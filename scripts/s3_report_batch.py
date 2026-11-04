#!/usr/bin/env python
"""Neuron reports for a list of neurons, in chunks, resumable from what S3 already holds.

- Each chunk is one `run_neuron_report.py` call, so the scan and gather are shared within it.
- After a chunk, each neuron's page, cluster images and `mean.pt` go to `<s3-prefix>/<neuron>/`.
  - `hits.pt`, the per-cluster `hits.html` and the rest stay behind, and the local folder is deleted.
- A neuron counts as done once its `index.html` is in S3, so a recycled VM loses at most one chunk.

    python scripts/s3_report_batch.py --config <yaml> --neurons '[6, 8]' --s3-prefix s3://bucket/path
"""

import itertools
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from in_tfm.neuron_run import report_name
from in_tfm.run_config import load_run_config

KEPT = ["index.html", "*/mean.pt", "*/hadamard.jpg"]


class Args(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    config: Path
    neurons: list[int]
    s3_prefix: str
    chunk: int = 8


def finished_neurons(s3_prefix: str) -> set[int]:
    listing = subprocess.run(
        ["aws", "s3", "ls", f"{s3_prefix.rstrip('/')}/", "--recursive"],
        capture_output=True,
        text=True,
    ).stdout
    return {int(m) for m in re.findall(r"/(\d+)/index\.html$", listing, re.M)}


def run_chunk(config: Path, neurons: list[int]) -> None:
    command = [
        sys.executable,
        "scripts/run_neuron_report.py",
        "--config",
        str(config),
        "--neurons",
        json.dumps(neurons),
    ]
    subprocess.run(command, check=True)


def upload(report_dir: Path, destination: str) -> None:
    includes = [arg for pattern in KEPT for arg in ("--include", pattern)]
    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            "--quiet",
            "--recursive",
            str(report_dir),
            destination,
            "--exclude",
            "*",
            *includes,
        ],
        check=True,
    )


def main() -> None:
    args = Args()  # ty: ignore[missing-argument]
    run_config = load_run_config(["--config", str(args.config)])
    prefix = args.s3_prefix.rstrip("/")
    done = finished_neurons(prefix)
    remaining = [n for n in args.neurons if n not in done]
    print(
        f"[batch] {len(done & set(args.neurons))} done in S3, {len(remaining)} to run",
        flush=True,
    )
    started = time.time()
    for chunk in itertools.batched(remaining, args.chunk):
        run_chunk(args.config, list(chunk))
        for neuron in chunk:
            report_dir = Path(run_config.out_dir) / report_name(
                neuron, run_config.polarity
            )
            upload(report_dir, f"{prefix}/{neuron}/")
            shutil.rmtree(report_dir)
        left = len(remaining) - len(finished_neurons(prefix) & set(remaining))
        print(
            f"[batch] chunk {list(chunk)} uploaded, {left} left, {time.time() - started:.0f}s",
            flush=True,
        )
    missing = sorted(set(args.neurons) - finished_neurons(prefix))
    print(f"[batch] done, missing in S3: {missing}", flush=True)


if __name__ == "__main__":
    main()
