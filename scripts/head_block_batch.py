#!/usr/bin/env python
"""Forests for many (neuron, head) pairs of layer 0 o_proj in one process, sized for a T4.

    python scripts/head_block_batch.py --neurons '[0,1,2,3,4,5,6,7,8,9]' --size 4000000

- What the pairs share is done once.
  - One forward pass captures the whole o_proj input for every point, all heads together.
    - A pair's rows are its head's slice of it times the neuron's weights there, so no pair runs the model.
  - One process, so the Triton kernels compile once instead of once per pair.
- Each pair writes its own result file, and the tree and entropy arrays `head_block_scale.save_point_arrays` names.
  - A restart skips pairs whose result file exists.
  - With `s3_prefix`, the folder is synced every `upload_every` pairs, under its own prefix so no results file
    from another run is overwritten.
- The defaults fit a 16 GB T4: smaller distance blocks, float64 rows on the CPU, and a smaller entropy batch.
"""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from pydantic_settings import BaseSettings, SettingsConfigDict

from head_block_forests import draw_windows
from head_block_scale import BlockPoints, HeadBlockScaleConfig, run_forest
from in_tfm.density_peaks import ivf_forest
from in_tfm.density_peaks.probabilities import frontier_entropy
from in_tfm.models.attn_only_2l.model import BOS_TOKEN, load_trained_attn_only
from in_tfm.tiny_attn_only_training import TOKENS_FILE, story_starts


class HeadBlockBatchConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True, cli_parse_args=True)

    checkpoint_dir: Path = Path("checkpoints/tiny_attn_only_2l_v4")
    neurons: list[int] = list(range(10))
    heads: list[int] = list(range(8))
    size: int = 4_000_000
    cell_size: int = 1000
    n_probe: int = 64
    peak_probe: int = 256
    margin_steps: float = 10.0
    block_elements: int = 2**26
    entropy_state: int = 2**30
    """Distances the entropy holds at once, trees x points. 4 GB of float32."""
    entropy_edges: int = 2**25
    """Edges one relaxation step takes. At 2^27 its index arrays ran a T4 out of memory at 4M points."""
    exact_on_device: bool = False
    seed: int = 0
    batch_size: int = 256
    device: str = "cuda"
    out_dir: Path = Path("reports/head_block_batch")
    s3_prefix: str | None = None
    upload_every: int = 8


@torch.no_grad()
def head_inputs(model, windows, batch_size: int, device: str) -> Float[np.ndarray, "n hidden"]:
    """The o_proj input at every position, which holds every head's z side by side."""
    o_proj = model.model.layers[0].self_attn.o_proj
    captured = {}
    handle = o_proj.register_forward_hook(lambda _m, inputs, _o: captured.__setitem__("z", inputs[0]))
    parts = []
    try:
        for start in range(0, len(windows), batch_size):
            model(input_ids=torch.from_numpy(windows[start:start + batch_size]).to(device))
            parts.append(captured["z"].float().reshape(-1, captured["z"].shape[-1]).cpu().numpy())
    finally:
        handle.remove()
    return np.concatenate(parts)


def pair_config(config: HeadBlockBatchConfig, neuron: int, head: int) -> HeadBlockScaleConfig:
    return HeadBlockScaleConfig(
        _cli_parse_args=False, checkpoint_dir=config.checkpoint_dir, neuron=neuron, head=head, sizes=[config.size],
        cell_size=config.cell_size, n_probe=config.n_probe, peak_probe=config.peak_probe,
        margin_steps=config.margin_steps, block_elements=config.block_elements, exact_up_to=0,
        compare_dense_up_to=0, entropy_backend="frontier", exact_on_device=config.exact_on_device, seed=config.seed,
        device=config.device, out=config.out_dir / "results.jsonl",
    )


def result_path(config: HeadBlockBatchConfig, neuron: int, head: int) -> Path:
    return config.out_dir / f"result_neuron{neuron}_head{head}_n{config.size}.json"


def upload(config: HeadBlockBatchConfig) -> None:
    if config.s3_prefix:
        subprocess.run(["aws", "s3", "sync", "--quiet", str(config.out_dir), config.s3_prefix], check=False)


def main() -> None:
    config = HeadBlockBatchConfig()
    print(config, flush=True)
    ivf_forest.BLOCK_ELEMENTS = config.block_elements
    frontier_entropy.STATE_ELEMENTS = config.entropy_state
    frontier_entropy.EDGES_PER_STEP = config.entropy_edges
    config.out_dir.mkdir(parents=True, exist_ok=True)
    pending = [(n, h) for n in config.neurons for h in config.heads if not result_path(config, n, h).exists()]
    print(f"{len(pending)} pairs to run", flush=True)
    if not pending:
        return
    model, tokenizer = load_trained_attn_only(config.checkpoint_dir)
    model = model.to(config.device).eval()
    tokens = np.load(config.checkpoint_dir / TOKENS_FILE.format(split="train"), mmap_mode="r")
    seq_len = model.config.max_position_embeddings
    starts = story_starts(tokens, tokenizer.convert_tokens_to_ids(BOS_TOKEN), seq_len)
    windows, _ = draw_windows(tokens, starts, -(-config.size // seq_len), seq_len, np.random.default_rng(config.seed))
    began = time.perf_counter()
    z = head_inputs(model, windows, config.batch_size, config.device)[:config.size]
    print(f"{len(z):,} points in {time.perf_counter() - began:.0f}s", flush=True)
    current = windows.reshape(-1)[:config.size]
    previous = np.concatenate([np.full((len(windows), 1), -1), windows[:, :-1]], axis=1).reshape(-1)[:config.size]
    weight = model.model.layers[0].self_attn.o_proj.weight.detach().float().cpu().numpy()
    head_dim = model.config.head_dim
    for done, (neuron, head) in enumerate(pending, start=1):
        block = slice(head * head_dim, (head + 1) * head_dim)
        points = BlockPoints(rows=z[:, block] * weight[neuron, block], current=current, previous=previous)
        began = time.perf_counter()
        row, forest, _ = run_forest(points, config.size, pair_config(config, neuron, head), exact=False)
        row |= {"neuron": neuron, "head": head, "pair_seconds": time.perf_counter() - began}
        result_path(config, neuron, head).write_text(json.dumps(row))
        print(f"neuron {neuron} head {head}: {row['pair_seconds']:.0f}s, forest {row['forest_seconds']:.0f}s, "
              f"entropy {row['entropy_seconds']:.0f}s, {row['trees']} trees, {100 * row['confident']:.1f}% confident",
              flush=True)
        del forest
        torch.cuda.empty_cache()
        if done % config.upload_every == 0:
            upload(config)
    upload(config)
    print("done", flush=True)


if __name__ == "__main__":
    main()
