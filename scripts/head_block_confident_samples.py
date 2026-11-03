#!/usr/bin/env python
"""Examples of one pair's confident points, sampled from entropy buckets below the confident cut.

    python scripts/head_block_confident_samples.py --results-dir reports/head_block_batch --neuron 0 --head 3

- The confident share is one number, but the points under it range from exact copies at 0 bits to near the cut.
  - Each bucket gets its own section, so the two ends can be compared.
- Examples are drawn at random within a bucket, so large trees show up as often as they hold points there.
- Tokens are marked as on the cluster page, with the value split from `head_block_report.Attention`.
"""

from pathlib import Path

import numpy as np
from pydantic_settings import BaseSettings, SettingsConfigDict

from head_block_report import (
    Attention,
    block_weights,
    example,
    pair_name,
    section,
    shared_inputs,
)
from in_tfm.html_report import page, render_template


class ConfidentSamplesConfig(BaseSettings):
    model_config = SettingsConfigDict(
        extra="forbid", cli_kebab_case=True, cli_parse_args=True
    )

    checkpoint_dir: Path = Path("checkpoints/tiny_attn_only_2l_v4")
    results_dir: Path = Path("reports/head_block_batch")
    neuron: int = 0
    head: int = 3
    size: int = 4_000_000
    run_size: int = 4_000_000
    kind: str = "probe64"
    seed: int = 0
    bucket_edges: list[float] = [0.0, 0.001, 0.1, 0.25, 0.5]
    examples: int = 40
    context: int = 12
    out: Path | None = None


def bucket_points(
    entropy, low: float, high: float, count: int, windows, context: int, rng
) -> list[int]:
    """Random points in [low, high), skipping contexts already shown, since TinyStories repeats stories."""
    seen, chosen = set(), []
    for point in rng.permutation(np.flatnonzero((entropy >= low) & (entropy < high)))[
        : 50 * count
    ]:
        window, position = divmod(int(point), windows.shape[1])
        text = tuple(windows[window, max(0, position - context) : position + 1])
        if text not in seen:
            seen.add(text)
            chosen.append(int(point))
        if len(chosen) == count:
            break
    return chosen


def bucket_section(
    points: list[int],
    tree,
    entropy,
    low: float,
    high: float,
    shared,
    attention: Attention,
    context: int,
) -> str:
    in_bucket = ((entropy >= low) & (entropy < high)).mean()
    patterns, parts = attention.rows(np.array(points)) if points else ([], [])
    examples = [
        example(p, "", shared.windows, entropy, shared.texts, context, pattern, part)
        | {"tree": int(tree[p])}
        for p, pattern, part in zip(points, patterns, parts)
    ]
    for item in examples:
        item["tag"] = item["tag"].removeprefix(" · ")
    return section(
        f"{low:g} to {high:g} bits",
        f"{100 * in_bucket:.1f}% of all points. {len(examples)} random examples.",
        render_template("head_block_samples.html", examples=examples),
    )


def main() -> None:
    config = ConfidentSamplesConfig()
    shared = shared_inputs(config)
    name = pair_name(config, config.neuron, config.head)
    tree = np.load(config.results_dir / f"tree_{name}.npy")
    entropy = np.load(config.results_dir / f"entropy_{name}.npy").astype(np.float64)
    attention = Attention(
        shared.model,
        shared.windows,
        config.head,
        block_weights(shared.model, config.neuron, config.head),
    )
    rng = np.random.default_rng(config.seed)
    edges = config.bucket_edges
    sections = [
        bucket_section(
            bucket_points(
                entropy, low, high, config.examples, shared.windows, config.context, rng
            ),
            tree,
            entropy,
            low,
            high,
            shared,
            attention,
            config.context,
        )
        for low, high in zip(edges[:-1], edges[1:])
    ]
    body = "".join(
        [
            f"<h1>Neuron {config.neuron}, head {config.head}: confident points</h1>",
            f'<p class="meta">Points below {edges[-1]:g} bits, in entropy buckets. Each example names its tree. '
            "Underlined: the previous token. Outlined: the current token. Bold: the most-attended token. "
            "Shading: each token's part of this neuron's block value, attention times (value . weights). "
            "Red adds, blue takes away. Hover a token for its numbers.</p>",
            *sections,
        ]
    )
    out = config.out or config.results_dir / f"confident_samples_{name}.html"
    out.write_text(
        page(f"Confident points, neuron {config.neuron} head {config.head}", body)
    )
    print(out)


if __name__ == "__main__":
    main()
