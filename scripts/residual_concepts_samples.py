#!/usr/bin/env python
"""Concept cards for one layer 0 o_proj neuron, on examples sampled across its activation range.

    python scripts/residual_concepts_samples.py --neuron 215

- Windows come from the training stream, one per story start, 256 tokens from BOS.
- The binned value is the neuron's own value, the o_proj output, which is the sum of the heads.
- Each sign gets its own equal-width buckets from 0 to its extreme, as `bucketed_sample` lays them out.
  - Buckets holding fewer than `per_bucket` positions merge into a neighbour, so a side can have fewer buckets.
- Each bucket shows `per_bucket` examples, each from a different window, and no window repeats on the page.
- The cards are the ones from `ov_graph_report.py`. The embedding card is context: it is not part of the binned value.
"""

from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from in_tfm.device import default_device
from in_tfm.hit_selection.strategies.bucketed import group_of_each_value, plan_buckets
from in_tfm.html_report import page, render_template
from in_tfm.models.attn_only_2l.model import BOS_TOKEN, AttnOnlyForCausalLM, load_trained_attn_only
from in_tfm.models.attn_only_2l.ov_graph import ov_flows
from in_tfm.presenters.text.colored_tokens import display_text
from in_tfm.presenters.text.concept_cards import concept_card_context
from in_tfm.tiny_attn_only_training import TOKENS_FILE, story_starts

LAYER = 0


class SamplesConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True, cli_parse_args=True)

    checkpoint_dir: Path = Path("checkpoints/tiny_attn_only_2l_v4")
    split: str = "train"
    neuron: int = 215
    n_windows: int = 1000
    n_buckets: int = 10
    per_bucket: int = 5
    context: int = 20
    """Tokens shown before the current one."""
    batch_size: int = 32
    seed: int = 0
    device: str = default_device()
    out: Path | None = None


class Part(BaseModel):
    s: int
    a: float
    vw: float
    part: float


class Concept(BaseModel):
    stage: str
    label: str
    value: float
    parts: list[Part]
    outside: float = 0.0
    """The value from positions before the shown tokens."""


class Example(BaseModel):
    window: int
    position: int
    value: float
    residual: float
    tokens: list[str]
    offset: int
    concepts: list[Concept]


class Bucket(BaseModel):
    low: float
    high: float
    n_positions: int
    examples: list[Example]


class Side(BaseModel):
    sign: str
    n_positions: int
    buckets: list[Bucket]
    """Highest |value| first."""


class Pick(BaseModel):
    window: int
    position: int


def draw_windows(tokens: np.ndarray, bos: int, n: int, seq_len: int, rng: np.random.Generator) -> Int[np.ndarray, "n seq"]:
    """Distinct story starts, so no two windows open on the same story."""
    chosen = rng.choice(story_starts(tokens, bos, seq_len), size=n, replace=False)
    return np.stack([tokens[s:s + seq_len] for s in np.sort(chosen)]).astype(np.int64)


@torch.no_grad()
def neuron_values(model: AttnOnlyForCausalLM, windows: np.ndarray, neuron: int, batch_size: int,
                  device: str) -> Float[np.ndarray, "n seq"]:
    o_proj = model.model.layers[LAYER].self_attn.o_proj
    captured = {}
    handle = o_proj.register_forward_hook(lambda _m, _inp, out: captured.__setitem__("out", out[..., neuron]))
    values = []
    try:
        for start in range(0, len(windows), batch_size):
            model(input_ids=torch.from_numpy(windows[start:start + batch_size]).to(device))
            values.append(captured["out"].float().cpu().numpy())
    finally:
        handle.remove()
    return np.concatenate(values)


def pick_bucket(members: Int[np.ndarray, "m"], seq_len: int, per_bucket: int, used_windows: set[int],
                rng: np.random.Generator) -> list[Pick]:
    picks = []
    for flat in rng.permutation(members):
        window, position = divmod(int(flat), seq_len)
        if window in used_windows:
            continue
        used_windows.add(window)
        picks.append(Pick(window=window, position=position))
        if len(picks) == per_bucket:
            break
    return picks


def side_picks(values: Float[np.ndarray, "n seq"], sign: int, config: SamplesConfig, used_windows: set[int],
               rng: np.random.Generator) -> list[tuple[float, float, int, list[Pick]]]:
    """(low, high, size, picks) per bucket, highest |value| first, with low and high in signed units."""
    flat = sign * values.ravel()
    candidates = np.flatnonzero(flat > 0)
    layout = plan_buckets(flat[candidates], config.n_buckets, 0.0, config.per_bucket)
    group_of = group_of_each_value(flat[candidates], layout)
    edges = layout.group_edges()
    buckets = []
    for group in reversed(range(len(layout.groups))):
        members = candidates[group_of == group]
        picks = pick_bucket(members, values.shape[1], config.per_bucket, used_windows, rng)
        low, high = sorted((sign * float(edges[group]), sign * float(edges[group + 1])), key=abs)
        buckets.append((low, high, len(members), picks))
    return buckets


@torch.no_grad()
def example(model: AttnOnlyForCausalLM, tokenizer, windows: np.ndarray, pick: Pick, neuron: int,
            context: int) -> Example:
    """The window is cut at the position, which changes nothing since attention is causal."""
    t = pick.position
    ids = torch.from_numpy(windows[pick.window, :t + 1])[None]
    flows = ov_flows(model, ids, LAYER)
    offset = max(0, t - context)
    embed = float(model.model.embed_tokens.weight[ids[0, t], neuron])
    concepts = [Concept(stage="embedding", label="W_E", value=embed, parts=[Part(s=t, a=1.0, vw=embed, part=embed)])]
    for head in range(flows.pattern.shape[0]):
        attention = flows.pattern[head, t]
        value_score = flows.value_scores[head, :, neuron]
        parts = attention * value_score
        concepts.append(Concept(
            stage=f"layer {LAYER} W_O", label=f"head {head}", value=float(parts.sum()),
            parts=[Part(s=s, a=float(attention[s]), vw=float(value_score[s]), part=float(parts[s]))
                   for s in range(offset, t + 1)],
            outside=float(parts[:offset].sum()),
        ))
    value = float(flows.outputs[t, neuron])
    return Example(
        window=pick.window, position=t, value=value, residual=embed + value,
        tokens=[display_text(tokenizer.decode([tok])) for tok in windows[pick.window, offset:t + 1].tolist()],
        offset=offset, concepts=concepts,
    )


def build_side(model, tokenizer, windows, values, sign: int, config: SamplesConfig, used_windows: set[int],
               rng: np.random.Generator) -> Side:
    buckets = [
        Bucket(low=low, high=high, n_positions=size,
               examples=[example(model, tokenizer, windows, pick, config.neuron, config.context) for pick in picks])
        for low, high, size, picks in side_picks(values, sign, config, used_windows, rng)
    ]
    return Side(sign="positive" if sign > 0 else "negative", n_positions=int((sign * values > 0).sum()), buckets=buckets)


def split_gap(sides: list[Side], values: Float[np.ndarray, "n seq"]) -> float:
    """Examples are recomputed one at a time on the CPU, so this checks they match the batched scan."""
    return max(abs(ex.value - float(values[ex.window, ex.position]))
               for side in sides for bucket in side.buckets for ex in bucket.examples)


def largest_contribution(sides: list[Side]) -> float:
    """Brightness is global for the dim, so every card on the page shares this scale."""
    return max((abs(c.value) for side in sides for bucket in side.buckets for ex in bucket.examples
                for c in ex.concepts), default=1e-9)


def main() -> None:
    config = SamplesConfig()
    print(config)
    rng = np.random.default_rng(config.seed)
    model, tokenizer = load_trained_attn_only(config.checkpoint_dir)
    tokens = np.load(config.checkpoint_dir / TOKENS_FILE.format(split=config.split), mmap_mode="r")
    seq_len = model.config.max_position_embeddings
    windows = draw_windows(tokens, tokenizer.convert_tokens_to_ids(BOS_TOKEN), config.n_windows, seq_len, rng)
    values = neuron_values(model.to(config.device), windows, config.neuron, config.batch_size, config.device)
    model = model.cpu()
    print(f"{values.size} positions: min {values.min():+.3f}, max {values.max():+.3f}")

    used_windows: set[int] = set()
    sides = [build_side(model, tokenizer, windows, values, sign, config, used_windows, rng) for sign in (1, -1)]
    for side in sides:
        print(f"{side.sign}: {side.n_positions} positions, {len(side.buckets)} buckets, "
              f"{sum(len(b.examples) for b in side.buckets)} examples")
    print(f"largest gap between an example's split and its scanned value: {split_gap(sides, values):.2e}")

    body = render_template(
        "residual_concepts_samples.html",
        neuron=config.neuron,
        layer=LAYER,
        n_positions=int(values.size),
        n_windows=config.n_windows,
        context=config.context,
        sides=[side.model_dump() for side in sides],
        scale=largest_contribution(sides),
        **concept_card_context(),
    )
    out = config.out or Path(f"reports/residual_concepts/neuron_{config.neuron}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page(f"Residual dim {config.neuron} samples", body))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
