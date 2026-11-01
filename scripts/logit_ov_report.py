#!/usr/bin/env python
"""One HTML page explaining the predicted next token of one short text, through every OV dot product.

    python scripts/logit_ov_report.py --text "Once upon a time, there was a little girl named Lily. She"

- The target is the last position's logit for the argmax token, so it is what the model predicted.
- AttnLRP gives that logit's relevance at every `v_proj` and `o_proj` output, at every position.
- High relevance is chosen per (layer, projection), above the elbow of the sorted relevance.
  - Positive and negative relevance each get their own elbow.
  - `v_proj` and `o_proj` are two stages of one path, so pooling them would count the same flow twice.
- Each chosen unit gets a panel.
  - Its Hadamard product: the projection's input at that position times the unit's weight row.
    - It sums to the unit's activation minus its bias.
  - Its own AttnLRP relevance on the input tokens.
- AttnLRP splits the relevance of `A @ V` evenly between `A` and `V`.
  - So a layer's `v_proj` units hold half of its `o_proj` relevance. The other half goes to the QK path.
- QK units are not covered yet.
"""

from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from in_tfm.colormaps import rd_bk_gn, rd_wht_gn, token_cmap_dark, token_cmap_light
from in_tfm.hit_selection.strategies.elbow import find_elbow_index_in_sorted_data
from in_tfm.html_report import page, render_template
from in_tfm.models.attn_only_2l.adapter import patch_attn_only_for_attn_lrp
from in_tfm.models.attn_only_2l.model import BOS_TOKEN, AttnOnlyForCausalLM, load_trained_attn_only
from in_tfm.models.attn_only_2l.ov_units import (
    OvCapture, OvUnit, UnitKey, capture_ov, logit_relevance, projection, unit_keys, unit_relevance,
)
from in_tfm.presenters.text.colored_tokens import display_text, diverging_colors
from in_tfm.presenters.text.views import TokenColors


class LogitOvReportConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True, cli_parse_args=True)

    checkpoint_dir: Path = Path("checkpoints/tiny_attn_only_2l_v4")
    text: str = "Once upon a time, there was a little girl named Lily. She"
    out: Path = Path("reports/logit_ov/report.html")


class Cell(BaseModel):
    value: float
    colors: TokenColors


class TokenChip(BaseModel):
    text: str
    relevance: float
    colors: TokenColors
    current: bool


class UnitPanel(BaseModel):
    unit: OvUnit
    head: int
    logit_relevance: float
    activation: float
    bias: float
    rows: list[list[Cell]]
    row_labels: list[str]
    tokens: list[TokenChip]
    token_gap: float
    """The unit's activation minus its summed token relevance.

    - Position embeddings and biases are not leaves, so their share lands here.
    - AttnLRP's softmax rule is not fully conservative, so part of the gap is that rule's loss.
    """


class UnitGroup(BaseModel):
    title: str
    sign: str
    n_units: int
    cutoff: float
    total_share: float
    panels: list[UnitPanel]


def above_elbow(magnitudes: Float[np.ndarray, "n"]) -> tuple[Float[np.ndarray, "n_kept"], float]:
    """Indices of `magnitudes` above the elbow of their ascending sort, and the cutoff."""
    ascending = np.sort(magnitudes)
    if len(ascending) < 3:
        return np.arange(len(magnitudes)), 0.0
    cutoff = float(ascending[find_elbow_index_in_sorted_data(torch.from_numpy(ascending))])
    return np.flatnonzero(magnitudes > cutoff), cutoff


def chosen_units(key: UnitKey, relevance: Float[np.ndarray, "seq out"], sign: int) -> tuple[list[OvUnit], float]:
    signed = sign * relevance.ravel()
    candidates = np.flatnonzero(signed > 0)
    kept, cutoff = above_elbow(signed[candidates])
    flat = candidates[kept][np.argsort(-signed[candidates[kept]])]
    units = [OvUnit(layer=key[0], projection=key[1], neuron=int(i % relevance.shape[1]),
                    position=int(i // relevance.shape[1])) for i in flat]
    return units, cutoff


def hadamard_rows(product: Float[np.ndarray, "in_features"], head_dim: int) -> list[list[Cell]]:
    largest = float(np.abs(product).max())
    return [[Cell(value=float(v), colors=diverging_colors(float(v), largest, rd_wht_gn, rd_bk_gn)) for v in row]
            for row in product.reshape(-1, head_dim)]


def row_labels(projection_name: str, n_rows: int, head_dim: int) -> list[str]:
    """`o_proj` reads the heads side by side, so a row is a head. `v_proj` reads the residual, which has no such layout."""
    if projection_name == "o":
        return [f"h{h}" for h in range(n_rows)]
    return [f"{r * head_dim}" for r in range(n_rows)]


def token_chips(texts: list[str], relevance: Float[np.ndarray, "seq"], position: int) -> list[TokenChip]:
    largest = float(np.abs(relevance).max())
    return [TokenChip(text=display_text(text), relevance=float(r), current=i == position,
                      colors=diverging_colors(float(r), largest, token_cmap_light, token_cmap_dark))
            for i, (text, r) in enumerate(zip(texts, relevance))]


def unit_panel(model: AttnOnlyForCausalLM, input_ids: torch.Tensor, capture: OvCapture, texts: list[str],
               unit: OvUnit, logit_rel: float) -> UnitPanel:
    key = (unit.layer, unit.projection)
    layer = projection(model, key)
    weight_row = layer.weight[unit.neuron].detach()
    product = (capture.inputs[key][unit.position] * weight_row).numpy()
    head_dim = model.config.head_dim
    explained = unit_relevance(model, input_ids, unit)
    tokens = explained.tokens.numpy()
    return UnitPanel(
        unit=unit,
        head=unit.neuron // head_dim,
        logit_relevance=logit_rel,
        activation=float(capture.outputs[key][unit.position, unit.neuron]),
        bias=float(layer.bias[unit.neuron]) if layer.bias is not None else 0.0,
        rows=hadamard_rows(product, head_dim),
        row_labels=row_labels(unit.projection, len(product) // head_dim, head_dim),
        tokens=token_chips(texts, tokens, unit.position),
        token_gap=explained.target - float(tokens.sum()),
    )


def unit_group(model, input_ids, capture, texts, key: UnitKey, relevance: Float[np.ndarray, "seq out"],
               sign: int) -> UnitGroup:
    units, cutoff = chosen_units(key, relevance, sign)
    same_sign_total = (sign * relevance).clip(min=0).sum()
    held = sum(sign * relevance[u.position, u.neuron] for u in units)
    return UnitGroup(
        title=f"layer {key[0]} {key[1]}_proj",
        sign="positive" if sign > 0 else "negative",
        n_units=len(units),
        cutoff=cutoff,
        total_share=float(held / same_sign_total) if same_sign_total > 0 else 0.0,
        panels=[unit_panel(model, input_ids, capture, texts, u, float(relevance[u.position, u.neuron])) for u in units],
    )


def main() -> None:
    config = LogitOvReportConfig()
    print(config)
    model, tokenizer = load_trained_attn_only(config.checkpoint_dir)
    patch_attn_only_for_attn_lrp(model)
    input_ids = torch.tensor([[tokenizer.convert_tokens_to_ids(BOS_TOKEN), *tokenizer.encode(config.text)]])
    texts = [tokenizer.decode([t]) for t in input_ids[0].tolist()]

    capture = capture_ov(model, input_ids)
    predicted = int(capture.logits[-1].argmax())
    explained = logit_relevance(model, input_ids, predicted)
    token_sum = float(explained.tokens.sum())
    print(f"predicted {tokenizer.decode([predicted])!r}, logit {explained.target:.3f}, "
          f"token relevance {token_sum:.3f}, gap {explained.target - token_sum:.3f}")

    groups = []
    for key in reversed(unit_keys(model)):
        relevance = explained.units[key].numpy()
        for sign in (1, -1):
            group = unit_group(model, input_ids, capture, texts, key, relevance, sign)
            print(f"{group.title} {group.sign}: {group.n_units} units above {group.cutoff:.4f}, "
                  f"{100 * group.total_share:.0f}% of that sign's relevance")
            groups.append(group)

    body = render_template(
        "logit_ov_report.html",
        text=config.text,
        predicted=tokenizer.decode([predicted]),
        logit=explained.target,
        input_tokens=token_chips(texts, explained.tokens.numpy(), len(texts) - 1),
        token_gap=explained.target - token_sum,
        groups=groups,
    )
    config.out.parent.mkdir(parents=True, exist_ok=True)
    config.out.write_text(page(f"Logit OV report: {config.text}", body))
    print(f"wrote {config.out}")


if __name__ == "__main__":
    main()
