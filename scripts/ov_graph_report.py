#!/usr/bin/env python
"""What one residual dim holds after layer 0, for one short text: the embedding, then each head's part.

    python scripts/ov_graph_report.py --text "Once upon a time, there was a little girl named Lily. She"

- A layer 0 o_proj neuron writes residual dim n, and nothing else writes it apart from the embedding.
  - So the dim is W_E[token, n] plus one value per head, exactly. See `models.attn_only_2l.ov_graph`.
- The page picks the position and the dim, so one file holds every (position, dim) view.
  - It opens on the last position, where the model predicts the next token.
- Only layer 0 for now, where each V is its own token and nothing else.
"""

from pathlib import Path

import torch
from matplotlib.colors import to_hex
from pydantic_settings import BaseSettings, SettingsConfigDict

from in_tfm.colormaps import rd_bk_gn, rd_wht_gn, token_cmap_dark, token_cmap_light
from in_tfm.html_report import page, render_template
from in_tfm.models.attn_only_2l.model import BOS_TOKEN, load_trained_attn_only
from in_tfm.models.attn_only_2l.ov_graph import ov_flows
from in_tfm.presenters.text.colored_tokens import display_text


class OvGraphReportConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True, cli_parse_args=True)

    checkpoint_dir: Path = Path("checkpoints/tiny_attn_only_2l_v4")
    text: str = "Once upon a time, there was a little girl named Lily. She"
    layer: int = 0
    out: Path = Path("reports/ov_graph/report.html")


def rounded(tensor: torch.Tensor) -> list:
    return torch.round(tensor, decimals=5).tolist()


def sign_colors() -> dict[str, str]:
    """The ends of the project's diverging maps, so an edge reads like a heatmap cell."""
    return {"pos_light": to_hex(rd_wht_gn(1.0)), "neg_light": to_hex(rd_wht_gn(0.0)),
            "pos_dark": to_hex(rd_bk_gn(1.0)), "neg_dark": to_hex(rd_bk_gn(0.0))}


def main() -> None:
    config = OvGraphReportConfig()
    print(config)
    model, tokenizer = load_trained_attn_only(config.checkpoint_dir)
    input_ids = torch.tensor([[tokenizer.convert_tokens_to_ids(BOS_TOKEN), *tokenizer.encode(config.text)]])
    with torch.no_grad():
        predicted = int(model(input_ids=input_ids).logits[0, -1].argmax())
    flows = ov_flows(model, input_ids, config.layer)

    graph_data = {
        "tokens": [display_text(tokenizer.decode([t])) for t in input_ids[0].tolist()],
        "pattern": rounded(flows.pattern),
        "valueScores": rounded(flows.value_scores),
        "outputs": rounded(flows.outputs),
        "embeds": rounded(model.model.embed_tokens.weight[input_ids[0]].detach()),
        "tokenLight": [to_hex(token_cmap_light(x)) for x in (0.0, 0.5, 1.0)],
        "tokenDark": [to_hex(token_cmap_dark(x)) for x in (0.0, 0.5, 1.0)],
    }
    body = render_template(
        "ov_graph.html",
        text=config.text,
        layer=config.layer,
        predicted=tokenizer.decode([predicted]),
        graph_data=graph_data,
        colors=sign_colors(),
    )
    config.out.parent.mkdir(parents=True, exist_ok=True)
    config.out.write_text(page(f"Residual concepts: {config.text}", body))
    print(f"predicted {tokenizer.decode([predicted])!r}, wrote {config.out}")


if __name__ == "__main__":
    main()
