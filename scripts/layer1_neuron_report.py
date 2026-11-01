#!/usr/bin/env python
"""One layer 1 o_proj neuron on one short text: which value dims of each head feed it, as layer 0 concept cards.

    python scripts/layer1_neuron_report.py --neuron 215 --text "Once upon a time, there was a little girl named Lily. She"

- The neuron at the last position t is a sum over layer 1 heads g and sources s of A1_g[t, s] (v1_{g,s} . w_g).
  - w_g is the neuron's W_O row, block g.
- Per head, the page takes the source with the largest positive part: the head's top value vector.
  - Its part splits over its 32 dims j as A1_g[t, s] v1_{g,s}[j] w_g[j], the Hadamard product.
  - The page keeps the fewest positive dims holding `share` of the vector's positive sum.
- Each kept dim j is a layer 1 v_proj neuron: resid_s . W_V1[j]. It is shown as the 9 concept cards.
  - The cards are those of the residual dim page, read along the direction W_V1[j] instead of one dim.
  - W_E's card is embed_s . W_V1[j]. Layer 0 head h's card splits into tokens r as A0_h[s, r] (v0_{h,r} W_O0[block h] . W_V1[j]).
  - The cards sum to the value dim exactly.
- The top of the page is the neuron's own attribution on the input tokens, down both layers' OV paths, exact.
- Attention is held fixed everywhere, so QK gets no share.
"""

from pathlib import Path

import torch
from jaxtyping import Float
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from in_tfm.html_report import page, render_template
from in_tfm.models.attn_only_2l.model import BOS_TOKEN, AttnOnlyForCausalLM, load_trained_attn_only
from in_tfm.models.attn_only_2l.ov_graph import OvFlows, ov_flows
from in_tfm.presenters.text.colored_tokens import display_text
from in_tfm.presenters.text.concept_cards import concept_card_context

DIGITS = 5


class Layer1NeuronConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True, cli_parse_args=True)

    checkpoint_dir: Path = Path("checkpoints/tiny_attn_only_2l_v4")
    text: str = "Once upon a time, there was a little girl named Lily. She"
    neuron: int | None = None
    """A layer 1 o_proj neuron. By default, the one with the largest |value| at the last position."""
    share: float = 0.3
    out: Path = Path("reports/layer1_neuron/report.html")


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


class ValueDim(BaseModel):
    dim: int
    term: float
    value: float
    weight: float
    share: float
    """Of the value vector's positive Hadamard sum."""
    concepts: list[Concept]


class HeadRead(BaseModel):
    head: int
    value: float
    source: int | None
    source_part: float
    source_attention: float
    positive_sum: float
    dims: list[ValueDim]


class Layers(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    first: OvFlows
    second: OvFlows
    embeds: Float[torch.Tensor, "seq hidden"]


def r(x) -> float:
    return round(float(x), DIGITS)


def run_layers(model: AttnOnlyForCausalLM, input_ids: torch.Tensor) -> Layers:
    return Layers(first=ov_flows(model, input_ids, 0), second=ov_flows(model, input_ids, 1),
                  embeds=model.model.embed_tokens.weight[input_ids[0]].detach())


def value_dim_concepts(layers: Layers, direction: Float[torch.Tensor, "hidden"], s: int) -> list[Concept]:
    """resid_s . direction, split into W_E and each layer 0 head, and each head into its source tokens."""
    embed = layers.embeds[s] @ direction
    concepts = [Concept(stage="embedding", label="W_E", value=r(embed),
                        parts=[Part(s=s, a=1.0, vw=r(embed), part=r(embed))])]
    token_scores = layers.first.value_scores[:, :s + 1] @ direction
    attention = layers.first.pattern[:, s, :s + 1]
    for h in range(attention.shape[0]):
        parts = attention[h] * token_scores[h]
        concepts.append(Concept(
            stage="layer 0 W_O", label=f"head {h}", value=r(parts.sum()),
            parts=[Part(s=src, a=r(attention[h, src]), vw=r(token_scores[h, src]), part=r(parts[src]))
                   for src in range(s + 1)],
        ))
    return concepts


def fewest_holding(terms: Float[torch.Tensor, "n"], share: float) -> list[int]:
    """The fewest positive terms, largest first, whose sum reaches `share` of all positive terms."""
    positive_total = terms.clamp(min=0).sum()
    kept, held = [], 0.0
    for j in torch.argsort(terms, descending=True).tolist():
        if terms[j] <= 0 or held >= share * positive_total:
            break
        kept.append(j)
        held += float(terms[j])
    return kept


def head_read(model: AttnOnlyForCausalLM, layers: Layers, neuron: int, head: int, t: int, share: float) -> HeadRead:
    attn = model.model.layers[1].self_attn
    head_dim = attn.head_dim
    w = attn.o_proj.weight[neuron, head * head_dim:(head + 1) * head_dim].detach()
    values = layers.second.values[head, :t + 1]
    attention = layers.second.pattern[head, t, :t + 1]
    source_parts = attention * (values @ w)
    s = int(source_parts.argmax())
    if source_parts[s] <= 0:
        return HeadRead(head=head, value=r(source_parts.sum()), source=None, source_part=0.0, source_attention=0.0,
                        positive_sum=0.0, dims=[])
    terms = attention[s] * values[s] * w
    positive_sum = float(terms.clamp(min=0).sum())
    v_rows = attn.v_proj.weight[head * head_dim:(head + 1) * head_dim].detach()
    dims = [ValueDim(dim=j, term=r(terms[j]), value=r(values[s, j]), weight=r(w[j]), share=r(terms[j] / positive_sum),
                     concepts=value_dim_concepts(layers, v_rows[j], s))
            for j in fewest_holding(terms, share)]
    return HeadRead(head=head, value=r(source_parts.sum()), source=s, source_part=r(source_parts[s]),
                    source_attention=r(attention[s]), positive_sum=r(positive_sum), dims=dims)


def token_attribution(model: AttnOnlyForCausalLM, layers: Layers, neuron: int, t: int) -> Float[torch.Tensor, "seq"]:
    """The neuron reads resid_s . u_g through head g, with u_g = W_V1[block g]^T w_g. Each resid_s splits into tokens."""
    attn = model.model.layers[1].self_attn
    w_blocks = attn.o_proj.weight[neuron].detach().view(attn.n_heads, attn.head_dim)
    v_blocks = attn.v_proj.weight.detach().view(attn.n_heads, attn.head_dim, -1)
    reads = torch.einsum("gd,gdn->gn", w_blocks, v_blocks)
    second = layers.second.pattern[:, t]
    direct = torch.einsum("gr,rg->r", second, layers.embeds @ reads.T)
    through_layer_0 = torch.einsum("gs,hsr,hrg->r", second, layers.first.pattern, layers.first.value_scores @ reads.T)
    return direct + through_layer_0


def check_value_dims(heads: list[HeadRead], layers: Layers) -> float:
    """The cards of a value dim sum to the dim. Without a v_proj bias the gap is float error only."""
    return max((abs(sum(c.value for c in dim.concepts) - float(layers.second.values[head.head, head.source, dim.dim]))
                for head in heads for dim in head.dims), default=0.0)


def largest_card(heads: list[HeadRead]) -> float:
    return max((abs(c.value) for head in heads for dim in head.dims for c in dim.concepts), default=1e-9)


def main() -> None:
    config = Layer1NeuronConfig()
    print(config)
    model, tokenizer = load_trained_attn_only(config.checkpoint_dir)
    input_ids = torch.tensor([[tokenizer.convert_tokens_to_ids(BOS_TOKEN), *tokenizer.encode(config.text)]])
    t = input_ids.shape[1] - 1
    with torch.no_grad():
        predicted = int(model(input_ids=input_ids).logits[0, -1].argmax())
        layers = run_layers(model, input_ids)
    outputs = layers.second.outputs[t]
    neuron = config.neuron if config.neuron is not None else int(outputs.abs().argmax())
    with torch.no_grad():
        heads = [head_read(model, layers, neuron, head, t, config.share) for head in range(layers.second.pattern.shape[0])]
        tokens_attr = token_attribution(model, layers, neuron, t)
    print(f"neuron {neuron} at the last position: {float(outputs[neuron]):+.4f}, "
          f"token attribution sums to {float(tokens_attr.sum()):+.4f}")
    print(f"largest gap between a value dim and its cards: {check_value_dims(heads, layers):.2e}")
    for head in heads:
        print(f"head {head.head}: {head.value:+.4f}, top source {head.source}, {len(head.dims)} dims kept")

    body = render_template(
        "layer1_neuron.html",
        text=config.text,
        neuron=neuron,
        share=config.share,
        predicted=tokenizer.decode([predicted]),
        page_data={
            "tokens": [display_text(tokenizer.decode([tok])) for tok in input_ids[0].tolist()],
            "value": r(outputs[neuron]),
            "attribution": [r(x) for x in tokens_attr],
            "heads": [head.model_dump() for head in heads],
            "scale": largest_card(heads),
        },
        **concept_card_context(),
    )
    config.out.parent.mkdir(parents=True, exist_ok=True)
    config.out.write_text(page(f"Layer 1 neuron {neuron}", body))
    print(f"wrote {config.out}")


if __name__ == "__main__":
    main()
