from functools import partial
from typing import cast

import pytest
import torch
from transformers import GPTNeoConfig, GPTNeoForCausalLM
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock

from in_tfm.useful_neurons.attention import all_scores
from in_tfm.useful_neurons.gpt_neo.keep import keep_all
from in_tfm.useful_neurons.gpt_neo.terms import block_writers, terms_of
from in_tfm.useful_neurons.gpt_neo.trace import GptNeoTrace, trace_gpt_neo
from in_tfm.useful_neurons.neurons import (
    HeadOut,
    MlpDown,
    Neuron,
    OProj,
    Resid,
    ResidMid,
    Score,
)
from in_tfm.useful_neurons.walk import (
    AllLogits,
    find_block_neurons,
    find_useful_neurons,
)


@pytest.fixture
def model() -> GPTNeoForCausalLM:
    """Window 3 with 8 tokens, so the local layer really drops far sources."""
    torch.manual_seed(0)
    config = GPTNeoConfig(
        vocab_size=30,
        max_position_embeddings=16,
        hidden_size=16,
        num_layers=2,
        num_heads=4,
        attention_types=[[["global", "local"], 1]],
        window_size=3,
    )
    model = GPTNeoForCausalLM(config).eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "bias" in name or "ln" in name:
                param.add_(torch.randn_like(param) * 0.1)
    return model


@pytest.fixture
def trace(model: GPTNeoForCausalLM) -> GptNeoTrace:
    return trace_gpt_neo(model, torch.tensor([1, 5, 7, 2, 9, 3, 11, 4]))


def test_trace_reproduces_logits(model: GPTNeoForCausalLM, trace: GptNeoTrace) -> None:
    logits = model(input_ids=trace.tokens[None]).logits[0]
    torch.testing.assert_close(trace.logits, logits)


def value_without_bias(
    neuron: Neuron, trace: GptNeoTrace, model: GPTNeoForCausalLM
) -> float:
    def block(layer: int) -> GPTNeoBlock:
        return cast(GPTNeoBlock, model.transformer.h[layer])

    match neuron:
        case Resid(layer=0, pos=pos, dim=dim):
            return trace.resid[0][pos, dim].item()
        case Resid(layer=layer, pos=pos, dim=dim):
            bias = block(layer - 1).mlp.c_proj.bias[dim]
            return (trace.resid[layer][pos, dim] - bias).item()
        case ResidMid(layer=layer, pos=pos, dim=dim):
            bias = block(layer).attn.attention.out_proj.bias[dim]
            return (trace.resid_mid[layer][pos, dim] - bias).item()
        case OProj(layer=layer, pos=pos, head=head, dim=dim):
            return trace.blocks[layer].attn.head_writes[head, pos, dim].item()
        case HeadOut(layer=layer, pos=pos, head=head, dim=dim):
            return trace.blocks[layer].attn.z[head, pos, dim].item()
        case Score(layer=layer, pos=pos, src=src, head=head):
            attn = trace.blocks[layer].attn
            return (attn.q[head, pos] @ attn.k[head, src]).item()
        case MlpDown(layer=layer, pos=pos, dim=dim):
            return trace.blocks[layer].mlp.writes[pos, dim].item()
    raise ValueError(f"no value for {neuron}")


@pytest.mark.parametrize(
    "neuron",
    [
        Resid(layer=0, pos=4, dim=3),
        Resid(layer=2, pos=4, dim=3),
        ResidMid(layer=1, pos=6, dim=2),
        OProj(layer=1, pos=6, head=2, dim=5),
        HeadOut(layer=1, pos=6, head=2, dim=1),
        Score(layer=1, pos=6, src=4, head=2),
        MlpDown(layer=0, pos=5, dim=7),
    ],
)
def test_terms_sum_to_the_neuron_without_bias(
    neuron: Neuron, trace: GptNeoTrace, model: GPTNeoForCausalLM
) -> None:
    terms = terms_of(neuron, trace)
    assert terms is not None
    expected = value_without_bias(neuron, trace, model)
    assert terms.values.sum().item() == pytest.approx(expected, abs=1e-5)


def test_dropping_a_resid_dim_removes_its_query_term(
    model: GPTNeoForCausalLM, trace: GptNeoTrace
) -> None:
    """With the norms frozen, a dim's term through ln_1 is exactly what zeroing it removes."""
    keep = keep_all(model.config, len(trace.tokens))
    keep.resid[1][4, 2] = False
    ablated = trace_gpt_neo(model, trace.tokens, keep, frozen=trace.frozen_norms)

    block = trace.blocks[1]
    term = block.attn.W_Q[:, :, 2] * block.ln_1.scale[4, 2] * trace.resid[1][4, 2]
    expected = block.attn.q[:, 4] - term
    torch.testing.assert_close(ablated.blocks[1].attn.q[:, 4], expected)


def test_dropping_a_mid_dim_removes_its_mlp_input_term(
    model: GPTNeoForCausalLM, trace: GptNeoTrace
) -> None:
    keep = keep_all(model.config, len(trace.tokens))
    keep.resid_mid[0][5, 7] = False
    ablated = trace_gpt_neo(model, trace.tokens, keep, frozen=trace.frozen_norms)

    W_in = trace.blocks[0].mlp.W_in
    before = trace.blocks[0].ln_2.output[5] @ W_in.T
    after = ablated.blocks[0].ln_2.output[5] @ W_in.T
    term = W_in[:, 7] * trace.blocks[0].ln_2.scale[5, 7] * trace.resid_mid[0][5, 7]
    torch.testing.assert_close(before - after, term)


def test_every_picked_input_is_itself_expanded(trace: GptNeoTrace) -> None:
    terms = partial(terms_of, trace=trace)
    scores = all_scores([block.attn for block in trace.blocks])
    useful = find_useful_neurons(
        trace.unembed, terms, AllLogits(), share=0.9, scores=scores
    )
    assert useful.logit_picks.any()
    for inputs in useful.picks.values():
        assert all(input in useful.picks for input in inputs)


def test_block_walk_never_leaves_its_block(trace: GptNeoTrace) -> None:
    terms = partial(terms_of, trace=trace)
    picks = find_block_neurons(block_writers(trace), terms, share=0.9)
    stream_kinds = {"resid", "resid_mid", "token", "pos"}
    for neuron, inputs in picks.items():
        if neuron.kind in stream_kinds:
            assert inputs == []
    assert any(neuron.kind == "mlp_hidden" for neuron in picks)


def test_scores_start_their_own_walks(trace: GptNeoTrace) -> None:
    """Window 3 on the local layer, so its far sources get no score."""
    scores = all_scores([block.attn for block in trace.blocks])
    seq = len(trace.tokens)
    causal_pairs = seq * (seq + 1) // 2
    local_pairs = sum(min(pos + 1, 3) for pos in range(seq))
    assert len(scores) == 4 * (causal_pairs + local_pairs)

    head_out = terms_of(HeadOut(layer=1, pos=6, head=2, dim=1), trace)
    assert head_out is not None
    picked = [n for i in range(len(head_out.values)) for n in head_out.inputs_of(i)]
    assert picked and not any(isinstance(n, Score) for n in picked)
