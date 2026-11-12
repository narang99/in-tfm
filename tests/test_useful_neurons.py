from functools import partial

import pytest
import torch

from in_tfm.models.attn_only_2l.model import AttnOnlyConfig, AttnOnlyForCausalLM
from in_tfm.useful_neurons.attention import all_scores
from in_tfm.useful_neurons.attn_only.keep import keep_all
from in_tfm.useful_neurons.attn_only.terms import terms_of
from in_tfm.useful_neurons.attn_only.trace import Trace, trace_sequence
from in_tfm.useful_neurons.logit_shift import logit_shift
from in_tfm.useful_neurons.neurons import (
    HeadOut,
    Key,
    Neuron,
    OProj,
    Query,
    Resid,
    Score,
    Value,
)
from in_tfm.useful_neurons.random_control import random_keep
from in_tfm.useful_neurons.top_share import top_share_mask
from in_tfm.useful_neurons.walk import AllLogits, find_useful_neurons


@pytest.fixture
def model() -> AttnOnlyForCausalLM:
    torch.manual_seed(0)
    config = AttnOnlyConfig(
        hidden_size=16,
        num_attention_heads=2,
        head_dim=4,
        num_hidden_layers=1,
        max_position_embeddings=8,
        vocab_size=20,
        attention_bias=False,
    )
    return AttnOnlyForCausalLM(config).eval()


@pytest.fixture
def trace(model: AttnOnlyForCausalLM) -> Trace:
    return trace_sequence(model, torch.tensor([1, 5, 7, 2, 9, 3]))


def neuron_value(neuron: Neuron, trace: Trace) -> float:
    match neuron:
        case Resid(layer=layer, pos=pos, dim=dim):
            return trace.resid[layer][pos, dim].item()
        case OProj(layer=layer, pos=pos, head=head, dim=dim):
            return trace.layers[layer].head_writes[head, pos, dim].item()
        case HeadOut(layer=layer, pos=pos, head=head, dim=dim):
            return trace.layers[layer].z[head, pos, dim].item()
        case Value(layer=layer, pos=pos, head=head, dim=dim):
            return trace.layers[layer].v[head, pos, dim].item()
        case Query(layer=layer, pos=pos, head=head, dim=dim):
            return trace.layers[layer].q[head, pos, dim].item()
        case Key(layer=layer, pos=pos, head=head, dim=dim):
            return trace.layers[layer].k[head, pos, dim].item()
        case Score(layer=layer, pos=pos, src=src, head=head):
            attn = trace.layers[layer]
            return (attn.scaling * attn.q[head, pos] @ attn.k[head, src]).item()
    raise ValueError(f"no value for {neuron}")


def test_trace_reproduces_logits(model: AttnOnlyForCausalLM, trace: Trace) -> None:
    logits = model(input_ids=trace.tokens[None]).logits[0]
    torch.testing.assert_close(trace.logits, logits)


@pytest.mark.parametrize(
    "neuron",
    [
        Resid(layer=1, pos=4, dim=3),
        OProj(layer=0, pos=4, head=1, dim=3),
        HeadOut(layer=0, pos=4, head=1, dim=2),
        Value(layer=0, pos=2, head=0, dim=1),
        Score(layer=0, pos=4, src=2, head=1),
        Query(layer=0, pos=4, head=1, dim=2),
        Key(layer=0, pos=2, head=0, dim=3),
    ],
)
def test_terms_sum_to_the_neuron(neuron: Neuron, trace: Trace) -> None:
    terms = terms_of(neuron, trace)
    assert terms is not None
    assert terms.values.sum().item() == pytest.approx(
        neuron_value(neuron, trace), abs=1e-5
    )


def test_top_share_keeps_each_sign_separately() -> None:
    values = torch.tensor([5.0, 3.0, 2.0, -4.0, -1.0])
    kept = top_share_mask(values, share=0.6)
    assert kept.tolist() == [True, True, False, True, False]


def test_every_picked_input_is_itself_expanded(trace: Trace) -> None:
    terms = partial(terms_of, trace=trace)
    scores = all_scores(trace.layers)
    useful = find_useful_neurons(
        trace.unembed, terms, AllLogits(), share=0.9, scores=scores
    )
    assert useful.logit_picks.any()
    for inputs in useful.picks.values():
        assert all(input in useful.picks for input in inputs)


def test_dropping_a_value_removes_only_its_terms(
    model: AttnOnlyForCausalLM, trace: Trace
) -> None:
    seq = trace.tokens.shape[0]
    keep = keep_all(model.config, seq)
    keep.layers[0].v[1, 2, 3] = False
    ablated = trace_sequence(model, trace.tokens, keep)

    layer = trace.layers[0]
    removed = layer.pattern[1, :, 2] * layer.v[1, 2, 3]
    expected = layer.z[1, :, 3] - removed
    torch.testing.assert_close(ablated.layers[0].z[1, :, 3], expected)


def test_unchanged_logits_have_zero_kl(trace: Trace) -> None:
    shift = logit_shift(trace.logits, trace.logits, trace.tokens)
    assert max(shift.kl) == pytest.approx(0, abs=1e-6)
    assert all(pct == 0 for pct in shift.percent_change)


def test_random_keep_keeps_each_count_and_only_real_sources(
    model: AttnOnlyForCausalLM, trace: Trace
) -> None:
    seq = trace.tokens.shape[0]
    keep = keep_all(model.config, seq, fill=False)
    keep.resid[0][:3] = True
    keep.layers[0].pattern[1, 4, :3] = True
    shuffled = random_keep(keep, torch.Generator().manual_seed(0))

    assert shuffled.resid[0].sum() == keep.resid[0].sum()
    assert shuffled.layers[0].pattern.sum() == 3
    assert not shuffled.layers[0].pattern.triu(diagonal=1).any()
    assert not shuffled.resid[1].any()
