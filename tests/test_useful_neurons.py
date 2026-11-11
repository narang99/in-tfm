import pytest
import torch

from in_tfm.models.attn_only_2l.model import AttnOnlyConfig, AttnOnlyForCausalLM
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
from in_tfm.useful_neurons.terms import terms_of
from in_tfm.useful_neurons.top_share import top_share_mask
from in_tfm.useful_neurons.trace import Trace, trace_sequence


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
    traced = trace.resid[-1] @ trace.W_U.T + model.lm_head.bias
    torch.testing.assert_close(traced, logits)


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
