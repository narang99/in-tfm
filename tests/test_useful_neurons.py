import pytest
import torch

from in_tfm.models.attn_only_2l.model import AttnOnlyConfig, AttnOnlyForCausalLM
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


def test_trace_reproduces_logits(model: AttnOnlyForCausalLM, trace: Trace) -> None:
    logits = model(input_ids=trace.tokens[None]).logits[0]
    traced = trace.resid[-1] @ trace.W_U.T + model.lm_head.bias
    torch.testing.assert_close(traced, logits)


def test_top_share_keeps_each_sign_separately() -> None:
    values = torch.tensor([5.0, 3.0, 2.0, -4.0, -1.0])
    kept = top_share_mask(values, share=0.6)
    assert kept.tolist() == [True, True, False, True, False]
