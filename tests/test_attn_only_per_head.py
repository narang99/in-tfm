"""`PerHeadOutput` against the fused `o_proj` it replaced.

- The reference is the same model with each `o_proj` swapped back to one fused `nn.Linear` with the same weights.
- So the only difference between the two forward passes is the topology of the output projection.
"""

import copy

import pytest
import torch
from torch import nn

from in_tfm.models.attn_only_2l.model import AttnOnlyConfig, AttnOnlyForCausalLM, split_fused_o_proj

FLOAT32_TOLERANCE = 1e-5


def per_head_model(bias: bool, seed: int = 0) -> AttnOnlyForCausalLM:
    torch.manual_seed(seed)
    config = AttnOnlyConfig(
        hidden_size=32, num_attention_heads=4, head_dim=8, num_hidden_layers=2, max_position_embeddings=16,
        vocab_size=50, attention_bias=bias,
    )
    model = AttnOnlyForCausalLM(config).eval()
    for param in model.parameters():
        nn.init.normal_(param, std=0.3)
    return model


def with_fused_o_proj(model: AttnOnlyForCausalLM) -> AttnOnlyForCausalLM:
    fused_model = copy.deepcopy(model)
    for layer in fused_model.model.layers:
        per_head = layer.self_attn.o_proj
        fused = nn.Linear(per_head.weight.shape[1], per_head.weight.shape[0], bias=per_head.bias is not None,
                          dtype=per_head.weight.dtype)
        with torch.no_grad():
            fused.weight.copy_(per_head.weight)
            if per_head.bias is not None:
                fused.bias.copy_(per_head.bias)
        layer.self_attn.o_proj = fused
    return fused_model


def random_ids(model: AttnOnlyForCausalLM) -> torch.Tensor:
    return torch.randint(0, model.config.vocab_size, (3, model.config.max_position_embeddings),
                         generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize("bias", [True, False])
def test_per_head_logits_match_the_fused_projection(bias):
    model = per_head_model(bias)
    ids = random_ids(model)
    with torch.no_grad():
        expected = with_fused_o_proj(model)(input_ids=ids).logits
        actual = model(input_ids=ids).logits
    assert torch.allclose(actual, expected, atol=FLOAT32_TOLERANCE)


def test_a_fused_checkpoint_loads_into_the_per_head_model():
    fused_model = with_fused_o_proj(per_head_model(bias=True, seed=2))
    model = per_head_model(bias=True, seed=3)
    model.load_state_dict(split_fused_o_proj(fused_model.state_dict(), model.config.head_dim))
    ids = random_ids(model)
    with torch.no_grad():
        assert torch.allclose(model(input_ids=ids).logits, fused_model(input_ids=ids).logits, atol=FLOAT32_TOLERANCE)


def test_the_heads_sum_to_the_summed_neuron():
    o_proj = per_head_model(bias=True).model.layers[0].self_attn.o_proj
    z = torch.randn(2, 5, len(o_proj.heads) * o_proj.head_dim)
    per_head = [head(z_h) for head, z_h in zip(o_proj.heads, z.split(o_proj.head_dim, dim=-1))]
    with torch.no_grad():
        assert torch.allclose(sum(per_head) + o_proj.bias, o_proj(z), atol=FLOAT32_TOLERANCE)


def test_the_fused_weight_view_is_head_major():
    o_proj = per_head_model(bias=False).model.layers[0].self_attn.o_proj
    head_dim = o_proj.head_dim
    assert torch.equal(o_proj.weight[:, 2 * head_dim:3 * head_dim], o_proj.heads[2].weight)
