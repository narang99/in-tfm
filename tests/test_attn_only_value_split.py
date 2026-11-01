import torch

from in_tfm.models.attn_only_2l import AttnOnlyConfig, AttnOnlyForCausalLM, attention_patterns, value_split

HEAD_DIM, N_HEADS, SEQ = 8, 4, 12


def tiny_model() -> AttnOnlyForCausalLM:
    torch.manual_seed(0)
    config = AttnOnlyConfig(hidden_size=32, num_attention_heads=N_HEADS, head_dim=HEAD_DIM, num_hidden_layers=2,
                            max_position_embeddings=SEQ, vocab_size=50)
    return AttnOnlyForCausalLM(config).eval()


def head_block_values(model: AttnOnlyForCausalLM, ids: torch.Tensor, head: int, weights: torch.Tensor) -> torch.Tensor:
    o_proj = model.model.layers[0].self_attn.o_proj
    captured = {}
    handle = o_proj.register_forward_hook(lambda _m, inputs, _o: captured.__setitem__("z", inputs[0]))
    with torch.no_grad():
        model(input_ids=ids)
    handle.remove()
    return captured["z"][..., head * HEAD_DIM:(head + 1) * HEAD_DIM] @ weights


def test_value_split_parts_add_up_to_the_block_value():
    model = tiny_model()
    ids = torch.randint(0, 50, (3, SEQ))
    head, neuron = 2, 5
    weights = model.model.layers[0].self_attn.o_proj.weight[neuron, head * HEAD_DIM:(head + 1) * HEAD_DIM].detach()
    split = value_split(model, ids, 0, head, weights)
    torch.testing.assert_close(split.parts.sum(dim=-1), head_block_values(model, ids, head, weights))


def test_value_split_pattern_is_the_heads_attention():
    model = tiny_model()
    ids = torch.randint(0, 50, (2, SEQ))
    split = value_split(model, ids, 0, 1, torch.ones(HEAD_DIM))
    torch.testing.assert_close(split.pattern, attention_patterns(model, ids, 0)[:, 1])
