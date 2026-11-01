import torch

from in_tfm.models.attn_only_2l import AttnOnlyConfig, AttnOnlyForCausalLM
from in_tfm.models.attn_only_2l.ov_graph import ov_flows

SEQ = 10


def tiny_model() -> AttnOnlyForCausalLM:
    torch.manual_seed(0)
    config = AttnOnlyConfig(hidden_size=32, num_attention_heads=4, head_dim=8, num_hidden_layers=2,
                            max_position_embeddings=SEQ, vocab_size=50)
    return AttnOnlyForCausalLM(config).eval()


def residual_into_layer_1(model: AttnOnlyForCausalLM, ids: torch.Tensor) -> torch.Tensor:
    captured = {}
    handle = model.model.layers[1].register_forward_pre_hook(lambda _m, args: captured.__setitem__("r", args[0]))
    with torch.no_grad():
        model(input_ids=ids)
    handle.remove()
    return captured["r"][0]


def test_embedding_plus_head_flows_is_the_residual_after_layer_0():
    model = tiny_model()
    model.model.layers[0].self_attn.o_proj.bias.data.zero_()
    ids = torch.randint(0, 50, (1, SEQ))
    flows = ov_flows(model, ids, 0)
    per_head = torch.einsum("hts,hsn->htn", flows.pattern, flows.value_scores).sum(dim=0)
    embeds = model.model.embed_tokens.weight[ids[0]]
    torch.testing.assert_close(embeds + per_head, residual_into_layer_1(model, ids))
