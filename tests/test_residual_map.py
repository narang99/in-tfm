import pytest
import torch

from in_tfm.dependencies import (
    FixedWriter,
    NeuronWriter,
    ResidualStreamMap,
    Tap,
    check_terms_rebuild_input,
)
from in_tfm.dependencies.residual import FixedSource, ReportedSource
from in_tfm.models.attn_only_2l import AttnOnlyConfig, AttnOnlyForCausalLM
from in_tfm.models.attn_only_2l.adapter import TinyAttnOnly1LD128Adapter

HIDDEN, N_HEADS, SEQ = 8, 2, 5

EMBED = Tap(layer="embed", port="input")
HEAD = Tap(layer="head0")


def test_each_writer_gives_its_own_value_on_the_dim():
    stream_map = ResidualStreamMap(
        [FixedSource(name="embedding", tap=EMBED), ReportedSource(layer="head0")]
    )
    captures = {EMBED: torch.tensor([[1.0, 2.0]]), HEAD: torch.tensor([[3.0, -5.0]])}
    dependency = stream_map.dependency(1, 0, captures)
    assert [t.writer for t in dependency.terms] == [
        FixedWriter(name="embedding", dim=1, token=0),
        NeuronWriter(layer="head0", neuron=1, token=0),
    ]
    assert [t.term for t in dependency.terms] == [2.0, -5.0]
    assert dependency.total == -3.0


def random_attn_only(n_layers: int, bias: bool = False) -> AttnOnlyForCausalLM:
    torch.manual_seed(0)
    config = AttnOnlyConfig(
        hidden_size=HIDDEN,
        num_attention_heads=N_HEADS,
        head_dim=4,
        num_hidden_layers=n_layers,
        max_position_embeddings=SEQ,
        vocab_size=11,
        attention_bias=bias,
    )
    return AttnOnlyForCausalLM(config).eval()


def captured_taps(
    model: AttnOnlyForCausalLM, taps: list[Tap]
) -> tuple[dict[Tap, torch.Tensor], torch.Tensor]:
    """Plain torch hooks, so the map is checked independently of the capture code."""
    saved: dict[Tap, torch.Tensor] = {}
    hooks = []
    for tap in taps:
        module = model.get_submodule(tap.layer)
        if tap.port == "input":
            hook = module.register_forward_pre_hook(
                lambda m, args, tap=tap: saved.__setitem__(tap, args[0])
            )
        else:
            hook = module.register_forward_hook(
                lambda m, args, out, tap=tap: saved.__setitem__(tap, out)
            )
        hooks.append(hook)
    lm_head_in: list[torch.Tensor] = []
    hooks.append(
        model.lm_head.register_forward_pre_hook(
            lambda m, args: lm_head_in.append(args[0])
        )
    )
    with torch.no_grad():
        model(inputs_embeds=torch.randn(1, SEQ, HIDDEN))
    for hook in hooks:
        hook.remove()
    return {tap: t[0] for tap, t in saved.items()}, lm_head_in[0][0]


@pytest.mark.parametrize("n_layers", [1, 2])
def test_the_unembed_map_rebuilds_every_dim_at_every_token(n_layers):
    model = random_attn_only(n_layers)
    adapter = TinyAttnOnly1LD128Adapter(model, tokenizer=None)  # ty: ignore[invalid-argument-type]
    stream_map = adapter.dependency_map_for("lm_head")
    assert len(stream_map.taps) == 1 + n_layers * N_HEADS

    captures, lm_head_in = captured_taps(model, stream_map.taps)
    for token in range(SEQ):
        dependencies = [
            stream_map.dependency(dim, token, captures) for dim in range(HIDDEN)
        ]
        assert check_terms_rebuild_input(dependencies, lm_head_in[token]) < 1e-5


def test_a_model_with_an_o_proj_bias_has_no_map():
    adapter = TinyAttnOnly1LD128Adapter(random_attn_only(1, bias=True), tokenizer=None)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="bias"):
        adapter.dependency_map_for("lm_head")


def test_a_layer_that_also_reads_positions_has_no_map():
    adapter = TinyAttnOnly1LD128Adapter(random_attn_only(1), tokenizer=None)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="only lm_head"):
        adapter.dependency_map_for("model.layers.0.self_attn.q_proj")
