import pytest
import torch
from nnsight import NNsight
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from in_tfm.layers import submodule_getter
from in_tfm.models.attn_only_2l import (
    AttnOnly2LAdapter,
    AttnOnlyConfig,
    AttnOnlyForCausalLM,
)
from in_tfm.models.gemma3 import Gemma3Adapter
from in_tfm.sources import ModelBatch

N_HEADS, HEAD_DIM, HIDDEN, VOCAB, MAX_LENGTH = 2, 4, 8, 20, 6


class FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    bos_token = "<bos>"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def __call__(self, texts, **kwargs):
        self.seen.extend(texts)
        ids = torch.randint(0, VOCAB, (len(texts), kwargs["max_length"]))
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def tiny_attn_only_adapter() -> AttnOnly2LAdapter:
    config = AttnOnlyConfig(
        hidden_size=HIDDEN,
        num_attention_heads=N_HEADS,
        head_dim=HEAD_DIM,
        num_hidden_layers=2,
        max_position_embeddings=MAX_LENGTH,
        vocab_size=VOCAB,
    )
    return AttnOnly2LAdapter(
        AttnOnlyForCausalLM(config), FakeTokenizer(), max_length=MAX_LENGTH
    )


def tiny_gemma3_adapter() -> Gemma3Adapter:
    config = Gemma3TextConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=N_HEADS,
        num_key_value_heads=1,
        head_dim=HEAD_DIM,
        max_position_embeddings=MAX_LENGTH,
    )
    return Gemma3Adapter(
        Gemma3ForCausalLM(config), FakeTokenizer(), max_length=MAX_LENGTH
    )


@pytest.fixture(
    params=[tiny_attn_only_adapter, tiny_gemma3_adapter], ids=["attn_only_2l", "gemma3"]
)
def adapter(request):
    return request.param()


def test_getter_weight_rows_match_the_layer_output_width(adapter):
    weight = submodule_getter("layers.0.self_attn.q_proj")(adapter.get_model()).weight
    assert weight.shape == (N_HEADS * HEAD_DIM, HIDDEN)


def test_patch_runs_once(adapter, monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(
        type(adapter), "_apply_attn_lrp_patch", lambda self: calls.append(1)
    )
    adapter.patch_for_attn_lrp()
    adapter.patch_for_attn_lrp()
    assert calls == [1]


def test_source_batch_carries_its_gradient_leaf(adapter):
    source = adapter.make_source(["a", "b"])
    batch = source.to_model_batch(list(source.sample_ids()))
    assert isinstance(batch, ModelBatch)
    assert batch.grad_leaf_key in batch.kwargs
    assert batch.valid_mask.shape == (2, MAX_LENGTH)


def test_model_runs_on_the_source_batch(adapter):
    source = adapter.make_source(["a"])
    batch = source.to_model_batch(list(source.sample_ids()))
    with torch.no_grad():
        adapter.get_model()(**batch.kwargs)


@pytest.mark.parametrize(
    ("make_adapter", "expected"),
    [
        (tiny_attn_only_adapter, "<bos>a"),
        (tiny_gemma3_adapter, "a"),
    ],
)
def test_only_the_attn_only_model_gets_a_bos_prepended(make_adapter, expected):
    adapter = make_adapter()
    source = adapter.make_source(["a"])
    source.to_model_batch(list(source.sample_ids()))
    assert adapter.tokenizer.seen == [expected]


def test_patching_after_nnsight_wraps_the_model_is_rejected(adapter):
    NNsight(adapter.get_model())
    with pytest.raises(RuntimeError, match="before wrapping"):
        adapter.patch_for_attn_lrp()


def test_nnsight_stores_the_forward_it_wraps():
    """The reason for the ordering rule. If NNsight ever stops doing this, the rule can go."""

    class Doubler(torch.nn.Module):
        def forward(self, x):
            return x * 2

    module = Doubler()
    NNsight(module)
    Doubler.forward = lambda self, x: x * 3
    try:
        assert module(torch.ones(1)).item() == 2.0
    finally:
        del Doubler.forward
