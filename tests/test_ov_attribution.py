from typing import cast

import numpy as np
import pytest
import torch
from transformers import PreTrainedTokenizerBase

from in_tfm.layers import submodule_getter
from in_tfm.models.attn_only_2l import (
    AttnOnlyConfig,
    AttnOnlyForCausalLM,
    TinyAttnOnly1LD128Adapter,
)
from in_tfm.models.attn_only_2l.ov import compute_ov_relevance
from in_tfm.neuron_run import attr_fn_for
from in_tfm.run_config import load_run_config
from in_tfm.sources import ModelBatch

N_HEADS, HEAD_DIM, HIDDEN, VOCAB, LENGTH = 2, 4, 8, 20, 6
FIRING = 4


class FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    bos_token = "<bos>"
    bos_token_id = VOCAB - 1

    def __call__(self, texts, **kwargs):
        ids = torch.randint(0, VOCAB, (len(texts), kwargs["max_length"]))
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def model_with_layers(n_layers: int) -> AttnOnlyForCausalLM:
    """Weights far above the 0.02 init, so attention is not near uniform and every term matters."""
    torch.manual_seed(0)
    config = AttnOnlyConfig(
        hidden_size=HIDDEN,
        num_attention_heads=N_HEADS,
        head_dim=HEAD_DIM,
        num_hidden_layers=n_layers,
        max_position_embeddings=LENGTH,
        vocab_size=VOCAB,
        attention_bias=False,
    )
    model = AttnOnlyForCausalLM(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 1)
    return model


def one_sample(model: AttnOnlyForCausalLM) -> ModelBatch:
    tokenizer = cast(PreTrainedTokenizerBase, FakeTokenizer())
    adapter = TinyAttnOnly1LD128Adapter(model, tokenizer, max_length=LENGTH)
    source = adapter.make_source(["a"])
    return source.to_model_batch(list(source.sample_ids()))


def layer_output(model, batch, layer_name: str) -> torch.Tensor:
    captured = {}
    layer = submodule_getter(layer_name)(model)
    handle = layer.register_forward_hook(
        lambda module, args, out: captured.__setitem__("out", out)
    )
    with torch.no_grad():
        model(**batch.kwargs)
    handle.remove()
    return captured["out"][0]


@pytest.mark.parametrize(
    "layer_name",
    ["model.layers.0.self_attn.o_proj.heads.1", "model.layers.0.self_attn.o_proj"],
)
def test_a_heads_write_splits_exactly_over_source_tokens(layer_name):
    model = model_with_layers(1)
    batch = one_sample(model)
    relevance = compute_ov_relevance(
        model, batch, submodule_getter(layer_name), 3, FIRING
    )
    assert relevance.shape == (1, LENGTH, 1)
    expected = layer_output(model, batch, layer_name)[FIRING, 3].item()
    assert relevance.sum() == pytest.approx(expected, rel=1e-4)
    assert np.all(relevance[0, FIRING + 1 :] == 0)


def test_a_logit_splits_into_the_embedding_and_the_value_vectors():
    model = model_with_layers(1)
    batch = one_sample(model)
    relevance = compute_ov_relevance(
        model, batch, submodule_getter("lm_head"), 7, FIRING
    )
    assert relevance.shape == (1, LENGTH, HIDDEN)
    logit = layer_output(model, batch, "lm_head")[FIRING, 7].item()
    assert relevance.sum() == pytest.approx(
        logit - model.lm_head.bias[7].item(), rel=1e-4
    )


def test_the_embedding_term_sits_on_the_firing_token_alone():
    model = model_with_layers(1)
    with torch.no_grad():
        model.model.layers[0].self_attn.v_proj.weight.zero_()
    batch = one_sample(model)
    relevance = compute_ov_relevance(
        model, batch, submodule_getter("lm_head"), 7, FIRING
    )
    per_token = relevance[0].sum(-1)
    assert np.count_nonzero(per_token) == 1
    assert per_token[FIRING] != 0


def test_two_layers_are_rejected():
    model = model_with_layers(2)
    with pytest.raises(ValueError, match="single layer"):
        compute_ov_relevance(
            model, one_sample(model), submodule_getter("lm_head"), 0, FIRING
        )


def test_the_attnlrp_patch_a_run_installs_changes_nothing():
    model = model_with_layers(1)
    batch = one_sample(model)
    getter = submodule_getter("lm_head")
    plain = compute_ov_relevance(model, batch, getter, 7, FIRING)
    tokenizer = cast(PreTrainedTokenizerBase, FakeTokenizer())
    TinyAttnOnly1LD128Adapter(model, tokenizer, max_length=LENGTH).patch_for_attn_lrp()
    np.testing.assert_allclose(
        compute_ov_relevance(model, batch, getter, 7, FIRING), plain, rtol=1e-5
    )


def test_a_run_config_picks_ov_only_for_an_attn_only_model():
    config = load_run_config(
        ["--model", "self/tiny-attn-only-1l-d128", "--attribution", "ov"]
    )
    tokenizer = cast(PreTrainedTokenizerBase, FakeTokenizer())
    adapter = TinyAttnOnly1LD128Adapter(
        model_with_layers(1), tokenizer, max_length=LENGTH
    )
    assert attr_fn_for(config, adapter) is compute_ov_relevance
    default = load_run_config(["--model", "self/tiny-attn-only-1l-d128"])
    assert attr_fn_for(default, adapter) is adapter.attr_fn
