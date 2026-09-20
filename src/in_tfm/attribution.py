"""Pixel-space attribution for a single MLP neuron: Integrated Gradients, and AttnLRP via lxt.

AttnLRP needs `lxt.efficient` patched onto Dinov2's ops before any forward/backward pass - see
`patch_for_attn_lrp`. `embs_dinov2.compat` is imported first (below) since `lxt.efficient.core`
transitively imports a submodule that expects a `pytorch_utils` function newer `transformers`
versions removed; see that module for why.
"""

import types
from functools import partial

import numpy as np
import torch
from captum.attr import NeuronIntegratedGradients
from jaxtyping import Float

from . import compat  # noqa: F401  (import order matters - see compat.py)
from lxt.efficient.core import monkey_patch
from lxt.efficient.patches import (
    check_already_patched,
    dropout_forward,
    layer_norm_forward,
    non_linear_forward,
    patch_method,
    wrap_attention_forward,
)
import transformers.models.dinov2.modeling_dinov2 as modeling_dinov2
from transformers.models.dinov2.modeling_dinov2 import Dinov2Model

from .layers import LayerGetter
from .sources import ModelBatch


def neuron_ig(
    model: Dinov2Model,
    pixel_values: Float[torch.Tensor, "batch c h w"],
    layer_getter: LayerGetter,
    neuron_idx: int,
    token_idx: int,
) -> Float[torch.Tensor, "batch h w"]:
    """Sums over the channel axis before returning, matching compute_attnlrp_relevance's
    contract so callers (e.g. mk_overlay) don't need to care which attribution method produced
    the map.
    """
    forward_fn = lambda px: model(pixel_values=px)
    ig = NeuronIntegratedGradients(forward_fn, layer_getter(model))
    attr = ig.attribute(pixel_values, (token_idx, neuron_idx), internal_batch_size=4)
    return attr.sum(dim=1)


def patch_eager_attention(module: types.ModuleType) -> bool:
    """lxt's own patch_attention() replaces ALL_ATTENTION_FUNCTIONS with a plain dict, which
    breaks this transformers version's AttentionInterface.get_interface(). Since
    get_interface("eager", default) just returns `default` unconditionally, it's enough (and
    simpler) to patch the module-level eager_attention_forward fallback directly, as long as
    attn_implementation="eager" is forced by the caller.

    Works for any modeling module exposing eager_attention_forward, which is the shared
    interface Dinov2, Llama, Qwen and Gemma all use.
    """
    new_forward = wrap_attention_forward(module.eager_attention_forward)
    if check_already_patched(module.eager_attention_forward, new_forward):
        return False
    module.eager_attention_forward = new_forward
    return True


def patch_for_attn_lrp(model: Dinov2Model) -> None:
    """Monkey-patch a RadDino/Dinov2 model in place for AttnLRP.

    lxt.efficient has no built-in support for Dinov2Model, but Dinov2's attention already uses
    the same ALL_ATTENTION_FUNCTIONS/eager_attention_forward interface as Llama/Qwen/Gemma, so
    the primitives lxt.efficient.patches ships for those models cover Dinov2 too:
      - LayerNorm -> identity rule (stop-gradient through mean/std)
      - the MLP activation -> identity rule (gradient*input reproduces f(x) exactly)
      - Dropout -> identity, in case .train() is ever used
      - attention (Q@K^T, attn@V) -> uniform rule, via patch_dinov2_attention
    nn.Linear/nn.Conv2d and Dinov2's LayerScale (multiply by a learned, input-independent
    vector) need no patch: plain gradient*input is already a valid LRP rule for anything
    linear in the input.
    """
    # Discovered from the live model rather than hardcoded, so this keeps working if the
    # checkpoint's config.hidden_act ever resolves to a different ACT2FN class.
    activation_cls = type(model.encoder.layer[0].mlp.activation)

    patch_map = {
        torch.nn.LayerNorm: partial(patch_method, layer_norm_forward),
        torch.nn.Dropout: partial(patch_method, dropout_forward),
        activation_cls: partial(patch_method, non_linear_forward, keep_original=True),
        modeling_dinov2: patch_eager_attention,
    }

    # NOTE: LayerNorm/Dropout are patched at the class level, i.e. process-wide - any other
    # model sharing this process will also get LRP-flavored LayerNorm/Dropout.
    model.config._attn_implementation = "eager"
    monkey_patch(model, patch_map=patch_map, verbose=True)


def compute_attnlrp_relevance(
    model: torch.nn.Module,
    batch: ModelBatch,
    layer_getter: LayerGetter,
    neuron_idx: int,
    token_idx: int | None,
) -> Float[np.ndarray, "batch ..."]:
    """AttnLRP relevance for one MLP neuron, via gradient*input on the batch's gradient leaf.

    With `patch_for_attn_lrp` applied, backpropagating from any single scalar produces
    AttnLRP-conservative relevance at every patched op along the way; grad*input at the
    (unpatched, but linear) leaf then gives per-element relevance for free.

    The leaf comes from the batch rather than being assumed to be the model input: images
    differentiate `pixel_values` directly, text cannot differentiate integer `input_ids` and
    substitutes embeddings. See sources.ModelBatch.

    Returns relevance *unreduced* - still carrying the leaf's feature axis (channels for
    images, hidden for text). Collapsing it is the presenter's call, since which axis to sum
    differs by modality.

    token_idx=None explains the neuron's activation summed over all tokens ("this neuron
    firing anywhere in the input"); pass a single flat token index to instead explain just
    that one high-activating position.
    """
    model.eval()
    kwargs, leaf = batch.differentiable()

    captured: dict[str, torch.Tensor] = {}
    handle = layer_getter(model).register_forward_hook(
        lambda module, inp, out: captured.__setitem__("layer_out", out)
    )
    try:
        with torch.enable_grad():
            model(**kwargs)
            layer_out = captured["layer_out"][..., neuron_idx]  # (batch, seq_len)
            target = layer_out[:, token_idx] if token_idx is not None else layer_out.sum(dim=1)
            target.sum().backward()
    finally:
        handle.remove()

    if leaf.grad is None:
        # nothing connected the traced layer back to the leaf - usually the source named a
        # grad_leaf_key the model doesn't actually consume, so the forward pass ignored it
        raise RuntimeError(
            f"no gradient reached {batch.grad_leaf_key!r}; check that the model consumes it "
            f"(got kwargs: {sorted(batch.kwargs)})"
        )
    return (leaf.grad * leaf).detach().cpu().numpy()


def patch_gemma3_for_attn_lrp(model: torch.nn.Module) -> None:
    """Monkey-patch a Gemma3 model in place for AttnLRP.

    lxt ships a gemma3 patch map, and its MLP/RMSNorm rules are reused verbatim here. Only its
    attention entry is swapped: lxt's `patch_attention` is the broken-on-transformers-5.x path
    described in patch_eager_attention.

    Gemma3RMSNorm gets the same treatment LayerNorm gets in the vision path - stop-gradient
    through the normalizing statistic, so relevance flows only through the scaled input.
    """
    from lxt.efficient.models.gemma3 import gemma3_norm
    from lxt.efficient.patches import gated_mlp_forward
    from transformers.models.gemma3 import modeling_gemma3
    from transformers.models.gemma3.modeling_gemma3 import Gemma3MLP, Gemma3RMSNorm

    patch_map = {
        Gemma3MLP: partial(patch_method, gated_mlp_forward),
        Gemma3RMSNorm: partial(patch_method, gemma3_norm, method_name="_norm"),
        torch.nn.Dropout: partial(patch_method, dropout_forward),
        modeling_gemma3: patch_eager_attention,
    }
    model.config._attn_implementation = "eager"
    monkey_patch(model, patch_map=patch_map, verbose=True)
