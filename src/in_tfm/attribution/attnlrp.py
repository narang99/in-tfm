"""AttnLRP relevance for a single MLP neuron, via lxt.

AttnLRP needs `lxt.efficient` patched onto the model's ops before any forward/backward pass.
Each architecture's patch lives with its adapter in `models`. `compat` is imported first (below)
since `lxt.efficient.core` transitively imports a submodule that expects a `pytorch_utils`
function newer `transformers` versions removed; see that module for why.
"""

import types

import numpy as np
import torch
from jaxtyping import Float

from .. import compat  # noqa: F401  (import order matters - see compat.py)
from lxt.efficient.patches import check_already_patched, wrap_attention_forward

from ..layers import LayerGetter
from ..sources import ModelBatch


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
