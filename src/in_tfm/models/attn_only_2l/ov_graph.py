"""The OV path of one layer, split into the flow from each V through each head into each O neuron.

- o_n(t) = sum over heads h and sources s of A_h[t, s] * (v_{h,s} . W_O[n, block h]).
  - Each term is one V flowing through one head into one O neuron.
  - The attention pattern is held fixed, so no relevance goes to QK.
- In layer 0 each v_{h,s} is a function of token s alone, since shortformer keeps position out of the value path.
"""

import torch
from jaxtyping import Float, Int
from pydantic import BaseModel

from .model import AttnOnlyForCausalLM, _projection_outputs, _softmax_pattern


class OvFlows(BaseModel):
    """`flow[h, t, s, n]` is `pattern[h, t, s] * value_scores[h, s, n]`, built on demand since it is seq x seq x out."""

    model_config = {"arbitrary_types_allowed": True}

    pattern: Float[torch.Tensor, "head seq seq"]
    value_scores: Float[torch.Tensor, "head seq out"]
    """v_{h,s} . W_O[n, block h]: what V (h, s) adds to neuron n at full attention."""
    outputs: Float[torch.Tensor, "seq out"]


def ov_flows(model: AttnOnlyForCausalLM, input_ids: Int[torch.Tensor, "1 seq"], layer_idx: int) -> OvFlows:
    attn = model.model.layers[layer_idx].self_attn
    captured = _projection_outputs(model, input_ids, layer_idx, ("q", "k", "v"))
    query, key, value = (attn._to_heads(captured[name])[0] for name in ("q", "k", "v"))
    blocks = attn.o_proj.weight.detach().view(-1, attn.n_heads, attn.head_dim)
    value_scores = torch.einsum("hsd,nhd->hsn", value, blocks)
    pattern = _softmax_pattern(attn, query, key)
    outputs = torch.einsum("hts,hsn->tn", pattern, value_scores)
    return OvFlows(pattern=pattern, value_scores=value_scores, outputs=outputs)
