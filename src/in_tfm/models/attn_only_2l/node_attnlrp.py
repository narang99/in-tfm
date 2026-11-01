"""AttnLRP of any attribution graph node, on the input tokens.

- The model must be patched with `patch_attn_only_for_attn_lrp` first.
- One forward pass keeps its graph, and each node is one backward pass from its scalar.
- Unlike the graph's edges, this goes through QK as well, by AttnLRP's softmax rule.
"""

import torch
from jaxtyping import Float, Int

from .attribution_graph import Coordinate, Embed, HeadBlock, HeadOut, OProjOut, Resid, ValueNeuron
from .model import AttnOnlyForCausalLM


class _Tape:
    """The tensors a node can name, kept with their autograd history."""

    def __init__(self, model: AttnOnlyForCausalLM, input_ids: Int[torch.Tensor, "1 seq"]):
        self.model = model
        self.embeds = model.model.embed_tokens(input_ids).detach().requires_grad_(True)
        self.o_proj_in: dict[int, torch.Tensor] = {}
        self.o_proj_out: dict[int, torch.Tensor] = {}
        self.v_proj_out: dict[int, torch.Tensor] = {}
        self.resid_into_layer_1: torch.Tensor | None = None
        handles = []
        for layer, block in enumerate(model.model.layers):
            attn = block.self_attn
            handles.append(attn.o_proj.register_forward_hook(self._record_o_proj(layer)))
            handles.append(attn.v_proj.register_forward_hook(self._record_v_proj(layer)))
        handles.append(model.model.layers[1].register_forward_pre_hook(self._record_resid))
        try:
            with torch.enable_grad():
                model(inputs_embeds=self.embeds)
        finally:
            for handle in handles:
                handle.remove()

    def _record_o_proj(self, layer: int):
        def hook(_module, inp, out):
            self.o_proj_in[layer], self.o_proj_out[layer] = inp[0][0], out[0]
        return hook

    def _record_v_proj(self, layer: int):
        def hook(_module, _inp, out):
            self.v_proj_out[layer] = out[0]
        return hook

    def _record_resid(self, _module, args):
        self.resid_into_layer_1 = args[0][0]

    def scalar(self, node: Coordinate) -> torch.Tensor:
        head_dim = self.model.config.head_dim
        match node:
            case HeadBlock(layer=layer, pos=pos, neuron=neuron, head=head):
                block = slice(head * head_dim, (head + 1) * head_dim)
                weight = self.model.model.layers[layer].self_attn.o_proj.weight[neuron, block]
                return self.o_proj_in[layer][pos, block] @ weight
            case HeadOut(layer=layer, pos=pos, head=head, dim=dim):
                return self.o_proj_in[layer][pos, head * head_dim + dim]
            case ValueNeuron(layer=layer, pos=pos, head=head, dim=dim):
                return self.v_proj_out[layer][pos, head * head_dim + dim]
            case Resid(pos=pos, dim=dim):
                return self.resid_into_layer_1[pos, dim]
            case OProjOut(layer=layer, pos=pos, neuron=neuron):
                return self.o_proj_out[layer][pos, neuron]
            case Embed(pos=pos, dim=dim):
                return self.embeds[0, pos, dim]
        raise ValueError(f"no scalar for {node}")


def node_token_relevance(
    model: AttnOnlyForCausalLM, input_ids: Int[torch.Tensor, "1 seq"], nodes: list[Coordinate]
) -> dict[Coordinate, Float[torch.Tensor, "seq"]]:
    model.eval()
    tape = _Tape(model, input_ids)
    relevance = {}
    for node in nodes:
        tape.embeds.grad = None
        tape.scalar(node).backward(retain_graph=True)
        grad = tape.embeds.grad if tape.embeds.grad is not None else torch.zeros_like(tape.embeds)
        relevance[node] = (tape.embeds * grad)[0].sum(dim=-1).detach()
    model.zero_grad(set_to_none=True)
    return relevance
