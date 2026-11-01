"""An attribution graph down the OV paths of the 2-layer attention-only model, from one (neuron, head) root.

The skeleton follows hiccup-ide's graph builder: immutable coordinates are the nodes, each coordinate type knows its
parents, and a pruning rule decides which parents are kept.

- Every node is one scalar in the forward pass, and every edge is that scalar's exact split over its parents.
  - Attention is held fixed, so each step is linear and the parts of a node sum to its value.
  - The model has no biases and no normalisation, so nothing is left over.
- The chain, from the root towards the input:
  - `HeadBlock`: one head's block of one o_proj neuron, sum_j z[h, j] W_O[n, hj].
  - `HeadOut`: one dim of a head's output z, the o_proj input. It splits over sources s as A_h[t, s] v_s[j].
  - `ValueNeuron`: one v_proj output, x_s . W_V[hj]. It splits over the dims d of its input x_s.
    - Layer 1 reads the residual stream, so its parents are `Resid` nodes.
    - Layer 0 reads the embedding, so its parents are `Embed` nodes.
  - `Resid`: one residual dim into layer 1. It is a junction: W_E's value plus layer 0's o_proj neuron d.
  - `OProjOut`: one layer 0 o_proj neuron, split over all heads' z dims as z[h, j] W_O[n, hj].
  - `Embed`: one dim of one token's embedding. A leaf.
- Relevance starts as the root's value and flows to parents in proportion to their parts, which is LRP-0.
  - With everything linear, a node's relevance is the sum over paths from the root of the product of edge shares.
"""

from typing import Annotated, Literal, Union

import torch
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict, Field

from .model import AttnOnlyForCausalLM
from .ov_graph import ov_flows


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class HeadBlock(Frozen):
    type: Literal["HeadBlock"] = "HeadBlock"
    layer: int
    pos: int
    neuron: int
    head: int


class HeadOut(Frozen):
    type: Literal["HeadOut"] = "HeadOut"
    layer: int
    pos: int
    head: int
    dim: int


class ValueNeuron(Frozen):
    type: Literal["ValueNeuron"] = "ValueNeuron"
    layer: int
    pos: int
    head: int
    dim: int


class Resid(Frozen):
    """The residual stream into layer 1. The stream into layer 0 is the embedding itself, so it has no junction."""

    type: Literal["Resid"] = "Resid"
    pos: int
    dim: int


class OProjOut(Frozen):
    type: Literal["OProjOut"] = "OProjOut"
    layer: int
    pos: int
    neuron: int


class Embed(Frozen):
    type: Literal["Embed"] = "Embed"
    pos: int
    dim: int


Coordinate = Annotated[Union[HeadBlock, HeadOut, ValueNeuron, Resid, OProjOut, Embed], Field(discriminator="type")]
Parts = list[tuple[Coordinate, float]]

LAST_STAGE = 7


def stage(node: Coordinate) -> int:
    """Columns of the graph, root first. Every edge goes to a later stage, so stages are a topological order."""
    match node:
        case HeadBlock():
            return 0
        case HeadOut(layer=1):
            return 1
        case ValueNeuron(layer=1):
            return 2
        case Resid():
            return 3
        case OProjOut():
            return 4
        case HeadOut(layer=0):
            return 5
        case ValueNeuron(layer=0):
            return 6
        case Embed():
            return LAST_STAGE
    raise ValueError(f"no stage for {node}")


class OvState(BaseModel):
    """Every scalar the graph can name, from one forward pass of a batch of one."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    embeds: Float[torch.Tensor, "seq hidden"]
    patterns: list[Float[torch.Tensor, "head seq seq"]]
    values: list[Float[torch.Tensor, "seq head head_dim"]]
    head_outputs: list[Float[torch.Tensor, "seq head head_dim"]]
    o_proj_out: list[Float[torch.Tensor, "seq hidden"]]
    w_v: list[Float[torch.Tensor, "inner hidden"]]
    w_o: list[Float[torch.Tensor, "hidden inner"]]
    head_dim: int

    @property
    def resid_into_layer_1(self) -> Float[torch.Tensor, "seq hidden"]:
        return self.embeds + self.o_proj_out[0]


@torch.no_grad()
def ov_state(model: AttnOnlyForCausalLM, input_ids: Int[torch.Tensor, "1 seq"]) -> OvState:
    patterns, values, head_outputs, o_proj_out, w_v, w_o = [], [], [], [], [], []
    for layer in range(model.config.num_hidden_layers):
        attn = model.model.layers[layer].self_attn
        flows = ov_flows(model, input_ids, layer)
        z = torch.einsum("hts,hsd->thd", flows.pattern, flows.values)
        patterns.append(flows.pattern)
        values.append(flows.values.transpose(0, 1))
        head_outputs.append(z)
        o_proj_out.append(flows.outputs)
        w_v.append(attn.v_proj.weight.detach())
        w_o.append(attn.o_proj.weight.detach())
    return OvState(embeds=model.model.embed_tokens.weight[input_ids[0]].detach(), patterns=patterns, values=values,
                   head_outputs=head_outputs, o_proj_out=o_proj_out, w_v=w_v, w_o=w_o,
                   head_dim=model.config.head_dim)


def value(state: OvState, node: Coordinate) -> float:
    match node:
        case HeadBlock(layer=layer, pos=pos, neuron=neuron, head=head):
            return float(state.head_outputs[layer][pos, head] @ _o_block(state, layer, neuron, head))
        case HeadOut(layer=layer, pos=pos, head=head, dim=dim):
            return float(state.head_outputs[layer][pos, head, dim])
        case ValueNeuron(layer=layer, pos=pos, head=head, dim=dim):
            return float(state.values[layer][pos, head, dim])
        case Resid(pos=pos, dim=dim):
            return float(state.resid_into_layer_1[pos, dim])
        case OProjOut(layer=layer, pos=pos, neuron=neuron):
            return float(state.o_proj_out[layer][pos, neuron])
        case Embed(pos=pos, dim=dim):
            return float(state.embeds[pos, dim])
    raise ValueError(f"no value for {node}")


def parts(state: OvState, node: Coordinate) -> Parts:
    """Every parent with its part of the node's value. The parts sum to `value(state, node)`."""
    match node:
        case HeadBlock(layer=layer, pos=pos, neuron=neuron, head=head):
            terms = state.head_outputs[layer][pos, head] * _o_block(state, layer, neuron, head)
            return [(HeadOut(layer=layer, pos=pos, head=head, dim=j), float(terms[j])) for j in range(len(terms))]
        case OProjOut(layer=layer, pos=pos, neuron=neuron):
            terms = state.head_outputs[layer][pos].reshape(-1) * state.w_o[layer][neuron]
            return [(HeadOut(layer=layer, pos=pos, head=i // state.head_dim, dim=i % state.head_dim), float(terms[i]))
                    for i in range(len(terms))]
        case HeadOut(layer=layer, pos=pos, head=head, dim=dim):
            terms = state.patterns[layer][head, pos, :pos + 1] * state.values[layer][:pos + 1, head, dim]
            return [(ValueNeuron(layer=layer, pos=s, head=head, dim=dim), float(terms[s])) for s in range(pos + 1)]
        case ValueNeuron(layer=layer, pos=pos, head=head, dim=dim):
            reads = state.resid_into_layer_1[pos] if layer == 1 else state.embeds[pos]
            terms = reads * state.w_v[layer][head * state.head_dim + dim]
            parent = Resid if layer == 1 else Embed
            return [(parent(pos=pos, dim=d), float(terms[d])) for d in range(len(terms))]
        case Resid(pos=pos, dim=dim):
            return [(Embed(pos=pos, dim=dim), float(state.embeds[pos, dim])),
                    (OProjOut(layer=0, pos=pos, neuron=dim), float(state.o_proj_out[0][pos, dim]))]
        case Embed():
            return []
    raise ValueError(f"no parents for {node}")


def _o_block(state: OvState, layer: int, neuron: int, head: int) -> Float[torch.Tensor, "head_dim"]:
    return state.w_o[layer][neuron, head * state.head_dim:(head + 1) * state.head_dim]


class PruneRule(BaseModel):
    share: float = 0.8
    """Per sign, the fewest parents whose relevance reaches this share of that sign's total."""
    max_parents: int = 5
    floor: float = 0.01
    """Nodes and edges under this share of the root's |relevance| are dropped."""


def kept_parents(edge_relevance: list[tuple[Coordinate, float]], rule: PruneRule) -> list[tuple[Coordinate, float]]:
    kept: list[tuple[Coordinate, float]] = []
    for sign in (1, -1):
        side = sorted((e for e in edge_relevance if sign * e[1] > 0), key=lambda e: -abs(e[1]))
        total = sum(abs(r) for _, r in side)
        held = 0.0
        for edge in side:
            if held >= rule.share * total:
                break
            kept.append(edge)
            held += abs(edge[1])
    return sorted(kept, key=lambda e: -abs(e[1]))[:rule.max_parents]


class GraphNode(BaseModel):
    node: Coordinate
    stage: int
    value: float
    relevance: float
    kept_share: float
    """The share of this node's relevance its kept parents carry. 1 for a leaf."""


class GraphEdge(BaseModel):
    child: Coordinate
    parent: Coordinate
    relevance: float


class AttributionGraph(BaseModel):
    root: Coordinate
    nodes: list[GraphNode]
    edges: list[GraphEdge]


def build_graph(state: OvState, root: HeadBlock, rule: PruneRule) -> AttributionGraph:
    """Expands stage by stage, so a node's relevance is complete before it is split further."""
    relevance: dict[Coordinate, float] = {root: value(state, root)}
    floor = rule.floor * abs(relevance[root])
    edges: dict[tuple[Coordinate, Coordinate], float] = {}
    for current in range(LAST_STAGE + 1):
        for node in _drop_below_floor(relevance, edges, current, floor):
            node_value = value(state, node)
            node_parts = parts(state, node)
            if not node_parts or abs(node_value) < 1e-12:
                continue
            scale = relevance[node] / node_value
            kept = [(p, r) for p, r in kept_parents([(p, part * scale) for p, part in node_parts], rule)
                    if abs(r) >= floor]
            for parent, edge_relevance in kept:
                edges[(node, parent)] = edge_relevance
                relevance[parent] = relevance.get(parent, 0.0) + edge_relevance
    carried: dict[Coordinate, float] = {}
    for (child, _), edge_relevance in edges.items():
        carried[child] = carried.get(child, 0.0) + edge_relevance
    return AttributionGraph(
        root=root,
        nodes=[GraphNode(node=n, stage=stage(n), value=value(state, n), relevance=r,
                         kept_share=carried[n] / r if n in carried else 1.0)
               for n, r in relevance.items()],
        edges=[GraphEdge(child=c, parent=p, relevance=r) for (c, p), r in edges.items()],
    )


def _drop_below_floor(relevance: dict, edges: dict, current: int, floor: float) -> list:
    """The nodes of one stage that survive, largest first. Dropped nodes take their incoming edges with them."""
    at_stage = [n for n in relevance if stage(n) == current]
    for node in at_stage:
        if abs(relevance[node]) < floor:
            del relevance[node]
            for key in [k for k in edges if k[1] == node]:
                del edges[key]
    return sorted((n for n in at_stage if n in relevance), key=lambda n: -abs(relevance[n]))
