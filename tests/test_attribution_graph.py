import pytest
import torch

from in_tfm.models.attn_only_2l import AttnOnlyConfig, AttnOnlyForCausalLM
from in_tfm.models.attn_only_2l.attribution_graph import (
    Embed, HeadBlock, HeadOut, OProjOut, PruneRule, Resid, ValueNeuron, build_graph, ov_state, parts, value,
)

SEQ = 8


@pytest.fixture
def state():
    torch.manual_seed(0)
    config = AttnOnlyConfig(hidden_size=32, num_attention_heads=4, head_dim=8, num_hidden_layers=2,
                            max_position_embeddings=SEQ, vocab_size=50, attention_bias=False)
    model = AttnOnlyForCausalLM(config).eval()
    return ov_state(model, torch.randint(0, 50, (1, SEQ)))


@pytest.mark.parametrize("node", [
    HeadBlock(layer=1, pos=7, neuron=3, head=2),
    HeadOut(layer=1, pos=6, head=1, dim=4),
    ValueNeuron(layer=1, pos=5, head=0, dim=2),
    Resid(pos=4, dim=9),
    OProjOut(layer=0, pos=3, neuron=11),
    HeadOut(layer=0, pos=2, head=3, dim=7),
    ValueNeuron(layer=0, pos=1, head=2, dim=5),
])
def test_parts_sum_to_the_node_value(state, node):
    assert sum(part for _, part in parts(state, node)) == pytest.approx(value(state, node), abs=1e-5)


def test_o_proj_out_matches_its_head_blocks(state):
    blocks = sum(value(state, HeadBlock(layer=0, pos=5, neuron=4, head=h)) for h in range(4))
    assert blocks == pytest.approx(value(state, OProjOut(layer=0, pos=5, neuron=4)), abs=1e-5)


def test_without_pruning_the_leaves_hold_the_root(state):
    root = HeadBlock(layer=1, pos=7, neuron=3, head=2)
    graph = build_graph(state, root, PruneRule(share=1.0, max_parents=10_000, floor=0.0))
    leaves = sum(n.relevance for n in graph.nodes if isinstance(n.node, Embed))
    assert leaves == pytest.approx(value(state, root), abs=1e-4)


def test_a_dead_end_keeps_no_share(state):
    root = HeadBlock(layer=1, pos=7, neuron=3, head=2)
    graph = build_graph(state, root, PruneRule(share=0.8, max_parents=5, floor=0.05))
    children = {e.child for e in graph.edges}
    inner = [n for n in graph.nodes if not isinstance(n.node, Embed)]
    assert all((n.kept_share == 0.0) == (n.node not in children) for n in inner)
