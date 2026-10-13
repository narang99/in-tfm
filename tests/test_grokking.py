import torch

from in_tfm.grokking_tracking import LossDrop, largest_drops, loss_drops, most_token_selective_neurons, saved_epochs, states_to_report
from in_tfm.grokking_training import EpochMetrics
from in_tfm.models.grokking import MODULUS, GrokTransformer, GrokkingAdapter
from in_tfm.models.grokking.data import all_pairs, pairs_for, split_pairs
from in_tfm.models.grokking.source import PairSource


def metrics(test_losses: list[float]) -> list[EpochMetrics]:
    return [EpochMetrics(epoch=e, train_loss=0.0, test_loss=t, train_acc=1.0, test_acc=0.0) for e, t in enumerate(test_losses)]


def test_the_logits_are_read_at_the_equals_position_over_the_residues():
    model = GrokTransformer()
    tokens, labels = all_pairs()
    assert model.logits_for_tokens(tokens[:5]).shape == (5, 3, MODULUS)
    assert torch.equal(labels[:3], torch.tensor([0, 1, 2]))


def test_the_split_is_seeded_and_partitions_all_pairs():
    first, again = split_pairs(0.3, 1), split_pairs(0.3, 1)
    assert torch.equal(first.train_tokens, again.train_tokens)
    assert len(first.train_tokens) + len(first.test_tokens) == MODULUS**2
    assert len(first.train_tokens) == int(0.3 * MODULUS**2)


def test_distinct_pairs_have_each_residue_once_on_each_side():
    pairs = pairs_for("distinct", 0.3, 0)
    assert sorted(a for a, _ in pairs) == sorted(b for _, b in pairs) == list(range(MODULUS))


def test_a_value_neuron_depends_on_its_own_token_only():
    model = GrokTransformer().eval()
    source = PairSource([(3, 4), (3, 9)], model)
    batch = source.to_model_batch(source.sample_ids())
    captured: list[torch.Tensor] = []
    model.layers[0].self_attn.v_proj.register_forward_hook(lambda m, i, o: captured.append(o))
    model(**batch.kwargs)
    assert torch.allclose(captured[0][0, 0], captured[0][1, 0])
    assert not torch.allclose(captured[0][0, 1], captured[0][1, 1])


def test_attnlrp_puts_all_of_a_value_neurons_relevance_on_its_own_token():
    adapter = GrokkingAdapter(GrokTransformer().eval())
    adapter.patch_for_attn_lrp()
    source = adapter.make_source([(5, 7)])
    batch = source.to_model_batch(source.sample_ids())
    getter = lambda model: model.layers[0].self_attn.v_proj
    relevance = adapter.attr_fn(adapter.get_model(), batch, getter, 3, 1)
    per_token = relevance.sum(-1)[0]
    assert per_token[1] != 0
    assert per_token[0] == per_token[2] == 0


def test_the_largest_drops_are_taken_between_saved_states_in_training_order():
    series = metrics([10.0, 10.0, 9.0, 9.0, 3.0, 3.0, 2.0])
    drops = loss_drops(series, stride=2)
    assert [(d.before, d.after) for d in drops] == [(0, 2), (2, 4), (4, 6)]
    assert largest_drops(drops, 1)[0].before == 2


def test_min_gap_keeps_the_chosen_drops_apart():
    drops = [LossDrop(before=b, after=b + 1, test_loss_before=l, test_loss_after=0.0) for b, l in [(0, 9.0), (1, 8.0), (5, 3.0)]]
    assert [d.before for d in largest_drops(drops, 2, min_gap=3)] == [0, 5]


def test_each_drop_reports_the_state_before_and_after_it_plus_initial_and_final():
    chosen = [LossDrop(before=20, after=30, test_loss_before=5.0, test_loss_after=1.0)]
    assert [s.state for s in states_to_report(chosen, final_state=100)] == [0, 20, 30, 100]
    assert saved_epochs(metrics([0.0] * 7), stride=3) == [0, 3, 6]


def test_one_neuron_per_head_is_picked_with_its_flat_index_inside_that_head():
    neurons = most_token_selective_neurons(GrokTransformer())
    assert [n // 32 for n in neurons] == [0, 1, 2, 3]


def test_a_drop_can_span_several_strides():
    from in_tfm.grokking_tracking import drop_between

    drop = drop_between(metrics([8.0, 7.0, 6.0, 1.0]), 0, 3)
    assert (drop.before, drop.after, drop.drop) == (0, 3, 7.0)
