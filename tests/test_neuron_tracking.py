import torch

from in_tfm.layers import submodule_getter
from in_tfm.models.stl_inception import StlInception
from in_tfm.neuron_tracking import NeuronTracker, StepRecord, cosine_distance, largest_steps, neuron_weights, state_path
from in_tfm.stl_training import TrainConfig, train
from test_stl_training import Tiny

LAYER = "block_c.branch_1x1.0"


def record(step: int, distance: float) -> StepRecord:
    return StepRecord(step=step, epoch=0, lr=0.0, cosine_distance=distance, euclidean_distance=0.0)


def test_cosine_distance_is_zero_for_the_same_direction_and_one_for_orthogonal():
    a = torch.tensor([1.0, 0.0], dtype=torch.float64)
    assert cosine_distance(a, 3 * a) == 0.0
    assert cosine_distance(a, torch.tensor([0.0, 2.0], dtype=torch.float64)) == 1.0


def test_largest_steps_are_the_raw_top_ones_by_default_so_neighbours_and_step_1_count():
    records = [record(i + 1, d) for i, d in enumerate([9.0, 5.0, 4.9, 1.0])]
    assert [r.step for r in largest_steps(records, count=3)] == [1, 2, 3]


def test_a_gap_skips_neighbours_of_a_larger_step():
    distances = [9.0, 5.0, 4.9, 1.0, 4.0, 0.1, 3.0]
    records = [record(i + 1, d) for i, d in enumerate(distances)]
    assert [r.step for r in largest_steps(records, count=3, min_gap=2)] == [1, 3, 5]


def test_the_neuron_weights_are_the_row_the_hadamard_product_uses():
    model = StlInception()
    layer = submodule_getter(LAYER)(model)
    row = neuron_weights(model, submodule_getter(LAYER), 3)
    assert row.shape == (layer.weight.shape[1],)
    assert torch.equal(row.float(), layer.weight[3].flatten().detach())


def test_the_tracker_saves_the_initial_state_and_one_state_per_step(tmp_path):
    config = TrainConfig(_cli_parse_args=[], out_dir=tmp_path / "out", epochs=1, batch_size=4, device="cpu")
    tracker = NeuronTracker(submodule_getter(LAYER), LAYER, 0, tmp_path / "states", steps_per_epoch=2)
    train(config, Tiny(8), Tiny(4), observer=tracker)
    assert len(tracker.records) == 2
    assert all(state_path(tmp_path / "states", step).exists() for step in range(3))
    first, second = (torch.load(state_path(tmp_path / "states", s))["model"] for s in (0, 1))
    key = f"{LAYER}.weight"
    assert not torch.equal(first[key], second[key])
    assert tracker.records[0].cosine_distance > 0
