import torch

from in_tfm.neuron_tracking import NeuronTracker, StepRecord, cosine_distance, largest_steps, state_path
from in_tfm.stl_training import TrainConfig, train
from test_stl_training import Tiny

LAYER = "block_c.branch_1x1.0"


def record(step: int, distance: float) -> StepRecord:
    return StepRecord(step=step, epoch=0, lr=0.0, cosine_distance=distance, euclidean_distance=0.0)


def test_cosine_distance_is_zero_for_the_same_direction_and_one_for_orthogonal():
    a = torch.tensor([1.0, 0.0], dtype=torch.float64)
    assert cosine_distance(a, 3 * a) == 0.0
    assert cosine_distance(a, torch.tensor([0.0, 2.0], dtype=torch.float64)) == 1.0


def test_largest_steps_skip_neighbours_and_the_move_off_initialisation():
    distances = [9.0, 5.0, 4.9, 1.0, 4.0, 0.1, 3.0]
    records = [record(i + 1, d) for i, d in enumerate(distances)]
    picked = largest_steps(records, count=3, min_gap=2)
    assert [r.step for r in picked] == [2, 5, 7]


def test_largest_steps_can_include_the_first_step_when_asked():
    records = [record(1, 9.0), record(2, 1.0)]
    assert [r.step for r in largest_steps(records, 1, 1, skip_first_step=False)] == [1]


def test_the_tracker_saves_the_initial_state_and_one_state_per_step(tmp_path):
    config = TrainConfig(_cli_parse_args=[], out_dir=tmp_path / "out", epochs=1, batch_size=4, device="cpu")
    tracker = NeuronTracker(LAYER, 0, tmp_path / "states", steps_per_epoch=2)
    train(config, Tiny(8), Tiny(4), observer=tracker)
    assert len(tracker.records) == 2
    assert all(state_path(tmp_path / "states", step).exists() for step in range(3))
    first, second = (torch.load(state_path(tmp_path / "states", s))["model"] for s in (0, 1))
    key = f"{LAYER}.weight"
    assert not torch.equal(first[key], second[key])
    assert tracker.records[0].cosine_distance > 0
