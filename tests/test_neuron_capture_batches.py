import numpy as np

from in_tfm.neuron_capture import NeuronCapture


def capture_with_batch_size(batch_size: int) -> NeuronCapture:
    capture = NeuronCapture.__new__(NeuronCapture)
    capture.batch_size = batch_size
    return capture


def test_gather_reruns_whole_scan_batches_holding_a_hit():
    capture = capture_with_batch_size(4)
    hits = np.array([1, 2, 9, 9, 10])
    batches = capture._scan_batches_holding(hits, n_samples=11)
    assert [b.tolist() for b in batches] == [[0, 1, 2, 3], [8, 9, 10]]


def test_inputs_past_the_known_cutoffs_are_kept_and_read_back():
    import pytest
    import torch

    from in_tfm.neuron_capture import KeptInputs, inputs_past
    from in_tfm.run_config import KnownCutoffs

    cutoffs = KnownCutoffs(positive=4.25, negative=-4.0)
    outputs = torch.tensor([[[0.0], [5.0], [-4.0]], [[4.25], [1.0], [-9.0]]])
    inputs = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    valid = torch.tensor([[True, True, True], [True, True, False]])
    kept = KeptInputs.joined(
        [
            inputs_past(cutoffs, inputs[:1], outputs[:1], valid[:1], first_sample=0),
            inputs_past(cutoffs, inputs[1:], outputs[1:], valid[1:], first_sample=1),
        ],
        seq_len=3,
    )
    assert kept.keys.tolist() == [1, 2, 3]
    gathered = kept.gather(np.array([1, 0]), np.array([0, 2]))
    assert torch.equal(gathered.inputs, torch.stack([inputs[1, 0], inputs[0, 2]]))
    with pytest.raises(ValueError, match="do not belong to this corpus"):
        kept.gather(np.array([0]), np.array([0]))
