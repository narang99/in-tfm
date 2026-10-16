import torch

from in_tfm.neuron_tracking import StepRecord, TrackingLog
from in_tfm.tracking_reports import ClusterMatch, StateReport, match_clusters, save_distance_plot, states_to_report, write_index


def record(step: int) -> StepRecord:
    return StepRecord(step=step, epoch=0, lr=1e-3, cosine_distance=1e-5 * step, euclidean_distance=0.0)


def test_each_chosen_step_reports_the_state_before_and_after_it_plus_initial_and_final():
    states = states_to_report([record(3), record(9)], n_states=11)
    assert [s.state for s in states] == [0, 2, 3, 8, 9, 10]
    assert states[0].labels == ["initial"]
    assert states[-1].labels == ["final"]
    assert states[2].labels == ["after step 3"]


def test_adjacent_steps_share_a_state_and_both_labels():
    states = {s.state: s.labels for s in states_to_report([record(3), record(4)], n_states=11)}
    assert states[3] == ["after step 3", "before step 4"]


def test_a_cluster_is_matched_to_the_most_similar_cluster_on_the_other_side():
    a = {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 1.0])}
    b = {7: torch.tensor([0.1, 1.0]), 8: torch.tensor([1.0, 0.1])}
    assert {m.cluster_a: m.cluster_b for m in match_clusters(a, b)} == {0: 8, 1: 7}


def test_matching_against_no_clusters_gives_no_matches():
    assert match_clusters({0: torch.ones(2)}, {}) == []


def test_the_distance_plot_is_written(tmp_path):
    log = TrackingLog(layer_name="x", neuron=0, steps=[record(s) for s in range(1, 20)])
    save_distance_plot(log, [record(5)], tmp_path / "plot.png")
    assert (tmp_path / "plot.png").stat().st_size > 0


def test_the_index_lists_steps_states_and_matches_and_escapes_what_it_is_given(tmp_path):
    log = TrackingLog(layer_name="block_c.<x>", neuron=0, steps=[record(s) for s in range(1, 5)])
    states = [StateReport(state=0, labels=["initial"]), StateReport(state=3, labels=["after step 3"])]
    pairs = {"initial to final": [ClusterMatch(cluster_a=1, cluster_b=2, similarity=0.5)], "empty": []}
    html = write_index(tmp_path, log, [record(3)], states, pairs).read_text()
    assert 'href="state_0003/neuron_0/index.html"' in html
    assert "block_c.&lt;x&gt;" in html
    assert "0.500" in html
    assert "no clusters on one side" in html
