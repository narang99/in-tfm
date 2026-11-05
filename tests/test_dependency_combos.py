import torch
from conftest import firing, writer

from in_tfm.dependencies import FixedWriter, Tap
from in_tfm.dependencies.firings import ClusterMatch, WriterFiring, WriterReportRef
from in_tfm.dependencies.mean_matching import MeanMatcher
from in_tfm.dependencies.stats import combo_shares


def matched(w: WriterFiring, cluster_id: int) -> WriterFiring:
    w.match = ClusterMatch(
        cluster_id=cluster_id, distance=0.0, hadamard=[], closest_hadamard=[]
    )
    return w


def embedding(token_id: int) -> WriterFiring:
    return matched(
        WriterFiring(
            writer=FixedWriter(
                name="embedding", tap=Tap(layer="emb"), dim=0, token=0, by_token=True
            ),
            value=1.0,
            term=1.0,
        ),
        token_id,
    )


def test_a_combination_is_one_cluster_per_writer_counted_over_the_dims_firings():
    firings = [
        firing(0, 0, 1.0, [matched(writer(0, 1.0, 1.0), 3), embedding(7)]),
        firing(1, 0, 1.0, [embedding(7), matched(writer(0, 5.0, 5.0), 3)]),
        firing(2, 0, 1.0, [embedding(9), matched(writer(0, 1.0, 1.0), 3)]),
    ]
    first, second = combo_shares(firings)
    assert [p.cluster_id for p in first.parts] == [7, 3]
    assert (first.count, round(first.ratio, 2)) == (2, 0.67)
    assert [p.cluster_id for p in second.parts] == [9, 3]


def write_mean(report_dir, cluster_id: int, mean: list[float]) -> None:
    cluster_dir = report_dir / f"cluster_{cluster_id}"
    cluster_dir.mkdir(parents=True)
    torch.save(torch.tensor(mean), cluster_dir / "mean.pt")


def test_a_firing_matches_the_mean_nearest_by_cosine_whatever_its_scale(tmp_path):
    write_mean(tmp_path, 4, [1.0, 0.0])
    write_mean(tmp_path, 11, [0.0, 1.0])
    report = WriterReportRef(layer="o", neuron=0, report_dir=tmp_path, tile_shape=(2,))
    matcher = MeanMatcher(report, torch.tensor([1.0, 2.0]))
    near_4, near_11 = matcher.match(torch.tensor([[10.0, 0.1], [0.1, 3.0]]))
    assert (near_4.cluster_id, near_11.cluster_id) == (4, 11)
    assert near_4.distance < 0.01
