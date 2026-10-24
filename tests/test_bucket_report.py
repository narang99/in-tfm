import numpy as np

from in_tfm.bucket_graph import SampleMap
from in_tfm.bucket_map_image import HUGE_CANVAS, NORMAL_CANVAS, draw_base, draw_bucket
from in_tfm.bucket_report import MIN_SHOWN_PROBABILITY, SHOWN_CLUSTERS, card_marks, map_fractions, shown_clusters


def test_the_shown_clusters_are_the_most_probable_above_the_minimum():
    probabilities = np.array([0.5, 0.3, 0.1, 0.04, 0.03, 0.03])
    assert shown_clusters(probabilities) == [0, 1, 2]
    assert len(shown_clusters(np.full(10, 0.1))) == SHOWN_CLUSTERS


def test_the_most_probable_cluster_is_always_shown():
    probabilities = np.full(40, 1 / 40)
    assert probabilities.max() < MIN_SHOWN_PROBABILITY
    assert len(shown_clusters(probabilities)) == 1


def test_the_huge_map_is_the_same_picture_at_four_times_the_size():
    points = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.2]])
    labels, peaks = np.array([0, 1, 0]), {0: 0, 1: 1}
    normal, huge = draw_base(points, labels, peaks, NORMAL_CANVAS), draw_base(points, labels, peaks, HUGE_CANVAS)
    assert (normal.width, huge.width) == (NORMAL_CANVAS, HUGE_CANVAS)
    assert draw_bucket(huge, points, [2], [2]).size == huge.size


def test_a_mark_sits_where_its_node_is_drawn_in_percent_of_the_map():
    points = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    at = map_fractions(points)
    assert np.allclose(at[2], [50, 50])
    assert 0 < at[0, 0] < at[1, 0] < 100


def test_numbered_stars_link_to_their_cards_in_order():
    sample = SampleMap(
        hits=np.arange(3), points=np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]]), labels=np.zeros(3, dtype=int),
        peak_nodes={0: 0}, neighbours=6,
    )
    assert [m.href for m in card_marks(sample, [2, 1])] == ["#card-1", "#card-2"]
