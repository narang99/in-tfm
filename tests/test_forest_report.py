import numpy as np

from in_tfm.density_peaks import density_peak_forest, forest_trees
from in_tfm.forest_report import FOREST_PAGE, HitArrays, RootRule, within_depth, write_forest_page


def test_open_ended_depth_bin_takes_everything_deeper():
    depth = np.array([0, 3, 21, 400])
    assert within_depth(depth, 21, None).tolist() == [False, False, True, True]
    assert within_depth(depth, 1, 3).tolist() == [False, True, False, False]


def test_page_has_a_row_per_tree_and_its_images(tmp_path):
    rng = np.random.default_rng(0)
    inputs = np.zeros((20, 9))
    inputs[:12, :3] = rng.uniform(1.0, 1.2, (12, 3))
    inputs[12:, 6:] = rng.uniform(1.0, 1.2, (8, 3))
    forest = density_peak_forest(inputs, k=3)
    trees = forest_trees(forest, root_delta=0.5, root_density_quantile=0.5)
    hits = HitArrays(hadamards=inputs, inputs=inputs, activations=rng.uniform(0, 1, 20))
    rule = RootRule(k=3, root_delta=0.5, root_density_quantile=0.5, distance="weighted_jaccard")

    path = write_forest_page(tmp_path, forest, trees, hits, (3, 3), rule, "Neuron 0")

    html = path.read_text()
    assert path.name == FOREST_PAGE
    assert html.count('class="tree"') == 2
    assert 'href="index.html#cluster-1"' in html
    for name in ["decision_graph.png", "forest.png", "tree0_hadamard_0.png", "tree1_input_0.png"]:
        assert (tmp_path / "density_peaks" / name).exists()
