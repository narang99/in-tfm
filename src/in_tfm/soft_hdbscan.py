"""HDBSCAN's soft clustering, also for a precomputed distance matrix.

- The `hdbscan` package gives each point a probability per cluster with `all_points_membership_vectors`.
  - It multiplies a distance part and a tree part, then scales by the chance the point is in any cluster.
  - The distance part is 1 / the distance to each cluster's nearest exemplar, normalised.
  - The tree part comes from the condensed tree alone: how high up the tree the point merges with each cluster.
- The package only builds this from raw vectors, so a weighted Jaccard matrix gets no soft clustering from it.
  - Only the distance part needs vectors. With a full distance matrix it is a lookup.
  - So this file rebuilds the exemplars from the condensed tree, as `hdbscan.prediction.PredictionData` does.
  - The tree part reuses the package's own compiled helpers.
- `tests/test_soft_hdbscan.py` checks the result against the package on euclidean data, where both can run.
"""

import numpy as np
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import pairwise_distances

from .cluster_membership import ClusterProbabilities, HitMembership, Noise
from .clustering import ClusterSelectionMethod
from .density_peaks import comparable_rows
from .run_config import ClusteringConfig, HdbscanConfig


class SoftClustering(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    labels: Int[np.ndarray, "n_points"]
    """HDBSCAN's hard labels, -1 for noise."""
    probabilities: Float[np.ndarray, "n_points n_clusters"]
    """Each row sums to the point's chance of being in any cluster, so less than 1. Column i is label i."""


def clusters_below(cluster_tree: np.ndarray, cluster: int) -> list[int]:
    found, frontier = [], [cluster]
    while frontier:
        found.extend(frontier)
        frontier = cluster_tree["child"][np.isin(cluster_tree["parent"], frontier)].tolist()
    return found


def leaves_below(cluster_tree: np.ndarray, cluster: int) -> list[int]:
    return [c for c in clusters_below(cluster_tree, cluster) if not np.isin(c, cluster_tree["parent"])]


def max_lambda_per_parent(tree: np.ndarray) -> dict[int, float]:
    result: dict[int, float] = {}
    for parent, lambda_val in zip(tree["parent"].tolist(), tree["lambda_val"].tolist()):
        result[parent] = max(result.get(parent, -np.inf), lambda_val)
    return result


def exemplars(tree: np.ndarray, cluster_tree: np.ndarray, cluster: int, leaf_max_lambdas: dict[int, float]) -> Int[np.ndarray, "n"]:
    """The points that stay in each of the cluster's leaves longest, as `PredictionData` picks them."""
    points = [
        tree["child"][(tree["parent"] == leaf) & (tree["lambda_val"] == leaf_max_lambdas[leaf])]
        for leaf in leaves_below(cluster_tree, cluster)
    ]
    return np.concatenate(points) if points else np.zeros(0, dtype=int)


def distance_part(
    distances: Float[np.ndarray, "n n"], exemplars_per_cluster: list[Int[np.ndarray, "m"]]
) -> Float[np.ndarray, "n n_clusters"]:
    """`dist_membership_vector` without a softmax: 1 / the nearest exemplar's distance, normalised per point."""
    nearest = np.stack([distances[:, ex].min(axis=1) for ex in exemplars_per_cluster], axis=1)
    inverse = np.where(nearest > 0, 1.0 / np.where(nearest > 0, nearest, 1.0), np.finfo(np.float64).max / nearest.shape[1])
    return inverse / inverse.sum(axis=1, keepdims=True)


def soft_clustering_precomputed(
    distances: Float[np.ndarray, "n n"],
    min_cluster_size: int,
    cluster_selection_method: ClusterSelectionMethod,
    min_samples: int | None,
) -> SoftClustering:
    from hdbscan import HDBSCAN
    from hdbscan._prediction_utils import all_points_outlier_membership_vector, all_points_prob_in_some_cluster

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples,
        cluster_selection_method=cluster_selection_method, metric="precomputed",
    ).fit(distances.astype(np.float64))
    labels = np.asarray(clusterer.labels_)
    selected = np.array(sorted(clusterer.condensed_tree_._select_clusters()), dtype=np.intp)
    if selected.size == 0:
        return SoftClustering(labels=labels, probabilities=np.zeros((len(labels), 0)))
    tree = clusterer.condensed_tree_._raw_tree
    cluster_tree = tree[tree["child_size"] > 1]
    leaf_max_lambdas = max_lambda_per_parent(tree)
    by_distance = distance_part(distances, [exemplars(tree, cluster_tree, c, leaf_max_lambdas) for c in selected])
    by_tree = all_points_outlier_membership_vector(selected, tree, leaf_max_lambdas, cluster_tree)
    in_any = all_points_prob_in_some_cluster(selected, tree, leaf_max_lambdas, cluster_tree)
    combined = by_distance * by_tree
    combined /= combined.sum(axis=1, keepdims=True)
    return SoftClustering(labels=labels, probabilities=combined * in_any[:, None])


def hit_distances(
    points: Float[np.ndarray, "n dim"], clustering: ClusteringConfig, device: str | None
) -> Float[np.ndarray, "n n"]:
    """Every pair, compared as the density-peak forest compares them. On CUDA the Triton kernel builds the matrix."""
    if device is not None:
        from .density_peaks import torch_forest

        return torch_forest.pairwise_distances(points, clustering.distance, clustering.normalise, device)
    rows = comparable_rows(points.astype(np.float64), clustering.distance, clustering.normalise)
    if clustering.distance == "weighted_jaccard":
        from .weighted_jaccard import weighted_jaccard_block

        return np.clip(weighted_jaccard_block(rows, rows), 0.0, None)
    return pairwise_distances(rows)


def soft_memberships(
    points: Float[np.ndarray, "n dim"], clustering: ClusteringConfig, strategy: HdbscanConfig, device: str | None,
) -> list[HitMembership]:
    """HDBSCAN's noise stays noise. A member's row is renormalised over the clusters, so it sums to 1."""
    print(f"[cluster] {len(points)} points x {points.shape[1]} dims, soft hdbscan, {clustering.distance}")
    soft = soft_clustering_precomputed(
        hit_distances(points, clustering, device), strategy.min_cluster_size, strategy.selection_method,
        strategy.min_samples,
    )
    return [
        Noise() if label < 0 else ClusterProbabilities(probabilities=row / row.sum())
        for label, row in zip(soft.labels, soft.probabilities)
    ]
