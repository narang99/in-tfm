import numpy as np
import pytest
from jaxtyping import Float, Int

from in_tfm.density_peaks.persistence import clusters_at, merge_routes, peak_merges

PathEdges = Int[np.ndarray, "n k"]


def neighbours_on_a_line(n: int) -> PathEdges:
    """Each point lists the points beside it, with an end repeating its only neighbour."""
    return np.array(
        [
            [max(i - 1, 0) if i else 1, min(i + 1, n - 1) if i < n - 1 else n - 2]
            for i in range(n)
        ]
    )


def test_the_lower_of_two_hills_dies_at_the_valley():
    rho = np.array([1.0, 3.0, 1.5, 0.5, 2.0, 1.0])
    merges = peak_merges(rho, neighbours_on_a_line(6))
    assert merges.peaks.tolist() == [1, 4]
    assert merges.peaks[merges.dying].tolist() == [4]
    assert merges.peaks[merges.surviving].tolist() == [1]
    assert merges.saddle.tolist() == [3]
    assert merges.persistence.tolist() == pytest.approx([1.5])
    assert merges.peak_of_point.tolist() == [0, 0, 0, 1, 1, 1]


def test_a_route_runs_from_the_dying_peak_through_the_valley_to_the_surviving_peak():
    rho = np.array([1.0, 3.0, 1.5, 0.5, 2.0, 1.0])
    neighbours = neighbours_on_a_line(6)
    route = merge_routes(rho, neighbours, peak_merges(rho, neighbours))[0]
    assert route.points == [4, 3, 2, 1]
    assert route.merge_at == 1


def fewest_hops(
    edges: set[tuple[int, int]], allowed: set[int], start: int, end: int
) -> int:
    reached, frontier, hops = {start}, {start}, 0
    while end not in reached:
        frontier = {
            b for a, b in edges if a in frontier and b in allowed and b not in reached
        }
        assert frontier
        reached |= frontier
        hops += 1
    return hops


@pytest.mark.parametrize("seed", range(20))
def test_routes_follow_the_graph_above_the_saddle(seed: int):
    random = np.random.default_rng(seed)
    n = 40
    rho = random.random(n)
    neighbours = random.integers(0, n, size=(n, 3))
    seeds = np.flatnonzero(rho > 0.7)
    merges = peak_merges(rho, neighbours, seeds=seeds)
    edges = {
        pair
        for a, listed in enumerate(neighbours.tolist())
        for b in listed
        for pair in ((a, b), (b, a))
    }
    routes = merge_routes(rho, neighbours, merges)
    assert len(routes) == len(merges.dying)
    for i, route in enumerate(routes):
        saddle = merges.saddle[i]
        assert route.points[0] == merges.peaks[merges.dying[i]]
        assert route.points[-1] == merges.peaks[merges.surviving[i]]
        assert route.points[route.merge_at] == saddle
        assert len(set(route.points)) == len(route.points)
        assert all(step in edges for step in zip(route.points, route.points[1:]))
        assert rho[route.points].min() == rho[saddle]
        allowed = set(np.flatnonzero(rho >= rho[saddle]).tolist())
        assert route.merge_at == fewest_hops(
            edges, allowed, int(saddle), route.points[0]
        )
        assert len(route.points) - 1 - route.merge_at == fewest_hops(
            edges, allowed, int(saddle), route.points[-1]
        )


def test_merges_are_sorted_with_the_smallest_bump_first():
    rho = np.array([5.0, 1.0, 3.0, 2.5, 4.0])
    merges = peak_merges(rho, neighbours_on_a_line(5))
    assert merges.peaks[merges.dying].tolist() == [2, 4]
    assert merges.persistence.tolist() == pytest.approx([0.5, 3.0])


def test_separate_graph_pieces_never_merge():
    rho = np.array([2.0, 1.0, 3.0, 1.0])
    neighbours = np.array([[1], [0], [3], [2]])
    merges = peak_merges(rho, neighbours)
    assert len(merges.peaks) == 2
    assert len(merges.persistence) == 0


def test_a_tied_plateau_is_one_peak():
    rho = np.array([1.0, 2.0, 2.0, 2.0, 1.0])
    merges = peak_merges(rho, neighbours_on_a_line(5))
    assert merges.peaks.tolist() == [1]
    assert len(merges.persistence) == 0
    assert merges.peak_of_point.tolist() == [0] * 5


def test_seeds_silence_merges_with_other_peaks():
    rho = np.array([5.0, 1.0, 3.0, 2.5, 4.0, 0.5, 2.0])
    merges = peak_merges(rho, neighbours_on_a_line(7), seeds=np.array([0, 6]))
    assert merges.peaks.tolist() == [0, 6]
    assert merges.persistence.tolist() == pytest.approx([1.5])
    assert merges.saddle.tolist() == [5]


def test_a_seed_on_a_slope_merges_where_it_enters():
    rho = np.array([3.0, 2.0, 1.0])
    merges = peak_merges(rho, neighbours_on_a_line(3), seeds=np.array([0, 1]))
    assert merges.persistence.tolist() == [0.0]
    assert merges.peaks[merges.dying].tolist() == [1]


def test_thresholds_give_tomato_clusters():
    rho = np.array([5.0, 1.0, 3.0, 2.5, 4.0])
    merges = peak_merges(rho, neighbours_on_a_line(5))
    peak_ids = merges.peaks.tolist()
    assert clusters_at(merges, 0.0).tolist() == [
        peak_ids.index(p) for p in (0, 0, 2, 4, 4)
    ]
    assert clusters_at(merges, 0.5).tolist() == [
        peak_ids.index(p) for p in (0, 0, 4, 4, 4)
    ]
    assert clusters_at(merges, 3.0).tolist() == [peak_ids.index(0)] * 5


def superlevel_components(
    rho: Float[np.ndarray, "n"], neighbours: PathEdges, level: float
) -> list[set[int]]:
    alive = {i for i in range(len(rho)) if rho[i] >= level}
    pending, components = set(alive), []
    while pending:
        stack, component = [pending.pop()], set()
        while stack:
            here = stack.pop()
            component.add(here)
            linked = {int(j) for j in neighbours[here]} | {
                j for j in range(len(rho)) if here in neighbours[j].tolist()
            }
            for next_point in linked & pending:
                pending.remove(next_point)
                stack.append(next_point)
        components.append(component)
    return components


def brute_force_persistence(
    rho: Float[np.ndarray, "n"], neighbours: PathEdges
) -> list[float]:
    """From the definition: walk the levels down, and at each one kill all but the highest peak of a component."""
    peaks_alive: set[int] = set()
    deaths = []
    for level in sorted(set(rho.tolist()), reverse=True):
        for component in superlevel_components(rho, neighbours, level):
            present = component & peaks_alive
            if not present:
                peaks_alive.add(min(component))
            for peak in sorted(present, key=lambda p: rho[p])[:-1]:
                deaths.append(float(rho[peak] - level))
                peaks_alive.remove(peak)
    return sorted(deaths)


@pytest.mark.parametrize("seed", range(20))
def test_persistence_matches_its_definition_on_random_graphs(seed):
    rng = np.random.default_rng(seed)
    n = 30
    rho = rng.permutation(n).astype(float)
    neighbours = np.stack(
        [rng.choice(np.delete(np.arange(n), i), 3, replace=False) for i in range(n)]
    )
    assert peak_merges(rho, neighbours).persistence.tolist() == pytest.approx(
        brute_force_persistence(rho, neighbours)
    )
