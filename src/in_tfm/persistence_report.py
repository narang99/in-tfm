"""A page of a neuron's ToMATo merges, with a slider that steps from one merge to the next.

- It sits next to `index.html` as `persistence.html`, and the index links to it.
- It needs only the saved forest and a label per peak, so no model has to run.
- It holds two hierarchies, since they answer different questions.
  - Between the report's trees: are two trees separate hills?
  - Between every local mode: the raw hierarchy, mostly small bumps.
"""

import base64
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel, ConfigDict

from .colormaps import rd_bk_gn
from .density_peaks.persistence import MergeRoute, PeakMerges, merge_routes, peak_merges
from .html_report import page, render_template
from .presenters import HadamardShape, HitLabel
from .report_page import PERSISTENCE_PAGE, ForestData

LabelOfPoint = Callable[[int], HitLabel]
REPORT_PAGE = "index.html"
SAME_THRESHOLD_DECIMALS = 6
"""Merges whose persistence agrees to this many decimals are one slider step."""
ROUTE_STOPS = 20
EXAMPLES_PER_TREE = 8
EXAMPLES_PER_MODE = 4
"""Fewer than a tree's, since there are thousands of modes and every example holds its tile."""
TILE_LEVELS = 127
COPY_DECIMALS = 6
"""Rows that agree to this many decimals are one example, as in `merge_copies`."""
COLOR_TABLE_SIZE = 256


class HitDisplay(BaseModel):
    """What a page needs to draw any point of the forest."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    label_of: LabelOfPoint
    hadamards: Float[np.ndarray, "n_distinct hidden"]
    """The vectors that were clustered, so a tile shows what the distance saw."""
    shape: HadamardShape
    report_clusters: set[int]
    """Clusters with a section in the report, since a very small tree can have none."""


class TreeLink(BaseModel):
    name: str
    href: str | None
    """Its section of the cluster report. A cluster k holds the confident hits of tree k."""


class ExampleView(BaseModel):
    rho: float
    before: str
    token: str
    tile: str
    """Base64 of int8 values, each a share of `tile_max` in `TILE_LEVELS` levels."""
    tile_max: float


class PeakView(BaseModel):
    name: str
    rho: float
    hits: int
    examples: list[ExampleView]
    """The densest distinct hits among its members, the peak first."""
    density_is_capped: bool
    """See `points_at_capped_density`. Its rho, and so its persistence, is not a measurement."""
    trees: list[TreeLink] = []
    """Only between trees. Copies can put several trees on one peak."""


class RouteStop(BaseModel):
    tree: TreeLink
    example: ExampleView


class RouteView(BaseModel):
    stops: list[RouteStop]
    merge_stop: int
    """Position of the merge point in `stops`."""
    n_points: int
    """Length of the whole route, before it was thinned to `stops`."""
    step: int


class MergeView(BaseModel):
    dying: int
    surviving: int
    saddle_rho: float
    persistence: float
    route: RouteView | None = None
    """Only between trees, since the view between all modes has thousands of merges."""


class HierarchyView(BaseModel):
    peaks: list[PeakView]
    merges: list[MergeView]
    step_ends: list[int]
    """How many merges have happened at each slider step after the first, which has none."""
    hits_with_no_peak: int
    members_are: str
    """How a peak's hits were chosen, in words that follow "hits" on the page."""


class PersistenceData(BaseModel):
    trees: HierarchyView
    modes: HierarchyView
    tile_colors: list[tuple[int, int, int]]
    """`rd_bk_gn` as a lookup table, so the page's tiles are coloured as the report's are."""
    tile_levels: int
    tile_height: int
    tile_width: int


def points_at_capped_density(data: ForestData) -> Bool[np.ndarray, "n_distinct"]:
    """More copies than k means a kth distance of 0, and `log_density` gives those one fixed top value.

    - More than k points tying at the very top is what that looks like, since real densities rarely tie.
    """
    at_top = data.rho == data.rho.max()
    return at_top if at_top.sum() > data.neighbours.shape[1] else np.zeros_like(at_top)


def peak_of_tree_label(
    merges: PeakMerges, data: ForestData
) -> Int[np.ndarray, "n_distinct"]:
    """A tree's own hits, not its basin.

    - Between trees, a basin also holds any denser mode that is no tree's root, once it absorbs one.
    - So a basin's densest hits can be a different hill from the tree they would stand for."""
    return merges.peak_of_point[data.roots][data.tree]


def hits_per_peak(
    members: Int[np.ndarray, "n_distinct"], n_peaks: int, data: ForestData
) -> np.ndarray:
    assigned = members >= 0
    return np.bincount(
        members[assigned], weights=data.counts[assigned], minlength=n_peaks
    ).astype(int)


def step_ends(persistence: np.ndarray) -> list[int]:
    rounded = persistence.round(SAME_THRESHOLD_DECIMALS)
    return (np.flatnonzero(np.diff(rounded) > 0) + 1).tolist() + (
        [len(rounded)] if len(rounded) else []
    )


def tree_link(tree: int, display: HitDisplay) -> TreeLink:
    in_report = tree in display.report_clusters
    return TreeLink(
        name=f"T{tree}",
        href=f"{REPORT_PAGE}#cluster-{tree}" if in_report else None,
    )


def trees_of_peaks(
    merges: PeakMerges, data: ForestData, display: HitDisplay
) -> list[list[TreeLink]]:
    """Copies can put two roots on one plateau, so one peak can hold several trees."""
    links: list[list[TreeLink]] = [[] for _ in merges.peaks]
    for tree, root in enumerate(data.roots.tolist()):
        links[merges.peak_of_point[root]].append(tree_link(tree, display))
    return links


def encoded_tile(row: Float[np.ndarray, "hidden"]) -> tuple[str, float]:
    """Each tile keeps its own maximum, so a page can put any tiles on one scale without losing precision."""
    largest = float(np.abs(row).max())
    scaled = row / largest * TILE_LEVELS if largest > 0 else np.zeros_like(row)
    levels = np.round(scaled).astype(np.int8)
    return base64.b64encode(levels.tobytes()).decode("ascii"), largest


def example_points(
    members: Int[np.ndarray, "n_distinct"],
    n_peaks: int,
    rho: Float[np.ndarray, "n_distinct"],
    hadamards: Float[np.ndarray, "n_distinct hidden"],
    per_peak: int,
) -> list[list[int]]:
    """Densest first, skipping identical rows, since a copy group would otherwise fill a whole list."""
    assigned = np.flatnonzero(members >= 0)
    order = assigned[np.lexsort((-rho[assigned], members[assigned]))]
    chosen: list[list[int]] = [[] for _ in range(n_peaks)]
    seen: list[set[bytes]] = [set() for _ in range(n_peaks)]
    for point in order.tolist():
        peak = int(members[point])
        if len(chosen[peak]) == per_peak:
            continue
        row = hadamards[point].round(COPY_DECIMALS).tobytes()
        if row not in seen[peak]:
            seen[peak].add(row)
            chosen[peak].append(point)
    return chosen


def example_view(point: int, data: ForestData, display: HitDisplay) -> ExampleView:
    label = display.label_of(point)
    tile, tile_max = encoded_tile(display.hadamards[point])
    return ExampleView(
        rho=round(float(data.rho[point]), 3),
        before=label.before,
        token=label.token,
        tile=tile,
        tile_max=tile_max,
    )


def hierarchy_view(
    merges: PeakMerges,
    names: list[str],
    members: Int[np.ndarray, "n_distinct"],
    members_are: str,
    data: ForestData,
    display: HitDisplay,
    examples_per_peak: int,
) -> HierarchyView:
    n_peaks = len(merges.peaks)
    hits = hits_per_peak(members, n_peaks, data)
    examples = example_points(
        members, n_peaks, data.rho, display.hadamards, examples_per_peak
    )
    capped = points_at_capped_density(data)
    return HierarchyView(
        peaks=[
            PeakView(
                name=name,
                rho=round(float(data.rho[point]), 3),
                hits=int(count),
                examples=[example_view(p, data, display) for p in listed],
                density_is_capped=bool(capped[point]),
            )
            for name, point, count, listed in zip(
                names, merges.peaks.tolist(), hits, examples
            )
        ],
        merges=[
            MergeView(
                dying=dying,
                surviving=surviving,
                saddle_rho=round(float(data.rho[saddle]), 3),
                persistence=round(float(persistence), SAME_THRESHOLD_DECIMALS),
            )
            for dying, surviving, saddle, persistence in zip(
                merges.dying.tolist(),
                merges.surviving.tolist(),
                merges.saddle.tolist(),
                merges.persistence.tolist(),
            )
        ],
        step_ends=step_ends(merges.persistence),
        hits_with_no_peak=int(data.counts[members < 0].sum()),
        members_are=members_are,
    )


def hops_towards(start: int, end: int, step: int) -> list[int]:
    """Every step-th position after `start`, always ending on `end`."""
    distance = abs(end - start)
    direction = 1 if end > start else -1
    hops = list(range(step, distance, step)) + ([distance] if distance else [])
    return [start + direction * hop for hop in hops]


def thinned_route(n_points: int, merge_at: int, step: int) -> list[int]:
    towards_dying = hops_towards(merge_at, 0, step)
    towards_surviving = hops_towards(merge_at, n_points - 1, step)
    return towards_dying[::-1] + [merge_at] + towards_surviving


def smallest_step(n_points: int, merge_at: int, stops: int) -> int:
    """Stepping out from the merge point keeps it, and the route's two ends are always kept too."""
    step = 1
    while len(thinned_route(n_points, merge_at, step)) > stops:
        step += 1
    return step


def route_view(route: MergeRoute, data: ForestData, display: HitDisplay) -> RouteView:
    n_points = len(route.points)
    step = smallest_step(n_points, route.merge_at, ROUTE_STOPS)
    kept = thinned_route(n_points, route.merge_at, step)
    return RouteView(
        stops=[
            RouteStop(
                tree=tree_link(int(data.tree[route.points[position]]), display),
                example=example_view(route.points[position], data, display),
            )
            for position in kept
        ],
        merge_stop=kept.index(route.merge_at),
        n_points=n_points,
        step=step,
    )


def tile_color_table() -> list[tuple[int, int, int]]:
    colors = rd_bk_gn(np.linspace(0, 1, COLOR_TABLE_SIZE))[:, :3] * 255
    return [(int(r), int(g), int(b)) for r, g, b in colors]


def persistence_data(data: ForestData, display: HitDisplay) -> PersistenceData:
    between_trees = peak_merges(data.rho, data.neighbours, seeds=data.roots)
    between_modes = peak_merges(data.rho, data.neighbours)
    links = trees_of_peaks(between_trees, data, display)
    trees = hierarchy_view(
        between_trees,
        ["+".join(link.name for link in listed) for listed in links],
        peak_of_tree_label(between_trees, data),
        "labelled with this tree",
        data,
        display,
        EXAMPLES_PER_TREE,
    )
    for peak, listed in zip(trees.peaks, links):
        peak.trees = listed
    routes = merge_routes(data.rho, data.neighbours, between_trees)
    for merge, route in zip(trees.merges, routes):
        merge.route = route_view(route, data, display)
    return PersistenceData(
        trees=trees,
        modes=hierarchy_view(
            between_modes,
            [f"P{i}" for i in range(len(between_modes.peaks))],
            between_modes.peak_of_point,
            "in its basin",
            data,
            display,
            EXAMPLES_PER_MODE,
        ),
        tile_colors=tile_color_table(),
        tile_levels=TILE_LEVELS,
        tile_height=display.shape[0],
        tile_width=display.shape[1],
    )


def data_for_script_tag(data: PersistenceData) -> str:
    """`<` is escaped so that hit text can never close the script tag."""
    return json.dumps(data.model_dump(), ensure_ascii=False).replace("<", "\\u003c")


def write_persistence_page(
    report_dir: Path, neuron_idx: int, data: ForestData, display: HitDisplay
) -> Path:
    title = f"Neuron {neuron_idx}: peak merges"
    body = render_template(
        "persistence.html",
        title=title,
        data_json=data_for_script_tag(persistence_data(data, display)),
    )
    path = report_dir / PERSISTENCE_PAGE
    path.write_text(page(title, body))
    return path
