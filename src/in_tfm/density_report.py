"""One page of maps, one per density level, all on the bucket report's layout so every hit stays in place.

- Kept hits are coloured by their group at that level, and the hits not yet kept are faint.
- Each group is a numbered sphere at its densest hit on the map, linking to its row in the level's table.
- A row links each group to the clusters most of its hits belong to.
"""

from pathlib import Path

import numpy as np
from jaxtyping import Int
from pydantic import BaseModel

from .bucket_graph import SampleMap
from .bucket_map_image import NORMAL_CANVAS, cluster_colour, draw_groups
from .bucket_report import PEAK_MARK_SIZE, MapMark, map_fractions
from .density_levels import NOT_KEPT, DensityLevel, DensityLevels
from .html_report import page, render_template
from .run_config import DensityLevelsConfig

INDEX_PAGE = "index.html"
NOT_KEPT_COLOUR = (232, 232, 232)
SMALL_GROUP_COLOUR = (150, 150, 150)


class LevelView(BaseModel):
    anchor: str
    level: DensityLevel
    map_src: str
    marks: list[MapMark]
    n_on_map: int
    """Kept hits that are on the map."""


def group_anchor(level_anchor: str, number: int) -> str:
    return f"{level_anchor}-group-{number}"


def dot_colours(groups_of_nodes: Int[np.ndarray, "m"]) -> list[tuple[int, int, int]]:
    def colour(group: int) -> tuple[int, int, int]:
        if group == NOT_KEPT:
            return NOT_KEPT_COLOUR
        return SMALL_GROUP_COLOUR if group == 0 else cluster_colour(group)
    return [colour(int(g)) for g in groups_of_nodes]


def kept_on_top(groups_of_nodes: Int[np.ndarray, "m"]) -> Int[np.ndarray, "m"]:
    """Faint nodes first, then small groups, then the numbered groups."""
    rank = np.where(groups_of_nodes == NOT_KEPT, 0, np.where(groups_of_nodes == 0, 1, 2))
    return np.argsort(rank, kind="stable")


def sphere_nodes(groups_of_nodes: Int[np.ndarray, "m"], birth_of_nodes: Int[np.ndarray, "m"]) -> dict[int, int]:
    """Each numbered group's densest node on the map. A group with no hit on the map gets no sphere."""
    spheres = {}
    for number in sorted(set(groups_of_nodes[groups_of_nodes > 0].tolist())):
        members = np.flatnonzero(groups_of_nodes == number)
        spheres[number] = int(members[np.argmin(birth_of_nodes[members])])
    return spheres


def level_view(index: int, levels: DensityLevels, sample: SampleMap, report_dir: Path) -> LevelView:
    level = levels.levels[index]
    anchor = f"level-{round(level.share * 100)}"
    groups_of_nodes = levels.hit_groups[index][sample.hits]
    spheres = sphere_nodes(groups_of_nodes, levels.hit_birth[sample.hits])
    image = draw_groups(sample.points, dot_colours(groups_of_nodes), kept_on_top(groups_of_nodes), spheres, NORMAL_CANVAS)
    map_src = f"maps/{anchor}.png"
    image.save(report_dir / map_src)
    at = map_fractions(sample.points)
    marks = [
        MapMark(href=f"#{group_anchor(anchor, n)}", title=f"group {n}", left=at[node, 0], top=at[node, 1], size=PEAK_MARK_SIZE)
        for n, node in spheres.items()
    ]
    return LevelView(anchor=anchor, level=level, map_src=map_src, marks=marks, n_on_map=int((groups_of_nodes != NOT_KEPT).sum()))


def write_density_report(
    levels: DensityLevels, sample: SampleMap, settings: DensityLevelsConfig, neuron_idx: int, cluster_report: str,
    report_dir: Path,
) -> Path:
    """`cluster_report` is the cluster report's index, relative to `report_dir`."""
    (report_dir / "maps").mkdir(parents=True, exist_ok=True)
    views = [level_view(i, levels, sample, report_dir) for i in range(len(levels.levels))]
    title = f"Neuron {neuron_idx}: groups of hits at falling density"
    body = render_template(
        "density_levels.html", title=title, views=views, settings=settings, cluster_report=cluster_report,
        n_sampled=len(sample.hits), n_hits=levels.hit_groups.shape[1], group_anchor=group_anchor,
        colour_of=lambda n: "rgb(%d,%d,%d)" % cluster_colour(n),
    )
    path = report_dir / INDEX_PAGE
    path.write_text(page(title, body))
    return path
