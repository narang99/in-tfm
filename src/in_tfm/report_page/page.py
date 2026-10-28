"""The parts of a neuron's index page above its clusters: stat tiles, a glossary, and numbered sections."""

import json
import math
from pathlib import Path

from pydantic import BaseModel

from ..activation_ranges import ActivationRanges, load_activation_ranges
from ..html_report import render_template
from ..report_meta import ReportMeta
from .borders_section import BORDER_INDEX, BORDERS_DIR, borders_section
from .data import ForestData, SelectionCurve, load_forest_data, load_selection_curve
from .forest_section import forest_section
from .membership import membership_section
from .sections import PageSection, StatTile, activations_section, cluster_ranges_section


class PageData(BaseModel):
    curve: SelectionCurve | None
    ranges: ActivationRanges | None
    forest: ForestData | None
    mixed_hits_page: str | None = None
    border_links: dict[str, str] = {}


class PageParts(BaseModel):
    header_html: str
    sections_html: str
    n_sections: int
    has_charts: bool


MIXED_HITS_PAGE = "mixed_hits.html"
"""The sheet of hits split between clusters, linked from the page when it exists."""


def load_page_data(report_dir: Path) -> PageData:
    return PageData(
        curve=load_selection_curve(report_dir),
        ranges=load_activation_ranges(report_dir),
        forest=load_forest_data(report_dir),
        mixed_hits_page=MIXED_HITS_PAGE if (report_dir / MIXED_HITS_PAGE).exists() else None,
        border_links=load_border_links(report_dir),
    )


def load_border_links(report_dir: Path) -> dict[str, str]:
    path = report_dir / BORDERS_DIR / BORDER_INDEX
    return json.loads(path.read_text()) if path.exists() else {}


def page_sections(data: PageData, meta: ReportMeta) -> list[PageSection]:
    cluster_ids = list(meta.clusters)
    built = [
        activations_section(data.curve),
        cluster_ranges_section(data.ranges, cluster_ids),
        forest_section(data.forest),
        membership_section(data.forest),
        borders_section(data.forest, data.mixed_hits_page, data.border_links, set(cluster_ids)),
    ]
    return [section for section in built if section is not None]


def stat_tiles(data: PageData, meta: ReportMeta) -> list[StatTile]:
    tiles = []
    if data.curve is not None:
        tiles.append(StatTile(value=f"{data.curve.n_positive:,}", name="positions in the tail"))
    tiles += [
        StatTile(value=f"{meta.n_hits:,}", name="hits"),
        StatTile(value=f"{len(meta.clusters)}", name="clusters shown"),
    ]
    if data.forest is not None:
        tiles.append(StatTile(value=f"{data.forest.n_trees}", name="trees in the forest"))
        if not math.isnan(data.forest.max_entropy):
            confident = int((data.forest.hit_entropy < data.forest.max_entropy).sum())
            tiles.append(StatTile(value=f"{confident:,}", name="confident hits"))
    return tiles


def page_parts(data: PageData, meta: ReportMeta) -> PageParts:
    sections = page_sections(data, meta)
    guide = [(step, section.title, entry) for step, section in enumerate(sections, 1) for entry in section.guide]
    return PageParts(
        header_html=render_template("page_header.html", stats=stat_tiles(data, meta), guide=guide),
        sections_html=render_template("page_sections.html", sections=sections),
        n_sections=len(sections),
        has_charts=bool(sections),
    )
