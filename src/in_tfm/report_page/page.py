"""The parts of a neuron's index page above its clusters: stat tiles, a glossary, and numbered sections."""

import math
from pathlib import Path

from pydantic import BaseModel

from ..activation_ranges import ActivationRanges, load_activation_ranges
from ..html_report import render_template
from ..report_meta import ReportMeta
from .data import ForestData, SelectionCurve, load_forest_data, load_selection_curve
from .forest_section import forest_section
from .membership import membership_section
from .sections import PageSection, StatTile, activations_section, cluster_ranges_section


class PageData(BaseModel):
    curve: SelectionCurve | None
    ranges: ActivationRanges | None
    forest: ForestData | None


class PageParts(BaseModel):
    header_html: str
    sections_html: str
    n_sections: int
    has_charts: bool


def load_page_data(report_dir: Path) -> PageData:
    return PageData(
        curve=load_selection_curve(report_dir),
        ranges=load_activation_ranges(report_dir),
        forest=load_forest_data(report_dir),
    )


def page_sections(data: PageData, meta: ReportMeta) -> list[PageSection]:
    cluster_ids = list(meta.clusters)
    built = [
        activations_section(data.curve, data.ranges),
        cluster_ranges_section(data.ranges, cluster_ids),
        forest_section(data.forest),
        membership_section(data.forest),
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
