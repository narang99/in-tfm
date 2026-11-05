"""A residual dim's card as combinations of its writers' clusters, and the token section they link to.

- A stream dim is a sum of writers, so one firing is one cluster per writer.
  - The embedding's cluster is the token itself. A reported writer's is its matched cluster.
- A combination is shown with the share of the dim's firings it made, and no magnitudes.
  - How the writers' sizes compare may change what matches, so the page does not rank by them yet.
"""

import os
from collections import Counter
from pathlib import Path

import numpy as np
from jaxtyping import Float
from pydantic import BaseModel

from ..cluster_colors import cluster_style
from ..colormaps import rd_bk_gn
from ..dependencies.collect import ClusterDependencies, TokenRef
from ..dependencies.firings import Example
from ..dependencies.stats import ComboPart, ComboShare
from ..viz import apply_cmap
from .clusters import ReportsByWriter, section_href
from .tiles import png_data_uri

SHOWN_COMBOS = 12
EXAMPLES_PER_TOKEN = 3
TOKEN_IMAGE_PX = 128


class ShownPart(BaseModel):
    text: str
    href: str | None = None
    style: str = ""
    image_src: str | None = None
    """The matched cluster's own `hadamard.jpg`, from the writer's report."""


class ComboRow(BaseModel):
    parts: list[ShownPart]
    count: int
    ratio: float


class ComboTable(BaseModel):
    rows: list[ComboRow]
    n_combos: int
    rest_count: int
    rest_ratio: float


def token_anchor(token_id: int) -> str:
    return f"token-{token_id}"


def shown_part(
    part: ComboPart,
    tokens: dict[int, TokenRef],
    reports: ReportsByWriter,
    page_dir: Path,
) -> ShownPart:
    if part.cluster_id is None:
        return ShownPart(text=f"{part.label} (no report)")
    if part.kind == "fixed":
        token = tokens[part.cluster_id]
        return ShownPart(
            text=repr(token.text), href=f"#{token_anchor(part.cluster_id)}"
        )
    report = reports[(part.column.tap.layer, part.column.column)]
    image = report.report_dir / f"cluster_{part.cluster_id}" / "hadamard.jpg"
    return ShownPart(
        text=f"{part.label} c{part.cluster_id}",
        href=section_href(report, part.cluster_id, page_dir),
        style=cluster_style(part.cluster_id),
        image_src=os.path.relpath(image, page_dir),
    )


def combo_table(
    combos: list[ComboShare],
    tokens: dict[int, TokenRef],
    reports: ReportsByWriter,
    page_dir: Path,
) -> ComboTable:
    shown, rest = combos[:SHOWN_COMBOS], combos[SHOWN_COMBOS:]
    return ComboTable(
        rows=[
            ComboRow(
                parts=[shown_part(p, tokens, reports, page_dir) for p in c.parts],
                count=c.count,
                ratio=c.ratio,
            )
            for c in shown
        ],
        n_combos=len(combos),
        rest_count=sum(c.count for c in rest),
        rest_ratio=sum(c.ratio for c in rest),
    )


class TokenExample(BaseModel):
    before: str
    token: str
    after: str


class TokenSection(BaseModel):
    token_id: int
    text: str
    n_examples: int
    image_uri: str
    examples: list[TokenExample]

    @property
    def anchor(self) -> str:
        return token_anchor(self.token_id)


def square_image_uri(row: Float[np.ndarray, "n"]) -> str:
    """A d-dim row drawn as a near-square grid, row-major, padded with zeros."""
    side = int(np.ceil(np.sqrt(len(row))))
    grid = np.zeros(side * side)
    grid[: len(row)] = row
    limit = max(float(np.abs(row).max()), 1e-8)
    image = apply_cmap(
        grid.reshape(side, side),
        rd_bk_gn,
        vmin=-limit,
        vmax=limit,
        size=(TOKEN_IMAGE_PX, TOKEN_IMAGE_PX),
    )
    return png_data_uri(image)


def distinct_examples(examples: list[Example]) -> list[TokenExample]:
    """TinyStories repeats whole stories, so the same context is shown once."""
    shown = {}
    for e in examples:
        shown.setdefault(
            "".join(e.context),
            TokenExample(
                before="".join(e.context[: e.focus]),
                token=e.context[e.focus],
                after="".join(e.context[e.focus + 1 :]),
            ),
        )
    return list(shown.values())


def token_sections(collected: ClusterDependencies) -> list[TokenSection]:
    """Every token of the cluster's examples, most examples first."""
    counts = Counter(e.token_id for e in collected.examples if e.token_id is not None)
    sections = []
    for token_id, count in counts.most_common():
        if token_id not in collected.tokens:
            continue
        token = collected.tokens[token_id]
        examples = [e for e in collected.examples if e.token_id == token_id]
        sections.append(
            TokenSection(
                token_id=token_id,
                text=token.text,
                n_examples=count,
                image_uri=square_image_uri(np.asarray(token.hadamard)),
                examples=distinct_examples(examples)[:EXAMPLES_PER_TOKEN],
            )
        )
    return sections
