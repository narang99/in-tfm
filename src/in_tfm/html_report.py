"""The report page shell, rendered from the templates in `report_templates/`.

Markup, CSS and JS live in those files, and this module only fills them in.
Template variables are autoescaped.
Parameters named `body` (or ending in `_html`) are already-built fragments, and templates mark them `|safe`.
"""

from collections.abc import Sequence

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape
from pydantic import BaseModel

_env = Environment(
    loader=PackageLoader("in_tfm", "report_templates"),
    autoescape=select_autoescape(["html"]),
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=StrictUndefined,
)

ACCENTS = ("#3b82f6", "#8b5cf6", "#f59e0b", "#06b6d4", "#ec4899", "#64748b")
"""Blues, purples and ambers only: red and green already mean relevance sign inside the hits."""


class ClusterLink(BaseModel):
    cluster_id: int
    n_hits: int
    accent_class: str


def accent_class(position: int) -> str:
    return f"accent-{position % len(ACCENTS)}"


def _render(template: str, **context: object) -> str:
    return _env.get_template(template).render(**context)


SINK_TOGGLE = _render("sink_toggle.html")


def page(title: str, body: str) -> str:
    return _render("page.html", title=title, body=body, accents=ACCENTS)


def details(summary: str, body: str, start_open: bool = False) -> str:
    return _render("details.html", summary=summary, body=body, start_open=start_open)


def cluster_section(
    cluster_id: int, position: int, n_hits: int, n_unique_images: int, body: str
) -> str:
    return _render(
        "cluster_section.html",
        cluster_id=cluster_id,
        accent_class=accent_class(position),
        n_hits=n_hits,
        n_unique_images=n_unique_images,
        body=body,
    )


def cluster_nav(links: Sequence[ClusterLink]) -> str:
    return _render("cluster_nav.html", links=links)


def neuron_index(
    title: str,
    threshold: float,
    n_hits: int,
    n_clusters: int,
    page_controls: str,
    activation_plot: str,
    cluster_nav: str,
    sections: Sequence[str],
) -> str:
    return _render(
        "neuron_index.html",
        title=title,
        threshold=threshold,
        n_hits=n_hits,
        n_clusters=n_clusters,
        page_controls=page_controls,
        activation_plot=activation_plot,
        cluster_nav=cluster_nav,
        sections="\n".join(sections),
    )
