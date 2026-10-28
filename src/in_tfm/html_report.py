"""The report page shell, rendered from the templates in `report_templates/`.

Markup, CSS and JS live in those files, and this module only fills them in.
Template variables are autoescaped.
Parameters named `body` (or ending in `_html`) are already-built fragments, and templates mark them `|safe`.
"""

from collections.abc import Sequence

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape
from pydantic import BaseModel

from .cluster_colors import cluster_style

_env = Environment(
    loader=PackageLoader("in_tfm", "report_templates"),
    autoescape=select_autoescape(["html"]),
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=StrictUndefined,
)

class ClusterLink(BaseModel):
    cluster_id: int
    n_hits: int

    @property
    def style(self) -> str:
        return cluster_style(self.cluster_id)


def render_template(template: str, **context: object) -> str:
    return _env.get_template(template).render(**context)


SINK_TOGGLE = render_template("sink_toggle.html")


def page(title: str, body: str, plotly_js: str | None = None) -> str:
    """`plotly_js` loads plotly.js and the tab script, for a page with charts."""
    return render_template("page.html", title=title, body=body, plotly_js=plotly_js)


def details(summary: str, body: str, start_open: bool = False) -> str:
    return render_template("details.html", summary=summary, body=body, start_open=start_open)


def cluster_section(cluster_id: int, n_hits: int, n_unique_images: int, body: str) -> str:
    return render_template(
        "cluster_section.html",
        cluster_id=cluster_id,
        style=cluster_style(cluster_id),
        n_hits=n_hits,
        n_unique_images=n_unique_images,
        body=body,
    )


def cluster_nav(links: Sequence[ClusterLink]) -> str:
    return render_template("cluster_nav.html", links=links)


def neuron_index(
    title: str,
    threshold: float,
    n_hits: int,
    n_clusters: int,
    page_controls: str,
    activation_plot: str,
    cluster_nav: str,
    sections: Sequence[str],
    mixed_hits_page: str | None = None,
    header_html: str = "",
    page_sections_html: str = "",
    clusters_step: int = 1,
) -> str:
    return render_template(
        "neuron_index.html",
        mixed_hits_page=mixed_hits_page,
        title=title,
        threshold=threshold,
        n_hits=n_hits,
        n_clusters=n_clusters,
        page_controls=page_controls,
        activation_plot=activation_plot,
        cluster_nav=cluster_nav,
        sections="\n".join(sections),
        header_html=header_html,
        page_sections_html=page_sections_html,
        clusters_step=clusters_step,
    )
