#!/usr/bin/env python
"""Group the clusters of finished neuron reports into collapsible sections, by k-means over the cluster means.

    uv run python scripts/group_report_clusters.py --report-dir reports_x/state_0000/neuron_0
    uv run python scripts/group_report_clusters.py --reports-root reports_x --k 5

What it adds to each report page, without changing any library code:
- A PCA plot of the hits, coloured by group.
- An elbow plot of k-means over the cluster means, with a silhouette cross-check.
- A cluster similarity table, ordered by group.
- The cluster sections moved into one collapsed section per group.

Each group gets `group_{i}/mean.pt`, the unweighted mean of its clusters' means, and `groups.json` lists the members.
The original page is kept as `index.original.html` and every run starts from it, so a run can be repeated with another `--k`.

PCA needs every hit's vector, which a report does not keep.
- By default it uses the few hits each cluster displays, plus the cluster means.
- `--full-pca` recomputes all hits from the saved network, so it needs `--states-dir` and takes about a minute per report.
"""

import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from jaxtyping import Float
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from in_tfm.activation_range_plot import BACKGROUND, KELLY_COLORS

ORIGINAL_NAME = "index.original.html"
GROUPS_NAME = "groups.json"
NOISE_COLOR = "#555555"
MIN_CLUSTERS_TO_GROUP = 3
CLUSTER_NAV = '<nav class="cluster-nav">'
MATCH_MARKER = "best match of each group"


class GroupConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True, cli_parse_args=True, cli_implicit_flags=True)

    report_dir: Path | None = None
    reports_root: Path | None = None
    neuron: int = 0
    k: int | None = None
    seed: int = 0
    full_pca: bool = False
    states_dir: Path = Path("checkpoints/stl_tracking_resized/states")
    report_config: Path = Path("configs/stl_neuron_tracking.yaml")


class Cluster(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: int
    n_hits: int
    mean: torch.Tensor


class Group(BaseModel):
    id: int
    cluster_ids: list[int]
    n_hits: int


class Sweep(BaseModel):
    ks: list[int]
    inertia: list[float]
    silhouette: dict[int, float]
    chosen_k: int


def load_clusters(report_dir: Path) -> list[Cluster]:
    meta = json.loads((report_dir / "meta.json").read_text())
    return [
        Cluster(
            id=int(cid),
            n_hits=cluster["n_hits"],
            mean=torch.load(report_dir / f"cluster_{cid}" / "mean.pt").double(),
        )
        for cid, cluster in meta["clusters"].items()
    ]


def unit_means(clusters: list[Cluster]) -> Float[np.ndarray, "clusters hidden"]:
    return torch.nn.functional.normalize(torch.stack([c.mean for c in clusters]), dim=1).numpy()


def elbow_index(values: list[float]) -> int:
    """The point farthest from the straight line between the first and last points."""
    points = np.stack([np.arange(len(values)), values], axis=1).astype(float)
    points = (points - points[0]) / (points[-1] - points[0])
    return int(np.argmax(np.abs(points[:, 1] - points[:, 0])))


def fit_kmeans(x: np.ndarray, k: int, seed: int) -> KMeans:
    return KMeans(k, n_init=20, random_state=seed).fit(x)


def sweep_k(x: np.ndarray, forced_k: int | None, seed: int) -> Sweep:
    ks = list(range(1, len(x)))
    fits = {k: fit_kmeans(x, k, seed) for k in ks}
    inertia = [fits[k].inertia_ for k in ks]
    silhouette = {k: float(silhouette_score(x, fits[k].labels_)) for k in ks if k >= 2}
    return Sweep(ks=ks, inertia=inertia, silhouette=silhouette, chosen_k=forced_k or ks[elbow_index(inertia)])


def make_groups(clusters: list[Cluster], assignment: np.ndarray) -> list[Group]:
    """Numbered by hit count, largest first."""
    members: dict[int, list[Cluster]] = {}
    for cluster, label in zip(clusters, assignment):
        members.setdefault(int(label), []).append(cluster)
    ordered = sorted(members.values(), key=lambda m: -sum(c.n_hits for c in m))
    return [
        Group(id=i, cluster_ids=[c.id for c in m], n_hits=sum(c.n_hits for c in m)) for i, m in enumerate(ordered)
    ]


def group_mean(group: Group, clusters: dict[int, Cluster]) -> torch.Tensor:
    return torch.stack([clusters[cid].mean for cid in group.cluster_ids]).mean(dim=0)


def write_group_artifacts(report_dir: Path, groups: list[Group], clusters: dict[int, Cluster]) -> None:
    (report_dir / GROUPS_NAME).write_text(json.dumps([g.model_dump() for g in groups], indent=1))
    for group in groups:
        group_dir = report_dir / f"group_{group.id}"
        group_dir.mkdir(exist_ok=True)
        torch.save(group_mean(group, clusters).float(), group_dir / "mean.pt")


def group_of_cluster(groups: list[Group]) -> dict[int, int]:
    return {cid: g.id for g in groups for cid in g.cluster_ids}


def group_color(group_id: int) -> str:
    return KELLY_COLORS[group_id % len(KELLY_COLORS)]


def save_elbow_plot(sweep: Sweep, path: Path) -> None:
    fig = Figure(figsize=(10, 3.6))
    left, right = fig.subplots(1, 2)
    left.plot(sweep.ks, sweep.inertia, "o-")
    left.set_title("elbow on the cluster means")
    left.set_ylabel("inertia")
    right.plot(list(sweep.silhouette), list(sweep.silhouette.values()), "o-", color="tab:green")
    right.set_title("silhouette cross-check")
    right.set_ylabel("silhouette")
    for ax in (left, right):
        ax.axvline(sweep.chosen_k, color="tab:orange", ls="--", label=f"k={sweep.chosen_k}")
        ax.set_xlabel("k")
    left.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)


class PcaPoints(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectors: np.ndarray
    cluster_ids: np.ndarray
    """Cluster id of each vector, -1 for noise and dropped clusters."""
    note: str


def points_from_disk(report_dir: Path, clusters: list[Cluster]) -> PcaPoints:
    vectors, ids = [], []
    for cluster in clusters:
        for hit in torch.load(report_dir / f"cluster_{cluster.id}" / "hits.pt", weights_only=False):
            vectors.append(np.asarray(hit["hadamard"], dtype=np.float64))
            ids.append(cluster.id)
    return PcaPoints(vectors=np.stack(vectors), cluster_ids=np.array(ids), note="hits shown in the report")


def points_from_model(report_dir: Path, config: GroupConfig) -> PcaPoints:
    """Reruns pass 1, 2 and the clustering on the saved network, which the report folder does not keep."""
    from datasets import load_dataset

    import in_tfm.neuron_run as neuron_run
    from in_tfm.clustering import cluster_labels
    from in_tfm.hadamard import hadamard_from_rows
    from in_tfm.hit_selection import select_hits
    from in_tfm.models.stl_inception.adapter import StlInceptionAdapter, load_model
    from in_tfm.models.stl_inception.source import HubImages, spread_indices
    from in_tfm.neuron_tracking import state_path
    from in_tfm.run_config import load_run_config

    state = int(report_dir.parent.name.removeprefix("state_"))
    run = load_run_config(["--config", str(config.report_config), "--out-dir", str(report_dir.parent)])
    dataset = HubImages(load_dataset(run.stl.dataset, split=run.stl.split))
    model = load_model(str(state_path(config.states_dir, state)))
    adapter = StlInceptionAdapter(model, run.layer_name, dataset, run.stl.split, run.stl.attribution, run.stl.ig_steps)
    prepared = neuron_run.prepare_run(run, adapter, spread_indices(len(dataset), run.stl.n_images))
    neuron = run.neuron_idxs[0]
    scan = prepared.capture.scan()
    selection = select_hits(scan, neuron, "positive", run.hits, run.seed)
    gathered = prepared.capture.gather(selection.batch_idx, selection.token_idx)
    weight = prepared.layout.weight(neuron_run.layer_getter_for(run)(prepared.module).weight)
    vectors = hadamard_from_rows(gathered.inputs, weight, neuron)
    clustering = run.clustering
    labels = cluster_labels(vectors, clustering.min_cluster_size, clustering.selection_method, clustering.min_samples)
    return PcaPoints(vectors=vectors, cluster_ids=labels, note="all clustered hits")


def save_pca_plot(
    points: PcaPoints, clusters: list[Cluster], groups: list[Group], state_name: str, path: Path
) -> None:
    pca = PCA(n_components=4).fit(points.vectors)
    xy = pca.transform(points.vectors)
    means_xy = pca.transform(np.stack([c.mean.numpy() / c.mean.norm().item() for c in clusters]))
    of_cluster = group_of_cluster(groups)
    point_groups = np.array([of_cluster.get(int(c), -1) for c in points.cluster_ids])
    fig = Figure(figsize=(14, 6), facecolor=BACKGROUND)
    for ax, (a, b) in zip(fig.subplots(1, 2), [(0, 1), (2, 3)]):
        ax.set_facecolor(BACKGROUND)
        grey = point_groups == -1
        ax.scatter(xy[grey, a], xy[grey, b], s=3, c=NOISE_COLOR, alpha=0.4, linewidths=0)
        for group in groups:
            mask = point_groups == group.id
            ax.scatter(xy[mask, a], xy[mask, b], s=6, c=group_color(group.id), alpha=0.8, linewidths=0, label=f"group {group.id}")
        for cluster, (x, y) in zip(clusters, means_xy[:, [a, b]]):
            ax.scatter([x], [y], s=60, marker="*", c=group_color(of_cluster[cluster.id]), edgecolors="white", linewidths=0.6)
            if (a, b) == (0, 1):
                ax.text(x, y, f" {cluster.id}", color="white", fontsize=9, weight="bold")
        ax.set_xlabel(f"PC{a + 1} ({pca.explained_variance_ratio_[a]:.0%})", color="white")
        ax.set_ylabel(f"PC{b + 1} ({pca.explained_variance_ratio_[b]:.0%})", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#888888")
    fig.legend(*fig.axes[0].get_legend_handles_labels(), loc="lower center", ncol=len(groups), frameon=False, labelcolor="white")
    fig.suptitle(f"{state_name}: PCA of {points.note} ({len(points.vectors)} points), stars are cluster means", color="white")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=110, facecolor=BACKGROUND)


def save_group_mean_image(mean: torch.Tensor, shape: tuple[int, int], path: Path) -> None:
    fig = Figure(figsize=(2.2, 1.6))
    ax = fig.subplots()
    limit = float(mean.abs().max())
    ax.imshow(mean.reshape(shape).numpy(), cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=80)


def similarity_cell(value: float) -> str:
    color = "59,130,246" if value >= 0 else "245,158,11"
    return f'<td style="background:rgba({color},{0.6 * min(abs(value), 1):.2f})">{value:.2f}</td>'


def similarity_table_html(clusters: list[Cluster], groups: list[Group]) -> str:
    by_id = {c.id: c for c in clusters}
    order = [cid for g in groups for cid in g.cluster_ids]
    unit = torch.nn.functional.normalize(torch.stack([by_id[cid].mean for cid in order]), dim=1)
    sim = (unit @ unit.T).tolist()
    head = "".join(f'<th><a href="#cluster-{cid}">{cid}</a></th>' for cid in order)
    rows = "".join(
        f'<tr><th><a href="#cluster-{cid}">{cid}</a></th>{"".join(similarity_cell(v) for v in row)}</tr>'
        for cid, row in zip(order, sim)
    )
    return f'<div class="similarity"><table><tr><th></th>{head}</tr>{rows}</table></div>'


def group_summary_rows(groups: list[Group]) -> str:
    return "".join(
        f'<tr><td><a href="#group-{g.id}">group {g.id}</a></td>'
        f'<td>{", ".join(f"<a href=\"#cluster-{cid}\">{cid}</a>" for cid in g.cluster_ids)}</td>'
        f'<td>{g.n_hits}</td><td><img src="group_{g.id}/mean.png" alt="group {g.id} mean"></td></tr>'
        for g in groups
    )


STYLE = """<style>
.similarity { overflow-x:auto; margin-top:8px; }
.similarity table { border-collapse:collapse; font-size:0.7rem; font-variant-numeric:tabular-nums; }
.similarity th, .similarity td { padding:2px 6px; text-align:center; border:1px solid var(--line); }
.similarity th a { color:var(--muted); text-decoration:none; }
.groups-table { border-collapse:collapse; font-size:0.85rem; margin-top:8px; }
.groups-table th, .groups-table td { padding:4px 10px; border-bottom:1px solid var(--line); text-align:left; }
.groups-table img { height:48px; border-radius:2px; }
details.group { border:1px solid var(--line); border-left:8px solid var(--group); border-radius:8px; margin:24px 0; padding:4px 12px; }
details.group > summary { font-size:1rem; color:var(--fg); padding:8px 0; }
details.group > summary img { height:36px; vertical-align:middle; margin-left:12px; }
</style>
"""

OPEN_ON_LINK = """<script>
function openAncestors() {
  var el = document.getElementById(location.hash.slice(1));
  for (; el; el = el.parentElement) { if (el.tagName === 'DETAILS') { el.open = true; } }
}
addEventListener('hashchange', openAncestors);
openAncestors();
</script>
"""


def extras_html(clusters: list[Cluster], groups: list[Group], sweep: Sweep) -> str:
    return (
        STYLE
        + f'<details><summary>groups (k-means over the cluster means, k={sweep.chosen_k})</summary>'
        + f'<table class="groups-table"><tr><th>group</th><th>clusters</th><th>hits</th><th>mean</th></tr>{group_summary_rows(groups)}</table>'
        + '<img src="kmeans_elbow.png" alt="k-means elbow" style="max-width:700px"></details>\n'
        + '<details><summary>PCA projection of the hits (coloured by group)</summary>'
        + '<img src="pca_projection.jpg" alt="PCA projection"></details>\n'
        + f"<details><summary>cluster similarity, ordered by group (cosine of mean Hadamard products)</summary>{similarity_table_html(clusters, groups)}</details>\n"
    )


def split_sections(html: str) -> dict[int, str]:
    sections = re.findall(r'<section class="cluster .*?</section>\n?', html, flags=re.S)
    return {int(re.search(r'id="cluster-(\d+)"', s).group(1)): s for s in sections}


def grouped_sections_html(groups: list[Group], sections: dict[int, str]) -> str:
    return "".join(
        f'<details class="group" id="group-{g.id}" style="--group:{group_color(g.id)}"><summary><b>Group {g.id}</b>'
        f' &middot; clusters {", ".join(map(str, g.cluster_ids))} &middot; {g.n_hits} hits'
        f'<img src="group_{g.id}/mean.png" alt="mean"></summary>\n{"".join(sections[cid] for cid in g.cluster_ids)}</details>\n'
        for g in groups
    )


def rebuild_page(original: str, clusters: list[Cluster], groups: list[Group], sweep: Sweep) -> str:
    sections = split_sections(original)
    assert set(sections) == {c.id for c in clusters}, "the page's cluster sections do not match meta.json"
    page = original
    for section in sections.values():
        page = page.replace(section, "")
    assert page.count(CLUSTER_NAV) == 1
    nav_end = page.index("</nav>", page.index(CLUSTER_NAV)) + len("</nav>")
    page = page[:nav_end] + "\n" + grouped_sections_html(groups, sections) + page[nav_end:]
    page = page.replace(CLUSTER_NAV, extras_html(clusters, groups, sweep) + CLUSTER_NAV, 1)
    return page.replace("</body>", OPEN_ON_LINK + "</body>")


def group_report(report_dir: Path, config: GroupConfig) -> str:
    clusters = load_clusters(report_dir)
    if len(clusters) < MIN_CLUSTERS_TO_GROUP:
        return f"{len(clusters)} clusters, nothing to group"
    original_path = report_dir / ORIGINAL_NAME
    if not original_path.exists():
        shutil.copy(report_dir / "index.html", original_path)
    sweep = sweep_k(unit_means(clusters), config.k, config.seed)
    groups = make_groups(clusters, fit_kmeans(unit_means(clusters), sweep.chosen_k, config.seed).labels_)
    by_id = {c.id: c for c in clusters}
    write_group_artifacts(report_dir, groups, by_id)
    shape = tuple(json.loads((report_dir / "meta.json").read_text())["hadamard_shape"])
    for group in groups:
        save_group_mean_image(group_mean(group, by_id), shape, report_dir / f"group_{group.id}" / "mean.png")
    save_elbow_plot(sweep, report_dir / "kmeans_elbow.png")
    points = points_from_model(report_dir, config) if config.full_pca else points_from_disk(report_dir, clusters)
    save_pca_plot(points, clusters, groups, report_dir.parent.name, report_dir / "pca_projection.jpg")
    (report_dir / "index.html").write_text(rebuild_page(original_path.read_text(), clusters, groups, sweep))
    return f"k={sweep.chosen_k}, groups " + " | ".join(",".join(map(str, g.cluster_ids)) for g in groups)


class GroupMean(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: int
    unit_mean: torch.Tensor


def load_group_means(report_dir: Path) -> list[GroupMean]:
    groups = json.loads((report_dir / GROUPS_NAME).read_text())
    return [
        GroupMean(
            id=g["id"],
            unit_mean=torch.nn.functional.normalize(torch.load(report_dir / f"group_{g['id']}" / "mean.pt").double(), dim=0),
        )
        for g in groups
    ]


def previous_report_dir(report_dir: Path) -> Path | None:
    """The report of the state one step earlier, when it sits next to this one and has been grouped."""
    state = int(report_dir.parent.name.removeprefix("state_"))
    previous = report_dir.parent.parent / f"state_{state - 1:04d}" / report_dir.name
    return previous if (previous / GROUPS_NAME).exists() else None


def match_table_html(report_dir: Path, previous_dir: Path) -> str:
    current, previous = load_group_means(report_dir), load_group_means(previous_dir)
    similarity = torch.stack([g.unit_mean for g in current]) @ torch.stack([g.unit_mean for g in previous]).T
    previous_name = previous_dir.parent.name
    link = f"../../{previous_name}/{previous_dir.name}/index.html"
    rows = "".join(
        f'<tr><td><a href="#group-{g.id}">Group {g.id}</a></td>'
        f'<td><a href="{link}#group-{previous[b].id}">Group {previous[b].id}</a> ({previous_name})</td>'
        f"<td>{similarity[i, b].item():.3f}</td></tr>"
        for i, (g, b) in enumerate(zip(current, similarity.argmax(dim=1).tolist()))
    )
    return (
        f"<details open><summary>best match of each group in {previous_name} (cosine of group means)</summary>"
        f'<table class="groups-table"><tr><th>section in this state</th><th>most similar section in {previous_name}</th>'
        f"<th>cosine</th></tr>{rows}</table></details>\n"
    )


def add_match_table(report_dir: Path) -> str:
    """Run after every report is grouped, since a page is rebuilt from its original on each grouping run."""
    previous_dir = previous_report_dir(report_dir)
    if previous_dir is None or not (report_dir / GROUPS_NAME).exists():
        return "no previous state"
    index = report_dir / "index.html"
    html = index.read_text()
    marker = "<details><summary>PCA projection"
    assert html.count(marker) == 1 and MATCH_MARKER not in html
    index.write_text(html.replace(marker, match_table_html(report_dir, previous_dir) + marker))
    return f"matched to {previous_dir.parent.name}"


def report_dirs(config: GroupConfig) -> list[Path]:
    if config.report_dir is not None:
        return [config.report_dir]
    assert config.reports_root is not None, "pass --report-dir or --reports-root"
    found = sorted(config.reports_root.glob(f"state_*/neuron_{config.neuron}"))
    return [d for d in found if (d / "meta.json").exists()]


def main() -> None:
    config = GroupConfig()
    dirs = report_dirs(config)
    for report_dir in dirs:
        print(f"{report_dir.parent.name}: {group_report(report_dir, config)}", flush=True)
    for report_dir in dirs:
        print(f"{report_dir.parent.name}: {add_match_table(report_dir)}", flush=True)


if __name__ == "__main__":
    main()
