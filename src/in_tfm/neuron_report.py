"""Per-cluster inspection: sample hits from one HDBSCAN cluster of a neuron's Hadamard
products, run attribution on each, and hand them to a presenter for rendering.

Sampling needs the same handful of arrays/handles (model, source, sample ids,
layer/neuron/token/batch indices, cluster labels, hadamard products) - threading all of that
through free functions individually got unwieldy, hence one class holding it all.

Modality lives entirely in the injected `source` and `presenter`; this class only orchestrates.

Report writing is split in two so the presenter can be iterated on without a model or a GPU:

- `NeuronClusterHits.write_report` samples and attributes each cluster, caching exactly what the
  presenter receives to `cluster_{id}/hits.pt`, alongside `meta.json`.
- `render_report` builds `index.html` from those caches alone. `write_report` calls it too, so a
  re-render of a downloaded report produces the same page the pipeline did.
"""

import gc
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from tqdm import tqdm

from .activation_ranges import ActivationRanges, PLOT_NAME, RANGES_NAME, cluster_activations, write_activation_ranges
from .attribution import compute_attnlrp_relevance
from .device import default_device, empty_cache
from .hit_selection import BucketedPick, ElbowPick, HitSelection
from .html_report import (
    ClusterLink,
    accent_class,
    cluster_nav,
    cluster_section,
    details,
    neuron_index,
    page,
)
from .layers import LayerGetter
from .presenters import ClusterArtifacts, ClusterHit, ClusterPresenter, HadamardShape
from .report_page import (
    MIXED_HITS_PAGE, PLOTLY_JS, SELECTION_DATA, ForestData, PageData, cluster_stats_html, load_page_data, page_parts, save_forest_data,
    save_selection_curve, selection_curve,
)
from .report_meta import (
    BucketedPickMeta,
    ClusterMeta,
    ElbowPickMeta,
    ReportMeta,
    load_report_meta,
    write_report_meta,
)
from .sources import SampleId, SampleSource

AttrFn = Callable[..., np.ndarray]


HITS_CACHE_NAME = "hits.pt"
SAMPLE_SEED = 0
COLLECT_EVERY = 32
"""Hits attributed between garbage collections.

- Collecting after every hit was 44% of the cluster report's time on mixed4e:55, in a py-spy profile.
- Collecting now and then still frees each hit's batch and graph, so memory stays bounded."""


def save_hits(hits: list[ClusterHit], cluster_dir: Path) -> None:
    """Stored as plain dicts rather than pickled `ClusterHit`s, so adding a defaulted field to
    `ClusterHit` does not orphan caches written before it."""
    torch.save([hit.model_dump() for hit in hits], cluster_dir / HITS_CACHE_NAME)


def load_hits(cluster_dir: Path) -> list[ClusterHit]:
    # weights_only=False: the cache holds numpy arrays, which the safe loader rejects
    saved = torch.load(cluster_dir / HITS_CACHE_NAME, weights_only=False)
    return [ClusterHit.model_validate(d) for d in saved]


def cluster_dir_for(report_dir: Path, cluster_id: int) -> Path:
    return report_dir / f"cluster_{cluster_id}"




def render_report(report_dir: str | Path, presenter: ClusterPresenter) -> Path:
    """`index.html` from `meta.json` and each cluster's `hits.pt` - no model needed."""
    report_dir = Path(report_dir)
    meta = load_report_meta(report_dir)
    data = load_page_data(report_dir)
    sections = [
        _cluster_section(
            cid, cluster, position, cluster_dir_for(report_dir, cid), presenter, meta.hadamard_shape,
            cluster_stats_html(data.forest, cid),
        )
        for position, (cid, cluster) in enumerate(meta.clusters.items())
    ]
    report_path = report_dir / "index.html"
    report_path.write_text(_index_page(report_dir, data, meta, sections, presenter.page_controls(), data.mixed_hits_page))
    return report_path


def _cluster_section(
    cluster_id: int,
    cluster: ClusterMeta,
    position: int,
    cluster_dir: Path,
    presenter: ClusterPresenter,
    hadamard_shape: HadamardShape,
    stats_html: str = "",
) -> str:
    """The header is sticky and accent-coloured so that, mid-scroll, it is obvious which
    cluster the hits on screen belong to and when that changes."""
    hits = load_hits(cluster_dir)
    artifacts = ClusterArtifacts(cluster_dir=cluster_dir)
    body = presenter.render(hits, artifacts, hadamard_shape) if hits else "<p>no hits sampled.</p>"
    return cluster_section(cluster_id, position, cluster.n_hits, cluster.n_unique_images, stats_html + body)


def _cluster_nav(meta: ReportMeta) -> str:
    return cluster_nav(
        [
            ClusterLink(cluster_id=cid, n_hits=c.n_hits, accent_class=accent_class(i))
            for i, (cid, c) in enumerate(meta.clusters.items())
        ]
    )


def _activation_plot(meta: ReportMeta) -> str:
    """Reports written before `selection.npz` keep their saved image."""
    match meta.pick:
        case ElbowPickMeta():
            return details("activation threshold (elbow plot)", '<img class="elbow" src="elbow.png" alt="elbow plot">')
        case BucketedPickMeta():
            return details("activation buckets", '<img class="elbow" src="buckets.png" alt="bucket edges plot">')


def _activation_ranges_plot(report_dir: Path) -> str:
    """Absent from reports written before the plot existed, and for a neuron with no clusters."""
    if not (report_dir / PLOT_NAME).exists():
        return ""
    return details("activation ranges per cluster", f'<img src="{PLOT_NAME}" alt="activation range per cluster">')


def _legacy_plots(report_dir: Path, meta: ReportMeta) -> str:
    """The images a report saved before its charts had data files. Each is shown only if its chart is missing."""
    plots = "" if (report_dir / SELECTION_DATA).exists() else _activation_plot(meta)
    return plots if (report_dir / RANGES_NAME).exists() else plots + _activation_ranges_plot(report_dir)


def _index_page(
    report_dir: Path,
    data: PageData,
    meta: ReportMeta,
    sections: list[str],
    page_controls: str,
    mixed_hits_page: str | None,
) -> str:
    title = f"Neuron {meta.neuron_idx}"
    parts = page_parts(data, meta)
    body = neuron_index(
        title=title,
        threshold=meta.threshold,
        n_hits=meta.n_hits,
        n_clusters=len(sections),
        page_controls=page_controls,
        activation_plot=_legacy_plots(report_dir, meta),
        cluster_nav=_cluster_nav(meta),
        sections=sections,
        mixed_hits_page=mixed_hits_page,
        header_html=parts.header_html,
        page_sections_html=parts.sections_html,
        clusters_step=parts.n_sections + 1,
    )
    return page(title, body, PLOTLY_JS if parts.has_charts else None)


def cluster_counts(labels: Int[np.ndarray, "n_hits"]) -> dict[int, int]:
    """Cluster id -> hit count, excluding HDBSCAN's noise label (-1)."""
    ids, counts = np.unique(labels, return_counts=True)
    return {int(cid): int(count) for cid, count in zip(ids, counts) if cid != -1}


def clusters_by_frequency(labels: Int[np.ndarray, "n_hits"]) -> list[int]:
    counts = cluster_counts(labels)
    return sorted(counts, key=counts.get, reverse=True)


def cluster_unique_image_counts(
    labels: Int[np.ndarray, "n_hits"], batch_idx: Int[np.ndarray, "n_hits"]
) -> dict[int, int]:
    """Distinct source images per cluster - a cluster drawn from only one image is likely a
    repeated local artifact (e.g. an annotation glyph), not a genuine cross-image pattern."""
    return {cid: len(np.unique(batch_idx[labels == cid])) for cid in cluster_counts(labels)}



def sampled_hit_indices(labels: Int[np.ndarray, "n_hits"], cluster_id: int, max_n: int) -> Int[np.ndarray, "k"]:
    """A random but repeatable pick, so two runs over the same hits show the same ones.

    - Reports of the same hits under different attribution methods are only comparable if they show the same hits.
    """
    members = np.argwhere(labels == cluster_id).reshape(-1)
    return np.random.default_rng([SAMPLE_SEED, cluster_id]).permutation(members)[:max_n]


class NeuronClusterHits(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: torch.nn.Module
    source: SampleSource
    presenter: ClusterPresenter
    sample_ids: list[SampleId]
    layer_getter: LayerGetter
    neuron_idx: int
    selection: HitSelection
    labels: Int[np.ndarray, "n_hits"]
    """One per hit of `selection`, in the same order."""
    hdmds: Float[np.ndarray, "n_hits hidden"]
    hadamard_shape: HadamardShape
    noise_activations: Float[np.ndarray, "n"] | None = None
    """With it, the report gets a collapsed activation range plot."""
    device: str = Field(default_factory=default_device)
    _hits_since_collect: int = PrivateAttr(0)

    def sample_cluster(
        self, cluster_id: int, max_n: int = 5, attr_fn: AttrFn = compute_attnlrp_relevance
    ) -> list[ClusterHit]:
        return [self.cluster_hit(int(i), attr_fn) for i in tqdm(sampled_hit_indices(self.labels, cluster_id, max_n))]

    def cluster_hit(self, hit: int, attr_fn: AttrFn = compute_attnlrp_relevance) -> ClusterHit:
        """Hit `hit` of `selection`, attributed and ready for the presenter."""
        bid, tid = self.selection.batch_idx[hit], self.selection.token_idx[hit]
        sample_id = self.sample_ids[bid]
        singleton_batch = self.source.to_model_batch([sample_id]).to(self.device)
        relevance = attr_fn(self.model.to(self.device), singleton_batch, self.layer_getter, self.neuron_idx, tid)
        result = ClusterHit(
            sample_id=sample_id,
            token_idx=int(tid),
            relevance=relevance,
            model_input=singleton_batch.grad_leaf.detach().cpu(),
            hadamard=self.hdmds[hit],
            display_ids=None if singleton_batch.display_ids is None else singleton_batch.display_ids[0].cpu(),
        )
        del singleton_batch
        self._collect_now_and_then()
        return result

    def _collect_now_and_then(self) -> None:
        self._hits_since_collect += 1
        if self._hits_since_collect >= COLLECT_EVERY:
            gc.collect()
            empty_cache()
            self._hits_since_collect = 0

    def shown_cluster_ids(self, min_uniq_images: int) -> list[int]:
        """Largest first, without clusters drawn from fewer than `min_uniq_images` distinct images."""
        uniq_images = cluster_unique_image_counts(self.labels, self.selection.batch_idx)
        return [cid for cid in clusters_by_frequency(self.labels) if uniq_images[cid] >= min_uniq_images]

    def cluster_mean_hadamard(self, cluster_id: int) -> Float[np.ndarray, "hidden"]:
        return self._cluster_mean_hadamard(cluster_id).numpy()

    def _cluster_mean_hadamard(self, cluster_id: int) -> torch.Tensor:
        """Mean Hadamard product over every hit HDBSCAN placed in this cluster (not just the
        handful sampled for display) - a per-cluster prototype other data can later be matched
        against via cosine similarity."""
        cluster_hdmds = self.hdmds[self.labels == cluster_id]
        return torch.from_numpy(cluster_hdmds.mean(axis=0))

    def _cache_cluster(
        self, cluster_id: int, report_dir: Path, attr_fn: AttrFn, max_n: int
    ) -> None:
        """Writes the cluster's presenter input (`hits.pt`) and its mean Hadamard prototype
        (`mean.pt`) into its own `cluster_{id}/` subfolder."""
        cluster_dir = cluster_dir_for(report_dir, cluster_id)
        cluster_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._cluster_mean_hadamard(cluster_id), cluster_dir / "mean.pt")
        save_hits(self.sample_cluster(cluster_id, max_n=max_n, attr_fn=attr_fn), cluster_dir)

    def _meta(self, cluster_ids: list[int], min_uniq_images: int) -> ReportMeta:
        counts = cluster_counts(self.labels)
        uniq_images = cluster_unique_image_counts(self.labels, self.selection.batch_idx)
        return ReportMeta(
            neuron_idx=self.neuron_idx,
            pick=self._pick_meta(),
            n_hits=len(self.labels),
            min_uniq_images_per_cluster=min_uniq_images,
            hadamard_shape=self.hadamard_shape,
            clusters={
                cid: ClusterMeta(n_hits=counts[cid], n_unique_images=uniq_images[cid])
                for cid in cluster_ids
            },
            sample_ids=list(self.sample_ids),
        )

    def _pick_meta(self) -> ElbowPickMeta | BucketedPickMeta:
        threshold = self.selection.threshold
        match self.selection.pick:
            case ElbowPick():
                return ElbowPickMeta(cutoff=threshold)
            case BucketedPick(bucket_edges=edges):
                return BucketedPickMeta(floor=threshold, bucket_edges=edges.tolist())

    def _write_activation_ranges(self, report_dir: Path, cluster_ids: list[int]) -> None:
        if self.noise_activations is None:
            return
        ranges = ActivationRanges(
            clusters=cluster_activations(self.labels, self.selection.activations, cluster_ids),
            noise=self.noise_activations.tolist(),
        )
        write_activation_ranges(report_dir, ranges)

    def write_report(
        self,
        out_dir: str | Path,
        name: str | None = None,
        attr_fn: AttrFn = compute_attnlrp_relevance,
        max_n: int = 5,
        min_uniq_images: int = 2,
        forest: ForestData | None = None,
    ) -> Path:
        """Self-contained report folder for this neuron: `out_dir/{name}/index.html`, every
        cluster largest first (each in its own `cluster_{id}/` subfolder), the elbow plot and
        threshold metadata that picked this neuron's activation cutoff, all sitting under
        report_dir - so the whole folder can be zipped/tarred and read standalone.

        Clusters drawn from fewer than `min_uniq_images` distinct source images are dropped -
        see cluster_unique_image_counts for why.

        `name` defaults to `neuron_{idx}` but can be overridden.
        `forest` adds the forest sections to the page. Only density-peak clustering has one.
        """
        report_dir = Path(out_dir) / (name or f"neuron_{self.neuron_idx}")
        report_dir.mkdir(parents=True, exist_ok=True)

        save_selection_curve(report_dir, selection_curve(self.selection))
        if forest is not None:
            save_forest_data(report_dir, forest)

        cluster_ids = self.shown_cluster_ids(min_uniq_images)
        meta = self._meta(cluster_ids, min_uniq_images)
        write_report_meta(report_dir, meta)

        self._write_activation_ranges(report_dir, cluster_ids)
        for cid in cluster_ids:
            self._cache_cluster(cid, report_dir, attr_fn, max_n)
        return render_report(report_dir, self.presenter)
