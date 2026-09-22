"""Per-cluster inspection: sample hits from one HDBSCAN cluster of a neuron's Hadamard
products, run attribution on each, and hand them to a presenter for rendering.

Sampling needs the same handful of arrays/handles (model, source, sample ids,
layer/neuron/token/batch indices, cluster labels, hadamard products) - threading all of that
through free functions individually got unwieldy, hence one class holding it all.

Modality lives entirely in the injected `source` and `presenter`; this class only orchestrates.
"""

import gc
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from .attribution import compute_attnlrp_relevance
from .device import default_device, empty_cache
from .html_report import page
from .layers import LayerGetter
from .presenters import ClusterHit, ClusterPresenter
from .sources import SampleId, SampleSource
from .viz import save_elbow_plot

AttrFn = Callable[..., np.ndarray]


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


class NeuronClusterHits(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: torch.nn.Module
    source: SampleSource
    presenter: ClusterPresenter
    sample_ids: list[SampleId]
    layer_getter: LayerGetter
    neuron_idx: int
    token_idx: Int[np.ndarray, "n_hits"]
    batch_idx: Int[np.ndarray, "n_hits"]
    labels: Int[np.ndarray, "n_hits"]
    hdmds: Float[np.ndarray, "n_hits hidden"]
    threshold: float
    elbow_values: Float[np.ndarray, "n_pos"]
    elbow_idx: int
    device: str = Field(default_factory=default_device)

    def sample_cluster(
        self, cluster_id: int, max_n: int = 5, attr_fn: AttrFn = compute_attnlrp_relevance
    ) -> list[ClusterHit]:
        idxs = np.argwhere(self.labels == cluster_id).reshape(-1)
        idxs = np.random.permutation(idxs)[:max_n]
        model = self.model.to(self.device)
        results: list[ClusterHit] = []
        for i in tqdm(idxs):
            bid, tid = self.batch_idx[i], self.token_idx[i]
            sample_id = self.sample_ids[bid]
            singleton_batch = self.source.to_model_batch([sample_id]).to(self.device)
            relevance = attr_fn(model, singleton_batch, self.layer_getter, self.neuron_idx, tid)
            results.append(
                ClusterHit(
                    sample_id=sample_id,
                    token_idx=int(tid),
                    relevance=relevance,
                    model_input=singleton_batch.grad_leaf.detach().cpu(),
                    hadamard=self.hdmds[i],
                    display_ids=None if singleton_batch.display_ids is None else singleton_batch.display_ids[0].cpu(),
                )
            )
            del singleton_batch
            empty_cache()
            gc.collect()
        return results

    def _cluster_mean_hadamard(self, cluster_id: int) -> torch.Tensor:
        """Mean Hadamard product over every hit HDBSCAN placed in this cluster (not just the
        handful sampled for display) - a per-cluster prototype other data can later be matched
        against via cosine similarity."""
        cluster_hdmds = self.hdmds[self.labels == cluster_id]
        return torch.from_numpy(cluster_hdmds.mean(axis=0))

    def _cluster_section(
        self,
        cluster_id: int,
        count: int,
        n_unique_samples: int,
        report_dir: Path,
        attr_fn: AttrFn,
        max_n: int,
    ) -> str:
        """Html for one cluster's section of the report; a `cluster_{id}/` subfolder holding
        its artifacts and mean Hadamard prototype (`mean.pt`) is written as a side effect."""
        cluster_dir = report_dir / f"cluster_{cluster_id}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._cluster_mean_hadamard(cluster_id), cluster_dir / "mean.pt")

        header = f"<h2>Cluster {cluster_id} (n={count}, unique_samples={n_unique_samples})</h2>"
        hits = self.sample_cluster(cluster_id, max_n=max_n, attr_fn=attr_fn)
        body = self.presenter.render(hits, cluster_dir) if hits else "<p>no hits sampled.</p>"
        return f'<section class="cluster">\n{header}\n{body}\n</section>'

    def _write_meta(
        self,
        report_dir: Path,
        cluster_ids: list[int],
        counts: dict[int, int],
        uniq_images: dict[int, int],
        min_uniq_images: int,
    ) -> None:
        meta = {
            "neuron_idx": self.neuron_idx,
            "threshold": self.threshold,
            "n_hits": len(self.labels),
            "min_uniq_images_per_cluster": min_uniq_images,
            "n_clusters": len(cluster_ids),
            "clusters": {
                str(cid): {"n_hits": counts[cid], "n_unique_images": uniq_images[cid]} for cid in cluster_ids
            },
            "sample_ids": list(self.sample_ids),
        }
        (report_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    def write_report(
        self,
        out_dir: str | Path,
        name: str | None = None,
        attr_fn: AttrFn = compute_attnlrp_relevance,
        max_n: int = 5,
        min_uniq_images: int = 2,
    ) -> Path:
        """Self-contained report folder for this neuron: `out_dir/{name}/index.html`, every
        cluster largest first (each in its own `cluster_{id}/` subfolder), the elbow plot and
        threshold metadata that picked this neuron's activation cutoff, all sitting under
        report_dir - so the whole folder can be zipped/tarred and read standalone.

        Clusters drawn from fewer than `min_uniq_images` distinct source images are dropped -
        see cluster_unique_image_counts for why.

        `name` defaults to `neuron_{idx}` but can be overridden.
        """
        report_dir = Path(out_dir) / (name or f"neuron_{self.neuron_idx}")
        report_dir.mkdir(parents=True, exist_ok=True)

        save_elbow_plot(self.elbow_values, self.elbow_idx, report_dir / "elbow.png")

        counts = cluster_counts(self.labels)
        uniq_images = cluster_unique_image_counts(self.labels, self.batch_idx)
        cluster_ids = [
            cid for cid in clusters_by_frequency(self.labels) if uniq_images[cid] >= min_uniq_images
        ]
        self._write_meta(report_dir, cluster_ids, counts, uniq_images, min_uniq_images)

        sections = [
            self._cluster_section(cid, counts[cid], uniq_images[cid], report_dir, attr_fn, max_n)
            for cid in cluster_ids
        ]

        report_path = report_dir / "index.html"
        title = f"Neuron {self.neuron_idx}"
        report_path.write_text(
            page(
                title,
                f"<h1>{title}</h1>\n"
                f'<p class="meta">threshold: {self.threshold:.4f} &middot; '
                f"{len(self.labels)} hits &middot; {len(sections)} clusters</p>\n"
                '<img src="elbow.png" alt="elbow plot">\n' + "\n".join(sections),
            )
        )
        return report_path
