"""Per-cluster inspection: sample hits from one HDBSCAN cluster of a neuron's Hadamard
products, run pixel-space attribution on each, and render an overlay grid alongside the raw
Hadamard patterns that drove the clustering.

Sampling and rendering both need the same handful of arrays/handles (model, processor,
dcm_paths, layer/neuron/token/batch indices, cluster labels, hadamard products) - threading
all of that through free functions individually got unwieldy, hence one class holding it all.
"""

import gc
import json
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float, Int
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm
from transformers.image_processing_utils import BaseImageProcessor
from transformers.models.dinov2.modeling_dinov2 import Dinov2Model

from .attribution import compute_attnlrp_relevance
from .device import default_device, empty_cache
from .dicom import get_batch, inv_tfm
from .layers import LayerGetter
from .viz import (
    mk_overlay,
    render_hadamard_tiles,
    save_elbow_plot,
    save_image_grid,
    show_grid,
    show_single_channel_red_green_black,
    to_pil,
)

# Attribution methods disagree on return type (captum's neuron_ig returns an undetached
# tensor, compute_attnlrp_relevance an already-detached numpy array) - both are accepted here
# and normalized in sample_cluster.
AttrFn = Callable[
    [Dinov2Model, torch.Tensor, LayerGetter, int, int | None], torch.Tensor | np.ndarray
]


class ClusterHit(NamedTuple):
    image: np.ndarray
    attribution: torch.Tensor
    hadamard: np.ndarray


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

    model: Dinov2Model
    processor: BaseImageProcessor
    dcm_paths: list[Path]
    layer_getter: LayerGetter
    neuron_idx: int
    token_idx: Int[np.ndarray, "n_hits"]
    batch_idx: Int[np.ndarray, "n_hits"]
    labels: Int[np.ndarray, "n_hits"]
    hdmds: Float[np.ndarray, "n_hits hidden"]
    act_shape: tuple[int, int]
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
            batch = get_batch(self.processor, [self.dcm_paths[bid]])
            batch = batch.to(self.device)
            attr = attr_fn(model, batch["pixel_values"], self.layer_getter, self.neuron_idx, tid)
            batch = batch.to("cpu")
            inv_img = inv_tfm(self.processor, batch["pixel_values"])
            attr = attr[0]
            # normalize to a cpu tensor - see the AttrFn comment above for why this varies
            attr = attr.detach().cpu() if isinstance(attr, torch.Tensor) else torch.from_numpy(attr)
            results.append(ClusterHit(image=inv_img, attribution=attr, hadamard=self.hdmds[i]))
            del batch, attr
            empty_cache()
            gc.collect()
        return results

    def show_cluster(
        self, cluster_id: int, max_n: int = 5, attr_fn: AttrFn = compute_attnlrp_relevance, alpha: float = 0.7
    ) -> None:
        """Plots overlays and raw Hadamard products side by side, as a sanity check that the
        clustering is grouping visually-similar activations."""
        hits = self.sample_cluster(cluster_id, max_n=max_n, attr_fn=attr_fn)
        if not hits:
            print(f"no hits for cluster {cluster_id}")
            return

        overlays = [np.asarray(overlay) for overlay in self._mk_overlays(hits, alpha)]
        sampled_hdmds = self._hadamard_patches(hits)

        show_grid(overlays, 1, len(hits))
        plt.show()

        show_single_channel_red_green_black(sampled_hdmds, (5 * len(hits), 5), len(hits))
        plt.show()

    def _mk_overlays(self, hits: list[ClusterHit], alpha: float) -> list[Image.Image]:
        return [mk_overlay(hit.image, hit.attribution, (500, 500), alpha=alpha) for hit in hits]

    def _mk_originals(self, hits: list[ClusterHit]) -> list[Image.Image]:
        """The denormalized model input (inv_img), not the raw DICOM - what the overlay was
        blended onto, so the two line up pixel-for-pixel for comparison."""
        return [to_pil(hit.image, (500, 500)) for hit in hits]

    def _hadamard_patches(self, hits: list[ClusterHit]) -> list[np.ndarray]:
        return [hit.hadamard.reshape(self.act_shape) for hit in hits]

    def _cluster_mean_hadamard(self, cluster_id: int) -> torch.Tensor:
        """Mean Hadamard product over every hit HDBSCAN placed in this cluster (not just the
        handful sampled for display) - a per-cluster prototype other data can later be matched
        against via cosine similarity."""
        cluster_hdmds = self.hdmds[self.labels == cluster_id]
        return torch.from_numpy(cluster_hdmds.mean(axis=0))

    def _save_cluster_images(
        self, hits: list[ClusterHit], cluster_dir: Path, alpha: float
    ) -> tuple[Path, Path, Path]:
        original_path = cluster_dir / "original.jpg"
        overlay_path = cluster_dir / "overlay.jpg"
        hadamard_path = cluster_dir / "hadamard.jpg"
        save_image_grid(self._mk_originals(hits), original_path)
        save_image_grid(self._mk_overlays(hits, alpha), overlay_path)
        save_image_grid(render_hadamard_tiles(self._hadamard_patches(hits)), hadamard_path)
        return original_path, overlay_path, hadamard_path

    def _cluster_section(
        self,
        cluster_id: int,
        count: int,
        n_unique_images: int,
        report_dir: Path,
        attr_fn: AttrFn,
        max_n: int,
        alpha: float,
    ) -> str:
        """Markdown for one cluster's section of the report; a `cluster_{id}/` subfolder holding
        its images and mean Hadamard prototype (`mean.pt`) is written as a side effect."""
        cluster_dir = report_dir / f"cluster_{cluster_id}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._cluster_mean_hadamard(cluster_id), cluster_dir / "mean.pt")

        header = f"## Cluster {cluster_id} (n={count}, unique_images={n_unique_images})"
        hits = self.sample_cluster(cluster_id, max_n=max_n, attr_fn=attr_fn)
        if not hits:
            return f"{header}\n\nno hits sampled.\n"

        original_path, overlay_path, hadamard_path = self._save_cluster_images(hits, cluster_dir, alpha)
        return (
            f"{header}\n\n"
            f"![original]({cluster_dir.name}/{original_path.name})\n\n"
            f"![overlay]({cluster_dir.name}/{overlay_path.name})\n\n"
            f"<details>\n<summary>hadamard patterns</summary>\n\n"
            f"![hadamard]({cluster_dir.name}/{hadamard_path.name})\n\n"
            f"</details>\n"
        )

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
            "n_hits": int(len(self.labels)),
            "min_uniq_images_per_cluster": min_uniq_images,
            "n_clusters": len(cluster_ids),
            "clusters": {
                str(cid): {"n_hits": counts[cid], "n_unique_images": uniq_images[cid]} for cid in cluster_ids
            },
            "dcm_paths": [str(p) for p in self.dcm_paths],
        }
        (report_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    def write_report(
        self,
        out_dir: str | Path,
        name: str | None = None,
        attr_fn: AttrFn = compute_attnlrp_relevance,
        max_n: int = 5,
        alpha: float = 0.7,
        min_uniq_images: int = 2,
    ) -> Path:
        """Self-contained report folder for this neuron: `out_dir/{name}/index.md`, every
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
            self._cluster_section(cid, counts[cid], uniq_images[cid], report_dir, attr_fn, max_n, alpha)
            for cid in cluster_ids
        ]

        report_path = report_dir / "index.md"
        report_path.write_text(
            f"# Neuron {self.neuron_idx}\n\n"
            f"threshold: {self.threshold:.4f}\n\n"
            f"![elbow](elbow.png)\n\n" + "\n\n".join(sections)
        )
        return report_path
