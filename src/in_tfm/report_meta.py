"""What a neuron report's `meta.json` holds, and how older files are read.

- `ReportMeta` is the current structure.
  - It forbids unknown keys, so an old file never half-parses as the new structure.
- `LegacyReportMeta` is the structure before the hit selection method was stored explicitly.
  - It is kept only so `load_report_meta` can name the problem and `ReportMeta.migrate_file` can upgrade it.
"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .presenters import HadamardShape
from .sources import SampleId

META_NAME = "meta.json"


class ClusterMeta(BaseModel):
    n_hits: int
    n_unique_images: int


class ElbowPickMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["elbow"] = "elbow"
    cutoff: float
    """Signed, so it is negative for the negative tail."""


class BucketedPickMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["bucketed"] = "bucketed"
    floor: float
    """Signed, like the elbow cutoff."""
    bucket_edges: list[float]
    """Lower edge of every merged bucket, then the max.

    - These are magnitudes, like the curve the bucket plot draws, so they stay positive for the negative tail.
    """


PickMeta = Annotated[ElbowPickMeta | BucketedPickMeta, Field(discriminator="method")]


class KMeansSweep(BaseModel):
    ks: list[int]
    inertia: list[float]
    silhouette: dict[int, float]
    """Only for k of at least 2, where silhouette is defined."""


class KMeansGroupingMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["kmeans"] = "kmeans"
    k: int
    sweep: KMeansSweep | None
    """None when k was given rather than picked from the sweep's elbow."""
    members: list[list[int]]
    """Cluster ids of each group, largest group first."""


class ThresholdGroupingMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["thresholds"] = "thresholds"
    min_cosine: float
    max_median_gap: float
    members: list[list[int]]
    """Cluster ids of each group, largest group first."""


GroupingMeta = Annotated[KMeansGroupingMeta | ThresholdGroupingMeta, Field(discriminator="method")]


class LegacyReportMeta(BaseModel):
    """`threshold` was the elbow cutoff or the bucket floor, and `bucket_edges` being set was the only sign of which."""

    neuron_idx: int
    threshold: float
    bucket_edges: list[float] | None = None
    n_hits: int
    min_uniq_images_per_cluster: int
    n_clusters: int
    hadamard_shape: HadamardShape
    clusters: dict[int, ClusterMeta]
    sample_ids: list[SampleId]


class ReportMeta(BaseModel):
    """Everything `render_report` needs besides the cached hits."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[2] = 2
    neuron_idx: int
    pick: PickMeta
    n_hits: int
    min_uniq_images_per_cluster: int
    hadamard_shape: HadamardShape
    """Cached with the hits so a re-render can reshape their Hadamard vectors without the model."""
    clusters: dict[int, ClusterMeta]
    """Largest first, the order sections appear in the report."""
    sample_ids: list[SampleId]
    groupings: dict[str, GroupingMeta] = {}
    """Named so two ways of grouping the same clusters can be compared. Reports written before groupings have none."""
    default_grouping: str | None = None
    """The grouping the page shows, None for one group per cluster."""

    @property
    def threshold(self) -> float:
        match self.pick:
            case ElbowPickMeta(cutoff=cutoff):
                return cutoff
            case BucketedPickMeta(floor=floor):
                return floor

    @classmethod
    def from_legacy(cls, old: LegacyReportMeta) -> "ReportMeta":
        return cls(
            neuron_idx=old.neuron_idx,
            pick=legacy_pick(old),
            n_hits=old.n_hits,
            min_uniq_images_per_cluster=old.min_uniq_images_per_cluster,
            hadamard_shape=old.hadamard_shape,
            clusters=old.clusters,
            sample_ids=old.sample_ids,
        )

    @classmethod
    def migrate_file(cls, meta_path: Path) -> "ReportMeta":
        """Rewrites an old `meta.json` in place. A file already in the current structure is left untouched."""
        text = meta_path.read_text()
        try:
            return cls.model_validate_json(text)
        except ValidationError:
            meta = cls.from_legacy(LegacyReportMeta.model_validate_json(text))
        meta_path.write_text(meta.model_dump_json(indent=2))
        return meta


def legacy_pick(old: LegacyReportMeta) -> ElbowPickMeta | BucketedPickMeta:
    if old.bucket_edges is None:
        return ElbowPickMeta(cutoff=old.threshold)
    return BucketedPickMeta(floor=old.threshold, bucket_edges=old.bucket_edges)


class LegacyReportMetaError(ValueError):
    pass


def load_report_meta(report_dir: Path) -> ReportMeta:
    meta_path = report_dir / META_NAME
    text = meta_path.read_text()
    try:
        return ReportMeta.model_validate_json(text)
    except ValidationError as current_error:
        if not parses_as_legacy(text):
            raise
        raise LegacyReportMetaError(
            f"{meta_path} uses the old meta.json structure, from before the hit selection method was stored. "
            f"Upgrade it with `ReportMeta.migrate_file(Path({str(meta_path)!r}))`, "
            f"or every report under a folder with `scripts/migrate_report_meta.py <folder>`."
        ) from current_error


def parses_as_legacy(text: str) -> bool:
    try:
        LegacyReportMeta.model_validate_json(text)
    except ValidationError:
        return False
    return True


def write_report_meta(report_dir: Path, meta: ReportMeta) -> None:
    (report_dir / META_NAME).write_text(meta.model_dump_json(indent=2))
