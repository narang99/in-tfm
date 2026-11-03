"""What a neuron report's `meta.json` holds."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

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

    @property
    def threshold(self) -> float:
        match self.pick:
            case ElbowPickMeta(cutoff=cutoff):
                return cutoff
            case BucketedPickMeta(floor=floor):
                return floor


def load_report_meta(report_dir: Path) -> ReportMeta:
    return ReportMeta.model_validate_json((report_dir / META_NAME).read_text())


def write_report_meta(report_dir: Path, meta: ReportMeta) -> None:
    (report_dir / META_NAME).write_text(meta.model_dump_json(indent=2))
