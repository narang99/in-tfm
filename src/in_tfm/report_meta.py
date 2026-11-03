"""What a neuron report's `meta.json` holds."""

import json
import math
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .presenters import HadamardShape
from .run_config import Tail
from .sources import SampleId

META_NAME = "meta.json"


class ClusterMeta(BaseModel):
    n_hits: int
    n_unique_images: int


class ElbowPickMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["elbow"] = "elbow"
    tail: Tail = "positive"
    cutoff: float
    """Signed, so it is negative for the negative tail."""


class BucketedPickMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["bucketed"] = "bucketed"
    tail: Tail = "positive"
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

    version: Literal[3] = 3
    neuron_idx: int
    picks: list[PickMeta]
    """One per tail the hits came from, negative first."""
    n_hits: int
    min_uniq_images_per_cluster: int
    hadamard_shape: HadamardShape
    """Cached with the hits so a re-render can reshape their Hadamard vectors without the model."""
    clusters: dict[int, ClusterMeta]
    """Largest first, the order sections appear in the report."""
    sample_ids: list[SampleId]

    @property
    def thresholds(self) -> list[float]:
        return [threshold_of(pick) for pick in self.picks]


def threshold_of(pick: PickMeta) -> float:
    match pick:
        case ElbowPickMeta(cutoff=cutoff):
            return cutoff
        case BucketedPickMeta(floor=floor):
            return floor


def picks_from_version_2(pick: dict[str, Any]) -> list[dict[str, Any]]:
    """Version 2 held one pick, signed like the hits, so its sign gives the tail.

    - A version 2 signed report held a magnitude floor, so it reads as the positive tail."""
    value = pick["cutoff"] if pick["method"] == "elbow" else pick["floor"]
    tail = "negative" if math.copysign(1.0, value) < 0 else "positive"
    return [{**pick, "tail": tail}]


def load_report_meta(report_dir: Path) -> ReportMeta:
    raw = json.loads((report_dir / META_NAME).read_text())
    if raw.get("version") == 2:
        pick = raw.pop("pick")
        raw = {**raw, "version": 3, "picks": picks_from_version_2(pick)}
    return ReportMeta.model_validate(raw)


def write_report_meta(report_dir: Path, meta: ReportMeta) -> None:
    (report_dir / META_NAME).write_text(meta.model_dump_json(indent=2))
