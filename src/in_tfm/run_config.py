"""Everything a neuron-report run can be told, as one typed config.

Values come from two places, and the command line wins:

- a YAML file given with `--config`, for the settings that distinguish one experiment from
  another, so they live in git
- flags, for one-off changes on top of it

Nested sections use dotted flags, e.g. `--clustering.min-cluster-size 20`.
Overriding one key of a section keeps the file's other keys for that section.
"""

import argparse
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from .device import default_device


class ModelName(StrEnum):
    """HF models keep their HF id. Models we host ourselves are named `self/...`."""

    GEMMA3_270M = "google/gemma-3-270m"
    ATTN_ONLY_2L = "self/attn-only-2l"
    RAD_DINO = "microsoft/rad-dino"
    STL_INCEPTION = "self/stl-inception"


Modality = Literal["text", "image", "stl"]

MODALITY: dict[ModelName, Modality] = {
    ModelName.GEMMA3_270M: "text",
    ModelName.ATTN_ONLY_2L: "text",
    ModelName.RAD_DINO: "image",
    ModelName.STL_INCEPTION: "stl",
}


class Target(StrEnum):
    DOWN_PROJ = "down_proj"
    Q_PROJ = "q_proj"
    K_PROJ = "k_proj"
    FC2 = "fc2"
    CONV = "conv"


SUPPORTED_TARGETS: dict[ModelName, set[Target]] = {
    ModelName.GEMMA3_270M: {Target.DOWN_PROJ, Target.Q_PROJ, Target.K_PROJ},
    ModelName.ATTN_ONLY_2L: {Target.Q_PROJ, Target.K_PROJ},
    ModelName.RAD_DINO: {Target.FC2},
    ModelName.STL_INCEPTION: {Target.CONV},
}
"""The attention-only model has no MLP, so it has no `down_proj`. Dinov2 calls its MLP output `fc2`."""

Polarity = Literal["positive", "negative"]


class Section(BaseModel):
    """A typo in a config file should fail, not be ignored."""

    model_config = ConfigDict(extra="forbid")


class TextDataConfig(Section):
    dataset: str = Field("Salesforce/wikitext", description="hf dataset name")
    dataset_config: str = "wikitext-2-raw-v1"
    split: str = "train"
    n_samples: int = 64
    max_length: int = 128
    unit: Literal["paragraph", "article"] = Field(
        "paragraph",
        description="article joins a wikitext article's paragraphs, so a long max_length is mostly real tokens",
    )
    min_article_chars: int = 0


class ImageDataConfig(Section):
    dcm_dir: Path = Path("dicoms")
    n_dicoms: int = Field(1000, description="the first n in sorted file order, all of them if there are fewer")


class StlDataConfig(Section):
    root: Path = Path("data/stl10")
    split: Literal["train", "test"] = "test"
    n_images: int = Field(1000, description="the first n images of the split in dataset order")
    checkpoint: Path | None = Field(None, description="a snapshot from scripts/train_stl.py, random weights if unset")


class HitSelectionConfig(Section):
    method: Literal["elbow", "bucketed"] = Field(
        "elbow",
        description="elbow: everything above the elbow, uniformly subsampled to max_hits. "
        "bucketed: equal-width activation buckets, an equal share of max_hits from each",
    )
    max_hits: int = Field(20000, description="hits kept per neuron before clustering")
    n_buckets: int = 10
    min_bucket_size: int = Field(
        100, description="bucketed only: buckets with fewer positions than this are merged with their neighbour"
    )
    bucket_floor_fraction: float = Field(
        0.0, description="bucketed only: lowest bucket starts at this fraction of the max activation"
    )


class ClusteringConfig(Section):
    on: Literal["hadamard", "input"] = Field(
        "hadamard",
        description="hadamard: input * weight row. input: the raw input at each hit, "
        "L2-normalised the same way, an ablation of the weight row",
    )
    min_cluster_size: int = 10
    min_samples: int | None = Field(None, description="HDBSCAN min_samples, defaults to min_cluster_size")
    selection_method: Literal["eom", "leaf"] = "leaf"


class ReportConfig(Section):
    max_hits_per_cluster: int = 10
    min_uniq_samples_per_cluster: int = 2


class RunConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True)

    config: Path | None = Field(None, description="YAML file with defaults for any of the settings below")
    model: ModelName = Field(ModelName.GEMMA3_270M, description="google/gemma-3-270m or self/attn-only-2l")
    target: Target = Field(Target.DOWN_PROJ, description="down_proj: MLP neurons. q_proj / k_proj: neuron index = (kv_)head * head_dim + dim")
    layer_idx: int = 10
    layer_name: str | None = Field(
        None, description="conv only: dotted submodule path, for example block_b.branch_3x3.1.0. Used instead of layer_idx"
    )
    neurons: list[int] | None = Field(None, description="neuron indices, repeat the flag per index. Overrides the range")
    neuron_start: int = 90
    neuron_end: int = Field(90, description="inclusive")
    polarity: Literal["positive", "negative", "both"] = Field(
        "positive", description="which tail of the activation distribution counts as a hit"
    )
    batch_size: int = 8
    seed: int = Field(0, description="corpus shuffle and hit subsampling")
    out_dir: Path = Path("reports")
    device: str = Field(default_factory=default_device)
    text: TextDataConfig = TextDataConfig()
    image: ImageDataConfig = ImageDataConfig()
    stl: StlDataConfig = StlDataConfig()
    hits: HitSelectionConfig = HitSelectionConfig()
    clustering: ClusteringConfig = ClusteringConfig()
    report: ReportConfig = ReportConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Without this, an environment variable named `DEVICE` or `MODEL` would silently
        override a field. The config file arrives as init values."""
        return (init_settings,)

    @model_validator(mode="after")
    def target_exists_on_model(self) -> "RunConfig":
        if self.target not in SUPPORTED_TARGETS[self.model]:
            options = sorted(t.value for t in SUPPORTED_TARGETS[self.model])
            raise ValueError(f"{self.model.value} has no {self.target.value}, choose one of {options}")
        return self

    @model_validator(mode="after")
    def conv_target_names_its_layer(self) -> "RunConfig":
        if self.target is Target.CONV and self.layer_name is None:
            raise ValueError("the conv target needs layer_name, a dotted submodule path")
        return self

    @model_validator(mode="after")
    def only_the_models_data_section_is_set(self) -> "RunConfig":
        """`text.*` flags with an image model are a mistake, not something to ignore."""
        others = {"text", "image", "stl"} - {self.modality}
        if wrong := others & self.model_fields_set:
            raise ValueError(f"{self.model.value} reads {self.modality} data, but the {sorted(wrong)[0]} section was set")
        return self

    @property
    def modality(self) -> Modality:
        return MODALITY[self.model]

    @property
    def neuron_idxs(self) -> list[int]:
        return self.neurons or list(range(self.neuron_start, self.neuron_end + 1))


def config_path_from(argv: Sequence[str]) -> Path | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path)
    known, _ = parser.parse_known_args(argv)
    return known.config


def load_run_config(argv: Sequence[str]) -> RunConfig:
    """The file is read first and handed over as init values, with the flags layered on top.
    `RunConfig` itself cannot read the path from the flags it is parsing."""
    path = config_path_from(argv)
    values = yaml.safe_load(path.read_text()) if path else {}
    return RunConfig(_cli_parse_args=list(argv), **(values or {}))
