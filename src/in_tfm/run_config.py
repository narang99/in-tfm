"""Everything a neuron-report run can be told, as one typed config.

Values come from two places, and the command line wins:

- a YAML file given with `--config`, for the settings that distinguish one experiment from
  another, so they live in git
- flags, for one-off changes on top of it

Nested sections use dotted flags, e.g. `--clustering.strategy.min-cluster-size 20`.
Overriding one key of a section keeps the file's other keys for that section.
"""

import argparse
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

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
    MNIST_CNN = "self/mnist-cnn"
    MNIST_CNN_SMALL = "self/mnist-cnn-small"
    INCEPTION_V1 = "lucent/inception-v1"


Modality = Literal["text", "image", "stl", "mnist", "inception"]

MODALITY: dict[ModelName, Modality] = {
    ModelName.GEMMA3_270M: "text",
    ModelName.ATTN_ONLY_2L: "text",
    ModelName.RAD_DINO: "image",
    ModelName.STL_INCEPTION: "stl",
    ModelName.MNIST_CNN: "mnist",
    ModelName.MNIST_CNN_SMALL: "mnist",
    ModelName.INCEPTION_V1: "inception",
}


class Target(StrEnum):
    DOWN_PROJ = "down_proj"
    Q_PROJ = "q_proj"
    K_PROJ = "k_proj"
    FC2 = "fc2"
    CONV = "conv"
    LINEAR = "linear"


SUPPORTED_TARGETS: dict[ModelName, set[Target]] = {
    ModelName.GEMMA3_270M: {Target.DOWN_PROJ, Target.Q_PROJ, Target.K_PROJ},
    ModelName.ATTN_ONLY_2L: {Target.Q_PROJ, Target.K_PROJ},
    ModelName.RAD_DINO: {Target.FC2},
    ModelName.STL_INCEPTION: {Target.CONV},
    ModelName.MNIST_CNN: {Target.CONV, Target.LINEAR},
    ModelName.MNIST_CNN_SMALL: {Target.CONV, Target.LINEAR},
    ModelName.INCEPTION_V1: {Target.CONV},
}
"""The attention-only model has no MLP, so it has no `down_proj`. Dinov2 calls its MLP output `fc2`."""

Polarity = Literal["positive", "negative", "signed"]
"""`signed` takes hits from both tails into one space."""


class Section(BaseModel):
    """A typo in a config file should fail, not be ignored."""

    model_config = ConfigDict(extra="forbid")


class TextDataConfig(Section):
    dataset: str = Field("Salesforce/wikitext", description="hf dataset name")
    dataset_config: str | None = Field("wikitext-2-raw-v1", description="none for a dataset without configs")
    split: str = "train"
    n_samples: int = 64
    max_length: int = 128
    unit: Literal["paragraph", "article", "document"] = Field(
        "paragraph",
        description="article joins a wikitext article's paragraphs, so a long max_length is mostly real tokens. "
        "document takes each row as it is, for corpora like NeelNanda/pile-10k",
    )
    min_article_chars: int = 0


class ImageDataConfig(Section):
    dcm_dir: Path = Path("dicoms")
    n_dicoms: int = Field(1000, description="the first n in sorted file order, all of them if there are fewer")


class StlDataConfig(Section):
    dataset: str = Field("tanganke/stl10", description="hf dataset with `image` and `label` columns, 96x96")
    split: Literal["train", "test"] = "test"
    n_images: int = Field(1000, description="the first n images of the split in dataset order")
    checkpoint: Path | None = Field(None, description="a snapshot from scripts/train_stl.py, random weights if unset")
    attribution: Literal["deeplift", "integrated-gradients"] = Field(
        "deeplift", description="how a hit is explained in pixels, integrated-gradients is the slow reference"
    )
    ig_steps: int = Field(128, description="integrated-gradients only: interpolation steps per explained hit")


class MnistDataConfig(Section):
    dataset: str = Field("ylecun/mnist", description="hf dataset with a grayscale `image` and a `label` column, 28x28")
    split: Literal["train", "test"] = "test"
    images_per_class: int = Field(100, description="the first n images of each digit in the split")
    checkpoint: Path | None = Field(None, description="a snapshot from scripts/train_mnist.py, the model's own if unset")


class InceptionDataConfig(Section):
    image_dir: Path = Field(
        Path("data/inception_pool"), description="one folder per class holding jpgs, see scripts/prepare_inception_images.py"
    )
    n_images: int | None = Field(
        None, description="evenly spread over the sorted images, so every class is in it. All of them if unset"
    )


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
        0.0,
        description="bucketed only: lowest bucket starts at this fraction of the max activation. "
        "With signed polarity, hits need a magnitude above this fraction of the largest magnitude",
    )


class HdbscanConfig(Section):
    """Dense regions split by sparse gaps. Hits in no dense region are noise."""

    method: Literal["hdbscan"] = "hdbscan"
    min_cluster_size: int = 10
    min_samples: int | None = Field(None, description="defaults to min_cluster_size")
    selection_method: Literal["eom", "leaf"] = "leaf"
    soft: bool = Field(
        False,
        description="each hit gets a probability per cluster, as the hdbscan package's soft clustering gives it. "
        "Noise stays noise",
    )
    max_entropy: float = Field(
        0.5, gt=0, description="soft only: bits. Hits this mixed or more are left out of the cluster report",
    )


class DensityPeaksConfig(Section):
    """Each hit links to its nearest denser hit, and every hit lands in the tree of a root. See in_tfm.density_peaks."""

    method: Literal["density_peaks"] = "density_peaks"
    k: int = Field(5, description="a hit's density is 1 / the distance to its kth neighbour")
    peak_share: float = Field(
        0.8, gt=0, le=1,
        description="a tree's peak is its hits at least this dense relative to its root. Each hit's probability per "
        "tree comes from its geodesic distance to that tree's peak",
    )
    max_entropy: float = Field(
        0.5, gt=0,
        description="bits. Hits whose probabilities over trees are this mixed or more are left out of the cluster report",
    )


ClusteringStrategy = Annotated[HdbscanConfig | DensityPeaksConfig, Field(discriminator="method")]

MOVED_TO_STRATEGY = {
    "method": "method",
    "min_cluster_size": "hdbscan",
    "min_samples": "hdbscan",
    "selection_method": "hdbscan",
    "k": "density_peaks",
}
"""Keys that used to sit directly under `clustering`, and the strategy that now holds each."""


class ClusteringConfig(Section):
    """Settings every strategy shares sit here. Each strategy's own settings sit under `strategy`, picked by `method`."""

    on: Literal["hadamard", "input"] = Field(
        "hadamard",
        description="hadamard: input * weight row. input: the raw input at each hit, "
        "L2-normalised the same way, an ablation of the weight row",
    )
    distance: Literal["euclidean", "weighted_jaccard"] = Field(
        "euclidean",
        description="euclidean: between the L2-normalised vectors. weighted_jaccard: 1 - sum(min) / sum(max) between "
        "each hit's share of contribution per dim, with positive and negative parts kept apart",
    )
    normalise: bool = Field(
        True,
        description="false: no L2 on the clustered rows, and no L1 or L2 inside the distance, so a hit's size counts. "
        "The Hadamard tiles in the reports are then raw too",
    )
    strategy: ClusteringStrategy = Field(
        HdbscanConfig(), description="hdbscan or density_peaks, picked by strategy.method"
    )

    @model_validator(mode="before")
    @classmethod
    def strategy_keys_are_under_strategy(cls, data: Any) -> Any:
        """Old files put these keys directly under `clustering`. Say where each went, rather than only that it is unknown."""
        if isinstance(data, dict) and (moved := sorted(MOVED_TO_STRATEGY.keys() & data.keys())):
            key = moved[0]
            where = "strategy.method" if key == "method" else f"strategy.{key}, with strategy.method: {MOVED_TO_STRATEGY[key]}"
            raise ValueError(f"clustering.{key} moved to clustering.{where}")
        return data

    @model_validator(mode="before")
    @classmethod
    def strategy_without_a_method_is_hdbscan(cls, data: Any) -> Any:
        """HDBSCAN is the default, so tuning it should not need its method too, e.g. `--clustering.strategy.min-cluster-size 20`."""
        strategy = data.get("strategy") if isinstance(data, dict) else None
        if isinstance(strategy, dict) and "method" not in strategy:
            return {**data, "strategy": {"method": "hdbscan", **strategy}}
        return data


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
        None, description="conv and linear only: dotted submodule path, for example block_b.branch_3x3.1.0. Used instead of layer_idx"
    )
    neurons: list[int] | None = Field(None, description="neuron indices, repeat the flag per index. Overrides the range")
    neuron_start: int = 90
    neuron_end: int = Field(90, description="inclusive")
    polarity: Literal["positive", "negative", "both", "signed"] = Field(
        "positive",
        description="which tail of the activation distribution counts as a hit. "
        "both: each tail in its own space. signed: both tails sampled and clustered together, bucketed only",
    )
    batch_size: int = 8
    seed: int = Field(0, description="corpus shuffle and hit subsampling")
    out_dir: Path = Path("reports")
    device: str = Field(default_factory=default_device)
    text: TextDataConfig = TextDataConfig()
    image: ImageDataConfig = ImageDataConfig()
    stl: StlDataConfig = StlDataConfig()
    mnist: MnistDataConfig = MnistDataConfig()
    inception: InceptionDataConfig = InceptionDataConfig()
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
        if self.target in (Target.CONV, Target.LINEAR) and self.layer_name is None:
            raise ValueError(f"the {self.target.value} target needs layer_name, a dotted submodule path")
        return self

    @model_validator(mode="after")
    def only_the_models_data_section_is_set(self) -> "RunConfig":
        """`text.*` flags with an image model are a mistake, not something to ignore."""
        others = {"text", "image", "stl", "mnist", "inception"} - {self.modality}
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
