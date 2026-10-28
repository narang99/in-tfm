"""The capture, select, cluster and report sequence behind every neuron-report script.

Scripts only decide what the samples are and which adapter loads the model. The model is
patched for AttnLRP once, before NNsight wraps it, since the patch mutates process-wide state
and NNsight stores each module's forward at wrap time (see ModelAdapter.patch_for_attn_lrp).
"""

import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from nnsight import NNsight
from pydantic import BaseModel, ConfigDict
from sklearn.preprocessing import normalize

from .activation_ranges import sample_noise
from .cluster_membership import (
    NOISE_LABEL, ClusterProbabilities, HitMembership, confident_labels, membership_entropy, most_probable_labels,
    one_hot_memberships, probability_matrix,
)
from .clustering import cluster_labels
from .density_peaks import (
    DensityPeakForest, ForestTrees, calculate_probs, comparable_rows, density_peak_forest, forest_trees, torch_forest,
)
from .device import empty_cache
from .hadamard import near_square_shape, raw_hadamard_from_rows
from .hit_selection import HitSelection, select_hits
from .layers import LayerGetter, conv_getter, down_proj_getter, fc2_getter, k_proj_getter, q_proj_getter
from .border_pages import write_border_pages
from .mixed_hits_report import write_mixed_hits_page
from .models import ModelAdapter
from .neuron_capture import NeuronCapture, merge_positions
from .neuron_report import AttrFn, NeuronClusterHits
from .presenters import ClusterPresenter
from .report_page import ForestData, forest_data
from .run_config import ClusteringConfig, DensityPeaksConfig, HdbscanConfig, Polarity, RunConfig, Target
from .sources import SampleSource
from .token_layout import ConvLayout, IdentityLayout, TokenLayout


@contextmanager
def timed(label: str):
    start = time.perf_counter()
    yield
    print(f"[{label}] {time.perf_counter() - start:.2f}s")


def layer_getter_for(config: RunConfig) -> LayerGetter:
    getters = {
        Target.DOWN_PROJ: down_proj_getter,
        Target.Q_PROJ: q_proj_getter,
        Target.K_PROJ: k_proj_getter,
        Target.FC2: fc2_getter,
    }
    if config.target in (Target.CONV, Target.LINEAR):
        return conv_getter(config.layer_name)
    return getters[config.target](config.layer_idx)


def layout_for(layer: torch.nn.Module) -> TokenLayout:
    return ConvLayout(layer) if isinstance(layer, torch.nn.Conv2d) else IdentityLayout()


def polarities_for(config: RunConfig) -> list[Polarity]:
    return ["positive", "negative"] if config.polarity == "both" else [config.polarity]


def clustered_rows(
    rows: Float[torch.Tensor, "n_hits hidden"], weight: torch.Tensor, neuron_idx: int, clustering: ClusteringConfig
) -> Float[np.ndarray, "n_hits hidden"]:
    """`on: input` drops the weight row, an ablation of it. Both are L2-normalised unless `normalise` is off."""
    if clustering.on == "input":
        points = rows.detach().cpu().float().numpy()
    else:
        points = raw_hadamard_from_rows(rows, weight, neuron_idx)
    return normalize(points, "l2") if clustering.normalise else points


def report_name(neuron_idx: int, polarity: Polarity) -> str:
    """Positive keeps the historical `neuron_{idx}` so existing reports are not renamed."""
    suffixes = {"positive": "", "negative": "_neg", "signed": "_signed"}
    return f"neuron_{neuron_idx}{suffixes[polarity]}"


def forest_device(device: str) -> str | None:
    """The torch forest needs float64, which only CUDA does quickly. MPS has no float64, so it and the CPU use numpy."""
    return device if device.startswith("cuda") else None


def density_peak_trees(
    hdmd: Float[np.ndarray, "n_hits hidden"], clustering: ClusteringConfig, strategy: DensityPeaksConfig,
    device: str | None = None,
) -> tuple[DensityPeakForest, ForestTrees]:
    """`device` runs the forest with torch there. See `forest_device`."""
    if device is None:
        forest = density_peak_forest(hdmd, strategy.k, clustering.distance, clustering.normalise)
    else:
        forest = torch_forest.density_peak_forest(hdmd, strategy.k, clustering.distance, clustering.normalise, device)
    return forest, forest_trees(forest)


class Clustered(BaseModel):
    memberships: list[HitMembership]
    forest: ForestData | None = None
    """Only density peaks builds a forest. The report page draws it."""


def cluster_hits(
    hdmd: Float[np.ndarray, "n_hits hidden"], clustering: ClusteringConfig, device: str | None = None,
) -> Clustered:
    match clustering.strategy:
        case DensityPeaksConfig() as strategy:
            forest, trees = density_peak_trees(hdmd, clustering, strategy, device)
            probabilities = calculate_probs(forest, trees, strategy.peak_share)
            return Clustered(
                memberships=[ClusterProbabilities(probabilities=p) for p in probabilities[forest.distinct_of_hit]],
                forest=forest_data(
                    forest, trees, probabilities,
                    comparable_rows(hdmd.astype(np.float64), clustering.distance, clustering.normalise),
                    clustering.distance,
                    strategy.max_entropy,
                ),
            )
        case HdbscanConfig() as strategy:
            return Clustered(memberships=one_hot_memberships(cluster_labels(
                hdmd, strategy.min_cluster_size, strategy.selection_method, strategy.min_samples, clustering.distance,
                clustering.normalise,
            )))


def hit_memberships(
    hdmd: Float[np.ndarray, "n_hits hidden"], clustering: ClusteringConfig, device: str | None = None,
) -> list[HitMembership]:
    return cluster_hits(hdmd, clustering, device).memberships


def max_entropy_of(clustering: ClusteringConfig) -> float | None:
    """Only density peaks gives soft probabilities. HDBSCAN's are one-hot, so it needs no threshold."""
    match clustering.strategy:
        case DensityPeaksConfig(max_entropy=max_entropy):
            return max_entropy
        case HdbscanConfig():
            return None


class PreparedRun(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    module: torch.nn.Module
    """The raw model, which the layer getters index into and attribution runs on."""
    model: NNsight
    source: SampleSource
    presenter: ClusterPresenter
    capture: NeuronCapture
    layout: TokenLayout
    attr_fn: AttrFn


def prepare_run[SamplesT](config: RunConfig, adapter: ModelAdapter[SamplesT], samples: SamplesT) -> PreparedRun:
    module = adapter.get_model()
    adapter.patch_for_attn_lrp()  # before NNsight wraps: it stores each module's forward at wrap time
    model = NNsight(module).to(config.device)
    source = adapter.make_source(samples)
    clustered_label = "normalised inputs" if config.clustering.on == "input" else "hadamard products"
    presenter = adapter.make_presenter(source, clustered_label)
    layout = layout_for(layer_getter_for(config)(module))
    capture = NeuronCapture(
        model, source, layer_getter_for(config), config.neuron_idxs, config.batch_size, config.device, layout
    )
    return PreparedRun(
        module=module, model=model, source=source, presenter=presenter, capture=capture, layout=layout,
        attr_fn=adapter.attr_fn,
    )


def report_neuron(
    neuron_idx: int,
    polarity: Polarity,
    selection: HitSelection,
    rows: Float[torch.Tensor, "n_hits hidden"],
    sample_ids: list[str],
    weight: torch.Tensor,
    noise: Float[np.ndarray, "n"],
    run: PreparedRun,
    config: RunConfig,
) -> None:
    tag = f"neuron {neuron_idx} {polarity}"
    hdmd = clustered_rows(rows, weight, neuron_idx, config.clustering)
    with timed(f"{tag}: cluster"):
        clustered = cluster_hits(hdmd, config.clustering, forest_device(config.device))
    memberships = clustered.memberships
    labels = confident_labels(memberships, max_entropy_of(config.clustering))
    mixed = int((most_probable_labels(memberships) != labels).sum())
    print(f"[{tag}] {len(set(labels) - {NOISE_LABEL})} clusters, {mixed} of {len(labels)} hits too mixed to show")

    hits = NeuronClusterHits(
        model=run.module,
        source=run.source,
        presenter=run.presenter,
        sample_ids=sample_ids,
        layer_getter=layer_getter_for(config),
        neuron_idx=neuron_idx,
        selection=selection,
        labels=labels,
        hdmds=hdmd,
        hadamard_shape=near_square_shape(hdmd.shape[1]),
        noise_activations=noise,
        device=config.device,
    )
    max_entropy = max_entropy_of(config.clustering)
    if max_entropy is not None:
        with timed(f"{tag}: write mixed hits"):
            mixed_path = write_mixed_hits_page(
                hits, Path(config.out_dir) / report_name(neuron_idx, polarity),
                probability_matrix(memberships),
                membership_entropy(memberships), max_entropy, config.report.min_uniq_samples_per_cluster, run.attr_fn,
            )
        print(f"[{tag}] mixed hits -> {mixed_path}")
    if clustered.forest is not None:
        with timed(f"{tag}: write border pages"):
            border_paths = write_border_pages(
                hits, Path(config.out_dir) / report_name(neuron_idx, polarity), clustered.forest,
                set(hits.shown_cluster_ids(config.report.min_uniq_samples_per_cluster)), run.attr_fn,
            )
        print(f"[{tag}] {len(border_paths)} border pages")
    with timed(f"{tag}: write report"):
        report_path = hits.write_report(
            config.out_dir,
            name=report_name(neuron_idx, polarity),
            attr_fn=run.attr_fn,
            max_n=config.report.max_hits_per_cluster,
            min_uniq_images=config.report.min_uniq_samples_per_cluster,
            forest=clustered.forest,
        )
    print(f"[{tag}] report -> {report_path}")


def run_neuron_reports[SamplesT](config: RunConfig, adapter: ModelAdapter[SamplesT], samples: SamplesT) -> None:
    run = prepare_run(config, adapter, samples)

    with timed("pass 1: scan"):
        scan = run.capture.scan()
    print(f"scanned {len(scan.sample_ids)} samples, activations {tuple(scan.activations.shape)}, "
          f"{int(scan.valid_mask.sum())}/{scan.valid_mask.numel()} positions unpadded")

    targets = [(n, polarity) for n in config.neuron_idxs for polarity in polarities_for(config)]
    selections = {
        (n, polarity): select_hits(
            scan, n, polarity, config.hits, config.seed
        ) for (n, polarity) in targets
    }
    merged = merge_positions([(s.batch_idx, s.token_idx) for s in selections.values()], scan.valid_mask.shape[1])
    with timed("pass 2: gather"):
        gathered = run.capture.gather(merged.batch_idx, merged.token_idx)
    worst = run.capture.check_consistent(scan, gathered, merged.batch_idx, merged.token_idx)
    print(f"gathered {len(merged.batch_idx)} positions; pass 2 matches pass 1 to {worst:.2e}")

    run.model.to("cpu")
    empty_cache()
    weight = run.layout.weight(layer_getter_for(config)(run.module).weight)

    for (neuron_idx, polarity), index_map in zip(targets, merged.index_maps):
        report_neuron(
            neuron_idx, polarity, selections[(neuron_idx, polarity)], gathered.inputs[index_map],
            scan.sample_ids, weight, sample_noise(scan, neuron_idx, config.seed), run, config,
        )
