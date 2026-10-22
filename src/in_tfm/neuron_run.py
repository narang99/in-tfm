"""The capture, select, cluster and report sequence behind every neuron-report script.

Scripts only decide what the samples are and which adapter loads the model. The model is
patched for AttnLRP once, before NNsight wraps it, since the patch mutates process-wide state
and NNsight stores each module's forward at wrap time (see ModelAdapter.patch_for_attn_lrp).
"""

import time
from contextlib import contextmanager

import numpy as np
import torch
from jaxtyping import Float
from nnsight import NNsight
from pydantic import BaseModel, ConfigDict
from sklearn.preprocessing import normalize

from .activation_ranges import sample_noise
from .cluster_membership import (
    NOISE_LABEL, ClusterProbabilities, HitMembership, confident_labels, most_probable_labels, one_hot_memberships,
)
from .clustering import cluster_labels
from .density_peaks import DensityPeakForest, ForestTrees, calculate_probs, density_peak_forest, forest_trees
from .device import empty_cache
from .hadamard import near_square_shape, raw_hadamard_from_rows
from .hit_selection import HitSelection, select_hits
from .layers import LayerGetter, conv_getter, down_proj_getter, fc2_getter, k_proj_getter, q_proj_getter
from .models import ModelAdapter
from .neuron_capture import NeuronCapture, merge_positions
from .neuron_report import AttrFn, NeuronClusterHits
from .presenters import ClusterPresenter
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
    return f"neuron_{neuron_idx}" if polarity == "positive" else f"neuron_{neuron_idx}_neg"


def density_peak_trees(
    hdmd: Float[np.ndarray, "n_hits hidden"], clustering: ClusteringConfig, strategy: DensityPeaksConfig
) -> tuple[DensityPeakForest, ForestTrees]:
    forest = density_peak_forest(hdmd, strategy.k, clustering.distance, clustering.normalise)
    return forest, forest_trees(forest)


def hit_memberships(hdmd: Float[np.ndarray, "n_hits hidden"], clustering: ClusteringConfig) -> list[HitMembership]:
    match clustering.strategy:
        case DensityPeaksConfig() as strategy:
            forest, trees = density_peak_trees(hdmd, clustering, strategy)
            probabilities = calculate_probs(forest, trees, strategy.peak_share)
            return [ClusterProbabilities(probabilities=p) for p in probabilities[forest.distinct_of_hit]]
        case HdbscanConfig() as strategy:
            return one_hot_memberships(cluster_labels(
                hdmd, strategy.min_cluster_size, strategy.selection_method, strategy.min_samples, clustering.distance,
                clustering.normalise,
            ))


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
        memberships = hit_memberships(hdmd, config.clustering)
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
    with timed(f"{tag}: write report"):
        report_path = hits.write_report(
            config.out_dir,
            name=report_name(neuron_idx, polarity),
            attr_fn=run.attr_fn,
            max_n=config.report.max_hits_per_cluster,
            min_uniq_images=config.report.min_uniq_samples_per_cluster,
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
