"""The capture, select, cluster and report sequence behind every neuron-report script.

Scripts only decide what the samples are and which adapter loads the model. The model is
patched for AttnLRP once, before NNsight wraps it, since the patch mutates process-wide state
and NNsight stores each module's forward at wrap time (see ModelAdapter.patch_for_attn_lrp).
"""

import time
from contextlib import contextmanager

import torch
from jaxtyping import Float
from nnsight import NNsight
from pydantic import BaseModel, ConfigDict

from .attribution import compute_attnlrp_relevance
from .clustering import cluster_labels
from .device import empty_cache
from .hadamard import hadamard_from_rows, near_square_shape, normalized_rows
from .hit_selection import HitSelection, select_hits
from .layers import LayerGetter, down_proj_getter, k_proj_getter, q_proj_getter
from .models import ModelAdapter
from .neuron_capture import NeuronCapture, merge_positions
from .neuron_report import NeuronClusterHits
from .presenters import ClusterPresenter
from .run_config import Polarity, RunConfig, Target
from .sources import SampleSource


@contextmanager
def timed(label: str):
    start = time.perf_counter()
    yield
    print(f"[{label}] {time.perf_counter() - start:.2f}s")


def layer_getter_for(config: RunConfig) -> LayerGetter:
    getters = {Target.DOWN_PROJ: down_proj_getter, Target.Q_PROJ: q_proj_getter, Target.K_PROJ: k_proj_getter}
    return getters[config.target](config.layer_idx)


def polarities_for(config: RunConfig) -> list[Polarity]:
    return ["positive", "negative"] if config.polarity == "both" else [config.polarity]


def report_name(neuron_idx: int, polarity: Polarity) -> str:
    """Positive keeps the historical `neuron_{idx}` so existing reports are not renamed."""
    return f"neuron_{neuron_idx}" if polarity == "positive" else f"neuron_{neuron_idx}_neg"


class PreparedRun(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    module: torch.nn.Module
    """The raw model, which the layer getters index into and attribution runs on."""
    model: NNsight
    source: SampleSource
    presenter: ClusterPresenter
    capture: NeuronCapture


def prepare_run[SamplesT](config: RunConfig, adapter: ModelAdapter[SamplesT], samples: SamplesT) -> PreparedRun:
    module = adapter.get_model()
    adapter.patch_for_attn_lrp()  # before NNsight wraps: it stores each module's forward at wrap time
    model = NNsight(module).to(config.device)
    source = adapter.make_source(samples)
    clustered_label = "normalised inputs" if config.clustering.on == "input" else "hadamard products"
    presenter = adapter.make_presenter(source, clustered_label)
    capture = NeuronCapture(model, source, layer_getter_for(config), config.neuron_idxs, config.batch_size, config.device)
    return PreparedRun(module=module, model=model, source=source, presenter=presenter, capture=capture)


def report_neuron(
    neuron_idx: int,
    polarity: Polarity,
    selection: HitSelection,
    rows: Float[torch.Tensor, "n_hits hidden"],
    sample_ids: list[str],
    weight: torch.Tensor,
    run: PreparedRun,
    config: RunConfig,
) -> None:
    tag = f"neuron {neuron_idx} {polarity}"
    clustering = config.clustering
    hdmd = normalized_rows(rows) if config.clustering.on == "input" else hadamard_from_rows(rows, weight, neuron_idx)
    with timed(f"{tag}: cluster"):
        labels = cluster_labels(hdmd, clustering.min_cluster_size, clustering.selection_method, clustering.min_samples)
    print(f"[{tag}] {len(set(labels) - {-1})} clusters")

    hits = NeuronClusterHits(
        model=run.module,
        source=run.source,
        presenter=run.presenter,
        sample_ids=sample_ids,
        layer_getter=layer_getter_for(config),
        neuron_idx=neuron_idx,
        token_idx=selection.token_idx,
        batch_idx=selection.batch_idx,
        labels=labels,
        hdmds=hdmd,
        hadamard_shape=near_square_shape(hdmd.shape[1]),
        threshold=selection.threshold,
        elbow_values=selection.elbow_values.numpy(),
        elbow_idx=selection.elbow_idx,
        bucket_edges=selection.bucket_edges,
        device=config.device,
    )
    with timed(f"{tag}: write report"):
        report_path = hits.write_report(
            config.out_dir,
            name=report_name(neuron_idx, polarity),
            attr_fn=compute_attnlrp_relevance,
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
    selections = {t: select_hits(scan, *t, config.hits, config.seed) for t in targets}
    merged = merge_positions([(s.batch_idx, s.token_idx) for s in selections.values()], scan.valid_mask.shape[1])
    with timed("pass 2: gather"):
        gathered = run.capture.gather(merged.batch_idx, merged.token_idx)
    worst = run.capture.check_consistent(scan, gathered, merged.batch_idx, merged.token_idx)
    print(f"gathered {len(merged.batch_idx)} positions; pass 2 matches pass 1 to {worst:.2e}")

    run.model.to("cpu")
    empty_cache()
    weight = layer_getter_for(config)(run.module).weight

    for (neuron_idx, polarity), index_map in zip(targets, merged.index_maps):
        report_neuron(
            neuron_idx, polarity, selections[(neuron_idx, polarity)], gathered.inputs[index_map],
            scan.sample_ids, weight, run, config,
        )
