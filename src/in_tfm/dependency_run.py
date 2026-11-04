"""Dependency reports for the clusters of neuron reports a run config already made.

- The config is the neuron report's own, so the run finds the reports where that run wrote them.
- The samples are loaded the same way, and the report's `meta.json` must list the same ones.
  - Otherwise the hit positions in `clustered_hits.json` would point at other samples.
"""

from pathlib import Path

import torch

from .dependencies import DependencyMap, SupportsDependencies
from .dependencies.collect import (
    CaptureSetup,
    OriginCluster,
    collect_cluster_dependencies,
    write_cluster_dependencies,
)
from .dependency_page import write_dependency_page
from .models import ModelAdapter
from .neuron_run import layer_getter_for, report_name, timed
from .report_meta import ReportMeta, load_report_meta
from .run_config import RunConfig
from .sources import SampleSource


def dependency_map_of(adapter: object, target_layer: str) -> DependencyMap:
    if not isinstance(adapter, SupportsDependencies):
        raise ValueError(
            f"{type(adapter).__name__} has no dependency map, so it cannot make dependency reports"
        )
    return adapter.dependency_map_for(target_layer)


def target_weight(
    config: RunConfig, adapter: ModelAdapter, module: torch.nn.Module
) -> torch.Tensor:
    layer = layer_getter_for(config)(module)
    weight = layer.weight
    assert isinstance(weight, torch.Tensor)
    return adapter.layout_for(layer).weight(weight).detach().cpu()


def check_same_samples(
    meta: ReportMeta, source: SampleSource, report_dir: Path
) -> None:
    if list(source.sample_ids()) != meta.sample_ids:
        raise ValueError(
            f"{report_dir} was made from other samples than this config loads; "
            "its hit positions would point at the wrong samples"
        )


def cluster_ids_to_report(config: RunConfig, meta: ReportMeta) -> list[int]:
    return config.dependencies.cluster_ids or list(meta.clusters)


def run_dependency_reports[SamplesT, SourceT: SampleSource](
    config: RunConfig, adapter: ModelAdapter[SamplesT, SourceT], samples: SamplesT
) -> list[Path]:
    dependency_map = dependency_map_of(adapter, config.layer_name)
    module = adapter.get_model().eval()
    source = adapter.make_source(samples)
    weight = target_weight(config, adapter, module)
    setup = CaptureSetup(
        model=module,
        source=source,
        layout_for=adapter.layout_for,
        batch_size=config.batch_size,
        device=config.device,
    )
    pages = []
    for neuron in config.neuron_idxs:
        report_dir = Path(config.out_dir) / report_name(neuron, config.polarity)
        meta = load_report_meta(report_dir)
        check_same_samples(meta, source, report_dir)
        for cluster_id in cluster_ids_to_report(config, meta):
            origin = OriginCluster(
                report_dir=report_dir,
                layer=config.layer_name,
                neuron=neuron,
                cluster_id=cluster_id,
            )
            with timed(f"neuron {neuron} cluster {cluster_id}: collect"):
                collected = collect_cluster_dependencies(
                    setup,
                    origin,
                    weight[neuron],
                    dependency_map,
                    config.dependencies.selection,
                    config.seed,
                )
            write_cluster_dependencies(collected)
            with timed(f"neuron {neuron} cluster {cluster_id}: page"):
                page = write_dependency_page(
                    report_dir,
                    cluster_id,
                    config.dependencies.stats,
                    config.dependencies.view,
                )
            print(
                f"[neuron {neuron} cluster {cluster_id}] {len(collected.examples)} examples -> {page}"
            )
            pages.append(page)
    return pages
