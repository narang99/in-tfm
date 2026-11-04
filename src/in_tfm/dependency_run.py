"""Dependency reports for the clusters of neuron reports a run config already made.

- The config is the neuron report's own, so the run finds the reports where that run wrote them.
- The samples are loaded the same way, and the report's `meta.json` must list the same ones.
  - Otherwise the hit positions in `clustered_hits.json` would point at other samples.
"""

from pathlib import Path

import torch
from pydantic import BaseModel

from .dependencies import DependencyMap, SupportsDependencies
from .dependencies.collect import (
    CaptureSetup,
    ClusterDependencies,
    Matchers,
    OriginCluster,
    collect_cluster_dependencies,
    write_cluster_dependencies,
)
from .dependencies.matching import WriterReport, build_matcher
from .dependency_page import write_dependency_page
from .dependency_page.graph import write_graph_page
from .models import ModelAdapter
from .neuron_report import render_report
from .neuron_run import clustered_label, layer_getter_for, report_name, timed
from .report_meta import ReportMeta, load_report_meta
from .run_config import RunConfig, load_run_config
from .sources import SampleSource


def dependency_map_of(adapter: object, target_layer: str) -> DependencyMap:
    if not isinstance(adapter, SupportsDependencies):
        raise ValueError(
            f"{type(adapter).__name__} has no dependency map, so it cannot make dependency reports"
        )
    return adapter.dependency_map_for(target_layer)


def layer_weight(layer: torch.nn.Module, adapter: ModelAdapter) -> torch.Tensor:
    weight = layer.weight
    assert isinstance(weight, torch.Tensor)
    return adapter.layout_for(layer).weight(weight).detach().cpu()


def tile_shape_of(layer: torch.nn.Module) -> tuple[int, ...]:
    """How a Hadamard row on this layer is drawn, one tile per input channel for a conv."""
    if isinstance(layer, torch.nn.Conv2d):
        return (layer.in_channels, *layer.kernel_size)
    weight = layer.weight
    assert isinstance(weight, torch.Tensor)
    return (weight.shape[1],)


def writer_reports(
    writer_config: RunConfig, module: torch.nn.Module
) -> list[WriterReport]:
    """Every neuron of the writer config that has a report on disk."""
    layer = layer_getter_for(writer_config)(module)
    reports = [
        WriterReport(
            layer=writer_config.layer_name,
            neuron=neuron,
            report_dir=Path(writer_config.out_dir)
            / report_name(neuron, writer_config.polarity),
            clustering=writer_config.clustering,
            tile_shape=tile_shape_of(layer),
        )
        for neuron in writer_config.neuron_idxs
    ]
    return [r for r in reports if (r.report_dir / "meta.json").exists()]


def writer_matchers(
    config: RunConfig,
    adapter: ModelAdapter,
    module: torch.nn.Module,
    setup: CaptureSetup,
) -> Matchers:
    matchers: Matchers = {}
    for path in config.dependencies.writer_configs:
        writer_config = load_run_config(["--config", str(path)])
        weight = layer_weight(layer_getter_for(writer_config)(module), adapter)
        for report in writer_reports(writer_config, module):
            check_same_samples(
                load_report_meta(report.report_dir), setup.source, report.report_dir
            )
            with timed(f"matcher {report.layer}:{report.neuron}"):
                matchers[(report.layer, report.neuron)] = build_matcher(
                    setup, report, weight
                )
    return matchers


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


class DependencyRun(BaseModel, arbitrary_types_allowed=True):
    config: RunConfig
    setup: CaptureSetup
    dependency_map: DependencyMap
    weight: torch.Tensor
    matchers: Matchers


def report_cluster(run: DependencyRun, origin: OriginCluster) -> ClusterDependencies:
    tag = f"neuron {origin.neuron} cluster {origin.cluster_id}"
    with timed(f"{tag}: collect"):
        collected = collect_cluster_dependencies(
            run.setup,
            origin,
            run.weight[origin.neuron],
            run.dependency_map,
            run.config.dependencies.selection,
            run.config.seed,
            run.matchers,
        )
    write_cluster_dependencies(collected)
    with timed(f"{tag}: page"):
        page = write_dependency_page(
            origin.report_dir,
            origin.cluster_id,
            run.config.dependencies.stats,
            run.config.dependencies.view,
        )
    print(f"[{tag}] {len(collected.examples)} examples -> {page}")
    return collected


def report_neuron(run: DependencyRun, neuron: int) -> list[ClusterDependencies]:
    report_dir = Path(run.config.out_dir) / report_name(neuron, run.config.polarity)
    meta = load_report_meta(report_dir)
    check_same_samples(meta, run.setup.source, report_dir)
    return [
        report_cluster(
            run,
            OriginCluster(
                report_dir=report_dir,
                layer=run.config.layer_name,
                neuron=neuron,
                cluster_id=cluster_id,
            ),
        )
        for cluster_id in cluster_ids_to_report(run.config, meta)
    ]


def run_dependency_reports[SamplesT, SourceT: SampleSource](
    config: RunConfig, adapter: ModelAdapter[SamplesT, SourceT], samples: SamplesT
) -> Path:
    """Returns the run's graph page. Each origin report is re-rendered, so its clusters link to their pages."""
    module = adapter.get_model().eval()
    source = adapter.make_source(samples)
    setup = CaptureSetup(
        model=module,
        source=source,
        layout_for=adapter.layout_for,
        batch_size=config.batch_size,
        device=config.device,
    )
    run = DependencyRun(
        config=config,
        setup=setup,
        dependency_map=dependency_map_of(adapter, config.layer_name),
        weight=layer_weight(layer_getter_for(config)(module), adapter),
        matchers=writer_matchers(config, adapter, module, setup),
    )
    presenter = adapter.make_presenter(source, clustered_label(config.clustering))
    collections: list[ClusterDependencies] = []
    for neuron in config.neuron_idxs:
        collections += report_neuron(run, neuron)
        render_report(
            Path(config.out_dir) / report_name(neuron, config.polarity), presenter
        )
    graph = write_graph_page(
        collections, Path(config.out_dir), config.dependencies.view
    )
    print(f"[graph] {graph}")
    return graph
