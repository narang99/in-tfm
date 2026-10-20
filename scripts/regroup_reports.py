#!/usr/bin/env python
"""Group the clusters of finished neuron reports, and re-render their pages without the model.

    uv run python scripts/regroup_reports.py --reports-root reports/mnist_cnn_small --merge-clusters.method thresholds
    uv run python scripts/regroup_reports.py --report-dirs reports/x/neuron_3 --merge-clusters.method kmeans --merge-clusters.k 4

- Each grouping is added to the report's `meta.json` under its own name, and becomes the one the page shows.
- The page reuses each cluster's cached body, so a report rendered before that cache existed is skipped.
  - Regenerate such a report with `--merge-clusters.method` on `run_neuron_report.py` instead.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from in_tfm.cluster_merging import merge_report_clusters
from in_tfm.neuron_report import MissingRenderCacheError, render_report
from in_tfm.report_meta import META_NAME
from in_tfm.run_config import MergeClustersConfig


class RegroupConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True, cli_parse_args=True)

    report_dirs: list[Path] = Field([], description="neuron report folders, each with a meta.json")
    reports_root: Path | None = Field(None, description="every neuron report folder directly under it")
    merge_clusters: MergeClustersConfig = MergeClustersConfig(method="thresholds")
    seed: int = 0


def report_dirs(config: RegroupConfig) -> list[Path]:
    found = sorted(d.parent for d in config.reports_root.glob(f"*/{META_NAME}")) if config.reports_root else []
    return [*config.report_dirs, *found]


def main() -> None:
    config = RegroupConfig()
    for report_dir in report_dirs(config):
        name = merge_report_clusters(report_dir, config.merge_clusters, config.seed)
        try:
            render_report(report_dir)
        except MissingRenderCacheError as error:
            print(f"{report_dir}: grouped as {name}, but not re-rendered: {error}")
            continue
        print(f"{report_dir}: {name}")


if __name__ == "__main__":
    main()
