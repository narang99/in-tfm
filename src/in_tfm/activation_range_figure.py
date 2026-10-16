"""The per-cluster activation range figure, built from the helpers cut down from hiccup-ide.

- Clusters are sorted by mean activation, highest first, and dealt round-robin to the panels.
  - So every panel holds high and low clusters, and not only the top ones in the first panel.
  - hiccup-ide's grid function gave each panel a contiguous run of labels, so it was dropped.
- The neuron's noise sample is drawn in every panel, next to its shorth range.
- Panels are laid out and styled by the helpers in `activation_range_plot.py`.
"""

import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .activation_range_plot import (
    KELLY_COLORS,
    NOISE_COLOR,
    NOISE_RANGE_COLOR,
    draw_hlines,
    draw_label_panel,
    get_noise_range,
    grid_dims_for_labels,
    jittered_x,
    style_dark_panels_and_legends,
    subsample_and_color,
)

CLUSTERS_PER_PANEL = 8
MAX_POINTS_PER_SERIES = 200


def sorted_by_mean_descending(label_by_points: dict[int, list[float]]) -> dict[int, list[float]]:
    return dict(sorted(label_by_points.items(), key=lambda item: np.mean(item[1]), reverse=True))


def panels_needed(n_labels: int, clusters_per_panel: int = CLUSTERS_PER_PANEL) -> int:
    return max(1, math.ceil(n_labels / clusters_per_panel))


def round_robin_panels(labels: Sequence[int], n_panels: int) -> list[list[int]]:
    """Label i goes to panel i % n_panels, so a mean-sorted list spreads high and low labels over every panel."""
    return [list(labels[panel::n_panels]) for panel in range(n_panels)]


def noise_range_hlines(noise: np.ndarray) -> list[tuple]:
    noise_min, noise_max = get_noise_range(noise)
    return [
        (noise_min, NOISE_RANGE_COLOR, "--", 1.0, None),
        (noise_max, NOISE_RANGE_COLOR, "--", 1.0, "noise range"),
    ]


def draw_noise_series(ax, noise_points: np.ndarray, rng: np.random.Generator) -> None:
    ax.scatter(jittered_x(len(noise_points), rng), noise_points, color=NOISE_COLOR, alpha=0.25, s=10, label="noise")


def show_activation_range_grid(label_by_points: dict[int, list[float]], noise: np.ndarray, rng: np.random.Generator):
    ranked = sorted_by_mean_descending(label_by_points)
    subsampled, color_map = subsample_and_color(ranked, KELLY_COLORS[1:], MAX_POINTS_PER_SERIES, rng)
    noise_points = rng.choice(noise, size=min(len(noise), MAX_POINTS_PER_SERIES), replace=False)
    hlines = noise_range_hlines(noise)

    n_rows, n_cols = grid_dims_for_labels(len(ranked), CLUSTERS_PER_PANEL)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), sharey=True, squeeze=False)
    axes = axes.reshape(-1)
    shares = round_robin_panels(list(ranked), min(len(axes), panels_needed(len(ranked))))
    for i, ax in enumerate(axes):
        if i >= len(shares) or not shares[i]:
            ax.axis("off")
            continue
        draw_label_panel(ax, shares[i], subsampled, color_map, rng)
        draw_noise_series(ax, noise_points, rng)
        draw_hlines(ax, hlines)
        ax.legend()
    fig.tight_layout()
    return fig, axes


def save_activation_range_plot(
    label_by_points: dict[int, list[float]], noise: np.ndarray, output_path: Path, rng: np.random.Generator
) -> None:
    """Headless, so the backend is fixed here and not left to the machine's default."""
    matplotlib.use("Agg")
    fig, axes = show_activation_range_grid(label_by_points, noise, rng)
    style_dark_panels_and_legends(fig, axes)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="jpeg", facecolor=fig.get_facecolor(), pil_kwargs={"quality": 90})
    plt.close(fig)
