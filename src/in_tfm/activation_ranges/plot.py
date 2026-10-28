"""The plotting helpers the activation range figure uses, cut down from hiccup-ide.

- Adapted from `olt/act_ranges/plotting.py`, `stats.py` and `constants/general.py` of hiccup-ide (hariom/hp-do).
  - The unchanged copy is in git history, commit `8118be1`.
- Dropped: the interactive and jpeg grid functions and `noise_overlay`, which the figure builder replaces.
- Names are public now, since `figure.py` imports them.
"""

import math

import numpy as np
from jaxtyping import Float
from matplotlib.axes import Axes
from matplotlib.figure import Figure

Activations = Float[np.ndarray, "n"]

NOISE_COLOR = "white"
NOISE_RANGE_COLOR = "#888888"
BACKGROUND = "#212529"


def shorth(data: Activations, coverage: float) -> tuple[float, float]:
    """The narrowest interval holding `coverage` of the data, so a range that ignores outliers."""
    ordered = np.sort(data)
    window = math.ceil(coverage * len(ordered))
    widths = ordered[window - 1 :] - ordered[: len(ordered) - window + 1]
    start = int(np.argmin(widths))
    return float(ordered[start]), float(ordered[start + window - 1])


def get_noise_range(data: Activations, coverage: float = 0.9) -> tuple[float, float]:
    return shorth(data, coverage)


def jittered_x(n: int, rng: np.random.Generator) -> Float[np.ndarray, "n"]:
    """Series of equal length on integer x positions would stack into vertical stripes, so x is random."""
    return rng.uniform(0, max(n, 1), size=n)


def subsample_and_color(
    label_by_points: dict[int, list[float]], colors: list[str], max_points_per_label: int, rng: np.random.Generator
) -> tuple[dict[int, Activations], dict[int, str]]:
    """Each label is capped, and gets one color that every panel then shares."""
    subsampled = {
        label: rng.choice(points, size=min(len(points), max_points_per_label), replace=False)
        for label, points in label_by_points.items()
    }
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(subsampled)}
    return subsampled, color_map


def draw_label_panel(
    ax: Axes, panel_labels: list[int], subsampled: dict[int, Activations], color_map: dict[int, str], rng: np.random.Generator
) -> None:
    for label in panel_labels:
        ys = subsampled[label]
        ax.scatter(jittered_x(len(ys), rng), ys, color=color_map[label], label=label)


def draw_hlines(ax: Axes, hlines: list[tuple]) -> None:
    """Each line is `(y, color, linestyle, linewidth, label)`, and a falsy label draws it without a legend entry."""
    for y, color, linestyle, linewidth, label in hlines:
        kwargs = {"label": label} if label else {}
        ax.axhline(y, color=color, linestyle=linestyle, linewidth=linewidth, **kwargs)


def grid_dims_for_labels(n_labels: int, clusters_per_panel: int = 8) -> tuple[int, int]:
    """The nearest-square (rows, columns) grid that holds about `clusters_per_panel` labels per panel."""
    n_panels = max(1, math.ceil(n_labels / clusters_per_panel))
    n_cols = math.ceil(math.sqrt(n_panels))
    return math.ceil(n_panels / n_cols), n_cols


def style_dark_panels_and_legends(fig: Figure, axes: list[Axes]) -> None:
    """Axes that are switched off stay as they are."""
    fig.patch.set_facecolor(BACKGROUND)
    for ax in axes:
        if not ax.axison:
            continue
        ax.set_facecolor(BACKGROUND)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(BACKGROUND)
            legend.get_frame().set_edgecolor("white")
            for text in legend.get_texts():
                text.set_color("white")
