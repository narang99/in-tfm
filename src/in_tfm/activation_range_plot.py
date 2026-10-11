"""Copied verbatim from hiccup-ide, branch `hariom/hp-do`, commit 12ec85cd742fdfe6701b61cad63fdfc44e56c3d5.

- Sources:
  - `olt/src/olt/act_ranges/plotting.py`: the per-label scatter grid functions.
  - `olt/src/olt/act_ranges/stats.py`: `shorth`, `get_noise_range` and `noise_stats`.
  - `olt/src/olt/act_ranges/constants/general.py`: `KELLY_COLORS`.
- Function bodies, docstrings and comments are unchanged.
- The only edit is that the two `olt` imports are gone, since what they imported is defined below.
- Left behind: the other plots in `plotting.py`, which the reports do not use.
- Nothing in this repo uses this module yet, and it is not tested.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

KELLY_COLORS = [
    "#F2F3F4",  # white (skip as bg reference)
    "#F3C300",
    "#875692",
    "#F38400",
    "#A1CAF1",
    "#BE0032",
    "#C2B280",
    "#848482",
    "#008856",
    "#E68FAC",
    "#0067A5",
    "#F99379",
    "#604E97",
    "#F6A600",
    "#B3446C",
    "#DCD300",
    "#882D17",
    "#8DB600",
    "#654522",
    "#E25822",
    "#2B3D26",
]

# KELLY_COLORS[0] is white, reserved as the noise scatter's color below —
# excluded here too so no matched cluster gets a color easily confused with
# the noise/unmatched colors below.
_CLUSTER_COLORS = KELLY_COLORS[1:]
_NOISE_COLOR = "white"
_UNMATCHED_COLOR = "#888888"


_POS_FILL = "#8fd19e"  # light green
_NEG_FILL = "#f4a3a3"  # light red
_POS_MARKER = "#1b7a34"  # dark green
_NEG_MARKER = "#a11f1f"  # dark red
_HIST_COLOR = "#4dabf7"  # light blue
_NOISE_RANGE_COLOR = "#888888"  # grey — the two shorth noise-range lines
_THRESHOLD_COLOR = "#f59f00"  # amber — the above-noise threshold line (distinct)



def shorth(data, frac=0.5):
    data_sorted = np.sort(data)
    n = len(data_sorted)
    window = int(np.ceil(frac * n))
    min_width = np.inf
    best_start = 0
    for i in range(n - window + 1):
        width = data_sorted[i + window - 1] - data_sorted[i]
        if width < min_width:
            min_width = width
            best_start = i
    return data_sorted[best_start], data_sorted[best_start + window - 1]


def get_noise_range(data, frac=0.9):
    return shorth(data, frac)


def noise_stats(noise, tol=1e-6):
    """
    (noise_min, noise_med, noise_max, noise_radius) for one dep neuron's raw
    noise samples — noise_min/noise_max via get_noise_range (shorth), noise_med
    the plain median, and noise_radius = noise_max - noise_med + tol (the
    "1 unit" used everywhere distances are expressed in noise-radius units,
    e.g. analyser.get_activation_distance_from_noise,
    report_stats.compute_output_activation_noise_median_distances) — tol avoids
    a division by zero on a degenerate (single-valued) noise sample.
    """
    noise_min, noise_max = get_noise_range(noise)
    noise_med = np.median(noise)
    noise_radius = noise_max - noise_med + tol
    return noise_min, noise_med, noise_max, noise_radius


def _style_dark_axis(fig, ax):
    fig.patch.set_facecolor("#212529")
    ax.set_facecolor("#212529")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")


def _jittered_x(n, rng):
    """
    Continuous random x in [0, n) for n points, instead of the integer
    sequence range(n) — plotting several same-length series (e.g. many
    same-size subsampled clusters) each against their own range(n) stacks
    points from every series onto the same handful of integer x positions,
    reading as solid vertical lines/columns rather than a scatter. Continuous
    jitter spreads them out so overlapping series are visually distinguishable.
    rng is caller-supplied so results are reproducible across a single figure.
    """
    return rng.uniform(0, max(n, 1), size=n)


def _subsample_and_color(label_by_points, colors, max_points_per_label, rng):
    """Subsample each label to <=max_points_per_label points (without
    replacement) and assign each a color (cycled via modulo). Returned once so
    every panel — including the extra above-noise panel in
    save_label_points_grid_jpeg — keeps the same color per cluster."""
    subsampled = {
        label: rng.choice(points, size=min(len(points), max_points_per_label), replace=False)
        for label, points in label_by_points.items()
    }
    labels = list(subsampled.keys())
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(labels)}
    return subsampled, color_map


def _draw_label_panel(ax, panel_labels, subsampled, color_map, rng):
    """Overlay one jittered scatter series per label onto ax (see _jittered_x
    for why x is continuous jitter, not range(n))."""
    for label in panel_labels:
        ys = subsampled[label]
        xs = _jittered_x(len(ys), rng)
        ax.scatter(xs, ys, color=color_map[label], label=label)


def _draw_hlines(ax, hlines):
    """Draw pre-computed horizontal reference lines on a scatter panel. `hlines`
    is a list of (y, color, linestyle, linewidth, label) tuples — deliberately a
    dumb drawing primitive with no knowledge of noise/thresholds, so callers
    decide what the lines mean and where they sit. A falsy label draws the line
    without a legend entry (e.g. the lower of a two-line range, so the pair reads
    as one entry). With sharey=True the lines sit at the same height across every
    panel, so "which clusters clear the bar" reads consistently panel-to-panel."""
    for y, color, linestyle, linewidth, label in hlines:
        kwargs = {"label": label} if label else {}
        ax.axhline(y, color=color, linestyle=linestyle, linewidth=linewidth, **kwargs)


def noise_overlay(label_by_points, noise, above_noise_threshold=0.8, above_noise_percentile=50):
    """Convenience for the noise view, kept OUT of the plotting primitives so
    they stay noise-agnostic — the caller runs this and passes its results down.
    From raw sampled `noise` (what get_random_sampled_activations returns; shorth
    over it IS the neuron's noise, see stats.noise_stats) it returns:
      * `hlines`: the styled two-line shorth noise range (grey dashed) + the
        amber above-noise threshold line at noise_med + above_noise_threshold *
        noise_radius — ready to pass as show_label_points_grid /
        save_label_points_grid_jpeg's `hlines`.
      * `above_label_by_points`: the subset of label_by_points whose
        above_noise_percentile-th point percentile clears that threshold — i.e.
        the filtering is done here, in the caller's layer, and the already-filtered
        dict is handed back for save_label_points_grid_jpeg's
        extra_label_by_points (empty dict if nothing clears the bar).
    """
    noise_min, noise_med, noise_max, noise_radius = noise_stats(np.asarray(noise))
    threshold = noise_med + above_noise_threshold * noise_radius
    hlines = [
        (noise_min, _NOISE_RANGE_COLOR, "--", 1.0, None),
        (noise_max, _NOISE_RANGE_COLOR, "--", 1.0, "noise range"),
        (threshold, _THRESHOLD_COLOR, "-", 1.4, "above-noise threshold"),
    ]
    above = {
        label: points
        for label, points in label_by_points.items()
        if len(points) > 0 and np.percentile(points, above_noise_percentile) > threshold
    }
    return hlines, above


def show_label_points_grid(
    label_by_points,
    n_rows,
    n_cols,
    colors=None,
    figsize=None,
    sharey=True,
    max_points_per_label=200,
    rng=None,
    hlines=None,
):
    """
    Interactive grid of per-label scatter panels — one panel of overlaid
    scatters per chunk of labels, labels split evenly (ceil(n_labels /
    (n_rows*n_cols)) per panel) across n_rows*n_cols panels, e.g. for eyeballing
    every cluster's raw point population (layer_by_channel_by_label_by_points)
    at once instead of one cluster at a time.

    label_by_points: dict[label, list[float]] — the full point population per
    label; each label is subsampled to at most max_points_per_label points
    (without replacement — a label already at or under the cap is plotted
    as-is rather than resampled with possible duplicates).

    Each label within a panel is plotted with jittered (not bare range(n)) x
    positions (see _jittered_x) — several labels capped to the same
    max_points_per_label length would otherwise share identical integer x
    positions and stack into vertical stripes instead of reading as a
    scatter (the same fix save_combined_scatter_jpeg uses for its overview
    scatter).

    colors: cycled (via modulo) across labels if there are more labels than
    colors — defaults to constants.KELLY_COLORS[1:] (index 0 is reserved
    elsewhere in this module for a "noise" series color, see
    _CLUSTER_COLORS).

    rng: np.random.Generator for subsampling/jitter reproducibility — a
    fresh np.random.default_rng() is used if not given.

    hlines: optional list of (y, color, linestyle, linewidth, label) horizontal
    reference lines drawn on every active panel (see _draw_hlines) — a neutral
    drawing primitive with no noise semantics. For the noise view, the caller
    builds these via noise_overlay and passes them in, rather than this function
    knowing anything about noise.

    Panels beyond however many are actually needed to place every label are
    turned off (axis hidden) rather than left as empty ticked boxes — this
    is computed per-panel from panel_labels directly (not sliced off the end
    by label count), since a chunk size > 1 per panel means "panels with no
    labels" and "labels used up" aren't the same index.

    Returns (fig, axes) — axes flattened to 1D regardless of n_rows/n_cols.
    """
    rng = np.random.default_rng() if rng is None else rng
    if colors is None:
        colors = KELLY_COLORS[1:]

    subsampled, color_map = _subsample_and_color(
        label_by_points, colors, max_points_per_label, rng
    )
    labels = list(subsampled.keys())

    if figsize is None:
        figsize = (7 * n_cols, 5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=sharey)
    axes = np.array(axes).reshape(-1)

    n_panels = n_rows * n_cols
    labels_per_panel = -(-len(labels) // n_panels) if n_panels > 0 else 0

    for i, ax in enumerate(axes):
        panel_labels = labels[i * labels_per_panel : (i + 1) * labels_per_panel]
        if not panel_labels:
            ax.axis("off")
            continue
        _draw_label_panel(ax, panel_labels, subsampled, color_map, rng)
        if hlines:
            _draw_hlines(ax, hlines)
        ax.legend()

    fig.tight_layout()
    return fig, axes


def _square_grid(n_panels):
    """Arrange n_panels into the nearest-square grid (cols = ceil(sqrt(n)))."""
    n_panels = max(1, n_panels)
    n_cols = math.ceil(math.sqrt(n_panels))
    n_rows = math.ceil(n_panels / n_cols)
    return n_rows, n_cols


def grid_dims_for_labels(n_labels, clusters_per_panel=8):
    """Heuristic (n_rows, n_cols) for laying out per-label scatter panels: pack
    ~`clusters_per_panel` labels into each panel, then arrange the resulting
    panels into the nearest-square grid (see _square_grid). n_labels==0 collapses
    to a single (1, 1) panel (show_label_points_grid just turns it off)."""
    return _square_grid(math.ceil(n_labels / clusters_per_panel))


def _style_dark_panels_and_legends(fig, axes):
    """Dark-theme every active panel (skip the axis-off ones) + recolor its
    legend for a dark background, so the figure matches the rest of the report."""
    for ax in axes:
        if not ax.axison:
            continue
        _style_dark_axis(fig, ax)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor("#212529")
            legend.get_frame().set_edgecolor("white")
            for text in legend.get_texts():
                text.set_color("white")


def save_label_points_grid_jpeg(
    label_by_points,
    output_path,
    clusters_per_panel=8,
    colors=None,
    figsize=None,
    max_points_per_label=200,
    rng=None,
    hlines=None,
):
    """Headless (savefig, no plt.show) sibling of show_label_points_grid: dumps
    the per-cluster activation-range scatter grid to a dark-themed JPEG, sizing
    the grid from `clusters_per_panel` (see grid_dims_for_labels) so the caller
    only supplies the label->points map. Styled to match the rest of the report
    (dark axes + dark legends) so it reads cleanly as a tab of
    html_report.generate_html_report.

    label_by_points: dict[label, list[float]] — {cluster_label: [output
    activations]}. To get an "above noise only" view, the caller just passes the
    already-filtered subset (e.g. noise_overlay's second return value) as its own
    call — a separate image / separate tab — rather than this function composing
    it. hlines: optional horizontal reference lines drawn on every active panel
    (see _draw_hlines); for the noise view the caller passes noise_overlay's
    hlines. Both omitted leaves the grid exactly as before."""
    n_rows, n_cols = grid_dims_for_labels(len(label_by_points), clusters_per_panel)
    fig, axes = show_label_points_grid(
        label_by_points,
        n_rows,
        n_cols,
        colors=colors,
        figsize=figsize,
        max_points_per_label=max_points_per_label,
        rng=rng,
        hlines=hlines,
    )
    _style_dark_panels_and_legends(fig, axes)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path, format="jpeg", facecolor=fig.get_facecolor(), pil_kwargs={"quality": 90}
    )
    plt.close(fig)
