"""Plotting helpers for attribution maps: signed heatmaps, grids of overlays, and blending an
attribution heatmap onto its source image.
"""

import math
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from PIL import Image

from .colormaps import get_local_image_limits, rd_bk_gn, rd_wht_gn

ViewType = Literal["global", "local", "gray"]
Mode = Literal["dark", "light"]


def show_single_channel_red_green_black(
    images: list[np.ndarray],
    figsize: int | tuple[int, int] | None = None,
    ncols: int = 2,
    axis: str = "on",
    viztype: ViewType = "global",
    mode: Mode = "dark",
    suptitle: str = "",
    ax_titles: list[str] | None = None,
) -> list[Axes] | None:
    if len(images) == 1:
        ncols = 1
    fig, axs, v_limit = _plot_single_channel_red_green_black(
        images, figsize, ncols, axis, viztype, mode, suptitle, ax_titles
    )
    return axs


def save_single_channel_red_green_black(
    images: list[np.ndarray],
    output_path: str | Path,
    figsize: int | tuple[int, int] | None = None,
    ncols: int = 2,
    axis: str = "on",
    viztype: ViewType = "global",
    mode: Mode = "dark",
    suptitle: str = "",
    ax_titles: list[str] | None = None,
) -> None:
    """Headless sibling of show_single_channel_red_green_black: savefig + close instead of
    returning axes, for dumping report assets rather than notebook display."""
    if len(images) == 1:
        ncols = 1
    fig, _, _ = _plot_single_channel_red_green_black(
        images, figsize, ncols, axis, viztype, mode, suptitle, ax_titles
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_single_channel_red_green_black(
    images: list[np.ndarray],
    figsize: int | tuple[int, int] | None,
    ncols: int,
    axis: str,
    viztype: ViewType,
    mode: Mode,
    suptitle: str,
    ax_titles: list[str] | None,
) -> tuple[Figure, list[Axes], float] | tuple[None, None, None]:
    if not images:
        return None, None, None

    if images[0].dtype == np.uint8:
        images = [img.astype(np.float32) for img in images]

    all_min = min(img.min() for img in images)
    all_max = max(img.max() for img in images)
    v_limit = max(abs(all_min), abs(all_max))

    images = list(images)
    rows = math.ceil(len(images) / ncols)
    if figsize is None:
        figsize = (5 * rows, 5 * rows)
    elif isinstance(figsize, int):
        figsize = (figsize, figsize)

    fig, axs = plt.subplots(rows, ncols, figsize=figsize)
    fig.suptitle(suptitle)
    axs = axs.flatten() if len(images) > 1 else [axs]

    for i, img in enumerate(images):
        params = {}
        if viztype == "gray":
            params["cmap"] = "gray"
        else:
            params["cmap"] = rd_bk_gn if mode == "dark" else rd_wht_gn
            if viztype == "global":
                params["vmin"], params["vmax"] = -v_limit, v_limit
            elif viztype == "local":
                params["vmin"], params["vmax"] = get_local_image_limits(img)
            else:
                raise ValueError(f"invalid viztype {viztype}")

        axs[i].imshow(img, **params)
        axs[i].axis(axis)

    if ax_titles is not None:
        for i in range(len(axs)):
            if i < len(ax_titles):
                axs[i].set_title(ax_titles[i])

    plt.tight_layout()
    return fig, axs, v_limit


def show_grid(
    image_list: list[np.ndarray], rows: int, cols: int, ax_titles: list[str] | None = None
) -> None:
    if not image_list:
        print("empty image list, nothing to show")
        return
    ax_titles = ax_titles or []
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for i, img in enumerate(image_list):
        if i < len(axes):
            axes[i].imshow(img)
            axes[i].axis("off")  # hide the x/y tick coordinates
            if len(ax_titles) > i:
                axes[i].set_title(ax_titles[i])

    for j in range(len(image_list), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()


def compose_grid(
    images: list[Image.Image],
    cols: int | None = None,
    pad: int = 4,
    bg: tuple[int, int, int] = (0, 0, 0),
    border: int = 0,
) -> Image.Image:
    """Pastes same-size PIL images into a grid canvas.

    `pad` separates tiles from each other; `border` is the extra margin around the whole grid.

    No matplotlib figure/axes involved, so this is the preferred way to batch images together
    for headless report generation - matplotlib is reserved for interactive notebook display.
    """
    cols = cols or len(images)
    rows = math.ceil(len(images) / cols)
    w, h = images[0].size
    canvas = Image.new(
        "RGB", (cols * (w + pad) - pad + 2 * border, rows * (h + pad) - pad + 2 * border), bg
    )
    for i, img in enumerate(images):
        row, col = divmod(i, cols)
        canvas.paste(img, (border + col * (w + pad), border + row * (h + pad)))
    return canvas


def save_image_grid(
    images: list[Image.Image],
    output_path: str | Path,
    cols: int | None = None,
    pad: int = 4,
    border: int = 0,
) -> None:
    if not images:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compose_grid(images, cols, pad=pad, border=border).save(output_path)


def render_hadamard_tiles(
    patches: list[Float[np.ndarray, "h w"]], size: tuple[int, int] = (150, 150)
) -> list[Image.Image]:
    """Colors raw Hadamard patches red/black/green, same convention as
    show_single_channel_red_green_black's "global" viztype (shared vmin/vmax across all
    patches, symmetric about 0) - but rendered directly as PIL tiles for save_image_grid.
    """
    v_limit = max(np.abs(p).max() for p in patches)
    return [apply_cmap(p, rd_bk_gn, vmin=-v_limit, vmax=v_limit, size=size) for p in patches]


def save_elbow_plot(values: Float[np.ndarray, "n"], elbow_idx: int, output_path: str | Path) -> None:
    """Line plot of sorted positive activations with the chosen elbow cutoff marked - a genuine
    chart (axes, annotation), not an image composite, so matplotlib is the right tool here even
    though this runs headlessly during report generation."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(values)
    ax.axvline(elbow_idx, color="red", linestyle="--", label=f"threshold={values[elbow_idx]:.3f}")
    ax.set_xlabel("sorted index")
    ax.set_ylabel("activation")
    ax.legend()
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def to_pil(img: np.ndarray | torch.Tensor, size: tuple[int, int]) -> Image.Image:
    """Accepts either (H,W,3) or (3,H,W) - channel-first is transposed before the rest."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.shape[0] == 3:  # (3,H,W) -> (H,W,3)
        img = img.transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img).resize(size, Image.BILINEAR)


def apply_cmap(
    arr: Float[np.ndarray, "h w"],
    cmap: Colormap,
    vmin: float,
    vmax: float,
    size: tuple[int, int],
    interpolation: int = Image.NEAREST,
) -> Image.Image:
    arr = np.clip((arr - vmin) / (vmax - vmin + 1e-8), 0, 1)
    rgba = (cmap(arr) * 255).astype(np.uint8)  # cmap returns (H,W,4)
    return Image.fromarray(rgba, mode="RGBA").convert("RGB").resize(size, interpolation)


def mk_overlay(
    inv_img: np.ndarray,
    neuron_att: Float[torch.Tensor, "h w"],
    size: tuple[int, int],
    cmap: Colormap = rd_bk_gn,
    alpha: float = 0.8,
) -> Image.Image:
    """`inv_img` must already be denormalized (e.g. via inv_tfm) and `neuron_att` must already
    have its channel axis collapsed - attr_fn implementations (neuron_ig,
    compute_attnlrp_relevance) are responsible for that, not this function.
    """
    overlay = neuron_att.detach().cpu().numpy()
    overlay = overlay / np.abs(overlay).max()
    base = to_pil(inv_img, size=size)
    heat = apply_cmap(overlay, cmap, vmin=-1, vmax=1, size=size, interpolation=Image.BILINEAR)
    return Image.blend(base, heat, alpha=alpha)
