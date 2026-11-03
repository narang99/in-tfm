"""Plotting helpers for attribution maps: signed heatmaps, grids of overlays, and blending an
attribution heatmap onto its source image.
"""

import math
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from matplotlib.colors import Colormap
from PIL import Image

from .colormaps import rd_bk_gn


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
    """Colors raw Hadamard patches red/black/green, with one vmin/vmax shared across all
    patches and symmetric about 0."""
    v_limit = max(np.abs(p).max() for p in patches)
    return [apply_cmap(p, rd_bk_gn, vmin=-v_limit, vmax=v_limit, size=size) for p in patches]


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
    have its channel axis collapsed - attr_fn implementations (DeepLift,
    AttnLRP) are responsible for that, not this function.

    An all-zero map has nothing to scale by, so the plain image is returned instead of a NaN heat map.
    """
    overlay = neuron_att.detach().cpu().numpy()
    base = to_pil(inv_img, size=size)
    if not overlay.any():
        return base
    overlay = overlay / np.abs(overlay).max()
    heat = apply_cmap(overlay, cmap, vmin=-1, vmax=1, size=size, interpolation=Image.BILINEAR)
    return Image.blend(base, heat, alpha=alpha)
