"""A writer firing's Hadamard row beside its nearest clustered hit's, as one small PNG.

- A conv row is `(channels, k, k)` flattened channel-major, as `F.unfold` lays it out.
  - So it is drawn one k x k tile per channel, never folded into one near-square grid.
  - Folding mixes channels into one tile, the open bug in the neuron reports' Hadamard tiles.
- Both rows share one colour scale, symmetric about 0, so the pair compares directly.
- PIL, not matplotlib, since a page holds many of these.
"""

import base64
import io

import numpy as np
from jaxtyping import Float
from PIL import Image

from ..colormaps import rd_bk_gn
from ..viz import apply_cmap, compose_grid

TILE_PX = 54
"""Per k x k tile, so a 3 x 3 cell is 18 px."""
GAP_PX = 3
PAIR_GAP_PX = 12


def channel_tiles(
    row: Float[np.ndarray, "n"], tile_shape: tuple[int, ...]
) -> list[Float[np.ndarray, "h w"]]:
    if len(tile_shape) == 3:
        return list(row.reshape(tile_shape))
    return [row.reshape(1, -1)]


def tile_size(tile: Float[np.ndarray, "h w"]) -> tuple[int, int]:
    height, width = tile.shape
    cell = TILE_PX // max(height, 1) if height > 1 else 12
    return width * cell, height * cell


def row_image(
    row: Float[np.ndarray, "n"], tile_shape: tuple[int, ...], v_limit: float
) -> Image.Image:
    tiles = [
        apply_cmap(tile, rd_bk_gn, vmin=-v_limit, vmax=v_limit, size=tile_size(tile))
        for tile in channel_tiles(row, tile_shape)
    ]
    return compose_grid(tiles, pad=GAP_PX)


def side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    return compose_grid([left, right], pad=PAIR_GAP_PX)


def png_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def pair_uri(
    wild: list[float], closest: list[float], tile_shape: tuple[int, ...]
) -> str:
    wild_row, closest_row = np.asarray(wild), np.asarray(closest)
    v_limit = max(np.abs(wild_row).max(), np.abs(closest_row).max(), 1e-8)
    return png_data_uri(
        side_by_side(
            row_image(wild_row, tile_shape, v_limit),
            row_image(closest_row, tile_shape, v_limit),
        )
    )
