"""The sample graph drawn with PIL: one base picture, and a copy per bucket with that bucket's hits starred.

- Dots are coloured by the hit's most probable cluster, and each cluster's peak is a large labelled disc.
- Starred hits are the bucket's members in the sample, and the numbered ones match the cards under the picture.
"""

import colorsys
import math

import numpy as np
from jaxtyping import Float, Int
from PIL import Image, ImageDraw, ImageFont

CANVAS = 1400
MARGIN = 60
DOT_RADIUS = 5
PEAK_RADIUS = 15
STAR_RADIUS = 11
STAR_FILL = (255, 235, 0)
GOLDEN_RATIO_STEP = 0.618033988749895
FONT_SIZE = 20


def cluster_colour(cluster: int) -> tuple[int, int, int]:
    """Neighbouring ids get hues a golden-ratio step apart, so they stay distinguishable."""
    r, g, b = colorsys.hsv_to_rgb((cluster * GOLDEN_RATIO_STEP) % 1.0, 0.65, 0.85)
    return int(r * 255), int(g * 255), int(b * 255)


def to_canvas(points: Float[np.ndarray, "m 2"]) -> Float[np.ndarray, "m 2"]:
    low, high = points.min(axis=0), points.max(axis=0)
    return MARGIN + (points - low) / np.maximum(high - low, 1e-9) * (CANVAS - 2 * MARGIN)


def star_polygon(x: float, y: float, radius: float) -> list[tuple[float, float]]:
    """Ten vertices alternating between the outer radius and 0.45 of it, starting at the top."""
    return [
        (x + r * math.sin(i * math.pi / 5), y - r * math.cos(i * math.pi / 5))
        for i, r in ((i, radius if i % 2 == 0 else radius * 0.45) for i in range(10))
    ]


def font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=FONT_SIZE)


def draw_base(
    points: Float[np.ndarray, "m 2"], labels: Int[np.ndarray, "m"], peak_nodes: dict[int, int],
) -> Image.Image:
    """`peak_nodes` maps a cluster id to the node of its peak."""
    canvas_points = to_canvas(points)
    image = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for (x, y), label in zip(canvas_points, labels):
        draw.ellipse([x - DOT_RADIUS, y - DOT_RADIUS, x + DOT_RADIUS, y + DOT_RADIUS], fill=cluster_colour(int(label)))
    for cluster, node in peak_nodes.items():
        x, y = canvas_points[node]
        draw.ellipse([x - PEAK_RADIUS, y - PEAK_RADIUS, x + PEAK_RADIUS, y + PEAK_RADIUS],
                     fill=cluster_colour(cluster), outline=(0, 0, 0), width=2)
        draw.text((x, y), str(cluster), fill=(0, 0, 0), font=font(), anchor="mm", stroke_width=2, stroke_fill=(255, 255, 255))
    return image


def draw_bucket(
    base: Image.Image, points: Float[np.ndarray, "m 2"], starred: list[int], numbered: list[int],
) -> Image.Image:
    """Stars for every `starred` node, and a number beside each of `numbered`, in card order."""
    canvas_points = to_canvas(points)
    image = base.copy()
    draw = ImageDraw.Draw(image)
    for node in starred:
        x, y = canvas_points[node]
        draw.polygon(star_polygon(x, y, STAR_RADIUS), fill=STAR_FILL, outline=(0, 0, 0))
    for number, node in enumerate(numbered, start=1):
        x, y = canvas_points[node]
        draw.text((x + STAR_RADIUS, y - STAR_RADIUS), str(number), fill=(0, 0, 0), font=font(),
                  anchor="lb", stroke_width=3, stroke_fill=(255, 255, 255))
    return image
