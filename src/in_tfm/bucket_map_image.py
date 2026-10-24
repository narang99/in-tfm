"""The sample graph drawn with PIL: one base picture, and a copy per bucket with that bucket's hits starred.

- Dots are coloured by the hit's most probable cluster, and each cluster's peak is a large labelled disc.
- Starred hits are the bucket's members in the sample, and the numbered ones match the cards under the picture.
"""

import colorsys
import math

import numpy as np
from jaxtyping import Float, Int
from PIL import Image, ImageDraw, ImageFont

NORMAL_CANVAS = 1400
HUGE_CANVAS = 5600
"""Drawn separately and linked, so the browser can zoom into a crowded corner without loading it with the page."""
STAR_FILL = (255, 235, 0)
GOLDEN_RATIO_STEP = 0.618033988749895
FONT_SIZE = 20
MARGIN = 60
DOT_RADIUS = 5
PEAK_RADIUS = 15
STAR_RADIUS = 11


def cluster_colour(cluster: int) -> tuple[int, int, int]:
    """Neighbouring ids get hues a golden-ratio step apart, so they stay distinguishable."""
    r, g, b = colorsys.hsv_to_rgb((cluster * GOLDEN_RATIO_STEP) % 1.0, 0.65, 0.85)
    return int(r * 255), int(g * 255), int(b * 255)


def to_canvas(points: Float[np.ndarray, "m 2"], canvas: int) -> Float[np.ndarray, "m 2"]:
    margin = scaled(MARGIN, canvas)
    low, high = points.min(axis=0), points.max(axis=0)
    return margin + (points - low) / np.maximum(high - low, 1e-9) * (canvas - 2 * margin)


def star_polygon(x: float, y: float, radius: float) -> list[tuple[float, float]]:
    """Ten vertices alternating between the outer radius and 0.45 of it, starting at the top."""
    return [
        (x + r * math.sin(i * math.pi / 5), y - r * math.cos(i * math.pi / 5))
        for i, r in ((i, radius if i % 2 == 0 else radius * 0.45) for i in range(10))
    ]


def scaled(pixels: float, canvas: int) -> int:
    """`pixels` is for a normal canvas, and sizes grow with the canvas."""
    return max(1, round(pixels * canvas / NORMAL_CANVAS))


def font(canvas: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=scaled(FONT_SIZE, canvas))


def draw_base(
    points: Float[np.ndarray, "m 2"], labels: Int[np.ndarray, "m"], peak_nodes: dict[int, int], canvas: int,
) -> Image.Image:
    """`peak_nodes` maps a cluster id to the node of its peak."""
    canvas_points = to_canvas(points, canvas)
    dot, peak = scaled(DOT_RADIUS, canvas), scaled(PEAK_RADIUS, canvas)
    image = Image.new("RGB", (canvas, canvas), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for (x, y), label in zip(canvas_points, labels):
        draw.ellipse([x - dot, y - dot, x + dot, y + dot], fill=cluster_colour(int(label)))
    for cluster, node in peak_nodes.items():
        x, y = canvas_points[node]
        draw.ellipse([x - peak, y - peak, x + peak, y + peak],
                     fill=cluster_colour(cluster), outline=(0, 0, 0), width=scaled(2, canvas))
        draw.text((x, y), str(cluster), fill=(0, 0, 0), font=font(canvas), anchor="mm",
                  stroke_width=scaled(2, canvas), stroke_fill=(255, 255, 255))
    return image


def draw_groups(
    points: Float[np.ndarray, "m 2"], dot_colours: list[tuple[int, int, int]], draw_order: Int[np.ndarray, "m"],
    spheres: dict[int, int], canvas: int,
) -> Image.Image:
    """Nodes are drawn in `draw_order`, so later ones sit on top.

    `spheres` maps a group number to the node where it is drawn as a numbered disc, in that group's colour."""
    canvas_points = to_canvas(points, canvas)
    dot, peak = scaled(DOT_RADIUS, canvas), scaled(PEAK_RADIUS, canvas)
    image = Image.new("RGB", (canvas, canvas), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for node in draw_order:
        x, y = canvas_points[node]
        draw.ellipse([x - dot, y - dot, x + dot, y + dot], fill=dot_colours[node])
    for number, node in spheres.items():
        x, y = canvas_points[node]
        draw.ellipse([x - peak, y - peak, x + peak, y + peak],
                     fill=cluster_colour(number), outline=(0, 0, 0), width=scaled(2, canvas))
        draw.text((x, y), str(number), fill=(0, 0, 0), font=font(canvas), anchor="mm",
                  stroke_width=scaled(2, canvas), stroke_fill=(255, 255, 255))
    return image


def draw_bucket(
    base: Image.Image, points: Float[np.ndarray, "m 2"], starred: list[int], numbered: list[int],
) -> Image.Image:
    """Stars for every `starred` node, and a number beside each of `numbered`, in card order. Sized like `base`."""
    canvas = base.width
    canvas_points = to_canvas(points, canvas)
    star = scaled(STAR_RADIUS, canvas)
    image = base.copy()
    draw = ImageDraw.Draw(image)
    for node in starred:
        x, y = canvas_points[node]
        draw.polygon(star_polygon(x, y, star), fill=STAR_FILL, outline=(0, 0, 0))
    for number, node in enumerate(numbered, start=1):
        x, y = canvas_points[node]
        draw.text((x + star, y - star), str(number), fill=(0, 0, 0), font=font(canvas), anchor="lb",
                  stroke_width=scaled(3, canvas), stroke_fill=(255, 255, 255))
    return image
