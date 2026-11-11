"""A dependency's place on its sign's concentration curve, as a tiny inline SVG.

- SVG from the template, not Plotly, since a page can hold a hundred cards.
- Both signs share one y scale, so a sign with more total magnitude stands taller.
"""

from pydantic import BaseModel

from ..dependencies.stats import Concentration, ConcentrationMarker

WIDTH, HEIGHT, HALF_GAP = 160, 36, 6


class Sparkline(BaseModel):
    positive_points: str
    negative_points: str
    marker_x: float | None
    marker: ConcentrationMarker | None
    width: int = WIDTH
    height: int = HEIGHT

    @property
    def label(self) -> str:
        if self.marker is None:
            return "median contribution is zero"
        sign = "+" if self.marker.sign > 0 else "−"
        return (
            f"{sign} rank {self.marker.rank}: {self.marker.cumulative_share:.0%} "
            f"(+{self.marker.delta:.0%})"
        )


def half_width() -> float:
    return WIDTH / 2 - HALF_GAP


def polyline(curve: list[float], x0: float, y_max: float) -> str:
    """Rank on x within its half, cumulative magnitude on y, starting from zero."""
    if not curve:
        return ""
    step = half_width() / len(curve)
    points = [(x0, HEIGHT)] + [
        (x0 + (i + 1) * step, HEIGHT - HEIGHT * value / y_max)
        for i, value in enumerate(curve)
    ]
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def marker_x(
    marker: ConcentrationMarker | None, concentration: Concentration
) -> float | None:
    if marker is None:
        return None
    curve = concentration.positive if marker.sign > 0 else concentration.negative
    x0 = 0.0 if marker.sign > 0 else WIDTH / 2 + HALF_GAP
    return x0 + marker.rank * half_width() / len(curve)


def sparkline(concentration: Concentration, dim: int) -> Sparkline:
    ends = [c[-1] for c in (concentration.positive, concentration.negative) if c]
    y_max = max(ends, default=1.0) or 1.0
    marker = concentration.markers.get(dim)
    return Sparkline(
        positive_points=polyline(concentration.positive, 0.0, y_max),
        negative_points=polyline(concentration.negative, WIDTH / 2 + HALF_GAP, y_max),
        marker_x=marker_x(marker, concentration),
        marker=marker,
    )
