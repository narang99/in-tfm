"""One colour per cluster id, shared by the cluster cards and every chart, so a cluster looks the same everywhere.

- The palette is Kelly's 22 colours of maximum contrast, without white, which marks noise.
- Vivid red and the greens are left out, since red and green mean relevance sign inside the hits.
"""

KELLY_COLORS = [
    "#F2F3F4", "#F3C300", "#875692", "#F38400", "#A1CAF1", "#BE0032", "#C2B280", "#848482", "#008856", "#E68FAC",
    "#0067A5", "#F99379", "#604E97", "#F6A600", "#B3446C", "#DCD300", "#882D17", "#8DB600", "#654522", "#E25822",
    "#2B3D26",
]  # fmt: skip
"""Index 0 is white, reserved for noise."""
RED_AND_GREEN = {"#BE0032", "#008856", "#8DB600", "#2B3D26"}
CLUSTER_COLORS = [c for c in KELLY_COLORS[1:] if c not in RED_AND_GREEN]
DARK_INK = "#1a1a1a"
LIGHT_INK = "#ffffff"


def cluster_color(cluster_id: int) -> str:
    return CLUSTER_COLORS[int(cluster_id) % len(CLUSTER_COLORS)]


def relative_luminance(color: str) -> float:
    """WCAG relative luminance of a `#rrggbb` colour."""
    channels = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def ink_on(color: str) -> str:
    """Dark text on light colours like Kelly's yellows, white on the rest."""
    return DARK_INK if relative_luminance(color) > 0.35 else LIGHT_INK


def cluster_style(cluster_id: int) -> str:
    """CSS custom properties for a cluster's card or nav link."""
    color = cluster_color(cluster_id)
    return f"--accent:{color};--accent-ink:{ink_on(color)}"
