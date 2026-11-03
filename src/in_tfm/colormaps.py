"""Diverging red/black/green colormaps for signed attribution maps.

Red-black-green (rather than the usual red-blue) keeps zero-relevance pixels visually "off"
against a dark image background instead of a washed-out mid-tone; `set_bad("yellow")` flags
NaNs instead of silently rendering them as zero.
"""

import matplotlib.colors as mcolors

rd_bk_gn: mcolors.LinearSegmentedColormap = mcolors.LinearSegmentedColormap.from_list(
    "RdBkGn", ["#FF3131", "#333333", "#39FF14"]
)
rd_bk_gn.set_bad("yellow")

token_cmap_light: mcolors.LinearSegmentedColormap = mcolors.LinearSegmentedColormap.from_list(
    "TokenLight", ["#B22222", "#FFFFFF", "#00A550"]
)
token_cmap_dark: mcolors.LinearSegmentedColormap = mcolors.LinearSegmentedColormap.from_list(
    "TokenDark", ["#FF3131", "#1E1E1E", "#39FF14"]
)
"""Same red/green endpoints as the image maps, but centered on the html report's page color
instead of grey/black: a near-zero token then reads as ordinary unhighlighted text rather than
as a filled box, which is what makes a heatmap over running text legible."""
