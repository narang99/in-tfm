"""Colours for `report_templates/concept_cards.html`, taken from the project's diverging maps."""

from matplotlib.colors import to_hex

from ...colormaps import rd_bk_gn, rd_wht_gn, token_cmap_dark, token_cmap_light


def sign_colors() -> dict[str, str]:
    """The ends of the grid maps, so a card's border reads like a heatmap cell."""
    return {"pos_light": to_hex(rd_wht_gn(1.0)), "neg_light": to_hex(rd_wht_gn(0.0)),
            "pos_dark": to_hex(rd_bk_gn(1.0)), "neg_dark": to_hex(rd_bk_gn(0.0))}


def token_stops() -> dict[str, list[str]]:
    """The three stops of each token map. The page interpolates between them as matplotlib does."""
    return {"light": [to_hex(token_cmap_light(x)) for x in (0.0, 0.5, 1.0)],
            "dark": [to_hex(token_cmap_dark(x)) for x in (0.0, 0.5, 1.0)]}


def concept_card_context() -> dict[str, dict]:
    """What `concept_cards.html` needs from the page's template context."""
    return {"colors": sign_colors(), "token_stops": token_stops()}
