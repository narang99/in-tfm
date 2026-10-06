"""What the text report's templates read.

Presenters decide every value here, and the templates only lay them out.
Nothing is html-escaped on the way in, since the templates escape on the way out.
"""

from pydantic import BaseModel


class TokenColors(BaseModel):
    bg_light: str
    fg_light: str
    bg_dark: str
    fg_dark: str


class TokenView(BaseModel):
    text: str
    title: str
    firing: bool
    first: bool
    colors: TokenColors
    colors_without_first: TokenColors


class RelevanceChip(BaseModel):
    text: str
    value: float


class FiringChip(BaseModel):
    text: str
    count: int


class HitView(BaseModel):
    sample_id: str
    token_idx: int
    tokens: list[TokenView]
    top_relevance: list[RelevanceChip]
    top_relevance_without_first: list[RelevanceChip]


class TextClusterView(BaseModel):
    firing_tokens: list[FiringChip]
    vmax: float
    vmax_without_first: float
    hits: list[HitView]
    clustered_label: str
    hadamard_url: str
