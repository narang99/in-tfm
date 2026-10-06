"""What the text report's templates read.

Presenters decide every value here, and the templates only lay them out.
Nothing is html-escaped on the way in, since the templates escape on the way out.
"""

from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class SinkViews(BaseModel, Generic[T]):
    """The same thing computed twice: counting the attention sink (position 0), and leaving it out.

    The page ships both and its toggle shows one, so anything that depends on the sink is a pair.
    The `sink_views` template macro renders a pair, so the toggle's markup lives in one place.
    """

    with_sink: T
    without_sink: T


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
    colors: SinkViews[TokenColors]


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
    top_relevance: SinkViews[list[RelevanceChip]]


class TextClusterView(BaseModel):
    firing_tokens: list[FiringChip]
    scale: SinkViews[float]
    hits: list[HitView]
    clustered_label: str
    hadamard_url: str
