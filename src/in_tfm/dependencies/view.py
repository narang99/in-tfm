"""What a dependency page shows, out of everything the stats hold.

- A residual stream can have hundreds of dependencies, each with many writers and writer-cluster
  combinations, so the page cannot list them all.
- A `ViewPolicy` decides, in three steps over any list of items with a share:
  - `rank` orders them
  - `group` merges them into rows
  - `trim` splits the rows into the ones shown and a collapsed rest
- Each step is its own small class, picked by `method`, so a new rule is one class and one branch.
  - `describe` names the steps, and the page prints it under every table, so a reader sees how it was cut.
"""

from collections.abc import Callable
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, Field


class ShareRank(BaseModel):
    """Largest share first, where share is fired ratio times |median contribution|."""

    method: Literal["share"] = "share"


class NoGrouping(BaseModel):
    method: Literal["none"] = "none"


class CumulativeShareTrim(BaseModel):
    """The fewest top rows whose shares reach `mass` of the total share."""

    method: Literal["cumulative_share"] = "cumulative_share"
    mass: float = Field(0.9, gt=0, le=1)


class NoTrim(BaseModel):
    method: Literal["none"] = "none"


TrimStep = Annotated[CumulativeShareTrim | NoTrim, Field(discriminator="method")]


class ViewPolicy(BaseModel):
    rank: ShareRank = ShareRank()
    group: NoGrouping = NoGrouping()
    trim: TrimStep = CumulativeShareTrim()

    def describe(self) -> str:
        return ", ".join(
            [
                describe_rank(self.rank),
                describe_group(self.group),
                describe_trim(self.trim),
            ]
        )


class Row[T](BaseModel):
    label: str
    share: float
    members: list[T]


class TrimmedView[T](BaseModel):
    shown: list[Row[T]]
    rest: list[Row[T]]
    description: str

    @property
    def rest_share(self) -> float:
        return sum(row.share for row in self.rest)


def describe_rank(step: ShareRank) -> str:
    match step:
        case ShareRank():
            return "ranked by share"


def describe_group(step: NoGrouping) -> str:
    match step:
        case NoGrouping():
            return "not grouped"


def describe_trim(step: TrimStep) -> str:
    match step:
        case CumulativeShareTrim(mass=mass):
            return f"shown until {mass:.0%} of the total share"
        case NoTrim():
            return "nothing hidden"


def ranked[T](
    items: list[T], step: ShareRank, share_of: Callable[[T], float]
) -> list[T]:
    match step:
        case ShareRank():
            return sorted(items, key=lambda item: -share_of(item))


def grouped[T](
    items: list[T],
    step: NoGrouping,
    share_of: Callable[[T], float],
    label_of: Callable[[T], str],
) -> list[Row[T]]:
    match step:
        case NoGrouping():
            return [
                Row(label=label_of(item), share=share_of(item), members=[item])
                for item in items
            ]


def n_shown(rows: list[Row], step: TrimStep) -> int:
    match step:
        case NoTrim():
            return len(rows)
        case CumulativeShareTrim(mass=mass):
            covered = np.cumsum([row.share for row in rows])
            if not len(rows) or covered[-1] == 0:
                return len(rows)
            return min(int(np.searchsorted(covered, mass * covered[-1])) + 1, len(rows))


def view[T](
    items: list[T],
    policy: ViewPolicy,
    share_of: Callable[[T], float],
    label_of: Callable[[T], str],
) -> TrimmedView[T]:
    rows = grouped(
        ranked(items, policy.rank, share_of), policy.group, share_of, label_of
    )
    cut = n_shown(rows, policy.trim)
    return TrimmedView(shown=rows[:cut], rest=rows[cut:], description=policy.describe())
