import pytest
from pydantic import BaseModel

from in_tfm.dependencies.view import CumulativeShareTrim, NoTrim, ViewPolicy, view


class Item(BaseModel):
    name: str
    share: float


ITEMS = [Item(name=n, share=s) for n, s in [("c", 1.0), ("a", 6.0), ("b", 3.0)]]


def shown_and_rest(policy: ViewPolicy) -> tuple[list[str], list[str]]:
    trimmed = view(ITEMS, policy, lambda i: i.share, lambda i: i.name)
    return [r.label for r in trimmed.shown], [r.label for r in trimmed.rest]


def test_rows_are_ranked_by_share_and_trimmed_at_the_cumulative_mass():
    assert shown_and_rest(ViewPolicy()) == (["a", "b"], ["c"])
    assert shown_and_rest(ViewPolicy(trim=CumulativeShareTrim(mass=0.6))) == (
        ["a"],
        ["b", "c"],
    )


def test_nothing_is_hidden_without_a_trim():
    assert shown_and_rest(ViewPolicy(trim=NoTrim())) == (["a", "b", "c"], [])


def test_the_rest_keeps_its_share_and_the_policy_describes_itself():
    trimmed = view(ITEMS, ViewPolicy(), lambda i: i.share, lambda i: i.name)
    assert trimmed.rest_share == 1.0
    assert (
        trimmed.description
        == "ranked by share, not grouped, shown until 90% of the total share"
    )


def test_a_trim_is_picked_by_its_method_name_in_a_config():
    policy = ViewPolicy.model_validate({"trim": {"method": "none"}})
    assert policy.trim == NoTrim()
    with pytest.raises(ValueError):
        ViewPolicy.model_validate({"trim": {"method": "top_k"}})
