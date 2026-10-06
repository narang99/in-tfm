import numpy as np
import torch

from in_tfm.presenters.base import ClusterHit
from in_tfm.presenters.text import TextPresenter
from in_tfm.presenters.text.colored_tokens import colored_tokens


class FakeTokenizer:
    def convert_ids_to_tokens(self, ids):
        return [f"t{i}" for i in ids.tolist()]


class FakeSource:
    tokenizer = FakeTokenizer()


def hit_at(token_idx: int, length: int = 60) -> ClusterHit:
    relevance = np.zeros((1, length, 2), dtype=np.float32)
    relevance[0, 0] = 5.0
    relevance[0, token_idx] += 1.0
    return ClusterHit(
        sample_id="s",
        token_idx=token_idx,
        relevance=relevance,
        model_input=torch.zeros(1, length, 2),
        hadamard=np.zeros(4),
        display_ids=torch.arange(length),
    )


def test_relevance_runs_from_the_start_to_the_firing_token():
    presenter = TextPresenter(FakeSource())
    assert len(presenter._relevance_up_to_firing(hit_at(30))) == 31
    assert len(presenter._relevance_up_to_firing(hit_at(0))) == 1


def test_top_relevance_can_leave_out_position_zero():
    presenter = TextPresenter(FakeSource(), top_tokens=2)
    tokens = [f"t{i}" for i in range(4)]
    relevance = np.array([9.0, 1.0, 3.0, 2.0])
    with_sink = presenter._top_relevance(tokens, relevance, first=0)
    without_sink = presenter._top_relevance(tokens, relevance, first=1)
    assert "t0" in with_sink
    assert "t0" not in without_sink
    assert without_sink.index("t2") < without_sink.index("t3")


def test_each_token_carries_colors_for_both_views():
    html = colored_tokens(["a", "b"], np.array([5.0, 1.0]), vmax=5.0, vmax_without_first=1.0, firing_idx=1)
    assert html.count("--x-bg-l") == 2
    assert html.count("--bg-l") == 2
    assert 'class="tok tok-first"' in html
    assert 'class="tok tok-firing"' in html


def test_the_report_shows_both_scales_in_one_page():
    html = TextPresenter(FakeSource())._hit_block(hit_at(3), np.array([5.0, 0.0, 0.0, 1.0]), 5.0, 1.0)
    assert "when-sink" in html and "when-no-sink" in html
