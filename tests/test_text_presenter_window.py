import numpy as np
import torch

from in_tfm.presenters.base import ClusterHit
from in_tfm.presenters.text import GAP_MARKER, TextPresenter


class FakeTokenizer:
    def convert_ids_to_tokens(self, ids):
        return [f"t{i}" for i in ids.tolist()]


class FakeSource:
    tokenizer = FakeTokenizer()


def hit_at(token_idx: int, length: int = 60) -> ClusterHit:
    relevance = np.zeros((1, length, 2), dtype=np.float32)
    relevance[0, 0] = 5.0
    return ClusterHit(
        sample_id="s",
        token_idx=token_idx,
        relevance=relevance,
        model_input=torch.zeros(1, length, 2),
        hadamard=np.zeros(4),
        display_ids=torch.arange(length),
    )


def test_window_far_from_the_start_keeps_position_zero_in_front():
    window = TextPresenter(FakeSource(), context_tokens=3)._window(hit_at(30))
    assert window.tokens[:2] == ["t0", GAP_MARKER]
    assert window.tokens[window.firing_pos] == "t30"
    assert window.starts_at_bos


def test_window_at_the_start_is_not_changed():
    window = TextPresenter(FakeSource(), context_tokens=3)._window(hit_at(1))
    assert window.tokens == ["t0", "t1", "t2", "t3", "t4"]
    assert window.firing_pos == 1


def test_top_relevance_lists_the_sink_last_and_not_among_the_ranked_tokens():
    presenter = TextPresenter(FakeSource(), top_tokens=2)
    html = presenter._top_relevance([f"t{i}" for i in range(4)], np.array([9.0, 1.0, 3.0, 2.0]))
    assert html.index("t2") < html.index("t3") < html.index("t0")
    assert "sink" in html


def test_a_hit_firing_at_position_zero_is_not_the_sink():
    window = TextPresenter(FakeSource(), context_tokens=3)._window(hit_at(0))
    assert not window.starts_at_bos
    assert window.scale_relevance.max() == 10.0
