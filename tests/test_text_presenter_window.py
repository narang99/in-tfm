import numpy as np
import torch

from in_tfm.html_report import render_template
from in_tfm.presenters import ClusterArtifacts
from in_tfm.presenters.base import ClusterHit
from in_tfm.presenters.text import TextPresenter
from in_tfm.presenters.text.colored_tokens import token_views
from in_tfm.presenters.text.views import HitView, SinkViews


class FakeTokenizer:
    def convert_ids_to_tokens(self, ids):
        return [f"t{i}" for i in ids.tolist()]


class FakeSource:
    tokenizer = FakeTokenizer()


def scale(with_sink: float, without_sink: float) -> SinkViews[float]:
    return SinkViews(with_sink=with_sink, without_sink=without_sink)


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
    with_sink = [chip.text for chip in presenter._top_relevance(tokens, relevance, first=0)]
    without_sink = [chip.text for chip in presenter._top_relevance(tokens, relevance, first=1)]
    assert "t0" in with_sink
    assert without_sink == ["t2", "t3"]


def test_each_token_carries_colors_for_both_views():
    tokens = token_views(["a", "b"], np.array([5.0, 1.0]), scale(5.0, 1.0), firing_idx=1)
    assert [t.first for t in tokens] == [True, False]
    assert [t.firing for t in tokens] == [False, True]
    assert tokens[1].colors.with_sink != tokens[1].colors.without_sink


def test_token_html_has_both_color_sets_and_marks_first_and_firing():
    hit_view = TextPresenter(FakeSource())._hit_view(hit_at(1, length=2), np.array([5.0, 1.0]), scale(5.0, 1.0))
    html = render_template("text_hit.html", hit=hit_view)
    assert html.count("--x-bg-l") == 2
    assert html.count("--bg-l") == 2
    assert 'class="tok tok-first"' in html
    assert 'class="tok tok-firing"' in html


def test_the_report_shows_both_scales_in_one_page():
    hit_view = TextPresenter(FakeSource())._hit_view(hit_at(3), np.array([5.0, 0.0, 0.0, 1.0]), scale(5.0, 1.0))
    html = render_template("text_hit.html", hit=hit_view)
    assert "when-sink" in html and "when-no-sink" in html


def test_render_writes_artifacts_and_links_them_relative_to_the_report(tmp_path):
    cluster_dir = tmp_path / "cluster_4"
    cluster_dir.mkdir()
    hit = hit_at(3).model_copy(update={"hadamard": np.arange(4, dtype=np.float32) - 1.5})
    html = TextPresenter(FakeSource()).render([hit], ClusterArtifacts(cluster_dir=cluster_dir), (2, 2))
    assert 'src="cluster_4/hadamard.jpg"' in html
    assert (cluster_dir / "hadamard.jpg").exists()
    assert (cluster_dir / "hits.html").exists()
    assert "1 sampled hits in context" in html


def test_token_text_is_escaped():
    tokens = token_views(["<b>"], np.array([1.0]), scale(1.0, 1.0))
    hit_view = HitView(
        sample_id="s", token_idx=0, tokens=tokens, top_relevance=SinkViews(with_sink=[], without_sink=[])
    )
    html = render_template("text_hit.html", hit=hit_view)
    assert "&lt;b&gt;" in html
    assert "<b>" not in html
