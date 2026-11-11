import numpy as np
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

from in_tfm.qk_pairs import (
    attention_of,
    capture_query_key,
    causal_pairs,
    elbow_pairs,
    gather_rows,
    head_scores,
    scan_pairs,
    worst_gather_mismatch,
)
from in_tfm.sources import ModelBatch

LAYER = 1
SEQ = 12


def tiny_qwen3() -> Qwen3Model:
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=50,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        attn_implementation="eager",
    )
    return Qwen3Model(config).eval()


class RandomTokens:
    """The last two samples are right-padded, so padding has to be masked out."""

    def __init__(self, model: Qwen3Model, n_samples: int = 6) -> None:
        generator = torch.Generator().manual_seed(1)
        self.ids = torch.randint(0, 50, (n_samples, SEQ), generator=generator)
        self.mask = torch.ones(n_samples, SEQ, dtype=torch.long)
        self.mask[-2:, -3:] = 0
        self.embed = model.embed_tokens

    def sample_ids(self) -> list[str]:
        return [str(i) for i in range(len(self.ids))]

    def to_model_batch(self, ids) -> ModelBatch:
        rows = [int(i) for i in ids]
        with torch.no_grad():
            embeds = self.embed(self.ids[rows])
        return ModelBatch(
            kwargs={"inputs_embeds": embeds, "attention_mask": self.mask[rows]},
            grad_leaf_key="inputs_embeds",
            valid_mask=self.mask[rows].bool(),
        )


def test_softmax_of_the_scores_is_the_models_attention():
    model = tiny_qwen3()
    source = RandomTokens(model)
    batch = source.to_model_batch(source.sample_ids())
    with torch.no_grad():
        attentions = model(**batch.kwargs, output_attentions=True).attentions[LAYER]
    query_key = capture_query_key(model, LAYER, batch)
    pairs = causal_pairs(batch.valid_mask)
    for head in range(4):
        scores = head_scores(query_key, attention_of(model, LAYER), head)
        weights = scores.masked_fill(~pairs, -torch.inf).softmax(-1)
        real_queries = batch.valid_mask
        torch.testing.assert_close(
            weights[real_queries],
            attentions[:, head][real_queries],
            atol=1e-5,
            rtol=1e-4,
        )


def test_gathered_rows_sum_to_the_scanned_scores():
    model = tiny_qwen3()
    source = RandomTokens(model)
    scan = scan_pairs(model, source, LAYER, [0, 3], batch_size=4, device="cpu")
    hits = [elbow_pairs(scan, head, 10, seed=0, device="cpu") for head in [0, 3]]
    gathered = gather_rows(model, source, LAYER, scan, hits, device="cpu")
    for pair_rows in gathered:
        assert len(pair_rows.hits.sample_idx) > 0
        assert worst_gather_mismatch(pair_rows) < 2e-3
        assert (pair_rows.hits.key_idx <= pair_rows.hits.query_idx).all()


def test_padding_and_future_keys_never_become_hits():
    model = tiny_qwen3()
    source = RandomTokens(model)
    scan = scan_pairs(model, source, LAYER, [1], batch_size=6, device="cpu")
    hits = elbow_pairs(scan, 1, 10_000, seed=0, device="cpu")
    padded = source.mask.numpy() == 0
    assert not padded[hits.sample_idx, hits.key_idx].any()
    assert not padded[hits.sample_idx, hits.query_idx].any()
    assert np.all(hits.key_idx <= hits.query_idx)


def test_the_report_page_marks_key_and_query(tmp_path):
    from in_tfm.qk_report import PairReportInputs, write_pair_report

    model = tiny_qwen3()
    source = RandomTokens(model)
    scan = scan_pairs(model, source, LAYER, [0], batch_size=6, device="cpu")
    hits = elbow_pairs(scan, 0, 10_000, seed=0, device="cpu")
    pair_rows = gather_rows(model, source, LAYER, scan, [hits], device="cpu")[0]
    labels = np.arange(len(hits.sample_idx)) % 2
    path = write_pair_report(
        tmp_path,
        "tiny head 0",
        PairReportInputs(
            pair_rows=pair_rows,
            labels=labels,
            sample_ids=scan.sample_ids,
            token_ids=source.ids,
        ),
        decode=lambda ids: f"t{ids[0]}",
        pad_id=-1,
        examples_per_cluster=3,
        context=2,
        seed=0,
    )
    html = path.read_text()
    assert "cluster 0" in html and "cluster 1" in html
    assert 'class="qk-key"' in html or 'class="qk-both"' in html
    assert (tmp_path / "pairs.pt").exists()


def test_excluding_the_first_key_drops_every_pair_that_attends_to_it():
    model = tiny_qwen3()
    source = RandomTokens(model)
    scan = scan_pairs(model, source, LAYER, [2], batch_size=4, device="cpu")
    kept_all = elbow_pairs(scan, 2, 10_000, seed=0, device="cpu")
    without_first = elbow_pairs(
        scan, 2, 10_000, seed=0, device="cpu", exclude_first_key=True
    )
    first = scan.first_real.numpy()
    assert (kept_all.key_idx == first[kept_all.sample_idx]).any()
    assert not (without_first.key_idx == first[without_first.sample_idx]).any()
