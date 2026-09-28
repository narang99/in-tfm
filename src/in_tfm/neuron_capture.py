"""Two-pass capture of a linear layer's neurons, for corpora too large to keep whole.

`activations.get_activations` keeps the layer's full input and output for every position, which
is what limits it to a few thousand short samples. A neuron's activation over all positions is
one float per position though, and the input is only needed at the positions that turn out to
be hits. So:

- `scan` streams the corpus and keeps just the requested neuron columns (exact, so the
  elbow threshold is computed exactly rather than from a sketch).
- `gather` re-runs only the samples containing hits and keeps the layer input at those
  positions.

The second pass is only sound because a sample occupies the same columns in any batch - see
sources.TextSource._encode - and `check_consistent` verifies it: the activation re-read in
pass two must match pass one.

Takes a list of neurons even when a run uses one, so scanning several costs one pass.
"""

import gc
from collections.abc import Sequence
from itertools import batched

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from nnsight.modeling.base import NNsight
from pydantic import BaseModel, ConfigDict
from sklearn.preprocessing import normalize
from tqdm import tqdm

from .activations import cat_captures
from .device import default_device, empty_cache
from .layers import LayerGetter
from .sources import SampleId, SampleSource


class ScanResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sample_ids: list[SampleId]
    neuron_idxs: list[int]
    activations: Float[torch.Tensor, "batch seq n_neurons"]
    valid_mask: Bool[torch.Tensor, "batch seq"]

    def neuron_column(self, neuron_idx: int) -> Float[torch.Tensor, "batch seq 1"]:
        """Keeps a trailing axis of width one: the threshold and hit helpers index the last
        axis of a (batch, seq, hidden) tensor, so they take `neuron_idx=0` on this."""
        column = self.neuron_idxs.index(neuron_idx)
        return self.activations[..., column : column + 1]


class GatherResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    inputs: Float[torch.Tensor, "n_hits hidden"]
    activations: Float[torch.Tensor, "n_hits n_neurons"]
    """Re-read while gathering, so it can be compared against the scan."""


class SimilarityScanResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sample_ids: list[SampleId]
    valid_mask: Bool[torch.Tensor, "batch seq"]
    similarities: dict[int, Float[torch.Tensor, "batch seq n_clusters"]]
    """Keyed by neuron_idx. Column order within a neuron's tensor matches the cluster order it
    was scanned with - see `NeuronCapture.scan_cluster_similarities`."""


class MergedPositions(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    batch_idx: Int[np.ndarray, "n_union"]
    token_idx: Int[np.ndarray, "n_union"]
    index_maps: list[Int[np.ndarray, "n_hits"]]
    """One per input selection: where each of its positions sits in the merged arrays."""


def merge_positions(
    selections: Sequence[tuple[Int[np.ndarray, "n_hits"], Int[np.ndarray, "n_hits"]]],
    seq_len: int,
) -> MergedPositions:
    """Union of several neurons' hit positions, so one gather serves them all. Neurons that
    fire on the same token would otherwise pay for the same forward pass once each."""
    keys = [batch.astype(np.int64) * seq_len + token for batch, token in selections]
    union, inverse = np.unique(np.concatenate(keys), return_inverse=True)
    boundaries = np.cumsum([len(k) for k in keys])[:-1]
    return MergedPositions(
        batch_idx=union // seq_len,
        token_idx=union % seq_len,
        index_maps=np.split(inverse, boundaries),
    )


class NeuronCapture:
    def __init__(
        self,
        model: NNsight,
        source: SampleSource,
        layer_getter: LayerGetter,
        neuron_idxs: Sequence[int],
        batch_size: int = 8,
        device: str | None = None,
    ) -> None:
        self.device = device or default_device()
        self.model = model.to(self.device)
        self.source = source
        self.layer_getter = layer_getter
        self.neuron_idxs = list(neuron_idxs)
        self.batch_size = batch_size

    def scan(self) -> ScanResult:
        ids = list(self.source.sample_ids())
        activations: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for chunk in tqdm(list(batched(ids, self.batch_size)), desc="scan"):
            _, outputs, valid_mask = self._trace(chunk)
            activations.append(outputs[..., self.neuron_idxs].cpu())
            masks.append(valid_mask.cpu())
            del outputs
            self._release()
        return ScanResult(
            sample_ids=ids,
            neuron_idxs=self.neuron_idxs,
            activations=cat_captures(activations),
            valid_mask=cat_captures(masks),
        )

    def scan_cluster_similarities(
        self,
        weight: Float[torch.Tensor, "out_hidden hidden"],
        cluster_means: dict[int, Float[np.ndarray, "n_clusters hidden"]],
    ) -> SimilarityScanResult:
        """Streams the corpus once and scores every neuron's cluster means against the
        Hadamard vector at every valid position - the distribution a cluster's inference
        threshold is fit against (see `in_tfm.inference.fit_and_patch_inference_thresholds`).

        Never keeps a Hadamard vector for more than the current batch: `layer.input` is shared
        across every neuron in `cluster_means`, so scoring several neurons' clusters still
        costs one pass, the same way `scan` scores several neurons' raw activations in one -
        each batch's Hadamard products are computed, reduced to similarity scores, and dropped
        before the next batch is traced.
        """
        normalized_means = {n: torch.from_numpy(normalize(m, "l2")) for n, m in cluster_means.items()}
        self.model = self.model.to(self.device)
        ids = list(self.source.sample_ids())
        masks: list[torch.Tensor] = []
        sims: dict[int, list[torch.Tensor]] = {n: [] for n in cluster_means}
        for chunk in tqdm(list(batched(ids, self.batch_size)), desc="similarity scan"):
            inputs, outputs, valid_mask = self._trace(chunk)
            del outputs
            # `_trace` detaches what it captures, but `weight` is a live model parameter, so the
            # multiply below would re-attach the result to the graph and keep every batch's
            # Hadamard products alive - the exact leak this streaming scan exists to avoid.
            with torch.no_grad():
                for neuron_idx, means in normalized_means.items():
                    hdmd = inputs * weight[neuron_idx].to(inputs.device)
                    hdmd = hdmd / (hdmd.norm(dim=-1, keepdim=True) + 1e-12)
                    sims[neuron_idx].append((hdmd @ means.to(inputs.device).T).cpu())
            masks.append(valid_mask.cpu())
            del inputs
            self._release()
        return SimilarityScanResult(
            sample_ids=ids,
            valid_mask=cat_captures(masks),
            similarities={n: cat_captures(s) for n, s in sims.items()},
        )

    def gather(
        self,
        batch_idx: Int[np.ndarray, "n_hits"],
        token_idx: Int[np.ndarray, "n_hits"],
    ) -> GatherResult:
        """`batch_idx` indexes into the scan's samples. Results come back in the order the
        positions were given, whatever order the samples were re-run in."""
        ids = list(self.source.sample_ids())
        samples = np.unique(batch_idx)
        order: list[np.ndarray] = []
        inputs: list[torch.Tensor] = []
        activations: list[torch.Tensor] = []
        for chunk in tqdm(list(batched(samples, self.batch_size)), desc="gather"):
            chunk = np.asarray(chunk)
            hit_rows = np.flatnonzero(np.isin(batch_idx, chunk))
            in_batch = torch.from_numpy(np.searchsorted(chunk, batch_idx[hit_rows]))
            tokens = torch.from_numpy(token_idx[hit_rows])

            layer_inputs, outputs, _ = self._trace([ids[i] for i in chunk])
            order.append(hit_rows)
            inputs.append(layer_inputs[in_batch.to(self.device), tokens.to(self.device)].cpu())
            activations.append(
                outputs[in_batch.to(self.device), tokens.to(self.device)][:, self.neuron_idxs].cpu()
            )
            del layer_inputs, outputs
            self._release()

        restore_order = np.argsort(np.concatenate(order))
        return GatherResult(
            inputs=torch.cat(inputs)[restore_order],
            activations=torch.cat(activations)[restore_order],
        )

    def check_consistent(
        self,
        scan: ScanResult,
        gathered: GatherResult,
        batch_idx: Int[np.ndarray, "n_hits"],
        token_idx: Int[np.ndarray, "n_hits"],
        atol: float = 1e-3,
        rtol: float = 1e-3,
    ) -> float:
        """Returns the largest absolute disagreement, so a run can log how tight it was."""
        expected = scan.activations[batch_idx, token_idx]
        worst = (gathered.activations - expected).abs().max().item()
        if not torch.allclose(gathered.activations, expected, atol=atol, rtol=rtol):
            raise ValueError(
                f"pass two disagrees with pass one (max abs diff {worst:.3g}); a hit found by "
                f"the scan is not being re-found at the same position"
            )
        return worst

    def _trace(
        self, ids: Sequence[SampleId]
    ) -> tuple[
        Float[torch.Tensor, "batch seq hidden"],
        Float[torch.Tensor, "batch seq out_hidden"],
        Bool[torch.Tensor, "batch seq"],
    ]:
        batch = self.source.to_model_batch(ids).to(self.device)
        # nnsight leaves autograd on, which keeps every layer's intermediates alive for a
        # backward pass that never comes - at 4 x 2048 tokens that alone exhausts a 15GB T4.
        with torch.no_grad(), self.model.trace(**batch.kwargs):
            layer = self.layer_getter(self.model)
            inputs = layer.input.save()
            outputs = layer.output.save()
        return inputs.detach(), outputs.detach(), batch.valid_mask

    def _release(self) -> None:
        empty_cache()
        gc.collect()
