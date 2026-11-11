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
from tqdm import tqdm

from .device import default_device, empty_cache
from .layers import LayerGetter
from .run_config import KnownCutoffs
from .sources import SampleId, SampleSource
from .token_layout import TokenLayout


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


def past_cutoffs(
    cutoffs: KnownCutoffs, activations: Float[torch.Tensor, "batch seq n_neurons"]
) -> Bool[torch.Tensor, "batch seq"]:
    """At or past a cut-off, so a hit the selection keeps strictly past it is always here."""
    beyond = torch.zeros(
        activations.shape[:2], dtype=torch.bool, device=activations.device
    )
    if cutoffs.positive is not None:
        beyond |= (activations >= cutoffs.positive).any(-1)
    if cutoffs.negative is not None:
        beyond |= (activations <= cutoffs.negative).any(-1)
    return beyond


class KeptInputs(BaseModel):
    """Layer inputs the scan kept, keyed by `sample * seq_len + token` in ascending order."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    keys: Int[np.ndarray, "n_kept"]
    inputs: Float[torch.Tensor, "n_kept hidden"]
    activations: Float[torch.Tensor, "n_kept n_neurons"]
    seq_len: int = 0

    @classmethod
    def joined(cls, parts: list["KeptInputs"], seq_len: int) -> "KeptInputs":
        return cls(
            keys=np.concatenate([p.keys for p in parts]),
            inputs=torch.cat([p.inputs for p in parts]),
            activations=torch.cat([p.activations for p in parts]),
            seq_len=seq_len,
        )

    def gather(
        self,
        batch_idx: Int[np.ndarray, "n_hits"],
        token_idx: Int[np.ndarray, "n_hits"],
    ) -> GatherResult:
        """The same result `NeuronCapture.gather` gives, read from what the scan kept."""
        keys = batch_idx.astype(np.int64) * self.seq_len + token_idx
        rows = np.searchsorted(self.keys, keys)
        found = (rows < len(self.keys)) & (
            self.keys[np.minimum(rows, len(self.keys) - 1)] == keys
        )
        if not found.all():
            raise ValueError(
                f"{int((~found).sum())} hits fall inside the known cut-offs, so the scan did not keep "
                f"their inputs. The cut-offs do not belong to this corpus."
            )
        return GatherResult(
            inputs=self.inputs[rows], activations=self.activations[rows]
        )


def inputs_past(
    keep_past: KnownCutoffs,
    layer_inputs: Float[torch.Tensor, "batch seq hidden"],
    neuron_outputs: Float[torch.Tensor, "batch seq n_neurons"],
    valid_mask: Bool[torch.Tensor, "batch seq"],
    first_sample: int,
) -> KeptInputs:
    """Row-major order within a batch, so keys come out ascending."""
    sample, token = torch.nonzero(
        past_cutoffs(keep_past, neuron_outputs) & valid_mask, as_tuple=True
    )
    seq_len = valid_mask.shape[1]
    keys = (sample + first_sample).cpu().numpy().astype(
        np.int64
    ) * seq_len + token.cpu().numpy()
    return KeptInputs(
        keys=keys,
        inputs=layer_inputs[sample, token].cpu(),
        activations=neuron_outputs[sample, token].cpu(),
    )


class NeuronCapture:
    def __init__(
        self,
        model: NNsight,
        source: SampleSource,
        layer_getter: LayerGetter,
        layout: TokenLayout,
        neuron_idxs: Sequence[int],
        batch_size: int = 8,
        device: str | None = None,
    ) -> None:
        self.layout = layout
        self.device = device or default_device()
        self.model = model.to(torch.device(self.device))
        self.source = source
        self.layer_getter = layer_getter
        self.neuron_idxs = list(neuron_idxs)
        self.batch_size = batch_size

    def scan(self) -> ScanResult:
        return self.scan_keeping_inputs(None)[0]

    def scan_keeping_inputs(
        self, keep_past: KnownCutoffs | None
    ) -> tuple[ScanResult, "KeptInputs | None"]:
        """With `keep_past`, the scan also keeps the layer input at every position past a cut-off.

        - The hits are then all in hand after one pass, and `gather` is not needed.
        """
        ids = list(self.source.sample_ids())
        activations: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        kept: list[KeptInputs] = []
        for start, chunk in zip(
            range(0, len(ids), self.batch_size),
            tqdm(list(batched(ids, self.batch_size)), desc="scan"),
        ):
            layer_inputs, outputs, valid_mask = self._trace(chunk)
            neuron_outputs = outputs[..., self.neuron_idxs]
            activations.append(neuron_outputs.cpu())
            masks.append(valid_mask.cpu())
            if keep_past is not None:
                kept.append(
                    inputs_past(
                        keep_past, layer_inputs, neuron_outputs, valid_mask, start
                    )
                )
            del layer_inputs, outputs, neuron_outputs
            self._release()
        scan = ScanResult(
            sample_ids=ids,
            neuron_idxs=self.neuron_idxs,
            activations=cat_captures(activations),
            valid_mask=cat_captures(masks),
        )
        if keep_past is None:
            return scan, None
        return scan, KeptInputs.joined(kept, seq_len=scan.valid_mask.shape[1])

    def gather(
        self,
        batch_idx: Int[np.ndarray, "n_hits"],
        token_idx: Int[np.ndarray, "n_hits"],
    ) -> GatherResult:
        """`batch_idx` indexes into the scan's samples. Results come back in the order the
        positions were given, whatever order the samples were re-run in.

        - Each hit is re-run in the same batch the scan ran it in, neighbours included.
          - A half-precision kernel can round differently for another batch shape.
          - At layer 55 of Qwen3-32B that drifted by 0.69, far past the check's tolerance.
        """
        ids = list(self.source.sample_ids())
        order: list[np.ndarray] = []
        inputs: list[torch.Tensor] = []
        activations: list[torch.Tensor] = []
        for chunk in tqdm(
            self._scan_batches_holding(batch_idx, len(ids)), desc="gather"
        ):
            hit_rows = np.flatnonzero(np.isin(batch_idx, chunk))
            in_batch = torch.from_numpy(np.searchsorted(chunk, batch_idx[hit_rows]))
            tokens = torch.from_numpy(token_idx[hit_rows])

            layer_inputs, outputs, _ = self._trace([ids[i] for i in chunk])
            order.append(hit_rows)
            inputs.append(
                layer_inputs[in_batch.to(self.device), tokens.to(self.device)].cpu()
            )
            activations.append(
                outputs[in_batch.to(self.device), tokens.to(self.device)][
                    :, self.neuron_idxs
                ].cpu()
            )
            del layer_inputs, outputs
            self._release()

        restore_order = np.argsort(np.concatenate(order))
        return GatherResult(
            inputs=torch.cat(inputs)[restore_order],
            activations=torch.cat(activations)[restore_order],
        )

    def _scan_batches_holding(
        self, batch_idx: Int[np.ndarray, "n_hits"], n_samples: int
    ) -> list[Int[np.ndarray, "batch"]]:
        """The scan's batches, as sample indices, that contain at least one hit."""
        starts = np.unique(batch_idx // self.batch_size) * self.batch_size
        return [
            np.arange(start, min(start + self.batch_size, n_samples))
            for start in starts
        ]

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
        # A bf16 model's tensors would reach numpy, which has no bf16. Everything after capture is float32.
        return (
            self.layout.inputs(inputs.detach()).float(),
            self.layout.outputs(outputs.detach()).float(),
            batch.valid_mask,
        )

    def _release(self) -> None:
        empty_cache()
        gc.collect()


def consistency_tolerance(model: torch.nn.Module) -> float:
    """Pass two re-runs a hit in a different batch, and a half-precision kernel may round it differently.

    - Four of the model dtype's rounding steps, and never below the float32 default of 1e-3.
    """
    dtype = next(model.parameters()).dtype
    return max(1e-3, 4 * torch.finfo(dtype).eps)


def cat_captures(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Concat only works if all tensors are of the same size

    Text models can have variable sized batches, their `Source`'s are right now hard-coded to return same sized batches
    Add a specific check with informative error so that a new Source does not accidentally start variable sized batches
    """
    widths = {t.shape[1] for t in tensors}
    if len(widths) > 1:
        raise ValueError(
            f"captured batches disagree on sequence width ({sorted(widths)}); a source must "
            f"pad to a fixed width - see sources.TextSource._encode"
        )
    return torch.cat(tensors, dim=0)
