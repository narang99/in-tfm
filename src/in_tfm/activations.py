"""Capturing per-token MLP activations from RadDino via nnsight tracing.

`get_activations` streams DICOMs off disk in batches and hooks one `mlp.fc2` layer's
input/output through NNsight's `.trace()`, rather than running the whole model and reading
`outputs.hidden_states` - we need the *pre*-fc2 features (fc2's input) to build Hadamard
products downstream, and HF's stock output objects don't expose that.
"""

import gc
from itertools import islice
from pathlib import Path

import torch
from jaxtyping import Float
from nnsight.modeling.base import NNsight
from tqdm import tqdm
from transformers.image_processing_utils import BaseImageProcessor

from .dicom import get_batch
from .layers import LayerGetter


def clear_mem() -> None:
    """Drop Jupyter's cached Out/In history (if running under IPython) before freeing CUDA
    memory - `torch.cuda.empty_cache()` alone can't reclaim tensors still referenced by cell
    outputs."""
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None:
            ip.run_line_magic("reset", "out")
    except ImportError:
        pass
    torch.cuda.empty_cache()
    gc.collect()


def get_activations(
    model: NNsight,
    processor: BaseImageProcessor,
    layer_getter: LayerGetter,
    dcm_base: Path,
    n_iter: int = 14,
    bs: int = 4,
    device: str = "cuda",
) -> tuple[
    list[Path],
    Float[torch.Tensor, "batch seq hidden"],
    Float[torch.Tensor, "batch seq hidden"],
]:
    """Inputs/outputs keep the (batch, seq_len, hidden) shape rather than flattening
    batch/seq_len together, so a later `argwhere` on the activations can be traced back to
    (batch_idx, token_idx) pairs.
    """
    it = dcm_base.glob("*.dcm")
    all_dcm_paths: list[Path] = []
    all_inputs: list[torch.Tensor] = []
    all_outputs: list[torch.Tensor] = []
    model = model.to(device)
    for _ in tqdm(range(n_iter)):
        dcm_paths = list(islice(it, bs))
        all_dcm_paths.extend(dcm_paths)
        batch = get_batch(processor, dcm_paths)
        batch = batch.to(device)
        with model.trace(**batch):
            lay = layer_getter(model)
            inputs = lay.input.save()
            outputs = lay.output.save()
        batch = batch.to("cpu")
        all_inputs.append(inputs.detach().cpu())
        all_outputs.append(outputs.detach().cpu())

        del inputs, outputs, batch
        torch.cuda.empty_cache()
        gc.collect()

    outputs = torch.cat(all_outputs, dim=0)
    inputs = torch.cat(all_inputs, dim=0)
    return all_dcm_paths, inputs, outputs
