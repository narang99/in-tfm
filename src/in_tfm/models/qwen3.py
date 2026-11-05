"""Qwen3 decoders, Marin 32B among them, cut after the deepest layer a run reads.

- A neuron at layer L depends only on the embedding and layers 0..L, so the later layers are never loaded.
  - The final norm still runs. It sits after every captured layer, so it changes nothing captured.
- Only the safetensors shards that hold the kept weights are downloaded.
  - The model loads from a local folder with a filtered weight index and a shortened config.
  - At layer 16 of Marin 32B that is about 18 GB instead of 64 GB.
"""

import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

# isort: split
from .. import compat  # noqa: F401  (must precede any lxt import, see compat.py)

# isort: split
from huggingface_hub import hf_hub_download, snapshot_download
from lxt.efficient.core import monkey_patch
from lxt.efficient.patches import (
    dropout_forward,
    gated_mlp_forward,
    patch_method,
    rms_norm_forward,
)
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3MLP,
    Qwen3Model,
    Qwen3RMSNorm,
)

from ..attribution import compute_attnlrp_relevance, patch_eager_attention
from ..layers import LayerGetter
from ..sources import ModelBatch
from . import text

LAYER_OF_NAME = re.compile(r"(?:^|\.)layers\.(\d+)\.")
INDEX_NAME = "model.safetensors.index.json"
TRUNCATED_ROOT = Path.home() / ".cache" / "in_tfm" / "truncated"


def layers_needed(layer_name: str) -> int:
    """`layers.16.mlp.gate_proj` needs layers 0..16, so 17."""
    match = LAYER_OF_NAME.search(layer_name)
    if match is None:
        raise ValueError(f"{layer_name!r} names no decoder layer")
    return int(match.group(1)) + 1


def kept_weight(weight_name: str, n_layers: int) -> bool:
    """The embedding, the final norm and the first `n_layers` layers. `lm_head` is never read."""
    match = LAYER_OF_NAME.search(weight_name)
    if match is not None:
        return int(match.group(1)) < n_layers
    return weight_name.startswith("model.")


def link_into(folder: Path, files: list[Path]) -> None:
    for file in files:
        target = folder / file.name
        if not target.exists():
            target.symlink_to(file.resolve())


def truncated_checkpoint(name: str, n_layers: int) -> Path:
    """A local folder that `from_pretrained` reads as a model of `n_layers` layers."""
    index = json.loads(Path(hf_hub_download(name, INDEX_NAME)).read_text())
    weight_map = {
        weight: shard
        for weight, shard in index["weight_map"].items()
        if kept_weight(weight, n_layers)
    }
    shards = sorted(set(weight_map.values()))
    snapshot = Path(snapshot_download(name, allow_patterns=[*shards, "config.json"]))

    folder = TRUNCATED_ROOT / name.replace("/", "--") / f"layers-{n_layers}"
    folder.mkdir(parents=True, exist_ok=True)
    link_into(folder, [snapshot / shard for shard in shards])
    (folder / INDEX_NAME).write_text(
        json.dumps({"metadata": index.get("metadata", {}), "weight_map": weight_map})
    )
    config = AutoConfig.from_pretrained(snapshot)
    config.num_hidden_layers = n_layers
    if getattr(config, "layer_types", None) is not None:
        config.layer_types = config.layer_types[:n_layers]
    config.save_pretrained(folder)
    return folder


def patch_qwen3_for_attn_lrp(model: torch.nn.Module) -> None:
    """lxt's qwen3 patch map, with its attention entry swapped as in `gemma3.patch_gemma3_for_attn_lrp`.

    - Its RMSNorm rule also covers `q_norm` and `k_norm`, which are `Qwen3RMSNorm` over each head's 128 dims.
    """
    patch_map = {
        Qwen3MLP: partial(patch_method, gated_mlp_forward),
        Qwen3RMSNorm: partial(patch_method, rms_norm_forward),
        torch.nn.Dropout: partial(patch_method, dropout_forward),
        modeling_qwen3: patch_eager_attention,
    }
    text.force_eager_attention(model)
    monkey_patch(model, patch_map=patch_map, verbose=True)


@contextmanager
def checkpointed_layers(model: torch.nn.Module) -> Iterator[None]:
    """Each decoder layer recomputes its activations in the backward pass, instead of keeping them.

    - AttnLRP over 2,048 tokens and 17 Marin layers would keep about 20 GB of activations, and an L4 has about 4 GB spare.
    - The forward of each layer is wrapped as it is, so whatever replaced it (the lxt patches, NNsight) still runs.
    - HF's own `gradient_checkpointing_enable` only acts in training mode, and AttnLRP runs in eval mode.
    - The KV cache is off, since a recomputed layer would append its keys to the cache a second time.
    """
    layers = list(model.get_submodule("layers"))
    originals = [layer.forward for layer in layers]
    config = model.config
    assert isinstance(config, PretrainedConfig)
    use_cache = config.use_cache
    config.use_cache = False
    for layer, forward in zip(layers, originals):
        layer.forward = partial(checkpoint, forward, use_reentrant=False)
    try:
        yield
    finally:
        config.use_cache = use_cache
        for layer, forward in zip(layers, originals):
            layer.forward = forward


def checkpointed_attnlrp_relevance(
    model: torch.nn.Module,
    batch: ModelBatch,
    layer_getter: LayerGetter,
    neuron_idx: int,
    token_idx: int | None,
):
    with checkpointed_layers(model):
        return compute_attnlrp_relevance(
            model, batch, layer_getter, neuron_idx, token_idx
        )


class Qwen3Adapter(text.DecoderTextAdapter):
    """Holds the bare `Qwen3Model`, with no `lm_head`.

    - BOS is never a hit, as in the Marin atlas.
    """

    bos_is_a_hit = False
    attr_fn = staticmethod(checkpointed_attnlrp_relevance)

    def get_model(self) -> torch.nn.Module:
        return self.hf_model

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        layer_name: str,
        max_length: int = 128,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
    ) -> "Qwen3Adapter":
        """Weights are frozen, so AttnLRP's backward pass keeps no gradient for them.
        In bf16 those gradients would be as large as the weights."""
        folder = truncated_checkpoint(name, layers_needed(layer_name))
        model = Qwen3Model.from_pretrained(folder, dtype=dtype, device_map=device)
        model.requires_grad_(False)
        return cls(model, text.load_tokenizer(name), max_length)

    def _apply_attn_lrp_patch(self) -> None:
        patch_qwen3_for_attn_lrp(self.get_model())
