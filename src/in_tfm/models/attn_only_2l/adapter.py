from pathlib import Path
from typing import ClassVar

import torch

from ...attribution import patch_eager_attention
from ...dependencies import (
    FixedSource,
    ReportedSource,
    ResidualStreamMap,
    StreamSource,
    Tap,
)
from .. import text
from . import model as attn_only_model
from .model import AttnOnlyConfig, load_attn_only_2l, load_trained_attn_only


def patch_attn_only_for_attn_lrp(model: torch.nn.Module) -> None:
    """Attention is the only rule needed.

    - The model has no normalization and no MLP, so softmax is its only nonlinearity.
    - Everything else is linear in its input, where plain gradient*input is a valid LRP rule.
    - The patch is scoped to this module's own `eager_attention_forward`, so it cannot reach
      any other architecture.
    """
    text.force_eager_attention(model)
    patch_eager_attention(attn_only_model)


class AttnOnly2LAdapter(text.DecoderTextAdapter):
    # TransformerLens prepends BOS by default, and the practice notebooks run this model through it.
    prepend_bos = True

    @classmethod
    def from_pretrained(cls, max_length: int = 128) -> "AttnOnly2LAdapter":
        hf_model, tokenizer = load_attn_only_2l()
        return cls(hf_model, tokenizer, max_length)

    def _apply_attn_lrp_patch(self) -> None:
        patch_attn_only_for_attn_lrp(self.get_model())


def stream_sources(
    n_layers: int, n_heads: int, head_writers: bool = True
) -> list[StreamSource]:
    """The final residual stream: the embeddings plus every head of every layer.

    - The text source passes the embeddings in as `inputs_embeds`, so `embed_tokens` never runs.
      - They are read as the first layer's input instead.
    """
    embedding = FixedSource(
        name="embedding", tap=Tap(layer="model.layers.0", port="input"), by_token=True
    )
    if not head_writers:
        layers = [
            ReportedSource(layer=f"model.layers.{layer}.self_attn.o_proj")
            for layer in range(n_layers)
        ]
        return [embedding, *layers]
    heads = [
        ReportedSource(layer=f"model.layers.{layer}.self_attn.o_proj.heads.{head}")
        for layer in range(n_layers)
        for head in range(n_heads)
    ]
    return [embedding, *heads]


class TrainedAttnOnlyAdapter(AttnOnly2LAdapter):
    """Our own TinyStories models, see `attn_only_2l.train`. Subclasses name a checkpoint.

    - The whole causal LM is the model, so `lm_head` is a layer the pipeline can report on.
      - Layer paths then start at `model.`, for example `model.layers.0.self_attn.o_proj.heads.2`.
    - Training windows start at BOS, so samples do too.
    """

    default_checkpoint: ClassVar[Path]
    head_writers: ClassVar[bool] = True
    """False makes each layer's whole o_proj one writer, for when its reports are of o_proj dims."""

    @classmethod
    def from_pretrained(
        cls, max_length: int = 128, checkpoint: Path | None = None
    ) -> "TrainedAttnOnlyAdapter":
        hf_model, tokenizer = load_trained_attn_only(
            checkpoint or cls.default_checkpoint
        )
        return cls(hf_model, tokenizer, max_length)

    def get_model(self) -> torch.nn.Module:
        return self.hf_model

    def embedding(self) -> torch.nn.Module:
        return self.hf_model.get_submodule("model.embed_tokens")

    def dependency_map_for(self, target_layer: str) -> ResidualStreamMap:
        """Only the unembed for now. `q_proj` and `k_proj` also read the positional embedding,
        which is added inside attention and is no module's output."""
        if target_layer != "lm_head":
            raise ValueError(f"only lm_head has a dependency map, not {target_layer}")
        config = self.hf_model.config
        assert isinstance(config, AttnOnlyConfig)
        if config.attention_bias:
            raise ValueError(
                "an o_proj bias would be a stream writer the map leaves out"
            )
        return ResidualStreamMap(
            stream_sources(
                config.num_hidden_layers,
                config.num_attention_heads,
                self.head_writers,
            )
        )


class TinyAttnOnly1LD128Adapter(TrainedAttnOnlyAdapter):
    default_checkpoint = Path("checkpoints/tiny_attn_only_1l_d128")


class TinyAttnOnly1LAdapter(TrainedAttnOnlyAdapter):
    """The d256 model. Its writer reports are of whole o_proj dims, so o_proj is one writer."""

    default_checkpoint = Path("checkpoints/tiny_attn_only_1l")
    head_writers = False
