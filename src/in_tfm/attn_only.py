"""An attention-only transformer with shortformer positional embeddings, shaped like an HF
decoder so the rest of the pipeline takes it unchanged.

This loads `callummcdougall/attn_only_2L_half`, the 2-layer attention-only model the mech-interp
practice notebooks use for induction heads. That repo holds a bare TransformerLens state dict
with no `config.json`, and TransformerLens 2.17.0 is what loads it in practice. Rather than take
the dependency, this module reproduces that model's forward pass. It was checked against
TransformerLens 2.17.0 and agrees on logits to 2e-5.

Two things differ from a stock HF decoder, both read off the checkpoint:

- No normalization anywhere. The residual stream reaches `q_proj` unscaled, so the Hadamard
  decomposition is over the raw stream. There is also no `q_norm`, which removes the caveat that
  a large raw query value might only mean the token had a large post-norm query.
- Shortformer positional embeddings. Position is added to the `q_proj` and `k_proj` inputs only,
  never to the value path and never to the residual stream. Since `q_proj` stays linear in its
  input, a query or key coordinate's activation splits *exactly* into a token term and a
  position term, which RoPE models do not allow.
"""

import torch
from huggingface_hub import hf_hub_download
from jaxtyping import Bool, Float, Int
from torch import nn
from transformers import AutoTokenizer, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput
from transformers.models.llama import modeling_llama

ATTN_ONLY_2L_REPO = "callummcdougall/attn_only_2L_half"
ATTN_ONLY_2L_WEIGHTS = "attn_only_2L_half.pth"
ATTN_ONLY_2L_TOKENIZER = "EleutherAI/gpt-neox-20b"


class AttnOnlyConfig(PretrainedConfig):
    """Defaults are this one checkpoint's shapes, since the repo ships no config to read them
    from. `vocab_size` is one row wider than the tokenizer: the checkpoint pads 50277 to an even
    number, and that last row is never a real token."""

    model_type = "attn_only_shortformer"

    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        head_dim: int = 64,
        num_hidden_layers: int = 2,
        max_position_embeddings: int = 2048,
        vocab_size: int = 50278,
        **kwargs,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size
        super().__init__(**kwargs)


class AttnOnlyAttention(nn.Module):
    """Named `self_attn` with `q_proj`/`k_proj`/`v_proj`/`o_proj` so `layers.q_proj_getter` and
    `layers.k_proj_getter` resolve on this model with no special case."""

    def __init__(self, config: AttnOnlyConfig) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        # no grouped-query attention here, but HF's eager kernel reads this to expand kv heads
        self.num_key_value_groups = 1
        inner = self.n_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, inner)
        self.k_proj = nn.Linear(config.hidden_size, inner)
        self.v_proj = nn.Linear(config.hidden_size, inner)
        self.o_proj = nn.Linear(inner, config.hidden_size)

    def forward(
        self,
        hidden_states: Float[torch.Tensor, "batch seq hidden"],
        position_embeds: Float[torch.Tensor, "seq hidden"],
        attention_mask: Float[torch.Tensor, "batch 1 seq seq"],
    ) -> Float[torch.Tensor, "batch seq hidden"]:
        qk_input = hidden_states + position_embeds
        query = self._to_heads(self.q_proj(qk_input))
        key = self._to_heads(self.k_proj(qk_input))
        value = self._to_heads(self.v_proj(hidden_states))
        # Reached through the module rather than imported by name on purpose: AttnLRP patches
        # the module's global, and only a call-time lookup sees that patch. See
        # attribution.patch_attn_only_for_attn_lrp.
        attn_out, _ = modeling_llama.eager_attention_forward(
            self, query, key, value, attention_mask, scaling=self.scaling, dropout=0.0
        )
        return self.o_proj(attn_out.reshape(*hidden_states.shape[:2], -1))

    def _to_heads(
        self, x: Float[torch.Tensor, "batch seq inner"]
    ) -> Float[torch.Tensor, "batch head seq head_dim"]:
        """Splits head-major, so a flat neuron index is `head * head_dim + dim` - the convention
        `layers.q_proj_getter` documents and `_converted_state_dict` lays the weights out for."""
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)


class AttnOnlyDecoderLayer(nn.Module):
    def __init__(self, config: AttnOnlyConfig) -> None:
        super().__init__()
        self.self_attn = AttnOnlyAttention(config)

    def forward(
        self,
        hidden_states: Float[torch.Tensor, "batch seq hidden"],
        position_embeds: Float[torch.Tensor, "seq hidden"],
        attention_mask: Float[torch.Tensor, "batch 1 seq seq"],
    ) -> Float[torch.Tensor, "batch seq hidden"]:
        return hidden_states + self.self_attn(hidden_states, position_embeds, attention_mask)


class AttnOnlyModel(nn.Module):
    """The decoder stack. `embed_tokens` and `layers` keep HF's names, which is what lets
    `sources.TextSource` and the layer getters take this as the model."""

    def __init__(self, config: AttnOnlyConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layers = nn.ModuleList(
            AttnOnlyDecoderLayer(config) for _ in range(config.num_hidden_layers)
        )

    def forward(
        self,
        inputs_embeds: Float[torch.Tensor, "batch seq hidden"] | None = None,
        attention_mask: Int[torch.Tensor, "batch seq"] | None = None,
        input_ids: Int[torch.Tensor, "batch seq"] | None = None,
    ) -> BaseModelOutput:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        seq = inputs_embeds.shape[1]
        position_embeds = self.pos_embed.weight[:seq]
        mask = self._additive_mask(attention_mask, seq, inputs_embeds)
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = layer(hidden, position_embeds, mask)
        return BaseModelOutput(last_hidden_state=hidden)

    def _additive_mask(
        self,
        attention_mask: Int[torch.Tensor, "batch seq"] | None,
        seq: int,
        like: torch.Tensor,
    ) -> Float[torch.Tensor, "batch 1 seq seq"]:
        """Causal, widened by the source's padding mask when there is one.

        Blocked entries get `dtype.min` rather than `-inf`, which TransformerLens uses. A query
        row that is entirely padding would softmax `-inf` to nan and poison the backward pass for
        the whole batch. Those rows are dropped by `valid_mask` downstream either way, so a large
        finite penalty loses nothing.
        """
        blocked: Bool[torch.Tensor, "batch 1 seq seq"] = torch.triu(
            torch.ones(seq, seq, dtype=torch.bool, device=like.device), diagonal=1
        )[None, None]
        if attention_mask is not None:
            blocked = blocked | ~attention_mask[:, None, None, :].bool()
        return torch.zeros_like(blocked, dtype=like.dtype).masked_fill(
            blocked, torch.finfo(like.dtype).min
        )


class AttnOnlyForCausalLM(nn.Module):
    """Holds the stack as `.model`, matching HF, so `NNsight(hf_model.model)` and
    `hf_model.model.embed_tokens` keep working in the report scripts."""

    def __init__(self, config: AttnOnlyConfig) -> None:
        super().__init__()
        self.config = config
        self.model = AttnOnlyModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, **kwargs) -> CausalLMOutput:
        hidden = self.model(**kwargs).last_hidden_state
        return CausalLMOutput(logits=self.lm_head(hidden))


def _converted_state_dict(
    tl_state: dict[str, torch.Tensor], config: AttnOnlyConfig
) -> dict[str, torch.Tensor]:
    """TransformerLens keeps attention weights per head: `(n_heads, hidden, head_dim)` going in
    and `(n_heads, head_dim, hidden)` coming out. `nn.Linear` wants `(out_features, in_features)`
    with the head axis folded into `out_features`, and folding it head-major is what makes a flat
    neuron index `head * head_dim + dim`.
    """
    inner = config.num_attention_heads * config.head_dim
    converted = {
        "model.embed_tokens.weight": tl_state["embed.W_E"],
        "model.pos_embed.weight": tl_state["pos_embed.W_pos"],
        "lm_head.weight": tl_state["unembed.W_U"].T,
        "lm_head.bias": tl_state["unembed.b_U"],
    }
    for idx in range(config.num_hidden_layers):
        prefix = f"model.layers.{idx}.self_attn"
        for proj, weight, bias in (
            ("q_proj", "W_Q", "b_Q"),
            ("k_proj", "W_K", "b_K"),
            ("v_proj", "W_V", "b_V"),
        ):
            converted[f"{prefix}.{proj}.weight"] = (
                tl_state[f"blocks.{idx}.attn.{weight}"].permute(0, 2, 1).reshape(inner, -1)
            )
            converted[f"{prefix}.{proj}.bias"] = tl_state[f"blocks.{idx}.attn.{bias}"].reshape(-1)
        converted[f"{prefix}.o_proj.weight"] = (
            tl_state[f"blocks.{idx}.attn.W_O"].reshape(inner, -1).T
        )
        converted[f"{prefix}.o_proj.bias"] = tl_state[f"blocks.{idx}.attn.b_O"]
    return converted


def load_attn_only_2l(
    dtype: torch.dtype = torch.float32,
) -> tuple[AttnOnlyForCausalLM, AutoTokenizer]:
    """The checkpoint is fp16 and is upcast by default, since clustering and relevance downstream
    are not worth doing in half precision.

    `weights_only=True` because this is a pickle fetched from a model hub. The load is strict, so
    a checkpoint that does not match `AttnOnlyConfig`'s shapes fails here rather than silently
    producing a differently-wired model. TransformerLens's `mask` and `IGNORE` buffers are
    dropped: they are constants this forward pass rebuilds.
    """
    path = hf_hub_download(ATTN_ONLY_2L_REPO, ATTN_ONLY_2L_WEIGHTS)
    tl_state = torch.load(path, map_location="cpu", weights_only=True)
    config = AttnOnlyConfig()
    model = AttnOnlyForCausalLM(config)
    model.load_state_dict(_converted_state_dict(tl_state, config))
    tokenizer = AutoTokenizer.from_pretrained(ATTN_ONLY_2L_TOKENIZER)
    return model.to(dtype).eval(), tokenizer
