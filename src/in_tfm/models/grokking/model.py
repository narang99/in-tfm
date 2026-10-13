"""The one-layer transformer of Nanda et al., "Progress measures for grokking via mechanistic interpretability".

- The input is the three tokens `a b =`, and the answer `(a + b) mod P` is read from the `=` position.
- It has no LayerNorm, and no biases in the embedding, the attention or the unembedding.
  - So `v_proj` is exactly `w . x`, which is what the Hadamard product needs.
  - The MLP keeps its two biases, as in the paper.
- Names follow the HF decoders (`layers`, `self_attn.v_proj`), so `layers.v_proj_getter` resolves with no special case.
- The forward takes embeddings rather than token ids, because AttnLRP needs a float leaf to differentiate.
"""

import torch
from jaxtyping import Float, Int
from pydantic import BaseModel
from torch import nn

MODULUS = 113
EQUALS_TOKEN = MODULUS
SEQ_LEN = 3
VOCAB_SIZE = MODULUS + 1
"""The residues, then `=`. The unembedding only predicts the residues."""

TokenEmbeds = Float[torch.Tensor, "batch seq d_model"]


class GrokConfig(BaseModel):
    d_model: int = 128
    n_heads: int = 4
    d_head: int = 32
    d_mlp: int = 512
    n_layers: int = 1


def eager_attention_forward(
    module: nn.Module,
    query: Float[torch.Tensor, "batch head seq d_head"],
    key: Float[torch.Tensor, "batch head seq d_head"],
    value: Float[torch.Tensor, "batch head seq d_head"],
    attention_mask: Float[torch.Tensor, "batch 1 seq seq"] | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[Float[torch.Tensor, "batch seq head d_head"], Float[torch.Tensor, "batch head seq seq"]]:
    """Signed like HF's per-model kernels, which is what `attribution.patch_eager_attention` looks for."""
    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask
    weights = nn.functional.softmax(scores, dim=-1)
    return torch.matmul(weights, value).transpose(1, 2).contiguous(), weights


class GrokAttention(nn.Module):
    def __init__(self, config: GrokConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.scaling = config.d_head**-0.5
        inner = config.n_heads * config.d_head
        self.q_proj = nn.Linear(config.d_model, inner, bias=False)
        self.k_proj = nn.Linear(config.d_model, inner, bias=False)
        self.v_proj = nn.Linear(config.d_model, inner, bias=False)
        self.o_proj = nn.Linear(inner, config.d_model, bias=False)
        self.mixed = nn.Identity()
        """Marks the attention-weighted values before `o_proj`, so a hook can read a head's OV output (index `h * d_head + d`)."""

    def forward(self, hidden: TokenEmbeds, causal_mask: Float[torch.Tensor, "1 1 seq seq"]) -> TokenEmbeds:
        query, key, value = (self._to_heads(proj(hidden)) for proj in (self.q_proj, self.k_proj, self.v_proj))
        # a module global, not a captured reference, so the AttnLRP patch is visible
        mixed, _ = eager_attention_forward(self, query, key, value, causal_mask, scaling=self.scaling)
        return self.o_proj(self.mixed(mixed.reshape(*hidden.shape[:2], -1)))

    def _to_heads(self, x: Float[torch.Tensor, "batch seq inner"]) -> Float[torch.Tensor, "batch head seq d_head"]:
        """Head-major, so a flat neuron index is `head * d_head + dim`."""
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)


class GrokMlp(nn.Module):
    def __init__(self, config: GrokConfig) -> None:
        super().__init__()
        self.up_proj = nn.Linear(config.d_model, config.d_mlp)
        self.act = nn.ReLU()
        self.down_proj = nn.Linear(config.d_mlp, config.d_model)

    def forward(self, hidden: TokenEmbeds) -> TokenEmbeds:
        return self.down_proj(self.act(self.up_proj(hidden)))


class GrokLayer(nn.Module):
    def __init__(self, config: GrokConfig) -> None:
        super().__init__()
        self.self_attn = GrokAttention(config)
        self.mlp = GrokMlp(config)

    def forward(self, hidden: TokenEmbeds, causal_mask: Float[torch.Tensor, "1 1 seq seq"]) -> TokenEmbeds:
        hidden = hidden + self.self_attn(hidden, causal_mask)
        return hidden + self.mlp(hidden)


class GrokTransformer(nn.Module):
    def __init__(self, config: GrokConfig = GrokConfig()) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, config.d_model)
        self.pos_embed = nn.Parameter(torch.empty(SEQ_LEN, config.d_model))
        self.layers = nn.ModuleList(GrokLayer(config) for _ in range(config.n_layers))
        self.unembed = nn.Linear(config.d_model, MODULUS, bias=False)
        self.register_buffer("causal_mask", causal_mask(SEQ_LEN), persistent=False)
        self._init_weights()

    def _init_weights(self) -> None:
        """Gaussian with std `1 / sqrt(fan_in)` and zero biases, the scheme of the paper's code.

        - The embeddings take `d_model` as their fan-in, since they are read out into the residual stream.
        """
        for name, param in self.named_parameters():
            if name.endswith("bias"):
                nn.init.zeros_(param)
            else:
                nn.init.normal_(param, std=param.shape[-1] ** -0.5)

    def forward(self, inputs_embeds: TokenEmbeds) -> Float[torch.Tensor, "batch seq modulus"]:
        hidden = inputs_embeds + self.pos_embed[: inputs_embeds.shape[1]]
        for layer in self.layers:
            hidden = layer(hidden, self.causal_mask)
        return self.unembed(hidden)

    def logits_for_tokens(self, tokens: Int[torch.Tensor, "batch seq"]) -> Float[torch.Tensor, "batch seq modulus"]:
        return self(self.embed_tokens(tokens))


def causal_mask(seq_len: int) -> Float[torch.Tensor, "1 1 seq seq"]:
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    return mask[None, None]
