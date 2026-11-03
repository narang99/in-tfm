import math

from torch import nn

from ..model import AttnOnlyConfig, AttnOnlyForCausalLM
from .config import TinyAttnOnlyTrainConfig


def build_model(config: TinyAttnOnlyTrainConfig) -> AttnOnlyForCausalLM:
    shape = config.shape
    model = AttnOnlyForCausalLM(AttnOnlyConfig(
        hidden_size=shape.d_model, num_attention_heads=shape.n_heads, head_dim=shape.head_dim,
        num_hidden_layers=shape.n_layers, max_position_embeddings=shape.seq_len, vocab_size=config.vocab_size,
        attention_bias=False,
    ))
    init_weights(model, shape.n_layers, config.init_std)
    return model


def init_weights(model: AttnOnlyForCausalLM, n_layers: int, std: float) -> None:
    """GPT-2 style, at a larger scale. The output projections are scaled down, since each adds to an unnormalized stream.

    - Without normalization the residual stream is as small as the init, so every attention score starts near 0.
      - A score is bilinear in two small weights, so its gradient starts small too.
    - At std 0.02 neither TinyStories nor repeated random tokens formed a previous-token head.
      - On repeated random tokens, std 0.1 with lr 3e-3 formed both heads in a few thousand steps.
      - std 0.05, or lr 1e-3, did not.
    """
    for name, param in model.named_parameters():
        if name.endswith("bias"):
            nn.init.zeros_(param)
        elif "o_proj" in name:
            nn.init.normal_(param, std=std / math.sqrt(2 * n_layers))
        else:
            nn.init.normal_(param, std=std)
