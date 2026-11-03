"""Train a small attention-only shortformer model on a corpus we know, so clustering can use that same corpus.

- `loop` is the training loop itself. Read it first.
- `config` holds every setting, and `model_setup` builds and initialises the model.
- `corpus` trains the BPE tokenizer and encodes the corpus, and `windows` draws a batch from it.
- `evaluation` measures the val loss along the way.

- The architecture is `AttnOnlyForCausalLM`, the shim for `callummcdougall/attn_only_2L_half`.
  - It has no MLP and no normalization, and position goes only into the queries and keys.
  - The attention projections have no bias. The unembed keeps one, which learns the unigram logits.
- The token arrays are saved next to the checkpoint, so a later report reads the exact training stream.
"""

from .config import LOG_FILE, TOKENS_FILE, ModelShape, TinyAttnOnlyTrainConfig
from .corpus import encode_stories, prepare_tokens, story_starts, train_bpe
from .loop import lr_at, save_weights, train
from .loss import next_token_loss
from .model_setup import build_model, init_weights
from .windows import sample_windows

__all__ = [
    "LOG_FILE", "TOKENS_FILE", "ModelShape", "TinyAttnOnlyTrainConfig", "build_model", "encode_stories", "init_weights",
    "lr_at", "next_token_loss", "prepare_tokens", "sample_windows", "save_weights", "story_starts", "train", "train_bpe",
]
