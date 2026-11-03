import numpy as np
import torch
from jaxtyping import Int


def sample_windows(
    tokens: Int[np.ndarray, "n_tokens"], starts: Int[np.ndarray, "n_starts"], batch_size: int, seq_len: int,
    rng: np.random.Generator,
) -> Int[torch.Tensor, "batch seq_plus_one"]:
    """Each window starts at a story's BOS, like the report adapter, which prepends BOS.

    - One extra token per window, so the inputs and the next-token targets come from one slice.
    """
    chosen = rng.choice(starts, size=batch_size)
    return torch.from_numpy(np.stack([tokens[s:s + seq_len + 1] for s in chosen]).astype(np.int64))
