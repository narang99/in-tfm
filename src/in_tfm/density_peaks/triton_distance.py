"""A tiled float32 distance kernel for CUDA, used by `torch_forest`.

- `torch.cdist` with p=1 reads both rows from memory for every pair, which made the GPU forest take hours.
- Here a program loads a tile of rows from each side once and compares every pair in the tile.
- Weighted Jaccard is finished from the L1 sum and the row sums inside the kernel, so only the result is written.
"""

import torch
import triton
import triton.language as tl
from jaxtyping import Float

from .forest import PeakDistance

COLUMN_CHUNK = 2**20
"""Columns per launch. The grid's second axis takes at most 65,535 blocks, which is about 2M columns of 32."""

TILE_CONFIGS = [
    triton.Config({"BM": 32, "BN": 32, "BK": 8}, num_warps=4),
    triton.Config({"BM": 64, "BN": 32, "BK": 4}, num_warps=4),
    triton.Config({"BM": 64, "BN": 64, "BK": 4}, num_warps=8),
    triton.Config({"BM": 32, "BN": 32, "BK": 16}, num_warps=4),
]


@triton.autotune(configs=TILE_CONFIGS, key=["D"])
@triton.jit
def distance_kernel(
    a_ptr, b_ptr, sums_a_ptr, sums_b_ptr, out_ptr, M, N, D, stride_a, stride_b, stride_out,
    JACCARD: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    rows_a = tl.program_id(0) * BM + tl.arange(0, BM)
    rows_b = tl.program_id(1) * BN + tl.arange(0, BN)
    total = tl.zeros((BM, BN), dtype=tl.float32)
    for start in range(0, D, BK):
        dims = start + tl.arange(0, BK)
        a = tl.load(
            a_ptr + rows_a[:, None] * stride_a + dims[None, :],
            mask=(rows_a[:, None] < M) & (dims[None, :] < D), other=0.0,
        )
        b = tl.load(
            b_ptr + rows_b[:, None] * stride_b + dims[None, :],
            mask=(rows_b[:, None] < N) & (dims[None, :] < D), other=0.0,
        )
        diff = a[:, None, :] - b[None, :, :]
        if JACCARD:
            total += tl.sum(tl.abs(diff), axis=2)
        else:
            total += tl.sum(diff * diff, axis=2)
    if JACCARD:
        sums_a = tl.load(sums_a_ptr + rows_a, mask=rows_a < M, other=0.0)
        sums_b = tl.load(sums_b_ptr + rows_b, mask=rows_b < N, other=0.0)
        both = sums_a[:, None] + sums_b[None, :]
        union = (both + total) / 2
        overlap = (both - total) / 2
        result = 1.0 - overlap / tl.where(union > 0, union, 1.0)
    else:
        result = tl.sqrt(total)
    tl.store(
        out_ptr + rows_a[:, None] * stride_out + rows_b[None, :], result,
        mask=(rows_a[:, None] < M) & (rows_b[None, :] < N),
    )


def triton_distances(
    a: Float[torch.Tensor, "a dim"], b: Float[torch.Tensor, "b dim"],
    sums_a: Float[torch.Tensor, "a"] | None, sums_b: Float[torch.Tensor, "b"] | None, distance: PeakDistance,
) -> Float[torch.Tensor, "a b"]:
    a, b = a.contiguous(), b.contiguous()
    out = torch.empty((len(a), len(b)), dtype=torch.float32, device=a.device)
    jaccard = distance == "weighted_jaccard"
    sums_a = sums_a.contiguous() if jaccard else a
    sums_b = sums_b.contiguous() if jaccard else b
    for start in range(0, len(b), COLUMN_CHUNK):
        b_part, sums_part = b[start:start + COLUMN_CHUNK], sums_b[start:start + COLUMN_CHUNK]
        out_part = out[:, start:start + COLUMN_CHUNK]
        grid = lambda meta: (triton.cdiv(len(a), meta["BM"]), triton.cdiv(len(b_part), meta["BN"]))  # noqa: E731
        distance_kernel[grid](
            a, b_part, sums_a, sums_part, out_part, len(a), len(b_part), a.shape[1], a.stride(0), b_part.stride(0),
            out_part.stride(0), JACCARD=jaccard,
        )
    return out
