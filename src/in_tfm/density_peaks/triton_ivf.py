"""The IVF search with the distance and the top-k in one CUDA kernel, for `ivf_forest.probed_search`.

- The torch path writes a block of distances per (query chunk, list), then runs `topk`, then merges with sorts.
  - On an A100 at 1M points, `topk` took 36% of the GPU time, the distances 31% and the merge sorts 17%.
- Here a program takes up to `BLOCK_PAIRS` (query, list) pairs that probe one list, and scans the list in tiles.
  - Each distance is packed with its point id into one int64 key: the float's bits above, the id below.
    - Distances are never negative, so their bits order like the floats, and ties go to the lower id.
  - The running top `TOP` keys and a new tile are joined and sorted, and the first half is kept.
- Each (query, probe slot) writes its own top keys, so programs never write to the same place.
  - One `topk` per query over its slots then gives the final list.
- `tests/test_triton_ivf.py` checks it against the torch path, and skips without CUDA.
"""

import torch
import triton
import triton.language as tl
from jaxtyping import Float, Int

TOP = 32
"""Keys kept per (query, list). Also the tile width, since the sort joins two blocks of the same shape."""
BLOCK_PAIRS = 32
CHUNK_PAIRS = 2**21
"""(query, probe slot) pairs per launch. Their keys take CHUNK_PAIRS * TOP * 8 bytes, 512 MB."""
EMPTY = torch.iinfo(torch.int64).max


@triton.jit
def ivf_scan_kernel(
    rows_ptr, sums_ptr, members_ptr, starts_ptr, ranks_ptr,
    work_list_ptr, work_start_ptr, work_count_ptr, pair_query_ptr, pair_slot_ptr, query_ids_ptr, out_ptr,
    D, n_probe,
    JACCARD: tl.constexpr, DENSER: tl.constexpr, BM: tl.constexpr, K: tl.constexpr, BK: tl.constexpr,
):
    work = tl.program_id(0)
    list_id = tl.load(work_list_ptr + work)
    first_pair = tl.load(work_start_ptr + work)
    n_pairs = tl.load(work_count_ptr + work)
    lanes = tl.arange(0, BM)
    valid = lanes < n_pairs
    query_pos = tl.load(pair_query_ptr + first_pair + lanes, mask=valid, other=0)
    slot = tl.load(pair_slot_ptr + first_pair + lanes, mask=valid, other=0)
    query = tl.load(query_ids_ptr + query_pos, mask=valid, other=0)
    if JACCARD:
        query_sum = tl.load(sums_ptr + query, mask=valid, other=0.0)
    if DENSER:
        query_rank = tl.load(ranks_ptr + query, mask=valid, other=0)
    member_start = tl.load(starts_ptr + list_id)
    member_end = tl.load(starts_ptr + list_id + 1)
    running = tl.full((BM, K), 0x7FFFFFFFFFFFFFFF, dtype=tl.int64)
    for tile in range(member_start, member_end, K):
        at = tile + tl.arange(0, K)
        present = at < member_end
        other = tl.load(members_ptr + at, mask=present, other=0)
        total = tl.zeros((BM, K), dtype=tl.float32)
        for start in range(0, D, BK):
            dims = start + tl.arange(0, BK)
            a = tl.load(rows_ptr + query[:, None] * D + dims[None, :], mask=valid[:, None] & (dims[None, :] < D),
                        other=0.0)
            b = tl.load(rows_ptr + other[:, None] * D + dims[None, :], mask=present[:, None] & (dims[None, :] < D),
                        other=0.0)
            diff = a[:, None, :] - b[None, :, :]
            if JACCARD:
                total += tl.sum(tl.abs(diff), axis=2)
            else:
                total += tl.sum(diff * diff, axis=2)
        if JACCARD:
            other_sum = tl.load(sums_ptr + other, mask=present, other=0.0)
            both = query_sum[:, None] + other_sum[None, :]
            union = (both + total) / 2
            distance = 1.0 - ((both - total) / 2) / tl.where(union > 0, union, 1.0)
        else:
            distance = tl.sqrt(total)
        excluded = (~present[None, :]) | (other[None, :] == query[:, None])
        if DENSER:
            other_rank = tl.load(ranks_ptr + other, mask=present, other=0)
            excluded = excluded | (other_rank[None, :] <= query_rank[:, None])
        keys = (distance.to(tl.int32, bitcast=True).to(tl.int64) << 32) | other[None, :].to(tl.int64)
        keys = tl.where(excluded, 0x7FFFFFFFFFFFFFFF, keys)
        joined = tl.reshape(tl.join(running, keys), (BM, 2 * K))
        ordered = tl.sort(joined, dim=1)
        running = tl.split(tl.permute(tl.reshape(ordered, (BM, 2, K)), (0, 2, 1)))[0]
    columns = tl.arange(0, K)
    target = out_ptr + ((query_pos * n_probe + slot) * K)[:, None] + columns[None, :]
    tl.store(target, running, mask=valid[:, None])


def work_items(probes: Int[torch.Tensor, "q n_probe"]) -> tuple[torch.Tensor, ...]:
    """(query, slot) pairs sorted by list, then cut into blocks of at most BLOCK_PAIRS that share one list."""
    n_probe = probes.shape[1]
    lists = probes.flatten()
    order = torch.argsort(lists, stable=True)
    pair_query, pair_slot = order // n_probe, order % n_probe
    present, counts = torch.unique_consecutive(lists[order], return_counts=True)
    segment_start = torch.cumsum(counts, 0) - counts
    blocks = (counts + BLOCK_PAIRS - 1) // BLOCK_PAIRS
    segment = torch.repeat_interleave(torch.arange(len(present), device=probes.device), blocks)
    block_in_segment = torch.arange(len(segment), device=probes.device) - torch.repeat_interleave(
        torch.cumsum(blocks, 0) - blocks, blocks)
    start = segment_start[segment] + block_in_segment * BLOCK_PAIRS
    count = torch.clamp(counts[segment] - block_in_segment * BLOCK_PAIRS, max=BLOCK_PAIRS)
    return present[segment], start, count, pair_query, pair_slot


def fused_probed_search(
    signed: Float[torch.Tensor, "n dim"], sums: Float[torch.Tensor, "n"] | None, members: Int[torch.Tensor, "n"],
    starts: Int[torch.Tensor, "l1"], queries: Int[torch.Tensor, "q"], probes: Int[torch.Tensor, "q n_probe"], n: int,
    denser_ranks: Int[torch.Tensor, "n"] | None = None,
) -> tuple[Int[torch.Tensor, "q n"], Float[torch.Tensor, "q n"]]:
    """The n closest points among each query's probed lists, closest first, with -1 at inf for empty slots."""
    assert n <= TOP, f"at most {TOP} neighbours per query"
    signed = signed.contiguous()
    jaccard = sums is not None
    out = torch.full((len(queries), probes.shape[1], TOP), EMPTY, dtype=torch.int64, device=signed.device)
    work_list, work_start, work_count, pair_query, pair_slot = work_items(probes)
    ivf_scan_kernel[(len(work_list),)](
        signed, sums if jaccard else signed, members, starts, denser_ranks if denser_ranks is not None else members,
        work_list, work_start, work_count, pair_query, pair_slot, queries, out,
        signed.shape[1], probes.shape[1],
        JACCARD=jaccard, DENSER=denser_ranks is not None, BM=BLOCK_PAIRS, K=TOP, BK=8, num_warps=4,
    )
    keys = torch.topk(out.view(len(queries), -1), n, dim=1, largest=False, sorted=True).values
    empty = keys == EMPTY
    found = torch.where(empty, -1, keys & 0xFFFFFFFF)
    distance = (keys >> 32).to(torch.int32).view(torch.float32)
    return found, torch.where(empty, torch.inf, distance)
