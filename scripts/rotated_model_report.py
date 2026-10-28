#!/usr/bin/env python
"""The neuron report on the attn-only model with its residual stream rotated, to test what the clusters depend on.

- A random rotation R is applied to everything that writes to or reads from the residual stream.
  - Writers: the token and position embeddings, and both layers' o_proj. Each output vector v becomes R v.
  - Readers: q, k and v projections of both layers, and the unembedding. Each weight W becomes W Rᵀ.
- The model has no norm layers, so every reader's output is unchanged. The script checks that before reporting.
- Hits are picked once, from the original model's scan, and both models report on exactly those positions.
  - Sampling separately would not give the same hits: float32 rounding moves a few values across bucket edges,
    and one changed bucket size reshuffles every later draw.
  - The rotated model's activations at those positions are checked against the original scan.
- So only the residual coordinates differ between the two reports.
  - Clusters on per-coordinate Hadamard products can move.
  - Clusters on raw inputs with euclidean distance cannot, since a rotation keeps euclidean distances.
- Reports go to `<out_dir>/original` and `<out_dir>/rotated`, each with `hit_positions.npz`.

    python scripts/rotated_model_report.py --rotation-seed 1 --config configs/attn_only_2l_signed_q_proj_l1_pile.yaml
"""

import copy
import sys
import time

import torch
from jaxtyping import Float

from in_tfm.models.attn_only_2l import AttnOnly2LAdapter, AttnOnlyForCausalLM
from in_tfm.neuron_run import prepare_run, report_selected, scan_and_select, timed
from in_tfm.run_config import load_run_config
from in_tfm.sources.wikitext import load_texts

SEED_FLAG = "--rotation-seed"
CHECK_TEXTS = 16
CHECK_LENGTH = 64
TOLERANCE = 1e-3


def pop_seed(argv: list[str]) -> tuple[int | None, list[str]]:
    """`RunConfig` forbids unknown flags, so the seed is taken off before it parses the rest."""
    if SEED_FLAG not in argv:
        return None, argv
    at = argv.index(SEED_FLAG)
    return int(argv[at + 1]), argv[:at] + argv[at + 2 :]


def random_rotation(size: int, seed: int) -> Float[torch.Tensor, "size size"]:
    """Haar-random orthogonal matrix: QR of a Gaussian matrix, with the signs of R's diagonal folded into Q."""
    gaussian = torch.randn(size, size, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    q, r = torch.linalg.qr(gaussian)
    return q * torch.sign(torch.diagonal(r))


@torch.no_grad()
def rotate_residual(model: AttnOnlyForCausalLM, rotation: Float[torch.Tensor, "hidden hidden"]) -> None:
    """In place. Embedding rows are written vectors, so they become rows @ Rᵀ. nn.Linear weights are (out, in)."""

    def set_weight(param: torch.nn.Parameter, value: torch.Tensor) -> None:
        param.copy_(value.to(param.dtype))

    R = rotation
    stack = model.model
    set_weight(stack.embed_tokens.weight, stack.embed_tokens.weight.double() @ R.T)
    set_weight(stack.pos_embed.weight, stack.pos_embed.weight.double() @ R.T)
    for layer in stack.layers:
        attn = layer.self_attn
        for reader in (attn.q_proj, attn.k_proj, attn.v_proj):
            set_weight(reader.weight, reader.weight.double() @ R.T)
        set_weight(attn.o_proj.weight, R @ attn.o_proj.weight.double())
        set_weight(attn.o_proj.bias, R @ attn.o_proj.bias.double())
    set_weight(model.lm_head.weight, model.lm_head.weight.double() @ R.T)


@torch.no_grad()
def layer_outputs(model: AttnOnlyForCausalLM, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}
    handles = [
        getattr(layer.self_attn, name).register_forward_hook(
            lambda _m, _i, out, key=f"layer {i} {name}": captured.__setitem__(key, out)
        )
        for i, layer in enumerate(model.model.layers)
        for name in ("q_proj", "k_proj", "v_proj")
    ]
    try:
        captured["logits"] = model(input_ids=input_ids).logits
    finally:
        for handle in handles:
            handle.remove()
    return captured


def check_same_outputs(original: AttnOnlyForCausalLM, rotated: AttnOnlyForCausalLM, tokenizer, texts: list[str]) -> None:
    ids = tokenizer([tokenizer.bos_token + t for t in texts[:CHECK_TEXTS]], return_tensors="pt", truncation=True,
                    max_length=CHECK_LENGTH, padding="max_length").input_ids
    before, after = layer_outputs(original, ids), layer_outputs(rotated, ids)
    worst = 0.0
    for key in before:
        diff = (before[key] - after[key]).abs().max().item()
        scale = before[key].abs().max().item()
        print(f"[rotation check] {key}: max abs diff {diff:.2e} (values up to {scale:.2f})")
        worst = max(worst, diff / scale)
    if worst > TOLERANCE:
        raise ValueError(f"the rotated model differs: worst relative diff {worst:.2e}")


def main() -> None:
    start = time.perf_counter()
    seed, argv = pop_seed(sys.argv[1:])
    if seed is None:
        raise ValueError(f"{SEED_FLAG} is required")
    config = load_run_config(argv)
    with timed("load samples"):
        texts = load_texts(config.text, config.seed)
    with timed("load models"):
        original = AttnOnly2LAdapter.from_pretrained(config.text.max_length)
        rotated_model = copy.deepcopy(original.hf_model)
        rotate_residual(rotated_model, random_rotation(rotated_model.config.hidden_size, seed))
        rotated = AttnOnly2LAdapter(rotated_model, original.tokenizer, config.text.max_length)
    check_same_outputs(original.hf_model, rotated.hf_model, original.tokenizer, texts)

    original_config = config.model_copy(update={"out_dir": config.out_dir / "original"})
    original_run = prepare_run(original_config, original, texts)
    selected = scan_and_select(original_run, original_config)
    print("=== original")
    report_selected(original_run, original_config, selected)
    print("=== rotated, on the original's hits")
    rotated_config = config.model_copy(update={"out_dir": config.out_dir / "rotated"})
    report_selected(prepare_run(rotated_config, rotated, texts), rotated_config, selected)
    print(f"[total] {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
