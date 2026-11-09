#!/usr/bin/env python
"""Gate, up and act of one Marin 32B MLP neuron on hand-written prompts, at the words marked with [[ ]].

- A marked word is read at its last token, the first position where the model has seen all of it.

- Forward passes only, one prompt at a time, from BOS as in the Marin atlas.
- Only layers 0..L are loaded, through `truncated_checkpoint`.

    python scripts/marin_minimal_pairs.py --prompts configs/marin_l16_19311_pairs.yaml --out pairs.json
"""

import argparse
import json
import re
from pathlib import Path

import torch
import yaml
from pydantic import BaseModel
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

from in_tfm.models.qwen3 import truncated_checkpoint

MARKED = re.compile(r"\[\[(.+?)\]\]")


class PromptItem(BaseModel):
    id: str
    text: str


class PromptGroup(BaseModel):
    name: str
    prediction: str
    note: str = ""
    items: list[PromptItem]


class PromptSet(BaseModel):
    groups: list[PromptGroup]


class TargetReading(BaseModel):
    word: str
    token: str
    position: int
    gate: float
    up: float
    act: float


class ItemReading(BaseModel):
    group: str
    id: str
    text: str
    targets: list[TargetReading]
    tokens: list[str]
    act_per_token: list[float]


def strip_markers(marked: str) -> tuple[str, list[tuple[str, int]]]:
    """Clean text, and for each marked word its text and the index of its last character."""
    clean, targets, cursor = [], [], 0
    for match in MARKED.finditer(marked):
        clean.append(marked[cursor : match.start()])
        start = sum(len(piece) for piece in clean)
        targets.append((match.group(1), start + len(match.group(1)) - 1))
        clean.append(match.group(1))
        cursor = match.end()
    clean.append(marked[cursor:])
    return "".join(clean), targets


def encode_from_bos(
    tokenizer: PreTrainedTokenizerBase, text: str
) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
    ids, offsets = list(encoded["input_ids"]), list(encoded["offset_mapping"])
    bos = tokenizer.bos_token_id
    if bos is not None and ids[0] != bos:
        ids, offsets = [bos, *ids], [(0, 0), *offsets]
    return ids, offsets


def token_at(offsets: list[tuple[int, int]], char: int) -> int:
    for position, (start, end) in enumerate(offsets):
        if start <= char < end and end > start:
            return position
    raise ValueError(f"no token covers character {char}")


class NeuronProbe:
    """Keeps the gate and up pre-activations of one neuron from the last forward pass."""

    def __init__(self, mlp: torch.nn.Module, neuron: int) -> None:
        self.neuron = neuron
        self.gate = torch.empty(0)
        self.up = torch.empty(0)
        mlp.gate_proj.register_forward_hook(self._keep("gate"))
        mlp.up_proj.register_forward_hook(self._keep("up"))

    def _keep(self, name: str):
        def hook(_module, _inputs, output):
            setattr(self, name, output[0, :, self.neuron].float().cpu())

        return hook

    @property
    def act(self) -> torch.Tensor:
        return torch.nn.functional.silu(self.gate) * self.up


def read_item(
    model: Qwen3Model,
    tokenizer: PreTrainedTokenizerBase,
    probe: NeuronProbe,
    group: str,
    item: PromptItem,
) -> ItemReading:
    text, marked = strip_markers(item.text)
    ids, offsets = encode_from_bos(tokenizer, text)
    with torch.no_grad():
        model(input_ids=torch.tensor([ids], device=model.device))
    tokens = tokenizer.convert_ids_to_tokens(ids)
    act = probe.act
    targets = []
    for word, last_char in marked:
        position = token_at(offsets, last_char)
        targets.append(
            TargetReading(
                word=word,
                token=tokens[position],
                position=position,
                gate=float(probe.gate[position]),
                up=float(probe.up[position]),
                act=float(act[position]),
            )
        )
    return ItemReading(
        group=group,
        id=item.id,
        text=text,
        targets=targets,
        tokens=tokens,
        act_per_token=[round(float(a), 4) for a in act],
    )


def print_group(group: PromptGroup, readings: list[ItemReading]) -> None:
    print(f"\n## {group.name}\nprediction: {group.prediction}")
    print(f"{'id':<22} {'word':<14} {'token':<14} {'gate':>7} {'up':>7} {'act':>7}")
    for reading in readings:
        for target in reading.targets:
            print(
                f"{reading.id:<22} {target.word:<14} {target.token!r:<14} "
                f"{target.gate:>7.3f} {target.up:>7.3f} {target.act:>7.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="marin-community/marin-32b-base")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--neuron", type=int, default=19311)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    prompt_set = PromptSet.model_validate(yaml.safe_load(args.prompts.read_text()))
    folder = truncated_checkpoint(args.model, args.layer + 1)
    model = Qwen3Model.from_pretrained(
        folder, dtype=torch.bfloat16, device_map=args.device
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    probe = NeuronProbe(model.layers[args.layer].mlp, args.neuron)

    readings = []
    for group in prompt_set.groups:
        group_readings = [
            read_item(model, tokenizer, probe, group.name, item) for item in group.items
        ]
        print_group(group, group_readings)
        readings.extend(group_readings)

    args.out.write_text(json.dumps([r.model_dump() for r in readings], indent=1))
    print(f"\n[done] {len(readings)} prompts, written to {args.out}")


if __name__ == "__main__":
    main()
