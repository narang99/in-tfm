#!/usr/bin/env python
"""Heatmaps of neuron spectra (neurons grouped, frequencies across) for chosen states, drawn with PIL.

    uv run python scripts/plot_grokking_spectra.py --states '[4400,6000,40000]' --out-dir experiments_grokking/figures
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pydantic_settings import BaseSettings, SettingsConfigDict

from in_tfm.grokking_groups import group_by_spectrum
from in_tfm.grokking_circuits import hellinger_spectra
from in_tfm.grokking_fourier import mlp_frequency_spectra
from in_tfm.models.grokking import load_model
from in_tfm.neuron_tracking import state_path

CELL = 8


class PlotConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True, cli_parse_args=True)

    states: list[int]
    states_dir: Path = Path("checkpoints/grokking/states")
    out_dir: Path


def heat(value: float) -> tuple[int, int, int]:
    level = int(255 * (1 - min(value, 1.0)))
    return (level, level, 255)


def draw_spectra(shares: np.ndarray, labels: np.ndarray, title: str) -> Image.Image:
    order = np.lexsort((shares.argmax(axis=1), labels))
    n, freqs = shares.shape
    margin = 24
    image = Image.new("RGB", (freqs * CELL + 2 * margin, n * 1 + 2 * margin), "white")
    draw = ImageDraw.Draw(image)
    draw.text((margin, 6), title, fill="black")
    for row, neuron in enumerate(order):
        for f in range(freqs):
            value = shares[neuron, f]
            if value > 0.02:
                draw.rectangle([margin + f * CELL, margin + row, margin + (f + 1) * CELL - 1, margin + row], fill=heat(value * 2))
    boundaries = np.where(np.diff(labels[order]) != 0)[0] + 1
    for b in boundaries:
        draw.line([margin, margin + b, margin + freqs * CELL, margin + b], fill=(200, 0, 0))
    return image


def main() -> None:
    config = PlotConfig()
    config.out_dir.mkdir(parents=True, exist_ok=True)
    for state in config.states:
        model = load_model(str(state_path(config.states_dir, state)))
        spectra = mlp_frequency_spectra(model)
        labels = group_by_spectrum(spectra)
        shares = spectra / spectra.sum(axis=1, keepdims=True)
        path = config.out_dir / f"mlp_spectra_state_{state:05d}.png"
        draw_spectra(shares, labels, f"state {state}: 512 MLP neurons (rows, grouped) x frequency 1..56 (columns), red lines split groups").save(path)
        print(path)


if __name__ == "__main__":
    main()
