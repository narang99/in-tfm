#!/usr/bin/env python
"""Prints example contexts for the concepts each side found, from compare_sae_features' hits file.

    python scripts/inspect_sae_concepts.py sae_comparison.hits.json --neuron 90

- Each of our clusters, next to the SAE cluster it overlaps most, and the hits only one side has.
- SAE clusters that match none of ours, and the other way round.
- Latents that top many hits our clustering calls noise.

With `--cards-json`, it instead writes one card per cluster on either side, for every neuron, to
build a report page from.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from in_tfm.sae_comparison import NOISE, HitTable, HitTables

MATCH_JACCARD = 0.3


def load_table(path: Path, neuron: int) -> HitTable:
    tables = HitTables.model_validate_json(path.read_text()).neurons
    return next(t for t in tables if t.neuron == neuron)


def examples(contexts: list[str], members: np.ndarray, n: int, rng: np.random.Generator) -> list[str]:
    picked = rng.choice(np.flatnonzero(members), size=min(n, int(members.sum())), replace=False)
    return [contexts[i].replace("\n", "⏎")[-70:] for i in picked]


def show(title: str, contexts: list[str], members: np.ndarray, n: int, rng: np.random.Generator) -> None:
    print(f"    {title} ({int(members.sum())} hits)")
    for line in examples(contexts, members, n, rng) if members.any() else []:
        print(f"      | {line}")


def best_match(members: np.ndarray, other: np.ndarray) -> tuple[int, float]:
    candidates = [c for c in set(other[members].tolist()) if c != NOISE]
    if not candidates:
        return NOISE, 0.0
    scores = {c: (members & (other == c)).sum() / (members | (other == c)).sum() for c in candidates}
    best = max(scores, key=scores.get)
    return best, float(scores[best])


def by_size(labels: np.ndarray) -> list[int]:
    return [c for c, _ in Counter(labels[labels != NOISE].tolist()).most_common()]


def top_latent_of(members: np.ndarray, top_latent: np.ndarray) -> str:
    latent, count = Counter(top_latent[members].tolist()).most_common(1)[0]
    return f"latent {latent} tops {count / members.sum():.0%}"


class ConceptCard(BaseModel):
    side: str
    """"ours" or "sae"."""
    cluster: int
    n_hits: int
    top_latent: int
    top_latent_share: float
    match: int
    """The other side's cluster with the highest jaccard, -1 when every shared hit is noise there."""
    jaccard: float
    other_side_noise: float
    """Fraction of this cluster's hits the other side calls noise."""
    examples: list[str]


class NeuronCards(BaseModel):
    neuron: int
    n_hits: int
    cards: list[ConceptCard]


def concept_cards(table: HitTable, n_examples: int) -> NeuronCards:
    labels = {"ours": np.array(table.ours), "sae": np.array(table.sae)}
    top_latent = np.array([row[0] for row in table.top_latents])
    rng = np.random.default_rng(0)
    cards = []
    for side, other in (("ours", "sae"), ("sae", "ours")):
        for cluster in by_size(labels[side]):
            members = labels[side] == cluster
            match, jaccard = best_match(members, labels[other])
            latent, count = Counter(top_latent[members].tolist()).most_common(1)[0]
            cards.append(
                ConceptCard(
                    side=side, cluster=cluster, n_hits=int(members.sum()), top_latent=latent,
                    top_latent_share=count / members.sum(), match=match, jaccard=jaccard,
                    other_side_noise=float((labels[other][members] == NOISE).mean()),
                    examples=examples(table.context, members, n_examples, rng),
                )
            )
    return NeuronCards(neuron=table.neuron, n_hits=len(table.ours), cards=cards)


def write_cards(hits_json: Path, out: Path, n_examples: int) -> None:
    tables = HitTables.model_validate_json(hits_json.read_text()).neurons
    cards = [concept_cards(t, n_examples).model_dump() for t in tables]
    out.write_text(json.dumps(cards, ensure_ascii=False, separators=(",", ":")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("hits_json", type=Path)
    parser.add_argument("--neuron", type=int)
    parser.add_argument("--cards-json", type=Path)
    parser.add_argument("--n-clusters", type=int, default=15)
    parser.add_argument("--n-examples", type=int, default=4)
    args = parser.parse_args()
    if args.cards_json:
        write_cards(args.hits_json, args.cards_json, args.n_examples)
        return

    table = load_table(args.hits_json, args.neuron)
    ours, sae = np.array(table.ours), np.array(table.sae)
    top_latent = np.array([row[0] for row in table.top_latents])
    contexts, rng, n = table.context, np.random.default_rng(0), args.n_examples

    print(f"=== neuron {args.neuron}: our clusters, largest first, against their best SAE cluster")
    for cluster in by_size(ours)[: args.n_clusters]:
        mine = ours == cluster
        match, jaccard = best_match(mine, sae)
        print(f"\n  ours {cluster} -> sae {match}, jaccard {jaccard:.2f}, {top_latent_of(mine, top_latent)}")
        show("ours", contexts, mine, n, rng)
        if match != NOISE:
            show("only in the SAE cluster", contexts, (sae == match) & ~mine, n, rng)
        show("ours that the SAE calls noise", contexts, mine & (sae == NOISE), n, rng)

    print(f"\n=== SAE clusters matching none of ours (jaccard < {MATCH_JACCARD}), largest first")
    for cluster in by_size(sae):
        members = sae == cluster
        if best_match(members, ours)[1] < MATCH_JACCARD:
            print(f"\n  sae {cluster}: {(ours[members] == NOISE).mean():.0%} our noise, {top_latent_of(members, top_latent)}")
            show("sae", contexts, members, n, rng)

    print(f"\n=== our clusters matching no SAE cluster (jaccard < {MATCH_JACCARD}), largest first")
    for cluster in by_size(ours):
        members = ours == cluster
        if best_match(members, sae)[1] < MATCH_JACCARD:
            print(f"\n  ours {cluster}: {(sae[members] == NOISE).mean():.0%} SAE noise, {top_latent_of(members, top_latent)}")
            show("ours", contexts, members, n, rng)

    print("\n=== latents topping the most hits that we call noise")
    noise_tops = Counter(top_latent[ours == NOISE].tolist()).most_common(8)
    for latent, count in noise_tops:
        print(f"\n  latent {latent}: tops {count} of our noise hits")
        show("hits", contexts, (top_latent == latent) & (ours == NOISE), n, rng)


if __name__ == "__main__":
    main()
