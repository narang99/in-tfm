# Our neuron clusters against Gemma Scope 2 SAE latents

Branch `exp/gemma-sae-features`.

## Question

- How do the concepts our pipeline extracts for a neuron differ from the features an SAE extracts?
- Gemma 3 270m has public SAEs on every layer (Gemma Scope 2), so it is the testbed.

## Setup

- Model: `google/gemma-3-270m`, layer 11, `q_proj`, neurons 90 and 300.
  - These are the neurons of the earlier layer 11 ablations (`experiments/hdbscan-leaf`).
- Corpus: 100 wikitext-2 articles, 2048 tokens each, seed 0.
  - That is about 200k tokens, far fewer than the SAE was trained on.
- Hits: bucketed sampling, as in a neuron report: 10 activation buckets, 10,000 hits per neuron.
  - The first run used elbow selection (12,925 and 5,977 hits). Its numbers are kept below for reference.
- Our clusters: HDBSCAN (`leaf`, `min_cluster_size` 10) on the L2-normalised Hadamard products.
- SAE: `gemma-scope-2-270m-pt-res-all`, `layer_10_width_16k_l0_small`, through sae-lens.
  - It reads `blocks.10.hook_resid_post`, the residual entering layer 11.
  - The neuron reads that same residual, after layer 11's `input_layernorm`.
- Backend: cuml HDBSCAN on a Colab T4.
- Commands:

```
python scripts/compare_sae_features.py --config configs/gemma3_q_proj_sae.yaml --result-json out/l10_16k_small.json
python scripts/inspect_sae_concepts.py out/l10_16k_small.hits.json --neuron 90
```

## How a neuron relates to an SAE

- The neuron computes w · norm(x), and Gemma's RMSNorm is x / rms(x) * (1 + γ).
  - So the activation is (w ⊙ (1 + γ)) · x / rms(x).
- The SAE writes x = Σ_j f_j d_j + b_dec + error.
- Substituting splits the activation exactly into
  - one term per active latent, f_j (w_eff · d_j) / rms(x),
  - a decoder-bias term,
  - an error term.
- Our Hadamard product splits the same dot product per residual coordinate instead.
- For the SAE-basis clustering, each hit's latent terms form a vector.
  - It is L2-normalised and clustered with the same HDBSCAN settings as ours.

## Checks

- The recomputed activation matches the captured one to 1.3e-6 for both neurons.
  - So the norm folding is right.
- On 4,000 random non-BOS tokens the SAE has L0 22.8 against a target of 20, and FVU 0.041.
  - So the SAE is fed the activation it was trained on.

## Numbers

Bucketed hits, the main run:

| | neuron 90 | neuron 300 |
|---|---|---|
| hits | 10,000 | 10,000 |
| our clusters / noise | 51 / 76% | 43 / 85% |
| SAE-basis clusters / noise | 73 / 72% | 90 / 73% |
| median share of activation from latents | 0.40 | 0.43 |
| from the decoder bias | 0.47 | 0.25 |
| from the SAE error | 0.13 | 0.32 |
| median top latent's share of the latent sum | 0.27 | 0.27 |
| median latents to reach 80% | 6 | 6 |
| distinct top latents | 1,064 | 1,320 |

| pair (bucketed) | neuron 90 | hits | neuron 300 | hits |
|---|---|---|---|---|
| ours vs SAE clusters | 0.89 | 1,273 | 0.89 | 920 |
| ours vs top latent | 0.47 | 2,369 | 0.68 | 1,499 |
| ours vs token | 0.60 | 2,369 | 0.59 | 1,499 |
| SAE clusters vs token | 0.53 | 2,830 | 0.51 | 2,673 |

- The bias share rises for neuron 90 because bucketed hits include weak activations.
  - There the constant bias term is a larger fraction of a small activation.
- At clusters of 20 hits or more:
  - neuron 90 has 24 concepts found by both, 26 found only in the SAE basis, and 14 only by us,
  - neuron 300 has 14 found by both, 38 only in the SAE basis, and 13 only by us.
- The concepts read the same as in the elbow run below.
  - New SAE-only ones: `even though` / `While` / `As for` (latent 311), `an` (latent 777), and `the [decline/form/case] of` (latent 1128).
- Report page with every concept and its example contexts: https://claude.ai/artifact/Qfcwa5jFTyk4wLEUWKivvf
  - Built from `bucketed/cards_bucketed.json`, written by `inspect_sae_concepts.py --cards-json`.

Elbow hits, the first run:

| | neuron 90 | neuron 300 |
|---|---|---|
| hits | 12,925 | 5,977 |
| our clusters / noise | 74 / 75% | 23 / 88% |
| SAE-basis clusters / noise | 89 / 68% | 56 / 75% |
| median share of activation from latents | 0.52 | 0.52 |
| from the decoder bias | 0.35 | 0.16 |
| from the SAE error | 0.14 | 0.32 |
| median top latent's share of the latent sum | 0.26 | 0.25 |
| median latents to reach 80% | 7 | 7 |
| distinct top latents | 740 | 853 |

Adjusted Rand for the elbow run, each pair over the hits neither side calls noise:

| pair | neuron 90 | hits | neuron 300 | hits |
|---|---|---|---|---|
| ours vs SAE clusters | 0.94 | 1,894 | 0.87 | 440 |
| ours vs top latent | 0.62 | 3,258 | 0.43 | 729 |
| ours vs token | 0.36 | 3,258 | 0.47 | 729 |
| SAE clusters vs token | 0.50 | 4,078 | 0.48 | 1,479 |

- The 0.94 and 0.87 hold on 15% and 7% of the hits only.
  - Those are the dense cores both sides agree on.
  - The index says nothing about the rest, so the concepts below were read by hand.

## Concepts, read from example contexts (elbow run)

### Where both sides find the same concept

- Neuron 90, wikitext formatting, one latent each:
  - `=` opening a heading after a newline (latent 91 tops 92% of the cluster).
  - `=` closing a heading (jaccard 0.88).
  - the space before the newline that ends a title line (jaccard 0.92).
  - the newline after `= Title =` (jaccard 0.97).
  - the possessive `'` in `Madero's` (latent 367, jaccard 0.85).
- Neuron 90, content:
  - the surname or second part of a person's name, as in `Karl [Blake] Bashoff` (latent 441).
  - word pieces inside Indian and other non-English names, as in `Nag[av]arma`.
- Neuron 300:
  - possessive `'` (jaccard 1.00).
  - `each` / `Each` / `every` (latent 799, jaccard 0.94).
  - `=` after a newline (jaccard 0.97).
  - `used` / `found` / `discovered` after an auxiliary (jaccard 0.62).
- On these, the SAE often finds the same concept with one latent.
  - The SAE-basis clustering still fragments it, or calls much of it noise.
  - So the top latent matches our clusters better than the SAE clustering does here.

### Concepts only the SAE basis finds

- Each of these is a coherent SAE cluster, and our clustering calls 75-100% of its hits noise.
- Neuron 90:
  - the newline after a sentence ends a paragraph (latent 525, 453 hits, 88% our noise).
  - the first word of a section heading, as in `= = [Race] = =` (latent 651, 296 hits).
  - a sentence-final `.` (latent 575, 179 hits).
  - `on` before a date or a place (latent 139).
  - the first piece of a place name, as in `[T]imbuktu`, `[Sc]apa Flow` (latent 18).
  - `of` in `the city of Warwick` (latent 16177).
- Neuron 300, mostly meaning that spans several tokens:
  - size words: `larger`, `small` (latent 3601).
  - width words: `wide`, `narrow` (latent 12201).
  - `short`, `shortness` (latent 5495).
  - `death`, `dead`, `deaths` (latent 1863).
  - `first`, `last`, `oldest known` (latent 147).
  - nationality adjectives: `Soviet`, `British`, `French` (latent 14353).
  - abstract nouns after `the`: `role`, `wishes`, `usefulness` (latent 1128).
  - auxiliaries `has`, `was`, `be` (latent 1324).
  - `Central` in proper names (latent 7638).

### Concepts only we find, or find differently

- Neuron 90:
  - `@-@` hyphens, which the SAE spreads over many latents (latent 174 tops only 20%).
  - the digits of a year, as in `19[7]8` (latent 840 tops 15%).
  - the `ʻ` in `Hawaiʻi` (no SAE cluster).
  - `)` closing a title, as in `= Species ( film [)] =`.
- Neuron 300:
  - words of song and album titles in quotes (`"Sweet Love"`, `"Bossy"`).
  - section-heading words (`Description`, `Geology`, `Conception`), which have no single SAE latent (the top one covers 11%).
- These are mostly one token type, or one formatting context, with many latents mixed in.

## Findings

- **The two bases agree on concepts tied to one token or one formatting slot.**
  - There, one SAE latent usually covers most of our cluster.
- **The SAE basis finds semantic concepts that span several tokens, and we miss them.**
  - Examples: `wide`/`narrow`, `death`/`dead`/`deaths`, size words, nationality words.
  - In the Hadamard basis these hits are spread out.
    - The token's own embedding dominates the residual, and so dominates w ⊙ x.
    - Two different tokens with a shared meaning are far apart there, so HDBSCAN cannot group them.
  - The SAE separates the meaning direction from the token, so the hits sit together.
- **Much of what the neuron reads is outside the SAE's latents.**
  - Latents carry about half the activation.
  - The decoder bias carries 16-35%, a constant offset from the mean residual.
  - The error carries 14-32%, and up to 100% for some of our clusters (neuron 300 cluster 14).
  - So for parts of these neurons the coordinate basis keeps signal the SAE throws away.
- **One hit is not one feature.**
  - The top latent carries only about a quarter of the latent sum, and 80% needs about 7 latents.
- **Weights alone would not find the driving latents.**
  - The latents that top the most hits sit at read-strength ranks in the hundreds or thousands.
  - So how often a latent fires matters as much as how strongly the neuron reads it.

## Is the small corpus the reason we miss concepts?

- Partly, but not mainly.
- What the corpus size changes:
  - The SAE's latents were learned on far more text than ours, and we only apply them here.
    - So the SAE can name a concept from 20 hits.
    - Our HDBSCAN needs at least 10 hits close together in 640 dimensions.
  - Wikitext is heavy on formatting (`= = Heading = =`, `@-@`), and that formatting dominates both sides' clusters.
- What it does not explain:
  - Several missed concepts have hundreds of hits.
    - Neuron 90's paragraph-ending newline has 453 hits, and we call 88% of them noise.
  - We fragment that same latent into small clusters (19 and 16 hits) plus noise.
  - So the misses come from the Hadamard geometry, not from too few examples.
- To test it directly, re-run with a larger and more varied corpus.
  - If the corpus is the cause, the SAE-only concepts should start to appear among our clusters.

## Not yet done

- A larger corpus, as above.
- A wider SAE (262k) and a larger L0, to see whether the SAE-only concepts survive.
- More neurons, and `k_proj`.
- The bucketed run's summary and cards are in `bucketed/`; its 5 MB hits file is not committed.
  - The commands above regenerate everything in about 5 minutes on a T4.
