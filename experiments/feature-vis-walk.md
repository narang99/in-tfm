# Does feature vis follow paths through a neuron's hits?

Run 2026-11-01. The hypothesis was not supported.

## Question

- Does activation maximisation of InceptionV1 mixed4e:55 walk through the density-peak forest of the neuron's real hits?

## Setup

- One-cell activation maximisation of mixed4e:55.
  - Raw pixels at 224, TV 1e4, lucent's redirected ReLU.
- Starts: grey, and 3 typical hits of tree 0, the background tree.
- Each snapshot's Hadamard vector is placed among the 14,240 hits.
  - Placement is by the forest's distance, and by direction.
- Trees, named from their core crops:
  - Background: tree 0, mostly cars.
  - Cats: trees 1, 2, 23 and 27.
  - Snouts: trees 20, 24, 31, 36 and 39.

## Code

- Branch [`worktree-feature-viz`](https://github.com/narang99/in-tfm/tree/worktree-feature-viz).
  - [`scripts/build_hit_space.py`](https://github.com/narang99/in-tfm/blob/7c8e27e6722d05faec9df1b931bb1c4008dd3877/scripts/build_hit_space.py) saves every hit's vector and the forest, in about 2 minutes on a T4.
    - It matched the saved report's forest exactly.
  - [`scripts/feature_vis_walk.py`](https://github.com/narang99/in-tfm/blob/7c8e27e6722d05faec9df1b931bb1c4008dd3877/scripts/feature_vis_walk.py) runs the climbs and places their snapshots.
  - [`scripts/forest_path_walk.py`](https://github.com/narang99/in-tfm/blob/7c8e27e6722d05faec9df1b931bb1c4008dd3877/scripts/forest_path_walk.py) runs the stepwise walks.

```
python scripts/feature_vis_walk.py --hit-space reports/hit_space/hit_space.npz --image-dir data/inception_pool \
    --cat-image cat.jpg --cat-trees 1,7 --snout-trees 8,20 --out reports/feature_vis_walk
python scripts/forest_path_walk.py --hit-space hit_space.npz --image-dir data/inception_pool \
    --target cats=1,2,23,27 --target snouts=20,24,31,36,39 --out reports/forest_path_walk
```

## Results: climbs

- No climb reached a cat core or a snout core.
- Paths matched straight pixel blends and Hadamard blends between the same two ends.
- From grey, feature vis grows car fronts.
  - All 16 uncapped seeds end on hit 11273, a white ambulance front in tree 0.
- A real cat photo's vector sits inside the hits, in tree 1 or 2, at 1.0 to 1.5 times the median neighbour distance.
  - Its geodesic to the snouts goes through cat muzzles, to the fox nose tree 24 or the dog muzzle tree 20.

## Results: stepwise walks

- Fitting the image to each next point on the geodesic, 64 steps each, reached every point in all 8 walks.
- A direct fit to the last point reached it within its first 64 steps.
  - Its MSE was lower in 7 of 8 walks.
  - So the path is not needed.
- The fits are adversarial.
  - Matching a fox nose's vector leaves the taxi image with a texture on top.

## Not tried

- lucent's FFT parameterisation, a natural-image prior.
  - Raw pixels can set one cell's vector to almost anything.
- A start inside a cat tree.
- Normalised distances for the forest.
  - Above the hits' activation range (about 1,561), the unnormalised distance places vectors by size.

## Outputs

- `s3://narang99-private/feature_vis_walk/`
  - `feature_vis_walk_2026-11-01.tar.gz` and `README.md` for the climbs, with `hit_space.npz` inside.
  - `forest_path_walk_2026-11-01.tar.gz` and `forest_path_walk_README.md` for the stepwise walks.
