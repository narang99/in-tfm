# Running the neuron-report experiment on Colab via colab-mcp

Goal: stop running 518×518 ViT forward/backward on MPS. Keep a Colab T4 notebook open, drive it
from Claude Code over MCP, iterate there. Keep the Mac path working.

## 0. Target environment (measured, not assumed)

| | Colab (T4, py3.13.15) | Mac (py3.13.7) |
|---|---|---|
| torch | 2.11.0+cu128 | 2.14.0 |
| transformers | 5.16.1 | 5.17.0 |
| numpy | 2.1.3 | 2.5.3 |
| scikit-learn | 1.6.1 | 1.9.1 |
| hdbscan | 0.8.44 | 0.8.44 |
| pillow | 11.3.0 | — |

`requires-python = ">=3.13"` stays as-is — Colab is 3.13.15.

Both `microsoft/rad-dino` and `ndonyapour/dicom-sample-files` are `gated: False` on the HF API,
so the notebook pulls model and data anonymously. No secrets are read or referenced.

## 1. The real constraint: every high floor in `pyproject.toml` forces a reinstall

The current pins were whatever `uv lock` happened to pick on the Mac, not what the code needs.
Four of them sit *above* Colab's preinstalls, and two of those are actively dangerous:

- **`torch>=2.14.0` vs 2.11.0+cu128** — uv would pull a generic PyPI wheel (~2.5GB) over the
  cu128 build matched to the T4 driver. This is the one that breaks the runtime outright.
- **`numpy>=2.5.3` vs 2.1.3** — numpy is the shared ABI floor under torch, scikit-learn, scipy
  and cuml. Upgrading it risks breaking all four at once, for nothing we use.
- **`scikit-learn>=1.9.1` vs 1.6.1** — we import exactly one symbol, `preprocessing.normalize`.
  Ancient API. The floor is pure noise and drags numpy/scipy with it.
- **`transformers>=5.17.0` vs 5.16.1** — both are 5.x, so `AttentionInterface` and the dropped
  `find_pruneable_heads_and_indices` both behave the way `attribution.py` and `compat.py` assume.
  Staying on Colab's 5.16.1 costs nothing and avoids a reinstall.

Plus one dependency that is **entirely unused** — I grepped `src/` and `scripts/`: zero hits for
pandas. `pandas>=3.0.6` against Colab's 2.x would have been another large compiled reinstall for
a package we never import.

### Proposed `pyproject.toml` delta

| package | now | proposed | why |
|---|---|---|---|
| `torch` | `>=2.14.0` | `>=2.11` | keep Colab's cu128 build |
| `numpy` | `>=2.5.3` | `>=2.1` | shared ABI; don't disturb |
| `scikit-learn` | `>=1.9.1` | `>=1.3` | only `normalize` is used |
| `transformers` | `>=5.17.0` | `>=5.16` | 5.x is what matters, not the minor |
| `matplotlib` | `>=3.11.2` | `>=3.8` | Colab ships ~3.10 |
| `pydantic` | `>=2.13.5` | `>=2.7` | nothing version-specific in use |
| `hdbscan` | `>=0.8.44` | `>=0.8.39` | see §2 |
| `pandas` | `>=3.0.6` | **removed** | unused |
| `jupyter`, `ipykernel` | deps | dev group | not needed on Colab; big tree |
| `requires-python` | `>=3.13` | unchanged | Colab is 3.13.15 |

Lowering a floor doesn't change the Mac: `uv lock` keeps resolving to the newer versions already
installed there. I'll re-run the Mac smoke test to confirm.

Net effect on Colab — the only fresh installs are `captum`, `jaxtyping`, `lxt`, `nnsight`,
`pydicom`, `rad-dino` (+`einops`). All light, all pure-Python-ish. **Nothing compiled is touched.**

## 2. Clustering: cuml when present, hdbscan otherwise

New `src/in_tfm/clustering.py` exposing one generic function:

```python
def cluster_labels(
    points: Float[np.ndarray, "n_points dim"], min_cluster_size: int
) -> Int[np.ndarray, "n_points"]:
```

Backend resolution, cached at module level so it happens once:

- try `cuml.cluster.HDBSCAN` → use it, **return before importing hdbscan**
- else `hdbscan.HDBSCAN`
- log which one was chosen, so it shows up in cell stdout

That early return is deliberate. You've hit hdbscan/cuml API-compat breaks before, and the
version that coexists cleanly is 0.8.39. Rather than pin Colab down to a version with no cp313
wheel (0.8.39 predates 3.13, so it would mean a source build), the resolver just never imports
the CPU package when cuml is live — they're never in the same process. `hdbscan>=0.8.39` records
your known-good floor and is already satisfied by Colab's 0.8.44, so nothing reinstalls. If an
API break shows up anyway, *then* we pin `==0.8.39` and eat the build.

`scripts/run_neuron_report.py` drops `from hdbscan import HDBSCAN` and calls `cluster_labels`.
`default_device()` already returns `"cuda"` on Colab with no change.

### I was wrong last round about cuml being a nice-to-have

I said clustering was only 10s of the 581s Mac run, so GPU HDBSCAN didn't matter. That holds at
8 DICOMs and breaks at the config we actually want. HDBSCAN on 3072-dim vectors degrades to
near-quadratic, and hit count scales with DICOM count:

```
  8 dicoms -> ~1195 hits, ~  9.9s/neuron CPU,   1.8 min for neurons 90-100
 44 dicoms -> ~6572 hits, ~299.2s/neuron CPU,  54.8 min for neurons 90-100
```

So at the target config CPU clustering becomes the **dominant** cost — worse than the AttnLRP
passes it was rounding-error against on the Mac. If cuml turns out not to be preinstalled, the
~2-3 min RAPIDS install is worth paying, which is the opposite of what I told you.

## 3. Notebook — three cells

Split so the slow cell caches across iterations.

**Cell 1 — environment (once per runtime)**
```python
!pip install -q uv
!git clone https://github.com/narang99/in-tfm /content/in-tfm
!uv pip install --system -e /content/in-tfm
import torch, transformers, numpy, sklearn
print(torch.__version__, torch.cuda.get_device_name(0))
print(transformers.__version__, numpy.__version__, sklearn.__version__)
try:
    import cuml; print("cuml", cuml.__version__)
except ImportError:
    print("no cuml -> CPU hdbscan (see plan §2)")
```
`origin/main` is at `0cdfef4 rad dino works` and in sync, so a plain clone gets the fixed code.
`-e` matters: later `git pull` picks up edits with no reinstall.

**Cell 2 — data (once per runtime)**
```python
from huggingface_hub import hf_hub_download
import zipfile, pathlib
z = hf_hub_download("ndonyapour/dicom-sample-files", "CT-chest.zip", repo_type="dataset")
out = pathlib.Path("/content/dicoms"); out.mkdir(exist_ok=True)
with zipfile.ZipFile(z) as f:
    for n in f.namelist():
        if n.endswith(".dcm"):
            (out / pathlib.Path(n).name).write_bytes(f.read(n))
print(len(list(out.glob("*.dcm"))), "dicoms")
```

**Cell 3 — the run (re-edited each iteration)**
```python
!cd /content/in-tfm && git pull -q
!MPLBACKEND=Agg python /content/in-tfm/scripts/run_neuron_report.py \
    --dcm-dir /content/dicoms --out-dir /content/reports \
    --n-dicoms 8 --batch-size 4 --neuron-start 90 --neuron-end 90 \
    --max-hits-per-cluster 4 --min-cluster-size 10
```

Each `!python` is a fresh subprocess, which keeps `patch_for_attn_lrp`'s process-wide
LayerNorm/Dropout class patching from leaking between runs. Worth remembering if we ever move
the run inline into the kernel — then a re-run inherits the patches and needs a kernel restart.

## 4. Two stages, not one

**Stage 1 — parity.** Cell 3 exactly as above: the same 8 DICOMs / neuron 90 / 4 hits config that
produced `reports/neuron_90` on the Mac. Should take ~1 min. I check cluster count and threshold
against the Mac run (`threshold=0.5627`, 1195 hits, 11 clusters, 7 surviving). If those don't
match, the port is wrong and scaling up would just hide it.

**Stage 2 — the real run.** 44 DICOMs, neurons 90–100, `--max-hits-per-cluster 10`. Feasible only
with the clustering backend settled in §2.

## 5. Artifacts out (no persistent storage)

- **For me — stdout.** `meta.json` plus timings is all I need to judge an iteration:
  ```python
  import json, glob
  for p in sorted(glob.glob("/content/reports/*/meta.json")):
      print(p, json.dumps(json.load(open(p)))[:600])
  ```
- **For you — the JPEGs.** `!zip -qr /content/reports.zip /content/reports` then
  `files.download(...)` through the tab you already have open. Drive mount as fallback if that's
  awkward under MCP — one interactive auth per runtime, but survives runtime death.

## 6. Order of work

1. `pyproject.toml` — the delta in §1
2. `src/in_tfm/clustering.py` + wire into `scripts/run_neuron_report.py`; fix the stale
   "Clustering runs on CPU via sklearn-contrib hdbscan" paragraph in the module docstring
3. Re-run the Mac smoke test (8 DICOMs, neuron 90) — confirms lowered floors broke nothing
4. Commit, push to `main`
5. `open_colab_browser_connection` — needs you at the keyboard, 60s auth timeout
6. Build cells 1–3, run stage 1, compare against the Mac numbers
7. Stage 2

## RESULTS (measured 2026-09-20, Tesla T4)

Everything below §0-§6 was the plan; this is what actually happened.

**The dependency strategy worked.** After `uv pip install --system --no-deps -e .` plus the six
light deps, Colab's stack was untouched:

```
torch        2.11.0+cu128 | cuda: True | Tesla T4
transformers 5.16.1
numpy        2.1.3
sklearn      1.6.1
cuml         26.02.000
device       cuda | cluster backend: cuml (gpu)
```

`--no-deps` on our own package is load-bearing. Installing it with deps lets uv re-resolve torch
and swap the +cu128 build for a generic wheel.

**cuml is preinstalled** (26.02.000) - no RAPIDS install needed, GPU clustering is free.

**Parity, 8 dicoms / neuron 90 / max-hits 4, both running the sorted-glob fix:**

|  | Mac (MPS) | T4 |
|---|---|---|
| dicoms selected | 0000-0001..0008 | identical |
| threshold | 0.6243 | 0.6243 |
| hits above threshold | 619 | 619 |
| clusters found | 7 | 6 |
| clusters kept | 4 | 5 |
| **total** | **234.69s** | **21.15s** |

**11.1x end to end.** (Not the ~27x an earlier note claimed - that compared against the
pre-fix mac baseline, which clustered differently and so ran a different number of LRP passes.
234.69s vs 21.15s is the only like-for-like pair.)

The deterministic half of the pipeline is identical across machines: same images, same
threshold to 4dp, same hit count. Everything upstream of clustering ports exactly.

**Clustering does not agree, and that is expected.** cuml's HDBSCAN and the CPU build are
different implementations, not the same algorithm on different hardware:

| cluster size | Mac (cpu hdbscan) | T4 (cuml) |
|---|---|---|
| | - | 287 |
| | 131 | 131 |
| | 41 | 44 |
| | 30 | 30 |
| | 13 | 13 |

They agree on the tight, well-separated clusters (131/30/13) and disagree on the diffuse mass:
cuml groups 287 points into a cluster that the CPU build leaves as noise. So a report generated
on Colab can legitimately show a large cluster that the same data on the mac does not, and
cluster ids are not comparable across backends. Worth remembering before reading anything into
a cluster that appears only on one of them.

## Two bugs the port exposed

**1. `Path.glob` order is filesystem-dependent.** `--n-dicoms 8` picked a different 8 images on
Mac than on Colab - only 3 of 8 overlapped - which moved the threshold (0.5627 vs 0.6585) and hit
count (1195 vs 630). It reads exactly like a broken port until you diff `dcm_paths` in
`meta.json`. Fixed in 4752ee1 by sorting the glob.

**2. A non-editable install silently ignores `git pull`.** Without `-e`, the script imports
`in_tfm` from dist-packages while `git pull` only updates the checkout, so an edit appears to
have no effect - the re-run produced byte-identical numbers. Cell 1 now uses `-e`.

## Still open

- `reports/` is never cleaned between runs, so cluster folders from a previous run with more
  clusters linger on disk. `index.html` only links current ones, so it is cosmetic, but a stale
  `cluster_N/` next to a fresh `meta.json` is misleading.
- `--max-hits-per-cluster` is still the dominant knob: 15.9s of the 21.2s total is LRP passes.
- Stop the runtime when idle. Credits burn on wall-clock, not compute.

## 7. Open risk, restated

The corpus is chest **CT**; rad-dino is a chest **X-ray** model. Fine as a pipeline smoke test,
but the clusters are not semantically meaningful. Once the loop is fast, swapping in real CXRs is
a cheap edit to cell 2 and the first thing worth doing with the speedup.
