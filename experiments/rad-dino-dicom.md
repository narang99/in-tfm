# RadDino on chest X-ray DICOMs

The project started here, on `microsoft/rad-dino` (DINOv2) over DICOM chest X-rays.

## What it did

- Neuron reports on an MLP `fc2` layer of the vision encoder.
- AttnLRP relevance in pixel space, drawn as overlays on the X-ray.
- The text pipeline was ported from it.
  - The port plan was `PLAN.md`.
  - The first Colab setup notes were `COLAB_PLAN.md`.

## Reproduce

- Removed from `main`. The last commit on `main` with it is [`0c79252`](https://github.com/narang99/in-tfm/tree/0c792524ada4568bd24eac7bfea2795820e6a864).
  - Adapter: [`src/in_tfm/models/rad_dino.py`](https://github.com/narang99/in-tfm/blob/0c792524ada4568bd24eac7bfea2795820e6a864/src/in_tfm/models/rad_dino.py)
  - Source: [`src/in_tfm/sources/dicom.py`](https://github.com/narang99/in-tfm/blob/0c792524ada4568bd24eac7bfea2795820e6a864/src/in_tfm/sources/dicom.py)
  - Config: [`configs/rad_dino_fc2.yaml`](https://github.com/narang99/in-tfm/blob/0c792524ada4568bd24eac7bfea2795820e6a864/configs/rad_dino_fc2.yaml)
  - Plans: [`PLAN.md`](https://github.com/narang99/in-tfm/blob/0c792524ada4568bd24eac7bfea2795820e6a864/PLAN.md), [`COLAB_PLAN.md`](https://github.com/narang99/in-tfm/blob/0c792524ada4568bd24eac7bfea2795820e6a864/COLAB_PLAN.md)
- `git checkout 0c792524a`, put DICOMs in `dicoms/`, then:
  - `uv run python scripts/run_neuron_report.py --config configs/rad_dino_fc2.yaml --image.n-dicoms 8`
