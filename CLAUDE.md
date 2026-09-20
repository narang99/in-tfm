# Coding and documentation guidelines

- Prefer small readable functions which do one thing. A function can be created to make code more readable even if the code block is not reused.
- Prefer code which conveys meaning using descriptive function and variable names. Don't write docstrings for every small function. Make sure you never simply write a docstring that restates what the function definition says
- Docstrings should talk about why/how of non-trivial parts. Don't state simple things. 
- Prefer bullet points. Don't merge sentences into more complex sentences which are harder to read.
- Always prefer strong typing, along with descriptive type names for long type definitions. 
  - Pydantic is preferred over dataclasses since it gives easy JSON marshalling
  - Use jaxtyping for typing tensors
- Don't use matplotlib for non-interactive work (batch image saving, report generation, etc.) - it's slow for this. Prefer compositing with PIL directly.
  - Reserve matplotlib for interactive/exploratory plotting (notebook `show_*` helpers).

# Running experiments on Colab

Measured on a T4, 2026-09-20. Details and timings in `COLAB_PLAN.md`.

## Dependencies

- Colab is a pre-built environment, not an empty one. Every floor in `pyproject.toml` above its
  preinstall forces a reinstall.
  - `torch`: uv swaps the `+cu128` build for a generic PyPI wheel and the GPU silently disappears.
  - `numpy`: it is the shared ABI floor under torch, scikit-learn, scipy and cuml.
- So floors state what the code needs, not what `uv lock` picked on a mac. Check Colab's actual
  versions before raising one.
- Install our own package with `--no-deps -e`, then only the deps Colab lacks. Without `--no-deps`
  uv re-resolves the whole graph even when the floors are right.
- `-e` is mandatory. Without it `git pull` updates the checkout while the script keeps importing
  the copy in `dist-packages`, so an edit has no effect and the re-run silently produces identical
  numbers. This failure mode looks like a broken fix, not a broken install.

## Verify the environment, don't trust it

- Assert in the setup cell: `torch.cuda.is_available()` and `"+cu" in torch.__version__`. A
  runtime you were told is a GPU can come back as CPU, and the install can strip CUDA.
- Print versions before *and* after install - that is what makes a torch swap visible.
- cuml ships preinstalled on GPU runtimes, so GPU HDBSCAN is free; no RAPIDS install needed.

## Notebook mechanics

- Run the real work as a subprocess (`!python ...`), not in the kernel. Fresh imports each time,
  and `patch_for_attn_lrp`'s process-wide class patching cannot leak between runs.
- The kernel caches imports, so verifying an editable reinstall in-kernel gives a false negative.
  Verify in a subprocess too.
- `colab-mcp` always opens Colab's scratchpad (`/notebooks/empty.ipynb`, hardcoded) - it cannot
  target a notebook. The scratchpad is a persistent singleton: it survives across sessions, but
  every experiment shares it. Confirm what you are attached to with
  `_message.blocking_request("get_ipynb")` before writing cells.
- Keep experiment config in the repo, not in cells. The notebook clones, installs and invokes;
  anything that distinguishes one experiment from another belongs in git where it is reviewable.

## Porting is a bug detector

- A second machine with a different filesystem is a free test for order-dependence. The
  `Path.glob` bug (same flag, different images on each machine) was only visible once there were
  two machines.
- Re-run the baseline after fixing such a bug. Comparing against a stale pre-fix artifact
  produced a speedup claim that was off by 2.5x.
- Swapping a backend for speed changes results, not just runtime - cuml and CPU hdbscan agree on
  tight clusters and disagree on diffuse ones. Verify rather than assuming a drop-in is
  semantically free.
