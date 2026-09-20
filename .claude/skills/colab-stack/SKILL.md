---
name: colab-stack
description: >-
  Run an experiment on a Colab GPU by pushing to GitHub and pulling it there, then ship the
  result as a stack of dependent pull requests. Use this whenever work needs a GPU the local
  machine doesn't have, whenever a change is about to be tested on Colab, and whenever a branch
  is ready to open a PR for - including when the user says "test this on colab", "run it on the
  GPU", "my mac hangs", "raise a PR", "stack these", or asks to split a large change into
  reviewable pieces. Also use it when an experiment run needs verifying, because it carries the
  rules for telling a finished run from a crashed one.
---

# Colab experiments, shipped as stacked PRs

Two halves of one loop. They are in the same skill because the same constraint drives both:
**Colab can only see code that is already on GitHub.** Every iteration is a real commit, so
commit hygiene stops being optional and the branches you end up with are exactly the layers a
stack wants.

Dependency floors, install flags and notebook mechanics live in `CLAUDE.md` under "Running
experiments on Colab" and are already in context. Don't restate them here; this skill is the
loop and the shipping.

## The loop

Setup cells run once per VM. Only the run cell re-runs.

| cell | changes | purpose |
|---|---|---|
| setup | rarely | `pip install uv`, clone, `uv pip install --system --no-deps -e`, then the light deps |
| assert | rarely | print versions, assert `torch.cuda.is_available()` and `"+cu" in torch.__version__` |
| data | rarely | download the corpus |
| checkout | per branch | `git fetch && git checkout <branch> && git pull` then re-run the editable install |
| run | every iteration | `!MPLBACKEND=Agg python .../script.py <flags>` |

Then: **edit locally → commit → push → run the checkout cell → run the run cell.**

There is no way to shortcut the push. A dirty local tree is invisible to Colab, so "it works on
my machine" and "it works in the notebook" are different claims until you have pushed. When an
edit appears to have no effect, check that it was pushed *and* that the checkout cell ran, before
suspecting the code.

Keep the run in a subprocess (`!python`), never in the kernel. Fresh imports each time, and
process-wide monkey-patching can't leak between runs.

## Verifying a run

This is the part that goes wrong quietly, so it is worth more care than it looks.

**A run that printed plausible numbers is not a run that finished.** Reports often write their
metadata *before* rendering artifacts, so a crash halfway leaves a `meta.json` full of correct,
convincing values next to zero images. Reading those numbers and declaring success is a real
failure mode — it happened in this repo and produced a confident "verified, behaviour-preserving"
claim about a run that had exited 1.

So check, in this order:

1. the process exited 0
2. the **final** line the script prints (a total-time line, or whatever comes last) is present
3. the artifacts exist on disk — count the files, don't assume
4. only then compare the numbers

**Keep a regression target.** When refactoring, the previous run's threshold / hit count /
cluster count is worth more than any test you could write, because it exercises the whole
pipeline on real data. Record it in the commit message so the next person can check it.

**Re-run the baseline after any fix that changes inputs.** Comparing against a stale pre-fix
artifact produced a speedup claim here that was off by 2.5x. A baseline from before the fix is
not a baseline.

**A backend swap changes results, not just speed.** GPU and CPU implementations of the same
algorithm agree on easy cases and diverge on hard ones. Say which backend produced a number.

## Runtime hygiene

Release the VM when done — credits burn on wall-clock, not compute:

```python
from google.colab import runtime
runtime.unassign()
```

**Do not verify this by running a cell.** Any execution makes Colab allocate a fresh VM, so
checking that it is off turns it back on. Verification and the goal are mutually exclusive; ask
the user to glance at the runtime indicator instead.

## Shipping as a stack

A stack is a chain of PRs where the bottom targets `main` and each one above targets the branch
below it. Each layer shows only its own diff, so reviewers get small changes instead of one
large one, and you can start the next piece before the previous merges.

**Where to cut a branch:** when the concern changes. Foundational things — shared types, an
interface, a schema — go lower; whatever depends on them goes above. The rule that decides it:
*if code in one layer depends on code in another, the dependency must be in the same branch or a
lower one.* In practice a refactor that enables a feature is the bottom layer and the feature is
the top.

### With `gh stack` (preferred)

```bash
gh extension install github/gh-stack   # once; needs gh >= 2.0
gh stack init <bottom-branch>          # trunk defaults to the repo default branch
gh stack add <next-branch>             # from the top of the stack
gh stack submit                        # pushes, opens PRs, registers the stack on GitHub
gh stack sync                          # fetch, cascading rebase, push, refresh PR state
gh stack merge                         # merges bottom-up, all-or-nothing
```

`submit` is what creates the actual stack object, which is what gives you the stack map in the
merge box and the automatic cascading rebase. That rebase is the whole reason to use the
extension — keeping dependent branches in sync by hand is where stacks usually fall apart.

### Without it

`gh pr create --base <branch-below>` produces the same dependency chain and merges correctly
bottom-up, with GitHub retargeting the upper PRs as lower ones merge. What it does *not* create
is a registered stack, so there's no stack map and no cascading rebase. That's a fine fallback,
and `gh stack link <pr> <pr> ...` retrofits a real stack onto PRs opened this way.

### Constraints worth knowing before you start

- All branches must be in the same repository. Cross-fork stacks are unsupported.
- Merge bottom-up. Merging a mid-stack PR brings everything below it along.
- Branch protection and CI apply to **every** layer, judged against the bottom PR's base branch.
- Not supported in GitHub Desktop.

### What goes in the PR body

Reviewers of a stack have less context than usual, because each layer is deliberately partial.
Put in each body:

- its position, and what it sits on — `Stack — 2 of 2. Based on #1, not main.`
- the reasoning that isn't visible in the diff: why an approach was chosen over the obvious
  alternative, what a subtle change actually prevents
- how it was verified — the concrete numbers, not "tested"
- known limitations, stated plainly. A reviewer finding a flaw you knew about and didn't mention
  loses trust in everything else in the description.

Commit messages carry the same weight. `fix: sort dicom glob` is useless six months on; say what
broke, what the symptom looked like, and why the fix works.

## Don't paper over a surprise

Every real bug in this repo's Colab port first appeared as something small and ignorable: a
warning about a secret timing out (wrong notebook), a re-run producing byte-identical numbers
(non-editable install), an `IndexError` on one sample (padding side). Each looked like noise.

When something is slightly off and the easy read is "harmless", spend the two minutes. The cost
of checking is tiny next to the cost of a confident wrong conclusion built on top of it.
