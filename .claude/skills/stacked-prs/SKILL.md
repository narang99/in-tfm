---
name: stacked-prs
description: >-
  Open and manage a chain of dependent pull requests with the `gh stack` extension, where the
  bottom PR targets main and each one above targets the branch below it. Use this whenever a
  change is large enough to split into reviewable pieces, whenever the next piece of work
  depends on a branch that hasn't merged yet, and whenever the user says "stack these", "raise
  stacked PRs", "split this up", "open a PR on top of that one", or asks how to review or merge
  a chain of dependent branches. Also use it when a stack needs rebasing after its base moved.
---

# Stacked pull requests

A stack is a chain of PRs where the bottom targets the trunk (usually `main`) and each one above
targets the branch below it:

```
   ┌── feat/frontend     → PR #3 (base: feat/api-endpoints)  ← top
  ┌── feat/api-endpoints → PR #2 (base: feat/auth-layer)
 ┌── feat/auth-layer     → PR #1 (base: main)                ← bottom
main
```

Each PR shows only its own layer's diff. That is the whole point: reviewers get a small, focused
change instead of one large one, and you can start the next piece before the previous merges
rather than sitting idle waiting for review.

## Where to cut a branch

The rule that decides it: **if code in one layer depends on code in another, the dependency must
be in the same branch or a lower one.** Foundational things — shared types, an interface, a
schema migration — go lower; whatever consumes them goes above.

Cut a new branch when the *concern* changes, not when a line count is hit. In practice:

- a refactor that enables a feature is the bottom layer, the feature is the top
- backend before frontend, core logic before the tests that exercise it
- when the current branch is already big enough that you'd apologise for its size in review

## The commands

```bash
gh extension install github/gh-stack   # once; needs gh >= 2.0

gh stack init <branch>       # start a stack; trunk defaults to the repo's default branch
gh stack add <branch>        # add a layer on top (run from the topmost branch)
gh stack submit              # push branches, open PRs, register the stack on GitHub
gh stack sync                # fetch, cascading rebase, push, refresh PR state
gh stack view --short        # see the chain and its PR links
gh stack merge               # merge bottom-up, all-or-nothing
```

`gh stack add -Am "message"` stages everything, commits, and auto-names the branch — useful when
you're moving fast and don't want to invent names.

Navigation, when you've lost your place: `gh stack up` / `down` / `top` / `bottom` / `switch`.

**`submit` is what creates the stack object**, and that object is the reason to use the extension
at all. It gives you the stack map in the merge box and, more importantly, the automatic
cascading rebase. Keeping dependent branches in sync by hand is where stacks usually fall apart —
rebase the bottom and every branch above it needs replaying, in order, resolving the same
conflicts repeatedly. `gh stack init` turns on `git rerere` so those resolutions are at least
remembered.

## Without the extension

`gh pr create --base <branch-below>` builds the same dependency chain, and it merges correctly
bottom-up — GitHub retargets the upper PRs to the trunk as lower ones merge. What you don't get
is a *registered* stack: no stack map, no cascading rebase.

That's a reasonable fallback when you can't install extensions, and it's retrofittable:

```bash
gh stack link <pr-or-branch> <pr-or-branch> ...   # bottom to top; creates the stack on GitHub
```

`gh stack link` is also the right tool if branches are managed by something other than git
locally — Jujutsu, Sapling, git-town — since it touches no local tracking state.

## Merging

Merge **bottom-up**. Merging a mid-stack PR brings everything below it along; PRs above stay open
and retarget automatically.

- `gh stack merge` — the whole current stack
- `gh stack merge 42` — everything up to and including PR #42
- merge the top PR on the web to land the entire stack at once

Branch protection and CI apply to **every** layer, evaluated against the bottom PR's base branch
— so a mid-stack PR that never directly targets `main` still has to satisfy `main`'s rules. You
cannot bypass merge requirements on a stack.

If the base branch uses a merge queue, the stack is queued rather than merged directly, the queue
picks the merge method, and the PRs can land in separate groups rather than together.

## What goes in each PR body

Reviewers of a stack have *less* context than usual, because each layer is deliberately partial.
Compensate:

- **Position first** — `Stack — 2 of 2. Based on #1, not main.` Say explicitly that the diff looks
  small because it's compared against the layer below, and which PR to read first.
- **The reasoning the diff can't show** — why this approach over the obvious alternative, what a
  subtle change actually prevents. A one-line fix whose value is invisible in the diff deserves
  the most explanation, not the least.
- **How it was verified** — the concrete numbers. "Tested" tells a reviewer nothing; a before/after
  table tells them whether to trust the change.
- **Known limitations, plainly.** A reviewer who finds a flaw you knew about and didn't mention
  stops trusting the rest of the description.

Commit messages carry the same weight, and outlive the PR. `fix: sort glob` is useless in six
months; say what broke, what the symptom looked like, and why the fix works — especially when the
symptom was misleading, because that's what saves the next person the debugging session.

## Constraints worth knowing before you start

- All branches must be in the **same repository**. Cross-fork stacks are not supported.
- Not supported in GitHub Desktop.
- Restructuring — reordering, dropping, folding layers — goes through `gh stack modify`, or
  `gh stack unstack` followed by a fresh `init`. Don't hand-edit base branches of a registered
  stack; you'll desync local tracking from GitHub.
- `gh stack sync` in a non-interactive shell aborts on a diverged stack rather than guessing.

## Verify the chain before you call it done

Cheap, and catches a base branch pointed at the wrong layer:

```bash
gh pr list --json number,headRefName,baseRefName \
  -q '.[] | "#\(.number) \(.headRefName) -> \(.baseRefName)"'
```

Every PR should target the branch below it, and exactly one should target the trunk.
