---
name: colab-experiments
description: >-
  Run an experiment on a Colab GPU by pushing code to GitHub and pulling it there, driving the
  notebook over colab-mcp. Use this whenever work needs a GPU the local machine doesn't have,
  whenever a change is about to be tested on Colab, and when the user says something like "test this on colab". 
---

# Running experiments on Colab

The constraint that shapes everything: **Colab can only see code that is already on GitHub.**
There is no shortcut. A dirty local tree is invisible to the notebook, so "works on my machine"
and "works in the notebook" are different claims until you have pushed.

Dependency floors, install flags and notebook mechanics live in `CLAUDE.md` under "Running
experiments on Colab" and are already in context. This skill is the loop and the verification.

## Before the first cell

- **The runtime is a CPU one on first connect.**
  - After `open_colab_browser_connection`, the notebook always comes up on CPU.
  - Prompt the user to switch the machine type to GPU (T4) *before* running anything.
  - Do not run the setup cell first "to save time": changing the runtime restarts the VM and
    throws the install away.
  - Wait for the user to say the switch is done, then run setup.
- **`HF_TOKEN` is in the notebook's secrets.**
  - Gated models (e.g. `google/gemma-3-270m`) are fine to use; do not swap in an ungated mirror.
  - Expose it to subprocesses from the kernel, and never print it:
    `os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")` (`from google.colab import userdata`).
  - If the download still fails (401/403), stop and prompt the user to get access to that model.

## The loop

Setup cells run once per VM. Only the run cell re-runs.

| cell | changes | purpose |
|---|---|---|
| setup | rarely | `pip install uv`, clone, `uv pip install --system --no-deps -e`, then the light deps |
| assert | rarely | print versions, assert `torch.cuda.is_available()` and `"+cu" in torch.__version__` |
| data | rarely | download the corpus |
| checkout | per branch | `git fetch && git checkout <branch> && git pull`, then re-run the editable install |
| run | every iteration | `!MPLBACKEND=Agg python .../script.py <flags>` for a short run, a detached `Popen` for a long one (see below) |
| status | while a long run works | tail the run's log, and check the process is alive |

Then: **edit locally → commit → push → run the checkout cell → run the run cell.**

When an edit appears to have no effect, check that it was pushed *and* that the checkout cell
ran, before suspecting the code. A non-editable install produces exactly this symptom and looks
like a broken fix rather than a broken install.

Keep the actual run in a subprocess (`!python ...`), never in the kernel. Fresh imports each
time, and process-wide monkey-patching can't leak between runs.

## Long runs: detach, then check a status cell

- A run of more than a couple of minutes goes in the background, logged to a file.
  - The kernel stays free, so other cells run while it works: status checks, uploads, small fixes.
  - A foreground `!python ...` holds the kernel until the run ends.
    - `run_code_cell` moves it to the background after 120 s, but every later cell waits behind it.
- Launch it with `subprocess.Popen` in its own session, with stdin closed:

```python
import subprocess
alive = subprocess.run(["pgrep", "-f", "scripts/space_reports.py"], capture_output=True, text=True).stdout.split()
if alive:
    print("already running, pids", alive)
else:
    log = open("/content/run.log", "w")
    proc = subprocess.Popen(
        "MPLBACKEND=Agg PYTHONUNBUFFERED=1 python scripts/space_reports.py --config <yaml>; echo EXIT $?",
        shell=True, cwd="/content/in-tfm", stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print("started, pid", proc.pid)
```

- The `pgrep` guard makes re-running the cell safe, since it never starts a second run.
- `echo EXIT $?` puts the exit code in the log, for the checks under "Verifying a run".
- Read progress with a separate status cell, as often as needed:

```python
!tr '\r' '\n' < /content/run.log | grep -E "^\[|Error|Traceback|EXIT" | tail -20
!pgrep -af scripts/space_reports.py || echo "no run alive"
```

- `tr '\r' '\n'` turns tqdm's carriage returns into lines, so the stage lines can be grepped.
- Set `PYTHONUNBUFFERED=1` in the run's environment.
  - Python buffers stdout to a file, so stage lines otherwise show up many minutes late.
  - Until then, judge progress by the files the run has written.
- To wait for a run from another script, wait for its `EXIT` line in the log, not for its pid.
  - A run started with `Popen` is a child of the kernel, so once it exits it stays a zombie until the kernel reaps it.
  - `kill -0` succeeds on a zombie, so `while kill -0 $pid` never ends.
  - On 2026-10-24 that stalled a chain of runs until an unrelated `subprocess.run` in the notebook happened to reap it.

### Never `!cmd &`, even with `nohup`

- Colab's `!` waits until every process started from it releases the terminal.
  - A job backgrounded with `&` keeps the terminal's stdin unless it is redirected, so the cell never returns.
- Symptoms, seen on 2026-10-24:
  - The launch cell stays "executing" for as long as the run lasts.
  - Every cell sent after it shows `execution_count: null`, and `run_code_cell` returns empty outputs.
  - Reconnecting with `open_colab_browser_connection` does not help, since the kernel itself is busy.
- Interrupting the stuck cell kills the run too, despite `nohup`.
  - The run ended in a `KeyboardInterrupt` after 25 minutes of work, and had to start over.
- If it happens anyway, stop the cell yourself through Claude in Chrome, then relaunch with `Popen` as above.
  - The Colab tab is open in the user's Chrome, so its stop button, or Runtime > Interrupt execution, can be clicked.
  - Ask the user only if the browser tools fail.

## Verifying a run

This is the part that goes wrong quietly, so it deserves more care than it looks like it needs.

**A run that printed plausible numbers is not a run that finished.** Reports commonly write
their metadata *before* rendering artifacts, so a crash halfway through leaves a `meta.json`
full of correct, convincing values sitting next to zero images. Reading those numbers and
declaring success is a real failure mode — it happened in this repo and produced a confident
"verified, behaviour-preserving" claim about a run that had exited 1.

Check in this order:

1. the process exited 0
2. the **final** line the script prints — a total-time line, or whatever comes last — is present
3. the artifacts exist on disk; count the files rather than assuming
4. only then compare the numbers

**Keep a regression target.** When refactoring, the previous run's key numbers are worth more
than any unit test you could write, because they exercise the whole pipeline on real data.
Record them in the commit message so the next person can check against them.

**Re-run the baseline after any fix that changes inputs.** Comparing against a stale pre-fix
artifact produced a speedup claim here that was off by 2.5x. A baseline from before the fix is
not a baseline.

**A backend swap changes results, not just speed.** GPU and CPU implementations of the same
algorithm agree on easy cases and diverge on hard ones, so always say which backend produced a
number.

## Reading the results

Two audiences, two channels. Print the small structured stuff — metadata, timings, counts — to
cell stdout, which is all that's needed to judge an iteration. Zip the images and hand them to
the user separately; don't try to move binaries through cell output.

Before drawing conclusions from a cluster, an outlier or a pattern, ask whether the **corpus**
could explain it. A preprocessing convention produces clusters that look exactly as legitimate
as real ones, and nothing in the output distinguishes them — only knowing the data does.

## Runtime hygiene

Release the VM when done; credits burn on wall-clock, not compute:

```python
from google.colab import runtime
runtime.unassign()
```

**Do not verify this by running a cell.** Any cell execution makes Colab allocate a fresh VM, so
checking that it's off turns it back on. Verification and the goal are mutually exclusive here —
ask the user to glance at the runtime indicator instead.

## Don't paper over a surprise

Every real bug in this repo's Colab port first appeared as something small and ignorable:

- a warning that a secret timed out → attached to the wrong notebook entirely
- a re-run producing byte-identical numbers → the install wasn't editable, so the fix never loaded
- one `IndexError` on one sample → the tokenizer padded left, shifting every token index

Each looked like noise. When something is slightly off and the easy read is "harmless", spend
the two minutes. The cost of checking is tiny next to the cost of a confident wrong conclusion
built on top of it.
