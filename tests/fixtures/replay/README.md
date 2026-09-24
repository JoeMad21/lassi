# Notebook replay fixtures (SYNTHETIC)

Every file here is a scripted, synthetic fixture for the notebook replay of
task P1.9 (tests/replay/). The recorded responses were written by hand for
these tests. They are not model output: upstream LASSI published no generated
code, and no model arm can serve before P3. Nothing here is upstream source or
upstream prompt text (OQ-018), and no value here is a measurement.

## What the replay checks

For each scenario in `scenarios.json`, in both directions (CUDA to OpenMP and
OpenMP to CUDA), the replay runs the pinned notebook's pipeline function and
our faithful stages on the same scripted inputs and compares their decisions:
every message sent, each extracted block, which attempts compiled and ran, the
final correction count, Sim-T, and Sim-L. It is a check of decision logic on
synthetic fixtures, not a measurement. Its fence-quirk hit count counts hits on
these synthetic replies only; it is not the Evaluation Protocol's fence-quirk
replay count and is never reported as a measurement.

## Files

- `scenarios.json`: the scenarios, labeled `"synthetic": true`.
  - `item`: the bench item whose manifest layout and run arguments both sides
    use (its real sources are never read; the programs below stand in for them).
  - `programs`: made-up source and reference programs per language. `plain`
    has LF line ends; `spaced` has runs of spaces, a tab, and CRLF line ends,
    so both sides must read it as text mode does.
  - `compile_kinds`: each scripted compile outcome, success or failure, and
    the raw stderr file it hands both sides per target language. The stderr
    files are the captured compiler outputs in `tests/toolchains/fixtures/`
    (see that README for their provenance); they are reused here as scripted
    outcomes, not re-measured.
  - `runs`: each scripted program run: an exit status (`exit`) or a death by
    signal (`signal`), and made-up stdout and stderr. The notebook's side sees
    a death by signal N as Popen reports it (-N); our executor reports it in
    the shell's form (128 + N).
  - `scenarios`: per scenario, its recording, its compile outcomes in order
    (the target reference's first), its run outcomes in order (the reference
    run's first), and `expect`, the decisions its script leads to, derived by
    hand from the script: attempts, the attempts that compiled and ran, the
    final correction count (null when the loop never started), the end
    (`complete`, `baseline-compile`, `baseline-run`, or `upstream-crash`), the
    fence-quirk hits, the attempt whose stdout stands as the output (null for
    none), and whether a stale-output warning is expected.
- `recordings/<scenario>.json`: the scripted replies in the replay backend's
  format (lassi/llm/replay.py), labeled `"synthetic": true`: the context
  summary reply, the source description reply, then one reply per attempt.
  They hold reply text only; the messages both sides send are compared at test
  time and never stored, since they hold upstream prompt text.

All files are plain ASCII. `scenarios.json` and the recordings were written by
a one-off script from literals; edit them by hand or regenerate them the same
way, and keep `expect` derived from the script, never from a run.
