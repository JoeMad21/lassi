# Confirmed counterexamples (task P16.7)

This directory holds the confirmed counterexamples that
`tools/export_counterexamples.py` copies out of a run tree: one JSON file per
confirmed divergence the adversary stage recorded (Trial.adversary,
Divergence). It is the only place the tool writes, and it lies outside every
bench suite (bible Adversarial Setups, item 1; Agent Rule 5; OQ-055, decision
12 (b)).

## Rules

- A counterexample is a synthetic fixture, never a held-out input. No suite
  manifest under `assets/bench` lists a file here, and the tool never edits a
  manifest or an item's held-out inputs.
- No registry reader and no train data source reaches a file here: the
  readers open a suite manifest only as `assets/bench/<name>.yaml`, and
  `load_suite`, handed a file from here, refuses it because it lacks a
  manifest's keys; `Suite.item` and `Suite.train_view` read only what a
  manifest names; `data.synthetic` names only a `.jsonl` file in
  `tests/fixtures/train/`; and `bench` reads only a suite's train split.
- Counterexamples of eval or unassigned items never train, tune a prompt, or
  feed a corpus harvest. Whether those of train items may train the
  adversary is decided with P8's planning (P16.12).
- Each file carries its run's provenance (commit, dirty flag, host and
  platform, device, driver as sdk, date, and pins; Agent Rule 1), so it
  names where its run ran and is never presented as another host's
  evidence. A file whose `exploratory` is true comes from a dirty tree, or
  from a run whose tree state was not recorded, and is never reported as a
  measurement. `exploratory: false` does not by itself make a value
  [MEASURED]: AGENTS.md Results still asks for a logged run with its
  provenance manifest, and for a workstation run a clean commit in the
  local distro's clone and the local provenance writer.

## Format

Each file is plain ASCII JSON with LF line ends, named
`<run id>--<suite>--<direction>--<item>--run<NN>--p<NN>.json`. Its keys, in
order: `format` (`lassi-counterexample-v1`), `exploratory`, `provenance`,
`trial_id`, `bench_item` (suite, item, split, direction), `adversary` (agent
and config), `proposal`, `candidate_sha256` (the sha256 of the reply the
adversary stage ran), `candidate_failure`, `stats` (binary_io's statistics of
the candidate against the target reference), and `inputs` (each input's
sha256 and its lassi_io bytes in base64). The tool's module docstring holds
the full rules and refusals.

## Exporting

    uv run tools/export_counterexamples.py <runs root>/runs/<run id>

A refusal prints its reason and exits with status 2, before anything is
written. An existing file with other bytes is never overwritten, and the
directory may not be reached through a link. After an export of a run whose
status is not `complete`, the tool names that status: check that the run
finished before committing its files.

No counterexample is committed by task P16.7; the phase gate (P16.G) exports
its runs' counterexamples here.
