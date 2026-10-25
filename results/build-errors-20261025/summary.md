# cpp -> tt build errors by bucket, 2026-10-25 (task P17.24)

Provenance: provenance.json (rx 20261025-050114-classify-3e3d; commit 3b41733, clean; alpha01, Ubuntu 22.04.5 LTS;
2026-10-25 05:01 on alpha01's clock, 08:01 EDT). The run read records only: it started no model and built or ran no
candidate or reference program. The command, exactly as provenance.json records it, from the slot's checkout:

```
prlimit --pid $$ --core=1:1 && d="$LASSI_RUNS_ROOT/classify-20261025" && mkdir -p "$d" && python3 tools/classify_build_errors.py "$LASSI_RUNS_ROOT" > "$d/report.txt" 2> "$d/stderr.txt"; rc=$?; echo "rc=$rc"; git log --oneline -1; git status --short | wc -l; wc -l "$d/report.txt"; head -60 "$d/report.txt"; cat "$d/stderr.txt"
```

The classifier's own exit status is the job log's first line, `rc=0` (`rx job tail 20261025-050114-classify-3e3d`;
provenance.json's rc is the whole line's, whose last command is cat). stderr.txt is empty. report.txt is the tool's
whole output: the bucketing rule, each run's counts with the provenance its own records carry (commit, clean or
dirty, device, pins, date), and the counts overall, per run, per project, per arm (the model), per item, and per
attempt index, then every kept trial. It read the runs root as it stood at 05:01, before the window's matrix wrote
any trial.

## Counts [MEASURED]

Counts of the records of clean runs under the runs root, as the tool's fixed rule buckets them (report.txt, "runs"
and "overall"):

- 92 runs, 336 trial records read, 134 of them cpp -> tt trials, in 62 runs.
- 13097 compile-stage errors bucketed across every attempt of those trials.
- In 121 of the 134 trials, some bucket recurs across the trial's attempts: the corrections met the same kind of
  error again.

report.txt also holds one run from a dirty tree, p413-fw-20261004-224906 (exploratory, never [MEASURED]): 2 records,
1 cpp -> tt trial, no compile error. With it the totals the report prints are 93 runs, 338 records, 135 trials in 63
runs; the errors (13097), the recurring trials (121), and every bucket count are the same with it or without it.

The ten largest buckets of the 13097 (report.txt, "overall"; <Q> is a quoted token, <N> a number):

| Count | Bucket |
| ---: | --- |
| 2113 | field designator <Q> does not refer to any field in type <Q> |
| 1429 | no matching function for call to <Q> |
| 1081 | no member named <Q> in namespace <Q>; did you mean simply <Q>? |
| 1059 | use of undeclared identifier <Q>; did you mean <Q>? |
| 1048 | variable type <Q> is an abstract class |
| 1039 | no member named <Q> in namespace <Q> |
| 934 | initialization of non-aggregate type <Q> with a designated initializer list |
| 837 | use of undeclared identifier <Q> |
| 557 | no member named <Q> in <Q> |
| 358 | the file <Q> cannot be written: the harness provides that file for the build. ... (abridged) |

Of the 13097, 556 are the harness's own compile-stage diagnostics, not the compiler's: 358 for a reply that rewrote
a harness file, 127 for a missing expected file, 51 for a fenced block never closed, 13 for a repeated block, and 7
for a repeated unfenced line (report.txt, "overall"). Attempt 0, the first generation, has 274 and 100 undeclared
identifiers (with and without a suggestion), 190 field designators, and 189 calls that match no function as its
largest buckets (report.txt, "per attempt index").

## Scope

- The cpp -> tt trials are those of the RNGD sweeps of 2026-10-22 to 2026-10-24 (sweep-20261022-2100 to
  sweep-20261024-0017; 113 trials in 55 runs, 12967 errors), the Tier A runs of 2026-10-20 (20261020-tiera1-* and 20261020-tiera30-*; 10
  trials, 130 errors), and 12 trials of the mock-reference arm in demo-mock-rehearsal-20261007-tt, p4-gate-tier-a,
  p413-dry-check, and the dirty p413-fw-20261004-224906, with no compile error among them. The other runs hold no
  cpp -> tt trial. Every trial id ends in run01: one sample each. Every TT run in them was on ttsim v1.3.4, a
  simulator, with tt-metal 5280a9cf; a build error stops a program before it runs.
- The counts measure the records, not the models: they weigh every attempt alike, so a trial that reached the cap
  adds more errors than one that passed early, and they are no score.

## Reading (not measured)

The masks hide every identifier, so the report does not show which buckets name a tt-metal API name and which name a
program's own; the buckets read as wrong names, overloads, and initializer forms, which is the kind of error the
lookup section (task P17.26, already built) is meant to answer. Nothing here measures whether it helps, and no prompt
or feature is tuned on these counts: every tt-pairs-v0 item is unassigned (Agent Rule 5).
