# The Full Suite On alpha01

`tools/suite.py` starts LASSI's full RNGD suite on alpha01 as one big rx job that needs no session once it
runs, and then follows, stops, and pulls it. It is the owner's tool,
so the owner can launch the full suite unattended, with a help option on every command (the owner's direction of 2026-10-26). Every command documents itself: `uv run tools/suite.py --help` and `uv run tools/suite.py <command>
--help`.

The job is `plans/runs/suite/run-suite.sh`, a generalization of last night's `plans/runs/night/run-night.sh`. It
downloads the models, builds the TT assets the recipes need, runs `plans/runs/rngd-sweep/run.sh` once per model
and recipe directory, one model served at a time, and writes the tables. Every TT program runs on ttsim, a
simulator; no Wormhole card is used.

## Before A Window

- The window is yours to start: launching is an alpha01 step (Agent Rule 17), and the launcher asks before it
  starts anything unless you pass `--yes`.
- The gate's `rngd` device class must be enabled for the window (`uv run tools/rx.py doctor`, devices_enabled).
  The launcher checks this and stops if it is off.
- Launch from a clean commit that is pushed to origin, with every recipe directory you name committed in it
  (sweep8 is P17.34's). A dirty tree or an unpushed commit is refused unless you pass `--allow-dirty`, and such a
  run is exploratory, never [MEASURED]. A clean worktree of the commit keeps other work out of the job
  (plans/LESSONS.md, alpha01): `git worktree add --detach C:/dev/lassi-suite <commit>`, then run the commands
  below from C:/dev/lassi-suite.
- Run the commands from Git Bash or PowerShell at the repository root. The launcher sets `MSYS_NO_PATHCONV=1` and
  `PYTHONIOENCODING=utf-8` for every program it starts, so Git Bash never rewrites a path in the job's command.
  Quote a glob (`--only 'v1*'`) so your shell does not expand it.

## Steps

```
uv run tools/suite.py plan --until 01:45        # what launch would do; contacts nothing
uv run tools/suite.py launch --until 01:45      # checks, shows the plan, asks "Start this on alpha01? [y/N]"
uv run tools/suite.py status <job id>           # the job's state and the end of its log
uv run tools/suite.py wait <job id>             # wait up to an hour; run it again to keep waiting
uv run tools/suite.py stop                      # stop early (below)
uv run tools/suite.py pull <job id>             # after it ends: results/<name>/ and the counts
```

`launch` prints the job id and the exact commands to watch, stop, and pull it. After `pull`, write
`results/<name>/summary.md` from `provenance.json` and the pulled tables, as for `results/night-20261025/`.

`launch` checks, in order, and stops with what to do when a check fails:

1. `.rip-stop` is absent at the repository root.
2. The tree is clean and HEAD is on an origin branch (unless `--allow-dirty`).
3. `rx doctor` answers, the gate's owner STOP is not set, the `rngd` class is enabled, and the scratch disk has
   at least 65 GB free (the gate's 60 GB big-job floor and run-night.sh's 5 GB margin).
4. `rx job status` lists no running job. It does not say which jobs are big, and the gate runs one big job at a
   time, so any running job stops the launch.

Then it prints the plan, asks, and runs
`uv run tools/rx.py job start --big --slot <slot> --name <name> --timeout <s> -- 'bash plans/runs/suite/run-suite.sh ...'`.

## Options

`plan` and `launch` take the same options (`plan` ignores `--yes`). Exactly one of `--hours` and `--until` is
required.

| Option | Default | Meaning |
| --- | --- | --- |
| `--models LIST` | every row of the table | Comma-separated model slugs from the table, served one at a time in this order. An unknown name is refused. |
| `--table TSV` | `plans/runs/suite/models.tsv` | The model table: run.sh's six columns and the revision's size in bytes (`-` when unknown: never downloaded, served only from the cache). The default holds the night's two rows, llama33-70b and gpt-oss-120b. |
| `--sweep DIR` | `projects/lassi-demo/sweep8` | A recipe directory holding `<slug>/<combo>-<half>.yaml`; repeat it for more, run in the order given. Every recipe in it must load with the repository's recipe loader, and each selected recipe must name its row's model. |
| `--only GLOB` | every recipe | Run only the recipes whose name (`<combo>-<half>`, for example `v1ex-maxabs`) matches; repeat it for more. |
| `--hours H` | - | The deadline H hours from now. |
| `--until WHEN` | - | The deadline: a US Eastern clock time `HH:MM` (its next time), `"YYYY-MM-DD HH:MM"`, or epoch seconds. EDT until the first Sunday of November, EST after it. |
| `--skip-download` | off | Download nothing; run.sh skips a model whose snapshot is not in the host's cache. |
| `--name NAME` | `suite` | The rx job's name, part of its id. |
| `--slot SLOT` | `suite` | The rx worktree slot on alpha01. |
| `--yes` | off | Start without asking. |
| `--allow-dirty` | off | Launch from a dirty tree or an unpushed commit; the run is exploratory, never [MEASURED]. |
| `--silicon` | - | Refused with exit status 2 (see Silicon). |

The rx job's timeout is the deadline plus 900 seconds, the night's margin (deadline 01:45, timeout at 02:00), and
it is computed again when the job starts. A deadline that leaves under 30 minutes for each run.sh call (run.sh
serves no model with less), or a timeout past the gate's 172800 s limit, is refused.

The other commands:

| Command | Options | What it does |
| --- | --- | --- |
| `status [ID]` | `--lines N` (40; 0 for none) | `rx job status`, and for an id `rx job tail ID -n N`. |
| `wait ID` | `--timeout S` (3600), `--interval S` (300) | `rx job wait`; exit status 3 while the job still runs. |
| `stop` | `--clear` | Creates `$LASSI_RUNS_ROOT/rngd-sweep/STOP` with one `rx exec`; `--clear` removes it. |
| `pull ID` | `--name NAME` (the job's name, date, and time, as `suite-20261026-170035`) | Pulls the job record into `results/<name>/provenance.json` and the job's output directory into `results/<name>/night/`, then prints the job's counts. |

## Stopping Early

`stop` creates the sweep's STOP file, the route run.sh already honours. run.sh stops before its next model,
run-suite.sh makes no further run.sh call and ends a running download, and the job then writes its tables and
ends. Recipes already running go on to their time limit. rx has no job stop, and `rx job kill` ends only the job's
runner, not its command, which keeps the server and the cards (OQ-082); do not use it to stop the suite.

The STOP file stays until it is removed, and while it exists a new job refuses to start (exit status 2, said in
its log). Clear it with `uv run tools/suite.py stop --clear` before the next launch.

## What It Touches On alpha01

- The gate: one big job and its record under the gate's runs directory, and the worktree slot
  `lassi-wt/<slot>`, a checkout of the commit, where the generated TT assets are built.
- `$LASSI_RUNS_ROOT/suite/<job id>/`: the job's own output (below), what `pull` fetches.
- `$LASSI_RUNS_ROOT/rngd-sweep/<pass>/` and `$LASSI_RUNS_ROOT/runs/sweep-<pass>-*`: run.sh's output and the run
  trees, one pass per run.sh call.
- `$LASSI_SCRATCH/.cache/huggingface`: model downloads. Each runs only while the scratch disk keeps 65 GB free
  after the bytes it still needs; the scratch root's 150G cap stays lifted while OQ-074's lift holds, and the job
  logs the root's size at its start and end.
- The RNGD cards and port 8123: run.sh claims only cards that read 0.00 GiB in `furiosa-smi status`, polling
  `furiosa-smi ps` and status every 60 s, serves one model at a time, checks that `/v1/models` lists it, and waits
  for the server's full exit and the port to close before the next (Agent Rule 8).
- ttsim and the tt-metal install under `$LASSI_TOOLCHAINS`. No Wormhole card is opened.
- The STOP file `$LASSI_RUNS_ROOT/rngd-sweep/STOP`, only through `stop`.

The job's output directory holds `recipes.tsv` (every recipe of the run with its recipe_hash, model id, and the
assets it needs) and `recipes/` (each selected recipe's resolved text, as a run saves it), `models-<slug>.tsv`,
`calls.tsv` (each run.sh call, its pass, and what it left out), `sweep-<n>-<slug>.log`, `passes/<pass>/`
(run.sh's summary.tsv and each served model's model list), the asset build logs, `fetch-<slug>.log` and `.done`,
`tables/` (tabulate.py over the whole runs root: trials.tsv, passes.tsv, not_verifiable.tsv, hardware_check.tsv),
`features-<pass>.tsv`, and `build-errors.txt`. `pull` copies every plain-ASCII file of it and names the files it
leaves out, the download logs' progress bars, as last night's record left out its two download logs.

## How Long It Takes

The job ends by the deadline, and the rx timeout ends it at most 900 s later. Within that, each run.sh call gets an
equal share of the time left, so with the default two models the first runs until about the midpoint and the
second until the deadline, as last night. A model whose recipes finish early leaves its time to the next.

Last night's run, the nearest guide (plans/runs/2026-10-25-night.md; times EDT; sweep7, the same two models): rx
20261025-170035-night-395c started at 20:00:35; the Llama download (73.8 GB) ended at 20:11:57, then the three
asset builds; Llama-3.3-70B was served at 20:12, READY at 20:14, and its 8 recipes ended at 22:09; gpt-oss-120b,
downloaded in the background meanwhile (198.6 GB), was READY at 22:13 and its 10 recipes ended by 00:12; the
tables were written and the job ended at 00:12:13 with 66 GB free. sweep8 (10 recipes per model) has not run yet, so its time is not
known. With both models already in the cache the downloads end quickly; `--skip-download` skips them.

## Results

Every count `pull` prints is copied from the trial records through tabulate.py. tt-pairs-v0 is a development set
(OQ-083 (b)), so these are development results. A run from a clean pushed commit is [MEASURED] on ttsim v1.3.4 on
alpha01 only as cited from its provenance.json in summary.md; a dirty run is exploratory and never [MEASURED].
ttsim is a simulator: its results are never silicon results and its timing is never performance (Agent Rule 2).

## Silicon

`--silicon` is accepted and always refused: this launcher never runs model-generated programs on the Wormhole
cards. Under alpha01's identity IOMMU domains a program that holds a card can make it read or write any host
memory (docs/tt-silicon-iommu.md; OQ-078), and silicon in the correction loop also waits on an Agent Rule 9
change worded by the owner (OQ-099).

## Files

- `tools/suite.py`: the launcher.
- `plans/runs/suite/run-suite.sh`: the job (`bash plans/runs/suite/run-suite.sh --help`).
- `plans/runs/suite/recipes.py`: its recipe helper (the manifest, the assets needed, and run.sh's SWEEP_SKIP).
- `plans/runs/suite/models.tsv`: the default model table.
- `tests/tools/test_suite.py`: tests with fakes for rx and git.
