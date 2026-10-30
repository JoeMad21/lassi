"""The stage runner behind `lassi run <recipe>`: load a recipe, run its stages over every trial, write the run tree.

run_recipe (bible Project Recipes; Component Interfaces; Result Record):

1. loads the recipe with the registry (every component registers when this
   module is imported, since it imports lassi.llm, lassi.toolchains,
   lassi.executors, lassi.profilers, lassi.agents (the fuzzer and llm adversaries),
   lassi.core.stages, lassi.core.oracle_stage, which imports lassi.oracles,
   lassi.core.adversary_stage (task P16.5), and lassi.scoring, which
   registers every ScoreProfile);
2. refuses what it cannot run, before any directory is created or any model
   is asked: first, right after the recipe loads, a bound executor (the
   single one, or one bound for any language) that declares `silicon`
   (lassi.core.capabilities SILICON, the tt_silicon executor's; task
   P17.16) when a listed stage declares `computes_reward`, even with the
   human start; when a listed stage runs model-generated code
   (`runs_model_code`, as run_loop declares) or an adversary's proposals
   (`runs_proposals`), even with the human start, naming OQ-078 (the
   sandbox does not keep a program from host memory under the cards'
   identity IOMMU domain, so until the owner answers, silicon runs only
   tracked references a human starts; plans/spikes/tt-card-dma.md); and
   otherwise unless RunOptions.silicon is True, which only
   `lassi run <recipe> --silicon` sets: silicon runs are evaluation runs a
   human starts, and RL rewards never execute on silicon (Agent Rule 9;
   OQ-075 (6a)). A recipe cannot set it, no environment variable stands in
   for it, and every refusal comes before any executor is built, so nothing
   reads the cards' files in a run a human did not start (_check_silicon).
   Then: arms without a model (the model registry that maps arms to
   backends comes later) and arms beside a model; a recipe with no
   llm.sampling.max_tokens or no prompt set, since it never picks a value;
   a fix toggle turned off (faithful: true turns them all off) when no
   listed stage names that fix in `reproduces`, a report toggle turned off,
   or a section this runner does not carry out (judges and the like),
   since it never ignores a choice; a stage list whose order cannot
   work (one generating stage at most, every compiling stage after it, and
   every stage that builds the reference programs, `builds_references`,
   before any stage that asks the model);
   a stage that builds the source reference under a fix that is on
   (`source_build_fix`, as baseline declares for baseline_both) when no
   toolchain is bound for a direction's source language;
   a template prompt set that lacks a template a stage renders or uses a
   placeholder the stage does not fill; context packs, or a listed stage
   that declares `needs_context`, with a template set whose
   summarize_context.txt or knowledge.txt (lassi.core.stages
   CONTEXT_PROMPTS) is missing or does not render with its fields (task
   P17.21); a fragment prompt set or context packs that
   do not load (lassi.prompts.load_recipe_assets, which checks every file
   against its manifest; a template set's packs are checked the same way,
   _template_set_pack), two packs that serve one language, a fragment a
   stage needs for a direction, a context pack a stage needs for a target
   language, and a Trial.context field a stage joins into a fragment
   set's prompt that no earlier stage fills; the retrieval section (task
   P17.22) with a fragment set, with a template set whose retrieved.txt or
   passage.txt (RETRIEVAL_PROMPTS) is missing or does not render, or with
   a corpus whose manifest is missing or whose index file is missing or
   fails the manifest's sha256 (_retrieval_index, which loads the index
   once per run); the lookup section (task P17.26) with a fragment set,
   then without a context pack serving a direction's target language,
   then with a lookup.core header that pack has no header part for, then
   with a template set whose looked_up.txt or declaration.txt
   (LOOKUP_TEMPLATES), or with lookup.nearest set its undeclared.txt
   (NEAREST_TEMPLATES), or with lookup.prompts listing generate its
   core_api.txt (CORE_TEMPLATES), is missing or does not render
   (_check_lookup);
   the examples section (OQ-083 (b)) with a fragment set, or with a
   template set whose examples.txt or example.txt (EXAMPLE_TEMPLATES) is
   missing or does not render (_check_examples), and a pack whose
   MANIFEST.yaml is missing or whose files are missing or fail it
   (_examples_pack, which loads the pack once per run);
   the skeleton section (OQ-096 (a)) with a fragment set
   (_check_skeleton), and a skeleton that no longer loads or whose sha256
   is not the resolved recipe's skeleton.sha256 (_run_skeleton, which loads
   it once per run);
   loop.restart_after (task P17.25) with a fragment set
   (_check_restart); a project (the recipe's `project`, the first
   trial id segment) that is not one plain segment or that names an entry
   of the run tree; trial ids that repeat; missing bench sources; an
   item with more than one source file under a fragment set, or more than
   one target file with fixes.fence_tag off, since each reads one file; and
   an oracle section that no listed stage uses (none names Oracle in
   `requires`), whose Oracle declares none of the capabilities a listed
   stage accepts (`oracle_capabilities`), or that its Oracle refuses
   (factory(**config) raises ValueError, as for an unset oracle.passfail or
   oracle.threshold); an Oracle that reads the references' agreement
   (`agreement_setting`, binary_io's threshold from_baseline) while
   fixes.baseline_both is off, no listed stage builds the references, an
   item of the run declares no tolerance in its suite manifest, or a
   direction's source and target are one language (task P16.8); a backend that declares
   `unload_before_run` but has no unload(), or `needs_reference` but has no
   with_reference() (or, with fixes.fence_tag off, no with_untagged_fence());
   llm.sampling.reasoning_effort with a backend that does not declare
   `reasoning_effort` (OQ-080; _reasoning_effort);
   a `score` profile or, when the recipe names metrics, any registered
   ScoreProfile that cannot be built, and a `metrics` name that no
   registered provider offers, that two offer, or that is named twice
   (lassi.scoring.run_scoring.plan_scoring);
   an adversary section and an adversary stage (one that declares
   runs_proposals; task P16.5) without each other, that stage listed twice
   or before the oracle stage, an adversary.budget.inputs that is not an
   int of at least 1, an item of the run with no held-out inputs, a target
   language's executor that runs no program or runs programs without
   `sandboxed`, and an adversary Agent that refuses its config or has no
   propose() (_check_adversary);
   and, when any bound executor runs programs, a sandbox.wall_s that is
   neither a number of seconds above 0 nor `baseline_x10`, or a
   sandbox.mem_gb that is not above 0 (the limits of every run,
   lassi.core.stages attempt_limits and reference_limits), and each bound
   executor that runs programs without `sandboxed` when a listed stage
   declares `runs_model_code` (Agent Rule 6); with executors bound per
   language (lassi.core.recipe, task P4.5), a direction's target language
   with no executor, and its source language with none when a listed stage
   builds the source reference under its `source_build_fix` (baseline under
   baseline_both), each message naming executor.<language>; a device()
   that returns neither None nor one non-empty line of printable ASCII with
   no leading or trailing blank; and an executor's `pins` (task P4.11) that
   are not pin name -> pairs with a VERSION, that name a pin with no Trial
   toolchain_pins field, or that give a pin other pairs than a toolchain or
   an earlier executor gives the same pin name.
   Right after the stage plan is checked, before any component is built
   (the Oracle and the ScoreProfiles included), it probes the device of
   every binding whose component declares `takes_device` (task P17.2;
   lassi.core.devices), in binding order: the component class's optional
   framework() (build metadata only) and then the kind's probe from
   RunOptions.probes, and it refuses a device the host lacks (no probe for
   the kind, a framework build that cannot reach it, a probe's refusal, or
   an index past the host's count), naming the key; no run falls back to
   another device. A framework() result that is neither a FrameworkBuild
   nor None is refused too, and so is any error framework() raises (a
   framework extra that is not installed among them), naming the key.
   Right after the probes it refuses a run in which model.device and any
   executor's device section (single or per language) name the same kind
   and share an index, unless the backend declares `unload_before_run`
   (one GPU, one role, task P17.5; _check_one_role).
   Then, for a recipe that binds a profiler (task P17.7; _check_profiler),
   it refuses one with nothing to profile (no listed stage declares
   `runs_model_code`, or no target language's executor runs programs), one
   whose profiled executor declares `simulator` (Agent Rule 2), one that
   declares `supports_power` without `takes_device`, and a power profiler
   beside a profiled executor (the single one, or each target language's
   that runs programs) whose device section does not name the profiler's
   kind and indices, naming profiler.device and that executor's key.
   A stage it does not implement already fails at load, unregistered;
3. builds the components: the backend as factory(model.id, **config), whose
   config holds the model section's keys other than backend, id, and
   recording_sha256 (its device section, server settings such as
   model.base_url, an in-process model's model.revision and model.seed,
   and the replay backend's model.recording, which the loader has checked
   under tests/fixtures/ and hashed into the resolved recipe, passed as the
   checked file; task P16.13), a setting
   the backend refuses being a RunError, as are a recording the backend
   cannot read and one whose bytes, by the backend's recording_sha256, are
   not those model.recording_sha256 records; then, for a backend that declares
   `model_check` (lassi.core.capabilities MODEL_CHECK), it reads the serving
   record with serving(), so a model the server does not list, a server
   that does not answer, or a record that would hold the API key is a
   RunError before any directory exists and before any unload or chat
   request (task P17.3); each toolchain
   with its pinned compiler and a clean environment (below), each executor
   as factory(**config) (one, or one per language; a DeviceUnavailable it
   raises, as the gpu executor does for a node or an index this host lacks,
   is a RunError naming the device key, task P17.5), asking each for its
   device() and reading the pins it declares, then the profiler, once per
   run, as factory(**config) (a setting it refuses, or telemetry this host
   lacks, is a RunError; task P17.7), and the ScoreProfiles
   `score` and `metrics` need
   (lassi.scoring.profiles.build_profile, with the bench root for a profile
   that declares reads_bench_sources);
4. loads the suite manifest assets/bench/<bench.suite>.yaml and finds the
   fetched sources (tools/fetch_bench.py puts them under $LASSI_SCRATCH),
   none for a tracked suite, whose files are all in the repository (task
   P16.8; a bench root given for one is a RunError);
5. runs every trial, direction by direction in recipe order, item by item in
   sorted order (the items bench.items selects, else every item of the
   recipe's split; an unknown item, one outside the split, or a repeat is a
   RunError before anything is written), then run 1 to trials.n: a backend
   that declares `needs_reference` gets the item's reference target, with
   with_untagged_fence() when fixes.fence_tag is off (the mock's faithful
   reply form), a backend that declares `unload_before_run` is asked to
   unload (upstream's setup unload, lassi.core.capabilities), then the
   stages run in recipe order on a fresh RunContext (which carries the
   prompt set's fragments, the context packs by language, the target
   language's executor, which runs every attempt, with executors per
   language every language's executor, so the baseline runs each reference
   on its own language's, the run's profiler, which run_loop wraps
   around every attempt run, the run's registry, from which the
   adversary stage builds the trial's adversary, and the run's one
   documentation index when the recipe sets retrieval), then the
   trial's final block, whose alignment is that of the attempt whose output
   stands (_final). A stage that sets final.end_reason ends the trial:
   no later stage runs, and the final block keeps the end reason. With
   `score`, the bound profile then sets final.score, and each Attempt.score
   when it declares scores_attempts, before the trial is written;
6. with `metrics`, scores every trial again with each profile that gives a
   named metric (these Scores stay out of the trials) and builds the run
   metric tables (lassi.scoring.run_scoring.compute_metrics), and, when an
   adversary metric is named, the adversary metric tables from the trials'
   adversary records alone (lassi.analysis.adversary adversary_tables;
   task P16.6);
7. writes the run tree and prints one line per trial and the run directory.

With RunOptions.observer set, the run also sends progress events
(lassi.core.progress, task P4.8) to that observer, wrapped in a
GuardedObserver shared by every trial: trial-start (the run id, the
trial's place in the run and the run's trial count, and the item's source
files), stage-start before each stage, and trial-end after the trial is
scored and written, just before its line is printed; the stages send the
rest through RunContext.observer. An observer that raises is dropped for
the rest of the run, and no record, file, or exit status depends on one.

The run tree is <runs root>/runs/<run_id>, where the runs root is the
runs_root option, else $LASSI_RUNS_ROOT, else the recipe's runs_root. The
runs root and the run directory must be absolute, resolve outside the
repository, and, when $LASSI_SCRATCH is set (the gate sets it on the build
host), resolve inside it (Agent Rule 7). The tree holds:

- recipe.resolved.yaml: the resolved recipe, which reruns the run alone
  (Design Principle 5);
- toolchains.json: per toolchain, its languages, pinned executable, compile
  environment, and pin files;
- provenance.json: the commit, the dirty flag, host, Python, the executor
  and its device, the start and end times (UTC), recipe hash, pin versions
  (every bound toolchain's and every bound executor's), and a status. The
  single executor form records `executor` (its name) and
  `device` (its device(), null for an executor that names none); executors
  per language record `executor` and `devices`, each a mapping from language
  to that language's executor name and device. `device_records` holds
  the record of each device section in binding order ([] without one), and
  `driver` the distinct drivers they name, joined by "; " (null when none
  does). It is
  written before the first trial with status "running", and again at the
  end with "complete", or with "failed" when a trial or a write raised, so
  every trial.json in the tree has a commit and a date beside it. Each
  trial's provenance copies the commit, dirty flag, device (the device of
  its target language with executors per language), driver (as sdk),
  started_utc (as date), and device_records of the manifest written first, and the final
  manifest is that same manifest with only its status and finish time
  changed, so a trial and its manifest never disagree. For a backend that
  declares `model_check`, `serving` holds its serving record
  (lassi.llm._serving: base_url, the model entry less its volatile fields,
  version, and version_from), which the trials do not copy;
- run.md: the page a person reads (Readability Standards, Run row). Its
  summary shows the manifest's provenance, with an unknown value (null) as
  "-" as trial.md shows it, since provenance is not a measurement; with
  executors per language, one Device row per language; with `serving`, a
  Server row after the Driver row (lassi.llm.serving_line). When a direction's
  target language, or its source language when the baseline runs the
  source reference too, runs on an executor that declares `simulator`
  (lassi.core.capabilities SIMULATOR), its Trials section opens with a
  note naming those languages: their run wall times are simulator wall
  time, not performance, and the wall_s column is each trial's pipeline
  wall time (task P4.6). A Guard coverage section follows the Trials
  section in every run (task P17.14; lassi.analysis.guard_coverage): the
  host-compute guard's checked and not-checked attempts, the not-checked
  rate, and each host_compute_not_checked code's count, for the run and
  per target language. With
  `metrics` it ends in a Metrics section
  (lassi.scoring.run_scoring.metrics_section), with the adversary tables
  when an adversary metric is named;
- one directory per trial (one level per trial_id segment) with trial.json,
  trial.md (each run's wall_s row labeled as simulator wall time when the
  trial's target-language executor declares `simulator`), and
  attempt<NN>/build, each attempt's fresh build directory
  (the run directory is the stages' build root, so a rerun of the recipe
  never meets an earlier run's builds);
- texts/, the text store of prompts and replies; parquet/, the mirror, which
  with `metrics` also holds the metric tables and the named components'
  values (lassi.scoring.run_scoring.write_metrics), and, when an adversary
  metric is named and the adversary stage took a trial, the adversary
  metric tables as parquet/adversary_metrics
  (lassi.analysis.adversary write_adversary_parquet; task P16.6).

Pinned toolchains (Agent Rule 10): a toolchain class that declares PIN and
PIN_BIN is built as factory(executable=<toolchains root>/<PREFIX_NAME>/
<PIN_BIN>, runner=SandboxedCompileRunner(...)); one that declares PIN and no
PIN_BIN uses a host compiler the project does not install, and its
executable is the pin's EXECUTABLE, an absolute path taken as given
(_host_executable; toolchains/gcc.pin). Such a class that also defines
check_tree builds against the pin's installed tree, <resolved toolchains
root>/<PREFIX_NAME>, which must be a directory that resolves inside the
root and which check_tree(tree, pin) must accept (its ValueError becomes a
RunError); it is built as factory(executable=..., runner=..., tree=<that
tree>) (_checked_tree; toolchains/tt-metal.pin). The toolchains root is the
toolchains_root option, else $LASSI_TOOLCHAINS, resolved once (links and
`..` segments followed), and the pin is read from toolchains/<PIN>.pin. The
executable, the tree, each linked prefix, and the sandbox's read-only
toolchains root all come from that resolved root, which is what a compile
sees, and an executable, tree, or prefix that resolves outside it is
refused. The compile
environment is the parent's PATH, LANG=C and LC_ALL=C (ASCII diagnostics),
and each variable the pin names (NVHPC_CUDA_HOME for toolchains/nvhpc.pin);
nothing else, so a variable such as NVCC_PREPEND_FLAGS never reaches a
compile, and HOME stays out on purpose (P0.20). Every compile of generated
sources runs in the sandbox (lassi.executors.sandbox.SandboxedCompileRunner,
P0.20): its build dir is the one writable directory, the toolchains root is
read-only, $HOME, $LASSI_SCRATCH, and $LASSI_RUNS_ROOT (those set) and the
run's own runs root are hidden, it runs under prlimit --core=1, and it gets
a private TMPDIR under its build dir. TMPDIR itself is still required, and
when $LASSI_SCRATCH is set it must resolve inside it (Agent Rule 7). The
scratch root is hidden, and TMPDIR checked against it, only when
$LASSI_SCRATCH is set, as the gate sets it on the build host; without it a
compile still hides $HOME and the runs root. Before a toolchain is used, the
compile layout is checked as the sandbox checks it, and `<executable>
--version` runs once through the toolchain's own compile runner (so in the
sandbox, with the builds' environment and view, in a fresh directory under
TMPDIR) and must print the pin's EXPECT_VERSION; that also proves the
compiler reachable in the compile's view. A class may declare VERSION_ARGS,
the check's arguments after the executable (hipcc-gfx942 names its target
before --version); without it they are --version alone. Every refusal
comes before the run directory exists, and so does a
SandboxUnavailableError from the --version check. A class without PIN (a
test fake) is built as factory().
build_toolchain is that construction for one registry name, public so that
a tool compiles exactly as a run does; the runner builds every bound
toolchain with it. An executor may declare pins too (task P4.11; Agent
Rule 10): `pins`, pin name -> the pin file's pairs, as BuiltToolchain.pins
holds a toolchain's (the ttsim executor declares ttsim and tt-metal). A
trial's toolchain_pins records the pins of the toolchain that builds its
target language and of the executor that runs it; provenance.json records
every bound toolchain's and executor's pins. A pin name that a toolchain
and an executor, or two executors, declare must carry the same pairs, or
the run is refused before any directory exists; the runner reads the
attribute, never an executor's name (Agent Rule 3). git, for the commit
and dirty flag, runs through
lassi.toolchains.EnvRunner, the audited command runner, and every compiler
command through the sandbox's runner, so this module starts no process
itself.
"""

from __future__ import annotations

import dataclasses
import os
import platform
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lassi import (
    agents,  # noqa: F401  (registers the Agents fuzzer and llm, the adversaries)
    profilers,  # noqa: F401  (registers the Profilers timing, nvml, and rocm_smi)
)
from lassi.analysis.guard_coverage import coverage_markdown, guard_coverage
from lassi.bench import Direction, Suite, load_suite, sources_dir
from lassi.bench import registry as bench_registry
from lassi.core import oracle_stage  # noqa: F401  (registers Stage "oracle" and, through lassi.oracles, the oracles)
from lassi.core.adversary_stage import ADVERSARY_KIND, RUNS_PROPOSALS, adversary_binding, budget_inputs
from lassi.core.capabilities import (
    ALIGNS_OUTPUT_FILES,
    COMPUTES_REWARD,
    DEVICE_KEY,
    LOWERS_SOURCE,
    MODEL_CHECK,
    REASONING_EFFORT,
    SILICON,
    SIMULATOR,
    SUPPORTS_POWER,
    TAKES_DEVICE,
    UNLOAD_BEFORE_RUN,
    declares,
    unload_before_run,
)
from lassi.core.devices import (
    DEFAULT_PROBES,
    DeviceProbe,
    DeviceUnavailable,
    FrameworkBuild,
    device_driver,
    parse_device,
    probe_device,
)
from lassi.core.fragments import fragment_key, pack_language
from lassi.core.interfaces import Executor, Profiler, Sampling, Toolchain
from lassi.core.parquet import write_run_parquet
from lassi.core.progress import STAGE_START, TRIAL_END, TRIAL_START, GuardedObserver, Observer, ProgressEvent, notify
from lassi.core.recipe import (
    DIALECT_LOOKUP,
    EXAMPLES,
    LOOKUP,
    LOOKUP_CORE,
    LOOKUP_GENERATE,
    LOOKUP_NEAREST,
    REASONING_EFFORT_KEY,
    RECORDING_KEY,
    RECORDING_SHA256_KEY,
    RESTART_AFTER,
    RESTART_ON_REPEAT,
    RETRIEVAL,
    SKELETON,
    SKELETON_SHA256_KEY,
    SOURCE_MLIR,
    UNCAPPED,
    Recipe,
    RecipeError,
    executor_languages,
    load_recipe,
    resolved_data,
    resolved_yaml,
)
from lassi.core.record import (
    TOOLCHAIN_PIN_NAMES,
    DeviceRecord,
    Final,
    Provenance,
    ToolchainPins,
    Trial,
    arm_segment,
    from_dict,
    json_text,
    make_trial_id,
    standing_attempt,
)
from lassi.core.registry import DEFAULT_REGISTRY, Binding, Entry, Registry, RegistryError, device_path
from lassi.core.stages import (
    BASELINE_X10,
    CONTEXT_PROMPTS,
    COORD_GAP_AS_ERROR,
    CORE_TEMPLATES,
    CORRECT_ON_MISMATCH,
    CORRECT_ON_SIM_GAP,
    DIALECT_TEMPLATES,
    EXAMPLE_TEMPLATES,
    GENERATE_FIELDS,
    LOOKUP_TEMPLATES,
    NEAREST_TEMPLATES,
    PURPOSE,
    RETRIEVAL_PROMPTS,
    RUNS_CODE,
    SANDBOXED,
    SOURCE_MLIR_TEMPLATES,
    RunContext,
    pack_parts,
)
from lassi.core.store import TextStore, write_trial
from lassi.core.trial_md import fenced, fmt, fmt_provenance
from lassi.executors import workdir
from lassi.executors.sandbox import SandboxedCompileRunner
from lassi.llm import ServingError, model_info, serving_line
from lassi.prompts import assets as prompt_assets
from lassi.prompts import examples as examples_pack
from lassi.prompts import load_recipe_assets, rag_corpus, render
from lassi.prompts import skeletons as skeleton_files
from lassi.prompts.dialect_defs import DialectDefs, load_defs
from lassi.prompts.retrieval import GENERATOR as INDEX_GENERATOR
from lassi.prompts.retrieval import Index, load_index
from lassi.scoring.run_scoring import (
    RunMetrics,
    ScoringPlan,
    compute_metrics,
    metrics_section,
    plan_scoring,
    score_trial,
    write_metrics,
)
from lassi.scoring.score_run import ScoreError
from lassi.toolchains import EnvRunner
from lassi.toolchains.pins import linked_prefixes, prefix_pin_name, read_pin, recorded_version

REPO = Path(__file__).resolve().parents[2]
# Where suite manifests live: bench.suite names <BENCH_DIR>/<suite>.yaml.
BENCH_DIR = REPO / "assets" / "bench"

RUNS_DIR = "runs"
RESOLVED_RECIPE = "recipe.resolved.yaml"
TOOLCHAINS_JSON = "toolchains.json"
PROVENANCE_JSON = "provenance.json"
RUN_MD = "run.md"
PARQUET_DIR = "parquet"

# The status provenance.json records: while trials run, after a trial or a write raised, and at the end.
RUNNING, FAILED, COMPLETE = "running", "failed", "complete"

# A run id or suite name: one plain path segment.
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
# Names the run tree uses for itself: a recipe (the first trial_id segment) with one of these names would
# put its trial directories among the text store or the Parquet tables.
_RUN_TREE_NAMES = frozenset({"texts", PARQUET_DIR, RESOLVED_RECIPE, TOOLCHAINS_JSON, PROVENANCE_JSON, RUN_MD})
# Recipe sections this runner does not carry out; a recipe that sets one is refused, never run without it. The
# adversary section is carried out by a stage that declares RUNS_PROPOSALS (task P16.5; _check_adversary).
_NOT_CARRIED_OUT = ("refine", "agents", "judges")
# The capability of the stage that aligns attempts (the oracle stage), which a stage that runs proposals follows.
_ALIGNS = "aligns"
# The capability of an executor that runs no program.
_COMPILE_ONLY = "compile_only"
# How long git may take to report the commit or the dirty flag, in seconds.
_GIT_TIMEOUT_S = 60.0
# How long a pinned compiler's --version may take, in seconds, and the name prefix of the fresh directory under
# TMPDIR it runs in.
_VERSION_TIMEOUT_S = 120.0
_VERSION_PROBE_PREFIX = "lassi-version-check."
# The --version check's arguments after the executable, for a class that declares no VERSION_ARGS (hipcc-gfx942
# declares its target before --version, since hipcc without one starts a program that asks for a GPU; OQ-002).
_VERSION_ARGS = ("--version",)
# The build dir, under the runs root, against which the compile layout is checked before a run (never created).
_LAYOUT_PROBE = "lassi-layout-check"
# The environment variables that name the roots every compile hides (P0.20), in order.
_COMPILE_HIDDEN_VARIABLES = ("HOME", "LASSI_SCRATCH", "LASSI_RUNS_ROOT")
# A context pack's name, as lassi.prompts.assets checks it: one plain path segment (_template_set_pack).
_PACK_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# The retrieval corpora under the assets root, and the index file's name where the generator writes it by default:
# $LASSI_SCRATCH/rag/<corpus>@<VERSION of the manifest's source pin>/index.json (task P17.22; _retrieval_index).
RAG_DIR = "rag"
INDEX_FILE = "index.json"


class RunError(RuntimeError):
    """A run that cannot start or a recipe choice the runner cannot carry out; the message says what to do."""


@dataclass(frozen=True)
class RunOptions:
    """How to run a recipe; every None falls back as each field says.

    - runs_root: the runs root; None means $LASSI_RUNS_ROOT, then the recipe's
      runs_root. The result must be absolute, outside the repository, and
      inside $LASSI_SCRATCH when that is set.
    - run_id: the run directory's name; None means the UTC start time as
      YYYYMMDD-HHMMSS.
    - bench_root: the fetched sources of the suite; None means
      lassi.bench.sources_dir($LASSI_SCRATCH, suite). A tracked suite takes
      none.
    - toolchains_root: where pinned toolchains are installed; None means
      $LASSI_TOOLCHAINS.
    - registry: the components a recipe may bind; None means DEFAULT_REGISTRY.
    - roots: the directories searched for a recipe named in `extends`; None
      means lassi.core.recipe.default_roots().
    - observer: gets the run's progress events (lassi.core.progress); None
      means none are sent. Nothing the run writes or returns depends on it.
    - probes: the device probes by kind (lassi.core.devices, task P17.2);
      None means lassi.core.devices.DEFAULT_PROBES, the host probes that
      lassi.executors registers.
    - silicon: the human start of a silicon run (task P17.16; Agent Rule 9;
      OQ-075 (6a)); only `lassi run --silicon` sets it, never a recipe or
      an environment variable. False refuses every executor that declares
      `silicon`, and True still refuses one beside a reward or beside model
      output (OQ-078; _check_silicon).
    """

    runs_root: Path | None = None
    run_id: str | None = None
    bench_root: Path | None = None
    toolchains_root: Path | None = None
    registry: Registry | None = None
    roots: Sequence[Path] | None = None
    observer: Observer | None = None
    probes: Mapping[str, DeviceProbe] | None = None
    silicon: bool = False


# The options run_recipe uses when none are given: every field falls back as RunOptions says.
_DEFAULT_OPTIONS = RunOptions()


@dataclass(frozen=True)
class _Settings:
    """The run choices read from a loaded recipe.

    `fragments` holds a fragment prompt set's fragments (empty for a template
    set) and `packs` the context packs by the language each serves.
    """

    backend: str
    model_id: str
    sampling: Sampling
    max_corrections: int | None
    prompts: str
    directions: tuple[Direction, ...]
    trials: int
    split: str
    fragments: Mapping[str, str] = dataclasses.field(default_factory=dict)
    packs: Mapping[str, str] = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class _Bench:
    """The suite, the root of its fetched sources, and the items the run covers, sorted."""

    suite: Suite
    root: Path
    items: tuple[str, ...]


@dataclass(frozen=True)
class BuiltToolchain:
    """One built toolchain: its registry name, the languages a recipe binds it for, the object, and its pins.

    build_toolchain returns one with no languages; the runner fills them in
    for each toolchain a recipe binds. `executable` and `environment` are
    None for a toolchain without a pin; `pins` maps each pin it uses to the
    pin file's pairs. `version` holds the non-blank lines the pinned
    compiler's `--version` printed when build_toolchain checked it against
    the pin's EXPECT_VERSION, and `version_status` the exit status it
    observed (empty and None without a pin).
    """

    name: str
    languages: tuple[str, ...]
    toolchain: Toolchain
    executable: str | None
    environment: dict[str, str] | None
    pins: dict[str, dict[str, str]]
    version: tuple[str, ...] = ()
    version_status: int | None = None


@dataclass(frozen=True)
class _DeclaredPins:
    """The pins one bound executor declares (its `pins`), where the recipe binds it, and the languages it serves.

    `languages` is None for the single form, whose executor serves every
    language; `label` names the executor and its recipe path for messages.
    """

    label: str
    languages: tuple[str, ...] | None
    pins: Mapping[str, Mapping[str, str]]


@dataclass(frozen=True)
class _Executors:
    """The run's executors and what provenance.json records of them.

    The single form binds one executor, `single`, for every language; the
    per-language form binds one per language in `by_language`, and `single`
    is None. `record` holds the manifest's executor keys: `executor` and
    `device` for the single form, `executor` and `devices` (each by
    language) for the per-language form. `declared` holds the pins each
    bound executor declares (task P4.11), in binding order.
    """

    single: Executor | None
    by_language: Mapping[str, Executor]
    record: Mapping[str, Any]
    declared: tuple[_DeclaredPins, ...] = ()

    def for_language(self, language: str) -> Executor:
        """Return the executor that runs programs in `language`."""
        return self.single if self.single is not None else self.by_language[language]

    def pins_for(self, language: str) -> list[Mapping[str, Mapping[str, str]]]:
        """Return the pins of the executor that runs programs in `language`, as one mapping per declaring executor."""
        return [item.pins for item in self.declared if item.languages is None or language in item.languages]


@dataclass(frozen=True)
class _Run:
    """Everything the trial loop and the run files read: the recipe, its components, the bench, and provenance.

    `pins` holds the pin versions of every bound toolchain and every bound
    executor that declares pins, and `target_pins` those of the toolchain
    that builds each target language and of the executor that runs it
    (_run_pins). `scoring` holds the
    built ScoreProfiles of `score` and `metrics` (lassi.scoring.run_scoring).
    `observer` is RunOptions.observer in a GuardedObserver, or None.
    `devices` holds the record of each device section, in binding order
    (probe_devices). `serving` is the backend's serving record when it declares
    model_check, else None (_served_model). `profiler` is the run's one
    Profiler, built before the first trial (_built_profiler), or None when
    the recipe binds none. `retrieval` is the run's one documentation index
    (_retrieval_index), or None when the recipe sets no retrieval section.
    `examples` is the run's worked examples pack (_examples_pack), empty
    when the recipe sets no examples section. `skeleton` is the run's host
    program skeleton (_run_skeleton), or None when the recipe sets no
    skeleton section. `dialect` is the run's dialect definitions
    (_dialect_defs; task P17.41), or None without a dialect_lookup section,
    and `frontend` its lowering toolchain (_frontend), or None without a
    source_mlir section.
    """

    recipe: Recipe
    registry: Registry
    settings: _Settings
    bench: _Bench
    backend: Any
    executors: _Executors
    toolchains: tuple[BuiltToolchain, ...]
    pins: ToolchainPins
    target_pins: Mapping[str, ToolchainPins]
    run_dir: Path
    store: TextStore
    started: datetime
    commit: str | None
    dirty: bool | None
    scoring: ScoringPlan
    observer: GuardedObserver | None = None
    devices: tuple[DeviceRecord, ...] = ()
    serving: Mapping[str, Any] | None = None
    profiler: Profiler | None = None
    retrieval: Index | None = None
    examples: tuple[examples_pack.Example, ...] = ()
    skeleton: skeleton_files.Skeleton | None = None
    dialect: DialectDefs | None = None
    frontend: BuiltToolchain | None = None


def run_recipe(path: Path, options: RunOptions = _DEFAULT_OPTIONS) -> Path:
    """Run the recipe at `path` and return its run directory, `<runs root>/runs/<run_id>`.

    Raises RecipeError when the recipe does not load and RunError when the
    run cannot start (see the module docstring); both come before any
    directory is created or any chat request is sent. A backend that
    declares model_check has its model list and version read then
    (_served_model). A component's own errors,
    such as SandboxUnavailableError from an executor or a sandboxed compile,
    propagate; one raised during the trials leaves provenance.json with
    status "failed", and so does a RunError from a ScoreProfile that cannot
    score a trial or from metric tables that cannot be built. The
    toolchains' --version checks run in the compile sandbox before any
    directory is created, so a sandbox that cannot run them raises
    SandboxUnavailableError then.
    """
    run = _prepare(Path(path), options, datetime.now(timezone.utc))
    run_dir = run.run_dir
    _write(run_dir / RESOLVED_RECIPE, resolved_yaml(run.recipe))
    _write(run_dir / TOOLCHAINS_JSON, json_text(_toolchains_record(run.toolchains)))
    manifest = _provenance(run)
    _write(run_dir / PROVENANCE_JSON, json_text(manifest))
    try:
        trials = _run_trials(run, manifest)
        metrics = _metrics(run, trials)
        provenance = _final_provenance(manifest, COMPLETE)
        _write(run_dir / RUN_MD, _run_md(run, provenance, trials, metrics))
        write_run_parquet(trials, run_dir / PARQUET_DIR)
        if metrics is not None:
            write_metrics(metrics, run_dir / PARQUET_DIR)
    except BaseException:
        _write(run_dir / PROVENANCE_JSON, json_text(_final_provenance(manifest, FAILED)))
        raise
    _write(run_dir / PROVENANCE_JSON, json_text(provenance))
    print(f"run directory: {run_dir}", flush=True)
    return run_dir


def _prepare(path: Path, options: RunOptions, started: datetime) -> _Run:
    """Load and check the recipe, build its components, and create the run directory; RunError before any mkdir."""
    registry = DEFAULT_REGISTRY if options.registry is None else options.registry
    recipe = _load(path, options, registry)
    _check_silicon(recipe, registry, options.silicon)
    settings = _settings(recipe)
    runs_root = resolve_runs_root(options.runs_root, recipe)
    run_dir = runs_root / RUNS_DIR / run_id_for(options.run_id, started)
    check_location(run_dir, f"the run directory {run_dir}")
    if run_dir.exists():
        raise RunError(f"the run directory {run_dir} already exists; choose another run id")
    bench = _bench(recipe, settings, options)
    _check_plan(recipe, registry, settings, bench)
    retrieval = _retrieval_index(recipe)
    examples = _examples_pack(recipe)
    skeleton = _run_skeleton(recipe)
    dialect = _dialect_defs(recipe)
    _check_adversary(recipe, registry, settings, bench)
    devices = probe_devices(recipe, registry, DEFAULT_PROBES if options.probes is None else options.probes)
    _check_one_role(recipe, registry, devices)
    _check_profiler(recipe, registry, settings, devices)
    _check_oracle(recipe, registry, bench)
    scoring = _scoring(recipe, registry, bench)
    backend = _backend(recipe, registry, settings)
    _check_backend(recipe, settings, backend)
    serving = _served_model(recipe, settings, backend)
    toolchains = _toolchains(recipe, registry, _toolchains_root(options), runs_root)
    frontend = _frontend(recipe, registry, _toolchains_root(options), runs_root)
    toolchains = toolchains if frontend is None else (*toolchains, frontend)
    executors = _executors(recipe, registry, settings)
    profiler = _built_profiler(recipe, registry)
    pins, target_pins = _run_pins(recipe, settings, toolchains, executors)
    commit, dirty = git_state()
    try:
        run_dir.mkdir(parents=True)
    except FileExistsError:
        raise RunError(f"the run directory {run_dir} already exists; choose another run id") from None
    return _Run(
        recipe=recipe,
        registry=registry,
        settings=settings,
        bench=bench,
        backend=backend,
        executors=executors,
        toolchains=toolchains,
        pins=pins,
        target_pins=target_pins,
        run_dir=run_dir,
        store=TextStore(run_dir),
        started=started,
        commit=commit,
        dirty=dirty,
        scoring=scoring,
        observer=None if options.observer is None else GuardedObserver(options.observer),
        devices=devices,
        serving=serving,
        profiler=profiler,
        retrieval=retrieval,
        examples=examples,
        skeleton=skeleton,
        dialect=dialect,
        frontend=frontend,
    )


def probe_devices(recipe: Recipe, registry: Registry, probes: Mapping[str, DeviceProbe]) -> tuple[DeviceRecord, ...]:
    """Probe the device of every binding whose component declares takes_device; RunError for one the host lacks.

    Bindings go in recipe order (the model's first). For each, the section is
    read (lassi.core.devices.parse_device), the component class's framework
    build is asked for (framework_build), and then the kind's probe runs
    (probe_device). It runs before any component is built and before any
    directory exists, and builds the records in memory. A refusal names the
    key, the kind, and the reason; no run falls back to another device.
    """
    records: list[DeviceRecord] = []
    for binding in recipe.bindings:
        entry = registry.get(binding.interface, binding.name)
        if TAKES_DEVICE not in entry.capabilities:
            continue
        key = device_path(binding)
        spec = parse_device(binding.config[DEVICE_KEY], key)
        framework = framework_build(recipe, binding, entry)
        try:
            records.append(probe_device(key, spec, framework, probes))
        except DeviceUnavailable as error:
            named = " ".join([spec.kind, *(str(index) for index in spec.indices)])
            raise RunError(
                f"{recipe.path}: {key} names {named}, which this host cannot provide: {error}; "
                "a run never falls back to another device"
            ) from None
    return tuple(records)


def _check_one_role(recipe: Recipe, registry: Registry, devices: Sequence[DeviceRecord]) -> None:
    """Refuse a run in which the model's GPU is also an executor's, unless the backend unloads before each run.

    One GPU, one role (task P17.5; Agent Rule 13's intent for GPUs): the
    LLMBackend binding's device record (model.device) and the record of
    any Executor binding, single or per language, that name the same kind
    and share an index are a RunError naming both keys, the kind, and the
    shared indices. A model without a device section or on kind cpu, and a
    backend that declares unload_before_run (the runner unloads it before
    each run of generated code), pass. The records are compared as
    written, kind and indices. It runs after every probe and before any
    component is built.
    """
    records = {record.key: record for record in devices}
    model = next(binding for binding in recipe.bindings if binding.interface == "LLMBackend")
    held = records.get(device_path(model))
    if held is None or held.kind == "cpu":
        return
    if UNLOAD_BEFORE_RUN in registry.get(model.interface, model.name).capabilities:
        return
    for binding in recipe.bindings:
        record = records.get(device_path(binding)) if binding.interface == "Executor" else None
        if record is None or record.kind != held.kind:
            continue
        shared = sorted(set(held.indices) & set(record.indices))
        if shared:
            named = ", ".join(str(index) for index in shared)
            raise RunError(
                f"{recipe.path}: {held.key} and {record.key} both name {held.kind} index {named}; a model's GPU "
                f"never runs candidates unless its backend declares {UNLOAD_BEFORE_RUN!r} (one GPU, one role)"
            )


def _profiler_binding(recipe: Recipe) -> Binding | None:
    """Return the recipe's Profiler binding (profiler.kind), or None when it binds no profiler."""
    return next((binding for binding in recipe.bindings if binding.interface == "Profiler"), None)


def _check_profiler(recipe: Recipe, registry: Registry, settings: _Settings, devices: Sequence[DeviceRecord]) -> None:
    """Refuse a profiler the run cannot honor, before any component is built and before any directory exists.

    Task P17.7. The profiled executors are those that run the attempts of
    the directions' target languages and run programs
    (_profiled_executors). Refused: a profiler with nothing to profile (no
    listed stage declares runs_model_code, or no profiled executor), since
    the runner never ignores a choice; a profiled executor that declares
    simulator, whose timing is never performance (Agent Rule 2); a profiler
    that declares supports_power without takes_device; and, for one that
    declares supports_power, a profiled executor that does not run on the
    profiler's device (_check_power_device). It runs after the probes, so
    the profiler's own device (profiler.device) is one this host has.
    """
    binding = _profiler_binding(recipe)
    if binding is None:
        return
    entry = registry.get(binding.interface, binding.name)
    named = f"Profiler {entry.name!r} ({binding.where})"
    stages = recipe.data["stages"]
    if not any(getattr(registry.get("Stage", name).factory, "runs_model_code", False) for name in stages):
        raise RunError(
            f"{recipe.path}: {named} would profile nothing: no listed stage runs attempts (none declares "
            "runs_model_code; run_loop does); list run_loop or remove the profiler section"
        )
    profiled = _profiled_executors(recipe, registry, settings)
    if profiled is None:
        return
    if not profiled:
        raise RunError(
            f"{recipe.path}: {named} would profile nothing: no target language's executor runs programs "
            f"(none declares {RUNS_CODE!r}); remove the profiler section"
        )
    for executor, executor_entry in profiled:
        if SIMULATOR in executor_entry.capabilities:
            raise RunError(
                f"{recipe.path}: {named} would profile the runs of Executor {executor_entry.name!r} "
                f"({executor.where}), which declares {SIMULATOR!r}; simulator timing is never performance "
                "(Agent Rule 2)"
            )
    if SUPPORTS_POWER not in entry.capabilities:
        return
    if TAKES_DEVICE not in entry.capabilities:
        raise RunError(
            f"{recipe.path}: {named} declares {SUPPORTS_POWER!r} without {TAKES_DEVICE!r}; a power profiler names "
            "the one GPU it reads in its device section"
        )
    records = {record.key: record for record in devices}
    for executor, executor_entry in profiled:
        _check_power_device(recipe, records[device_path(binding)], executor, executor_entry, records)


def _profiled_executors(recipe: Recipe, registry: Registry, settings: _Settings) -> list[tuple[Binding, Entry]] | None:
    """Return the Executor bindings whose runs a profiler profiles, with their entries, each once, in target order.

    For each target language of the directions, that is the executor that
    runs its attempts (the single one, or executor.<language> with
    executors per language) when it declares runs_code; a compile-only
    target never runs an attempt. A language that is only a source is not
    profiled, since the baseline's reference runs are not. With executors
    per language, it returns None when a target language has no executor:
    _executors refuses that run, naming executor.<language>, before the
    profiler is built.
    """
    bindings = [binding for binding in recipe.bindings if binding.interface == "Executor"]
    languages = executor_languages(recipe.data)
    by_language = dict(zip(languages, bindings, strict=True)) if languages else {}
    found: dict[str, tuple[Binding, Entry]] = {}
    for direction in settings.directions:
        binding = by_language.get(direction.target) if languages else bindings[0]
        if binding is None:
            return None
        entry = registry.get(binding.interface, binding.name)
        if RUNS_CODE in entry.capabilities:
            found.setdefault(binding.where, (binding, entry))
    return list(found.values())


def _check_power_device(
    recipe: Recipe, held: DeviceRecord, executor: Binding, entry: Entry, records: Mapping[str, DeviceRecord]
) -> None:
    """Refuse a profiled executor that does not run on a power profiler's device (`held`, the profiler.device record).

    The executor must declare takes_device, and its device record must name
    the profiler's kind and the same indices, as written, so every attempt
    run of the run is on the GPU whose power is read and none is measured on
    one GPU while it runs on another. The message names profiler.device and
    the executor's device key, or its binding path when it takes no device.
    """
    named = " ".join([held.kind, *(str(index) for index in held.indices)])
    if TAKES_DEVICE not in entry.capabilities:
        raise RunError(
            f"{recipe.path}: {held.key} names {named}, but Executor {entry.name!r} ({executor.where}) takes no "
            "device section, so its runs are not on the GPU whose power is read; a power profiler needs each "
            "profiled executor on its device"
        )
    record = records[device_path(executor)]
    if record.kind != held.kind or list(record.indices) != list(held.indices):
        runs = " ".join([record.kind, *(str(index) for index in record.indices)])
        raise RunError(
            f"{recipe.path}: {held.key} names {named}, but {record.key} names {runs}; a power profiler reads the "
            "GPU every profiled attempt runs on, so both name the same kind and indices"
        )


def _built_profiler(recipe: Recipe, registry: Registry) -> Profiler | None:
    """Return the recipe's Profiler built as factory(**config), or None when it binds none; RunError before any mkdir.

    Task P17.7. It is built once per run, after the executors and before
    the first trial. A setting it refuses (ValueError, such as a missing
    profiler.interval_ms) is a RunError naming the profiler and the
    refusal; telemetry it cannot read on this host (a DeviceUnavailable,
    which lassi.profilers.power.TelemetryUnavailable is) is a RunError
    naming profiler.device and the run's lack of telemetry. Any other error
    propagates as it is.
    """
    binding = _profiler_binding(recipe)
    if binding is None:
        return None
    entry = registry.get(binding.interface, binding.name)
    try:
        return entry.factory(**binding.config)
    except DeviceUnavailable as error:
        raise RunError(
            f"{recipe.path}: {device_path(binding)}: Profiler {entry.name!r} ({binding.where}) cannot read power "
            f"on this host: {error}; the run lacks its telemetry"
        ) from None
    except ValueError as error:
        raise RunError(
            f"{recipe.path}: Profiler {entry.name!r} ({binding.where}) refused its settings: {error}"
        ) from None


def framework_build(recipe: Recipe, binding: Binding, entry: Entry) -> FrameworkBuild | None:
    """Return the framework build a component class names with framework(), or None when it defines none.

    framework() is called on the class, before anything is built, and reads
    build metadata only. A result that is neither a FrameworkBuild nor None
    is a RunError naming the component, and so is any error it raises (a
    framework that is not installed, or a framework() the class cannot
    call), naming the device key too.
    """
    method = getattr(entry.factory, "framework", None)
    if not callable(method):
        return None
    try:
        build = method()
    except Exception as error:
        raise RunError(
            f"{recipe.path}: {device_path(binding)}: {entry.interface} {entry.name!r} ({binding.where}) could not "
            f"report its framework build: {type(error).__name__}: {error}"
        ) from None
    if build is not None and not isinstance(build, FrameworkBuild):
        raise RunError(
            f"{recipe.path}: {entry.interface} {entry.name!r} ({binding.where}) framework() returned {build!r}; "
            "it must return a FrameworkBuild or None"
        )
    return build


def _backend_config(recipe: Recipe) -> Mapping[str, Any]:
    """Return the config of the recipe's LLMBackend binding.

    It holds the model section's keys other than backend, id, and
    recording_sha256, with model.recording as the file the loader checked
    (lassi.core.recipe _bind_recording; task P16.13).
    """
    return next(binding.config for binding in recipe.bindings if binding.interface == "LLMBackend")


def _backend(recipe: Recipe, registry: Registry, settings: _Settings) -> Any:
    """Build the backend as factory(model.id, **config); a setting it refuses (ValueError) is a RunError.

    The HTTP backends' ValueError messages name the setting and never quote its
    value (lassi.llm._http.checked_base_url), and hf_local's quote only a
    model.id, model.revision, model.seed, or model.device value, none a
    secret, so the RunError quotes them. Construction sends no request. A
    backend that takes model.recording reads the file again here, so an
    OSError (a file removed since load) is a RunError naming model.recording,
    and _check_recording compares the bytes it read with the loader's. When
    the recipe sets llm.sampling.reasoning_effort, the factory also gets
    reasoning_effort=<value> (_reasoning_effort).
    """
    factory = registry.get("LLMBackend", settings.backend).factory
    config = dict(_backend_config(recipe))
    effort = _reasoning_effort(recipe, registry, settings)
    if effort is not None:
        config[REASONING_EFFORT] = effort
    try:
        backend = factory(settings.model_id, **config)
    except ValueError as error:
        raise RunError(f"{recipe.path}: model: LLMBackend {settings.backend!r} refused its settings: {error}") from None
    except OSError as error:
        if RECORDING_KEY not in config:
            raise
        raise RunError(
            f"{recipe.path}: model.{RECORDING_KEY}: LLMBackend {settings.backend!r} could not read its recording "
            f"{config[RECORDING_KEY]} ({error.strerror or error}); it was readable when the recipe loaded"
        ) from None
    _check_recording(recipe, settings, backend)
    return backend


def _reasoning_effort(recipe: Recipe, registry: Registry, settings: _Settings) -> str | None:
    """Return llm.sampling.reasoning_effort, or None when the recipe does not set it (OQ-080).

    A set value goes only to a backend that declares REASONING_EFFORT, read
    from its registry entry, never its name; any other backend is a RunError
    naming the key, raised before the backend is built and before any
    directory exists, so no run claims an effort its requests never carried.
    """
    effort = recipe.data["llm"]["sampling"].get(REASONING_EFFORT)
    if effort is None or REASONING_EFFORT in registry.get("LLMBackend", settings.backend).capabilities:
        return effort
    raise RunError(
        f"{recipe.path}: {REASONING_EFFORT_KEY} is {effort}, but LLMBackend {settings.backend!r} does not declare "
        f"{REASONING_EFFORT!r}, so its requests would not carry it; remove {REASONING_EFFORT_KEY} or bind a backend "
        "that declares it, such as openai_compat"
    )


def _check_recording(recipe: Recipe, settings: _Settings, backend: Any) -> None:
    """Refuse a backend whose recording bytes are not the ones model.recording_sha256 records (task P16.13).

    The loader hashes model.recording into the resolved recipe, and the
    backend reads the file again when it is built, reporting the sha256 of
    the bytes it read as recording_sha256 (lassi.llm.replay). A recording
    changed in between, or a backend that reports no sha256, is a RunError
    before any directory exists, so recipe.resolved.yaml describes the bytes
    the trials replay (Agent Rule 10).
    """
    stated = recipe.data.get("model", {}).get(RECORDING_SHA256_KEY)
    if stated is None:
        return
    read = getattr(backend, RECORDING_SHA256_KEY, None)
    where = f"{recipe.path}: model.{RECORDING_SHA256_KEY}: LLMBackend {settings.backend!r}"
    if read is None:
        raise RunError(
            f"{where} takes model.{RECORDING_KEY} but reports no {RECORDING_SHA256_KEY} (the sha256 of the bytes it "
            "read), so the run cannot show that the resolved recipe describes the recording it replays"
        )
    if read != stated:
        raise RunError(
            f"{where} read model.{RECORDING_KEY} with the sha256 {read}, but the recipe loaded it with the sha256 "
            f"{stated}; the recording changed after the recipe was loaded, so rerun once it holds the bytes intended"
        )


def _served_model(recipe: Recipe, settings: _Settings, backend: Any) -> Mapping[str, Any] | None:
    """Return the backend's serving record when it declares model_check, else None; RunError when it cannot.

    serving() confirms the model id on the server and reads the record
    (lassi.llm._serving) before any directory exists; the trials reuse the
    entry it checked. Its ServingError (a model the server does not list, an
    unreachable server, a timeout, an unset key variable, a record that would
    hold the key) quotes no key and no base_url value. The record is rendered
    as provenance.json writes it once here, so one that JSON cannot hold
    (NaN) is refused before any directory too (plans/LESSONS.md, Audits).
    The message names model.base_url only for a backend that takes it, so an
    in-process backend's failed load (task P17.4) names model.id alone.
    """
    if not declares(backend, MODEL_CHECK):
        return None
    try:
        record = backend.serving()
        json_text(record)
    except (ServingError, ValueError) as error:
        where = " at model.base_url" if "base_url" in getattr(backend, "config_keys", ()) else ""
        raise RunError(
            f"{recipe.path}: model.id: LLMBackend {settings.backend!r} could not confirm its model{where}: {error}"
        ) from None
    return record


def _scoring(recipe: Recipe, registry: Registry, bench: _Bench) -> ScoringPlan:
    """Build the ScoreProfiles `score` and `metrics` need and check every metrics name; RunError says why not.

    lassi.scoring.run_scoring.plan_scoring decides, with the run's bench
    root for a profile that reads bench sources, before any directory is
    created or any model is asked.
    """
    try:
        return plan_scoring(recipe.data, registry, bench.root)
    except (OSError, ValueError) as error:
        raise RunError(f"{recipe.path}: {error}") from error


def _check_backend(recipe: Recipe, settings: _Settings, backend: Any) -> None:
    """Refuse a backend that lacks a method the run will call on it.

    A backend that declares `needs_reference` gets each item's reference
    target through with_reference(), and, with fixes.fence_tag off, is asked
    for its faithful reply form, one untagged fence, through
    with_untagged_fence(). One that declares `unload_before_run` needs unload(),
    and one that declares `model_check` needs serving().
    """
    if "needs_reference" in backend.capabilities:
        if not hasattr(backend, "with_reference"):
            raise RunError(f"LLMBackend {settings.backend!r} needs the reference target but has no with_reference()")
        if not recipe.data["fixes"]["fence_tag"] and not hasattr(backend, "with_untagged_fence"):
            raise RunError(
                f"LLMBackend {settings.backend!r} needs the reference target, but fixes.fence_tag is off and it has "
                "no with_untagged_fence() to answer in the one untagged fence that faithful extraction reads"
            )
    if declares(backend, MODEL_CHECK) and not callable(getattr(backend, "serving", None)):
        raise RunError(f"LLMBackend {settings.backend!r} declares {MODEL_CHECK!r} but has no serving()")
    if declares(backend, UNLOAD_BEFORE_RUN) and not callable(getattr(backend, "unload", None)):
        raise RunError(f"LLMBackend {settings.backend!r} declares {UNLOAD_BEFORE_RUN!r} but has no unload()")


# ---------------------------------------------------------------------------
# Reading the recipe and the options


def _load(path: Path, options: RunOptions, registry: Registry) -> Recipe:
    """Load the recipe; when the loader refuses one whose model section cannot run, raise the RunError that says why.

    A recipe with arms but no model fails the loader's binding check (a
    generating stage needs an LLMBackend, and only model binds one), so the
    resolved mapping is read again to explain the refusal.
    """
    try:
        return load_recipe(path, roots=options.roots, registry=registry)
    except RecipeError:
        problem = _model_problem(path, resolved_data(path, roots=options.roots))
        if problem is None:
            raise
        raise RunError(problem) from None


def _model_problem(path: Path, data: Mapping[str, Any]) -> str | None:
    """Return why the recipe's model section cannot run, or None when it names a backend and a model id."""
    model = data.get("model")
    if model is None:
        if isinstance(data.get("arms"), list) and data["arms"]:
            return (
                f"{path}: the recipe lists arms but no model; the model registry that maps arms to backends "
                "comes later, so set model: {backend: <name>, id: <model id>} to run one model"
            )
        return f"{path}: the recipe names no model; set model.backend and model.id"
    if not isinstance(model, dict):
        return None
    for key in ("backend", "id"):
        if key not in model:
            return f"{path}: model.{key} has no value; set it in the recipe"
    return None


def _settings(recipe: Recipe) -> _Settings:
    """Return the run choices of a loaded recipe; raise RunError for a choice the runner cannot carry out."""
    data = recipe.data
    problem = _model_problem(recipe.path, data)
    if problem is not None:
        raise RunError(problem)
    if data.get("arms"):
        raise RunError(
            f"{recipe.path}: the recipe sets both model and arms; the runner runs model alone and never drops arms "
            "silently, so remove arms (the model registry that maps arms to backends comes later)"
        )
    sampling = data["llm"]["sampling"]
    if "max_tokens" not in sampling:
        raise RunError(
            f"{recipe.path}: llm.sampling.max_tokens has no value; set it in the recipe (the runner never picks one)"
        )
    if sampling["max_tokens"] < 1:
        raise RunError(f"{recipe.path}: llm.sampling.max_tokens must be at least 1, not {sampling['max_tokens']}")
    if "prompts" not in data:
        raise RunError(f"{recipe.path}: prompts has no value; name the prompt set under assets/prompts/")
    _check_carried_out(recipe)
    fragments, packs = _recipe_assets(recipe)
    cap = data["loop"]["max_corrections"]
    return _Settings(
        fragments=fragments,
        packs=packs,
        backend=data["model"]["backend"],
        model_id=data["model"]["id"],
        sampling=Sampling(
            temperature=float(sampling["temperature"]),
            top_p=float(sampling["top_p"]),
            max_tokens=sampling["max_tokens"],
        ),
        max_corrections=None if cap == UNCAPPED else cap,
        prompts=data["prompts"],
        directions=tuple(Direction(entry["source"], entry["target"]) for entry in data["directions"]),
        trials=data["trials"]["n"],
        split=data["bench"]["split"],
    )


def _check_carried_out(recipe: Recipe) -> None:
    """Refuse a recipe choice the stages cannot honor, so the saved recipe never claims behavior the run lacked.

    A fix turned off is checked against the listed stages in _check_stages.
    """
    data = recipe.data
    for name, on in sorted(data["report"].items()):
        if not on:
            raise RunError(
                f"{recipe.path}: report.{name} is false, but the runner always writes trial.md and the Parquet "
                "mirror; set it to true"
            )
    unused = [key for key in _NOT_CARRIED_OUT if key in data]
    if unused:
        raise RunError(
            f"{recipe.path}: the recipe sets {', '.join(unused)}, which this runner does not carry out yet; "
            "remove them rather than have the run ignore them"
        )


def _recipe_assets(recipe: Recipe) -> tuple[dict[str, str], dict[str, str]]:
    """Return the fragments of the recipe's prompt set and its context packs keyed by the language each serves.

    A fragment set (a MANIFEST.yaml under the assets root) and its packs
    load with lassi.prompts.load_recipe_assets from the default assets
    root, checking every file against its manifest. A template set has no
    fragments (an empty mapping), and its packs load with
    _template_set_pack, checked the same way (task P17.21; the runner then
    checks that the set can serve them, _check_template_context). A pack
    serves the language its entry keys end in, read from its first key
    (lassi.core.fragments.pack_language; the loader refuses a pack whose
    keys end in different languages). Raises RunError, with the loader's
    message, for a set or pack that does not load, and when two packs serve
    one language.
    """
    data = recipe.data
    root = prompt_assets.default_root()
    names = data.get("context", [])
    manifest = root / "prompts" / str(data["prompts"]) / prompt_assets.MANIFEST
    try:
        if manifest.is_file():
            assets = load_recipe_assets(data, root=root)
            fragments, texts = dict(assets.fragments), dict(assets.packs)
        else:
            fragments, texts = {}, {name: _template_set_pack(root, name) for name in names}
        packs: dict[str, str] = {}
        by_language: dict[str, str] = {}
        for name in names:
            language = pack_language(next(iter(prompt_assets.load_tree(root / "context" / name))))
            if language in packs:
                raise RunError(
                    f"{recipe.path}: the context packs {by_language[language]!r} and {name!r} both serve "
                    f"{language!r}; name one pack per language"
                )
            packs[language], by_language[language] = texts[name], name
    except ValueError as error:
        raise RunError(f"{recipe.path}: {error}") from error
    return fragments, packs


def _template_set_pack(root: Path, name: Any) -> str:
    """Return the text of context pack `name` for a template prompt set, checked as load_recipe_assets checks one.

    load_recipe_assets loads packs only beside a fragment set's fragments,
    so here the same public steps run alone: `name` must be one plain path
    segment, <root>/context/<name> a directory, every file what its
    manifest records (lassi.prompts.assets load_tree), and the text is the
    entries joined (pack_text). Raises ValueError naming the pack.
    """
    if not isinstance(name, str) or not _PACK_NAME.fullmatch(name):
        raise ValueError(f"the recipe's context must name packs matching {_PACK_NAME.pattern}, got {name!r}")
    pack_dir = root / "context" / name
    if not pack_dir.is_dir():
        raise ValueError(f"unknown context pack {name!r}: {pack_dir} is not a directory")
    return prompt_assets.pack_text(name, prompt_assets.load_tree(pack_dir))


def _retrieval_index(recipe: Recipe) -> Index | None:
    """Load the documentation index the recipe's retrieval section names, once per run; None without the section.

    The corpus is <assets root>/RAG_DIR/<retrieval.index> under
    lassi.prompts.assets.default_root(), whose MANIFEST.yaml
    lassi.prompts.rag_corpus.read_manifest checks. The index file is where
    the generator writes it by default, $LASSI_SCRATCH/rag/<the manifest's
    corpus>@<VERSION of its source pin>/INDEX_FILE, and it loads only with
    the manifest's sha256 (lassi.prompts.retrieval load_index). A missing
    manifest, an unset LASSI_SCRATCH, and an index that is missing or
    fails its sha256 are each a RunError naming retrieval.index, raised
    before any directory exists.
    """
    section = recipe.data.get(RETRIEVAL)
    if section is None:
        return None
    where = f"{recipe.path}: {RETRIEVAL}.index {section['index']!r}"
    manifest = prompt_assets.default_root() / RAG_DIR / section["index"] / prompt_assets.MANIFEST
    if not manifest.is_file():
        raise RunError(
            f"{where} names no corpus: {manifest} does not exist; name a directory under assets/{RAG_DIR}/ that "
            f"holds its {prompt_assets.MANIFEST}"
        )
    scratch = os.environ.get("LASSI_SCRATCH", "")
    if not scratch:
        raise RunError(f"{where}: its index lies under $LASSI_SCRATCH, where {INDEX_GENERATOR} writes it, "
                       "but LASSI_SCRATCH is not set")
    try:
        document = rag_corpus.read_manifest(manifest)
        source = document["source"]
        pin = source.get("pin") if isinstance(source, dict) else None
        if not isinstance(pin, str):
            raise ValueError(f"{manifest} names no source pin; regenerate it with uv run {INDEX_GENERATOR} build")
        path = Path(scratch) / RAG_DIR / f"{document['corpus']}@{read_pin(pin)['VERSION']}" / INDEX_FILE
        return load_index(path, sha256=document["index"]["sha256"])
    except (OSError, KeyError, ValueError) as error:
        raise RunError(f"{where}: {error}") from error


def _examples_pack(recipe: Recipe) -> tuple[examples_pack.Example, ...]:
    """Load the worked examples pack the recipe's examples section names, once per run; () without the section.

    The pack is <assets root>/EXAMPLES_DIR/<examples.pack> under
    lassi.prompts.assets.default_root(), read only when every file matches
    its MANIFEST.yaml (lassi.prompts.examples load_pack). A missing manifest,
    and a fragment that is missing or fails it, are each a RunError naming
    examples.pack, raised before any directory exists.
    """
    section = recipe.data.get(EXAMPLES)
    if section is None:
        return ()
    where = f"{recipe.path}: {EXAMPLES}.pack {section['pack']!r}"
    tree = prompt_assets.default_root() / examples_pack.EXAMPLES_DIR / section["pack"]
    manifest = tree / examples_pack.MANIFEST
    if not manifest.is_file():
        raise RunError(
            f"{where} names no pack: {manifest} does not exist; name a directory under "
            f"assets/{examples_pack.EXAMPLES_DIR}/ that holds its {examples_pack.MANIFEST}"
        )
    try:
        return examples_pack.load_pack(tree)
    except (OSError, ValueError) as error:
        raise RunError(f"{where}: {error}") from error


def _run_skeleton(recipe: Recipe) -> skeleton_files.Skeleton | None:
    """Load the host program skeleton the recipe's skeleton section names, once per run; None without the section.

    The loader read it already and entered its sha256 at skeleton.sha256
    (lassi.core.recipe _bind_skeleton); it is read again here, from
    <assets root>/skeletons/<skeleton.name>, and kept only when it still
    loads and its sha256 is the resolved recipe's, so every trial's prompts
    carry the bytes the recipe hash names. Either failure is a RunError
    naming skeleton.name, raised before any directory exists.
    """
    section = recipe.data.get(SKELETON)
    if section is None:
        return None
    where = f"{recipe.path}: {SKELETON}.name {section['name']!r}"
    try:
        skeleton = skeleton_files.load_skeleton(skeleton_files.skeleton_dir(section["name"]))
    except (OSError, ValueError) as error:
        raise RunError(f"{where}: {error}") from error
    if skeleton.sha256 != section[SKELETON_SHA256_KEY]:
        raise RunError(
            f"{where}: the skeleton's sha256 is now {skeleton.sha256}, not the recipe's {SKELETON}."
            f"{SKELETON_SHA256_KEY} {section[SKELETON_SHA256_KEY]}; it changed after the recipe loaded, so load the "
            "recipe again"
        )
    return skeleton


def resolve_runs_root(option: Path | None, recipe: Recipe) -> Path:
    """Return the runs root: `option`, else $LASSI_RUNS_ROOT, else the recipe's runs_root; absolute, outside the repo.

    The root is checked with check_location. A recipe without runs_root (a
    train recipe may leave it out) and no other source is a RunError naming
    the three sources. `lassi run` and `lassi train` share it (task P17.8).
    """
    if option is not None:
        root, source = Path(option), "the runs_root option"
    elif os.environ.get("LASSI_RUNS_ROOT"):
        try:
            root = workdir.runs_root()
        except ValueError as error:
            raise RunError(str(error)) from error
        source = "$LASSI_RUNS_ROOT"
    elif "runs_root" in recipe.data:
        root, source = Path(recipe.data["runs_root"]), f"runs_root in {recipe.path}"
    else:
        raise RunError(
            f"{recipe.path}: no runs root: pass --runs-root, set LASSI_RUNS_ROOT, or set runs_root in the recipe"
        )
    if not root.is_absolute():
        raise RunError(f"the runs root {root} (from {source}) must be an absolute path")
    check_location(root, f"the runs root {root} (from {source})")
    return root


def _within(path: Path, root: Path) -> bool:
    """Return True when the resolved `path` is `root` or lies under it."""
    return path == root or root in path.parents


def check_location(path: Path, what: str) -> None:
    """Refuse a run path that resolves inside the repository, or outside $LASSI_SCRATCH when that is set.

    The path is resolved, so a link into the repository is refused too;
    `what` names the path in the message (Agent Rule 7). Run trees and
    train trees (task P17.8) follow it.
    """
    resolved = path.resolve()
    if _within(resolved, REPO):
        raise RunError(f"{what} is inside the repository {REPO}; run trees stay out of git (Agent Rule 7)")
    scratch = os.environ.get("LASSI_SCRATCH", "")
    if scratch and not _within(resolved, Path(scratch).resolve()):
        raise RunError(
            f"{what} is outside $LASSI_SCRATCH ({scratch}); on the build host run trees stay on the scratch disk, "
            "never on the root filesystem (Agent Rule 7)"
        )


def run_id_for(option: str | None, started: datetime) -> str:
    """Return a run or train id: `option`, else the UTC start time as YYYYMMDD-HHMMSS; it must be one plain name."""
    run_id = started.strftime("%Y%m%d-%H%M%S") if option is None else option
    if not isinstance(run_id, str) or not _NAME.fullmatch(run_id):
        raise RunError(f"a run id must match {_NAME.pattern}, got {run_id!r}")
    return run_id


def _toolchains_root(options: RunOptions) -> Path | None:
    """Return the toolchains root: the option, else $LASSI_TOOLCHAINS, else None."""
    if options.toolchains_root is not None:
        return Path(options.toolchains_root)
    value = os.environ.get("LASSI_TOOLCHAINS", "")
    return Path(value) if value else None


# ---------------------------------------------------------------------------
# The bench and the plan


def _bench(recipe: Recipe, settings: _Settings, options: RunOptions) -> _Bench:
    """Load the recipe's suite and find its fetched sources; the items are those `_items` selects, sorted."""
    name = recipe.data["bench"]["suite"]
    manifest = BENCH_DIR / f"{name}.yaml"
    if not _NAME.fullmatch(name) or not manifest.is_file():
        raise RunError(f"{recipe.path}: bench.suite {name!r} names no suite manifest; expected {manifest}")
    try:
        suite = load_suite(manifest)
    except (OSError, ValueError) as error:
        raise RunError(f"cannot load the suite manifest {manifest}: {error}") from error
    items = _items(recipe, suite, settings.split)
    return _Bench(suite=suite, root=_sources_root(suite, options), items=items)


def _items(recipe: Recipe, suite: Suite, split: str) -> tuple[str, ...]:
    """Return the items the run covers, sorted: those bench.items names, else every item of the recipe's split.

    A selected item the suite lacks, one outside the split, or one named
    twice is a RunError naming it.
    """
    in_split = sorted(item for item, spec in suite.items.items() if spec.split == split)
    selected = recipe.data["bench"].get("items")
    if selected is None:
        if not in_split:
            raise RunError(f"{recipe.path}: suite {suite.name!r} has no items in the split {split!r}")
        return tuple(in_split)
    for index, item in enumerate(selected):
        if item not in suite.items:
            raise RunError(
                f"{recipe.path}: bench.items names {item!r}, which suite {suite.name!r} does not have; "
                f"items: {', '.join(sorted(suite.items))}"
            )
        if item not in in_split:
            raise RunError(f"{recipe.path}: bench.items names {item!r}, which is not in the split {split!r}")
        if item in selected[:index]:
            raise RunError(f"{recipe.path}: bench.items names {item!r} twice")
    return tuple(sorted(selected))


def _fetch_hint(suite: Suite) -> str:
    """Return how to fetch a suite's pinned sources on the build host; a tracked suite's files are in the repository."""
    if suite.tracked:
        return f"{suite.name} is a tracked suite, so its files are in the repository under assets/; check them there"
    return f"fetch them on the build host with `uv run tools/fetch_bench.py assets/bench/{suite.name}.yaml`"


def _sources_root(suite: Suite, options: RunOptions) -> Path:
    """Return the root of the suite's fetched sources: the bench_root option, else under $LASSI_SCRATCH.

    A tracked suite (lassi.bench.registry; task P16.8) has no fetched
    sources: every file the run reads is tracked, so its root is
    TRACKED_ROOT, no LASSI_SCRATCH is needed, and a bench_root option given
    for it is a RunError, so no run reads elsewhere.
    """
    if suite.tracked:
        if options.bench_root is not None:
            raise RunError(
                f"{suite.name} is a tracked suite: every language is in the repository, so it takes no bench root, "
                f"not {options.bench_root}"
            )
        return bench_registry.TRACKED_ROOT
    if options.bench_root is not None:
        root = Path(options.bench_root)
    else:
        scratch = os.environ.get("LASSI_SCRATCH", "")
        if not scratch:
            raise RunError(
                f"no bench sources for {suite.name}: LASSI_SCRATCH is not set and no bench root was given; "
                + _fetch_hint(suite)
            )
        try:
            root = sources_dir(Path(scratch), suite)
        except ValueError as error:
            raise RunError(f"{error}; {_fetch_hint(suite)}") from error
    if not root.is_dir():
        raise RunError(f"the sources of {suite.name} are not at {root}; {_fetch_hint(suite)}")
    return root


def _check_sandbox(recipe: Recipe) -> None:
    """Refuse run limits the stages cannot apply: sandbox.wall_s and sandbox.mem_gb, read for every run of a program.

    wall_s is a number of seconds above 0 or BASELINE_X10 (ten times the
    reference run's wall time, lassi.core.stages attempt_limits), and mem_gb
    a number of GB above 0.
    """
    sandbox = recipe.data.get("sandbox", {})
    wall = sandbox.get("wall_s")
    if wall != BASELINE_X10 and not (isinstance(wall, (int, float)) and not isinstance(wall, bool) and wall > 0):
        raise RunError(
            f"{recipe.path}: sandbox.wall_s is {wall!r}, but an executor that runs programs needs a number of seconds "
            f"above 0 or {BASELINE_X10!r}; set it in the recipe"
        )
    memory = sandbox.get("mem_gb")
    if not (isinstance(memory, (int, float)) and not isinstance(memory, bool) and memory > 0):
        raise RunError(
            f"{recipe.path}: sandbox.mem_gb is {memory!r}, but an executor that runs programs needs a number of GB "
            "above 0; set it in the recipe"
        )


def _check_silicon(recipe: Recipe, registry: Registry, silicon: bool) -> None:
    """Refuse a silicon executor in a run a human did not start, in a run with a reward, or beside model output.

    Each bound executor that declares SILICON, in the single form or for
    any language, is refused when a listed stage declares COMPUTES_REWARD,
    or runs model-generated code or an adversary's proposals
    (_model_output_stage), whatever `silicon` says, and otherwise unless
    `silicon` (RunOptions, set only by `lassi run --silicon`) is True: RL
    rewards never execute on silicon, and silicon runs are evaluation runs a
    human starts (Agent Rule 9; OQ-075 (6a)). Model output waits on OQ-078:
    under the cards' identity IOMMU domain a program holding a card can
    reach host memory through the card's DMA, which the sandbox cannot stop
    (plans/spikes/tt-card-dma.md), so until the owner answers, silicon runs
    only tracked references a human starts (its recommendation (c)). The
    runner reads the capabilities and the attribute, never a name (Agent
    Rule 3), and builds nothing here.
    """
    for binding in recipe.bindings:
        if binding.interface != "Executor" or SILICON not in registry.get("Executor", binding.name).capabilities:
            continue
        named = f"Executor {binding.name!r} ({binding.where})"
        for stage in recipe.data["stages"]:
            if COMPUTES_REWARD in registry.get("Stage", stage).capabilities:
                raise RunError(
                    f"{recipe.path}: stage {stage!r} computes a reward, and {named} runs programs on silicon; RL "
                    "rewards never execute on silicon, with or without --silicon (Agent Rule 9)"
                )
        found = _model_output_stage(recipe, registry)
        if found is not None:
            stage, what = found
            raise RunError(
                f"{recipe.path}: stage {stage!r} runs {what}, and {named} runs programs on silicon, where the sandbox "
                "does not keep a program from host memory (a card's DMA under the cards' identity IOMMU domain; "
                "plans/spikes/tt-card-dma.md); until OQ-078 is answered, silicon runs only tracked references a "
                "human starts, with or without --silicon"
            )
        if not silicon:
            raise RunError(
                f"{recipe.path}: {named} runs programs on silicon, and silicon runs are evaluation runs a human starts "
                "(Agent Rule 9): start this run with `lassi run <recipe> --silicon`, which no recipe can set and no "
                "unattended session passes"
            )


def _model_output_stage(recipe: Recipe, registry: Registry) -> tuple[str, str] | None:
    """Return the first listed stage that runs model output, with what it runs, or None when none does.

    A stage runs model output when its factory declares `runs_model_code`
    (run_loop) or its entry declares RUNS_PROPOSALS (the adversary stage).
    """
    for stage in recipe.data["stages"]:
        entry = registry.get("Stage", stage)
        if getattr(entry.factory, "runs_model_code", False):
            return stage, "model-generated code"
        if RUNS_PROPOSALS in entry.capabilities:
            return stage, "an adversary's proposals"
    return None


def _check_sandboxed(recipe: Recipe, registry: Registry, binding: Binding, capabilities: frozenset[str]) -> None:
    """Refuse an executor that runs programs outside the sandbox when a listed stage runs model-generated code.

    Such a stage declares `runs_model_code` (run_loop); the executor must
    then declare SANDBOXED (Agent Rule 6). baseline runs only bench
    references, so it needs no such declaration. With executors per
    language, every bound executor that runs programs is checked, and the
    message names its recipe path, executor.<language>.
    """
    if SANDBOXED in capabilities:
        return
    for name in recipe.data["stages"]:
        if getattr(registry.get("Stage", name).factory, "runs_model_code", False):
            raise RunError(
                f"{recipe.path}: stage {name!r} runs model-generated code, but Executor {binding.name!r} "
                f"({binding.where}) runs programs without declaring {SANDBOXED!r}; model-generated code runs only "
                "in the sandbox (Agent Rule 6)"
            )


def _executors(recipe: Recipe, registry: Registry, settings: _Settings) -> _Executors:
    """Check and build the recipe's executors, one or one per language, and name each one's device.

    Everything is checked before any executor is built: the languages the
    per-language form must bind (_check_executor_languages), the sandbox
    limits when any bound executor runs programs, and the sandboxed check for
    each one that does. Each executor is then built as factory(**config),
    asked for its device() once (_device, _checked_device), and read for
    the pins it declares (_declared_pins); RunError before any directory
    exists.
    """
    bindings = [binding for binding in recipe.bindings if binding.interface == "Executor"]
    languages = executor_languages(recipe.data)
    if languages:
        _check_executor_languages(recipe, registry, settings, set(languages))
    entries = [registry.get("Executor", binding.name) for binding in bindings]
    if any(RUNS_CODE in entry.capabilities for entry in entries):
        _check_sandbox(recipe)
    for binding, entry in zip(bindings, entries, strict=True):
        if RUNS_CODE in entry.capabilities:
            _check_sandboxed(recipe, registry, binding, entry.capabilities)
    built = [_built_executor(recipe, binding, entry) for binding, entry in zip(bindings, entries, strict=True)]
    devices = [_checked_device(_device(executor), binding) for executor, binding in zip(built, bindings, strict=True)]
    # The single form's one executor serves every language (None); each per-language executor serves its own.
    served: list[tuple[str, ...] | None] = [(language,) for language in languages] if languages else [None]
    declared = tuple(
        _declared_pins(executor, binding, serves)
        for executor, binding, serves in zip(built, bindings, served, strict=True)
    )
    if not languages:
        record = {"executor": bindings[0].name, "device": devices[0]}
        return _Executors(single=built[0], by_language={}, record=record, declared=declared)
    record = {
        "executor": {language: binding.name for language, binding in zip(languages, bindings, strict=True)},
        "devices": dict(zip(languages, devices, strict=True)),
    }
    by_language = dict(zip(languages, built, strict=True))
    return _Executors(single=None, by_language=by_language, record=record, declared=declared)


def _built_executor(recipe: Recipe, binding: Binding, entry: Entry) -> Executor:
    """Return factory(**config) for an Executor binding; a DeviceUnavailable it raises becomes a RunError.

    The gpu executor raises DeviceUnavailable at construction for a node or
    an index this host lacks (task P17.5); the RunError names the device
    key, the executor, and the reason. Any other error propagates as it is,
    so a refused setting stays a ValueError.
    """
    try:
        return entry.factory(**binding.config)
    except DeviceUnavailable as error:
        raise RunError(
            f"{recipe.path}: {device_path(binding)}: Executor {entry.name!r} ({binding.where}) cannot use its device "
            f"on this host: {error}; a run never falls back to another device"
        ) from None


def _check_executor_languages(recipe: Recipe, registry: Registry, settings: _Settings, bound: set[str]) -> None:
    """Refuse executors per language that leave a language the run needs without an executor.

    Every direction's target language needs one, since the attempts run
    there. A stage that builds the source reference under a fix that is on
    (`source_build_fix`, baseline under baseline_both) runs it on the source
    language's executor, so that language needs one too. Each message names
    the key to set, executor.<language>.
    """
    for direction in settings.directions:
        if direction.target not in bound:
            raise RunError(
                f"{recipe.path}: no executor is bound for the target language {direction.target!r}; "
                f"set executor.{direction.target}"
            )
    for name in recipe.data["stages"]:
        fix = getattr(registry.get("Stage", name).factory, "source_build_fix", None)
        if fix is None or recipe.data["fixes"].get(fix, True) is False:
            continue
        for direction in settings.directions:
            if direction.source not in bound:
                raise RunError(
                    f"{recipe.path}: stage {name!r} builds the source reference too while fixes.{fix} is on, and "
                    f"runs it on the source language's executor, but no executor is bound for {direction.source!r}; "
                    f"set executor.{direction.source}, or turn fixes.{fix} off"
                )


def _checked_device(device: Any, binding: Binding) -> str | None:
    """Return an executor's device after checking it: None, or one non-empty line of printable ASCII.

    The line has no leading or trailing blank (whitespace). A device enters
    provenance.json, every trial's provenance, and run.md, so anything else
    (a tab or another character that is not printable ASCII, a line break,
    a leading or trailing blank) is a RunError naming the binding, before
    any directory exists.
    """
    if device is None:
        return None
    if not isinstance(device, str) or not device.isascii() or not device.isprintable() or device.strip() != device:
        raise RunError(
            f"Executor {binding.name!r} ({binding.where}): device() must return one non-empty line of printable "
            f"ASCII with no leading or trailing blank, got {device!r}"
        )
    if not device:
        raise RunError(f"Executor {binding.name!r} ({binding.where}): device() returned an empty string")
    return device


def _declared_pins(executor: Executor, binding: Binding, languages: tuple[str, ...] | None) -> _DeclaredPins:
    """Return the pins an executor declares, checked, for the languages it serves (None: every language).

    An executor may declare `pins` (task P4.11; Agent Rule 10), as
    BuiltToolchain.pins holds a toolchain's: a mapping from pin name to the
    pin file's pairs, each with a VERSION string; an executor without it
    declares none. Anything else is a RunError naming the binding, before
    any directory exists. The runner reads the attribute and never the
    executor's name (Agent Rule 3).
    """
    label = f"Executor {binding.name!r} ({binding.where})"
    pins = getattr(executor, "pins", None)
    if pins is None:
        return _DeclaredPins(label, languages, {})
    valid = isinstance(pins, Mapping) and all(
        isinstance(name, str) and isinstance(pin, Mapping) and isinstance(pin.get("VERSION"), str)
        for name, pin in pins.items()
    )
    if not valid:
        raise RunError(
            f"{label}: pins must map each pin name to the pin file's pairs, each with a VERSION, as a toolchain's "
            f"pins do; got {pins!r}"
        )
    return _DeclaredPins(label, languages, {name: dict(pin) for name, pin in pins.items()})


def _check_plan(recipe: Recipe, registry: Registry, settings: _Settings, bench: _Bench) -> None:
    """Refuse, before anything is built, a plan whose project, stages, prompts, trial ids, or sources cannot work.

    The project (the recipe's `project` key, else its name) is the first
    trial id segment, so it must be one plain segment that names no entry of
    the run tree.
    """
    project = recipe.data["project"]
    if not _NAME.fullmatch(project):
        raise RunError(
            f"{recipe.path}: the project {project!r} is the first trial id segment, so it must match {_NAME.pattern}; "
            "set the project key"
        )
    if project in _RUN_TREE_NAMES:
        raise RunError(f"{recipe.path}: the project may not be {project!r}, which the run tree uses itself")
    _check_stages(recipe, registry, settings)
    _check_trials(recipe, settings, bench)
    _check_source_mlir(recipe, registry, settings, bench)


# What generate.txt's source_mlir field is filled with to see whether the template shows it (_check_source_mlir).
_SOURCE_MLIR_PROBE = "lassi-source-mlir-probe"


def _check_source_mlir(recipe: Recipe, registry: Registry, settings: _Settings, bench: _Bench) -> None:
    """Refuse a source_mlir section the run could not carry out (task P17.41; lassi.core.stages lowered_source).

    In this order: a fragment set, whose prompts (upstream's) show no MLIR;
    a toolchain that is not registered or does not declare LOWERS_SOURCE; a
    template set lacking SOURCE_MLIR_TEMPLATES or whose generate.txt never
    shows $source_mlir, so the module would be lowered and never shown; and
    a bench item without a compute unit (its manifest's `compute`). The
    toolchain itself is built by _frontend. A recipe without the section is
    never checked.
    """
    section = recipe.data.get(SOURCE_MLIR)
    if section is None:
        return
    where = f"{recipe.path}: {SOURCE_MLIR}"
    if settings.fragments:
        raise RunError(f"{where} is set, but {settings.prompts!r} is a fragment prompt set, upstream's prompts, "
                       f"which show no MLIR; use a template set or remove {SOURCE_MLIR}")
    try:
        entry = registry.get("Toolchain", section["toolchain"])
    except RegistryError as error:
        raise RunError(f"{where}.toolchain: {error}") from error
    if LOWERS_SOURCE not in entry.capabilities:
        raise RunError(f"{where}.toolchain {entry.name!r} does not declare {LOWERS_SOURCE!r}; name a Toolchain that "
                       "lowers a compute unit to MLIR, such as cgeist-affine")
    _check_templates(recipe, settings, f"{SOURCE_MLIR} is set", SOURCE_MLIR_TEMPLATES)
    fields = {**dict.fromkeys(GENERATE_FIELDS, ""), "source_mlir": _SOURCE_MLIR_PROBE}
    if _SOURCE_MLIR_PROBE not in render(settings.prompts, "generate", fields):
        raise RunError(f"{where} is set, but generate.txt of {settings.prompts!r} never shows $source_mlir, so the "
                       "lowered module would never reach the model; add the field or remove the section")
    missing = [item for item in bench.items if bench.suite.items[item].compute is None]
    if missing:
        raise RunError(f"{where} is set, but {bench.suite.name} declares no compute unit (the item key `compute`) for "
                       f"{', '.join(missing)}; add one per item or remove {SOURCE_MLIR}")


def _frontend(
    recipe: Recipe, registry: Registry, root: Path | None, build_root: Path | None
) -> BuiltToolchain | None:
    """Build the source_mlir section's lowering toolchain once per run with build_toolchain; None without it.

    It is built as every toolchain is (its pin, the compile sandbox, and the
    --version check, for a pinned class), with no language: its pins join
    every trial's (_run_pins), and toolchains.json records it.
    """
    section = recipe.data.get(SOURCE_MLIR)
    if section is None:
        return None
    return build_toolchain(section["toolchain"], root, registry, build_root=build_root)


def _check_stages(recipe: Recipe, registry: Registry, settings: _Settings) -> None:
    """Refuse a stage order that cannot run, a target language with no toolchain, and a prompt a stage cannot render.

    A fix turned off needs a listed stage that reproduces its quirk
    (_check_fixes). A trial gets one first attempt, so at most one stage
    declares `generates`, and a stage that declares `compiles` builds that
    attempt, so it comes after it. A stage that declares `builds_references`
    runs before any model call, so no stage that asks the model
    (_asks_model) comes before it. The prompts are checked by _check_prompts,
    loop.correct_on_mismatch by _check_mismatch, a template set's context by
    _check_template_context, the retrieval section's prompts by
    _check_retrieval_prompts, the lookup section by _check_lookup,
    loop.restart_after by _check_restart, and loop.correct_on_sim_gap by
    _check_sim_gap.
    """
    entries = [registry.get("Stage", name) for name in recipe.data["stages"]]
    _check_fixes(recipe, entries)
    generated = False
    asked: str | None = None
    for index, entry in enumerate(entries):
        where = f"{recipe.path}: stages[{index}] {entry.name!r}"
        if "builds_references" in entry.capabilities and asked is not None:
            raise RunError(
                f"{where} builds the reference programs before any model call, but {asked} asks the model; "
                "list it before every stage that asks the model"
            )
        if _asks_model(entry) and asked is None:
            asked = f"stages[{index}] {entry.name!r}"
        if "compiles" in entry.capabilities and not generated:
            raise RunError(f"{where} builds the attempt a generating stage appends; list such a stage before it")
        if "generates" in entry.capabilities:
            if generated:
                raise RunError(f"{where} is a second generating stage; a trial has one first attempt, so list one")
            generated = True
    bound = recipe.data.get("toolchain", {})
    if any("Toolchain" in entry.requires for entry in entries):
        for direction in settings.directions:
            if direction.target not in bound:
                raise RunError(
                    f"{recipe.path}: no toolchain is bound for the target language {direction.target!r}; "
                    f"set toolchain.{direction.target}"
                )
    _check_source_toolchains(recipe, entries, settings)
    _check_prompts(recipe, entries, settings)
    _check_mismatch(recipe, registry, entries, settings)
    _check_template_context(recipe, entries, settings)
    _check_retrieval_prompts(recipe, settings)
    _check_lookup(recipe, settings)
    _check_examples(recipe, settings)
    _check_skeleton(recipe, settings)
    _check_restart(recipe, settings)
    _check_sim_gap(recipe, settings)
    _check_coord_gap(recipe, settings)
    _check_dialect_lookup(recipe, settings)


def _check_dialect_lookup(recipe: Recipe, settings: _Settings) -> None:
    """Refuse the dialect_lookup section with a fragment set, or a template set that cannot render its block (P17.41).

    Upstream's prompts carry no definitions, so a fragment set cannot carry
    them; a template set needs each template of DIALECT_TEMPLATES
    (lassi.core.stages: the block, one definition, and one unknown name)
    rendering with its fields. The tree itself is read by _dialect_defs. A
    recipe without the section is never checked.
    """
    if DIALECT_LOOKUP not in recipe.data:
        return
    if settings.fragments:
        raise RunError(
            f"{recipe.path}: {DIALECT_LOOKUP} is set, but {settings.prompts!r} is a fragment prompt set, upstream's "
            f"prompts, which carry no dialect definitions; use a template set or remove {DIALECT_LOOKUP}"
        )
    _check_templates(recipe, settings, f"{DIALECT_LOOKUP} is set", DIALECT_TEMPLATES)


def _dialect_defs(recipe: Recipe) -> DialectDefs | None:
    """Load the definitions tree the recipe's dialect_lookup section names, once per run; None without the section.

    The tree is <assets root>/context/<dialect_lookup.defs>
    (lassi.prompts.dialect_defs load_defs), read only when every file
    matches its MANIFEST.yaml; a missing or bad tree is a RunError naming
    dialect_lookup.defs, raised before any directory exists.
    """
    section = recipe.data.get(DIALECT_LOOKUP)
    if section is None:
        return None
    try:
        return load_defs(section["defs"])
    except (OSError, ValueError) as error:
        raise RunError(
            f"{recipe.path}: {DIALECT_LOOKUP}.defs {section['defs']!r}: {error}; build it with "
            "uv run tools/build_ttkernel_context.py build --tree <tt-mlir at the pin>"
        ) from error


def _check_coord_gap(recipe: Recipe, settings: _Settings) -> None:
    """Refuse loop.coord_gap_as_error set true with a fragment prompt set (task P17.41; lassi.core.stages).

    Its correction carries this project's wording (addressing_text) in
    correct.txt, which only a template set holds. A key that is false or
    absent is never checked.
    """
    key = f"loop.{COORD_GAP_AS_ERROR}"
    if recipe.data["loop"].get(COORD_GAP_AS_ERROR) is True and settings.fragments:
        raise RunError(
            f"{recipe.path}: {key} is true, but {settings.prompts!r} is a fragment prompt set, upstream's prompts, "
            f"which hold no addressing-error correction; use a template set or remove {key}"
        )


def _check_sim_gap(recipe: Recipe, settings: _Settings) -> None:
    """Refuse loop.correct_on_sim_gap set true with a fragment prompt set (OQ-098; lassi.core.stages, simulator gaps).

    A gap's correction carries this project's wording (sim_gap_text) in
    correct.txt, which only a template set holds; a fragment set holds
    upstream's prompts, whose loop runs on no simulator. A key that is false
    or absent is never checked.
    """
    key = f"loop.{CORRECT_ON_SIM_GAP}"
    if recipe.data["loop"].get(CORRECT_ON_SIM_GAP) is True and settings.fragments:
        raise RunError(
            f"{recipe.path}: {key} is true, but {settings.prompts!r} is a fragment prompt set, upstream's prompts, "
            f"which hold no simulator-gap correction; use a template set or remove {key}"
        )


def _check_restart(recipe: Recipe, settings: _Settings) -> None:
    """Refuse loop.restart_after, or loop.restart_on_repeat set true, with a fragment prompt set (tasks P17.25, P17.50).

    A restart sends attempt 0's prompt again, generate.txt as a template
    set renders it; a fragment set holds upstream's prompts, whose loop
    never restarts. A recipe without either key is never checked.
    """
    loop = recipe.data["loop"]
    key = f"loop.{RESTART_AFTER}" if RESTART_AFTER in loop else f"loop.{RESTART_ON_REPEAT}"
    if (RESTART_AFTER in loop or loop.get(RESTART_ON_REPEAT) is True) and settings.fragments:
        raise RunError(
            f"{recipe.path}: {key} is set, but {settings.prompts!r} is a fragment prompt set, upstream's prompts, "
            f"whose loop never restarts; a restart sends generate.txt, which only a template set holds, so use a "
            f"template set or remove {key}"
        )


def _check_template_context(recipe: Recipe, entries: Sequence[Any], settings: _Settings) -> None:
    """Refuse context packs, or a stage that summarizes one, that a template prompt set cannot serve (task P17.21).

    A fragment set is checked by _check_prompts. Under a template set, a
    listed stage that declares `needs_context` needs a pack for every target
    language, and when the recipe names packs or lists such a stage, each
    template of CONTEXT_PROMPTS (lassi.core.stages: the summary request and
    the knowledge block) must render with its fields.
    """
    if settings.fragments:
        return
    needing = [entry for entry in entries if getattr(entry.factory, "needs_context", False)]
    for entry in needing:
        for direction in settings.directions:
            _check_pack(f"{recipe.path}: stage {entry.name!r}", entry, settings, direction)
    if settings.packs or needing:
        why = "context names packs" if settings.packs else f"stage {needing[0].name!r} summarizes a context pack"
        _check_templates(recipe, settings, why, CONTEXT_PROMPTS)


def _check_retrieval_prompts(recipe: Recipe, settings: _Settings) -> None:
    """Refuse the retrieval section with a fragment set, or a template set that cannot render its blocks (P17.22).

    Upstream's prompts hold no passages, so a fragment set cannot carry
    them; a template set needs each template of RETRIEVAL_PROMPTS
    (lassi.core.stages: the retrieved block and one passage) rendering with
    its fields. A recipe without the section is never checked.
    """
    if RETRIEVAL not in recipe.data:
        return
    if settings.fragments:
        raise RunError(
            f"{recipe.path}: {RETRIEVAL} is set, but {settings.prompts!r} is a fragment prompt set, upstream's "
            f"prompts, which carry no documentation passages; use a template set or remove {RETRIEVAL}"
        )
    _check_templates(recipe, settings, f"{RETRIEVAL} is set", RETRIEVAL_PROMPTS)


def _check_lookup(recipe: Recipe, settings: _Settings) -> None:
    """Refuse the lookup section when the run could not carry it out (task P17.26; lassi.core.stages).

    In this order: a fragment set, whose prompts (upstream's) carry no
    declarations; a direction whose target language no context pack of the
    recipe serves, since the declarations come from that pack; a
    lookup.core header that pack has no header part for (pack_parts); and
    a template set in which a template of LOOKUP_TEMPLATES (the looked-up
    block and one declaration), or with lookup.nearest set the template of
    NEAREST_TEMPLATES (one undeclared line), or with lookup.prompts listing
    generate the template of CORE_TEMPLATES (the generation prompt's core
    block), does not render with its fields. A recipe without the section is
    never checked.
    """
    if LOOKUP not in recipe.data:
        return
    if settings.fragments:
        raise RunError(
            f"{recipe.path}: {LOOKUP} is set, but {settings.prompts!r} is a fragment prompt set, upstream's "
            f"prompts, which carry no declarations; use a template set or remove {LOOKUP}"
        )
    for direction in settings.directions:
        if direction.target not in settings.packs:
            served = ", ".join(sorted(settings.packs)) or "none"
            raise RunError(
                f"{recipe.path}: {LOOKUP} is set, but no context pack serves the target language "
                f"{direction.target!r}, whose declarations it would carry; the recipe's context serves: {served}; "
                f"name a pack for {direction.target!r} in context or remove {LOOKUP}"
            )
        _check_core_parts(recipe, settings.packs[direction.target], direction.target)
    _check_templates(recipe, settings, f"{LOOKUP} is set", LOOKUP_TEMPLATES)
    if LOOKUP_NEAREST in recipe.data[LOOKUP]:
        _check_templates(recipe, settings, f"{LOOKUP}.{LOOKUP_NEAREST} is set", NEAREST_TEMPLATES)
    if LOOKUP_GENERATE in recipe.data[LOOKUP]["prompts"]:
        _check_templates(recipe, settings, f"{LOOKUP}.prompts lists {LOOKUP_GENERATE}", CORE_TEMPLATES)


def _check_core_parts(recipe: Recipe, pack: str, language: str) -> None:
    """Refuse a lookup.core header that the pack serving `language` has no header part for (stages.pack_parts)."""
    parts = pack_parts(pack)
    missing = [header for header in recipe.data[LOOKUP].get(LOOKUP_CORE, ()) if header not in parts]
    if missing:
        raise RunError(
            f"{recipe.path}: {LOOKUP}.{LOOKUP_CORE} names {', '.join(missing)}, for which the context pack serving "
            f"{language!r} has no header part; its parts are: {', '.join(parts) or 'none'}"
        )


def _check_examples(recipe: Recipe, settings: _Settings) -> None:
    """Refuse the examples section with a fragment set, or a template set that cannot render its block (OQ-083 (b)).

    Upstream's prompts hold no worked examples, so a fragment set cannot
    carry them; a template set needs each template of EXAMPLE_TEMPLATES
    (lassi.core.stages: the examples block and one example) rendering with
    its fields. The pack itself is checked by _examples_pack. A recipe
    without the section is never checked.
    """
    if EXAMPLES not in recipe.data:
        return
    if settings.fragments:
        raise RunError(
            f"{recipe.path}: {EXAMPLES} is set, but {settings.prompts!r} is a fragment prompt set, upstream's "
            f"prompts, which carry no worked examples; use a template set or remove {EXAMPLES}"
        )
    _check_templates(recipe, settings, f"{EXAMPLES} is set", EXAMPLE_TEMPLATES)


def _check_skeleton(recipe: Recipe, settings: _Settings) -> None:
    """Refuse the skeleton section with a fragment prompt set (OQ-096 (a); lassi.core.stages, the skeleton block).

    Upstream's prompts hold no skeleton, so a fragment set cannot carry it.
    The block's template is the skeleton's own block.txt, which the loader
    rendered, so a template set needs no template for it. A recipe without
    the section is never checked.
    """
    if SKELETON in recipe.data and settings.fragments:
        raise RunError(
            f"{recipe.path}: {SKELETON} is set, but {settings.prompts!r} is a fragment prompt set, upstream's "
            f"prompts, which carry no host program skeleton; use a template set or remove {SKELETON}"
        )


def _check_templates(
    recipe: Recipe, settings: _Settings, why: str, templates: Mapping[str, Sequence[str]]
) -> None:
    """Refuse a template of `templates` (name -> fields) that the template set lacks or cannot render with its fields.

    Each is rendered once with every field set to "", which finds a missing
    file and a placeholder no field fills; `why` says why the run needs it.
    """
    for prompt, fields in templates.items():
        try:
            render(settings.prompts, prompt, dict.fromkeys(fields, ""))
        except ValueError as error:
            raise RunError(
                f"{recipe.path}: {why}, so the template prompt set {settings.prompts!r} needs {prompt}.txt, "
                f"rendering with the fields {', '.join(fields)}: {error}"
            ) from error


def _check_mismatch(recipe: Recipe, registry: Registry, entries: Sequence[Any], settings: _Settings) -> None:
    """Refuse loop.correct_on_mismatch set true when the run could not carry it out (lassi.core.stages).

    It needs a listed stage that declares `mismatch_prompt_fields` (run_loop),
    an Oracle that declares aligns_output_files, whose per-output statistics
    the correction prompt's coarse report reads, and a template prompt set
    in which each such template renders with empty fields; a fragment set
    holds upstream's prompts, which have no such correction. A key that is
    false or absent is never checked.
    """
    key = f"loop.{CORRECT_ON_MISMATCH}"
    if recipe.data["loop"].get(CORRECT_ON_MISMATCH) is not True:
        return
    templates = [getattr(entry.factory, "mismatch_prompt_fields", None) for entry in entries]
    templates = [fields for fields in templates if fields]
    if not templates:
        raise RunError(f"{recipe.path}: {key} is true, but no listed stage carries it out; list run_loop or remove it")
    oracles = [binding for binding in recipe.bindings if binding.interface == "Oracle"]
    if not any(ALIGNS_OUTPUT_FILES in registry.get("Oracle", binding.name).capabilities for binding in oracles):
        raise RunError(
            f"{recipe.path}: {key} is true, but the recipe binds no Oracle that declares {ALIGNS_OUTPUT_FILES!r}, "
            f"whose per-output statistics the correction prompt's coarse report reads; bind one or remove {key}"
        )
    if settings.fragments:
        raise RunError(
            f"{recipe.path}: {key} is true, but {settings.prompts!r} is a fragment prompt set, upstream's prompts, "
            f"which hold no oracle-mismatch correction; use a template set or remove {key}"
        )
    for fields in templates:
        for prompt, names in fields.items():
            try:
                render(settings.prompts, prompt, dict.fromkeys(names, ""))
            except ValueError as error:
                message = f"{recipe.path}: {key} is true, but its correction prompt cannot render: {error}"
                raise RunError(message) from error


def _asks_model(entry: Any) -> bool:
    """Return True for a stage that asks the model: it requires an LLMBackend, or it generates or corrects attempts."""
    return "LLMBackend" in entry.requires or bool({"generates", "compiles"} & entry.capabilities)


def _check_source_toolchains(recipe: Recipe, entries: Sequence[Any], settings: _Settings) -> None:
    """Refuse a stage that builds the source reference too, under a fix that is on, when no toolchain builds it.

    A stage class names that fix in `source_build_fix` (baseline names
    baseline_both); with the fix on, every direction's source language needs
    a bound toolchain.
    """
    bound = recipe.data.get("toolchain", {})
    for entry in entries:
        fix = getattr(entry.factory, "source_build_fix", None)
        if fix is None or recipe.data["fixes"].get(fix, True) is False:
            continue
        for direction in settings.directions:
            if direction.source not in bound:
                raise RunError(
                    f"{recipe.path}: stage {entry.name!r} builds the source reference too while fixes.{fix} is on, "
                    f"but no toolchain is bound for the source language {direction.source!r}; "
                    f"set toolchain.{direction.source}, or turn fixes.{fix} off"
                )


def _check_fixes(recipe: Recipe, entries: Sequence[Any]) -> None:
    """Refuse a fix turned off when no listed stage names it in `reproduces`, so no quirk is asked for in vain."""
    data = recipe.data
    reproduced = {name for entry in entries for name in getattr(entry.factory, "reproduces", ())}
    for name, on in sorted(data["fixes"].items()):
        if not on and name not in reproduced:
            raise RunError(
                f"{recipe.path}: fixes.{name} is off (faithful: {fmt(data['faithful'])}), which asks for an upstream "
                "quirk that no listed stage reproduces; list a stage that reproduces it, or turn the fix on and "
                "leave faithful off"
            )


def _check_prompts(recipe: Recipe, entries: Sequence[Any], settings: _Settings) -> None:
    """Refuse a prompt a stage cannot build from the recipe's prompt set and context packs.

    With a template set, each stage's `prompt_fields` templates are rendered
    once with empty fields, which finds a missing set or file and a
    placeholder the stage does not fill; a stage that reads only fragments
    is refused (its context packs are checked by _check_template_context).
    With a fragment set, a stage that renders templates but
    declares no `fragment_keys` is refused, every key a stage declares must
    be in the set for every direction, a stage that declares
    `needs_context` needs a pack for every target language, and each
    Trial.context field a stage names in `joins_context` must be named in
    `fills_context` by an earlier stage whenever a pack serves a target
    (_check_context_order).
    """
    filled: set[str] = set()
    for entry in entries:
        templates = getattr(entry.factory, "prompt_fields", {})
        keys = getattr(entry.factory, "fragment_keys", None)
        where = f"{recipe.path}: stage {entry.name!r}"
        if not settings.fragments:
            if keys and not templates:
                raise RunError(f"{where} reads prompt fragments, but {settings.prompts!r} is a template prompt set")
            for prompt, fields in templates.items():
                try:
                    render(settings.prompts, prompt, dict.fromkeys(fields, ""))
                except ValueError as error:
                    raise RunError(f"{where} cannot use the prompt set: {error}") from error
            continue
        _check_context_order(where, entry, settings, filled)
        filled.update(getattr(entry.factory, "fills_context", ()))
        if keys is None:
            if templates:
                raise RunError(f"{where} renders templates, but {settings.prompts!r} is a fragment prompt set")
            continue
        for direction in settings.directions:
            missing = [key for key in (fragment_key(template, direction) for template in keys)
                       if key not in settings.fragments]
            if missing:
                raise RunError(f"{where}: the prompt set {settings.prompts!r} has no fragment {', '.join(missing)}")
            _check_pack(where, entry, settings, direction)


def _check_pack(where: str, entry: Any, settings: _Settings, direction: Direction) -> None:
    """Refuse a stage that declares `needs_context` when no context pack of the recipe serves the target language."""
    if getattr(entry.factory, "needs_context", False) and direction.target not in settings.packs:
        served = ", ".join(sorted(settings.packs)) or "none"
        raise RunError(
            f"{where} needs a context pack for the target language {direction.target!r}; "
            f"the recipe's context serves: {served}"
        )


def _check_context_order(where: str, entry: Any, settings: _Settings, filled: set[str]) -> None:
    """Refuse a stage that joins a Trial.context field into its prompt when no earlier stage fills that field.

    A fragment set's generation prompt joins the summary and the description
    whenever a pack serves the target; with a field left empty the model
    would get empty markers, a prompt upstream never sends.
    """
    missing = [name for name in getattr(entry.factory, "joins_context", ()) if name not in filled]
    served = [direction.target for direction in settings.directions if direction.target in settings.packs]
    if missing and served:
        raise RunError(
            f"{where} joins the Trial.context field(s) {', '.join(missing)} into its prompt when a context pack "
            f"serves {served[0]!r}, but no earlier listed stage fills them; list the stages that fill them first"
        )


def _check_trials(recipe: Recipe, settings: _Settings, bench: _Bench) -> None:
    """Refuse trial ids that are invalid or repeat (in any letter case), and bench sources that cannot be read.

    Each item's file counts are checked against the prompt set and the
    fence_tag fix by _check_one_file.
    """
    seen: dict[str, str] = {}
    for direction in settings.directions:
        for item in bench.items:
            try:
                arm = arm_segment(settings.model_id)
                trial_id = make_trial_id(recipe.data["project"], arm, bench.suite.name, direction.name, item, 1)
            except ValueError as error:
                raise RunError(f"{recipe.path}: {error}") from error
            key = trial_id.casefold()
            if key in seen:
                raise RunError(
                    f"{recipe.path}: the trials {seen[key]} and {trial_id} would share one trial directory; "
                    "list each direction once"
                )
            seen[key] = trial_id
            try:
                sources = bench.suite.source_files(item, direction, bench.root, purpose=PURPOSE)
                bench.suite.reference_target(item, direction, bench.root, purpose=PURPOSE)
                program = bench.suite.reference_language(item, direction.target)
                for language in dict.fromkeys((direction.target, direction.source, program)):
                    bench.suite.support_files(item, bench.root, purpose=PURPOSE, language=language)
            except (OSError, ValueError) as error:
                raise RunError(
                    f"cannot read {bench.suite.name}/{item} ({direction.name}) under {bench.root}: {error}; "
                    + _fetch_hint(bench.suite)
                ) from error
            _check_one_file(recipe, settings, bench, item, direction, len(sources))
            _check_reference_bindings(recipe, bench, item, direction)


def _check_reference_bindings(recipe: Recipe, bench: _Bench, item: str, direction: Direction) -> None:
    """Refuse a target whose reference is another language's program when that language has no toolchain or executor.

    The baseline builds such a reference with the reference language's
    toolchain and runs it on that language's executor (lassi.bench
    Suite.reference_language; task P17.36), so with executors bound per
    language both must be bound for it, each message naming the key. A
    reference that is the direction's own source language is refused too
    while fixes.baseline_both is on: the baseline would build that one
    program as both references and compare it with itself. With the fix
    off the baseline builds it once, as the target's reference (task
    P17.43). The caller has already read the reference language's
    support files.
    """
    program = bench.suite.reference_language(item, direction.target)
    if program == direction.target:
        return
    where = f"{recipe.path}: {bench.suite.name}/{item} ({direction.name})"
    if program == direction.source and recipe.data.get("fixes", {}).get("baseline_both", True) is not False:
        raise RunError(
            f"{where}: the {direction.target} reference is the item's {program} program, the direction's own source, "
            "so the baseline would build one program twice and compare it with itself; translate from another "
            "language, or turn fixes.baseline_both off so the baseline builds it once"
        )
    if program not in recipe.data.get("toolchain", {}):
        raise RunError(
            f"{where}: the {direction.target} reference is the item's {program} program, which the baseline builds "
            f"with the {program} toolchain; set toolchain.{program}"
        )
    executors = recipe.data.get("executor", {})
    if isinstance(executors, dict) and "kind" not in executors and program not in executors:
        raise RunError(
            f"{where}: the {direction.target} reference is the item's {program} program, which the baseline runs "
            f"on the {program} executor; set executor.{program}"
        )


def _check_one_file(
    recipe: Recipe, settings: _Settings, bench: _Bench, item: str, direction: Direction, source_count: int
) -> None:
    """Refuse an item a fragment set or the fence_tag quirk cannot take, before any model is asked.

    A fragment set joins one source text into its prompts
    (lassi.core.stages.source_as_read), and with fixes.fence_tag off the
    first fenced block of a reply is the one target file.
    """
    where = f"{recipe.path}: {bench.suite.name}/{item} ({direction.name})"
    if settings.fragments and source_count != 1:
        raise RunError(
            f"{where} has {source_count} {direction.source!r} source files; a fragment prompt set joins one source"
        )
    targets = bench.suite.item(item, purpose=PURPOSE).languages[direction.target].files
    if recipe.data["fixes"].get("fence_tag", True) is False and len(targets) != 1:
        raise RunError(
            f"{where} has {len(targets)} {direction.target!r} files; with fixes.fence_tag off the first fenced "
            "block of a reply is the one target file"
        )


def _check_oracle(recipe: Recipe, registry: Registry, bench: _Bench) -> None:
    """Refuse an oracle section no listed stage uses or accepts, one its Oracle refuses, or one the run cannot serve.

    A listed stage uses an Oracle when its registry entry names Oracle in
    `requires` (the load already refuses such a stage when no oracle is
    bound). A stage class that names `oracle_capabilities` (the oracle stage)
    needs its Oracle to declare one of them. Each bound Oracle is built once
    as factory(**config) and dropped, so a choice its section leaves unset,
    such as oracle.passfail or oracle.threshold, fails here, before any
    directory is created, and not when the first trial builds its stages.
    An Oracle whose `agreement_setting` is set (binary_io's threshold
    from_baseline) is checked by _check_agreement.
    """
    bindings = [binding for binding in recipe.bindings if binding.interface == "Oracle"]
    if not bindings:
        return
    stages = [registry.get("Stage", name) for name in recipe.data["stages"]]
    if not any("Oracle" in entry.requires for entry in stages):
        raise RunError(
            f"{recipe.path}: sets oracle, but no listed stage uses an Oracle; "
            "list the oracle stage or remove the section"
        )
    for binding in bindings:
        entry = registry.get("Oracle", binding.name)
        for stage in stages:
            accepted = tuple(getattr(stage.factory, "oracle_capabilities", ()))
            if accepted and not set(accepted) & entry.capabilities:
                declared = ", ".join(sorted(entry.capabilities)) or "nothing"
                raise RunError(
                    f"{recipe.path}: {binding.where}: stage {stage.name!r} needs an Oracle that declares one of "
                    f"{', '.join(accepted)}, but Oracle {binding.name!r} declares: {declared}"
                )
        try:
            oracle = entry.factory(**binding.config)
        except ValueError as error:
            raise RunError(f"{recipe.path}: {binding.where}: {error}") from error
        setting = getattr(oracle, "agreement_setting", None)
        if setting is not None:
            _check_agreement(recipe, stages, bench, setting)


def _check_adversary(recipe: Recipe, registry: Registry, settings: _Settings, bench: _Bench) -> None:
    """Refuse, before any directory exists, an adversary section or stage the run cannot carry out (task P16.5).

    A stage that declares RUNS_PROPOSALS (the adversary stage) runs the
    proposals of the Agent bound at adversary.kind, so each needs the
    other: the section without such a stage and the stage without the
    section are refused, and so is the stage listed twice or before a stage
    that aligns (the oracle stage; _check_proposal_stage). Then an
    adversary.budget.inputs that is not an int of at least 1, an item of the
    run with no held-out inputs (it has no input contract), a target
    language's executor that runs no program or runs programs without
    declaring sandboxed (_check_proposal_executors), and an Agent that
    refuses its config or has no propose() (_check_agent). The stage's
    need for an Oracle that declares aligns_output_files is checked at load
    (its `requires`).
    """
    stages = [registry.get("Stage", name) for name in recipe.data["stages"]]
    listed = [index for index, entry in enumerate(stages) if RUNS_PROPOSALS in entry.capabilities]
    binding = adversary_binding(recipe.bindings)
    if binding is None and not listed:
        return
    if binding is None:
        raise RunError(
            f"{recipe.path}: stages[{listed[0]}] {stages[listed[0]].name!r} runs an adversary's proposals, but the "
            "recipe has no adversary section; set adversary: {kind: <Agent>, budget: {inputs: <N>}}, with the "
            "Agent's own keys, or remove the stage"
        )
    if not listed:
        raise RunError(
            f"{recipe.path}: sets adversary (Agent {binding.name!r} at {ADVERSARY_KIND}), but no listed stage runs "
            "its proposals; list the adversary stage after the oracle stage, or remove the section"
        )
    name = stages[listed[0]].name
    _check_proposal_stage(recipe, stages, listed)
    try:
        budget_inputs(binding.config)
    except ValueError as error:
        raise RunError(f"{recipe.path}: {error}") from None
    missing = [item for item in bench.items if bench.suite.items[item].contract is None]
    if missing:
        raise RunError(
            f"{recipe.path}: stage {name!r} checks every proposal against its item's input contract, which only an "
            f"item with held-out inputs has, but {bench.suite.name} item(s) {', '.join(missing)} declare no held-out "
            "inputs; select items that do with bench.items"
        )
    _check_proposal_executors(recipe, registry, settings, name)
    _check_agent(recipe, registry, binding)


def _check_proposal_stage(recipe: Recipe, stages: Sequence[Entry], listed: Sequence[int]) -> None:
    """Refuse a stage that runs proposals listed twice, or listed before every stage that aligns attempts."""
    first = listed[0]
    where = f"{recipe.path}: stages[{first}] {stages[first].name!r}"
    if len(listed) > 1:
        raise RunError(f"{where} is listed {len(listed)} times; a trial's adversary record is one, so list it once")
    if not any(_ALIGNS in entry.capabilities for entry in stages[:first]):
        raise RunError(
            f"{where} takes a trial whose standing attempt passed the oracle on the held-out inputs, which a stage "
            f"that declares {_ALIGNS!r} records; list the oracle stage before it"
        )


def _check_proposal_executors(recipe: Recipe, registry: Registry, settings: _Settings, stage: str) -> None:
    """Refuse a target language's executor that runs no program, or runs programs without declaring sandboxed.

    Every accepted proposal runs the target reference and the candidate,
    model-generated code, on the target language's executor (Agent Rule 6),
    whether or not run_loop is listed. With executors per language, a target
    language with no executor is left to _executors, which names
    executor.<language>.
    """
    bindings = [binding for binding in recipe.bindings if binding.interface == "Executor"]
    languages = executor_languages(recipe.data)
    by_language = dict(zip(languages, bindings, strict=True)) if languages else {}
    for direction in settings.directions:
        binding = by_language.get(direction.target) if languages else (bindings[0] if bindings else None)
        if binding is None:
            continue
        capabilities = registry.get("Executor", binding.name).capabilities
        named = f"Executor {binding.name!r} ({binding.where}), the {direction.target!r} executor,"
        if _COMPILE_ONLY in capabilities or RUNS_CODE not in capabilities:
            why = f"declares {_COMPILE_ONLY!r}" if _COMPILE_ONLY in capabilities else f"does not declare {RUNS_CODE!r}"
            raise RunError(
                f"{recipe.path}: stage {stage!r} runs the target reference and the candidate on each accepted "
                f"proposal, but {named} {why}, so it runs no program; bind an executor that runs programs, or "
                "remove the stage and the adversary section"
            )
        if SANDBOXED not in capabilities:
            raise RunError(
                f"{recipe.path}: stage {stage!r} runs the candidate, model-generated code, on each accepted proposal, "
                f"but {named} runs programs without declaring {SANDBOXED!r}; model-generated code runs only in the "
                "sandbox (Agent Rule 6)"
            )


def _check_agent(recipe: Recipe, registry: Registry, binding: Binding) -> None:
    """Build the adversary once as factory(**config) and drop it; RunError for a refused config or no propose()."""
    entry = registry.get(binding.interface, binding.name)
    try:
        agent = entry.factory(**binding.config)
    except ValueError as error:
        where = f"{recipe.path}: {binding.where}"
        raise RunError(f"{where}: Agent {binding.name!r} refused its settings: {error}") from None
    if not callable(getattr(agent, "propose", None)):
        raise RunError(
            f"{recipe.path}: {binding.where}: Agent {binding.name!r} declares proposes_inputs but has no propose(view)"
        )


def _check_agreement(recipe: Recipe, stages: Sequence[Any], bench: _Bench, setting: str) -> None:
    """Refuse an oracle `setting` that reads the references' agreement when the run could never measure it.

    The baseline stage measures the source reference's agreement with the
    target reference only with fixes.baseline_both on and only for an item
    whose suite manifest declares a tolerance, so the fix must be on, a
    listed stage must build the references (`builds_references`), and every
    item the run covers must declare a tolerance. A direction whose source
    and target are one language builds one reference (the baseline stage),
    which the baseline compares with itself: a trivial agreement that
    would make from_baseline an exact match, not the references' own
    agreement, so it is refused too (task P16.8).
    """
    reads = f"{recipe.path}: {setting} reads the references' agreement"
    directions = recipe.data["directions"]
    same = [f"{entry['source']}-{entry['target']}" for entry in directions if entry["source"] == entry["target"]]
    if same:
        raise RunError(
            f"{reads}, which the baseline measures only between two languages; direction(s) {', '.join(same)} have "
            "one language, so the baseline builds one reference and compares it with itself, an agreement that "
            "would make from_baseline an exact match; set a number"
        )
    if recipe.data.get("fixes", {}).get("baseline_both", True) is False:
        raise RunError(
            f"{reads}, which the baseline measures only with fixes.baseline_both on (faithful: true turns it off); "
            "turn fixes.baseline_both on, or set a number"
        )
    if not any("builds_references" in stage.capabilities for stage in stages):
        raise RunError(f"{reads}, but no listed stage builds the reference programs; list the baseline stage")
    missing = [item for item in bench.items if bench.suite.items[item].tolerance is None]
    if missing:
        raise RunError(
            f"{reads}, which the baseline measures only for an item whose suite manifest declares a tolerance; "
            f"{bench.suite.name} item(s) {', '.join(missing)} declare none"
        )


# ---------------------------------------------------------------------------
# Toolchains and pins


def _toolchains(
    recipe: Recipe, registry: Registry, root: Path | None, build_root: Path | None = None
) -> tuple[BuiltToolchain, ...]:
    """Build each toolchain the recipe binds, once per registry name, with build_toolchain, and set its languages.

    `build_root` is the run's runs root, which every compile hides (see
    build_toolchain).
    """
    languages: dict[str, list[str]] = {}
    for language, name in sorted(recipe.data.get("toolchain", {}).items()):
        languages.setdefault(name, []).append(language)
    return tuple(
        dataclasses.replace(build_toolchain(name, root, registry, build_root=build_root), languages=tuple(bound))
        for name, bound in languages.items()
    )


def build_toolchain(
    name: str, root: Path | None, registry: Registry = DEFAULT_REGISTRY, *, build_root: Path | None = None
) -> BuiltToolchain:
    """Build the toolchain registered as `name` exactly as the stage runner builds it, with no languages.

    A class that declares PIN gets its pinned executable under the
    toolchains root `root` (None means none is set, which is refused), or,
    without PIN_BIN, the pin's EXECUTABLE (a host compiler, which must be
    an absolute path to an existing file; the root is still required, since
    the compile sandbox exposes it; with check_tree, also the installed tree
    it builds against, checked before any process starts), a
    clean compile environment, and the linked prefixes its pin names, run
    through SandboxedCompileRunner (see the module docstring), after its
    `--version` output was checked against the pin's EXPECT_VERSION through
    that same runner; a class without PIN is built as factory().
    `build_root`, when given, is the directory the build dirs lie under (a
    run passes its runs root): every compile hides it, apart from its own
    build dir, and the compile layout is checked against a build dir under
    it first. Raises RunError, saying what to install or set, when the pin
    file, the root, the executable (or it resolves outside the root), the
    installed tree (or check_tree refuses it), a
    linked prefix, TMPDIR (unset, or outside $LASSI_SCRATCH), or every
    hidden root is missing, when a hidden root is not absolute, when the
    sandbox refuses the compile layout, or when the compiler is not the
    pinned version; SandboxUnavailableError when the sandbox cannot run the
    --version check. The runner builds every bound toolchain through this,
    so a tool that calls it compiles with the same command, executable,
    environment, and sandbox as a run.
    """
    factory = registry.get("Toolchain", name).factory
    if getattr(factory, "PIN", None) is None:
        return BuiltToolchain(name, (), factory(), None, None, {})
    return _pinned_toolchain(name, factory, root, build_root)


def _pinned_toolchain(name: str, factory: type, root: Path | None, build_root: Path | None) -> BuiltToolchain:
    """Build a toolchain with its pinned executable and the sandboxed compile runner, checked by --version.

    A class with PIN_BIN finds its compiler under the toolchains root
    (_pinned_executable); a class with PIN and no PIN_BIN uses a host
    compiler the project does not install, the pin's EXECUTABLE
    (_host_executable), and one that also defines check_tree builds
    against the pin's installed tree (_checked_tree). Every refusal is a
    RunError that says what is missing, raised before any process starts:
    the pin file, the root, the executable, the tree (or check_tree refuses
    it), TMPDIR (unset, or outside $LASSI_SCRATCH), a hidden root, a linked
    prefix, or a compile layout the sandbox refuses. Then
    `<executable> --version`, with the class's VERSION_ARGS if it declares
    them (_version_argv), must print the pin's EXPECT_VERSION in the
    compile sandbox. A class that declares TOOL_PINS also gets the tools of
    those pins, each checked (_tool_pins), as factory(..., tools=...), and
    their pins join the toolchain's (task P17.36).
    """
    pin_name = factory.PIN
    host = getattr(factory, "PIN_BIN", None) is None
    check_tree = getattr(factory, "check_tree", None) if host else None
    pin = _pin(pin_name, ("VERSION",) if host and check_tree is None else ("VERSION", "PREFIX_NAME"))
    if host:
        resolved, executable = _host_executable(name, pin_name, pin, root)
    else:
        resolved, executable = _pinned_executable(name, factory, pin, root)
    keywords = {} if check_tree is None else {"tree": _checked_tree(name, pin_name, pin, resolved, check_tree)}
    tool_pins, tools = _tool_pins(name, factory, resolved)
    tmpdir = _checked_tmpdir()
    hidden_roots = _compile_hidden_roots(build_root)
    environment = _compile_environment()
    pins = {pin_name: pin, **_linked_pins(name, pin_name, pin, resolved, environment)}
    runner = _compile_runner(name, environment, resolved, hidden_roots, build_root)
    version, status = _checked_version(name, _version_argv(factory, executable), runner, Path(tmpdir), pin_name, pin)
    for tool_pin, tool in tools.values():
        _checked_version(name, [tool, "--version"], runner, Path(tmpdir), tool_pin, tool_pins[tool_pin])
    if tool_pins:
        keywords["tools"] = {key: tool for key, (_, tool) in tools.items()}
        pins.update(tool_pins)
    toolchain = factory(executable=str(executable), runner=runner, **keywords)
    return BuiltToolchain(name, (), toolchain, str(executable), environment, pins, version, status)


def _tool_pins(
    name: str, factory: type, root: Path
) -> tuple[dict[str, dict[str, str]], dict[str, tuple[str, str]]]:
    """Return the pins a class's TOOL_PINS names and, by pin key, each tool's pin and path (task P17.36).

    TOOL_PINS maps a pin name to the keys of that pin whose values are tool
    paths relative to its install, <root>/<PREFIX_NAME> (ttkernel-ttmetal:
    tt-mlir's TTMLIR_OPT and TTMLIR_TRANSLATE). The install must be a
    directory inside the resolved root that the class's check_tool_pin, when
    it defines one, accepts, and each tool a file resolving inside the root.
    These are filesystem checks, made before any process starts; the caller
    then runs each tool's `--version` through the compile runner, which
    must print that pin's EXPECT_VERSION (_checked_version). Every refusal
    is a RunError; a class without TOOL_PINS gets ({}, {}).
    """
    found: dict[str, dict[str, str]] = {}
    tools: dict[str, tuple[str, str]] = {}
    check = getattr(factory, "check_tool_pin", None)
    for tool_pin, keys in getattr(factory, "TOOL_PINS", {}).items():
        pin = _pin(tool_pin)
        home = root / pin["PREFIX_NAME"]
        install = f"toolchains/{tool_pin}.sh"
        if not home.is_dir() or not _within(home.resolve(), root):
            raise RunError(
                f"toolchain {name!r} needs the installed {tool_pin} at {home}, which is not a directory inside the "
                f"toolchains root; install it on the build host with {install}"
            )
        try:
            if check is not None:
                check(tool_pin, home, pin)
        except (OSError, ValueError) as error:
            raise RunError(f"toolchain {name!r}: {error}") from error
        for key in keys:
            tool = home / pin.get(key, "")
            if not pin.get(key) or not tool.is_file() or not _within(tool.resolve(), root):
                raise RunError(
                    f"toolchain {name!r}: the pinned tool {tool} ({key} in toolchains/{tool_pin}.pin) is not a file "
                    f"inside the toolchains root; install it on the build host with {install}"
                )
            tools[key] = (tool_pin, str(tool))
        found[tool_pin] = pin
    return found, tools


def _host_executable(name: str, pin_name: str, pin: Mapping[str, str], root: Path | None) -> tuple[Path, str]:
    """Return the resolved toolchains root and a host pin's EXECUTABLE, as the pin gives it.

    A host compiler (toolchains/gcc.pin) is not installed under the
    toolchains root, so its path is the pin's EXECUTABLE: it must be set, an
    absolute path on this host (Path.is_absolute, so the host's own path
    rules apply), and an existing file, or a RunError names it before any
    process starts. The compile sandbox still exposes the toolchains root
    read-only, as for every compile, so a root must be set. The --version
    check then proves the compiler is the pinned one and reachable in the
    compile's view.
    """
    if root is None:
        raise RunError(
            f"toolchain {name!r} uses the pinned host {pin_name} {pin['VERSION']} in the compile sandbox, which "
            "exposes the toolchains root read-only, but no toolchains root is set; set LASSI_TOOLCHAINS (the gate "
            "sets it on the build host)"
        )
    if not root.is_absolute():
        raise RunError(f"the toolchains root {root} must be an absolute path (LASSI_TOOLCHAINS or the option)")
    executable = pin.get("EXECUTABLE", "")
    if not executable:
        raise RunError(
            f"toolchains/{pin_name}.pin has no EXECUTABLE, the absolute path of the host compiler toolchain {name!r} "
            "runs; add it"
        )
    if not Path(executable).is_absolute():
        raise RunError(
            f"toolchain {name!r}: EXECUTABLE {executable!r} in toolchains/{pin_name}.pin is not an absolute path; "
            "a pinned compiler is never looked up on PATH (Agent Rule 10)"
        )
    if not Path(executable).is_file():
        raise RunError(
            f"toolchain {name!r}: the pinned host compiler {executable} (EXECUTABLE in toolchains/{pin_name}.pin) "
            "does not exist on this host; install the package it names, or run on the build host"
        )
    return root.resolve(), executable


def _checked_tree(
    name: str, pin_name: str, pin: Mapping[str, str], root: Path, check_tree: Callable[[Path, Mapping[str, str]], None]
) -> Path:
    """Return the installed tree a host compiler builds against, <root>/<PREFIX_NAME>, once check_tree accepts it.

    `root` is the resolved toolchains root. The tree must be a directory
    that resolves inside it, since a compile sees only that root, and
    `check_tree(tree, pin)`, the toolchain class's own check of its install
    (lassi.toolchains.ttmetal_build), must raise no ValueError. Every
    refusal is a RunError, raised before any process starts.
    """
    tree = root / pin["PREFIX_NAME"]
    if not tree.is_dir() or not _within(tree.resolve(), root):
        raise RunError(
            f"toolchain {name!r} builds against the installed {pin_name} tree {tree}, which is not a directory "
            f"inside the toolchains root; install it on the build host with toolchains/{pin_name}.sh"
        )
    try:
        check_tree(tree, pin)
    except (OSError, ValueError) as error:
        raise RunError(f"toolchain {name!r}: {error}") from error
    return tree


def _pinned_executable(name: str, factory: type, pin: Mapping[str, str], root: Path | None) -> tuple[Path, Path]:
    """Return the resolved toolchains root and the pinned executable under it; RunError says what is missing.

    The root is resolved once (symbolic links and `..` segments followed),
    and the executable, every linked prefix, and the sandbox's read-only
    toolchains root all come from that resolved root, which is what a
    compile sees; an unresolved path would not exist in the compile's view.
    The executable must also resolve inside the root, since nothing outside
    it is exposed to a compile.
    """
    pin_name = factory.PIN
    if root is None:
        raise RunError(
            f"toolchain {name!r} uses the pinned {pin_name} {pin['VERSION']}, but no toolchains root is set; "
            f"set LASSI_TOOLCHAINS (the gate sets it on the build host), where toolchains/{pin_name}.sh installs it"
        )
    if not root.is_absolute():
        raise RunError(f"the toolchains root {root} must be an absolute path (LASSI_TOOLCHAINS or the option)")
    try:
        relative = factory.PIN_BIN.format_map(pin)
    except (KeyError, ValueError) as error:
        message = f"toolchain {name!r}: PIN_BIN {factory.PIN_BIN!r} does not fit toolchains/{pin_name}.pin"
        raise RunError(f"{message} ({error!r})") from None
    resolved = root.resolve()
    executable = resolved / pin["PREFIX_NAME"] / relative
    if not executable.is_file():
        raise RunError(
            f"toolchain {name!r}: the pinned compiler {executable} does not exist; "
            f"install it on the build host with toolchains/{pin_name}.sh"
        )
    target = executable.resolve()
    if not _within(target, resolved):
        raise RunError(
            f"toolchain {name!r}: the pinned compiler {executable} resolves to {target}, outside the toolchains root "
            f"{resolved}, and a compile sees only that root; install it there with toolchains/{pin_name}.sh"
        )
    return resolved, executable


def _linked_pins(
    name: str, pin_name: str, pin: Mapping[str, str], root: Path, environment: dict[str, str]
) -> dict[str, dict[str, str]]:
    """Set each linked prefix the pin names in `environment` and return the linked pins, by pin name.

    `root` is the resolved toolchains root. Raises RunError when a linked
    prefix is not the installed prefix of its own pin under it, or resolves
    outside it, where a compile could not see it.
    """
    linked_pins: dict[str, dict[str, str]] = {}
    for variable, prefix in linked_prefixes(pin).items():
        try:
            linked_name = prefix_pin_name(prefix)
        except ValueError as error:
            raise RunError(f"{pin_name}.pin: {error}") from error
        linked = _pin(linked_name)
        home = root / prefix
        if linked["PREFIX_NAME"] != prefix or not home.is_dir() or not _within(home.resolve(), root):
            raise RunError(
                f"toolchain {name!r} needs {home} ({pin_name}.pin), which is not the installed {linked_name} pin "
                f"inside the toolchains root; install it on the build host with toolchains/{linked_name}.sh"
            )
        environment[variable] = str(home)
        linked_pins[linked_name] = linked
    return linked_pins


def _pin(name: str, keys: Sequence[str] = ("VERSION", "PREFIX_NAME")) -> dict[str, str]:
    """Return toolchains/<name>.pin as read_pin reads it; RunError when it is missing or lacks one of `keys`."""
    try:
        pin = read_pin(name)
    except (OSError, ValueError) as error:
        raise RunError(f"cannot read the pin file toolchains/{name}.pin: {error}") from error
    for key in keys:
        if not pin.get(key):
            raise RunError(f"toolchains/{name}.pin has no {key}")
    return pin


def _checked_tmpdir() -> str:
    """Return $TMPDIR after checking that it is set and, when $LASSI_SCRATCH is set, resolves inside it.

    Each compile gets a private TMPDIR under its own build dir
    (SandboxedCompileRunner); the parent's TMPDIR holds the --version
    check's directory, and on the build host it must stay on the scratch
    disk (Agent Rule 7). Both paths are resolved, so a `..` segment or a
    link cannot lead outside, and a sibling that only shares a name prefix
    is refused. Without $LASSI_SCRATCH (the gate sets it on the build host)
    there is no scratch root to check against.
    """
    tmpdir = os.environ.get("TMPDIR", "")
    if not tmpdir:
        raise RunError(
            "TMPDIR is not set, so a compiler would write its temporary files to /tmp on the root filesystem "
            "(Agent Rule 7); set TMPDIR to a directory on the scratch disk (the gate sets it on the build host)"
        )
    scratch = os.environ.get("LASSI_SCRATCH", "")
    if scratch and not _within(Path(tmpdir).resolve(), Path(scratch).resolve()):
        raise RunError(
            f"TMPDIR ({tmpdir}) is outside $LASSI_SCRATCH ({scratch}); on the build host temporary files stay on "
            "the scratch disk, never on the root filesystem (Agent Rule 7); set TMPDIR inside $LASSI_SCRATCH"
        )
    return tmpdir


def _compile_hidden_roots(build_root: Path | None) -> tuple[Path, ...]:
    """Return the roots every compile hides: $HOME, $LASSI_SCRATCH, and $LASSI_RUNS_ROOT (those set), then `build_root`.

    Raises RunError when one of them is not an absolute path, so a bad value
    never changes what a compile sees without notice, and when there is
    none, since a sandbox needs at least one hidden root. Only the roots
    set are hidden: without the gate's $LASSI_SCRATCH the scratch root is
    not known, so a compile then hides $HOME and the build root (for a run,
    its runs root) alone.
    """
    candidates = [(f"${variable}", os.environ.get(variable, "")) for variable in _COMPILE_HIDDEN_VARIABLES]
    if build_root is not None:
        candidates.append(("the build root", str(build_root)))
    roots: list[Path] = []
    for what, value in candidates:
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            raise RunError(f"{what} must be an absolute path to be hidden from compiles, got {value!r}")
        if path not in roots:
            roots.append(path)
    if not roots:
        names = ", ".join(f"${variable}" for variable in _COMPILE_HIDDEN_VARIABLES)
        raise RunError(f"none of {names} is set, so a compile would have no directory to hide; set them")
    return tuple(roots)


def _compile_environment() -> dict[str, str]:
    """Return the clean compile environment: the parent's PATH, LANG=C, and LC_ALL=C.

    HOME stays out on purpose (P0.20), and so does TMPDIR: each compile gets
    its own, a private directory under its build dir (SandboxedCompileRunner).
    """
    return {"PATH": os.environ.get("PATH", os.defpath), "LANG": "C", "LC_ALL": "C"}


def _compile_runner(
    name: str, environment: Mapping[str, str], root: Path, hidden_roots: tuple[Path, ...], build_root: Path | None
) -> SandboxedCompileRunner:
    """Return the toolchain's sandboxed compile runner after checking its layout; RunError names what is refused.

    The sandbox checks its layout (the hidden roots, the toolchains root
    `root`, and the build dir) when a compile builds its spec. Checking it
    here, against a build dir under `build_root`, makes a layout it refuses
    (a hidden root that is a filesystem root, a runs root inside the
    toolchains root, a build dir under /tmp) stop the run before the run
    directory exists, never at its first compile. Nothing is created.
    """
    runner = SandboxedCompileRunner(environment=environment, toolchains=root, hidden_roots=hidden_roots)
    if build_root is not None:
        try:
            runner.spec(build_root / _LAYOUT_PROBE)
        except ValueError as error:
            raise RunError(
                f"toolchain {name!r}: the compile sandbox refuses a build dir under {build_root} with the toolchains "
                f"root {root} and the hidden roots {', '.join(map(str, hidden_roots))}: {error}"
            ) from error
    return runner


def _version_argv(factory: type, executable: Path | str) -> list[str]:
    """Return the --version check's command: `executable`, then the class's VERSION_ARGS, else --version alone."""
    return [str(executable), *getattr(factory, "VERSION_ARGS", _VERSION_ARGS)]


def _checked_version(
    name: str,
    argv: Sequence[str],
    runner: SandboxedCompileRunner,
    tmpdir: Path,
    pin_name: str,
    pin: Mapping[str, str],
) -> tuple[tuple[str, ...], int]:
    """Run the --version check `argv` (_version_argv) once through `runner`; return its non-blank lines and status.

    It runs through the toolchain's own compile runner, so it runs the same
    executable with the same environment in the same view as every build,
    under prlimit --core=1, and proves the compiler reachable there before
    the first build. Its build dir is a fresh directory under the parent's
    TMPDIR, removed afterwards. It must exit 0 and print the pin's
    EXPECT_VERSION, so a run is never labeled with a pin its compiler is not
    (Agent Rule 10); a pin without EXPECT_VERSION is refused before anything
    runs. RunError also covers a directory TMPDIR cannot hold and a layout
    the sandbox refuses there; SandboxUnavailableError propagates.
    """
    expected = pin.get("EXPECT_VERSION", "")
    if not expected:
        raise RunError(
            f"toolchains/{pin_name}.pin has no EXPECT_VERSION, so toolchain {name!r} cannot check that "
            f"{argv[0]} is the pinned compiler (Agent Rule 10); add the --version text it must print"
        )
    try:
        probe = Path(tempfile.mkdtemp(prefix=_VERSION_PROBE_PREFIX, dir=tmpdir))
    except OSError as error:
        raise RunError(f"toolchain {name!r}: cannot make the --version check's directory in TMPDIR: {error}") from error
    try:
        result = runner(list(argv), probe, _VERSION_TIMEOUT_S)
    except ValueError as error:
        raise RunError(
            f"toolchain {name!r}: the compile sandbox refuses the --version check's build dir {probe} under TMPDIR "
            f"({tmpdir}): {error}"
        ) from error
    finally:
        shutil.rmtree(probe, ignore_errors=True)
    output = result.stdout + result.stderr
    if result.returncode != 0 or expected not in output:
        raise RunError(
            f"toolchain {name!r}: {' '.join(argv)} exited {result.returncode} and must print the "
            f"EXPECT_VERSION of toolchains/{pin_name}.pin ({expected!r}); the compiler is not the pinned one, "
            "so the run would be labeled with a pin it does not use (Agent Rule 10)"
        )
    return tuple(line.rstrip() for line in output.splitlines() if line.strip()), result.returncode


def _run_pins(
    recipe: Recipe, settings: _Settings, toolchains: Sequence[BuiltToolchain], executors: _Executors
) -> tuple[ToolchainPins, dict[str, ToolchainPins]]:
    """Return the run's pin versions and each target language's, from the toolchains and the executors' pins.

    The run's (provenance.json) are every bound toolchain's and every bound
    executor's pins. A target language's (each of its trials'
    toolchain_pins) are those of the toolchains bound for it, of a
    toolchain bound for no language (the source_mlir lowering toolchain,
    task P17.41, which serves every trial), and of the executor that runs
    it, so a per-language executor's pins join only the trials whose target
    it runs. A pin that an executor declares with other
    pairs than a toolchain or another executor that declares the same pin
    name is a RunError naming both (_check_shared_pins), and so is a pin
    with no ToolchainPins field (_pin_versions); both come before any
    directory exists.
    """
    _check_shared_pins(recipe, toolchains, executors)
    everything = [*(built.pins for built in toolchains), *(item.pins for item in executors.declared)]
    target_pins = {
        direction.target: _pin_versions(
            [
                *(built.pins for built in toolchains if direction.target in built.languages or not built.languages),
                *executors.pins_for(direction.target),
            ]
        )
        for direction in settings.directions
    }
    return _pin_versions(everything), target_pins


def _check_shared_pins(recipe: Recipe, toolchains: Sequence[BuiltToolchain], executors: _Executors) -> None:
    """Refuse a pin an executor declares with other pairs than an earlier declarer of that pin name.

    The earlier declarers are the bound toolchains, then the executors in
    binding order; the same pin declared twice with the same pairs is one
    pin (a ttsim run's tt-metal pin, which the host toolchain declares
    too). Two toolchains are not compared here, as before task P4.11.
    """
    first: dict[str, tuple[str, Mapping[str, str]]] = {}
    for built in toolchains:
        for name, pin in built.pins.items():
            first.setdefault(name, (f"Toolchain {built.name!r}", pin))
    for item in executors.declared:
        for name, pin in item.pins.items():
            label, earlier = first.setdefault(name, (item.label, pin))
            if dict(earlier) != dict(pin):
                keys = sorted(key for key in {*earlier, *pin} if earlier.get(key) != pin.get(key))
                raise RunError(
                    f"{recipe.path}: {label} and {item.label} both declare the pin {name!r} but differ in "
                    f"{', '.join(keys)} ({label}: VERSION {earlier.get('VERSION')!r}; {item.label}: VERSION "
                    f"{pin.get('VERSION')!r}); a run records one version of each pin (Agent Rule 10)"
                )


def _pin_versions(pin_maps: Iterable[Mapping[str, Mapping[str, str]]]) -> ToolchainPins:
    """Return the recorded version of every pin the given pin maps hold (pin name -> pairs), under its field.

    The recorded version is VERSION, or <VERSION>-<BUILD> for a pin that names
    a build variant (lassi.toolchains.pins recorded_version; task P17.42), so
    a trial on the Tracy build of tt-metal records 5280a9cf-tracy.
    """
    versions: dict[str, str] = {}
    for pins in pin_maps:
        for pin_name, pin in pins.items():
            field = pin_name.replace("-", "_")
            if field not in TOOLCHAIN_PIN_NAMES:
                raise RunError(f"the pin {pin_name!r} has no Trial toolchain_pins field; fields: {TOOLCHAIN_PIN_NAMES}")
            versions[field] = recorded_version(pin)
    return ToolchainPins(**versions)


def _toolchains_record(toolchains: Sequence[BuiltToolchain]) -> dict[str, Any]:
    """Return toolchains.json: per toolchain name, its languages, executable, environment, and pin files."""
    return {
        built.name: {
            "languages": list(built.languages),
            "executable": built.executable,
            "environment": built.environment,
            "pins": built.pins,
        }
        for built in toolchains
    }


# ---------------------------------------------------------------------------
# Trials


def _run_trials(run: _Run, manifest: Mapping[str, Any]) -> list[Trial]:
    """Run and write every trial: directions in recipe order, items sorted, runs 1 to n.

    Each trial carries the provenance the first-written `manifest` gives its
    direction's target language (_trial_provenance). The recipe's `score`
    profile, when it binds one, scores each trial before it is written
    (_scored). Its trial.md labels its run wall times as simulator wall time
    when its target language's executor declares SIMULATOR (task P4.6). The
    observer, when the run has one, gets each trial's trial-end event after
    the trial is written and before its line is printed.
    """
    trials: list[Trial] = []
    count = len(run.settings.directions) * len(run.bench.items) * run.settings.trials
    for direction in run.settings.directions:
        provenance = _trial_provenance(manifest, direction.target)
        simulator = declares(run.executors.for_language(direction.target), SIMULATOR)
        for item in run.bench.items:
            for number in range(1, run.settings.trials + 1):
                place = (len(trials) + 1, count)
                trial = _scored(run, _run_trial(run, provenance, direction, item, number, place))
                write_trial(trial, run.run_dir, run.store, simulator=simulator)
                trials.append(trial)
                notify(run.observer, TRIAL_END, trial)
                final = trial.final
                ended = "" if final.end_reason is None else f"  end {final.end_reason.code}"
                print(
                    f"{trial.trial_id}  stage {fmt(final.stage_reached)}  corrections {final.corrections}  "
                    f"wall_s {fmt(final.wall_s)}{ended}",
                    flush=True,
                )
    return trials


def _run_trial(
    run: _Run, provenance: Provenance, direction: Direction, item: str, number: int, place: tuple[int, int]
) -> Trial:
    """Run the recipe's stages on one new trial, each built on a fresh RunContext, and set its final block.

    The trial id's first segment is the recipe's project. A backend that
    declares `needs_reference` gets the item's reference target, as one
    untagged fence when fixes.fence_tag is off (faithful extraction reads
    that form). A backend that declares `unload_before_run` is asked to
    unload first, as upstream's setup unloads the model before anything
    runs. A stage that sets final.end_reason ends the trial: no later stage
    runs, and the final block keeps the end reason. Every trial starts with
    an empty Trial.requests, so its model requests are recorded and a trial
    that ends before any model call records none ([], never None). The
    observer gets the trial-start event (_trial_start; `place` is the
    trial's 1-based place in the run and the run's trial count) before the
    backend unloads, and a stage-start event before each stage runs.
    """
    settings, suite = run.settings, run.bench.suite
    backend = run.backend
    if "needs_reference" in backend.capabilities:
        backend = backend.with_reference(suite.reference_target(item, direction, run.bench.root, purpose=PURPOSE))
        if not run.recipe.data["fixes"]["fence_tag"]:
            backend = backend.with_untagged_fence()
    trial = Trial(
        trial_id=make_trial_id(
            run.recipe.data["project"], arm_segment(settings.model_id), suite.name, direction.name, item, number
        ),
        recipe_hash=run.recipe.recipe_hash,
        toolchain_pins=run.target_pins[direction.target],
        provenance=provenance,
        bench_item=suite.bench_item(item, direction),
        model=model_info(backend, settings.sampling),
        requests=[],
    )
    context = _context(run, backend, direction, item)
    _trial_start(run, trial, direction, item, place)
    started = time.monotonic()
    unload_before_run(backend)
    names = run.recipe.data["stages"]
    stages = [run.registry.get("Stage", name).factory(context=context) for name in names]
    for name, stage in zip(names, stages, strict=True):
        notify(run.observer, STAGE_START, trial, name)
        trial = stage(trial)
        if trial.final.end_reason is not None:
            break
    return dataclasses.replace(trial, final=_final(trial, time.monotonic() - started))


def _context(run: _Run, backend: Any, direction: Direction, item: str) -> RunContext:
    """Return a fresh RunContext for one trial of `item` in `direction`, with the run's observer."""
    settings = run.settings
    return RunContext(
        recipe=run.recipe,
        backend=backend,
        sampling=settings.sampling,
        toolchains={language: built.toolchain for built in run.toolchains for language in built.languages},
        executor=run.executors.for_language(direction.target),
        executors=run.executors.by_language,
        store=run.store,
        suite=run.bench.suite,
        sources_root=run.bench.root,
        item=item,
        direction=direction,
        build_root=run.run_dir,
        prompts=settings.prompts,
        max_corrections=settings.max_corrections,
        fragments=settings.fragments,
        packs=settings.packs,
        observer=run.observer,
        profiler=run.profiler,
        registry=run.registry,
        retrieval=run.retrieval,
        examples=run.examples,
        skeleton=run.skeleton,
        dialect=run.dialect,
        frontend=None if run.frontend is None else run.frontend.toolchain,
    )


def _trial_start(run: _Run, trial: Trial, direction: Direction, item: str, place: tuple[int, int]) -> None:
    """Send the observer the trial-start event: the run id, the trial's place and the run's count, and the source.

    Nothing is read when the run has no observer or it was dropped. The
    source is the item's files in the direction's source language as
    Suite.source_files reads them, None when they cannot be read, so the
    event never stops a run.
    """
    observer = run.observer
    if observer is None or observer.dropped:
        return
    try:
        source = run.bench.suite.source_files(item, direction, run.bench.root, purpose=PURPOSE)
    except (OSError, ValueError):
        source = None
    number, count = place
    observer(
        ProgressEvent(
            kind=TRIAL_START, trial=trial, run_id=run.run_dir.name, number=number, count=count, source=source
        )
    )


def _scored(run: _Run, trial: Trial) -> Trial:
    """Return the trial with the bound ScoreProfile's scores (lassi.scoring.run_scoring.score_trial).

    A profile that cannot score the trial raises RunError, which leaves
    provenance.json with status "failed".
    """
    try:
        return score_trial(run.scoring, trial)
    except ScoreError as error:
        raise RunError(f"{run.recipe.path}: score {run.scoring.score!r}: {error}") from error


def _metrics(run: _Run, trials: Sequence[Trial]) -> RunMetrics | None:
    """Return the run's named metrics (lassi.scoring.run_scoring.compute_metrics); None when it names none.

    A profile that cannot score a trial, or tables that cannot be built,
    raise RunError, which leaves provenance.json with status "failed".
    """
    try:
        return compute_metrics(run.scoring, trials)
    except (ScoreError, ValueError) as error:
        raise RunError(f"{run.recipe.path}: metrics: {error}") from error


def _final(trial: Trial, wall_s: float) -> Final:
    """Return the final block of a trial whose stages have run, taking `wall_s` as its wall time.

    It holds the last attempt's stage, the correction count, the alignment
    mean of the attempt whose output stands (standing_attempt; None when no
    attempt ran), and the end reason a stage set. A stage that sets the end
    reason ends the trial, so the oracle stage never aligns a trial that
    ended at the correction cap, and its alignment stays None, unless
    run_loop judged the attempt whose output stands under
    loop.correct_on_mismatch, whose alignment then stands.
    """
    last = trial.attempts[-1].stage_reached if trial.attempts else None
    named = standing_attempt(trial)
    return Final(
        stage_reached=last,
        alignment=None if named is None else named.alignment.mean,
        corrections=max(len(trial.attempts) - 1, 0),
        wall_s=wall_s,
        end_reason=trial.final.end_reason,
    )


# ---------------------------------------------------------------------------
# Provenance and run.md


def git_state() -> tuple[str | None, bool | None]:
    """Return the repository's commit and whether `git status --porcelain` lists anything; None when git fails."""
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    commit = (head.strip() or None) if head is not None else None
    dirty = bool(status.strip()) if status is not None else None
    return commit, dirty


def _git(*args: str) -> str | None:
    """Return the stdout of `git <args>` run in the repository, or None when git is missing or fails."""
    try:
        result = EnvRunner(os.environ)(["git", *args], REPO, _GIT_TIMEOUT_S)
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None


def _utc(moment: datetime) -> str:
    """Return a UTC time as ISO 8601 with seconds, such as 2026-09-23T12:34:56+00:00."""
    return moment.isoformat(timespec="seconds")


def _device(executor: Executor) -> str | None:
    """Return the device an executor's programs run on: its device() (task P4.5; Agent Rule 1).

    An executor without device() (a test fake) names "none (compile only)"
    when it declares compile_only and no device (None, null) otherwise.
    """
    named = getattr(executor, "device", None)
    if callable(named):
        return named()
    return "none (compile only)" if "compile_only" in getattr(executor, "capabilities", ()) else None


def _provenance(run: _Run) -> dict[str, Any]:
    """Return provenance.json as first written: where the run came from, when it started, what it ran on, the pins.

    Its status is "running" and `finished_utc` is null until
    _final_provenance closes it. The executor keys are the run's
    (_Executors.record): executor and device, or, with executors per
    language, executor and devices by language. `device_records` holds the
    record of each device section in binding order ([] when the recipe names
    none; task P17.2), and `driver` the distinct drivers those records name,
    joined by "; " (lassi.core.devices.device_driver), null when none names
    one. `serving` holds the backend's serving record when it declares
    model_check (task P17.3; _served_model) and is absent otherwise.
    """
    pins = dataclasses.asdict(run.pins)
    return {
        "run_id": run.run_dir.name,
        "status": RUNNING,
        "recipe": run.recipe.name,
        "recipe_path": Path(run.recipe.path).as_posix(),
        "recipe_chain": list(run.recipe.chain),
        "recipe_hash": run.recipe.recipe_hash,
        "commit": run.commit,
        "dirty": run.dirty,
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        **run.executors.record,
        "driver": device_driver(run.devices),
        "device_records": [dataclasses.asdict(item) for item in run.devices],
        **({} if run.serving is None else {"serving": dict(run.serving)}),
        "started_utc": _utc(run.started),
        "finished_utc": None,
        "pins": {name: version for name, version in pins.items() if version is not None},
    }


def _final_provenance(manifest: Mapping[str, Any], status: str) -> dict[str, Any]:
    """Return the first-written manifest with its final `status` and the finish time; every other value stays.

    The trials copied their provenance from that first manifest, so building
    the final one from it (not from the run again) keeps every trial's copy
    equal to provenance.json.
    """
    return {**manifest, "status": status, "finished_utc": _utc(datetime.now(timezone.utc))}


def _trial_provenance(manifest: Mapping[str, Any], language: str) -> Provenance:
    """Return the Trial provenance a run manifest gives a trial whose target is `language`.

    Every trial carries a copy of provenance.json: commit and dirty keep
    their manifest keys; device is the manifest's "device", or, with
    executors per language, its "devices" entry for `language`; sdk is the
    manifest's "driver", date its "started_utc", and device_records its
    "device_records", the whole list. A key the manifest lacks raises
    KeyError, and a value of the wrong type raises ValueError: the copy never
    fills in a value the manifest does not hold.
    """
    device = manifest["devices"][language] if "devices" in manifest else manifest["device"]
    return Provenance(
        commit=manifest["commit"],
        dirty=manifest["dirty"],
        device=device,
        sdk=manifest["driver"],
        date=manifest["started_utc"],
        device_records=[from_dict(DeviceRecord, item) for item in manifest["device_records"]],
    )


def _md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Return a Markdown table block; a pipe in a cell is escaped and a line break becomes a space."""

    def cell(text: str) -> str:
        return " ".join(text.replace("|", "\\|").splitlines())

    lines = [header, ["---"] * len(header), *rows]
    return "".join("| " + " | ".join(cell(text) for text in line) + " |\n" for line in lines)


def _pin_rows(toolchains: Sequence[BuiltToolchain]) -> list[tuple[str, ...]]:
    """Return one run.md row per pin each toolchain uses, or one 'not pinned' row for a toolchain without one."""
    rows: list[tuple[str, ...]] = []
    for built in toolchains:
        languages = ", ".join(built.languages)
        if not built.pins:
            rows.append((built.name, languages, "-", "not pinned", "-"))
        for pin_name, pin in built.pins.items():
            prefix = pin.get("PREFIX_NAME") or f"none, host {pin.get('EXECUTABLE', '-')}"
            rows.append((built.name, languages, pin_name, pin["VERSION"], prefix))
    return rows


def _trial_blocks(trials: Sequence[Trial]) -> list[str]:
    """Return one heading and table per arm (model id), in order of first appearance, linking each trial.md."""
    arms: dict[str, list[Trial]] = {}
    for trial in trials:
        arms.setdefault(trial.model.id, []).append(trial)
    header = ("trial", "direction", "item", "stage reached", "corrections", "wall_s", "trial.md")
    blocks: list[str] = []
    for arm, arm_trials in arms.items():
        rows = [
            (
                f"`{trial.trial_id}`",
                trial.bench_item.direction,
                trial.bench_item.item,
                fmt(trial.final.stage_reached),
                fmt(trial.final.corrections),
                fmt(trial.final.wall_s),
                f"[trial.md]({trial.trial_id}/trial.md)",
            )
            for trial in arm_trials
        ]
        blocks += [f"### Arm {arm}\n", _md_table(header, rows)]
    return blocks or ["No trials.\n"]


def _simulator_note(run: _Run, trials: Sequence[Trial]) -> list[str]:
    """Return the Trials section's note on simulator wall times, or nothing (task P4.6; Agent Rule 2).

    The note is written when the run has trials and _simulated_languages
    names any language; it names them. Otherwise run.md is as before.
    """
    languages = _simulated_languages(run)
    if not trials or not languages:
        return []
    return [
        f"The {', '.join(languages)} programs of this run ran on a simulator executor: their run wall times are "
        "simulator wall time, not performance, and trial.md labels each one it shows. The wall_s column below is "
        "each trial's pipeline wall time, which includes those runs, so it is not performance either.\n"
    ]


def _simulated_languages(run: _Run) -> list[str]:
    """Return, sorted, each language whose programs the run's trials run on an executor that declares SIMULATOR.

    A direction's target language counts when its executor declares it, and
    so does its source language when a listed stage builds the source
    reference under a fix that is on (`source_build_fix`, baseline under
    baseline_both), since that reference then runs on the source language's
    executor and its run time is part of the trial's pipeline wall time.
    """
    stages = [run.registry.get("Stage", name).factory for name in run.recipe.data["stages"]]
    fixes = [getattr(stage, "source_build_fix", None) for stage in stages]
    sources = any(fix is not None and run.recipe.data["fixes"].get(fix, True) is not False for fix in fixes)
    found: set[str] = set()
    for direction in run.settings.directions:
        languages = [direction.target]
        if sources and direction.source != direction.target:
            languages.append(direction.source)
        found.update(language for language in languages if declares(run.executors.for_language(language), SIMULATOR))
    return sorted(found)


def _executor_rows(provenance: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return run.md's executor rows: Executor and Device, or, per language, the executors and one Device row each."""
    if "devices" not in provenance:
        return [("Executor", provenance["executor"]), ("Device", fmt_provenance(provenance["device"]))]
    names = "; ".join(f"{language}: {name}" for language, name in sorted(provenance["executor"].items()))
    devices = sorted(provenance["devices"].items())
    return [("Executor", names), *((f"Device ({language})", fmt_provenance(device)) for language, device in devices)]


def _serving_rows(provenance: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return run.md's Server row (lassi.llm.serving_line) when provenance has `serving`, else no row."""
    if "serving" not in provenance:
        return []
    return [("Server", serving_line(provenance["serving"]))]


def _run_md(
    run: _Run, provenance: Mapping[str, Any], trials: Sequence[Trial], metrics: RunMetrics | None = None
) -> str:
    """Return run.md: the run summary, the resolved recipe, the toolchain pins, and one trials table per arm.

    The Trials section opens with _simulator_note when it applies, and the
    Guard coverage section (task P17.14) follows it. With
    `metrics` (the recipe names metrics), a Metrics section follows
    that (lassi.scoring.run_scoring.metrics_section). It is
    deterministic for given records, plain ASCII (non-ASCII becomes
    backslash escapes) with LF newlines, and names no absolute run path, so
    it reads the same wherever the run tree is copied.
    """
    summary = [
        ("Recipe", run.recipe.name),
        ("Recipe hash", f"`{run.recipe.recipe_hash}`"),
        ("Commit", fmt_provenance(provenance["commit"])),
        ("Dirty", fmt_provenance(provenance["dirty"])),
        *_executor_rows(provenance),
        ("Driver", fmt_provenance(provenance["driver"])),
        *_serving_rows(provenance),
        ("Started (UTC)", provenance["started_utc"]),
        ("Finished (UTC)", fmt(provenance["finished_utc"])),
        ("Trials", str(len(trials))),
    ]
    blocks = [f"# Run {run.run_dir.name}\n", _md_table(("Field", "Value"), summary)]
    blocks += ["## Resolved recipe\n", fenced(resolved_yaml(run.recipe), "yaml")]
    pin_header = ("Toolchain", "Languages", "Pin", "Version", "Install prefix")
    blocks += ["## Toolchain pins\n", _md_table(pin_header, _pin_rows(run.toolchains))]
    blocks += ["## Trials\n", *_simulator_note(run, trials), *_trial_blocks(trials)]
    blocks.append(coverage_markdown(guard_coverage(trials)))
    if metrics is not None:
        blocks.append(metrics_section(metrics, run.scoring.score))
    return "\n".join(blocks).encode("ascii", "backslashreplace").decode("ascii")


def _write(path: Path, text: str) -> None:
    """Write `text` to `path` as ASCII bytes, so newlines stay LF on every OS."""
    path.write_bytes(text.encode("ascii"))
