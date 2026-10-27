"""Load run recipes: extends chains, defaults, faithful overrides, checks, and the recipe hash.

A project is a recipe, not code (bible, Design Principle 1). Loading a recipe:

1. parses each file strictly (a repeated key is an error, never a silent overwrite, and nesting is bounded);
2. refuses any key the schema below does not list, naming the file and dotted path;
3. follows `extends` to the root ancestor;
4. merges from the root down, the child winning (mappings merge, anything else is replaced, a
   kind section that names a new kind replaces the inherited one, an executor section that
   switches between its two forms replaces the inherited one, and a `device` mapping replaces the
   inherited one whole, since a device is one choice);
5. materializes the defaults (`project`: the loaded file's recipe name; `faithful: false`; every fix on; no
   `extract` leniency, no LOOP_OPT_INS key, no `retrieval` section, no `lookup` section, and no
   llm.sampling.reasoning_effort, each off when absent, so a recipe without one keeps its resolved recipe and hash);
6. applies the faithful overrides (bible, Project Recipes Notes; Design Principle 4);
7. checks types and required choices, never picking a value for a choice left open, and refuses an EXTRACT
   leniency set true under faithful: true or with fixes.fence_tag off (_check_extract), a LOOP_OPT_INS key
   set true (restart_after: set at all) under faithful: true, and a loop.restart_after past loop.max_corrections
   (_check_loop; task P17.25), naming its key, and a `retrieval` section (task P17.22: every
   key required, k and max_chars within RETRIEVAL_K and RETRIEVAL_MAX_CHARS, prompts distinct RETRIEVAL_PROMPTS)
   under faithful: true (_check_retrieval), and a `lookup` section (task P17.26: max_chars and prompts required,
   max_chars within LOOKUP_MAX_CHARS, prompts distinct LOOKUP_PROMPTS; core required exactly when prompts lists
   generate, nearest within LOOKUP_NEAREST only when it lists correct) under faithful: true (_check_lookup), and
   llm.sampling.reasoning_effort (OQ-080: one of REASONING_EFFORTS) under faithful: true (_check_sampling);
8. binds components and checks their capabilities without constructing any (Component Interfaces), the
   adversary section's Agent among them, which must declare proposes_inputs (task P16.2); refuses a faithful
   recipe that binds a toolchain declaring a proxy capability (PROXY_CAPABILITIES), a toolchain upstream
   never used (NOT_UPSTREAM_TOOLCHAINS), or an offload build (OFFLOAD_CAPABILITIES) with an executor that
   runs programs and takes no device section (task P16.2); and reads the device section of every component
   that declares `takes_device` (task P17.2): one is required, never picked, and
   lassi.core.devices.parse_device refuses a bad one naming the key;
9. checks model.recording, the replay backend's recording (task P16.13; _bind_recording), and
   adversary.recording, an adversary's own replay backend's (task P16.4): a path relative to the repository
   root (REPO, read at each load, never the working directory), written with / separators and no empty,
   `.`, or `..` segment, that names a file under tests/fixtures/ exactly as spelled and still lies inside
   tests/fixtures/ once links are followed (recording_file); anything else is refused naming the key. The
   sha256 of the file's bytes enters the section as <section>.recording_sha256, which a recipe may state and
   which must then match, so a rerun from the resolved recipe reads the same bytes; the backend's binding
   carries the checked file, and the adversary's binding the sha256 beside the path as written;
10. serializes the resolved mapping to canonical YAML and hashes it.

The optional `examples` section (EXAMPLES; OQ-083 (b)) is like `lookup`: never materialized in step 5, so a recipe
without it keeps its resolved recipe and hash, and in step 7 every key required (pack a plain name, max_chars within
EXAMPLES_MAX_CHARS, prompts [generate] or [generate, correct]) and the section refused under faithful: true, set or
inherited (_check_examples).

The optional `skeleton` section (SKELETON; OQ-096 (a)) is never materialized either, so a recipe without it keeps
its resolved recipe and hash. In step 7 name and prompts are required (prompts [generate] or [generate, correct]),
and _bind_skeleton refuses the section under faithful: true, set or inherited, then reads the skeleton by name
(lassi.prompts.skeletons), refusing one that does not match its manifest, a stated skeleton.sha256 that is not its
sha256, and a direction whose target is not its language, and enters its sha256 at skeleton.sha256.

The optional `dialect_lookup` and `source_mlir` sections (task P17.41; DIALECT_LOOKUP, SOURCE_MLIR) are never
materialized either, so a recipe without them keeps its resolved recipe and hash. In step 7 each one's required keys
are checked, and either is refused under faithful: true, set or inherited (_check_unfaithful_sections); the runner
reads the definitions tree and builds the lowering toolchain (lassi.core.runner).

`executor` has two forms (task P4.5). The single form is a kind section, `{kind: <name>, <config>...}`,
binding one Executor at executor.kind for every language. The per-language form is a non-empty mapping
without `kind`, from a language to an Executor name (`cuda: native`) or to a kind section with that
executor's config (`omp: {kind: native, harness: ...}`), like `toolchain`; it binds one Executor per
language, at executor.<language> or executor.<language>.kind, in sorted language order
(executor_languages). A mapping with `kind` is always the single form, so a file that sets only config keys
of an inherited single form restates its kind. Within the per-language form a child merges entry by entry:
two mappings for one language merge unless the child names another kind, which replaces the inherited entry,
and any other child entry replaces it.

The canonical YAML, with a two-line header, is the resolved recipe every run saves
and can be rerun from (Design Principle 5). It always holds `project`, the first
trial id segment, so a rerun from it writes the same trial ids.

Train recipes (task P17.8; bible Project Recipes, Notes, train recipes) load with
load_train_recipe: steps 1 to 4, 7, 8, and 10 as above, against TRAIN_SCHEMA, with no
default materialized and no faithful override, so each file of the chain must be a train
recipe. A train recipe binds one Trainer at trainer.kind, whose section holds its device
and config keys; it names exactly one data source, bench or data.synthetic; bench.split
must be train (Agent Rule 5); and its method, weights, and data source must be among the
names its Trainer declares (methods, weight_modes, data_sources), read without building it.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import yaml

from lassi.bench.registry import FORBIDDEN_PURPOSES, HELD_SPLITS
from lassi.core.capabilities import (
    DEVICE_KEY,
    PROPOSES_INPUTS,
    REASONING_EFFORT,
    REASONING_EFFORTS,
    RUNS_CODE,
    TAKES_DEVICE,
)
from lassi.core.devices import DeviceSectionError, parse_device
from lassi.core.registry import DEFAULT_REGISTRY, Binding, Registry, RegistryError, check_bindings, device_path
from lassi.prompts import skeletons as skeleton_files

# The value of loop.max_corrections that means no cap (upstream LASSI's loop; bible LASSI quirk table).
UNCAPPED = "uncapped"

# The text of model.api_key_env: a variable name (task P17.3), stricter than a backend's own check for direct callers.
_ENV_NAME_TEXT = re.compile(r"[A-Z_][A-Z0-9_]{0,63}")
# The text of model.revision: a full commit id, the only pin (Agent Rule 10; task P17.4). A branch or tag is not one.
_REVISION_TEXT = re.compile(r"[0-9a-f]{40}")
# model.seed's bound: the seed of call k is seed + k (bible Serving Rules), far inside torch.manual_seed's range.
_SEED_LIMIT = 2**32

# The repository root, against which model.recording is read (task P16.13); load_recipe reads it at each load.
REPO = Path(__file__).resolve().parents[2]
# model.recording: the replay backend's recording, a path relative to REPO under RECORDING_DIR (task P16.13). The
# loader enters the sha256 of its bytes beside it, at model.recording_sha256, so the recipe hash covers the bytes.
RECORDING_KEY = "recording"
RECORDING_SHA256_KEY = "recording_sha256"
RECORDING_DIR = ("tests", "fixtures")
# The kind section whose Agent may take a recording too, adversary.recording (task P16.4; _KIND_SECTIONS).
_ADVERSARY = "adversary"
# The text of model.recording_sha256: a sha256 hex digest as hashlib writes it.
_SHA256_TEXT = re.compile(r"[0-9a-f]{64}")

# Every named fix toggle and what turning it on changes. Fixes default to on; faithful: true turns them all off.
FIXES: dict[str, str] = {
    "fence_tag": (
        "strip only the exact fence language tag;"
        " off keeps upstream's quirk that also drops a leading 'c' after cpp/c++"
    ),
    "prompt_spaces": (
        "keep the generation prompt's runs of spaces;"
        " off keeps upstream's quirk that cuts every run of spaces to one before the first generation call"
    ),
    "baseline_both": (
        "build and run the source reference as well as the target reference before any model call;"
        " off keeps upstream's baseline, which builds and runs only the target reference"
    ),
    "prompt_newlines": (
        "keep the correction prompt's line feeds;"
        " off keeps upstream's quirk that removes every line feed from a correction prompt before sending it"
    ),
    "parsed_diagnostics": (
        "send the parsed compile diagnostics, capped in count and bytes, in a correction prompt;"
        " off keeps upstream's behavior of sending the whole raw compiler stderr, read in text mode"
    ),
    "execution_gate": (
        "run every compiling attempt within the correction cap;"
        " off keeps upstream's quirk that runs a compiling attempt only while its correction count is at most 7,"
        " so a later compiling attempt ends the trial unexecuted, with an earlier run's stdout as stale output"
    ),
}

# Every opt-in leniency of the extract section and what turning it on changes. Unlike a fix, each is off when absent
# and the loader materializes none, so a recipe without the section keeps its resolved recipe and hash. Each reads
# this project's FILE-block format, which a reply has only with fixes.fence_tag on, and none is upstream behavior, so
# one set true is refused under faithful: true or with fixes.fence_tag off (_check_extract), never turned off silently.
EXTRACT: dict[str, str] = {
    "unfenced_file_lines": (
        "read a reply that holds no FILE block from its unfenced `// FILE: <path>` lines, each file running to the"
        " next such line, with a parse-stage unfenced-file warning per file; off reads only fenced FILE blocks"
    ),
}

# Every opt-in key of the loop section beside max_corrections, each a bool but RESTART_AFTER, and what turning it on
# changes. Like an extract leniency, each is off when absent and the loader materializes none, so a recipe without one
# keeps its resolved recipe and hash. None is upstream behavior: upstream's loop corrects only compile errors and
# failed runs, never asks a fresh generation, and runs on no simulator, so a bool set true, or RESTART_AFTER set at
# all, is refused under faithful: true (_check_loop), never turned off silently.
LOOP_OPT_INS: dict[str, str] = {
    "correct_on_mismatch": (
        "ask for a correction after a clean run whose output files fail the oracle on the held-out inputs (an output"
        " outside its threshold, or a missing or extra one), from a coarse per-output report (OQ-077), within"
        " loop.max_corrections; off ends the loop at the first clean run, as upstream's loop does"
    ),
    "correct_on_sim_gap": (
        "ask for a correction after a run that stopped at a simulator gap with no kernel JIT error, naming the gap's"
        " class and saying that the simulator cannot run that construct, a limit of the simulator and not necessarily"
        " of the program, so an equivalent construct the simulator runs should be used if one exists (OQ-098), within"
        " loop.max_corrections; off ends the trial at sim-gap right after that attempt, with no correction"
    ),
    "restart_after": (
        "after that many corrections in a row that each failed to reach a clean run, ask for a fresh generation from"
        " attempt 0's prompt in place of the next correction, within loop.max_corrections; absent, every request"
        " after attempt 0 is a correction, as in upstream's loop"
    ),
    "coord_gap_as_error": (
        "read a run that stopped at ttsim's coord_to_tile gap on a NOC coordinate (UnimplementedFunctionality:"
        " coord_to_tile: coord (x,y), a coordinate that maps to no tile of the simulated device) as a failed run, fed"
        " back as an addressing error in the kernel, within loop.max_corrections (task P17.41); off reads it as any"
        " other simulator gap (OQ-098)"
    ),
}
# The one LOOP_OPT_INS key that is an int (task P17.25): from 1 to the resolved loop.max_corrections, both included,
# or any int of at least 1 when the loop is uncapped. The schema checks the type and _check_loop the cap.
RESTART_AFTER = "restart_after"

# The optional retrieval section (task P17.22): documentation passages from an index before the prompts it lists
# (lassi.core.stages). Off when absent, and never materialized, so a recipe without it keeps its resolved recipe and
# hash. Upstream's prompts carry no passages, so it is refused under faithful: true (_check_retrieval). Its keys are
# all required: index (a corpus directory under assets/rag/ holding MANIFEST.yaml), k (passages per prompt) and
# max_chars (the passages' total text per prompt) within the bounds below, and prompts (distinct RETRIEVAL_PROMPTS).
RETRIEVAL = "retrieval"
RETRIEVAL_GENERATE = "generate"
RETRIEVAL_CORRECT = "correct"
RETRIEVAL_PROMPTS = (RETRIEVAL_GENERATE, RETRIEVAL_CORRECT)
RETRIEVAL_K = (1, 16)
RETRIEVAL_MAX_CHARS = (500, 32000)
_INDEX_TEXT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

# The optional lookup section (task P17.26): the target language's context pack's declarations of the identifiers a
# compile correction's listed errors name, before the prompts it lists (lassi.core.stages). Off when absent, and never
# materialized, so a recipe without it keeps its resolved recipe and hash. Upstream's prompts carry no declarations,
# so it is refused under faithful: true (_check_lookup). Two keys are required: max_chars (the declarations'
# total text per prompt) within the bounds below, and prompts (distinct LOOKUP_PROMPTS). A generation prompt shows no
# error to read, so when prompts lists generate the declarations it carries are those of the pack's header parts
# that core names (2026-10-25): core is then required, a non-empty list of distinct header paths as the pack's
# include lines name them (the runner refuses one the pack has no part for), and refused otherwise. nearest
# (optional, within LOOKUP_NEAREST, only when prompts lists correct) adds, for each identifier of a listed error that
# names no declaration, a line naming up to that many declared names nearest to it.
LOOKUP = "lookup"
LOOKUP_GENERATE = "generate"
LOOKUP_CORRECT = "correct"
LOOKUP_PROMPTS = (LOOKUP_GENERATE, LOOKUP_CORRECT)
LOOKUP_MAX_CHARS = (500, 32000)
LOOKUP_CORE = "core"
LOOKUP_NEAREST = "nearest"
LOOKUP_NEAREST_RANGE = (1, 8)
# A header path of lookup.core: no blank, `<`, or `>`, as the include line `// ==== #include <H> ====` holds it.
_HEADER_TEXT = re.compile(r"[^\s<>]+")

# The optional examples section (OQ-083 (b), 2026-10-25): worked examples, complete programs from a pack under
# assets/examples/ (lassi.prompts.examples), before the prompts it lists (lassi.core.stages). Off when absent, and
# never materialized, so a recipe without it keeps its resolved recipe and hash. Upstream's prompts carry no
# examples, so it is refused under faithful: true (_check_examples). Its keys are all required: pack (a directory
# under assets/examples/ holding MANIFEST.yaml), max_chars (the examples' total text per prompt) within the bounds
# below, and prompts (distinct EXAMPLES_PROMPTS holding generate: [generate] or [generate, correct]).
EXAMPLES = "examples"
EXAMPLES_GENERATE = "generate"
EXAMPLES_CORRECT = "correct"
EXAMPLES_PROMPTS = (EXAMPLES_GENERATE, EXAMPLES_CORRECT)
EXAMPLES_MAX_CHARS = (1000, 64000)

# The optional skeleton section (OQ-096, option (a), 2026-10-25): a host program skeleton, a directory under
# assets/skeletons/ (lassi.prompts.skeletons), whose block goes before the prompts it lists (lassi.core.stages). Off
# when absent, and never materialized, so a recipe without it keeps its resolved recipe and hash. Upstream's prompts
# carry no skeleton, so it is refused under faithful: true (_bind_skeleton). Two keys are required: name (a plain
# name) and prompts (distinct SKELETON_PROMPTS holding generate: [generate] or [generate, correct]). The loader reads
# the skeleton by name, refusing one that does not match its manifest, and enters its sha256 at skeleton.sha256,
# which a recipe may state and which must then match, so the recipe hash covers the skeleton's bytes.
SKELETON = "skeleton"
SKELETON_GENERATE = "generate"
SKELETON_CORRECT = "correct"
SKELETON_PROMPTS = (SKELETON_GENERATE, SKELETON_CORRECT)
SKELETON_SHA256_KEY = "sha256"

# The optional dialect_lookup section (task P17.41): for each error of an MLIR tool's parse or verify step that a
# compile correction lists, the exact definition and assembly format of the op or type it names or points at, and for
# a name the dialect lacks its nearest op and type names, from a definitions tree under assets/context/ that
# tools/build_ttkernel_context.py builds from the pinned dialect sources (lassi.prompts.dialect_defs), before the
# correction (lassi.core.stages). Off when absent, and never materialized, so a recipe without it keeps its resolved
# recipe and hash. Upstream's prompts carry no definitions, so it is refused under faithful: true
# (_check_dialect_lookup). defs (a tree name) and max_chars (the definitions' total text per prompt) are required;
# nearest (how many nearest names per unknown name) is optional, default DIALECT_NEAREST_DEFAULT.
DIALECT_LOOKUP = "dialect_lookup"
DIALECT_MAX_CHARS = (500, 32000)
DIALECT_NEAREST = "nearest"
DIALECT_NEAREST_RANGE = (1, 8)
DIALECT_NEAREST_DEFAULT = 4

# The optional source_mlir section (task P17.41): the item's compute-only C unit (its suite manifest's `compute`)
# lowered by a Toolchain that declares lowers_source (lassi.core.capabilities; the pinned cgeist, cgeist-affine) once
# per trial, and the module shown in the generation prompt's source_mlir field (lassi.core.stages). Off when absent,
# and never materialized, so a recipe without it keeps its resolved recipe and hash; upstream's prompts show no MLIR,
# so it is refused under faithful: true (_check_source_mlir). toolchain (a registry name) and max_chars (the largest
# module shown; a longer one is never cut, but left out with a note) are required.
SOURCE_MLIR = "source_mlir"
SOURCE_MLIR_MAX_CHARS = (1000, 64000)

# The opt-in sampling key llm.sampling.reasoning_effort (OQ-080; bible Serving Rules, reasoning effort): one of
# REASONING_EFFORTS, which a backend declaring the capability of that name takes (lassi.core.capabilities). Off when
# absent, and never materialized, so a recipe without it keeps its resolved recipe and hash. Upstream sends no such
# field, so it is refused under faithful: true, set or inherited (_check_sampling).
REASONING_EFFORT_KEY = f"llm.sampling.{REASONING_EFFORT}"


# Toolchain capabilities of a proxy build that is not upstream's: `openmp_multicore` builds OpenMP target regions
# for the host CPU (the -mp=multicore proxy, bible Harness Contract), where upstream builds them for GPU offload.
# A recipe with faithful: true may not bind a toolchain that declares one (Design Principle 4).
PROXY_CAPABILITIES = frozenset({"openmp_multicore"})
# Toolchains a recipe with faithful: true may not bind, each with the reason [DESIGN] (task P16.2; PHASE-NOTES All
# Phases, the faithful check): a build upstream never used. Here alone a registry name decides, not a capability:
# gcc-native declares only diagnostics and emits_warnings, which upstream's toolchains declare too, so no capability
# tells it apart, and its module (lassi/toolchains/gcc.py) lies outside this task. A capability declared there, read
# here in place of the name, is the follow-up.
NOT_UPSTREAM_TOOLCHAINS: Mapping[str, str] = MappingProxyType(
    {
        "gcc-native": "a host g++ -fopenmp build upstream never used",
        # gcc-native's build with the OpenMP-construct tag (task P17.40): as gcc-native, not upstream's.
        "gcc-openmp": "a host g++ -fopenmp build upstream never used",
    }
)
# Toolchain capabilities of a build whose programs upstream ran on a GPU: `openmp_offload` builds OpenMP target
# regions for GPU offload (-mp=gpu). A recipe with faithful: true may not pair such a toolchain with an executor that
# declares runs_code and not takes_device [DESIGN] (task P16.2), native and ttsim among them: such an executor gives
# its runs no GPU (in the sandbox only a run of the gpu executor, which takes a device section, sees a device node;
# bible Sandbox, gpu program runs), so the run is not upstream's, whatever the binary then does on the host. A
# compile-only executor runs nothing, and may be paired.
# Known cases, not complete: the check closes the two cases PHASE-NOTES All Phases states (gcc-native; nvcpp-cc80
# with the native executor) and what the capability rule above reaches. A faithful recipe still loads that runs a
# CUDA build (nvcc-sm80, capability cuda) or a HIP build (hipcc-gfx942, hip) on an executor with no GPU, or that binds
# nvcpp-cc80, an NVIDIA offload build, with the gpu executor on a rocm device (tests/core/test_faithful_bindings.py
# pins each).
OFFLOAD_CAPABILITIES = frozenset({"openmp_offload"})


class RecipeError(ValueError):
    """A recipe that cannot load; the message names the file (when there is one) and the dotted key path."""


@dataclass(frozen=True)
class Recipe:
    """A loaded recipe: `data` is the resolved mapping (treat it as read-only) and `recipe_hash` its sha256."""

    name: str
    path: Path
    chain: tuple[str, ...]
    data: dict[str, Any]
    canonical_yaml: str
    recipe_hash: str
    bindings: tuple[Binding, ...]


class _SchemaError(Exception):
    """A schema problem, reported as a RecipeError once the file it belongs to is known."""


def _required_message(path: str) -> str:
    """Return the message for a required choice that is missing or null."""
    return f"{path} is a required choice with no value; set it in the recipe or one it extends"


def _join(path: str, key: object) -> str:
    """Return the dotted path of `key` inside the mapping at `path`."""
    return f"{path}.{key}" if path else str(key)


def _describe(value: Any) -> str:
    """Say what a YAML value is, for error messages."""
    if value is None:
        return "null"
    kinds = {bool: "boolean", int: "integer", float: "number", str: "string", list: "list", dict: "mapping"}
    text = repr(value)
    if len(text) > 60:
        text = text[:57] + "..."
    return f"the {kinds.get(type(value), type(value).__name__)} {text}"


def _check(value: Any, spec: _Spec, path: str) -> None:
    """Check one value against its spec: a null is a required choice with no value, never a default."""
    if value is None:
        raise _SchemaError(_required_message(path))
    spec.check(value, path)


def _expect(value: Any, kind: type, expected: str, path: str) -> None:
    """Raise a type problem unless `value` is an instance of `kind`."""
    if not isinstance(value, kind):
        raise _SchemaError(f"{path} must be {expected}, not {_describe(value)}")


@dataclass(frozen=True)
class _Leaf:
    """A single value: `accepts` tests it and `expected` says in words what fits.

    With `shown` false, a refusal never quotes the value, since it may be a
    key (Agent Rule 12): the message says what fits and that the value is not
    shown.
    """

    expected: str
    accepts: Callable[[Any], bool]
    shown: bool = True

    def keys(self, value: Any, path: str) -> None:
        """A leaf holds no keys."""

    def check(self, value: Any, path: str) -> None:
        """Raise a type problem unless the value fits."""
        if self.accepts(value):
            return
        if not self.shown:
            raise _SchemaError(f"{path} must be {self.expected}; the value is not shown, since it may be a key")
        raise _SchemaError(f"{path} must be {self.expected}, not {_describe(value)}")


@dataclass(frozen=True)
class _Fields:
    """A mapping whose keys are the listed ones; any other key is unknown.

    A key is optional unless REQUIRED names its dotted path or `required` lists
    it. `required` serves the entries of open maps (an agent, a judge), whose
    paths REQUIRED cannot name.
    """

    fields: Mapping[str, _Spec]
    required: frozenset[str] = field(default_factory=frozenset)

    def spec_for(self, key: Any, path: str) -> _Spec:
        """Return the spec of `key`, or raise an unknown-key problem listing the keys allowed here."""
        spec = self.fields.get(key) if isinstance(key, str) else None
        if spec is None:
            raise _SchemaError(f"unknown key {_join(path, key)}; allowed here: {', '.join(sorted(self.fields))}")
        return spec

    def keys(self, value: Any, path: str) -> None:
        """Refuse unknown keys at any depth; values of the wrong type are left for the type check."""
        if isinstance(value, dict):
            for key, item in value.items():
                self.spec_for(key, path).keys(item, _join(path, key))

    def check(self, value: Any, path: str) -> None:
        """Check that the value is a mapping of allowed keys whose values fit their specs, with every required key."""
        _expect(value, dict, "a mapping", path)
        for key, item in value.items():
            _check(item, self.spec_for(key, path), _join(path, key))
        missing = [key for key in self.fields if key in self.required and key not in value]
        if missing:
            raise _SchemaError(_required_message(_join(path, missing[0])))


@dataclass(frozen=True)
class _OpenMap:
    """A mapping with any string keys, every value fitting one spec."""

    values: _Spec

    def keys(self, value: Any, path: str) -> None:
        """Refuse keys that are not strings and unknown keys inside the values."""
        if isinstance(value, dict):
            for key, item in value.items():
                _expect(key, str, "a string key", _join(path, key))
                self.values.keys(item, _join(path, key))

    def check(self, value: Any, path: str) -> None:
        """Check that the value is a mapping with string keys whose values fit the spec."""
        _expect(value, dict, "a mapping", path)
        for key, item in value.items():
            _expect(key, str, "a string key", _join(path, key))
            _check(item, self.values, _join(path, key))


@dataclass(frozen=True)
class _ListOf:
    """A list whose items all fit one spec; `non_empty` refuses an empty list."""

    items: _Spec
    non_empty: bool = False

    def keys(self, value: Any, path: str) -> None:
        """Refuse unknown keys inside the items."""
        if isinstance(value, list):
            for index, item in enumerate(value):
                self.items.keys(item, f"{path}[{index}]")

    def check(self, value: Any, path: str) -> None:
        """Check that the value is a list (non-empty when required) whose items fit the spec."""
        _expect(value, list, "a non-empty list" if self.non_empty else "a list", path)
        if self.non_empty and not value:
            raise _SchemaError(f"{path} must be a non-empty list, not an empty one")
        for index, item in enumerate(value):
            _check(item, self.items, f"{path}[{index}]")


@dataclass(frozen=True)
class _KindSection:
    """A mapping with a required `kind: str` naming a component; its other keys are that component's config.

    Config keys are checked against the bound component's config_keys when the
    bindings are checked, not here. Config values are strings, finite numbers,
    booleans, lists, and mappings with string keys, with no null anywhere.
    """

    def keys(self, value: Any, path: str) -> None:
        """Config keys are checked later, against the bound component."""

    def check(self, value: Any, path: str) -> None:
        """Check that the section is a mapping with a string kind, string keys, and no null anywhere in its config."""
        _expect(value, dict, "a mapping with a kind", path)
        if "kind" not in value:
            raise _SchemaError(_required_message(_join(path, "kind")))
        _check(value["kind"], _STR, _join(path, "kind"))
        for key, item in value.items():
            _expect(key, str, "a string key", _join(path, key))
            if key != "kind":
                _check_free(item, _join(path, key))


@dataclass(frozen=True)
class _ExecutorSection:
    """The executor section: a kind section (the single form) or a mapping from language to executor.

    A mapping with `kind`, and an empty mapping, are checked as a kind
    section. Any other mapping is the per-language form: each key a
    language, each value an Executor name (a string) or a kind section,
    checked at executor.<language>; a value that is neither is refused there.
    """

    def keys(self, value: Any, path: str) -> None:
        """Config keys are checked later, against each bound component."""

    def check(self, value: Any, path: str) -> None:
        """Check the section in whichever form it has; refuse an entry that is neither a name nor a kind section."""
        if _executor_form(value) != _PER_LANGUAGE:
            _KIND.check(value, path)
            return
        for language, entry in value.items():
            where = _join(path, language)
            _expect(language, str, "a string key", where)
            if entry is None:
                raise _SchemaError(_required_message(where))
            if isinstance(entry, str):
                continue
            if isinstance(entry, dict) and "kind" in entry:
                _KIND.check(entry, where)
                continue
            raise _SchemaError(
                f"{where} must be an Executor name or a kind section {{kind: <name>, ...}}, not {_describe(entry)} "
                f"(an executor section without {_join(path, 'kind')} binds one executor per language)"
            )


# The two forms of the executor section (_executor_form).
_SINGLE = "single"
_PER_LANGUAGE = "per-language"


def _executor_form(value: Any) -> str | None:
    """Return the form of an executor section: _SINGLE, _PER_LANGUAGE, or None when it is no mapping or is empty.

    A mapping with `kind` is the single form; a non-empty mapping without
    it is the per-language form.
    """
    if not isinstance(value, dict) or not value:
        return None
    return _SINGLE if "kind" in value else _PER_LANGUAGE


def executor_languages(data: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the languages a resolved recipe binds an Executor for, sorted; () for the single form.

    The recipe's Executor bindings follow this order in the per-language
    form, one per language.
    """
    section = data.get("executor")
    return tuple(sorted(section)) if _executor_form(section) == _PER_LANGUAGE else ()


def _check_free(value: Any, path: str) -> None:
    """Check a free-form config value: plain YAML data with string keys and no null anywhere.

    Other YAML types (sets, ordered pairs, dates, binary) are refused: a set of
    mixed types dumps in an order that changes from process to process, and
    pairs reload as lists, so either would break the recipe hash or the reload
    of the resolved file.
    """
    if value is None:
        raise _SchemaError(_required_message(path))
    if isinstance(value, dict):
        for key, item in value.items():
            _expect(key, str, "a string key", _join(path, key))
            _check_free(item, _join(path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_free(item, f"{path}[{index}]")
    elif not (isinstance(value, (str, bool)) or _is_number(value)):
        raise _SchemaError(f"{path} must be a string, finite number, boolean, list, or mapping, not {_describe(value)}")


_Spec = _Leaf | _Fields | _OpenMap | _ListOf | _KindSection | _ExecutorSection


def _is_int(value: Any) -> bool:
    """Return True for an int that is not a bool."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    """Return True for an int that is not a bool, or a finite float.

    A NaN never equals itself, so a reloaded recipe would not equal the one
    saved. An infinity is refused because every number a recipe sets is a
    sampling value, a limit, or a step: the sandbox always bounds a run, so
    "no limit" is never a value to write.
    """
    if isinstance(value, float):
        return math.isfinite(value)
    return _is_int(value)


def _is_env_name(value: Any) -> bool:
    """Return True for an environment variable name: uppercase letters, digits, and '_', no leading digit, <= 64."""
    return isinstance(value, str) and _ENV_NAME_TEXT.fullmatch(value) is not None


def _is_revision(value: Any) -> bool:
    """Return True for a full commit id: 40 lowercase hexadecimal characters."""
    return isinstance(value, str) and _REVISION_TEXT.fullmatch(value) is not None


def _is_seed(value: Any) -> bool:
    """Return True for an int that is not a bool, from 0 to 2**32 - 1."""
    return _is_int(value) and 0 <= value < _SEED_LIMIT


def _is_corrections(value: Any) -> bool:
    """Return True for a correction cap: an int of at least 0, or UNCAPPED."""
    if isinstance(value, str):
        return value == UNCAPPED
    return _is_int(value) and value >= 0


def _is_between(value: Any, bounds: tuple[int, int]) -> bool:
    """Return True for an int that is not a bool, from bounds[0] to bounds[1], both included."""
    return _is_int(value) and bounds[0] <= value <= bounds[1]


def _is_retrieval_prompts(value: Any) -> bool:
    """Return True for a non-empty list of distinct names among RETRIEVAL_PROMPTS."""
    if not isinstance(value, list) or not value:
        return False
    if not all(isinstance(item, str) and item in RETRIEVAL_PROMPTS for item in value):
        return False
    return len(set(value)) == len(value)


def _is_lookup_core(value: Any) -> bool:
    """Return True for a non-empty list of distinct header paths, each with no blank, `<`, or `>` (_HEADER_TEXT)."""
    if not isinstance(value, list) or not value:
        return False
    if not all(isinstance(item, str) and _HEADER_TEXT.fullmatch(item) for item in value):
        return False
    return len(set(value)) == len(value)


def _is_lookup_prompts(value: Any) -> bool:
    """Return True for a non-empty list of distinct names among LOOKUP_PROMPTS."""
    if not isinstance(value, list) or not value:
        return False
    if not all(isinstance(item, str) and item in LOOKUP_PROMPTS for item in value):
        return False
    return len(set(value)) == len(value)


def _is_examples_prompts(value: Any) -> bool:
    """Return True for a list of distinct names among EXAMPLES_PROMPTS that holds EXAMPLES_GENERATE."""
    return _holds_generate(value, EXAMPLES_PROMPTS, EXAMPLES_GENERATE)


def _is_skeleton_prompts(value: Any) -> bool:
    """Return True for a list of distinct names among SKELETON_PROMPTS that holds SKELETON_GENERATE."""
    return _holds_generate(value, SKELETON_PROMPTS, SKELETON_GENERATE)


def _holds_generate(value: Any, names: Sequence[str], generate: str) -> bool:
    """Return True for a list of distinct names among `names` that holds `generate`."""
    if not isinstance(value, list) or generate not in value:
        return False
    if not all(isinstance(item, str) and item in names for item in value):
        return False
    return len(set(value)) == len(value)


_STR = _Leaf("a string", lambda value: isinstance(value, str))
_BOOL = _Leaf("true or false", lambda value: isinstance(value, bool))
_INT = _Leaf("an integer", _is_int)
_NUMBER = _Leaf("a finite number", _is_number)
_NUMBER_OR_STR = _Leaf("a finite number or a string", lambda value: _is_number(value) or isinstance(value, str))
_AT_LEAST_ONE = _Leaf("an integer of at least 1", lambda value: _is_int(value) and value >= 1)
_CORRECTIONS = _Leaf(f"an integer of at least 0 or {UNCAPPED!r}", _is_corrections)
_POSITIVE = _Leaf("a finite number above zero", lambda value: _is_number(value) and value > 0)
# model.api_key_env names the variable that holds a key, never the key (Agent Rule 12; task P17.3). A value that does
# not fit is not shown, since it may be a key pasted in by mistake.
_ENV_NAME = _Leaf(
    "an environment variable name: uppercase letters, digits, and '_', not starting with a digit, "
    "at most 64 characters",
    _is_env_name,
    shown=False,
)
# model.revision and model.seed: an in-process model's pin and sampling seed (task P17.4).
_REVISION = _Leaf("a full commit id: 40 lowercase hexadecimal characters", _is_revision)
_SEED = _Leaf(f"an integer from 0 to {_SEED_LIMIT - 1}", _is_seed)
# model.recording_sha256: the sha256 of model.recording's bytes, which the loader enters (task P16.13).
_SHA256 = _Leaf(
    "a sha256: 64 lowercase hexadecimal characters",
    lambda value: isinstance(value, str) and _SHA256_TEXT.fullmatch(value) is not None,
)
# llm.sampling.reasoning_effort (OQ-080): one of REASONING_EFFORTS as written.
_REASONING_EFFORT = _Leaf(
    f"one of {', '.join(REASONING_EFFORTS)}", lambda value: isinstance(value, str) and value in REASONING_EFFORTS
)
_STRINGS = _ListOf(_STR)
_KIND = _KindSection()
_EXECUTOR = _ExecutorSection()
# The retrieval section (task P17.22; RETRIEVAL): a mapping with every key required.
_RETRIEVAL = _Fields(
    {
        "index": _Leaf(
            f"a corpus name matching {_INDEX_TEXT.pattern}, a directory under assets/rag holding MANIFEST.yaml",
            lambda value: isinstance(value, str) and _INDEX_TEXT.fullmatch(value) is not None,
        ),
        "k": _Leaf(
            f"an integer from {RETRIEVAL_K[0]} to {RETRIEVAL_K[1]}", lambda value: _is_between(value, RETRIEVAL_K)
        ),
        "max_chars": _Leaf(
            f"an integer from {RETRIEVAL_MAX_CHARS[0]} to {RETRIEVAL_MAX_CHARS[1]}",
            lambda value: _is_between(value, RETRIEVAL_MAX_CHARS),
        ),
        "prompts": _Leaf(
            f"a non-empty list of distinct names among {', '.join(RETRIEVAL_PROMPTS)}", _is_retrieval_prompts
        ),
    },
    required=frozenset({"index", "k", "max_chars", "prompts"}),
)
# The lookup section (task P17.26; LOOKUP): a mapping with max_chars and prompts required; core and nearest are
# checked against prompts by _check_lookup.
_LOOKUP = _Fields(
    {
        "max_chars": _Leaf(
            f"an integer from {LOOKUP_MAX_CHARS[0]} to {LOOKUP_MAX_CHARS[1]}",
            lambda value: _is_between(value, LOOKUP_MAX_CHARS),
        ),
        "prompts": _Leaf(f"a non-empty list of distinct names among {', '.join(LOOKUP_PROMPTS)}", _is_lookup_prompts),
        LOOKUP_CORE: _Leaf(
            "a non-empty list of distinct header paths as the pack's include lines name them, such as "
            "tt-metalium/host_api.hpp (no blank, '<', or '>')",
            _is_lookup_core,
        ),
        LOOKUP_NEAREST: _Leaf(
            f"an integer from {LOOKUP_NEAREST_RANGE[0]} to {LOOKUP_NEAREST_RANGE[1]}",
            lambda value: _is_between(value, LOOKUP_NEAREST_RANGE),
        ),
    },
    required=frozenset({"max_chars", "prompts"}),
)
# The examples section (OQ-083 (b); EXAMPLES): a mapping with every key required.
_EXAMPLES = _Fields(
    {
        "pack": _Leaf(
            f"a pack name matching {_INDEX_TEXT.pattern}, a directory under assets/examples holding MANIFEST.yaml",
            lambda value: isinstance(value, str) and _INDEX_TEXT.fullmatch(value) is not None,
        ),
        "max_chars": _Leaf(
            f"an integer from {EXAMPLES_MAX_CHARS[0]} to {EXAMPLES_MAX_CHARS[1]}",
            lambda value: _is_between(value, EXAMPLES_MAX_CHARS),
        ),
        "prompts": _Leaf(
            f"a list of distinct names among {', '.join(EXAMPLES_PROMPTS)} that holds {EXAMPLES_GENERATE}",
            _is_examples_prompts,
        ),
    },
    required=frozenset({"pack", "max_chars", "prompts"}),
)
# The skeleton section (OQ-096 (a); SKELETON): name and prompts required; sha256 is the loader's record of the
# skeleton's digest, which a recipe may state (_bind_skeleton).
_SKELETON = _Fields(
    {
        "name": _Leaf(
            f"a skeleton name matching {_INDEX_TEXT.pattern}, a directory under assets/skeletons holding MANIFEST.yaml",
            lambda value: isinstance(value, str) and _INDEX_TEXT.fullmatch(value) is not None,
        ),
        "prompts": _Leaf(
            f"a list of distinct names among {', '.join(SKELETON_PROMPTS)} that holds {SKELETON_GENERATE}",
            _is_skeleton_prompts,
        ),
        SKELETON_SHA256_KEY: _SHA256,
    },
    required=frozenset({"name", "prompts"}),
)
# The dialect_lookup section (task P17.41; DIALECT_LOOKUP): defs and max_chars required, nearest optional.
_DIALECT_LOOKUP = _Fields(
    {
        "defs": _Leaf(
            f"a definitions tree name matching {_INDEX_TEXT.pattern}, a directory under assets/context holding "
            "MANIFEST.yaml",
            lambda value: isinstance(value, str) and _INDEX_TEXT.fullmatch(value) is not None,
        ),
        "max_chars": _Leaf(
            f"an integer from {DIALECT_MAX_CHARS[0]} to {DIALECT_MAX_CHARS[1]}",
            lambda value: _is_between(value, DIALECT_MAX_CHARS),
        ),
        DIALECT_NEAREST: _Leaf(
            f"an integer from {DIALECT_NEAREST_RANGE[0]} to {DIALECT_NEAREST_RANGE[1]}",
            lambda value: _is_between(value, DIALECT_NEAREST_RANGE),
        ),
    },
    required=frozenset({"defs", "max_chars"}),
)
# The source_mlir section (task P17.41; SOURCE_MLIR): both keys required.
_SOURCE_MLIR = _Fields(
    {
        "toolchain": _Leaf(
            f"a Toolchain name matching {_INDEX_TEXT.pattern} that declares lowers_source",
            lambda value: isinstance(value, str) and _INDEX_TEXT.fullmatch(value) is not None,
        ),
        "max_chars": _Leaf(
            f"an integer from {SOURCE_MLIR_MAX_CHARS[0]} to {SOURCE_MLIR_MAX_CHARS[1]}",
            lambda value: _is_between(value, SOURCE_MLIR_MAX_CHARS),
        ),
    },
    required=frozenset({"toolchain", "max_chars"}),
)

# Every key a run recipe may hold and its type. Inside a fixed mapping every key is optional unless REQUIRED
# names it, and any other key is unknown. A train-only key (TRAIN_SCHEMA's base_model, method, trainer, data, and
# the rest) is unknown here; bench, adversary, runs_root, and extends are in both schemas.
# Entries of lists and open maps state every key they need: a direction both ends, an agent its model, and a judge
# its model, rubric, and use (an agent's enabled and a judge's mode are optional).
SCHEMA = _Fields(
    {
        "extends": _STR,  # recipe files only; never kept in the resolved mapping
        "project": _STR,  # the first trial id segment; defaults to the loaded file's recipe name
        "faithful": _BOOL,
        "fixes": _Fields({name: _BOOL for name in FIXES}),
        "extract": _Fields({name: _BOOL for name in EXTRACT}),  # opt-in leniencies; absent means off
        # reasoning_effort is an opt-in, absent means none is sent (OQ-080)
        "llm": _Fields(
            {
                "sampling": _Fields(
                    {"temperature": _NUMBER, "top_p": _NUMBER, "max_tokens": _INT, REASONING_EFFORT: _REASONING_EFFORT}
                )
            }
        ),
        # opt-ins off when absent; RESTART_AFTER is the one int among them (task P17.25)
        "loop": _Fields(
            {"max_corrections": _CORRECTIONS, **{name: _BOOL for name in LOOP_OPT_INS}, RESTART_AFTER: _AT_LEAST_ONE}
        ),
        "trials": _Fields({"n": _AT_LEAST_ONE}),
        "runs_root": _STR,
        "sandbox": _Fields({"network": _BOOL, "wall_s": _NUMBER_OR_STR, "mem_gb": _NUMBER}),
        "report": _Fields({"trial_md": _BOOL, "parquet": _BOOL}),
        "bench": _Fields({"suite": _STR, "split": _STR, "items": _ListOf(_STR, non_empty=True)}),
        "directions": _ListOf(
            _Fields({"source": _STR, "target": _STR}, required=frozenset({"source", "target"})), non_empty=True
        ),
        "prompts": _STR,
        "context": _STRINGS,
        RETRIEVAL: _RETRIEVAL,  # opt-in documentation passages (task P17.22); absent means off
        LOOKUP: _LOOKUP,  # opt-in declarations from the context pack (task P17.26); absent means off
        EXAMPLES: _EXAMPLES,  # opt-in worked examples from a pack (OQ-083 (b)); absent means off
        SKELETON: _SKELETON,  # opt-in host program skeleton (OQ-096 (a)); absent means off
        DIALECT_LOOKUP: _DIALECT_LOOKUP,  # opt-in dialect definitions for verify errors (task P17.41); absent means off
        SOURCE_MLIR: _SOURCE_MLIR,  # opt-in source lowered to MLIR in the generation prompt (task P17.41); absent: off
        "toolchain": _OpenMap(_STR),
        "stages": _ListOf(_STR, non_empty=True),
        "executor": _EXECUTOR,
        "oracle": _KIND,
        "profiler": _KIND,
        "adversary": _KIND,
        # device: a device section (P17.2); base_url, timeout_s, api_key_env: a served backend's settings (P17.3);
        # revision, seed: an in-process model's settings (P17.4); recording and its sha256: the replay backend's
        # recording (P16.13)
        "model": _Fields(
            {
                "backend": _STR,
                "id": _STR,
                "device": _KIND,
                "base_url": _STR,
                "timeout_s": _POSITIVE,
                "api_key_env": _ENV_NAME,
                "revision": _REVISION,
                "seed": _SEED,
                RECORDING_KEY: _STR,
                RECORDING_SHA256_KEY: _SHA256,
                "checkpoint": _STR,  # a trained checkpoint hf_local serves, and its content digest (2026-10-23)
                "checkpoint_sha256": _SHA256,
            }
        ),
        "arms": _STRINGS,
        "metrics": _STRINGS,
        "refine": _Fields({"counter_start": _NUMBER, "counter_step": _NUMBER, "max_iters": _INT}),
        "score": _STR,
        "agents": _OpenMap(_Fields({"model": _STR, "enabled": _BOOL}, required=frozenset({"model"}))),
        "judges": _OpenMap(
            _Fields(
                {"model": _STR, "rubric": _STR, "mode": _STR, "use": _STR},
                required=frozenset({"model", "rubric", "use"}),
            )
        ),
    }
)

# Choices every resolved recipe must state with a value; the loader never picks one.
REQUIRED: tuple[str, ...] = (
    "llm.sampling.temperature",
    "llm.sampling.top_p",
    "loop.max_corrections",
    "trials.n",
    "runs_root",
    "sandbox.network",
    "sandbox.wall_s",
    "sandbox.mem_gb",
    "report.trial_md",
    "report.parquet",
    "bench.suite",
    "bench.split",
    "directions",
    "stages",
    "executor.kind",  # in the single form; the per-language form names a kind in each entry instead
)

# The model section's own keys; every other model key is the backend's config (_bindings). recording_sha256 is
# the loader's record of model.recording's bytes, not a backend setting (task P16.13).
_MODEL_OWN = ("backend", "id", RECORDING_SHA256_KEY)

# Kind sections in binding order, with the interface each binds and the capabilities the section itself needs of the
# component it binds (Binding.needs, checked by check_bindings): the adversary section binds an Agent that must
# declare proposes_inputs (task P16.2), while an Agent bound to a role under agents needs none. Known cases, not
# complete: the rule holds for a run recipe (load_recipe). load_train_recipe binds only trainer.kind, so a train
# recipe's adversary section (method adversarial, part (b)) is bound and checked by neither rule until task P16.12
# binds it through the entry here; lassi train refuses the key until then.
_KIND_SECTIONS: tuple[tuple[str, str, frozenset[str]], ...] = (
    ("executor", "Executor", frozenset()),
    ("oracle", "Oracle", frozenset()),
    ("profiler", "Profiler", frozenset()),
    ("adversary", "Agent", frozenset({PROPOSES_INPUTS})),
)

# The train recipe's names (bible Training Module, Algorithms and Weight Modes; the train.yaml block's comments),
# and its three data sources: synthetic fixtures (data.synthetic), bench items (bench), and a built corpus
# (data.corpus, a manifest under assets/train/; lassi.train.data load_corpus). prompt_tuning is the frozen-weights mode.
METHODS: tuple[str, ...] = ("sft", "rft", "dpo", "grpo", "gspo", "ppo", "adversarial")
WEIGHT_MODES: tuple[str, ...] = ("full", "lora", "qlora", "dora", "prompt_tuning")
EPISODES: tuple[str, ...] = ("single_turn", "multi_turn")
DATA_SOURCES: tuple[str, ...] = ("synthetic", "bench", "corpus")
# The one split a train recipe may read (Agent Rule 5).
TRAIN_SPLIT = "train"
# A synthetic fixture's file name: one plain name ending in .jsonl, read from the fixture directory.
_SYNTHETIC_TEXT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.jsonl")
_CORPUS_TEXT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# The Trainer's declarations the loader reads: the recipe key whose value must be among them, the attribute, and
# what the value is called in a refusal.
_DECLARATIONS = (("method", "methods", "method"), ("weights", "weight_modes", "weight mode"))


def _one_of(names: Sequence[str]) -> _Leaf:
    """Return a leaf that accepts exactly one of `names` as written."""
    return _Leaf(f"one of {', '.join(names)}", lambda value: isinstance(value, str) and value in names)


def _is_synthetic_name(value: Any) -> bool:
    """Return True for a fixture file name: one plain name ending in .jsonl."""
    return isinstance(value, str) and _SYNTHETIC_TEXT.fullmatch(value) is not None


def _is_corpus_name(value: Any) -> bool:
    """Return True for a corpus name: one plain name, the directory of its manifest under assets/train/."""
    return isinstance(value, str) and _CORPUS_TEXT.fullmatch(value) is not None


# Every key a train recipe may hold: the bible's train.yaml block, plus extends, runs_root (where the train tree
# goes), trainer (a kind section binding the Trainer, with its device and config keys), and data (the synthetic
# fixture). A run-only key (llm, stages, model, and the rest) is unknown here, as a train-only key is in SCHEMA.
TRAIN_SCHEMA = _Fields(
    {
        "extends": _STR,  # recipe files only; never kept in the resolved mapping
        "runs_root": _STR,
        "base_model": _STR,
        "method": _one_of(METHODS),
        "episode": _one_of(EPISODES),
        "weights": _one_of(WEIGHT_MODES),
        "lora": _Fields({"r": _AT_LEAST_ONE, "targets": _STR}),
        "bench": _Fields(
            {"suite": _STR, "split": _STR, "items": _ListOf(_STR, non_empty=True)},
            required=frozenset({"suite", "split"}),
        ),
        "reward": _Fields({"profile": _STR, "executor": _STR, "cache": _BOOL}),
        "rollout": _Fields({"engine": _STR, "group_size": _AT_LEAST_ONE}),
        "adversary": _KIND,
        "export": _Fields({"merge": _BOOL, "fxb": _STR, "register_as": _STR}),
        "trainer": _KIND,
        "data": _Fields(  # exactly one of the two keys (_check_train_source)
            {
                "synthetic": _Leaf("a fixture file name: one plain name ending in .jsonl", _is_synthetic_name),
                "corpus": _Leaf("a corpus name: one plain name, a directory under assets/train", _is_corpus_name),
            }
        ),
    }
)

# Choices every resolved train recipe must state with a value; the loader never picks one.
TRAIN_REQUIRED: tuple[str, ...] = ("base_model", "method", "weights", "trainer.kind")


def default_roots() -> tuple[Path, ...]:
    """Return the directories searched for a recipe named in `extends`: the repository's projects/ directory."""
    return (Path(__file__).resolve().parents[2] / "projects",)


def load_recipe(path: Path, *, roots: Sequence[Path] | None = None, registry: Registry | None = None) -> Recipe:
    """Load, resolve, and check the recipe at `path`; construct no component.

    `roots` (default: default_roots()) are searched in order for a recipe named
    in `extends`; `registry` (default: DEFAULT_REGISTRY) holds the components
    the recipe may bind. Every failure raises RecipeError.
    """
    path = Path(path)
    search = tuple(Path(root) for root in (default_roots() if roots is None else roots))
    chain = _load_chain(path, search)
    data = _resolve(chain)
    try:
        SCHEMA.check(data, "")
        _check_required(data)
    except _SchemaError as exc:
        raise RecipeError(f"{path}: {exc}") from None
    _check_extract(path, data)
    _check_loop(path, data)
    _check_retrieval(path, data)
    _check_lookup(path, data)
    _check_examples(path, data)
    _bind_skeleton(path, data)
    _check_sampling(path, data)
    _check_unfaithful_sections(path, data)
    bindings = _bindings(data)
    registry = DEFAULT_REGISTRY if registry is None else registry
    try:
        check_bindings(bindings, registry)
    except RegistryError as exc:
        raise RecipeError(f"{path}: {exc}") from exc
    _check_faithful_bindings(path, data, bindings, registry)
    _check_devices(path, bindings, registry)
    bindings = _bind_recording(path, data, bindings)
    canonical = yaml.safe_dump(data, sort_keys=True, default_flow_style=False, allow_unicode=False, width=4096)
    return Recipe(
        name=_recipe_name(path),
        path=path,
        chain=tuple(_recipe_name(file) for file, _ in chain),
        data=data,
        canonical_yaml=canonical,
        recipe_hash=hashlib.sha256(canonical.encode("ascii")).hexdigest(),
        bindings=tuple(bindings),
    )


def load_train_recipe(path: Path, *, roots: Sequence[Path] | None = None, registry: Registry | None = None) -> Recipe:
    """Load, resolve, and check the train recipe at `path`; construct no component and ask no framework.

    Each file of the extends chain is read against TRAIN_SCHEMA and merged as
    load_recipe merges; nothing is materialized. The checks, in order: types
    and TRAIN_REQUIRED; one data source (_check_train_source); bench.split
    (_check_train_split); the Trainer binding (check_bindings); the Trainer's
    declarations (_check_trainer_declares); and its device section
    (_check_devices). The hash is load_recipe's. `roots` and `registry`
    default as for load_recipe. Every failure raises RecipeError.
    """
    path = Path(path)
    search = tuple(Path(root) for root in (default_roots() if roots is None else roots))
    chain = _load_chain(path, search, TRAIN_SCHEMA)
    data = _fresh(_merged(chain))
    try:
        TRAIN_SCHEMA.check(data, "")
        _check_required(data, TRAIN_REQUIRED)
    except _SchemaError as exc:
        raise RecipeError(f"{path}: {exc}") from None
    _check_train_source(path, data)
    _check_train_split(path, data)
    trainer = data["trainer"]
    config = {key: value for key, value in trainer.items() if key != "kind"}
    bindings = [Binding(interface="Trainer", name=trainer["kind"], where="trainer.kind", config=_fresh(config))]
    registry = DEFAULT_REGISTRY if registry is None else registry
    try:
        check_bindings(bindings, registry)
    except RegistryError as exc:
        raise RecipeError(f"{path}: {exc}") from exc
    _check_trainer_declares(path, data, bindings[0], registry)
    _check_devices(path, bindings, registry)
    canonical = yaml.safe_dump(data, sort_keys=True, default_flow_style=False, allow_unicode=False, width=4096)
    return Recipe(
        name=_recipe_name(path),
        path=path,
        chain=tuple(_recipe_name(file) for file, _ in chain),
        data=data,
        canonical_yaml=canonical,
        recipe_hash=hashlib.sha256(canonical.encode("ascii")).hexdigest(),
        bindings=tuple(bindings),
    )


def train_data_source(data: Mapping[str, Any]) -> str:
    """Return the data source of a loaded train recipe's mapping: "bench", "corpus", or "synthetic" (DATA_SOURCES)."""
    if "bench" in data:
        return "bench"
    return "corpus" if "corpus" in data.get("data", {}) else "synthetic"


def resolved_data(path: Path, *, roots: Sequence[Path] | None = None) -> dict[str, Any]:
    """Return the resolved mapping of the recipe at `path` (steps 1 to 6) without the type or binding checks.

    It lets a caller explain why a recipe that load_recipe refused cannot
    run (for example arms without a model, which fails the binding check).
    A file that cannot be read or merged raises RecipeError, as in load_recipe.
    """
    search = tuple(Path(root) for root in (default_roots() if roots is None else roots))
    return _resolve(_load_chain(Path(path), search))


def resolved_yaml(recipe: Recipe) -> str:
    """Return the resolved recipe text a run saves: two header comment lines, then the canonical YAML."""
    chain = " -> ".join(_ascii(name) for name in recipe.chain)
    return (
        f"# Resolved recipe {_ascii(recipe.name)} (chain: {chain})\n"
        f"# recipe_hash: {recipe.recipe_hash} is the sha256 of the YAML below these comment lines\n"
        f"{recipe.canonical_yaml}"
    )


def _ascii(text: str) -> str:
    """Return `text` with non-ASCII and control characters escaped, for a one-line ASCII comment."""
    return ascii(text)[1:-1]


def _recipe_name(path: Path) -> str:
    """Return a recipe's name: the file stem, or the directory name for a file called recipe.yaml.

    The directory is found on the absolute path with `..` collapsed (abspath,
    which follows no symlink), so `proj/variants/../recipe.yaml` is named proj.
    """
    return Path(os.path.abspath(path)).parent.name if path.name == "recipe.yaml" else path.stem


# ---------------------------------------------------------------------------
# Reading files and following extends (steps 1 to 3)


def _load_chain(path: Path, roots: Sequence[Path], schema: _Fields = SCHEMA) -> list[tuple[Path, dict[str, Any]]]:
    """Read `path` and every recipe it extends against `schema`; return (file, own mapping) from the root down."""
    chain: list[tuple[Path, dict[str, Any]]] = []
    seen: set[str] = set()
    current: Path | None = path
    while current is not None:
        try:
            identity = os.path.normcase(str(current.resolve()))
        except (OSError, ValueError) as exc:  # ValueError: a NUL character, which no OS allows in a path
            raise RecipeError(f"{current}: cannot read the recipe file ({exc})") from exc
        if identity in seen:
            names = " -> ".join([*(_recipe_name(file) for file, _ in chain), _recipe_name(current)])
            raise RecipeError(f"{chain[-1][0]}: extends makes a cycle: {names}")
        seen.add(identity)
        data = _read_recipe_file(current, schema)
        chain.append((current, data))
        current = _parent_path(current, data, roots)
    chain.reverse()
    return chain


def _read_recipe_file(path: Path, schema: _Fields = SCHEMA) -> dict[str, Any]:
    """Read one recipe file as UTF-8, parse it strictly, and refuse keys `schema` does not list."""
    try:
        text = path.read_bytes().decode("utf-8")
    except OSError as exc:
        raise RecipeError(f"{path}: cannot read the recipe file ({exc.strerror or exc})") from exc
    except UnicodeDecodeError as exc:
        raise RecipeError(f"{path}: not UTF-8 text ({exc.reason} at byte {exc.start})") from exc
    data = _parse_yaml(text, path)
    if data is None:
        raise RecipeError(f"{path}: the file is empty; a recipe is a mapping of keys")
    if not isinstance(data, dict):
        raise RecipeError(f"{path}: the top level must be a mapping of keys, not {_describe(data)}")
    try:
        schema.keys(data, "")
    except _SchemaError as exc:
        raise RecipeError(f"{path}: {exc}") from None
    return data


# What the safe constructors raise, instead of a yaml.YAMLError, for a scalar its explicit tag cannot read
# (`!!int abc`, `!!bool maybe`, `!!timestamp nope`, `!!int ''`).
_TAG_ERRORS = (AttributeError, LookupError, TypeError, ValueError)


def _parse_yaml(text: str, path: Path) -> Any:
    """Parse one YAML document with the safe loader.

    It refuses repeated keys, self-containing aliases, deep nesting, and
    aliases that expand to more than _MAX_VALUES values. A YAML error's
    message is _yaml_problem's, which quotes no source line.
    """
    try:
        loader = yaml.SafeLoader(text)
    except yaml.YAMLError as exc:
        raise RecipeError(f"{path}: not valid YAML: {_yaml_problem(exc)}") from None
    loader.name = path.name
    try:
        node = loader.get_single_node()
        if node is None:
            return None
        _check_nodes(node, loader, "", 0, frozenset(), {})
        return _construct(loader, node, whole=True)
    except _SchemaError as exc:
        raise RecipeError(f"{path}: {exc}") from None
    except yaml.YAMLError as exc:
        raise RecipeError(f"{path}: not valid YAML: {_yaml_problem(exc)}") from None
    except RecursionError as exc:
        raise RecipeError(f"{path}: not valid YAML: mappings and lists nest too deep to parse") from exc
    finally:
        loader.dispose()


def _yaml_problem(exc: yaml.YAMLError) -> str:
    """Return a YAML error as its context, its problem, and the problem's line and column, quoting no source line.

    PyYAML's own text (str(exc)) quotes the offending line, which could show
    a mistyped value such as a key written into api_key_env (task P17.11), so
    only exc.context, exc.problem, and the line and column (both from 1) of
    exc.problem_mark, else exc.context_mark, are used, for every key, since
    a syntax error comes before any key is known. Known cases, not
    complete: a problem text can itself hold source text, such as an
    unknown tag's name, and _construct's message for a scalar its explicit
    tag cannot read quotes the constructor's error, which can hold the
    scalar. A reader error gives its reason and character position, and any
    other error its class name.
    """
    if isinstance(exc, yaml.reader.ReaderError):
        return f"{exc.reason} at character {exc.position}"
    if not isinstance(exc, yaml.MarkedYAMLError):
        return type(exc).__name__
    parts = [str(part) for part in (exc.context, exc.problem) if part]
    mark = exc.problem_mark or exc.context_mark
    if mark is not None:
        parts.append(f"at line {mark.line + 1}, column {mark.column + 1}")
    return " ".join(parts) or type(exc).__name__


def _construct(loader: yaml.SafeLoader, node: yaml.Node, *, whole: bool = False) -> Any:
    """Build the value of `node` (the whole document when `whole`), reporting a scalar its explicit tag cannot read.

    Only the constructor calls sit inside this catch, so a bug elsewhere in
    the loader is never reported as a tag problem.
    """
    try:
        return loader.construct_document(node) if whole else loader.construct_object(node, deep=True)
    except _TAG_ERRORS as exc:
        raise _SchemaError(f"not valid YAML: a value does not fit its explicit tag ({exc!r})") from exc


_MERGE_TAG = "tag:yaml.org,2002:merge"
_VALUE_TAG = "tag:yaml.org,2002:value"  # the plain key `=`, which the safe loader reads as the string "="
_MERGE_KEY = object()  # stands for `<<` among a mapping's keys, so a second merge key counts as a repeat

# The deepest a value may sit inside mappings and lists. Real recipes nest four or five levels; the bound keeps
# every recursive step of loading (and yaml.safe_dump) far from Python's recursion limit, aliases included.
_MAX_DEPTH = 64

# The most values one file may hold once its aliases are expanded. Real recipes hold a few hundred. Loading
# copies every alias out (the canonical YAML has none), so without a bound a few hundred bytes of aliases to
# aliases would take hours and gigabytes to load (the billion laughs pattern).
_MAX_VALUES = 100_000


def _check_nodes(
    node: yaml.Node,
    loader: yaml.SafeLoader,
    path: str,
    depth: int,
    above: frozenset[int],
    measured: dict[int, tuple[int, int]],
) -> tuple[int, int]:
    """Refuse a repeated key, an alias to a mapping or list that contains it, and a document too deep or too big.

    `depth` counts the mappings and lists enclosing `node`, and `above` holds
    them. `measured` maps each node already checked to its height (0 for a
    scalar) and its expanded size (1 for a scalar; 1 plus its values' sizes for
    a mapping or list). A node reached again through an alias is checked once,
    but counts toward the depth at which it is reached and toward the size
    again. Returns the node's height and expanded size.
    """
    if id(node) in above:
        raise _SchemaError(f"{path} is an alias to a mapping or list that contains it")
    if depth > _MAX_DEPTH:
        raise _SchemaError(f"mappings and lists nest more than {_MAX_DEPTH} levels deep at {path}")
    measure = measured.get(id(node))
    if measure is None:
        inner = above | {id(node)}
        height, size = 0, 1
        for child_path, child in _child_nodes(node, loader, path):
            child_height, child_size = _check_nodes(child, loader, child_path, depth + 1, inner, measured)
            height, size = max(height, 1 + child_height), size + child_size
            if size > _MAX_VALUES:
                raise _SchemaError(
                    f"{path or 'the top level'} holds more than {_MAX_VALUES} values once its aliases are expanded;"
                    " a recipe needs far fewer"
                )
        measure = measured[id(node)] = (height, size)
    if depth + measure[0] > _MAX_DEPTH:
        raise _SchemaError(f"mappings and lists nest more than {_MAX_DEPTH} levels deep at {path or 'the top level'}")
    return measure


def _child_nodes(node: yaml.Node, loader: yaml.SafeLoader, path: str) -> list[tuple[str, yaml.Node]]:
    """Return the value nodes inside a list or mapping node with their paths, refusing a key a mapping repeats.

    A merge key (<<) counts as a key too: a second one in the same mapping would
    silently override the first, so it is a repeat like any other.
    """
    if isinstance(node, yaml.SequenceNode):
        return [(f"{path}[{index}]", child) for index, child in enumerate(node.value)]
    if not isinstance(node, yaml.MappingNode):
        return []
    children: list[tuple[str, yaml.Node]] = []
    keys: set[Any] = set()
    for key_node, value_node in node.value:
        if key_node.tag == _MERGE_TAG:
            key, shown, child_path = _MERGE_KEY, "<<", path
        elif isinstance(key_node, yaml.ScalarNode):
            key = "=" if key_node.tag == _VALUE_TAG else _construct(loader, key_node)
            shown, child_path = key, _join(path, key)
        else:
            continue  # a mapping or list as a key: construction refuses it as unhashable
        if key in keys:
            where = f"{_join(path, shown)} at line {key_node.start_mark.line + 1}"
            raise _SchemaError(f"duplicate key {where}; YAML would keep only the last value")
        keys.add(key)
        children.append((child_path, value_node))
    return children


def _parent_path(file: Path, data: Mapping[str, Any], roots: Sequence[Path]) -> Path | None:
    """Return the file that `file` extends, or None when it extends nothing.

    A name (no "/" and no .yaml or .yml suffix) is looked up in each root as
    <root>/<name>.yaml, then <root>/<name>/recipe.yaml. Anything else is a path
    relative to the extending file's directory, written with "/" separators.
    Every part must match the disk exactly, so a recipe that loads on one OS
    loads on all of them.
    """
    if "extends" not in data:
        return None
    value = data["extends"]
    if not isinstance(value, str) or not value.strip():
        raise RecipeError(f"{file}: extends must be a recipe name or a path to a recipe file, not {_describe(value)}")
    if "\\" in value or value.startswith("/") or PureWindowsPath(value).drive:
        raise RecipeError(
            f"{file}: extends {value!r} must be a recipe name or a relative path with / separators,"
            " so it finds the same file on every OS"
        )
    if value in (".", ".."):
        raise RecipeError(
            f"{file}: extends {value!r} names a directory, not a recipe name, and a name must stay inside a root;"
            f" write the path to the file instead, such as {value}/recipe.yaml"
        )
    if "/" in value or value.endswith((".yaml", ".yml")):
        candidates = [(file.parent, value)]
    else:
        candidates = [(root, spelled) for root in roots for spelled in (f"{value}.yaml", f"{value}/recipe.yaml")]
    for base, relative in candidates:
        if _is_file_as_spelled(base, relative):
            return base / relative
    tried = ", ".join(str(base / relative) for base, relative in candidates) or "nowhere (no roots to search)"
    raise RecipeError(
        f"{file}: extends {value!r}, but no recipe file was found; tried {tried}"
        " (each part must match the disk exactly, including case)"
    )


def _is_file_as_spelled(base: Path, relative: str) -> bool:
    """Return True when `relative` names a file under `base` and each of its parts exists exactly as written.

    Windows and macOS match names regardless of case and skip `name/..` without
    checking that `name` exists; Linux does neither. Checking each part against
    its directory listing gives the Linux answer everywhere.
    """
    current = base
    for part in PurePosixPath(relative).parts:
        if part != "..":
            try:
                if part not in os.listdir(current):
                    return False
            except OSError:
                return False
        current = current / part
    return current.is_file()


# ---------------------------------------------------------------------------
# Resolving and checking (steps 4 to 9)


def _resolve(chain: Sequence[tuple[Path, Mapping[str, Any]]]) -> dict[str, Any]:
    """Merge the chain from the root ancestor down, then materialize defaults and apply faithful (steps 4 to 6)."""
    data = _fresh(_merged(chain))
    _materialize_defaults(data, _recipe_name(chain[-1][0]))
    _apply_faithful(data)
    return data


def _merged(chain: Sequence[tuple[Path, Mapping[str, Any]]]) -> dict[str, Any]:
    """Merge each file's own keys but extends over the chain so far, from the root ancestor down (step 4)."""
    merged: dict[str, Any] = {}
    for _, own in chain:
        merged = _overlay(merged, {key: value for key, value in own.items() if key != "extends"})
    return merged


def _materialize_defaults(data: dict[str, Any], name: str) -> None:
    """Fill in the defaults the chain leaves unset (step 5).

    `project` becomes `name`, the recipe name of the file being loaded (never
    an ancestor's), `faithful` false, and every fix on. A value of the wrong
    type (or null) is left as it is for the type check to report.
    """
    data.setdefault("project", name)
    data.setdefault("faithful", False)
    fixes = data.setdefault("fixes", {})
    if isinstance(fixes, dict):
        for name in FIXES:
            fixes.setdefault(name, True)


def _apply_faithful(data: dict[str, Any]) -> None:
    """When the resolved `faithful` is true, uncap the loop and turn every fix off, whatever the chain said (step 6).

    This reproduces upstream behavior (bible, Project Recipes Notes; Design
    Principle 4). A loop or fixes value of the wrong type is left for the type
    check to report.
    """
    if data["faithful"] is not True:
        return
    loop = data.setdefault("loop", {})
    if isinstance(loop, dict):
        loop["max_corrections"] = UNCAPPED
    fixes = data["fixes"]
    if isinstance(fixes, dict):
        fixes.update({name: False for name in FIXES})


def _overlay(merged: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    """Merge one file's keys over the chain so far (step 4).

    A kind section whose child names a different kind replaces the inherited
    section instead of merging with it: the old component's config keys do not
    apply to the new one (the bible's tier-1 compile-only executor, `{kind: none}`,
    in place of a GPU executor with a host). Restating the same kind merges as usual.
    A train recipe's trainer section follows the same rule (task P17.8); run
    recipes never hold one. An executor section that switches between the
    single and the per-language form replaces the inherited one too, and two
    per-language sections merge entry by entry (_merge_executors).
    """
    sections = [section for section, _, _ in _KIND_SECTIONS] + ["trainer"]
    switched = {section for section in sections if _names_other_kind(merged.get(section), child.get(section))}
    inherited, own = merged.get("executor"), child.get("executor")
    forms = (_executor_form(inherited), _executor_form(own))
    if None not in forms and forms[0] != forms[1]:
        switched.add("executor")
    kept = {key: value for key, value in merged.items() if key not in switched}
    result = _merge(kept, child)
    if forms == (_PER_LANGUAGE, _PER_LANGUAGE):
        result["executor"] = _merge_executors(inherited, own)
    return result


def _merge_executors(inherited: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    """Merge two per-language executor sections: the child's entries win, entry by entry.

    Two mappings for one language merge unless the child names another
    kind, which replaces the inherited entry (its config keys belong to the
    old component), so a child mapping without kind adds or changes config
    keys of the inherited kind section; any other child entry replaces the
    inherited one.
    """
    merged = dict(inherited)
    for language, entry in child.items():
        old = merged.get(language)
        both_kinds = isinstance(old, dict) and isinstance(entry, dict)
        merged[language] = _merge(old, entry) if both_kinds and not _names_other_kind(old, entry) else entry
    return merged


def _names_other_kind(inherited: Any, value: Any) -> bool:
    """Return True when both are mappings with a kind and the child's kind differs from the inherited one."""
    both_kinds = isinstance(inherited, dict) and isinstance(value, dict) and "kind" in inherited and "kind" in value
    return both_kinds and inherited["kind"] != value["kind"]


def _merge(parent: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    """Return `parent` overlaid by `child`: two mappings merge recursively, and anything else from the child wins.

    A `device` value is one choice, so the child's replaces the inherited one
    whole at any depth, never merged key by key.
    """
    merged = dict(parent)
    for key, value in child.items():
        inherited = merged.get(key)
        both_maps = isinstance(inherited, dict) and isinstance(value, dict) and key != DEVICE_KEY
        merged[key] = _merge(inherited, value) if both_maps else value
    return merged


def _fresh(value: Any) -> Any:
    """Return a copy of nested mappings and lists that shares no object, so the canonical YAML has no aliases."""
    if isinstance(value, dict):
        return {key: _fresh(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_fresh(item) for item in value]
    return value


def _check_required(data: Mapping[str, Any], required: Sequence[str] = REQUIRED) -> None:
    """Raise a required-choice problem for the first `required` key the resolved mapping does not state.

    executor.kind is required of the single form only: the per-language form
    names a kind in each entry, which the schema check already required.
    """
    for dotted in required:
        if dotted == "executor.kind" and executor_languages(data):
            continue
        value: Any = data
        for part in dotted.split("."):
            if not isinstance(value, dict) or part not in value:
                raise _SchemaError(_required_message(dotted))
            value = value[part]


def _check_extract(path: Path, data: Mapping[str, Any]) -> None:
    """Refuse an EXTRACT leniency set true under faithful: true or with fixes.fence_tag off, naming its key.

    None is upstream behavior (Design Principle 4, Agent Rule 4), and each
    reads this project's FILE-block format, which a reply has only with the
    fence_tag fix on; faithful: true turns that fix off too, but its refusal
    comes first, since it says why. Runs after the type check, so the
    section, when present, maps known names to bools, and fixes.fence_tag is
    set. A leniency set false, or absent, is never refused.
    """
    for name, on in sorted(data.get("extract", {}).items()):
        if on is not True:
            continue
        key = f"extract.{name}"
        if data["faithful"] is True:
            raise RecipeError(
                f"{path}: {key} is true, but faithful: true reproduces upstream, which has no such leniency; "
                f"a recipe with faithful: true may not set it, so remove {key} or set faithful: false"
            )
        if data["fixes"]["fence_tag"] is not True:
            raise RecipeError(
                f"{path}: {key} is true, but fixes.fence_tag is off, so a reply is read as upstream reads it, from "
                f"its first fenced block, and no FILE line is read; turn fixes.fence_tag on or remove {key}"
            )


def _check_loop(path: Path, data: Mapping[str, Any]) -> None:
    """Refuse a LOOP_OPT_INS key under faithful: true, naming it, and a RESTART_AFTER past the correction cap.

    Upstream corrects only compile errors and failed runs, never asks a
    fresh generation, and runs on no simulator (Design Principle 4, Agent
    Rule 4), so under faithful: true a bool key set true (correct_on_mismatch,
    correct_on_sim_gap), or RESTART_AFTER set at all, is refused. Runs
    after the type check, so the loop section maps known names to their
    types. A bool set false, or a key absent, is never refused.
    RESTART_AFTER, an int of at least 1 by then, may not pass the resolved
    loop.max_corrections; an uncapped loop takes any such int.
    """
    loop = data["loop"]
    for name in sorted(LOOP_OPT_INS):
        value = loop.get(name)
        if not (value is True or (name == RESTART_AFTER and value is not None)) or data["faithful"] is not True:
            continue
        key = f"loop.{name}"
        raise RecipeError(
            f"{path}: {key} is {'true' if value is True else value}, but faithful: true reproduces upstream, whose "
            "loop corrects only compile errors and failed runs and never restarts; a recipe with faithful: true may "
            f"not set it, so remove {key} or set faithful: false"
        )
    after, cap = loop.get(RESTART_AFTER), loop["max_corrections"]
    if after is not None and cap != UNCAPPED and after > cap:
        key = f"loop.{RESTART_AFTER}"
        fits = f"set {key} from 1 to {cap}, or remove it" if cap else f"the loop asks no correction, so remove {key}"
        raise RecipeError(
            f"{path}: {key} is {after}, past loop.max_corrections, {cap}, so no restart could come within the cap; "
            f"{fits}"
        )


def _check_retrieval(path: Path, data: Mapping[str, Any]) -> None:
    """Refuse a `retrieval` section under faithful: true, inherited or not: upstream's prompts carry no passages.

    Runs after the type check, so a section that is present is a mapping
    of its four keys (Design Principle 4, Agent Rule 4).
    """
    if RETRIEVAL in data and data["faithful"] is True:
        raise RecipeError(
            f"{path}: {RETRIEVAL} is set, but faithful: true reproduces upstream, whose prompts carry no "
            f"documentation passages; a recipe with faithful: true may not set it, so remove {RETRIEVAL} or set "
            "faithful: false"
        )


def _check_lookup(path: Path, data: Mapping[str, Any]) -> None:
    """Refuse a `lookup` section under faithful: true, and core or nearest that its prompts do not call for.

    Runs after the type check, so a section that is present is a mapping
    of its keys, each of its type. Upstream's prompts carry no declarations,
    so the section is refused under faithful: true, inherited or not
    (Design Principle 4, Agent Rule 4). core is required when prompts lists
    generate and refused otherwise; nearest is refused unless prompts lists
    correct, since only a correction reads errors.
    """
    section = data.get(LOOKUP)
    if not isinstance(section, Mapping):
        return
    prompts = f"{LOOKUP}.prompts"
    if LOOKUP_GENERATE in section["prompts"] and LOOKUP_CORE not in section:
        raise RecipeError(
            f"{path}: {_required_message(f'{LOOKUP}.{LOOKUP_CORE}')}: {prompts} lists {LOOKUP_GENERATE}, whose "
            f"prompt shows no error to name a declaration, so {LOOKUP}.{LOOKUP_CORE} names the pack's header parts "
            f"it carries"
        )
    if LOOKUP_CORE in section and LOOKUP_GENERATE not in section["prompts"]:
        raise RecipeError(
            f"{path}: {LOOKUP}.{LOOKUP_CORE} is set, but {prompts} does not list {LOOKUP_GENERATE}, the only prompt "
            f"that carries the header parts it names; list {LOOKUP_GENERATE} or remove {LOOKUP}.{LOOKUP_CORE}"
        )
    if LOOKUP_NEAREST in section and LOOKUP_CORRECT not in section["prompts"]:
        raise RecipeError(
            f"{path}: {LOOKUP}.{LOOKUP_NEAREST} is set, but {prompts} does not list {LOOKUP_CORRECT}, the only "
            f"prompt whose errors name identifiers; list {LOOKUP_CORRECT} or remove {LOOKUP}.{LOOKUP_NEAREST}"
        )
    if data["faithful"] is True:
        raise RecipeError(
            f"{path}: {LOOKUP} is set, but faithful: true reproduces upstream, whose prompts carry no declarations "
            f"from a context pack; a recipe with faithful: true may not set it, so remove {LOOKUP} or set "
            "faithful: false"
        )


def _check_examples(path: Path, data: Mapping[str, Any]) -> None:
    """Refuse an `examples` section under faithful: true, inherited or not: upstream's prompts carry no examples.

    Runs after the type check, so a section that is present is a mapping
    of its three keys (Design Principle 4, Agent Rule 4).
    """
    if EXAMPLES in data and data["faithful"] is True:
        raise RecipeError(
            f"{path}: {EXAMPLES} is set, but faithful: true reproduces upstream, whose prompts carry no worked "
            f"examples; a recipe with faithful: true may not set it, so remove {EXAMPLES} or set faithful: false"
        )


# What upstream's prompts lack, for each opt-in section refused under faithful: true by _check_unfaithful_sections.
_UNFAITHFUL_SECTIONS = {
    DIALECT_LOOKUP: "carry no dialect definitions",
    SOURCE_MLIR: "show no source lowered to MLIR",
}


def _check_unfaithful_sections(path: Path, data: Mapping[str, Any]) -> None:
    """Refuse a dialect_lookup or source_mlir section under faithful: true, inherited or not (task P17.41).

    Runs after the type check, so a section that is present is a mapping of
    its keys (Design Principle 4, Agent Rule 4). A recipe without either is
    never refused here.
    """
    if data["faithful"] is not True:
        return
    for section, lacks in _UNFAITHFUL_SECTIONS.items():
        if section in data:
            raise RecipeError(
                f"{path}: {section} is set, but faithful: true reproduces upstream, whose prompts {lacks}; a recipe "
                f"with faithful: true may not set it, so remove {section} or set faithful: false"
            )


def _bind_skeleton(path: Path, data: dict[str, Any]) -> None:
    """Check the `skeleton` section and enter the skeleton's sha256 at skeleton.sha256 (OQ-096 (a); SKELETON).

    Runs after the type check, so a section that is present is a mapping of
    its keys. In order: the section is refused under faithful: true,
    inherited or not, since upstream's prompts carry no skeleton (Design
    Principle 4, Agent Rule 4); the skeleton named is read from
    <assets root>/skeletons/<name> (lassi.prompts.skeletons load_skeleton:
    its files must match its manifest and its block must render), each
    problem refused naming skeleton.name; a stated skeleton.sha256 must be
    the skeleton's; and every direction's target must be the skeleton's
    language. The sha256 then enters the section, so the resolved recipe
    and its hash name the skeleton's bytes (Agent Rule 10).
    """
    section = data.get(SKELETON)
    if not isinstance(section, dict):
        return
    if data["faithful"] is True:
        raise RecipeError(
            f"{path}: {SKELETON} is set, but faithful: true reproduces upstream, whose prompts carry no host program "
            f"skeleton; a recipe with faithful: true may not set it, so remove {SKELETON} or set faithful: false"
        )
    key = f"{SKELETON}.name"
    try:
        skeleton = skeleton_files.load_skeleton(skeleton_files.skeleton_dir(section["name"]))
    except ValueError as error:
        raise RecipeError(
            f"{path}: {key} {section['name']!r} names no skeleton that loads: {error}; name a directory under "
            f"assets/{skeleton_files.SKELETONS_DIR}/ that holds {skeleton_files.MANIFEST} and its files"
        ) from None
    stated = section.get(SKELETON_SHA256_KEY)
    if stated is not None and stated != skeleton.sha256:
        raise RecipeError(
            f"{path}: {SKELETON}.{SKELETON_SHA256_KEY} is {stated}, but the skeleton {skeleton.name}'s sha256 is "
            f"{skeleton.sha256}; it records the skeleton's bytes, so correct it or remove it"
        )
    others = sorted({direction["target"] for direction in data["directions"]} - {skeleton.language})
    if others:
        raise RecipeError(
            f"{path}: {SKELETON} names {skeleton.name}, a skeleton of {skeleton.language!r} programs, but the "
            f"directions also target {', '.join(repr(name) for name in others)}; a skeleton goes only before prompts "
            f"for its own language, so run those directions from a recipe without {SKELETON}"
        )
    section[SKELETON_SHA256_KEY] = skeleton.sha256


def _check_sampling(path: Path, data: Mapping[str, Any]) -> None:
    """Refuse llm.sampling.reasoning_effort under faithful: true, inherited or not: upstream sends no such field.

    Runs after the type check, so the key, when present, holds one of
    REASONING_EFFORTS (Design Principle 4, Agent Rule 4).
    """
    if REASONING_EFFORT in data["llm"]["sampling"] and data["faithful"] is True:
        raise RecipeError(
            f"{path}: {REASONING_EFFORT_KEY} is {data['llm']['sampling'][REASONING_EFFORT]}, but faithful: true "
            "reproduces upstream, whose requests carry no reasoning effort; a recipe with faithful: true may not set "
            f"it, so remove {REASONING_EFFORT_KEY} or set faithful: false"
        )


def _check_faithful_bindings(
    path: Path, data: Mapping[str, Any], bindings: Sequence[Binding], registry: Registry
) -> None:
    """Refuse a faithful recipe that binds a toolchain, or a toolchain and executor pair, upstream never ran.

    A faithful label on such a recipe would misstate what was reproduced
    (Design Principle 4, Agent Rule 4). Toolchain by toolchain in binding
    order, the first refusal is raised, naming the binding:

    - a toolchain that declares a PROXY_CAPABILITIES capability: upstream
      builds OpenMP target regions for GPU offload, and a proxy build checks
      outputs only (bible Harness Contract); the capability decides;
    - a toolchain named in NOT_UPSTREAM_TOOLCHAINS [DESIGN] (task P16.2);
    - a toolchain that declares an OFFLOAD_CAPABILITIES capability, when the
      executor of its language (the single form's, or the language's own)
      declares runs_code and not takes_device [DESIGN] (task P16.2), native
      and ttsim among them; the message also names that executor's binding.

    Recipes that are not faithful may bind any of them. Runs after
    check_bindings, so every name is registered.

    Known cases, not complete: a faithful recipe still loads that runs a
    CUDA build (nvcc-sm80) or a HIP build (hipcc-gfx942) on an executor with
    no GPU, or binds nvcpp-cc80 with the gpu executor on a rocm device (the
    comments at OFFLOAD_CAPABILITIES).
    """
    if data["faithful"] is not True:
        return
    for binding in bindings:
        if binding.interface != "Toolchain":
            continue
        problem = _unfaithful_toolchain(binding, bindings, registry)
        if problem is not None:
            raise RecipeError(f"{path}: {problem}")


def _unfaithful_toolchain(binding: Binding, bindings: Sequence[Binding], registry: Registry) -> str | None:
    """Return why a faithful recipe may not bind the Toolchain `binding` (see _check_faithful_bindings), or None."""
    who = f"{binding.where} binds Toolchain {binding.name!r}"
    capabilities = registry.get("Toolchain", binding.name).capabilities
    proxy = sorted(PROXY_CAPABILITIES & capabilities)
    if proxy:
        return (
            f"{who}, which declares {', '.join(proxy)} (a proxy build, not upstream's); a recipe with faithful: true "
            "may not bind it, so bind upstream's toolchain or set faithful: false"
        )
    if binding.name in NOT_UPSTREAM_TOOLCHAINS:
        return (
            f"{who} ({NOT_UPSTREAM_TOOLCHAINS[binding.name]}); a recipe with faithful: true may not bind it, so bind "
            "upstream's toolchain or set faithful: false"
        )
    offload = sorted(OFFLOAD_CAPABILITIES & capabilities)
    executor = _executor_for(bindings, binding.where.removeprefix("toolchain."))
    if not offload or executor is None:
        return None
    runs = registry.get("Executor", executor.name).capabilities
    if RUNS_CODE not in runs or TAKES_DEVICE in runs:
        return None
    return (
        f"{who}, which declares {', '.join(offload)} (upstream ran such programs on a GPU), and {executor.where} binds "
        f"Executor {executor.name!r}, which runs programs with no device section, so no GPU is exposed to them; a "
        "recipe with faithful: true may not pair them, so bind an executor with a GPU device section or a "
        "compile-only executor for that language, or set faithful: false"
    )


def _executor_for(bindings: Sequence[Binding], language: str) -> Binding | None:
    """Return the Executor binding that runs `language`'s programs (the single form's, or its own), else None."""
    paths = ("executor.kind", f"executor.{language}", f"executor.{language}.kind")
    return next((item for item in bindings if item.interface == "Executor" and item.where in paths), None)


def _check_devices(path: Path, bindings: Sequence[Binding], registry: Registry) -> None:
    """Refuse a component that declares takes_device without a device section, and any section parse_device refuses.

    Runs after check_bindings, so every name is registered and a device key
    sits only under a component that takes one. The loader never picks a
    device; a per-language entry written as a bare name cannot carry one.
    """
    for binding in bindings:
        entry = registry.get(binding.interface, binding.name)
        if TAKES_DEVICE not in entry.capabilities:
            continue
        where = device_path(binding)
        if DEVICE_KEY not in binding.config:
            bare = binding.where.startswith("executor.") and not binding.where.endswith(".kind")
            hint = "; a bare executor name cannot carry one, so write a kind section {kind, device}" if bare else ""
            raise RecipeError(
                f"{path}: {where} is a required choice with no value; {entry.interface} {entry.name!r} "
                f"({binding.where}) takes a device section {{kind, indices}} and the loader never picks one{hint}"
            )
        try:
            parse_device(binding.config[DEVICE_KEY], where)
        except DeviceSectionError as error:
            raise RecipeError(f"{path}: {error}") from None


def _bind_recording(path: Path, data: dict[str, Any], bindings: Sequence[Binding]) -> list[Binding]:
    """Check each section's recording, enter its sha256 there, and give the section's binding what it needs (step 9).

    Each file is recording_file's, and <section>.recording_sha256 becomes the
    sha256 of its bytes (_section_recording). Runs after check_bindings, so a
    component that does not list recording in its config_keys never reaches
    it with one. Every refusal names the key.

    model.recording (task P16.13): the LLMBackend binding's recording becomes
    the checked file's absolute path, which the runner passes as
    factory(model.id, recording=...); the resolved mapping keeps the path as
    written, so the recipe hash names no host path. A model section with no
    backend binds no LLMBackend: its recording is checked and hashed here,
    no binding carries it, and the runner refuses the recipe for the missing
    model.backend, as it does with any other model key. The backend reads
    the file again when the runner builds it, and the runner compares the
    sha256 of the bytes it read with model.recording_sha256.

    adversary.recording (task P16.4), an adversary's own replay backend's:
    the Agent binding's config gains adversary.recording_sha256 and keeps
    the recording as written, since the adversary stage copies that config
    into every trial's AdversaryRecord, so no host path enters a record. The
    Agent resolves the path with recording_file itself and compares the
    sha256 of the bytes its backend read (lassi.agents.llm).
    """
    model = _section_recording(path, "model", data.get("model", {}))
    adversary = _section_recording(path, _ADVERSARY, data.get(_ADVERSARY, {}))
    found = []
    for binding in bindings:
        if model is not None and binding.interface == "LLMBackend":
            binding = replace(binding, config=_fresh({**binding.config, RECORDING_KEY: str(model[0])}))
        elif adversary is not None and binding.where == f"{_ADVERSARY}.kind":
            binding = replace(binding, config=_fresh({**binding.config, RECORDING_SHA256_KEY: adversary[1]}))
        found.append(binding)
    return found


def _section_recording(path: Path, section: str, values: Any) -> tuple[Path, str] | None:
    """Check `section`.recording, set `section`.recording_sha256 in `values`, and return the file and that sha256.

    None when the section holds no recording; one that states the sha256
    without a recording is refused. A recipe that states the sha256 (a
    resolved recipe does) must state that of the file's bytes.
    """
    key, sha_key = f"{section}.{RECORDING_KEY}", f"{section}.{RECORDING_SHA256_KEY}"
    if not isinstance(values, dict) or RECORDING_KEY not in values:
        if isinstance(values, Mapping) and RECORDING_SHA256_KEY in values:
            raise RecipeError(
                f"{path}: {sha_key} is set without {key}; it records the sha256 of the recording's bytes, so set "
                f"{key} or remove it"
            )
        return None
    value = values[RECORDING_KEY]
    file = recording_file(value, f"{path}: {key}")
    try:
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
    except OSError as exc:
        raise RecipeError(f"{path}: {key} {value!r} cannot be read ({exc.strerror or exc})") from None
    stated = values.get(RECORDING_SHA256_KEY)
    if stated is not None and stated != digest:
        raise RecipeError(
            f"{path}: {sha_key} is {stated}, but the bytes of {key} {value!r} have the sha256 {digest}; the "
            "recording changed since the recipe stated it, so restore those bytes or state the new sha256"
        )
    values[RECORDING_SHA256_KEY] = digest
    return file, digest


def recording_file(value: object, where: str) -> Path:
    """Return the recording file `value` names, resolved; refuse it, naming `where`, unless it fits the rule below.

    `where` names the key in each refusal (such as "<recipe>: model.recording"
    or "adversary.recording"); a refusal is a RecipeError, a ValueError.
    `value` is a str, a path relative to REPO (read here, at each call)
    written with / separators: not empty, not absolute (no leading / or \\
    and no drive), no backslash, and no empty, `.`, or `..` segment, so the
    path in the recipe hash has no redundant segment (a link inside
    tests/fixtures/ could still give a file a second path; none is
    committed). It must lie under tests/fixtures/ as written, name a file
    whose every part matches the disk exactly (case included, as Linux reads
    it), and still lie inside tests/fixtures/ once every link is followed,
    so a link cannot lead a recording out of it (tasks P16.13 and P16.4).
    """
    fixtures = "/".join(RECORDING_DIR) + "/"
    if not isinstance(value, str):
        raise RecipeError(f"{where} must be a string, a path under {fixtures}, not {value!r}")
    if not value:
        raise RecipeError(f"{where} is empty; name a recording file under {fixtures}")
    where = f"{where} {value!r}"
    if value.startswith("/") or PureWindowsPath(value).anchor:
        raise RecipeError(f"{where} is an absolute path; write it relative to the repository root, under {fixtures}")
    if "\\" in value:
        raise RecipeError(f"{where} must use / separators, so it names the same file on every OS")
    parts = tuple(value.split("/"))
    if any(part in ("", ".", "..") for part in parts):
        raise RecipeError(
            f"{where} has an empty, '.', or '..' segment; write the file's plain path from the repository "
            "root, so the path in the recipe hash has no redundant segment"
        )
    if parts[: len(RECORDING_DIR)] != RECORDING_DIR or len(parts) <= len(RECORDING_DIR):
        raise RecipeError(f"{where} is not under {fixtures}; a recording is a test fixture and lives there")
    root = REPO
    if not _is_file_as_spelled(root, value):
        raise RecipeError(
            f"{where} names no file under the repository root {root} (each part must match the disk exactly, "
            "including case)"
        )
    try:
        file = root.joinpath(*parts).resolve(strict=True)
        inside = file.is_relative_to(root.joinpath(*RECORDING_DIR).resolve(strict=True))
    except (OSError, RuntimeError) as exc:
        raise RecipeError(f"{where} cannot be resolved ({exc})") from None
    if not inside:
        raise RecipeError(f"{where} resolves to {file}, outside {fixtures}; a link may not lead a recording out of it")
    return file


def _check_train_source(path: Path, data: Mapping[str, Any]) -> None:
    """Refuse a train recipe that names more than one data source (bench, data.synthetic, data.corpus) or none."""
    named = [key for key in ("bench", "data") if key in data]
    if len(named) != 1:
        found = "both" if named else "neither"
        raise RecipeError(
            f"{path}: a train recipe names exactly one data source, bench or data.synthetic (or data.corpus); this "
            f"one names {found}"
        )
    sources = [f"data.{key}" for key in ("synthetic", "corpus") if key in data.get("data", {})]
    if "data" in data and len(sources) != 1:
        found = " and ".join(sources) if sources else "neither"
        raise RecipeError(
            f"{path}: data.synthetic is a required choice with no value unless data.corpus names a corpus; a train "
            f"recipe names exactly one of data.synthetic and data.corpus, and this one names {found}"
        )


def _check_train_split(path: Path, data: Mapping[str, Any]) -> None:
    """Refuse a bench.split other than train; eval and unassigned with Agent Rule 5's words (lassi.bench.registry)."""
    if "bench" not in data:
        return
    split = data["bench"]["split"]
    if split in HELD_SPLITS:
        raise RecipeError(
            f"{path}: bench.split is {split!r}; the {split} split may never be used for "
            f"{FORBIDDEN_PURPOSES['train']} (Agent Rule 5), so a train recipe reads only the split {TRAIN_SPLIT!r}"
        )
    if split != TRAIN_SPLIT:
        raise RecipeError(f"{path}: bench.split is {split!r}; a train recipe reads only the split {TRAIN_SPLIT!r}")


def _check_trainer_declares(path: Path, data: Mapping[str, Any], binding: Binding, registry: Registry) -> None:
    """Refuse a Trainer without takes_device, a malformed declaration, or a recipe value it does not declare.

    The Trainer must declare takes_device, since a train recipe names its
    device with no default (bible Training Module, Compute). methods,
    weight_modes, and data_sources must be collections of strings; the
    recipe's method, weights, and data source must be among them. Each
    refusal names the Trainer, and the key or the attribute. Runs after
    check_bindings, so the Trainer is registered; nothing is built.
    """
    entry = registry.get(binding.interface, binding.name)
    who = f"{entry.interface} {entry.name!r} ({binding.where})"
    if TAKES_DEVICE not in entry.capabilities:
        raise RecipeError(
            f"{path}: {who} does not declare {TAKES_DEVICE!r}; a train recipe names its device at trainer.device "
            "with no default, so its Trainer must take one"
        )
    source = train_data_source(data)
    checks = [(key, data[key], attribute, noun) for key, attribute, noun in _DECLARATIONS]
    checks.append(("bench" if source == "bench" else f"data.{source}", source, "data_sources", "data source"))
    for key, value, attribute, noun in checks:
        declared = _declared_names(path, who, entry.factory, attribute)
        if value not in declared:
            carried = ", ".join(sorted(declared)) or "nothing"
            raise RecipeError(
                f"{path}: {key}: {who} does not carry out the {noun} {value!r}; it carries out: {carried}"
            )


def _declared_names(path: Path, who: str, factory: type, attribute: str) -> frozenset[str]:
    """Return a Trainer class's declared `attribute` as a frozenset; refuse one that is not a collection of strings."""
    value = getattr(factory, attribute, None)
    is_collection = isinstance(value, (list, tuple, set, frozenset))
    if not is_collection or not all(isinstance(item, str) for item in value):
        raise RecipeError(
            f"{path}: {who} declares {attribute} as {value!r}; it must be a collection of strings, the names it "
            "carries out"
        )
    return frozenset(value)


def _bindings(data: Mapping[str, Any]) -> list[Binding]:
    """Return the components the resolved mapping binds, in the order below; absent sections bind nothing.

    A per-language executor section binds one Executor per language, in
    executor_languages order, at executor.<language> for a name and
    executor.<language>.kind for a kind section. A kind section's binding
    carries the capabilities that section needs (_KIND_SECTIONS; needs, empty
    for every other binding). The LLMBackend's config holds
    every key of the model section but backend, id, and recording_sha256: its
    device section, when it has one, under `device`, any server setting
    (base_url, timeout_s, api_key_env; task P17.3), any in-process model
    setting (revision, seed; task P17.4), and the replay backend's recording
    as written (task P16.13), which _bind_recording then replaces with the
    checked file.
    """
    found: list[Binding] = []

    def bind(
        interface: str,
        name: str,
        where: str,
        config: Mapping[str, Any] | None = None,
        needs: frozenset[str] = frozenset(),
    ) -> None:
        config = _fresh(dict(config or {}))
        found.append(Binding(interface=interface, name=name, where=where, config=config, needs=needs))

    if "backend" in data.get("model", {}):
        model = data["model"]
        model_config = {key: value for key, value in model.items() if key not in _MODEL_OWN}
        bind("LLMBackend", model["backend"], "model.backend", model_config)
    for key in sorted(data.get("toolchain", {})):
        bind("Toolchain", data["toolchain"][key], f"toolchain.{key}")
    for section, interface, needs in _KIND_SECTIONS:
        if section == "executor" and executor_languages(data):
            for language in executor_languages(data):
                entry = data[section][language]
                if isinstance(entry, str):
                    bind(interface, entry, f"{section}.{language}", needs=needs)
                else:
                    config = {key: value for key, value in entry.items() if key != "kind"}
                    bind(interface, entry["kind"], f"{section}.{language}.kind", config, needs)
        elif section in data:
            config = {key: value for key, value in data[section].items() if key != "kind"}
            bind(interface, data[section]["kind"], f"{section}.kind", config, needs)
    for index, stage in enumerate(data.get("stages", [])):
        bind("Stage", stage, f"stages[{index}]")
    if "score" in data:
        bind("ScoreProfile", data["score"], "score")
    for role in sorted(data.get("agents", {})):
        bind("Agent", role, f"agents.{role}")
    for judge in sorted(data.get("judges", {})):
        bind("Judge", judge, f"judges.{judge}")
    return found
