"""Load run recipes: extends chains, defaults, faithful overrides, checks, and the recipe hash.

A project is a recipe, not code (bible, Design Principle 1). Loading a recipe:

1. parses each file strictly (a repeated key is an error, never a silent overwrite, and nesting is bounded);
2. refuses any key the schema below does not list, naming the file and dotted path;
3. follows `extends` to the root ancestor;
4. merges from the root down, the child winning (mappings merge, anything else is replaced, and a
   kind section that names a new kind replaces the inherited one);
5. materializes the defaults (`faithful: false`, every fix on);
6. applies the faithful overrides (bible, Project Recipes Notes; Design Principle 4);
7. checks types and required choices, never picking a value for a choice left open;
8. binds components and checks their capabilities without constructing any (Component Interfaces);
9. serializes the resolved mapping to canonical YAML and hashes it.

The canonical YAML, with a two-line header, is the resolved recipe every run saves
and can be rerun from (Design Principle 5).
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Mapping, Sequence

import yaml

from lassi.core.registry import DEFAULT_REGISTRY, Binding, Registry, RegistryError, check_bindings

# The value of loop.max_corrections that means no cap (upstream LASSI's loop; bible LASSI quirk table).
UNCAPPED = "uncapped"

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
}


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
    """A single value: `accepts` tests it and `expected` says in words what fits."""

    expected: str
    accepts: Callable[[Any], bool]

    def keys(self, value: Any, path: str) -> None:
        """A leaf holds no keys."""

    def check(self, value: Any, path: str) -> None:
        """Raise a type problem unless the value fits."""
        if not self.accepts(value):
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


_Spec = _Leaf | _Fields | _OpenMap | _ListOf | _KindSection


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


def _is_corrections(value: Any) -> bool:
    """Return True for a correction cap: an int of at least 0, or UNCAPPED."""
    if isinstance(value, str):
        return value == UNCAPPED
    return _is_int(value) and value >= 0


_STR = _Leaf("a string", lambda value: isinstance(value, str))
_BOOL = _Leaf("true or false", lambda value: isinstance(value, bool))
_INT = _Leaf("an integer", _is_int)
_NUMBER = _Leaf("a finite number", _is_number)
_NUMBER_OR_STR = _Leaf("a finite number or a string", lambda value: _is_number(value) or isinstance(value, str))
_AT_LEAST_ONE = _Leaf("an integer of at least 1", lambda value: _is_int(value) and value >= 1)
_CORRECTIONS = _Leaf(f"an integer of at least 0 or {UNCAPPED!r}", _is_corrections)
_STRINGS = _ListOf(_STR)
_KIND = _KindSection()

# Every key a run recipe may hold and its type. Inside a fixed mapping every key is optional unless REQUIRED
# names it, and any other key is unknown. Train recipes are not run recipes; their keys are unknown here.
# Entries of lists and open maps state every key they need: a direction both ends, an agent its model, and a judge
# its model, rubric, and use (an agent's enabled and a judge's mode are optional).
SCHEMA = _Fields(
    {
        "extends": _STR,  # recipe files only; never kept in the resolved mapping
        "faithful": _BOOL,
        "fixes": _Fields({name: _BOOL for name in FIXES}),
        "llm": _Fields({"sampling": _Fields({"temperature": _NUMBER, "top_p": _NUMBER, "max_tokens": _INT})}),
        "loop": _Fields({"max_corrections": _CORRECTIONS}),
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
        "toolchain": _OpenMap(_STR),
        "stages": _ListOf(_STR, non_empty=True),
        "executor": _KIND,
        "oracle": _KIND,
        "profiler": _KIND,
        "adversary": _KIND,
        "model": _Fields({"backend": _STR, "id": _STR}),
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
    "executor.kind",
)

# Kind sections in binding order, with the interface each binds.
_KIND_SECTIONS: tuple[tuple[str, str], ...] = (
    ("executor", "Executor"),
    ("oracle", "Oracle"),
    ("profiler", "Profiler"),
    ("adversary", "Agent"),
)


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
    bindings = _bindings(data)
    try:
        check_bindings(bindings, DEFAULT_REGISTRY if registry is None else registry)
    except RegistryError as exc:
        raise RecipeError(f"{path}: {exc}") from exc
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


def _load_chain(path: Path, roots: Sequence[Path]) -> list[tuple[Path, dict[str, Any]]]:
    """Read `path` and every recipe it extends; return (file, own mapping) from the root ancestor down."""
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
        data = _read_recipe_file(current)
        chain.append((current, data))
        current = _parent_path(current, data, roots)
    chain.reverse()
    return chain


def _read_recipe_file(path: Path) -> dict[str, Any]:
    """Read one recipe file as UTF-8, parse it strictly, and refuse keys the schema does not list."""
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
        SCHEMA.keys(data, "")
    except _SchemaError as exc:
        raise RecipeError(f"{path}: {exc}") from None
    return data


# What the safe constructors raise, instead of a yaml.YAMLError, for a scalar its explicit tag cannot read
# (`!!int abc`, `!!bool maybe`, `!!timestamp nope`, `!!int ''`).
_TAG_ERRORS = (AttributeError, LookupError, TypeError, ValueError)


def _parse_yaml(text: str, path: Path) -> Any:
    """Parse one YAML document with the safe loader.

    It refuses repeated keys, self-containing aliases, deep nesting, and
    aliases that expand to more than _MAX_VALUES values.
    """
    try:
        loader = yaml.SafeLoader(text)
    except yaml.YAMLError as exc:
        raise RecipeError(f"{path}: not valid YAML: {exc}") from exc
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
        raise RecipeError(f"{path}: not valid YAML: {exc}") from exc
    except RecursionError as exc:
        raise RecipeError(f"{path}: not valid YAML: mappings and lists nest too deep to parse") from exc
    finally:
        loader.dispose()


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
# Resolving and checking (steps 4 to 8)


def _resolve(chain: Sequence[tuple[Path, Mapping[str, Any]]]) -> dict[str, Any]:
    """Merge the chain from the root ancestor down, then materialize defaults and apply faithful (steps 4 to 6)."""
    merged: dict[str, Any] = {}
    for _, own in chain:
        merged = _overlay(merged, {key: value for key, value in own.items() if key != "extends"})
    data = _fresh(merged)
    _materialize_defaults(data)
    _apply_faithful(data)
    return data


def _materialize_defaults(data: dict[str, Any]) -> None:
    """Set `faithful` to false when no file sets it, and every fix the chain leaves unset to on (step 5).

    A value of the wrong type (or null) is left as it is for the type check to report.
    """
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
    """
    switched = {section for section, _ in _KIND_SECTIONS if _names_other_kind(merged.get(section), child.get(section))}
    kept = {key: value for key, value in merged.items() if key not in switched}
    return _merge(kept, child)


def _names_other_kind(inherited: Any, value: Any) -> bool:
    """Return True when both are mappings with a kind and the child's kind differs from the inherited one."""
    both_kinds = isinstance(inherited, dict) and isinstance(value, dict) and "kind" in inherited and "kind" in value
    return both_kinds and inherited["kind"] != value["kind"]


def _merge(parent: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    """Return `parent` overlaid by `child`: two mappings merge recursively, and anything else from the child wins."""
    merged = dict(parent)
    for key, value in child.items():
        inherited = merged.get(key)
        both_maps = isinstance(inherited, dict) and isinstance(value, dict)
        merged[key] = _merge(inherited, value) if both_maps else value
    return merged


def _fresh(value: Any) -> Any:
    """Return a copy of nested mappings and lists that shares no object, so the canonical YAML has no aliases."""
    if isinstance(value, dict):
        return {key: _fresh(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_fresh(item) for item in value]
    return value


def _check_required(data: Mapping[str, Any]) -> None:
    """Raise a required-choice problem for the first REQUIRED key the resolved mapping does not state."""
    for dotted in REQUIRED:
        value: Any = data
        for part in dotted.split("."):
            if not isinstance(value, dict) or part not in value:
                raise _SchemaError(_required_message(dotted))
            value = value[part]


def _bindings(data: Mapping[str, Any]) -> list[Binding]:
    """Return the components the resolved mapping binds, in the order below; absent sections bind nothing."""
    found: list[Binding] = []

    def bind(interface: str, name: str, where: str, config: Mapping[str, Any] | None = None) -> None:
        found.append(Binding(interface=interface, name=name, where=where, config=_fresh(dict(config or {}))))

    if "backend" in data.get("model", {}):
        bind("LLMBackend", data["model"]["backend"], "model.backend")
    for key in sorted(data.get("toolchain", {})):
        bind("Toolchain", data["toolchain"][key], f"toolchain.{key}")
    for section, interface in _KIND_SECTIONS:
        if section in data:
            config = {key: value for key, value in data[section].items() if key != "kind"}
            bind(interface, data[section]["kind"], f"{section}.kind", config)
    for index, stage in enumerate(data.get("stages", [])):
        bind("Stage", stage, f"stages[{index}]")
    if "score" in data:
        bind("ScoreProfile", data["score"], "score")
    for role in sorted(data.get("agents", {})):
        bind("Agent", role, f"agents.{role}")
    for judge in sorted(data.get("judges", {})):
        bind("Judge", judge, f"judges.{judge}")
    return found
