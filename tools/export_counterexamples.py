"""Export a run tree's confirmed counterexamples as synthetic fixtures (task P16.7; bible Adversarial Setups, item 1).

Usage:

    uv run tools/export_counterexamples.py <run dir> [--dest <dir>]

<run dir> is a run tree, <runs root>/runs/<run id>, as lassi.core.runner
writes it: provenance.json, the text store texts/, the binary store blobs/,
and one trial.json per trial one level per trial_id segment below it. The
tool writes one fixture per confirmed divergence (each entry of
Trial.adversary.divergences, in every trial) directly in
COUNTEREXAMPLES_DIR, tests/fixtures/adversary/counterexamples/, and nowhere
else: --dest, when given, must resolve to that directory, and any other
destination is refused. Contract findings, agreed and refused proposals,
and trials the adversary stage did not take give no fixture.

Agent Rule 5 and OQ-055, decision 12 (b): confirmed counterexamples join
only synthetic fixtures outside every suite, never the held-out inputs of an
eval or unassigned item. The tool reads no suite manifest and writes none;
it never writes under assets/ (every manifest and tracked source) or the
train layer's synthetic fixture directory (lassi.train.data SYNTHETIC_DIR),
and it refuses a destination that overlaps either, or that a link at or
above COUNTEREXAMPLES_DIR sends elsewhere. No registry reader reaches a
fixture: the readers open a suite manifest only as <bench dir>/<name>.yaml
(lassi.core.runner, lassi.train.data load_bench), and load_suite, handed a
fixture's path, refuses it, since a fixture lacks a manifest's keys (suite,
repo, commit, items). No train recipe can name one either: data.synthetic
names a .jsonl file in SYNTHETIC_DIR, and a fixture is a .json file outside
it.

A fixture is one plain ASCII JSON object with LF line ends (lassi.core.record
json_text), named <run id>--<suite>--<direction>--<item>--run<NN>--p<NN>.json
after its run, its trial (a run binds one project and one arm), and its
proposal's index, holding, in this order:

- format: FORMAT;
- exploratory: false only for a run from a clean tree at a known commit
  (provenance.json's dirty false and commit set), true otherwise, so a
  dirty-tree run's export, and one whose tree state was not recorded, is
  never read as a measurement (Agent Rule 1; AGENTS.md, Results). False
  does not by itself make a value [MEASURED]: AGENTS.md Results still asks
  for a logged run with its provenance manifest, and for a workstation run
  a clean commit in the local distro's clone and the local provenance
  writer;
- provenance: the run's, from provenance.json (Agent Rule 1): run_id,
  recipe, recipe_hash, status, commit, dirty, host and platform (where the
  run ran, so a fixture is never presented as another host's evidence),
  device (the run's device, or, with executors per language, the device of
  the trial's target language), sdk (the manifest's driver), date
  (started_utc, the date every trial's provenance copies), and pins;
- trial_id and bench_item {suite, item, split, direction}, the trial's;
- adversary {agent, config}, from Trial.adversary;
- proposal, the divergence's proposal index;
- candidate_sha256: the sha256 of the candidate's reply, the response text
  of the attempt the adversary stage took (lassi.core.adversary_stage
  taken_attempt, the standing attempt), as the text store keeps it;
- candidate_failure and stats: the Divergence's, binary_io's OutputStats of
  the candidate's run against the target reference's;
- inputs: {name: {sha256, base64}} per input of the divergence: the sha256
  Divergence.inputs records, and the file's bytes from the binary store in
  standard base64.

Refusals (a ValueError, or exit status 2 with a message on stderr, before
anything is written): a destination that does not resolve to
COUNTEREXAMPLES_DIR, that a link at or above COUNTEREXAMPLES_DIR sends
elsewhere (a symlink or a Windows junction), that overlaps assets/ or
SYNTHETIC_DIR, or that is not an existing directory; a run directory that
is missing or holds no trial at a trial's depth; a missing or unreadable
provenance.json, a key it lacks, a value of the wrong type, or a run_id
other than the run directory's name; a trial.json that does not load, or
whose trial_id does not name its place; a trial with a confirmed
divergence whose provenance copy or recipe_hash differs from
provenance.json's, whose standing attempt the stage would not take, or
whose direction's target language provenance.json's devices cannot name
(a trial without a confirmed divergence gives no fixture, and these three
checks skip it); an input that is not in the binary store or no longer matches its
digest; two fixtures whose names differ at most in letter case (they would
share one file on Windows); and a fixture file already in the destination
with other bytes, or a directory at a fixture's name, which is never
overwritten. An identical fixture already there is accepted. An export
that finds no confirmed divergence writes nothing and exits 0.

Known cases, not complete: a fixture holds every input's bytes, so a large
input set gives a large file; the tool does not compare the inputs with the
item's held-out inputs (the stage never stores those); and it does not
refuse a run that did not complete (status running or failed): the fixture
records the status, and main names it after the export.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lassi.core.adversary_stage import taken_attempt  # noqa: E402
from lassi.core.record import Divergence, Provenance, Trial, json_text, parse_trial_id, to_dict  # noqa: E402
from lassi.core.runner import COMPLETE, PROVENANCE_JSON  # noqa: E402
from lassi.core.store import TRIAL_JSON, BlobStore, TextStore, read_trial, sha256_text  # noqa: E402
from lassi.train.data import SYNTHETIC_DIR  # noqa: E402

# The one directory the tool writes to; read when the tool runs, so a test may point it elsewhere.
COUNTEREXAMPLES_DIR = REPO / "tests" / "fixtures" / "adversary" / "counterexamples"
# The directories a destination may not overlap: every suite manifest and tracked source, and the train fixtures.
DATA_ROOTS = (REPO / "assets", SYNTHETIC_DIR)
# The fixture format's name, the first key of every fixture.
FORMAT = "lassi-counterexample-v1"
# A trial directory lies one level per trial_id segment below the run directory.
_TRIAL_DEPTH = 6
# A run id: one plain path segment, as the runner requires.
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
# The provenance.json keys a fixture's provenance copies under its own name (date and sdk are renamed).
_COPIED_KEYS = ("run_id", "recipe", "recipe_hash", "status", "commit", "dirty", "host", "platform")
# The copied provenance.json keys whose value must be a string (run_id, commit, and dirty are checked apart).
_STRING_KEYS = ("recipe", "recipe_hash", "status", "host", "platform")


def export_counterexamples(run_dir: str | Path, dest: str | Path | None = None) -> list[Path]:
    """Write one fixture per confirmed divergence of the run tree at `run_dir`; return the paths written.

    `dest` None means COUNTEREXAMPLES_DIR; any other value must resolve to
    it. Every refusal (see the module docstring) raises ValueError before
    anything is written.
    """
    target = _destination(dest)
    run = _run_tree(run_dir)
    manifest = _manifest(run)
    blobs = BlobStore(run)
    planned: dict[str, bytes] = {}
    for trial in _trials(run):
        for name, fixture in _fixtures(trial, manifest, blobs):
            # Names that differ only by letter case share one file on Windows.
            if name.casefold() in {known.casefold() for known in planned}:
                raise ValueError(f"two confirmed divergences of {run.as_posix()} give the fixture name {name}")
            planned[name] = json_text(fixture).encode("ascii")
    for name, data in planned.items():
        existing = target / name
        if existing.exists() and (not existing.is_file() or existing.read_bytes() != data):
            raise ValueError(f"{existing.as_posix()} already holds another fixture; refusing to overwrite it")
    written: list[Path] = []
    for name, data in planned.items():
        written.append(_write_atomic(target / name, data))
    return written


def _destination(dest: str | Path | None) -> Path:
    """Return the resolved destination: COUNTEREXAMPLES_DIR, an existing directory outside every data root.

    COUNTEREXAMPLES_DIR must resolve to its own absolute spelling, so no
    link (a symlink or a Windows junction) at or above it sends writes
    elsewhere.
    """
    configured = Path(COUNTEREXAMPLES_DIR)
    allowed = configured.resolve()
    if os.path.normcase(os.path.abspath(configured)) != os.path.normcase(str(allowed)):
        raise ValueError(
            f"the destination {configured.as_posix()} resolves to {allowed.as_posix()}: a link at or above it "
            "would send the fixtures elsewhere"
        )
    chosen = allowed if dest is None else Path(dest).resolve()
    if chosen != allowed:
        raise ValueError(
            f"the destination {chosen.as_posix()} is not {allowed.as_posix()}: confirmed counterexamples go only "
            "into that synthetic fixture directory, outside every suite (Agent Rule 5)"
        )
    for root in DATA_ROOTS:
        resolved = Path(root).resolve()
        if allowed == resolved or resolved in allowed.parents or allowed in resolved.parents:
            raise ValueError(f"the destination {allowed.as_posix()} overlaps {resolved.as_posix()}, a data root")
    if not allowed.is_dir():
        raise ValueError(f"the destination {allowed.as_posix()} is not a directory; it is committed with its README")
    return allowed


def _run_tree(run_dir: str | Path) -> Path:
    """Return the run directory, resolved; ValueError when it is missing or not a directory."""
    path = Path(run_dir)
    if not path.is_dir():
        raise ValueError(f"the run directory {path.as_posix()} does not exist or is not a directory")
    return path.resolve()


def _refuse_constant(name: str) -> NoReturn:
    """Refuse a JSON constant (NaN, Infinity, -Infinity), which strict JSON does not allow."""
    raise ValueError(f"{name} is not a strict JSON value")


def _finite_float(text: str) -> float:
    """Return a JSON number as a float; refuse one that overflows a float."""
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"the number {text} overflows a float")
    return value


def _manifest(run: Path) -> dict[str, Any]:
    """Return the run's provenance.json, read strictly, with the keys a fixture copies checked.

    commit, dirty, the device keys, driver, and started_utc are checked as
    a trial's Provenance checks them; run_id must be the run directory's
    name, recipe, recipe_hash, status, host, and platform strings, and pins
    a mapping. ValueError names the file and the key.
    """
    path = run / PROVENANCE_JSON
    where = path.as_posix()
    if not path.is_file():
        raise ValueError(f"{where} is missing; every fixture carries its run's provenance (Agent Rule 1)")
    try:
        data = json.loads(path.read_text(encoding="utf-8"), parse_constant=_refuse_constant, parse_float=_finite_float)
    except (OSError, ValueError, RecursionError) as error:
        raise ValueError(f"cannot read {where}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{where} is not a JSON object")
    required = (*_COPIED_KEYS, "driver", "started_utc", "pins")
    missing = [key for key in required if key not in data]
    if missing or ("device" not in data and "devices" not in data):
        raise ValueError(f"{where} lacks {missing or ['device or devices']}; it is not a run manifest")
    if data["run_id"] != run.name or not _NAME.fullmatch(str(data["run_id"])):
        raise ValueError(f"{where}: run_id {data['run_id']!r} is not the run directory's name {run.name!r}")
    wrong = [key for key in _STRING_KEYS if not isinstance(data[key], str)]
    if wrong:
        raise ValueError(f"{where}: {wrong} must be strings")
    if not isinstance(data["pins"], dict) or ("devices" in data and not isinstance(data["devices"], dict)):
        raise ValueError(f"{where}: pins and devices must be JSON objects")
    known = {"commit": data["commit"], "dirty": data["dirty"], "sdk": data["driver"], "date": data["started_utc"]}
    try:
        Provenance(**known, device=data.get("device"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{where}: {error}") from error
    return data


def _trials(run: Path) -> list[Trial]:
    """Return every trial of the run tree, in path order; ValueError for none, or one that does not load."""
    store = TextStore(run)
    paths = sorted(run.glob("/".join(["*"] * _TRIAL_DEPTH + [TRIAL_JSON])))
    if not paths:
        raise ValueError(f"the run directory {run.as_posix()} holds no {TRIAL_JSON} at a trial's depth")
    trials: list[Trial] = []
    for path in paths:
        try:
            trial = read_trial(path, store)
        except (OSError, KeyError, ValueError, RecursionError) as error:
            raise ValueError(f"cannot load {path.as_posix()}: {error!r}") from error
        place = path.parent.relative_to(run).as_posix()
        if trial.trial_id != place:
            raise ValueError(f"{path.as_posix()} holds the trial {trial.trial_id!r}, not {place!r}, its place")
        trials.append(trial)
    return trials


def _device(manifest: Mapping[str, Any], direction: str) -> Any:
    """Return the device of a direction's target: the manifest's device, or its devices entry for the target.

    A direction is `<source>-<target>` and a language name may hold a
    hyphen, so the target is the one language of `devices` that ends the
    direction after a hyphen; none or several is refused.
    """
    if "devices" not in manifest:
        return manifest["device"]
    devices = manifest["devices"]
    targets = [name for name in devices if direction.endswith(f"-{name}") and len(direction) > len(name) + 1]
    if len(targets) != 1:
        raise ValueError(f"provenance.json's devices {sorted(devices)} name no single target of {direction!r}")
    return devices[targets[0]]


def _provenance(manifest: Mapping[str, Any], trial: Trial) -> dict[str, Any]:
    """Return a fixture's provenance for `trial`; ValueError when the trial's copy or recipe hash differs."""
    if trial.recipe_hash != manifest["recipe_hash"]:
        raise ValueError(f"trial {trial.trial_id}: recipe_hash {trial.recipe_hash!r} is not provenance.json's")
    copied = {key: manifest[key] for key in _COPIED_KEYS}
    device = _device(manifest, trial.bench_item.direction)
    record = {**copied, "device": device, "sdk": manifest["driver"], "date": manifest["started_utc"]}
    record["pins"] = manifest["pins"]
    own = trial.provenance
    for key, value in (("commit", own.commit), ("dirty", own.dirty), ("device", own.device), ("sdk", own.sdk)):
        if record[key] != value:
            raise ValueError(f"trial {trial.trial_id}: provenance {key} {value!r} is not provenance.json's")
    if own.date != record["date"]:
        raise ValueError(f"trial {trial.trial_id}: provenance date {own.date!r} is not provenance.json's started_utc")
    return record


def _exploratory(manifest: Mapping[str, Any]) -> bool:
    """Return False only for a run from a clean tree at a known commit (Agent Rule 1)."""
    return not (manifest["dirty"] is False and isinstance(manifest["commit"], str))


def _fixtures(trial: Trial, manifest: Mapping[str, Any], blobs: BlobStore) -> list[tuple[str, dict[str, Any]]]:
    """Return (file name, fixture) for each confirmed divergence of `trial`, in proposal order; [] without one."""
    record = trial.adversary
    if record is None or not record.divergences:
        return []
    candidate = taken_attempt(trial)
    if candidate is None:
        raise ValueError(f"trial {trial.trial_id} has an adversary record, but no attempt the adversary stage takes")
    head = {
        "format": FORMAT,
        "exploratory": _exploratory(manifest),
        "provenance": _provenance(manifest, trial),
        "trial_id": trial.trial_id,
        "bench_item": {
            "suite": trial.bench_item.suite,
            "item": trial.bench_item.item,
            "split": trial.bench_item.split,
            "direction": trial.bench_item.direction,
        },
        "adversary": {"agent": record.agent, "config": record.config},
    }
    parts = parse_trial_id(trial.trial_id)
    stem = f"{manifest['run_id']}--{parts.bench}--{parts.direction}--{parts.item}--run{parts.run:02d}"
    fixtures: list[tuple[str, dict[str, Any]]] = []
    for divergence in record.divergences:
        fixture = {
            **head,
            "proposal": divergence.proposal,
            "candidate_sha256": sha256_text(candidate.response_text),
            "candidate_failure": divergence.candidate_failure,
            "stats": [to_dict(entry) for entry in divergence.stats],
            "inputs": _inputs(divergence, blobs, f"trial {trial.trial_id}, proposal {divergence.proposal}"),
        }
        fixtures.append((f"{stem}--p{divergence.proposal:02d}.json", fixture))
    return fixtures


def _inputs(divergence: Divergence, blobs: BlobStore, where: str) -> dict[str, dict[str, str]]:
    """Return {name: {sha256, base64}} for each input of `divergence`, read from the binary store."""
    found: dict[str, dict[str, str]] = {}
    for name, digest in divergence.inputs.items():
        try:
            data = blobs.get(digest)
        except KeyError:
            raise ValueError(f"{where}: input {name!r} ({digest}) is not in the binary store") from None
        except ValueError as error:
            raise ValueError(f"{where}: input {name!r}: {error}") from error
        found[name] = {"sha256": digest, "base64": base64.b64encode(data).decode("ascii")}
    return found


def _write_atomic(target: Path, data: bytes) -> Path:
    """Write `data` to `target` through a temporary file in its directory and os.replace; return `target`."""
    handle, temp_name = tempfile.mkstemp(dir=target.parent, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(handle, "wb") as temp_file:
            temp_file.write(data)
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line: 0 after an export (none found included), 2 with a message on stderr for a refusal."""
    parser = argparse.ArgumentParser(
        prog="export_counterexamples.py",
        description="Export a run tree's confirmed counterexamples into tests/fixtures/adversary/counterexamples/.",
    )
    parser.add_argument("run_dir", type=Path, help="the run tree, <runs root>/runs/<run id>")
    parser.add_argument("--dest", type=Path, default=None, help="must be the counterexamples fixture directory")
    args = parser.parse_args(argv)
    try:
        paths = export_counterexamples(args.run_dir, args.dest)
    except ValueError as error:
        print(f"export_counterexamples: refused: {error}", file=sys.stderr)
        return 2
    if not paths:
        print(f"no confirmed divergence in {args.run_dir}; nothing written")
        return 0
    first = json.loads(paths[0].read_text(encoding="ascii"))
    for path in paths:
        print(path.as_posix())
    label = " (exploratory: the run's tree was dirty or its state was not recorded)" if first["exploratory"] else ""
    print(f"wrote {len(paths)} fixture(s){label}")
    status = first["provenance"]["status"]
    if status != COMPLETE:
        print(f"note: the run's status is {status!r}, not {COMPLETE!r}; its trials may be partial")
    return 0


if __name__ == "__main__":
    sys.exit(main())
