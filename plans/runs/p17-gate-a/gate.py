"""The checks and records of gate part (a)'s batch (task P17.11); run.sh beside it calls this module.

Run with the gate venv's interpreter from the repository root:

    python plans/runs/p17-gate-a/gate.py <command> [options]

Commands, each writing into the batch's report directory (ASCII JSON or text) and printing the few lines its
function names:

- fetch-model: fetch the recipe's model once under HF_HOME at its pinned revision and check it against the
  bible's gate model pin (MODEL_PIN; bible Toolchain Pins, gate model).
- speed: time hf_local's model load, a short greedy completion, and one long prompt on the recipe's device.
- trials: check the hf_local run's tree and copy its records (provenance.json, the resolved recipe).
- checkpoint: check one `lassi train` tree and copy its records (no weights).
- refusal: check one refused `lassi run` of a p17-gpu-*.yaml recipe, its strace log included.
- du-new: list what the batch added under HF_HOME and the new entries of the scratch root and its .cache.
- ascii: rewrite any non-ASCII byte in the report as an escape and write files.txt, the report's file list.
- verdict: one line per gate check from the records above and steps.tsv, then PASS or FAIL.

Exit status: 0 when the command's checks pass, 1 when one fails, 2 on bad usage. Only the standard library is
imported at module level; fetch-model imports huggingface_hub, speed imports lassi.llm.hf_local (and so torch),
and fetch-model, speed, and trials read the recipe with lassi.core.recipe. No command opens a device node,
starts a process, or reaches the network except fetch-model's one snapshot download when the model is absent
or does not match the pin.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

# The bible's gate model pin (Toolchain Pins, gate model; plans/spikes/p17-frameworks.md, Results 6).
MODEL_PIN: dict[str, Any] = {
    "files": 10,
    "bytes": 999_604_233,
    "sha256": {
        "model.safetensors": "f9523886352217ded3aeeef552b381af79d568c6d49a4b9e423288cea56b0a44",
        "config.json": "b1e58593cd31852f7da5c2fc31ddf6135b9c066c0fd9177a4bbe95717083adff",
        "tokenizer_config.json": "959e7f1d9a1b7641a6d6ce05ca97b75c7894fcb66cbe5a040406458fb1128ee4",
        "generation_config.json": "fdaccbcb02f3e1e7914ccb0f69ebe899071ffd27cf825166d823163e156870f2",
        "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    },
}
# The framework pins of the cpu extra (bible Toolchain Pins, framework pins), as installed versions.
FRAMEWORK_PINS = {"torch": "2.14.1+cpu", "transformers": "5.18.0", "trl": "1.14.1", "peft": "0.21.2",
                  "accelerate": "1.15.0"}
# The installed versions hf_local's serving record holds (lassi.llm.hf_local VERSION_NAMES).
SERVING_VERSIONS = ("torch", "transformers", "tokenizers", "huggingface-hub", "safetensors")
# The end reasons that stop a trial at the baseline (lassi.core.record END_REASONS).
BASELINE_ENDS = ("baseline-compile", "baseline-run", "baseline-disagree")
# The hf_local run: one item in both directions, one trial each.
DIRECTIONS = ("cuda-omp", "omp-cuda")
# Each refusal recipe (tests/fixtures/recipes/<name>.yaml): the device key it names and the kind.
REFUSALS = {
    "p17-gpu-hf-cuda": ("model.device", "cuda"),
    "p17-gpu-hf-rocm": ("model.device", "rocm"),
    "p17-gpu-exec-cuda": ("executor.cuda.device", "cuda"),
    "p17-gpu-exec-rocm": ("executor.cuda.device", "rocm"),
}
TRAIN_METHODS = ("sft", "dpo", "grpo")
TRAIN_STEPS = 2
# GPU device nodes, nested ones such as /dev/nvidia-caps/* too: no open of one is allowed (OQ-002); metadata
# calls (stat, access) are listed. Known cases, not complete: a path opened relative to a directory descriptor
# and an alias such as /dev/char/<major>:<minor> are not resolved.
GPU_NODE = re.compile(r"/dev/(kfd|dri(/.*)?|nvidia[^/]*(/.*)?)")
OPEN_CALLS = frozenset({"open", "openat", "openat2", "creat"})
METADATA_CALLS = frozenset({"access", "faccessat", "faccessat2", "stat", "lstat", "newfstatat", "statx"})
# One strace line: an optional pid, the call, then the first quoted argument (the path for every traced call).
_STRACE_CALL = re.compile(r'^(?:\[pid\s+\d+\]\s+|\d+\s+)?(?:<\.\.\.\s+)?([a-z0-9_]+)\(.*?"((?:[^"\\]|\\.)*)"')
# SYNTHETIC prompts for the speed step; nothing here is a bench item.
SPEED_SYSTEM = "SYNTHETIC speed check."
DECODE_PROMPT = "SYNTHETIC: write a C function that sums an array of integers, with a comment on every line."
PREFILL_LINE = "// SYNTHETIC filler line {0}: int value_{0} = {0} * 3 + 1;"
PREFILL_LINES = 420
DECODE_TOKENS = 256
# Files the batch keeps appending after the file list is written.
OPEN_FILES = ("stdout.txt", "stderr.txt")


def _write_json(path: Path, data: Any) -> None:
    """Write `data` as two-space JSON, ASCII only, with a final LF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(data, indent=2, ensure_ascii=True, sort_keys=False) + "\n").encode("ascii"))


def _read_json(path: Path) -> Any:
    """Return the JSON in `path`, or None when it is missing or not JSON."""
    try:
        return json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, ValueError):
        return None


def _say(*lines: str) -> None:
    """Print each line as printable ASCII, at most 200 characters."""
    for line in lines:
        print(line.encode("ascii", "backslashreplace").decode("ascii")[:200], flush=True)


def _copy(source: Path, target: Path) -> bool:
    """Copy one file when it exists; return whether it did."""
    if not source.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return True


def _sha256(path: Path) -> str:
    """Return the sha256 hex digest of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_kib(root: Path) -> int:
    """Return the disk use of a tree in KiB from lstat block counts (each inode once), 0 when it is missing."""
    seen: set[tuple[int, int]] = set()
    total = 0
    for directory, _, names in os.walk(root):
        for name in [".", *names]:
            try:
                info = os.lstat(os.path.join(directory, name))
            except OSError:
                continue
            if (info.st_dev, info.st_ino) not in seen:
                seen.add((info.st_dev, info.st_ino))
                total += getattr(info, "st_blocks", 0) * 512
    return total // 1024


def _recipe_model(recipe: Path) -> dict[str, Any]:
    """Return the resolved model section of a run recipe, read by lassi's own loader and registry.

    Importing lassi.core.runner registers every component, as `lassi run`
    does; hf_local's module imports no framework at import.
    """
    importlib.import_module("lassi.core.runner")
    from lassi.core.recipe import load_recipe

    return dict(load_recipe(recipe).data["model"])


# ---------------------------------------------------------------------------
# fetch-model


def snapshot_files(snapshot: Path, digests: Iterable[str]) -> dict[str, Any]:
    """Return a snapshot's file count, total bytes (links followed), and the sha256 of each file in `digests`."""
    files = sorted(path for path in snapshot.rglob("*") if path.is_file())
    names = {path.relative_to(snapshot).as_posix(): path for path in files}
    return {
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "names": sorted(names),
        "sha256": {name: _sha256(names[name]) if name in names else None for name in sorted(digests)},
    }


def model_problems(found: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    """Return how a snapshot differs from the expected pin: count, bytes, or a digest; [] when it matches."""
    problems = [f"{key} {found[key]} != {expected[key]}" for key in ("files", "bytes") if found[key] != expected[key]]
    for name, digest in sorted(expected["sha256"].items()):
        if found["sha256"].get(name) != digest:
            problems.append(f"sha256 of {name} {found['sha256'].get(name)} != {digest}")
    return problems


def fetch_model(args: argparse.Namespace) -> int:
    """Fetch the recipe's model at its revision only when HF_HOME lacks it, verify it, and write model.json.

    The offline lookup (local_files_only) runs first; one online
    snapshot_download at the 40-hex revision follows only when the model is
    absent or does not match the pin. Prints one line.
    """
    hub = importlib.import_module("huggingface_hub")
    model = _recipe_model(args.recipe)
    expected = MODEL_PIN if args.expect is None else _read_json(args.expect)
    home = Path(os.environ["HF_HOME"])
    record: dict[str, Any] = {"id": model["id"], "revision": model["revision"], "expected": expected,
                              "hf_home": str(home), "hf_home_kib_before": _tree_kib(home), "state": "absent"}
    found: dict[str, Any] | None = None
    try:
        path = Path(hub.snapshot_download(model["id"], revision=model["revision"], local_files_only=True))
        found = snapshot_files(path, expected["sha256"])
        record["state"] = "present" if not model_problems(found, expected) else "present-but-different"
    except Exception as error:  # the hub library's not-found errors differ by release
        record["offline_lookup"] = f"{type(error).__name__}: {error}"
    if record["state"] != "present":
        start = time.monotonic()
        try:
            path = Path(hub.snapshot_download(model["id"], revision=model["revision"]))
            found = snapshot_files(path, expected["sha256"])
            record["state"] = "fetched"
        except Exception as error:  # a failed download is recorded, not raised
            record["state"] = f"fetch failed: {type(error).__name__}: {error}"[:300]
        record["fetch_s"] = round(time.monotonic() - start, 1)
    problems = ["no snapshot"] if found is None else model_problems(found, expected)
    in_revision_dir = found is not None and path.name == model["revision"]
    record.update(snapshot=None if found is None else str(path), found=found, problems=problems,
                  passed=not problems and in_revision_dir, hf_home_kib_after=_tree_kib(home))
    _write_json(args.report / "model.json", record)
    verdict = "the pin matches" if record["passed"] else "MISMATCH: " + "; ".join(problems)[:120]
    counts = "no files" if found is None else f"{found['files']} files, {found['bytes']} bytes"
    _say(f"fetch-model: {record['state'][:80]}; {counts}; {verdict}")
    return 0 if record["passed"] else 1


# ---------------------------------------------------------------------------
# speed


def _timed_completion(backend: Any, prompt: str, max_tokens: int) -> dict[str, Any]:
    """Return one greedy completion's token counts, seconds, and tokens per second (generated, and prompt)."""
    from lassi.core.interfaces import Message, Sampling

    messages = [Message(role="system", content=SPEED_SYSTEM), Message(role="user", content=prompt)]
    start = time.perf_counter()
    completion = backend.complete(messages, Sampling(temperature=0.0, top_p=1.0, max_tokens=max_tokens))
    seconds = time.perf_counter() - start
    return {
        "prompt_tokens": completion.prompt_tokens,
        "completion_tokens": completion.completion_tokens,
        "seconds": round(seconds, 3),
        "completion_tokens_per_s": round(completion.completion_tokens / seconds, 2) if seconds > 0 else None,
        "prompt_tokens_per_s": round(completion.prompt_tokens / seconds, 2) if seconds > 0 else None,
    }


def speed(args: argparse.Namespace) -> int:
    """Time hf_local's load, a 256-token greedy completion, and a 1-token completion of a long prompt; speed.json.

    The prefill figure includes the one generated token. One run on a
    shared host, no spread; a value is [MEASURED] only with the clean-commit
    rx id the summary cites. Prints two lines.
    """
    hf_local = importlib.import_module("lassi.llm.hf_local")
    model = _recipe_model(args.recipe)
    backend = hf_local.HFLocalBackend(
        model["id"], device=model["device"], revision=model["revision"], seed=model["seed"]
    )
    start = time.perf_counter()
    backend.check()
    load_s = round(time.perf_counter() - start, 3)
    serving = backend.serving()
    decode = _timed_completion(backend, DECODE_PROMPT, DECODE_TOKENS)
    long_prompt = "\n".join(PREFILL_LINE.format(index) for index in range(PREFILL_LINES))
    prefill = _timed_completion(backend, long_prompt, 1)
    record = {"model": model["id"], "revision": model["revision"], "load_s": load_s, "decode": decode,
              "prefill": prefill, "dtype": serving["dtype"], "threads": serving["threads"],
              "device": serving["device"], "versions": serving["versions"],
              "omp_num_threads": os.environ.get("OMP_NUM_THREADS")}
    _write_json(args.report / "speed.json", record)
    _say(f"speed: load {load_s} s; dtype {serving['dtype']}; torch threads {serving['threads']}",
         f"  decode {decode['completion_tokens']} tokens in {decode['seconds']} s; prefill "
         f"{prefill['prompt_tokens']} tokens in {prefill['seconds']} s")
    return 0


# ---------------------------------------------------------------------------
# trials


def trial_row(path: Path) -> dict[str, Any]:
    """Return one trial.json's summary: id, direction, end reason, attempts, stages, reference run, alignment."""
    trial = _read_json(path) or {}
    final = trial.get("final") or {}
    attempts = trial.get("attempts") or []
    reference = trial.get("reference_run") or {}
    end = final.get("end_reason") or {}
    return {
        "trial_id": trial.get("trial_id"),
        "direction": (trial.get("bench_item") or {}).get("direction"),
        "end_reason": end.get("code"),
        "end_message": end.get("message"),
        "attempts": len(attempts),
        "last_attempt_stage": attempts[-1].get("stage_reached") if attempts else None,
        "final_stage": final.get("stage_reached"),
        "corrections": final.get("corrections"),
        "reference_exit_code": reference.get("exit_code"),
        "reference_wall_s": reference.get("wall_s"),
        "final_alignment": final.get("alignment"),
        "wall_s": final.get("wall_s"),
        "requests": len(trial.get("requests") or []),
    }


def run_checks(provenance: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], revision: str) -> dict[str, bool]:
    """Return the hf_local run's gate checks: status, trials past the baseline, device record, framework pins."""
    records = provenance.get("device_records") or []
    model = next((item for item in records if item.get("key") == "model.device"), {})
    serving = provenance.get("serving") or {}
    versions = serving.get("versions") or {}
    return {
        "status_complete": provenance.get("status") == "complete",
        "both_directions_once": sorted(str(row["direction"]) for row in rows) == sorted(DIRECTIONS),
        "every_trial_past_the_baseline": bool(rows) and all(row["end_reason"] not in BASELINE_ENDS for row in rows),
        "model_device_record_cpu": model.get("kind") == "cpu" and model.get("framework") == "torch"
        and model.get("framework_version") == FRAMEWORK_PINS["torch"],
        "serving_versions_recorded": set(SERVING_VERSIONS) <= set(versions)
        and versions.get("torch") == FRAMEWORK_PINS["torch"]
        and versions.get("transformers") == FRAMEWORK_PINS["transformers"],
        "serving_revision_pinned": serving.get("revision") == revision,
    }


def progress_lines(stderr: Path) -> int:
    """Return how many progress-bar updates (a "%|" bar, split at carriage returns too) a stderr log holds."""
    try:
        text = stderr.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return 0
    return sum(1 for line in re.split(r"[\r\n]", text) if "%|" in line)


def trials(args: argparse.Namespace) -> int:
    """Check the hf_local run tree and write trials.json; copy provenance.json and the resolved recipe. Two lines."""
    run_dir: Path = args.run_dir
    _copy(run_dir / "provenance.json", args.report / "hf-run-provenance.json")
    _copy(run_dir / "recipe.resolved.yaml", args.report / "hf-run-recipe.resolved.yaml")
    provenance = _read_json(run_dir / "provenance.json") or {}
    rows = [trial_row(path) for path in sorted(run_dir.rglob("trial.json"))]
    checks = run_checks(provenance, rows, _recipe_model(args.recipe)["revision"])
    record = {"run_dir": str(run_dir), "trials": rows, "checks": checks, "passed": all(checks.values()),
              "progress_lines_on_stderr": progress_lines(args.stderr)}
    _write_json(args.report / "trials.json", record)
    ends = ", ".join(f"{row['direction']} {row['end_reason'] or 'none'} ({row['attempts']} attempts)" for row in rows)
    failed = [name for name, passed in checks.items() if not passed]
    _say(f"hf-run: {len(rows)} trials: {ends or 'none'}",
         "  checks: " + ("all pass" if not failed else "FAILED " + ", ".join(failed)))
    return 0 if record["passed"] else 1


# ---------------------------------------------------------------------------
# checkpoint


def checkpoint_checks(provenance: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    """Return one train run's gate checks: status, steps, the cpu device record, and the pins in each checkpoint."""
    def pins_ok(source: Mapping[str, Any]) -> bool:
        pins = source.get("framework_pins") or {}
        packages = pins.get("packages") or {}
        build = pins.get("build") or {}
        torch_ok = build.get("version") == FRAMEWORK_PINS["torch"]
        return torch_ok and all(packages.get(name) == version for name, version in FRAMEWORK_PINS.items())

    def cpu_ok(source: Mapping[str, Any]) -> bool:
        return [item.get("kind") for item in source.get("device_records") or []] == ["cpu"]

    return {
        "status_complete": provenance.get("status") == "complete",
        "steps": provenance.get("steps") == TRAIN_STEPS,
        "device_record_cpu": cpu_ok(provenance),
        "framework_pins": pins_ok(provenance),
        "checkpoint_records": bool(records) and all(cpu_ok(item) and pins_ok(item) for item in records),
    }


def checkpoint(args: argparse.Namespace) -> int:
    """Check one train tree, copy its records (never weights) into the report, and write checks.json. One line."""
    train_dir: Path = args.train_dir
    out: Path = args.report
    _copy(train_dir / "provenance.json", out / "provenance.json")
    _copy(train_dir / "recipe.resolved.yaml", out / "recipe.resolved.yaml")
    provenance = _read_json(train_dir / "provenance.json") or {}
    records = []
    for name in provenance.get("checkpoints") or []:
        for file in ("checkpoint.json", "steps.jsonl"):
            _copy(train_dir / "output" / name / file, out / name / file)
        records.append(_read_json(train_dir / "output" / name / "checkpoint.json") or {})
    checks = checkpoint_checks(provenance, records)
    trainer = (records[-1].get("trainer_records") or {}) if records else {}
    summary = trainer.get("train_summary") or {}
    arguments = trainer.get("training_args") or {}
    record = {"train_dir": str(train_dir), "method": args.method, "checks": checks, "passed": all(checks.values()),
              "train_runtime": summary.get("train_runtime"), "global_step": trainer.get("global_step"),
              "precision": {key: arguments.get(key) for key in ("bf16", "fp16", "gradient_checkpointing")}}
    _write_json(out / "checks.json", record)
    failed = [name for name, passed in checks.items() if not passed]
    _say(f"train-{args.method}: steps {provenance.get('steps')}, train_runtime {record['train_runtime']} s; "
         + ("checks pass" if not failed else "FAILED " + ", ".join(failed)))
    return 0 if record["passed"] else 1


# ---------------------------------------------------------------------------
# refusal


def strace_calls(lines: Iterable[str]) -> list[tuple[str, str]]:
    """Return (call, path) for each strace line whose call is traced and whose first quoted argument is a path."""
    found = []
    for line in lines:
        match = _STRACE_CALL.match(line.strip())
        if match and match.group(1) in OPEN_CALLS | METADATA_CALLS:
            found.append((match.group(1), match.group(2)))
    return found


def node_calls(calls: Iterable[tuple[str, str]]) -> dict[str, list[str]]:
    """Return the GPU node opens and metadata calls among `calls`, each as "call path", in order."""
    opens = [f"{call} {path}" for call, path in calls if GPU_NODE.fullmatch(path) and call in OPEN_CALLS]
    metadata = [f"{call} {path}" for call, path in calls if GPU_NODE.fullmatch(path) and call in METADATA_CALLS]
    return {"gpu_node_opens": opens, "gpu_node_metadata": metadata}


def refusal_checks(name: str, rc: int, stderr: str, run_dir_exists: bool, opens: Sequence[str]) -> dict[str, bool]:
    """Return one refusal's checks: status 2, the message naming the key and kind 0, no run directory, no open."""
    key, kind = REFUSALS[name]
    return {
        "exit_status_2": rc == 2,
        "names_the_device": f"{key} names {kind} 0, which this host cannot provide: " in stderr,
        "no_run_directory": not run_dir_exists,
        "no_gpu_node_opened": not opens,
    }


def refusal(args: argparse.Namespace) -> int:
    """Check one refused `lassi run` and merge its entry into gpu-refusals.json. One line."""
    stderr = args.stderr.read_bytes().decode("utf-8", errors="replace") if args.stderr.is_file() else ""
    try:
        lines = args.strace.read_bytes().decode("utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    nodes = node_calls(strace_calls(lines))
    run_dir = args.runs_root / "runs" / args.run_id
    checks = refusal_checks(args.name, args.rc, stderr, run_dir.exists(), nodes["gpu_node_opens"])
    checks["strace_log_read"] = bool(lines)
    message = next((line for line in stderr.splitlines() if "cannot provide" in line), stderr.strip()[-300:])
    entry = {"recipe": f"tests/fixtures/recipes/{args.name}.yaml", "rc": args.rc, "message": message,
             "run_dir": str(run_dir), "strace_lines": len(lines), **nodes, "checks": checks,
             "passed": all(checks.values())}
    path = args.report / "gpu-refusals.json"
    merged = _read_json(path) or {}
    merged[args.name] = entry
    _write_json(path, merged)
    failed = [key for key, passed in checks.items() if not passed]
    _say(f"refuse {args.name}: rc {args.rc}; "
         + ("refused as expected" if not failed else "FAILED " + ", ".join(failed)))
    return 0 if entry["passed"] else 1


# ---------------------------------------------------------------------------
# du-new, ascii, verdict


def new_under(root: Path, stamp: float) -> dict[str, dict[str, int]]:
    """Return, per first-level entry of `root`, the count and bytes of files modified after `stamp`."""
    found: dict[str, dict[str, int]] = {}
    for directory, _, names in os.walk(root):
        for name in names:
            path = Path(directory, name)
            try:
                info = path.lstat()
            except OSError:
                continue
            if info.st_mtime > stamp:
                first = path.relative_to(root).parts[0]
                entry = found.setdefault(first, {"files": 0, "bytes": 0})
                entry["files"] += 1
                entry["bytes"] += info.st_size
    return found


def du_new(args: argparse.Namespace) -> int:
    """Write du-new.json: files newer than the stamp under HF_HOME, and new first-level scratch entries. One line."""
    stamp = args.stamp.stat().st_mtime
    before = set(args.before.read_text(encoding="utf-8").splitlines()) if args.before.is_file() else set()
    now = set(scratch_entries(args.scratch))
    record = {"hf_home": str(args.hf_home), "hf_home_new": new_under(args.hf_home, stamp),
              "scratch_entries_new": sorted(now - before), "scratch_entries_gone": sorted(before - now)}
    _write_json(args.report / "du-new.json", record)
    _say(f"du-new: HF_HOME changed under {sorted(record['hf_home_new']) or 'nothing'}; new scratch entries: "
         + (", ".join(record["scratch_entries_new"]) or "none"))
    return 0


def scratch_entries(scratch: Path) -> list[str]:
    """Return the first-level entries of the scratch root and of its .cache, as "<name>" and ".cache/<name>"."""
    names = []
    for prefix, directory in (("", scratch), (".cache/", scratch / ".cache")):
        try:
            names += [prefix + entry.name for entry in os.scandir(directory)]
        except OSError:
            continue
    return sorted(names)


def ascii_report(args: argparse.Namespace) -> int:
    """Rewrite non-ASCII bytes of every report file as escapes, then write files.txt (sha256, bytes, path). One line."""
    rewritten = []
    rows = []
    for path in sorted(item for item in args.dir.rglob("*") if item.is_file() and item.name != "files.txt"):
        name = path.relative_to(args.dir).as_posix()
        data = path.read_bytes()
        if not data.isascii():
            text = data.decode("utf-8", errors="backslashreplace")
            path.write_bytes(text.encode("ascii", errors="backslashreplace"))
            rewritten.append(name)
        if name in OPEN_FILES:
            rows.append(f"-  -  {name}")
        else:
            rows.append(f"{_sha256(path)}  {path.stat().st_size}  {name}")
    (args.dir / "files.txt").write_bytes(("\n".join(rows) + "\n").encode("ascii"))
    _say(f"ascii: {len(rows)} report files; rewritten to ASCII escapes: {', '.join(rewritten) or 'none'}")
    return 0


def read_steps(path: Path) -> dict[str, str]:
    """Return each step's exit status (a number or "skipped") from steps.tsv, by step name."""
    found = {}
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except OSError:
        return found
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) >= 2:
            found[fields[0]] = fields[1]
    return found


def gate_lines(report: Path, tree: str) -> list[tuple[str, bool]]:
    """Return (check, passed) for every gate check the batch can see, in the order of gate part (a)'s text."""
    steps = read_steps(report / "steps.tsv")

    def ok(step: str) -> bool:
        return steps.get(step) == "0"

    def passed(path: Path) -> bool:
        return bool((_read_json(path) or {}).get("passed"))

    refusals = _read_json(report / "gpu-refusals.json") or {}
    lines = [("env synced from the lock with the cpu extra and the five pins", ok("env") and ok("env-pins")),
             ("model fetched once at its revision and matching the pin", ok("fetch-model") and passed(
                 report / "model.json")),
             ("bench sources, upstream, and the slot's prompt files", all(ok(step) for step in (
                 "fetch-bench", "fetch-upstream", "extract-assets"))),
             ("hf_local on cpu: both directions past the baseline, device record and pins", ok("hf-run") and passed(
                 report / "trials.json"))]
    for method in TRAIN_METHODS:
        lines.append((f"lassi train {method}: checkpoint with device and pins", ok(f"train-{method}") and passed(
            report / f"train-{method}" / "checks.json")))
    for name in REFUSALS:
        lines.append((f"{name} refused before any directory, no node opened", bool(
            (refusals.get(name) or {}).get("passed"))))
    lines.append(("slot tree clean after the batch", tree == "clean"))
    return lines


def verdict(args: argparse.Namespace) -> int:
    """Print one line per gate check and the verdict, and write verdict.txt; exit 0 only on PASS."""
    lines = gate_lines(args.report, args.tree)
    result = "PASS" if all(passed for _, passed in lines) else "FAIL"
    text = [f"{'pass' if passed else 'FAIL'}  {check}" for check, passed in lines] + [f"verdict: {result}"]
    (args.report / "verdict.txt").write_bytes(("\n".join(text) + "\n").encode("ascii"))
    failed = [line for line in text[:-1] if line.startswith("FAIL")]
    _say(*(failed[:6] + [f"verdict: {result} ({len(lines) - len(failed)} of {len(lines)} checks pass)"]))
    return 0 if result == "PASS" else 1


# ---------------------------------------------------------------------------
# Command line


def build_parser() -> argparse.ArgumentParser:
    """Return the parser for every command."""
    parser = argparse.ArgumentParser(prog="gate.py", description="Checks and records of gate part (a)'s batch.")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch-model")
    fetch.add_argument("--recipe", type=Path, required=True)
    fetch.add_argument("--expect", type=Path, help="a JSON pin in MODEL_PIN's shape (stand-in only)")
    speed_parser = commands.add_parser("speed")
    speed_parser.add_argument("--recipe", type=Path, required=True)
    trials_parser = commands.add_parser("trials")
    trials_parser.add_argument("--recipe", type=Path, required=True)
    trials_parser.add_argument("--run-dir", type=Path, required=True)
    trials_parser.add_argument("--stderr", type=Path, required=True)
    train = commands.add_parser("checkpoint")
    train.add_argument("--train-dir", type=Path, required=True)
    train.add_argument("--method", choices=TRAIN_METHODS, required=True)
    refuse = commands.add_parser("refusal")
    refuse.add_argument("--name", choices=sorted(REFUSALS), required=True)
    refuse.add_argument("--rc", type=int, required=True)
    refuse.add_argument("--stderr", type=Path, required=True)
    refuse.add_argument("--strace", type=Path, required=True)
    refuse.add_argument("--runs-root", type=Path, required=True)
    refuse.add_argument("--run-id", required=True)
    du = commands.add_parser("du-new")
    du.add_argument("--stamp", type=Path, required=True)
    du.add_argument("--before", type=Path, required=True)
    du.add_argument("--hf-home", type=Path, required=True)
    du.add_argument("--scratch", type=Path, required=True)
    commands.add_parser("scratch-entries").add_argument("--scratch", type=Path, required=True)
    commands.add_parser("ascii").add_argument("--dir", type=Path, required=True)
    final = commands.add_parser("verdict")
    final.add_argument("--tree", required=True)
    for sub in (fetch, speed_parser, trials_parser, train, refuse, du, final):
        sub.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return its exit status."""
    args = build_parser().parse_args(argv)
    if args.command == "scratch-entries":
        _say(*scratch_entries(args.scratch))
        return 0
    commands = {"fetch-model": fetch_model, "speed": speed, "trials": trials, "checkpoint": checkpoint,
                "refusal": refusal, "du-new": du_new, "ascii": ascii_report, "verdict": verdict}
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
