"""Static checks of gate part (a)'s batch and its helper's checks (task P17.11).

The batch is plans/runs/p17-gate-a/run.sh, started on alpha01 by `rx job
start`, and gate.py beside it holds its checks and records. Bible: Build
Roadmap (P17 row, Gate part (a)), Agent Rules 6, 7, and 10; AGENTS.md,
Remote Execution; plans/p17-portable.md, Constraints (nothing on alpha01
opens /dev/kfd or a /dev/dri node; no RNGD card); PHASE-NOTES P17 (never
uv's --torch-backend on alpha01; every hipcc command names its target);
plans/LESSONS.md, alpha01 (the gate refuses command text naming a disabled
device's tools, a 1-byte core limit set by prlimit, a normally exiting
shell between strace and the program, the stdout budget).

What these tests fix:

- The launch command passes the gate's own deny and device patterns
  (tools/server/config.default.json), and neither file names a device tool
  or a device path the gate's device classes list; gate.py starts no
  process.
- run.sh sets the core limit with prlimit and checks it with getrlimit;
  syncs only from the lock with the cpu extra and never passes uv's
  --torch-backend; runs every lassi command with graphics off; puts a bash
  that exits normally between strace and lassi; reads the owner's gate
  STOP before every step; and runs its steps in the order run.sh's header
  lists. Every hipcc command (none today) would name --offload-arch=gfx942.
- gate.py's pins agree with the bible (gate model), the P17.1 spike (file
  digests), pyproject.toml (the cpu extra), run.sh (the freeze pins and the
  refusal recipes), and the refusal fixtures; its functions stay under 60
  lines.
- gate.py's checks, on SYNTHETIC inputs: the strace reader, the node calls,
  a refusal, the hf_local run, a checkpoint, the model pin, and the verdict.

Every strace line, record, and report here is SYNTHETIC; nothing runs the
batch, and no value in this module is a measurement.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
BATCH_DIR = REPO / "plans" / "runs" / "p17-gate-a"
RUN_SH = BATCH_DIR / "run.sh"
GATE_PY = BATCH_DIR / "gate.py"
GATE_CONFIG = REPO / "tools" / "server" / "config.default.json"
BIBLE = REPO / "docs" / "BIBLE.md"
SPIKE = REPO / "plans" / "spikes" / "p17-frameworks.md"
RECIPES = REPO / "tests" / "fixtures" / "recipes"
LAUNCH = "bash plans/runs/p17-gate-a/run.sh"
# Where each step starts in run.sh, in the order its header lists.
STEP_MARKERS = (
    "preflight || exit 2", "# Header.", "step env ", "check fetch-model ", "step fetch-bench ", "step fetch-upstream ",
    "step extract-assets ", "check speed ", "step hf-run ", 'step "train-$method"', 'short="refuse-', "# du:",
    "# close:",
)
# Device tools no file of the batch may name, whatever the gate's configuration says.
DEVICE_TOOLS = ("rocm-smi", "amd-smi", "rocminfo", "nvidia-smi", "tt-smi", "furiosa-smi", "offload-arch")
REFUSALS = {
    "p17-gpu-hf-cuda": ("model.device", "cuda"),
    "p17-gpu-hf-rocm": ("model.device", "rocm"),
    "p17-gpu-exec-cuda": ("executor.cuda.device", "cuda"),
    "p17-gpu-exec-rocm": ("executor.cuda.device", "rocm"),
}


def gate_module() -> ModuleType:
    """Import plans/runs/p17-gate-a/gate.py as a module, failing the test clearly while it does not exist."""
    if not GATE_PY.is_file():
        pytest.fail(f"{GATE_PY} does not exist yet (task P17.11)")
    spec = importlib.util.spec_from_file_location("p17_gate_a", GATE_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_text() -> str:
    """Return run.sh's text, failing the test clearly while it does not exist."""
    if not RUN_SH.is_file():
        pytest.fail(f"{RUN_SH} does not exist yet (task P17.11)")
    return RUN_SH.read_bytes().decode("ascii")


def code_lines(text: str) -> list[str]:
    """Return the lines of a shell script that are not comments, with trailing comments left in place."""
    return [line for line in text.splitlines() if not line.lstrip().startswith("#")]


def gate_patterns() -> tuple[list[str], list[str]]:
    """Return the gate's deny patterns and the patterns of every device class it lists."""
    config = json.loads(GATE_CONFIG.read_text(encoding="ascii"))
    devices = [pattern for device in config["devices"].values() for pattern in device["patterns"]]
    return config["deny_patterns"], devices


# ---------------------------------------------------------------------------
# The files and the launch command


@pytest.mark.parametrize("path", [RUN_SH, GATE_PY], ids=["run.sh", "gate.py"])
def test_the_batch_files_are_ascii_with_lf(path: Path) -> None:
    if not path.is_file():
        pytest.fail(f"{path} does not exist yet (task P17.11)")
    data = path.read_bytes()
    assert data.isascii() and b"\r" not in data and data.endswith(b"\n")


def test_the_launch_command_passes_the_gate_and_is_the_one_run_sh_documents() -> None:
    deny, devices = gate_patterns()
    for pattern in deny + devices:
        assert not re.search(pattern, LAUNCH), pattern
    header = " ".join(line.lstrip("#").strip() for line in run_text().splitlines()[:8])
    assert "rx.py job start --big --name p17-gate-a --timeout 21600 -- \\ 'bash plans/runs/p17-gate-a/run.sh'" in header


@pytest.mark.parametrize("path", [RUN_SH, GATE_PY], ids=["run.sh", "gate.py"])
def test_no_device_tool_is_named(path: Path) -> None:
    if not path.is_file():
        pytest.fail(f"{path} does not exist yet (task P17.11)")
    text = path.read_bytes().decode("ascii")
    for tool in DEVICE_TOOLS:
        assert tool not in text, tool


def test_run_sh_names_no_device_path_or_tool_the_gate_lists() -> None:
    _, devices = gate_patterns()
    text = run_text()
    for pattern in devices:
        assert not re.search(pattern, text), pattern


def test_gate_py_starts_no_process() -> None:
    tree = ast.parse(GATE_PY.read_bytes().decode("ascii"))
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert not imported & {"subprocess", "pty", "multiprocessing"}, imported
    assert "os.system" not in GATE_PY.read_text(encoding="ascii")


def test_no_command_deletes_escalates_or_reaches_another_host() -> None:
    for line in code_lines(run_text()):
        assert not re.search(r"\b(rm|rmdir|sudo|su|ssh|scp|rsync|sftp|systemctl|curl|wget)\s", line), line


# ---------------------------------------------------------------------------
# run.sh's safety rules


def test_the_core_limit_is_one_byte_by_prlimit_and_checked_by_getrlimit() -> None:
    text = run_text()
    assert "prlimit --pid $$ --core=1:1" in text
    assert "resource.getrlimit(resource.RLIMIT_CORE)==(1,1)" in text
    assert "ulimit" not in "\n".join(code_lines(text))


def test_uv_syncs_only_the_lock_with_the_cpu_extra_and_never_a_torch_backend() -> None:
    code = "\n".join(code_lines(run_text()))
    assert "--torch-backend" not in code
    syncs = [line for line in code.splitlines() if "uv sync" in line]
    assert syncs and all("--frozen" in line and "--extra cpu" in line for line in syncs), syncs
    assert not re.search(r"--extra (cuda|rocm)|uv pip install|pip install", code)
    assert "UV_TORCH_BACKEND" in code, "the backend variable is unset before any uv call"


def test_every_hipcc_command_names_its_target() -> None:
    for path in (RUN_SH, GATE_PY):
        for line in path.read_text(encoding="ascii").splitlines():
            if "hipcc" in line and not line.lstrip().startswith("#"):
                assert "--offload-arch=gfx942" in line, line


def test_every_lassi_command_runs_with_graphics_off() -> None:
    text = run_text()
    calls = [line for line in code_lines(text) if '"$LASSI"' in line]
    assert len(calls) == 3, calls
    assert all("--graphics off" in line for line in calls), calls
    assert "export LASSI_GRAPHICS=off" in text


def test_a_shell_that_exits_normally_sits_between_strace_and_lassi() -> None:
    code = code_lines(run_text())
    starts = [index for index, line in enumerate(code) if re.search(r"\bstrace -f\b", line)]
    assert starts, "the refusals run under strace"
    for index in starts:
        block = " ".join(code[index:index + 3])
        assert "-- \\ bash -c '\"$@\"; exit $?'" in " ".join(block.split()), block


def test_every_step_reads_the_owner_stop_before_it_starts() -> None:
    body = run_text().split("\nstep() {\n", 1)[1].split("\n}\n", 1)[0]
    assert body.find('[ -e "$GATE_STOP" ]') != -1
    assert body.find('[ -e "$GATE_STOP" ]') < body.find("timeout -k"), body


def test_the_steps_run_in_the_design_order() -> None:
    text = run_text()
    found = [text.find(marker) for marker in STEP_MARKERS]
    assert -1 not in found and found == sorted(found), dict(zip(STEP_MARKERS, found, strict=True))


# ---------------------------------------------------------------------------
# gate.py's pins agree with their sources


def test_the_model_pin_is_the_bible_s_and_the_spike_s() -> None:
    pin = gate_module().MODEL_PIN
    line = next(line for line in BIBLE.read_text(encoding="utf-8").splitlines() if line.startswith("- Gate model"))
    assert f"{pin['files']} files, {pin['bytes']:,} bytes" in line, line
    assert pin["sha256"]["model.safetensors"] in line, line
    spike = SPIKE.read_text(encoding="utf-8")
    for name, digest in pin["sha256"].items():
        assert f"{name} {digest}" in spike or digest in line, name


def test_the_framework_pins_are_the_cpu_extra_and_run_sh_s() -> None:
    pins = gate_module().FRAMEWORK_PINS
    extra = re.search(r"^cpu = \[(.*)\]$", (REPO / "pyproject.toml").read_text(encoding="ascii"), re.MULTILINE)
    assert extra is not None
    declared = dict(item.strip('" ').split("==") for item in extra.group(1).split(","))
    assert {name: version.split("+")[0] for name, version in pins.items()} == declared
    assert pins["torch"] == "2.14.1+cpu"
    freeze = re.search(r"^FREEZE_PINS=\((.*)\)$", run_text(), re.MULTILINE)
    assert freeze is not None and freeze.group(1).split() == [f"{name}=={version}" for name, version in pins.items()]


def test_the_refusal_table_matches_run_sh_and_the_fixtures() -> None:
    gate = gate_module()
    assert gate.REFUSALS == REFUSALS
    listed = re.search(r"^REFUSALS=\((.*)\)$", run_text(), re.MULTILINE)
    assert listed is not None and listed.group(1).split() == list(REFUSALS)
    for name, (key, kind) in REFUSALS.items():
        text = (RECIPES / f"{name}.yaml").read_text(encoding="ascii")
        data = yaml.safe_load(text)
        section = data["model"] if key == "model.device" else data["executor"]["cuda"]
        assert section["device"] == {"kind": kind, "indices": [0]}, name


def test_gate_py_functions_stay_under_sixty_lines_with_docstrings() -> None:
    tree = ast.parse(GATE_PY.read_bytes().decode("ascii"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            assert node.end_lineno - node.lineno < 60, node.name
            assert ast.get_docstring(node) or node.name in ("pins_ok", "cpu_ok", "ok", "passed"), node.name


# ---------------------------------------------------------------------------
# gate.py's checks on SYNTHETIC inputs


SYNTHETIC_STRACE = [
    '4242  newfstatat(AT_FDCWD, "/dev/kfd", {st_mode=S_IFCHR|0666, st_rdev=makedev(0xeb, 0)}, AT_SYMLINK_NOFOLLOW) = 0',
    '4242  faccessat2(AT_FDCWD, "/dev/kfd", R_OK|W_OK, 0) = -1 EACCES (Permission denied)',
    '4242  openat(AT_FDCWD, "/proc/cpuinfo", O_RDONLY|O_CLOEXEC) = 3',
    '[pid  4243] openat(AT_FDCWD, "/dev/null", O_RDWR|O_CLOEXEC) = 4',
    '4242  newfstatat(AT_FDCWD, "/dev/nvidiactl", 0x7ffc, AT_SYMLINK_NOFOLLOW) = -1 ENOENT (No such file or directory)',
    "4242  +++ exited with 2 +++",
]


def test_the_strace_reader_finds_metadata_calls_on_gpu_nodes_and_no_open() -> None:
    gate = gate_module()
    calls = gate.strace_calls(SYNTHETIC_STRACE)
    assert ("openat", "/dev/null") in calls and ("openat", "/proc/cpuinfo") in calls
    nodes = gate.node_calls(calls)
    assert nodes["gpu_node_opens"] == []
    assert nodes["gpu_node_metadata"] == [
        "newfstatat /dev/kfd", "faccessat2 /dev/kfd", "newfstatat /dev/nvidiactl"
    ]


@pytest.mark.parametrize(
    "line",
    [
        '4242  openat(AT_FDCWD, "/dev/kfd", O_RDWR|O_CLOEXEC) = -1 EACCES (Permission denied)',
        '[pid  77] openat(AT_FDCWD, "/dev/dri/renderD128", O_RDWR <unfinished ...>',
        'open("/dev/nvidia0", O_RDWR) = -1 ENOENT (No such file or directory)',
        '4242  openat(AT_FDCWD, "/dev/nvidia-caps/nvidia-cap1", O_RDONLY) = -1 ENOENT (No such file or directory)',
    ],
)
def test_any_open_of_a_gpu_node_is_found(line: str) -> None:
    gate = gate_module()
    assert len(gate.node_calls(gate.strace_calls([line]))["gpu_node_opens"]) == 1


def test_a_refusal_passes_only_with_status_2_the_message_no_directory_and_no_open() -> None:
    gate = gate_module()
    message = (
        "lassi run: x.yaml: executor.cuda.device names rocm 0, which this host cannot provide: /dev/kfd exists but "
        "SYNTHETIC; a run never falls back to another device"
    )
    good = gate.refusal_checks("p17-gpu-exec-rocm", 2, message, False, [])
    assert all(good.values()), good
    assert not gate.refusal_checks("p17-gpu-exec-rocm", 1, message, False, [])["exit_status_2"]
    assert not gate.refusal_checks("p17-gpu-hf-rocm", 2, message, False, [])["names_the_device"]
    assert not gate.refusal_checks("p17-gpu-exec-rocm", 2, message, True, [])["no_run_directory"]
    assert not gate.refusal_checks("p17-gpu-exec-rocm", 2, message, False, ["openat /dev/kfd"])["no_gpu_node_opened"]


def synthetic_provenance(**changes: Any) -> dict[str, Any]:
    """Return a SYNTHETIC hf_local run provenance that meets the gate, with top-level keys replaced."""
    record = {
        "status": "complete",
        "device_records": [{"key": "model.device", "kind": "cpu", "framework": "torch",
                            "framework_version": "2.14.1+cpu"}],
        "serving": {"revision": "ea3f2471cf1b1f0db85067f1ef93848e38e88c25", "versions": {
            "torch": "2.14.1+cpu", "transformers": "5.18.0", "tokenizers": "0.23.2", "huggingface-hub": "1.33.0",
            "safetensors": "0.8.0"}},
    }
    record.update(changes)
    return record


ROWS = [{"direction": "omp-cuda", "end_reason": "correction-cap"}, {"direction": "cuda-omp", "end_reason": None}]


def test_the_run_checks_need_both_directions_past_the_baseline_and_the_records() -> None:
    gate = gate_module()
    revision = "ea3f2471cf1b1f0db85067f1ef93848e38e88c25"
    assert all(gate.run_checks(synthetic_provenance(), ROWS, revision).values())
    stopped = [ROWS[0], {"direction": "cuda-omp", "end_reason": "baseline-run"}]
    assert not gate.run_checks(synthetic_provenance(), stopped, revision)["every_trial_past_the_baseline"]
    assert not gate.run_checks(synthetic_provenance(), ROWS[:1], revision)["both_directions_once"]
    assert not gate.run_checks(synthetic_provenance(status="failed"), ROWS, revision)["status_complete"]
    gpu = synthetic_provenance(device_records=[{"key": "model.device", "kind": "rocm"}])
    assert not gate.run_checks(gpu, ROWS, revision)["model_device_record_cpu"]
    assert not gate.run_checks(synthetic_provenance(serving={}), ROWS, revision)["serving_versions_recorded"]


def test_a_checkpoint_needs_the_cpu_record_and_the_pins_in_provenance_and_each_checkpoint() -> None:
    gate = gate_module()
    pins = {"build": {"version": "2.14.1+cpu"}, "packages": dict(gate.FRAMEWORK_PINS)}
    source = {"device_records": [{"kind": "cpu"}], "framework_pins": pins}
    provenance = {"status": "complete", "steps": 2, **source}
    assert all(gate.checkpoint_checks(provenance, [source]).values())
    assert not gate.checkpoint_checks(provenance, [])["checkpoint_records"]
    assert not gate.checkpoint_checks({**provenance, "steps": 1}, [source])["steps"]
    other = {**source, "framework_pins": {**pins, "packages": {**pins["packages"], "trl": "0.0.0"}}}
    assert not gate.checkpoint_checks(provenance, [other])["checkpoint_records"]


def test_the_model_check_names_each_difference() -> None:
    gate = gate_module()
    expected = {"files": 2, "bytes": 10, "sha256": {"config.json": "a" * 64}}
    assert gate.model_problems({"files": 2, "bytes": 10, "sha256": {"config.json": "a" * 64}}, expected) == []
    problems = gate.model_problems({"files": 3, "bytes": 10, "sha256": {"config.json": None}}, expected)
    assert problems == ["files 3 != 2", f"sha256 of config.json None != {'a' * 64}"]


def test_the_verdict_passes_only_when_every_gate_check_passes(tmp_path: Path) -> None:
    gate = gate_module()
    steps = ["step\trc\tseconds\tlimit_s"] + [
        f"{name}\t0\t1\t1" for name in ("env", "env-pins", "fetch-model", "fetch-bench", "fetch-upstream",
                                         "extract-assets", "hf-run", "train-sft", "train-dpo", "train-grpo")
    ]
    (tmp_path / "steps.tsv").write_bytes(("\n".join(steps) + "\n").encode("ascii"))
    for name in ("model.json", "trials.json"):
        (tmp_path / name).write_bytes(b'{"passed": true}\n')
    for method in ("sft", "dpo", "grpo"):
        (tmp_path / f"train-{method}").mkdir()
        (tmp_path / f"train-{method}" / "checks.json").write_bytes(b'{"passed": true}\n')
    refusals = {name: {"passed": True} for name in REFUSALS}
    (tmp_path / "gpu-refusals.json").write_bytes(json.dumps(refusals).encode("ascii"))
    assert all(passed for _, passed in gate.gate_lines(tmp_path, "clean"))
    assert not all(passed for _, passed in gate.gate_lines(tmp_path, "dirty"))
    refusals["p17-gpu-exec-rocm"]["passed"] = False
    (tmp_path / "gpu-refusals.json").write_bytes(json.dumps(refusals).encode("ascii"))
    assert [check for check, passed in gate.gate_lines(tmp_path, "clean") if not passed] == [
        "p17-gpu-exec-rocm refused before any directory, no node opened"
    ]
    refusals["p17-gpu-exec-rocm"]["passed"] = True
    (tmp_path / "gpu-refusals.json").write_bytes(json.dumps(refusals).encode("ascii"))
    pins_missing = [line.replace("env-pins\t0", "env-pins\t1") for line in steps]
    (tmp_path / "steps.tsv").write_bytes(("\n".join(pins_missing) + "\n").encode("ascii"))
    assert [check for check, passed in gate.gate_lines(tmp_path, "clean") if not passed] == [
        "env synced from the lock with the cpu extra and the five pins"
    ]
