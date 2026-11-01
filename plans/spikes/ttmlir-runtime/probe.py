"""Exploratory probe of the P17.48 driver and executor on ttsim (plans/spikes/ttmlir-runtime.md, 2026-10-30).

    uv run python plans/spikes/ttmlir-runtime/probe.py <case>

<case> is add (TT-Metal: add.mlir through --ttir-to-ttmetal-pipeline),
add-ttnn (TT-NN: the same module through --ttir-to-ttnn-backend-pipeline),
add-bf16-ttnn (TT-NN: add_bf16.mlir, the same add on bf16), or
histo (TT-NN: histo.mlir through --ttir-to-ttnn-backend-pipeline). In
$LASSI_RUNS_ROOT/ttmlir-runtime-spike/<stamp>-<case> it copies the module,
compiles it with the runtime prefix's ttmlir-opt and ttmlir-translate for the
recorded ttsim descriptor (these are the pinned tools on a hand-written
module, run directly, each under a timeout and the 1-byte core limit), writes
the inputs with lassi.harness.lassi_io generate_inputs, runs the flatbuffer
with the ttmlir-ttsim executor (in the sandbox, on ttsim), and compares the
outputs with a numpy reference (compare.py, run by the runtime venv's
Python, which holds the pinned numpy). It prints one JSON summary. Every
result is exploratory: the tree that runs it is not a clean commit.
"""

from __future__ import annotations

import json
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

from lassi.core.interfaces import Limits
from lassi.executors.ttmlir_ttsim import TtmlirTtsimExecutor
from lassi.executors.workdir import runs_root
from lassi.harness.lassi_io import generate_inputs
from lassi.toolchains.pins import read_pin

HERE = Path(__file__).resolve().parent
TOOL_TIMEOUT_S = 240
LIMITS = Limits(wall_s=420.0, memory_mb=16384, cpus=8)
CASES = {
    "add": {
        "module": "add.mlir",
        "pipeline": "--ttir-to-ttmetal-pipeline=system-desc-path={desc} enable-multicast-inference=false",
        "translate": "--ttmetal-to-flatbuffer",
        "suffix": ".ttm",
        "seed": 7,
        "inputs": [
            {"name": "a", "dtype": "f32", "shape": [64, 128], "dist": "uniform", "lo": -1.0, "hi": 1.0},
            {"name": "b", "dtype": "f32", "shape": [64, 128], "dist": "uniform", "lo": -1.0, "hi": 1.0},
        ],
        "outputs": ["c"],
    },
    "add-ttnn": {
        "module": "add.mlir",
        "pipeline": "--ttir-to-ttnn-backend-pipeline=system-desc-path={desc}",
        "translate": "--ttnn-to-flatbuffer",
        "suffix": ".ttnn",
        "seed": 7,
        "inputs": [
            {"name": "a", "dtype": "f32", "shape": [64, 128], "dist": "uniform", "lo": -1.0, "hi": 1.0},
            {"name": "b", "dtype": "f32", "shape": [64, 128], "dist": "uniform", "lo": -1.0, "hi": 1.0},
        ],
        "outputs": ["c"],
    },
    "add-bf16-ttnn": {
        "module": "add_bf16.mlir",
        "pipeline": "--ttir-to-ttnn-backend-pipeline=system-desc-path={desc}",
        "translate": "--ttnn-to-flatbuffer",
        "suffix": ".ttnn",
        "seed": 7,
        "inputs": [
            {"name": "a", "dtype": "bf16", "shape": [64, 128], "dist": "uniform", "lo": -1.0, "hi": 1.0},
            {"name": "b", "dtype": "bf16", "shape": [64, 128], "dist": "uniform", "lo": -1.0, "hi": 1.0},
        ],
        "outputs": ["c"],
    },
    "histo": {
        "module": "histo.mlir",
        "pipeline": "--ttir-to-ttnn-backend-pipeline=system-desc-path={desc}",
        "translate": "--ttnn-to-flatbuffer",
        "suffix": ".ttnn",
        "seed": 11,
        "inputs": [
            {"name": "counts0", "dtype": "i32", "shape": [4096], "dist": "integers", "lo": 0, "hi": 1},
            {"name": "index", "dtype": "i32", "shape": [16384], "dist": "integers", "lo": 0, "hi": 4096},
            {"name": "ones", "dtype": "i32", "shape": [16384], "dist": "integers", "lo": 1, "hi": 2},
        ],
        "outputs": ["counts"],
    },
}


def tool(argv: list[str], cwd: Path) -> dict[str, object]:
    """Run one pinned compiler tool under a timeout; return its argv, status, and stream tails."""
    started = time.monotonic()
    done = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=TOOL_TIMEOUT_S, check=False)
    return {
        "argv": argv,
        "rc": done.returncode,
        "seconds": round(time.monotonic() - started, 2),
        "stderr_tail": done.stderr[-2000:],
    }


def compile_case(case: dict, workdir: Path, desc: Path, runtime: Path, pin: dict[str, str]) -> tuple[list, Path]:
    """Lower and translate the case's module in `workdir`; return the steps and the flatbuffer's path."""
    module = workdir / case["module"]
    shutil.copyfile(HERE / case["module"], module)
    lowered = workdir / (module.stem + ".lowered.mlir")
    flatbuffer = workdir / (module.stem + case["suffix"])
    opt, translate = str(runtime / pin["TTMLIR_OPT"]), str(runtime / pin["TTMLIR_TRANSLATE"])
    steps = [tool([opt, case["pipeline"].format(desc=desc), module.name, "-o", lowered.name], workdir)]
    if steps[-1]["rc"] == 0:
        steps.append(tool([translate, case["translate"], lowered.name, "-o", flatbuffer.name], workdir))
    return steps, flatbuffer


def compare(case_name: str, workdir: Path, runtime: Path, pin: dict[str, str]) -> dict[str, object]:
    """Run compare.py with the runtime venv's Python (its pinned numpy); return its JSON."""
    python = str(runtime / pin["VENV_DIR"] / "bin" / "python")
    done = subprocess.run(
        [python, "-I", str(HERE / "compare.py"), case_name, str(workdir)],
        capture_output=True,
        text=True,
        timeout=TOOL_TIMEOUT_S,
        check=False,
    )
    if done.returncode != 0:
        return {"rc": done.returncode, "stderr_tail": done.stderr[-2000:]}
    return json.loads(done.stdout)


def main() -> int:
    """Compile, run, and compare one case; print the summary."""
    case_name = sys.argv[1]
    case = CASES[case_name]
    resource.setrlimit(resource.RLIMIT_CORE, (1, 1))
    executor = TtmlirTtsimExecutor()
    install = executor.install()
    pin = read_pin("tt-mlir-runtime")
    runtime = install.root / pin["PREFIX_NAME"]
    workdir = runs_root() / "ttmlir-runtime-spike" / f"{time.strftime('%Y%m%d-%H%M%S')}-{case_name}"
    workdir.mkdir(parents=True)
    steps, flatbuffer = compile_case(case, workdir, install.system_desc, runtime, pin)
    summary: dict[str, object] = {"case": case_name, "workdir": str(workdir), "device": executor.device()}
    summary["compile"] = steps
    if not flatbuffer.is_file():
        print(json.dumps(summary, indent=1))
        return 1
    summary["flatbuffer_head"] = flatbuffer.read_bytes()[:12].hex()
    (workdir / "@inputs").mkdir()
    generate_inputs(case["inputs"], case["seed"], workdir / "@inputs")
    args = [f"@inputs/{entry['name']}.lassiio" for entry in case["inputs"]]
    args += [f"{name}.lassiio" for name in case["outputs"]]
    run = executor.run(flatbuffer, args, LIMITS)
    summary["run"] = {
        "exit_code": run.exit_code,
        "hang": run.hang,
        "wall_s_simulator": round(run.wall_s, 1),
        "outputs": sorted(run.output_files),
        "sim_ub": run.sim_ub,
        "sim_gap": run.sim_gap,
        "diagnostics": [f"{item.stage}/{item.code}: {item.message[:400]}" for item in run.diagnostics],
        "stdout_tail": run.stdout[-1500:],
        "stderr_tail": run.stderr[-2500:],
    }
    if run.exit_code == 0:
        summary["compare"] = compare(case_name, workdir, runtime, pin)
    print(json.dumps(summary, indent=1))
    return 0 if run.exit_code == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
