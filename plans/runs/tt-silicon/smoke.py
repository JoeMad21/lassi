"""One Tier A TT reference on one Wormhole card, checked against its C++ counterpart (the owner's window of
2026-10-23, 00:00 to 06:00 EDT; the owner enabled the gate's tt_silicon class and said go).

Usage on the build host, from a clean commit, through `uv run tools/rx.py run`, one item and one card per call:

    uv run python plans/runs/tt-silicon/smoke.py ITEM --device N    # silicon: card N alone (TT_VISIBLE_DEVICES=N)
    uv run python plans/runs/tt-silicon/smoke.py ITEM --sim         # the same steps with ttsim as the device

ITEM is a tt-pairs-v0 item (eltwise_binary, eltwise_sfpu, loopback, matmul_multi_core, matmul_single_core).
The steps are those of tests/bench/test_tt_pairs_remote.py: fetch the suite's pinned files (idempotent), build the
TT reference with ttmetal-host and the C++ counterpart with gcc-native, stage the item's held-out inputs, run the
C++ side through the native executor (in the sandbox), run the TT side, and compare every output with binary_io
under the item's declared tolerance. With --sim the TT side runs through the ttsim executor. Without it the TT side
runs once on card N with the ttsim row's environment minus TT_METAL_SIMULATOR and plus TT_VISIBLE_DEVICES=N, so
UMD opens that one card and no other (Agent Rule 9: one placement, no looped opens). That run is outside the
sandbox, which gives a program no device node; the program is the item's tracked upstream reference, never model
output (Agent Rule 6 binds model-generated code). Nothing resets, flashes, or reconfigures a card.

The record, <runs root>/tt-silicon/<stamp>-<item>-<device>/record.json, holds the commit and its clean state, the
toolchain pins, the device (card N's type, serial, and firmware from sysfs, and the tenstorrent driver's version, or
ttsim v1.3.4 on tt-metal 5280a9cf), the date, both runs' exit status and hang, and the oracle's statistics. A wall
time is never performance (Agent Rule 2 for the simulator; one run on silicon is a smoke check, not a timing).
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from lassi import bench
from lassi.bench import registry
from lassi.core.interfaces import Limits, RunResult
from lassi.core.runner import BuiltToolchain, build_toolchain
from lassi.executors.native import NativeExecutor, _regular_files
from lassi.executors.ttsim import TtsimExecutor, program_environment
from lassi.oracles.binary_io import BinaryIOOracle
from lassi.toolchains import pins as pins_module

REPO = Path(__file__).resolve().parents[3]
MANIFEST = REPO / "assets" / "bench" / "tt-pairs-v0.yaml"
TO_TT, TO_CPP = bench.Direction("cpp", "tt"), bench.Direction("tt", "cpp")
# The remote test's limits; on silicon the wall limit is the program's timeout.
LIMITS = Limits(wall_s=120.0, memory_mb=8192, cpus=16)
SYSFS = Path("/sys/class/tenstorrent")
CARD_FILES = ("tt_card_type", "tt_serial", "tt_fw_bundle_ver", "tt_arc_fw_ver", "tt_eth_fw_ver")


def fetch_suite() -> None:
    """Fetch tt-pairs-v0's pinned files with tools/fetch_bench.py, as the remote test does; exit on a failure."""
    spec = importlib.util.spec_from_file_location("fetch_bench", REPO / "tools" / "fetch_bench.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.main([str(MANIFEST)]) != 0:
        sys.exit("smoke: tools/fetch_bench.py did not verify tt-pairs-v0's pinned files")


def build(built: BuiltToolchain, files: dict, harness: dict, workdir: Path) -> Path:
    """Build `files` with `harness` in a new `workdir`; return the artifact or exit with the compiler's tail."""
    workdir.mkdir(parents=True)
    result = built.toolchain.build(files, workdir, harness=harness)
    errors = [entry for entry in result.diagnostics if entry.severity == "error"]
    if result.artifact is None or errors:
        stderr = (workdir / "compile.stderr").read_bytes().decode("utf-8", "replace")
        sys.exit(f"smoke: the build in {workdir} failed:\n{stderr[-4000:]}")
    return Path(result.artifact)


def card_facts(device: int) -> dict[str, str]:
    """Return card `device`'s sysfs facts and the driver version (read-only files; no device node is opened)."""
    card = SYSFS / f"tenstorrent!{device}"
    facts = {name: (card / name).read_text().strip() for name in CARD_FILES if (card / name).is_file()}
    version = Path("/sys/module/tenstorrent/version")
    facts["driver"] = version.read_text().strip() if version.is_file() else "unknown"
    return facts


def run_on_card(artifact: Path, args: list[str], device: int) -> RunResult:
    """Run `artifact` once on card `device` in its build directory; return the run with the files it wrote."""
    workdir = artifact.parent
    tree = Path(os.environ["LASSI_TOOLCHAINS"]).resolve() / pins_module.read_pin("tt-metal")["PREFIX_NAME"]
    environment = {key: value for key, value in program_environment(_Install(tree), workdir, LIMITS).items()
                   if key != "TT_METAL_SIMULATOR"}
    (workdir / "@ttsim" / "tmp").mkdir(parents=True, exist_ok=True)
    environment.update({"TT_VISIBLE_DEVICES": str(device), "HOME": str(workdir),
                        "TMPDIR": str(workdir / "@ttsim" / "tmp")})
    before = _regular_files(workdir)
    command = ["prlimit", "--core=1:1", "--", "timeout", "--kill-after=30", str(int(LIMITS.wall_s * 2)),
               str(artifact), *args]
    start = time.monotonic()
    done = subprocess.run(command, cwd=workdir, env=environment, capture_output=True, text=True, check=False)
    wall = time.monotonic() - start
    after = _regular_files(workdir)
    outputs = {rel: after[rel] for rel in sorted(after) if rel not in before and not rel.startswith("@")}
    return RunResult(exit_code=done.returncode, hang=done.returncode in (124, 137), stdout=done.stdout,
                     stderr=done.stderr, output_files=outputs, wall_s=wall)


class _Install:
    """The one field of a ttsim install that program_environment reads besides the library: the tt-metal tree."""

    def __init__(self, tree: Path) -> None:
        """Hold `tree`; the library is never read, since its variable is dropped for silicon."""
        self.tree = tree
        self.library = Path("/nonexistent-on-silicon")


def provenance(target: str, facts: dict[str, str]) -> dict:
    """Return the run's provenance: commit, clean state, pins, device, and date."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True,
                                text=True).stdout.strip())
    pins = {name: pins_module.read_pin(name).get("COMMIT") or pins_module.read_pin(name).get("VERSION", "")
            for name in ("tt-metal", "ttsim", "gcc")}
    device = ("ttsim v1.3.4 on tt-metal 5280a9cf (simulator)" if target == "ttsim"
              else f"Tenstorrent Wormhole {facts.get('tt_card_type', '?')} card {target}")
    return {"commit": commit, "dirty": dirty, "pins": pins, "device": device, "card": facts,
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}


def main(argv: list[str] | None = None) -> int:
    """Run one item on one card (or ttsim); print and write the record; return 0 when every output passes."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("item")
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--device", type=int, help="the one card to open, 0 to 7")
    where.add_argument("--sim", action="store_true", help="run the TT side on ttsim instead")
    a = parser.parse_args(argv)
    target = "ttsim" if a.sim else str(a.device)
    facts = {} if a.sim else card_facts(a.device)
    fetch_suite()
    suite = bench.load_suite(MANIFEST)
    root = registry.sources_dir(Path(os.environ["LASSI_SCRATCH"]), suite)
    spec = suite.item(a.item, purpose="eval")
    toolchains = Path(os.environ["LASSI_TOOLCHAINS"])
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path(os.environ["LASSI_RUNS_ROOT"]) / "tt-silicon" / f"{stamp}-{a.item}-{target}"
    cpp = build(build_toolchain("gcc-native", toolchains), suite.reference_target(a.item, TO_CPP, root, purpose="eval"),
                suite.support_files(a.item, root, purpose="eval", language="cpp"), base / "cpp")
    tt = build(build_toolchain("ttmetal-host", toolchains), suite.reference_target(a.item, TO_TT, root, purpose="eval"),
               suite.support_files(a.item, root, purpose="eval", language="tt"), base / "tt")
    cpp_run = NativeExecutor().run(cpp, registry.stage_inputs(spec, cpp.parent), LIMITS)
    tt_args = registry.stage_inputs(spec, tt.parent)
    tt_run = TtsimExecutor().run(tt, tt_args, LIMITS) if a.sim else run_on_card(tt, tt_args, a.device)
    oracle = BinaryIOOracle(metric=spec.tolerance.metric, threshold=spec.tolerance.threshold)
    stats = oracle.compare(cpp_run.output_files, tt_run.output_files, sides=("cpp reference", f"tt on {target}"))
    record = {"item": a.item, "target": target, "provenance": provenance(target, facts),
              "cpp_run": {"exit_code": cpp_run.exit_code, "hang": cpp_run.hang},
              "tt_run": {"exit_code": tt_run.exit_code, "hang": tt_run.hang, "wall_s_not_performance": tt_run.wall_s,
                         "stdout_tail": tt_run.stdout[-3000:], "stderr_tail": tt_run.stderr[-3000:]},
              "outputs": [{"name": s.name, "passed": s.passed, "pcc": s.pcc, "max_abs": s.max_abs, "note": s.note}
                          for s in stats],
              "tolerance": {"metric": spec.tolerance.metric, "threshold": spec.tolerance.threshold}}
    (base / "record.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(json.dumps(record, indent=2, default=str))
    passed = tt_run.exit_code == 0 and not tt_run.hang and bool(stats) and all(s.passed for s in stats)
    print(f"SMOKE {a.item} on {target}: {'PASS' if passed else 'FAIL'}; record {base / 'record.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
