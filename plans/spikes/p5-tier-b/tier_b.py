"""Helper for the P5.3 batch, plans/spikes/p5-tier-b/run.sh (report: plans/spikes/p5-tier-b.md).

The batch checks the TurboQuant codec's programs (JoeMad21/TurboQuant-Tenstorrent at CODEC_COMMIT) against the
joint pin: tt-metal 5280a9cf on ttsim v1.3.4, a simulator, never silicon (Agent Rule 2). Subcommands:

- prepare (standard library): check the fetched checkout is the pinned commit; write report/source.tsv (path,
  bytes, sha256, git blob of every tracked file; no file text), report/license.txt, and report/programs.tsv (each
  program's host source and the kernels it names).
- copy (standard library; run.sh's own bounded step, only with --alpha01-copy): compare a copy of the codec found
  on alpha01 with the pinned checkout file by file (sha256 only, read only; git state read with
  --no-optional-locks and core.fsmonitor off; never a remote URL); report/alpha01-copy.json.
- batch (needs lassi; run under the slot's uv environment): a device-visibility probe through the ttsim executor,
  then each host program built with the Toolchain ttmetal-host in the compile sandbox, as written ("asis") and as
  a changed copy with its one device-id literal changed from 6 to 0 in the run directory only ("dev0", since ttsim
  presents one chip, id 0), and each built program run through lassi.executors.ttsim on N = 32 vectors. Per
  program and variant it records the build result and diagnostics, exit status, hang, sim_ub, sim_gap, findings,
  kernel JIT errors, simulator wall time (sizing only), each input file and where it came from, each output file's
  sha256, and the unpack_to_dest facts read from the JIT's generated descriptors. Order: every "asis" build; the
  six multi_tile stages' "dev0" build and run in pipeline order (each stage reads the previous stage's output, and
  the first stages read the codec's committed dumps); the "asis" runs; the baseline's "dev0" build and run last, so
  a hang there (a wall limit of 900 s plus a Watcher rerun) cannot starve the others. A stop file or a run
  directory over RUN_CAP_BYTES skips the step it is seen at and every later one; a budget that cannot hold a step
  skips that step.
- compare (needs NumPy): check each output against the codec's NumPy reference (reference/turboquant.py, its
  sha256 checked, imported from a copy of that one file) with the thresholds of the codec's own tests, each cited
  by file and line; report/compare.json and compare.txt.
- summarize (standard library): print the findings (the batch's stdout) and write report/summary.txt.

Every file under report/ is plain ASCII and holds no tt-metal or codec source text: counts, hashes, identifiers,
paths, verdicts, and simulator messages. Program streams, compiler stderr, and anything else that may quote source
go to local/ (pulled only into the untracked .rx/pulls/). Output dumps go to outputs/ (binary numbers; small).
Nothing here deletes a file it did not create, and tier_b.py's own writes stay inside the run directory it is given;
the lassi code that batch calls also makes, and removes, the --version check's directory under TMPDIR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CODEC_COMMIT = "81817eb09076d17b29491c6646b54c6927cac0bd"
NUM_VECTORS = 32
DIM = 128
# The one device-id literal each host program holds (MeshDevice::create_unit_mesh at the pin takes a physical chip
# id; tt_metal/distributed/mesh_device.cpp:466-483); ttsim presents a single chip, id 0 (bible, ttsim Facts).
DEVICE_ASIS = "create_unit_mesh(6)"
DEVICE_DEV0 = "create_unit_mesh(0)"
VARIANTS = ("asis", "dev0")
# The label every evidence file carries for the dev0 variant (summary.txt, compare.txt, compare.json).
DEV0_NOTE = f"dev0 = a changed copy: {DEVICE_ASIS} -> {DEVICE_DEV0} in the run directory only, not the codec's program"
ASIS_NOTE = f"asis = the codec's host program as written ({DEVICE_ASIS})"
# sha256 of reference/turboquant.py at CODEC_COMMIT (report/source.tsv of the dry run's real fetch); compare
# imports that file on the host, outside the sandbox, so it refuses any other bytes.
REFERENCE_SHA256 = "a4a79ca85104ecb26fc822d24cf66ed6d3f1ad9011bb62fe354c34839a3cf0b0"
# The run directory's start gate (messages call it a cap): a step starts only while the run directory's blocks total
# at most this (OQ-050's 1 GB). It bounds no step, so a step that starts under it can pass it.
RUN_CAP_BYTES = 1 << 30
# The batch's exit status when a stop file or the run directory cap skipped a step (run.sh exits with it).
HALTED_RC = 5
TOOLCHAIN = "ttmetal-host"
# A build's hard bound (the toolchain's timeout), and what the budget must still hold to start one: P4.10's
# builds of the Tier A host programs took well under a minute each (sizing only).
BUILD_TIMEOUT_S = 900.0
BUILD_NEED_S = 300.0
MEMORY_MB = 8192
CPUS = 16
PROBE_WALL_S = 60.0
# Wall limits (simulator wall time, sizing only): P4.9's six examples ran in at most 3.73 s each, and a busy
# core simulated about 430 kHz (results/p4-ttsim-runtime/runs.tsv); the baseline's stages do soft-float math on
# data-movement cores, its QJL stage on 32 cores, so it gets the most. An "asis" run fails at device open.
# A run starts only when the budget holds its wall limit and RUN_MARGIN_S; a hang's Watcher rerun (the ttsim
# executor's own, OQ-037) can take one wall limit more, bounded by the batch step's outer timeout.
ASIS_WALL_S = 120.0
RUN_MARGIN_S = 60.0
# A string literal in a host program that names a kernel source: a path with a directory named kernels.
KERNEL_LITERAL = re.compile(r'"((?:[A-Za-z0-9_.-]+/)*kernels/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.cpp)"')
# The private /dev of the sandbox (lassi.executors.sandbox, step 6): six bound nodes, pts, shm, and five links.
DEV_ALLOWED = frozenset({"null", "zero", "full", "random", "urandom", "tty", "pts", "shm", "fd", "stdin", "stdout",
                         "stderr", "ptmx"})
# The probe lists /dev and counts the entries of /sys/bus, /sys/class, and each /sys/devices entry but system (-1
# when a listing fails, which never counts as empty). /proc/bus/pci stays readable in the sandbox (a known limit,
# lassi/executors/sandbox.py, "/proc/bus/pci and the rest of /sys stay readable"): it lists PCI devices and opens
# none, so the probe records its presence and entry count without gating on it.
PROBE_SCRIPT = """#!/bin/sh
echo "dev: $(ls -A /dev | tr '\\n' ' ')"
for d in /sys/bus /sys/class /sys/devices/*; do
  case "$d" in /sys/devices/system) continue ;; esac
  [ -d "$d" ] || continue
  if list="$(ls -A "$d" 2>/dev/null)"; then
    if [ -z "$list" ]; then echo "sys: $d 0"; else echo "sys: $d $(printf '%s\\n' "$list" | wc -l)"; fi
  else
    echo "sys: $d -1"
  fi
done
if [ -e /proc/bus/pci ]; then
  echo "proc-bus-pci: present $(ls -A /proc/bus/pci 2>/dev/null | wc -l)"
else
  echo "proc-bus-pci: absent 0"
fi
echo "probe: done"
"""
WALL_LABEL = "simulator wall time, exploratory, sizing only, never performance"
# DataFormat values at the pin (tt_metal/api/tt-metalium/tt_backend_api_types.hpp:21-43) that ttsim issue #18's
# path needs: Float32 0, Int32 8, UInt32 24 (bible, ttsim Facts).
ISSUE18_FORMATS = frozenset({0, 8, 24})
INVALID_FORMAT = 255
LOCAL_CAP = 1 << 20
# Files of an alpha01 copy larger than this are recorded by size, not hashed (large dumps of big runs).
COPY_HASH_CAP = 64 << 20


@dataclass(frozen=True)
class Program:
    """One codec host program: its source, the headers it includes, arguments, input and output files."""

    name: str
    stage: str
    host: str
    headers: tuple[str, ...]
    args: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    wall_s: float


N_ARGS = ("--num-vectors", str(NUM_VECTORS))
PROGRAMS = (
    Program("baseline", "WHT, PolarQuant, QJL (baseline)", "host/turboquant_host.cpp",
            ("include/turboquant_layout.h",), (*N_ARGS, "--mode", "full"), (),
            ("dump_input.bin", "dump_rotated.bin", "dump_quant_indices.bin", "dump_quant_centroids.bin",
             "dump_qjl.bin", "dump_output.bin"), 900.0),
    Program("mt-rotation", "WHT (multi_tile, stage 0)", "multi_tile/host/turboquant_multi_tile.cpp",
            ("multi_tile/include/tile_layout.h",), N_ARGS, ("dump_input.bin",), ("dump_rotated_tile.bin",), 300.0),
    Program("mt-polarquant", "PolarQuant (multi_tile, stage 1)", "multi_tile/host/turboquant_multi_tile_pq.cpp",
            ("multi_tile/include/tile_layout.h",), N_ARGS, ("dump_rotated_tile.bin",), ("dump_centroids_tile.bin",),
            300.0),
    Program("mt-inverse-rotation", "inverse WHT (multi_tile, stage 2a)", "multi_tile/host/turboquant_multi_tile_ir.cpp",
            ("multi_tile/include/tile_layout.h",), (*N_ARGS, "--input", "dump_quant_centroids.bin"),
            ("dump_quant_centroids.bin",), ("dump_xhat_tile.bin",), 300.0),
    Program("mt-residual", "residual (multi_tile, stage 2b1)", "multi_tile/host/turboquant_multi_tile_resid.cpp",
            ("multi_tile/include/tile_layout.h",), N_ARGS, ("dump_input.bin", "dump_xhat_tile.bin"),
            ("dump_r_tile.bin",), 300.0),
    Program("mt-qjl-projection", "QJL projection (multi_tile, stage 2b2)",
            "multi_tile/host/turboquant_multi_tile_proj.cpp", ("multi_tile/include/tile_layout.h",), N_ARGS,
            ("dump_r_tile.bin",), ("dump_b_tile.bin", "dump_bits_tile.bin"), 300.0),
    Program("mt-rnorm", "r_norm (multi_tile, stage 2c)", "multi_tile/host/turboquant_multi_tile_rnorm.cpp",
            ("multi_tile/include/tile_layout.h",), N_ARGS, ("dump_r_tile.bin",), ("dump_rnorm_tile.bin",), 300.0),
)
# Codec files that can stand in for an input when no upstream run produced it: the committed dumps at CODEC_COMMIT
# (the codec's own silicon outputs at N = 32; their provenance is not recorded in the repository).
COMMITTED_INPUTS = ("dump_input.bin", "dump_quant_centroids.bin")
# Every file a program reads, in an order where each synthesized file comes after the files it is computed from.
INPUT_ORDER = ("dump_input.bin", "dump_quant_centroids.bin", "dump_rotated_tile.bin", "dump_xhat_tile.bin",
               "dump_r_tile.bin")


# ----------------------------------------------------------------------------------------------------------------
# Small shared helpers
# ----------------------------------------------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    """Return the hex sha256 of `data`."""
    return hashlib.sha256(data).hexdigest()


def write_ascii(path: Path, text: str) -> None:
    """Write `text` to `path` as ASCII, any other character as a backslash escape; parents are made."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("ascii", errors="backslashreplace"))


def write_json(path: Path, data: Any) -> None:
    """Write `data` as sorted, indented, ASCII JSON."""
    write_ascii(path, json.dumps(data, indent=1, sort_keys=True, ensure_ascii=True) + "\n")


def printable(text: str, limit: int = 200) -> str:
    """Return `text` on one line, printable ASCII only (others as '?'), cut to `limit` characters."""
    flat = " ".join(str(text).split())
    return "".join(c if c.isascii() and c.isprintable() else "?" for c in flat)[:limit]


def capped(data: bytes, cap: int = LOCAL_CAP) -> bytes:
    """Return `data`, or its first and last cap/2 bytes around a marker line when it is longer than `cap`."""
    if len(data) <= cap:
        return data
    half = cap // 2
    return data[:half] + f"\n[... {len(data) - 2 * half} bytes cut by the P5.3 batch ...]\n".encode() + data[-half:]


def read_text(path: Path) -> str:
    """Return a codec file's text, decoded as UTF-8 (the files hold UTF-8 comments); errors raise."""
    return path.read_bytes().decode("utf-8")


def tree_bytes(root: Path) -> int:
    """Return the bytes of disk blocks the files and directories under `root` take (symbolic links not followed)."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            try:
                info = os.lstat(os.path.join(dirpath, name))
            except OSError:
                continue
            blocks = getattr(info, "st_blocks", None)
            total += info.st_size if blocks is None else blocks * 512
    return total


def halted(reason: str) -> bool:
    """Return whether a skip reason ends the batch early (a stop file or the run directory cap), not the budget."""
    return reason.startswith(("stop file", "the run directory"))


# ----------------------------------------------------------------------------------------------------------------
# Codec layout: kernels named by a host program, the build file set, and bf16 helpers
# ----------------------------------------------------------------------------------------------------------------


def kernel_paths(host_text: str) -> list[str]:
    """Return the kernel sources a host program names, in first-seen order."""
    return list(dict.fromkeys(KERNEL_LITERAL.findall(host_text)))


def build_files(src: Path, program: Program, variant: str) -> tuple[dict[str, str], dict[str, Any]]:
    """Return (build files, facts) for `program` in `variant`: the host source and its headers at the build root,
    each kernel at the path the program names, with the headers beside it. ValueError when "dev0" cannot patch.

    A quoted include searches the including file's directory first, and the kernel JIT adds a kernel's own directory
    to its include path (tt_metal/impl/kernels/kernel.cpp:247-254 at the pin), so this layout finds every header.
    """
    host_text = read_text(src / program.host)
    facts: dict[str, Any] = {"host": program.host, "host_sha256": sha256_bytes(host_text.encode("utf-8"))}
    if variant == "dev0":
        count = host_text.count(DEVICE_ASIS)
        if count != 1:
            raise ValueError(f"{program.host} holds {count} copies of {DEVICE_ASIS}, not 1; dev0 is not built")
        host_text = host_text.replace(DEVICE_ASIS, DEVICE_DEV0)
        facts["patch"] = f"{DEVICE_ASIS} -> {DEVICE_DEV0} (one occurrence)"
    files = {Path(program.host).name: host_text}
    for header in program.headers:
        files[Path(header).name] = read_text(src / header)
    kernels = kernel_paths(host_text)
    for kernel in kernels:
        files[kernel] = read_text(src / kernel)
        for header in sorted((src / kernel).parent.glob("*.h")):
            files[header.relative_to(src).as_posix()] = read_text(header)
    facts.update(kernels=kernels, built_sha256=sha256_bytes(host_text.encode("utf-8")),
                 names_fp32_dest_acc_en="fp32_dest_acc_en" in host_text,
                 names_unpack_to_dest="unpack_to_dest" in host_text,
                 data_formats=sorted(set(re.findall(r"DataFormat::(\w+)", host_text))))
    return files, facts


def bf16_values(data: bytes) -> list[float]:
    """Return the float values of little-endian bf16 elements."""
    count = len(data) // 2
    halves = struct.unpack(f"<{count}H", data[: 2 * count])
    return list(struct.unpack(f"<{count}f", struct.pack(f"<{count}I", *(h << 16 for h in halves))))


def bf16_bytes(values: list[float]) -> bytes:
    """Return `values` as bf16 by truncation (the codec hosts' float_to_bf16, host/turboquant_host.cpp:43-46)."""
    words = struct.unpack(f"<{len(values)}I", struct.pack(f"<{len(values)}f", *values))
    return struct.pack(f"<{len(values)}H", *(w >> 16 for w in words))


def header_numbers(text: str, name: str) -> list[float]:
    """Return the numbers of the C array `name` in codebooks.h; ValueError when it is absent."""
    match = re.search(rf"{re.escape(name)}\[[^\]]*\]\s*=\s*\{{(.*?)\}};", text, re.S)
    if match is None:
        raise ValueError(f"codebooks.h has no array {name}")
    return [float(tok.rstrip("f")) for tok in re.findall(r"-?\d+(?:\.\d+)?f?", match.group(1))]


def wht_rows(rows: list[list[float]]) -> list[list[float]]:
    """Return the normalized Walsh-Hadamard transform of each row (reference/turboquant.py wht, in Python)."""
    out = []
    for row in rows:
        buf = list(row)
        h = 1
        while h < len(buf):
            for i in range(0, len(buf), 2 * h):
                for j in range(i, i + h):
                    buf[j], buf[j + h] = buf[j] + buf[j + h], buf[j] - buf[j + h]
            h *= 2
        scale = 1.0 / math.sqrt(len(buf))
        out.append([v * scale for v in buf])
    return out


def synthesize(name: str, src: Path, have: dict[str, bytes]) -> bytes | None:
    """Return a stand-in for input `name` computed from inputs in `have`, or None when it cannot be made.

    Used only when no upstream run produced the file and the codec commits none: the forward or inverse rotation
    with the kernels' sign table, or the residual, in double precision, then bf16 by truncation. Labeled
    "synthesized" in every record.
    """
    sign = header_numbers(read_text(src / "include" / "codebooks.h"), "SIGN_DIAGONAL_d128")
    rows = lambda data: [bf16_values(data)[r * DIM:(r + 1) * DIM] for r in range(NUM_VECTORS)]  # noqa: E731
    if name == "dump_rotated_tile.bin" and "dump_input.bin" in have:
        z = wht_rows([[s * v for s, v in zip(sign, row, strict=True)] for row in rows(have["dump_input.bin"])])
        return bf16_bytes([v for row in z for v in row])
    if name == "dump_xhat_tile.bin" and "dump_quant_centroids.bin" in have:
        w = wht_rows(rows(have["dump_quant_centroids.bin"]))
        return bf16_bytes([s * v for row in w for s, v in zip(sign, row, strict=True)])
    if name == "dump_r_tile.bin" and {"dump_input.bin", "dump_xhat_tile.bin"} <= set(have):
        x, xh = bf16_values(have["dump_input.bin"]), bf16_values(have["dump_xhat_tile.bin"])
        return bf16_bytes([a - b for a, b in zip(x, xh, strict=True)])
    return None


# ----------------------------------------------------------------------------------------------------------------
# prepare
# ----------------------------------------------------------------------------------------------------------------


def git(repo: Path, *args: str) -> str:
    """Return a read-only git command's stdout in `repo`; CalledProcessError on failure.

    No optional locks, no hooks, no fsmonitor program, and no global or system configuration file.
    """
    command = ["git", "--no-optional-locks", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false",
               "-C", str(repo), *args]
    env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=120, env=env).stdout


def source_manifest(src: Path) -> list[tuple[str, int, str, str]]:
    """Return (path, bytes, sha256, git blob) for every file tracked at HEAD of the checkout `src`."""
    rows = []
    for line in git(src, "ls-tree", "-r", "-l", "HEAD").splitlines():
        meta, path = line.split("\t", 1)
        _mode, kind, blob, _size = meta.split()
        if kind != "blob":
            continue
        data = (src / path).read_bytes()
        rows.append((path, len(data), sha256_bytes(data), blob))
    return rows


def license_facts(src: Path, rows: list[tuple[str, int, str, str]]) -> list[str]:
    """Return report/license.txt's lines: license files in the tree, and whether the README names one."""
    names = [p for p, *_ in rows if re.match(r"(?i)^(licen[cs]e|copying|notice)", Path(p).name)]
    readme = (src / "README.md").read_text(encoding="utf-8", errors="replace") if (src / "README.md").exists() else ""
    named = sorted(set(re.findall(r"\b(LICEN[CS]E(?:\.\w+)?)\b", readme)))
    return [f"license files in the tree at {CODEC_COMMIT}: {', '.join(names) or 'none'}",
            f"license file names the README mentions: {', '.join(named) or 'none'}",
            f"README names a license file the tree lacks: {bool(set(named) - {Path(p).name for p in names})}"]


def program_rows(src: Path) -> list[str]:
    """Return report/programs.tsv's lines: program, stage, host source, its sha256, kernels it names."""
    out = ["program\tstage\thost\thost_sha256\tkernels"]
    for program in PROGRAMS:
        text = read_text(src / program.host)
        out.append("\t".join([program.name, program.stage, program.host, sha256_bytes(text.encode()),
                              " ".join(kernel_paths(text))]))
    return out


def compare_copy(copy: Path, src: Path, rows: list[tuple[str, int, str, str]]) -> dict[str, Any]:
    """Compare an alpha01 copy of the codec with the pinned checkout, read only: git state and sha256 per file.

    Never reads a remote URL (plans/LESSONS.md). Walks until it has 5000 files, leaving out .git and build
    directories; a file over COPY_HASH_CAP bytes is recorded by size.
    """
    state: dict[str, Any] = {"copy": str(copy)}
    for key, args in (("head", ("rev-parse", "HEAD")), ("branch", ("rev-parse", "--abbrev-ref", "HEAD")),
                      ("status", ("status", "--porcelain=v1")),
                      ("has_pinned_commit", ("cat-file", "-t", CODEC_COMMIT))):
        try:
            state[key] = git(copy, *args).strip()
        except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired) as error:
            state[key] = f"unavailable ({type(error).__name__})"
    pinned = {path: digest for path, _size, digest, _blob in rows}
    found: dict[str, str] = {}
    state["walk_capped"] = False
    for dirpath, dirnames, filenames in os.walk(copy):
        dirnames[:] = [d for d in dirnames if d != ".git" and not d.startswith("build")]
        for name in filenames:
            path = Path(dirpath, name)
            if not path.is_file() or path.is_symlink():
                continue
            size = path.stat().st_size
            found[path.relative_to(copy).as_posix()] = (
                sha256_bytes(path.read_bytes()) if size <= COPY_HASH_CAP else f"not hashed: {size} bytes")
            if len(found) >= 5000:
                break
        if len(found) >= 5000:
            state["walk_capped"] = True
            break
    state["same"] = sorted(p for p in pinned if found.get(p) == pinned[p])
    state["different"] = sorted(p for p in pinned if p in found and found[p] != pinned[p])
    state["only_on_github"] = sorted(p for p in pinned if p not in found)
    state["only_in_copy"] = sorted(p for p in found if p not in pinned)
    state["status"] = state["status"].splitlines() if isinstance(state["status"], str) else state["status"]
    return state


def cmd_prepare(args: argparse.Namespace) -> int:
    """Check the checkout and write the source manifest, license facts, and program table."""
    src, report = Path(args.src), Path(args.report)
    head = git(src, "rev-parse", "HEAD").strip()
    if head != CODEC_COMMIT:
        print(f"p53: the checkout is at {head}, not {CODEC_COMMIT}; refusing")
        return 2
    rows = source_manifest(src)
    write_ascii(report / "source.tsv", "path\tbytes\tsha256\tgit_blob\n"
                + "".join(f"{p}\t{n}\t{d}\t{b}\n" for p, n, d, b in rows))
    write_ascii(report / "license.txt", "\n".join(license_facts(src, rows)) + "\n")
    write_ascii(report / "programs.tsv", "\n".join(program_rows(src)) + "\n")
    print(f"p53: codec at {head}: {len(rows)} tracked files, {sum(n for _, n, _, _ in rows)} bytes")
    print(f"p53: {license_facts(src, rows)[0]}")
    return 0


def cmd_copy(args: argparse.Namespace) -> int:
    """Compare an alpha01 copy with the pinned checkout (run.sh's own step); write report/alpha01-copy.json."""
    src, report = Path(args.src), Path(args.report)
    if git(src, "rev-parse", "HEAD").strip() != CODEC_COMMIT:
        print(f"p53: the checkout is not at {CODEC_COMMIT}; the copy comparison refuses")
        return 2
    state = compare_copy(Path(args.copy), src, source_manifest(src))
    write_json(report / "alpha01-copy.json", state)
    print(f"p53: alpha01 copy: head {printable(state['head'], 60)}, {len(state['same'])} same, "
          f"{len(state['different'])} different, {len(state['only_on_github'])} only on GitHub, "
          f"{len(state['only_in_copy'])} only in the copy, {len(state['status'])} status lines, "
          f"walk capped {state['walk_capped']}")
    return 0


# ----------------------------------------------------------------------------------------------------------------
# batch: probe, builds, runs
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class Batch:
    """The batch's paths, limits, and the lassi objects every step uses."""

    run: Path
    src: Path
    report: Path
    local: Path
    raw: Path
    outputs: Path
    deadline: float
    stop_files: list[Path]
    wall_scale: float
    built: Any = None
    executor: Any = None
    limits_type: Any = None


def stop_reason(batch: Batch, need_s: float) -> str:
    """Return why a step must be skipped ('' when it may run): a stop file, a run directory over RUN_CAP_BYTES, or
    a budget that cannot hold the step's `need_s` seconds."""
    for path in batch.stop_files:
        if path.exists():
            return f"stop file {path}"
    used = tree_bytes(batch.run)
    if used > RUN_CAP_BYTES:
        return f"the run directory holds {used >> 20} MiB, over its {RUN_CAP_BYTES >> 20} MiB cap"
    if time.monotonic() + need_s > batch.deadline:
        return f"the batch budget cannot hold a {need_s:.0f} s step"
    return ""


def device_probe(batch: Batch) -> dict[str, Any]:
    """Run PROBE_SCRIPT through the ttsim executor and check that the sandbox shows no device: its /dev holds only
    the sandbox's own entries and /sys/bus, /sys/class, and /sys/devices/* (system excepted) are empty. Records
    /proc/bus/pci's presence without gating on it (PROBE_SCRIPT)."""
    workdir = batch.raw / "probe"
    workdir.mkdir(parents=True)
    script = workdir / "probe.sh"
    script.write_text(PROBE_SCRIPT)
    script.chmod(0o755)
    result = batch.executor.run(script.resolve(), [], batch.limits_type(wall_s=PROBE_WALL_S, memory_mb=1024, cpus=1))
    dev: set[str] = set()
    sys_counts: dict[str, int] = {}
    pci = "not reported"
    for line in result.stdout.splitlines():
        if line.startswith("dev: "):
            dev = set(line[5:].split())
        elif line.startswith("sys: "):
            path, _, count = line[5:].rpartition(" ")
            sys_counts[path] = int(count) if count.isdigit() else -1
        elif line.startswith("proc-bus-pci: "):
            word, _, count = line[14:].partition(" ")
            pci = f"{word}, {count if count.isdigit() else '?'} entries" if word == "present" else word
    done = "probe: done" in result.stdout
    ok = done and result.exit_code == 0 and bool(dev) and dev <= DEV_ALLOWED and all(
        c == 0 for c in sys_counts.values()) and bool(sys_counts)
    return {"ok": ok, "exit_code": result.exit_code, "dev_entries": sorted(dev),
            "unexpected_dev": sorted(dev - DEV_ALLOWED),
            "sys_nonempty": sorted(p for p, c in sys_counts.items() if c != 0), "sys_checked": len(sys_counts),
            "proc_bus_pci": pci,
            "proc_bus_pci_note": "readable in the sandbox, a known limit (lassi/executors/sandbox.py); it lists "
                                 "PCI devices and opens none; recorded, not gated"}


def diagnostics_summary(diagnostics: list[Any]) -> dict[str, Any]:
    """Return counts by severity, warning codes, and the first errors (file:line and message) of a build."""
    severities: dict[str, int] = {}
    codes: dict[str, int] = {}
    errors = []
    for item in diagnostics:
        severities[item.severity] = severities.get(item.severity, 0) + 1
        if item.severity == "warning" and item.code:
            codes[item.code] = codes.get(item.code, 0) + 1
        if item.severity == "error" and len(errors) < 5:
            errors.append(f"{item.file or '-'}:{item.line or '-'}: {printable(item.message, 160)}")
    return {"by_severity": severities, "warning_codes": codes, "first_errors": errors}


def build_one(batch: Batch, program: Program, variant: str) -> dict[str, Any]:
    """Build `program` in `variant` in raw/<variant>-<name>/ with ttmetal-host; return the build record."""
    record: dict[str, Any] = {"program": program.name, "variant": variant, "step": "build"}
    reason = stop_reason(batch, BUILD_NEED_S)
    if reason:
        return {**record, "skipped": reason}
    workdir = batch.raw / f"{variant}-{program.name}"
    try:
        files, facts = build_files(batch.src, program, variant)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        return {**record, "skipped": printable(f"not assembled: {error}", 300)}
    workdir.mkdir(parents=True)
    started = time.monotonic()
    result = batch.built.toolchain.build(files, workdir)
    record.update(facts, build_s=round(time.monotonic() - started, 1), artifact=bool(result.artifact),
                  diagnostics=diagnostics_summary(result.diagnostics))
    stderr = workdir / (result.stderr_ref or "missing")
    if stderr.is_file():
        (batch.local / f"build-{variant}-{program.name}.stderr.txt").write_bytes(capped(stderr.read_bytes()))
    record["artifact_path"] = str(result.artifact) if result.artifact else None
    return record


def unpack_facts(cache: Path) -> dict[str, Any]:
    """Return the unpack_to_dest facts of the JIT's generated descriptors under `cache` (values only).

    Reads every header's `DST_ACCUM_MODE = <bool>` (tt_metal/jit_build/genfiles.cpp:487-490 at the pin) and the
    integer arrays unpack_src_format and unpack_dst_format (:291-294); 255 is DataFormat::Invalid.
    """
    accum: set[str] = set()
    formats: dict[str, set[int]] = {"unpack_src_format": set(), "unpack_dst_format": set()}
    headers = 0
    for path in sorted(cache.rglob("*.h")) if cache.is_dir() else []:
        text = path.read_text(encoding="utf-8", errors="replace")
        found = re.findall(r"\bDST_ACCUM_MODE\s*=\s*(true|false)\b", text)
        arrays = re.findall(r"\b(unpack_src_format|unpack_dst_format)\[\d+\]\s*=\s*\{([^}]*)\}", text)
        headers += bool(found or arrays)
        accum.update(found)
        for name, body in arrays:
            formats[name].update(int(v) for v in re.findall(r"-?\d+", body))
    used = {k: sorted(v - {INVALID_FORMAT}) for k, v in formats.items()}
    seen = set(used["unpack_src_format"]) | set(used["unpack_dst_format"])
    issue18 = "true" in accum or bool(ISSUE18_FORMATS & seen)
    return {"descriptor_headers": headers, "dst_accum_mode": sorted(accum), **used,
            "issue18_path_possible": issue18}


def resolve_inputs(batch: Batch, program: Program, variant: str, produced: dict[str, Path]) -> dict[str, Any]:
    """Return {name: (bytes, source)} for the program's inputs: an upstream dev0 output, the codec's committed
    file, or a synthesized stand-in, in that order ("asis" runs take no upstream output). Every pipeline file is
    resolved in INPUT_ORDER, so a synthesized file finds the files it is computed from."""
    have: dict[str, bytes] = {}
    sources: dict[str, str] = {}
    for name in INPUT_ORDER:
        if variant == "dev0" and name in produced:
            have[name] = produced[name].read_bytes()
            sources[name] = f"output of the dev0 run ({produced[name].parent.name})"
        elif name in COMMITTED_INPUTS:
            have[name], sources[name] = (batch.src / name).read_bytes(), f"the codec's committed {name}"
        else:
            made = synthesize(name, batch.src, have)
            if made is not None:
                have[name], sources[name] = made, "synthesized by tier_b.py (reference math, bf16 by truncation)"
    return {name: (have[name], sources[name]) for name in program.inputs if name in have}


def run_one(batch: Batch, program: Program, variant: str, build: dict[str, Any],
            produced: dict[str, Path]) -> dict[str, Any]:
    """Run a built program on ttsim through the executor; return the run record and keep its outputs."""
    record: dict[str, Any] = {"program": program.name, "variant": variant, "step": "run"}
    if "patch" in build:
        record["patch"] = build["patch"]
    wall = (ASIS_WALL_S if variant == "asis" else program.wall_s) * batch.wall_scale
    reason = stop_reason(batch, wall + RUN_MARGIN_S)
    if reason or not build.get("artifact"):
        return {**record, "skipped": reason or "no artifact"}
    artifact = Path(build["artifact_path"])
    inputs = resolve_inputs(batch, program, variant, produced)
    keep = batch.outputs / f"{variant}-{program.name}"
    for name, (data, _source) in inputs.items():
        (artifact.parent / name).write_bytes(data)
        write_inputs = keep / "inputs" / name
        write_inputs.parent.mkdir(parents=True, exist_ok=True)
        write_inputs.write_bytes(data)
    record["inputs"] = {n: {"source": s, "sha256": sha256_bytes(d), "bytes": len(d)} for n, (d, s) in inputs.items()}
    missing = [n for n in program.inputs if n not in inputs]
    if missing:
        return {**record, "skipped": f"inputs unavailable: {' '.join(missing)}"}
    limits = batch.limits_type(wall_s=wall, memory_mb=MEMORY_MB, cpus=CPUS)
    try:
        result = batch.executor.run(artifact, list(program.args), limits)
    except Exception as error:  # noqa: BLE001 - the executor's refusals (SandboxUnavailableError, ValueError)
        return {**record, "refused": printable(f"{type(error).__name__}: {error}", 300)}
    record.update(run_fields(result, wall))
    record["outputs"] = keep_outputs(result, program, keep, produced if variant == "dev0" else None)
    record["unpack_to_dest"] = unpack_facts(artifact.parent / "@ttsim" / "cache")
    for stream in ("stdout", "stderr"):
        (batch.local / f"run-{variant}-{program.name}.{stream}.txt").write_bytes(
            capped(getattr(result, stream).encode("utf-8", errors="replace")))
    return record


def run_fields(result: Any, wall: float) -> dict[str, Any]:
    """Return the record fields of a RunResult: status, flags, findings, JIT errors, notes, and wall time."""
    findings = [d for d in result.diagnostics if d.stage == "run" and d.severity == "error"]
    jit = [d for d in result.diagnostics if d.stage == "jit" and d.severity == "error"]
    notes = [d for d in result.diagnostics if d.severity == "note"]
    stderr = result.stderr
    return {"exit_code": result.exit_code, "hang": result.hang, "sim_ub": result.sim_ub, "sim_gap": result.sim_gap,
            "findings": [printable(d.message, 240) for d in findings[:8]],
            "finding_count": len(findings), "tensix_unpacr": any("tensix_unpacr" in d.message for d in findings),
            "jit_errors": [f"{d.file or '-'}:{d.line or '-'}: {printable(d.message, 160)}" for d in jit[:5]],
            "jit_error_count": len(jit), "watcher_note": bool(notes),
            "stderr_kind": [k for k, w in (("tt-fatal", "TT_FATAL"), ("tt-throw", "TT_THROW"),
                                           ("out-of-range", "out_of_range"), ("terminate", "terminate called"),
                                           ("cannot-open", "cannot open")) if w in stderr],
            "workdir_incomplete": result.workdir_incomplete, "stdout_truncated": result.stdout_truncated,
            "wall_s": {"value": round(result.wall_s, 2), "limit": wall, "label": WALL_LABEL}}


def keep_outputs(result: Any, program: Program, keep: Path, produced: dict[str, Path] | None) -> dict[str, Any]:
    """Copy the program's expected output files into `keep`; return name -> sha256 and bytes (or absent)."""
    out: dict[str, Any] = {}
    files = {Path(k).name: Path(v) for k, v in result.output_files.items()}
    for name in program.outputs:
        if name not in files or not files[name].is_file():
            out[name] = "absent"
            continue
        data = files[name].read_bytes()
        target = keep / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        out[name] = {"sha256": sha256_bytes(data), "bytes": len(data)}
        if produced is not None:
            produced[name] = target
    out["_other_new_files"] = sorted(n for n in files if n not in program.outputs)[:20]
    return out


def load_lassi(batch: Batch, toolchains: Path, runs_root: Path) -> None:
    """Import lassi and set the toolchain (build_toolchain, as tools/ttsim_smoke.py does) and the executor."""
    from lassi.core.interfaces import Limits  # noqa: PLC0415 - lassi is needed by batch only
    from lassi.core.runner import build_toolchain  # noqa: PLC0415
    from lassi.executors.ttsim import TtsimExecutor  # noqa: PLC0415

    batch.limits_type = Limits
    batch.executor = TtsimExecutor()
    batch.executor.install()
    batch.built = build_toolchain(TOOLCHAIN, toolchains, build_root=runs_root)
    batch.built.toolchain.timeout_s = BUILD_TIMEOUT_S


def step_line(record: dict[str, Any]) -> str:
    """Return one stdout line for a finished step."""
    head = f"{record['step']:<5} {record['variant']:<4} {record['program']:<20}"
    if "skipped" in record or "refused" in record:
        word = "skipped" if "skipped" in record else "refused"
        return f"{head} {word}: {record.get('skipped') or record['refused']}"[:200]
    if record["step"] == "build":
        sev = record["diagnostics"]["by_severity"]
        return f"{head} artifact={record['artifact']} errors={sev.get('error', 0)} warnings={sev.get('warning', 0)}"
    outs = sum(1 for v in record["outputs"].values() if isinstance(v, dict))
    return (f"{head} exit={record['exit_code']} hang={record['hang']} ub={record['sim_ub']} gap={record['sim_gap']} "
            f"jit={record['jit_error_count']} outputs={outs}/{len(record['outputs']) - 1}")


def save(batch: Batch, record: dict[str, Any]) -> bool:
    """Write the step's record to report/steps/ and print its line; return whether a halt skipped it."""
    write_json(batch.report / "steps" / f"{record['step']}-{record['variant']}-{record['program']}.json", record)
    print(step_line(record), flush=True)
    return halted(record.get("skipped", ""))


def cmd_batch(args: argparse.Namespace) -> int:
    """Probe the sandbox, then build and run every program (see the module docstring for the order).

    Returns 0, 2 (lassi's install checks refused), 3 (the probe failed), 4 (the budget could not hold the probe),
    or HALTED_RC (a stop file or the run directory's start gate skipped a step, the probe included).
    """
    run = Path(args.run)
    batch = Batch(run=run, src=run / "src", report=run / "report", local=run / "local", raw=run / "raw",
                  outputs=run / "outputs", deadline=time.monotonic() + args.budget_s,
                  stop_files=[Path(p) for p in args.stop_file], wall_scale=args.wall_scale)
    for directory in (batch.local, batch.raw, batch.outputs):
        directory.mkdir(parents=True, exist_ok=True)
    try:
        load_lassi(batch, Path(args.toolchains), Path(args.runs_root))
    except Exception as error:  # noqa: BLE001 - the install checks' refusals (RunError, SandboxUnavailableError)
        print(printable(f"p53: refused before any build: {type(error).__name__}: {error}", 400))
        return 2
    write_ascii(batch.report / "device.txt", batch.executor.device() + "\n")
    reason = stop_reason(batch, PROBE_WALL_S)
    if reason:
        write_json(batch.report / "device-probe.json", {"ok": False, "skipped": reason})
        print(printable(f"p53: {reason}; the probe and every build and run are skipped", 300))
        return HALTED_RC if halted(reason) else 4
    probe = device_probe(batch)
    write_json(batch.report / "device-probe.json", probe)
    print(f"probe sandbox dev={len(probe['dev_entries'])} unexpected={probe['unexpected_dev']} "
          f"sys_checked={probe['sys_checked']} sys_nonempty={probe['sys_nonempty']} "
          f"proc_bus_pci={probe['proc_bus_pci'].replace(' ', '')} ok={probe['ok']}", flush=True)
    if not probe["ok"]:
        print("p53: the sandbox probe did not confirm a device-free sandbox; no program runs")
        return 3
    return HALTED_RC if any(build_and_run(batch)) else 0


def build_and_run(batch: Batch) -> list[bool]:
    """Build and run every program in the module docstring's order; return save()'s halt flag of each step."""
    baseline, stages = PROGRAMS[0], PROGRAMS[1:]
    flags = []
    builds = {}
    for program in (*stages, baseline):
        builds["asis", program.name] = build_one(batch, program, "asis")
        flags.append(save(batch, builds["asis", program.name]))
    produced: dict[str, Path] = {}
    for program in stages:
        builds["dev0", program.name] = build_one(batch, program, "dev0")
        flags.append(save(batch, builds["dev0", program.name]))
        flags.append(save(batch, run_one(batch, program, "dev0", builds["dev0", program.name], produced)))
    for program in (*stages, baseline):
        flags.append(save(batch, run_one(batch, program, "asis", builds["asis", program.name], produced)))
    builds["dev0", baseline.name] = build_one(batch, baseline, "dev0")
    flags.append(save(batch, builds["dev0", baseline.name]))
    flags.append(save(batch, run_one(batch, baseline, "dev0", builds["dev0", baseline.name], produced)))
    return flags


# ----------------------------------------------------------------------------------------------------------------
# compare (NumPy)
# ----------------------------------------------------------------------------------------------------------------

# Each check: (program, check, codec file:line of the threshold, what the codec's own test compares).
CHECK_SOURCES = {
    "rotation": "tests/test_rotation.py:184,188 (max_abs < 0.005 and mean_abs < 0.001)",
    "indices": "tests/test_polarquant.py:229 (match >= 0.99; the test builds its codebook at :217, this check "
               "uses the kernels' own include/codebooks.h b4 d128 table)",
    "centroids": "tests/test_polarquant.py:277 (max_abs < 0.005)",
    "qjl_bits": "tests/test_qjl.py:251 (match >= 0.98; LFSR sign rows as tests/test_qjl.py:178-196)",
    "qjl_rnorm": "tests/test_qjl.py:289 (max relative error < 0.01)",
    "mt_rotation": "multi_tile/tests/test_rotation_tile.py:47 (max_abs < 0.01; the test compares with the "
                   "baseline's dump_rotated.bin, this check with the reference)",
    "mt_polarquant": "multi_tile/tests/test_polarquant_tile.py:45,89-90 (max_abs < 0.01 against searchsorted "
                     "side='right')",
    "mt_inverse_rotation": "multi_tile/tests/test_inverse_rotation_tile.py:84 (max_abs < 0.01)",
    "mt_residual": "multi_tile/tests/test_residual_tile.py:27 (max_abs < 0.001)",
    "mt_qjl_projection": "multi_tile/tests/test_qjl_project_tile.py:65,71 (strict match >= 0.998 where "
                         "|B| >= 0.01, overall >= 0.95; the test compares with the baseline's bits and masks by "
                         "|B_dev|, this check compares with the reference and masks by |B_ref|)",
    "mt_rnorm": "multi_tile/tests/test_rnorm_tile.py:48 (max relative error < 0.02)",
}


class Oracle:
    """The reference pieces every check uses: the codec's turboquant module, its sign diagonal, codebook, and S."""

    def __init__(self, np: Any, src: Path, scratch: Path) -> None:
        """Import reference/turboquant.py from a copy of that file alone (never the committed __pycache__), after
        checking its bytes against REFERENCE_SHA256; ValueError when they differ."""
        data = (src / "reference" / "turboquant.py").read_bytes()
        if sha256_bytes(data) != REFERENCE_SHA256:
            raise ValueError(f"reference/turboquant.py has sha256 {sha256_bytes(data)}, not {REFERENCE_SHA256}")
        scratch.mkdir(parents=True, exist_ok=True)
        (scratch / "turboquant.py").write_bytes(data)
        sys.path.insert(0, str(scratch))
        import turboquant  # noqa: PLC0415 - the codec's reference, copied above

        self.np, self.tq = np, turboquant
        header = read_text(src / "include" / "codebooks.h")
        self.sign_table = np.array(header_numbers(header, "SIGN_DIAGONAL_d128"))
        self.D = turboquant.make_sign_diagonal(DIM, seed=0)
        self.codebook = turboquant.Codebook(
            centroids=np.array(header_numbers(header, "CODEBOOK_CENTROIDS_b4_d128")),
            boundaries=np.array(header_numbers(header, "CODEBOOK_BOUNDARIES_b4_d128")), bits=4, dim=DIM)
        self.S = np.array(lfsr_rows(DIM, DIM, 0xCAFEBABE), dtype=np.float64)

    def bf16(self, path: Path, shape: tuple[int, ...]) -> Any:
        """Return a bf16 file as a float64 array of `shape` (the first prod(shape) elements)."""
        np = self.np
        raw = np.frombuffer(path.read_bytes(), dtype=np.uint16)[: int(np.prod(shape))]
        return (raw.astype(np.uint32) << 16).view(np.float32).astype(np.float64).reshape(shape)


def lfsr_rows(k: int, d: int, seed: int) -> list[list[float]]:
    """Return the kernels' QJL sign matrix S (k x d): Galois LFSR, taps 0x80200003, row seed seed ^ (j * 2654435761),
    a sign per step from the state's low bit (kernels/qjl_kernel.cpp, tests/test_qjl.py:178-196)."""
    rows = []
    for j in range(k):
        state = (seed ^ (j * 2654435761)) & 0xFFFFFFFF or 1
        row = []
        for _ in range(d):
            lsb = state & 1
            state >>= 1
            if lsb:
                state ^= 0x80200003
            row.append(1.0 if state & 1 else -1.0)
        rows.append(row)
    return rows


def check(name: str, passed: bool, **metrics: Any) -> dict[str, Any]:
    """Return one check's record, with its threshold source."""
    clean = {k: (round(float(v), 8) if isinstance(v, float) or hasattr(v, "dtype") else v) for k, v in metrics.items()}
    return {"check": name, "pass": bool(passed), "threshold_source": CHECK_SOURCES[name], **clean}


def guarded(name: str, compute: Any) -> dict[str, Any]:
    """Return compute()'s check, or a failed check naming the absent file when one of its files is missing."""
    try:
        return compute()
    except FileNotFoundError as error:
        return {"check": name, "pass": False, "absent": Path(error.filename or "?").name,
                "threshold_source": CHECK_SOURCES[name]}


def baseline_checks(o: Oracle, out: Path) -> list[dict[str, Any]]:
    """Return the baseline program's checks against the reference (see CHECK_SOURCES)."""
    np, tq, shape = o.np, o.tq, (NUM_VECTORS, DIM)

    def indices() -> Any:
        packed = np.frombuffer((out / "dump_quant_indices.bin").read_bytes(), dtype=np.uint8)[: NUM_VECTORS * DIM // 2]
        return np.stack([packed & 0xF, packed >> 4], axis=1).reshape(shape)

    def rotation() -> dict[str, Any]:
        ref = tq.apply_rotation(o.bf16(out / "dump_input.bin", shape), o.D)
        err = np.abs(o.bf16(out / "dump_rotated.bin", shape) - ref)
        return check("rotation", err.max() < 0.005 and err.mean() < 0.001, max_abs=err.max(), mean_abs=err.mean())

    def index_match() -> dict[str, Any]:
        rate = float(np.mean(indices() == tq.polarquant_encode(o.bf16(out / "dump_rotated.bin", shape), o.codebook)))
        return check("indices", rate >= 0.99, match=rate)

    def centroids() -> dict[str, Any]:
        err = np.abs(o.bf16(out / "dump_quant_centroids.bin", shape) - o.codebook.centroids[indices()]).max()
        return check("centroids", err < 0.005, max_abs=err)

    def qjl() -> tuple[Any, Any, Any]:
        raw = np.frombuffer((out / "dump_qjl.bin").read_bytes(), dtype=np.uint8)[: NUM_VECTORS * 18].reshape(-1, 18)
        bits = np.unpackbits(raw[:, :16], axis=1, bitorder="little").astype(bool)
        rn_dev = (raw[:, 16:18].copy().view(np.uint16).astype(np.uint32) << 16).view(np.float32)[:, 0]
        r = o.bf16(out / "dump_input.bin", shape) - tq.apply_inverse_rotation(
            o.bf16(out / "dump_quant_centroids.bin", shape), o.D)
        return bits, rn_dev.astype(np.float64), r

    def qjl_bits() -> dict[str, Any]:
        bits, _, r = qjl()
        rn = np.linalg.norm(r, axis=1)
        rate = float(np.mean(bits == (((r / (rn[:, None] + 1e-12)) @ o.S.T) > 0)))
        return check("qjl_bits", rate >= 0.98, match=rate)

    def qjl_rnorm() -> dict[str, Any]:
        _, rn_dev, r = qjl()
        rn = np.linalg.norm(r, axis=1)
        rel = float(np.max(np.abs(rn - rn_dev) / (rn + 1e-12)))
        return check("qjl_rnorm", rel < 0.01, max_rel=rel)

    return [guarded(name, fn) for name, fn in (("rotation", rotation), ("indices", index_match),
                                               ("centroids", centroids), ("qjl_bits", qjl_bits),
                                               ("qjl_rnorm", qjl_rnorm))]


def stage_checks(o: Oracle, name: str, out: Path) -> list[dict[str, Any]]:
    """Return a multi_tile program's check: its output against the reference applied to the inputs it read."""
    np, tq, shape, ins = o.np, o.tq, (NUM_VECTORS, DIM), out / "inputs"

    def rotation() -> dict[str, Any]:
        ref = tq.apply_rotation(o.bf16(ins / "dump_input.bin", shape), o.D)
        err = np.abs(o.bf16(out / "dump_rotated_tile.bin", shape) - ref)
        return check("mt_rotation", err.max() < 0.01, max_abs=err.max(), mean_abs=err.mean())

    def polarquant() -> dict[str, Any]:
        z, dev = o.bf16(ins / "dump_rotated_tile.bin", shape), o.bf16(out / "dump_centroids_tile.bin", shape)
        right = o.codebook.centroids[np.searchsorted(o.codebook.boundaries, z, side="right")]
        left = tq.polarquant_decode(tq.polarquant_encode(z, o.codebook), o.codebook)
        err = np.abs(dev - right).max()
        return check("mt_polarquant", err < 0.01, max_abs=err, max_abs_vs_reference_encode=np.abs(dev - left).max())

    def inverse_rotation() -> dict[str, Any]:
        ref = tq.apply_inverse_rotation(o.bf16(ins / "dump_quant_centroids.bin", shape), o.D)
        err = np.abs(o.bf16(out / "dump_xhat_tile.bin", shape) - ref)
        return check("mt_inverse_rotation", err.max() < 0.01, max_abs=err.max(), mean_abs=err.mean())

    def residual() -> dict[str, Any]:
        ref = o.bf16(ins / "dump_input.bin", shape) - o.bf16(ins / "dump_xhat_tile.bin", shape)
        err = np.abs(o.bf16(out / "dump_r_tile.bin", shape) - ref)
        return check("mt_residual", err.max() < 0.001, max_abs=err.max(), mean_abs=err.mean())

    def projection() -> dict[str, Any]:
        b_ref = o.bf16(ins / "dump_r_tile.bin", shape) @ o.S.T
        packed = np.frombuffer((out / "dump_bits_tile.bin").read_bytes(), dtype=np.uint8)[: NUM_VECTORS * 16]
        same = np.unpackbits(packed.reshape(-1, 16), axis=1, bitorder="little").astype(bool) == (b_ref > 0)
        strong = np.abs(b_ref) >= 0.01
        overall, strict = float(same.mean()), float(same[strong].mean()) if strong.any() else 1.0
        b_err = np.abs(o.bf16(out / "dump_b_tile.bin", shape) - b_ref).max()
        return check("mt_qjl_projection", strict >= 0.998 and overall >= 0.95, overall=overall, strict=strict,
                     b_max_abs=b_err)

    def rnorm() -> dict[str, Any]:
        ref = np.linalg.norm(o.bf16(ins / "dump_r_tile.bin", shape), axis=1)
        dev = o.bf16(out / "dump_rnorm_tile.bin", (NUM_VECTORS,))
        rel = float(np.max(np.abs(dev - ref) / np.maximum(ref, 1e-6)))
        return check("mt_rnorm", rel < 0.02, max_rel=rel)

    table = {"mt-rotation": ("mt_rotation", rotation), "mt-polarquant": ("mt_polarquant", polarquant),
             "mt-inverse-rotation": ("mt_inverse_rotation", inverse_rotation), "mt-residual": ("mt_residual", residual),
             "mt-qjl-projection": ("mt_qjl_projection", projection), "mt-rnorm": ("mt_rnorm", rnorm)}
    return [guarded(*table[name])] if name in table else []


def committed_checks(np: Any, src: Path, out: Path) -> list[dict[str, Any]]:
    """Return byte comparisons of the baseline's outputs with the codec's committed dumps (provenance unknown)."""
    rows = []
    for name in ("dump_input.bin", "dump_quant_indices.bin", "dump_quant_centroids.bin", "dump_qjl.bin",
                 "dump_output.bin"):
        if (out / name).is_file() and (src / name).is_file():
            a, b = (out / name).read_bytes(), (src / name).read_bytes()
            diff = sum(1 for i in range(min(len(a), len(b))) if a[i] != b[i])
            rows.append({"file": name, "identical": a == b, "bytes": [len(a), len(b)], "differing_bytes": diff})
    return rows


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare every kept output with the reference; write report/compare.json and compare.txt."""
    import numpy as np  # noqa: PLC0415 - compare runs under a Python that has NumPy

    run = Path(args.run)
    src, outputs, report = run / "src", run / "outputs", run / "report"
    try:
        o = Oracle(np, src, run / "raw" / "compare-ref")
    except ValueError as error:
        print(printable(f"p53: compare refuses to import the reference: {error}", 300))
        return 2
    result: dict[str, Any] = {"numpy": np.__version__, "python": sys.version.split()[0],
                              "variants": {"dev0": DEV0_NOTE, "asis": ASIS_NOTE},
                              "reference_sha256": REFERENCE_SHA256,
                              "sign_table_equals_make_sign_diagonal_128_seed0": bool(np.array_equal(o.sign_table, o.D)),
                              "programs": {}}
    for program in PROGRAMS:
        out = outputs / f"dev0-{program.name}"
        if not any((out / n).is_file() for n in program.outputs):
            continue
        try:
            checks = baseline_checks(o, out) if program.name == "baseline" else stage_checks(o, program.name, out)
        except (OSError, ValueError) as error:
            checks = [{"check": "error", "pass": False, "error": printable(str(error), 200)}]
        result["programs"][f"dev0-{program.name}"] = {"checks": checks}
    for variant in ("dev0", "asis"):
        out = outputs / f"{variant}-baseline"
        if out.is_dir():
            result.setdefault("committed_dumps", {})[variant] = committed_checks(np, src, out)
    write_json(report / "compare.json", result)
    lines = [DEV0_NOTE, f"numpy {result['numpy']}; sign table = make_sign_diagonal(128, 0): "
             f"{result['sign_table_equals_make_sign_diagonal_128_seed0']}"]
    for key, value in result["programs"].items():
        for item in value["checks"]:
            metrics = {k: v for k, v in item.items() if k not in ("check", "pass", "threshold_source")}
            lines.append(f"{key} {item['check']}: {'PASS' if item['pass'] else 'FAIL'} {json.dumps(metrics)}")
    write_ascii(report / "compare.txt", "\n".join(lines) + "\n")
    print(f"compare: {sum(len(v['checks']) for v in result['programs'].values())} checks over "
          f"{len(result['programs'])} program runs (report/compare.txt)")
    return 0


# ----------------------------------------------------------------------------------------------------------------
# summarize
# ----------------------------------------------------------------------------------------------------------------


def load_steps(report: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Return the step records by (step, variant, program)."""
    steps = {}
    for path in sorted((report / "steps").glob("*.json")):
        record = json.loads(path.read_text())
        steps[record["step"], record["variant"], record["program"]] = record
    return steps


def verdict_letter(check_record: dict[str, Any]) -> str:
    """Return P (pass), F (fail), or A (failed because a file it needs is absent) for a reference check."""
    return "P" if check_record["pass"] else ("A" if "absent" in check_record else "F")


def short_cell(steps: dict[tuple[str, str, str], dict[str, Any]], key: tuple[str, str, str], outputs: int) -> str:
    """Return a table cell for one step: skip, refused, a build's ok or FAIL, or a run's status fields."""
    record = steps.get(key, {"skipped": "not recorded"})
    if "skipped" in record or "refused" in record:
        return "skip" if "skipped" in record else "refused"
    if key[0] == "build":
        return "ok" if record["artifact"] else f"FAIL({record['diagnostics']['by_severity'].get('error', 0)}e)"
    outs = sum(1 for v in record["outputs"].values() if isinstance(v, dict))
    hang = "H" if record["hang"] else "-"
    return (f"{record['exit_code']} {hang} {record['sim_ub']} {record['sim_gap'] or '-'} "
            f"{record['jit_error_count']} {outs}/{outputs}")


def table_lines(steps: dict[tuple[str, str, str], dict[str, Any]], compare: dict[str, Any]) -> list[str]:
    """Return one line per program: both builds, the dev0 run, the asis run, and the dev0 comparison verdicts."""
    lines = [f"{'program':<20} {'build asis/dev0':<16} {'dev0 run (exit hang ub gap jit outs)':<40} "
             f"{'asis run':<14} reference checks (dev0)"]
    for program in PROGRAMS:
        n = len(program.outputs)
        builds = (f"{short_cell(steps, ('build', 'asis', program.name), n)}/"
                  f"{short_cell(steps, ('build', 'dev0', program.name), n)}")
        checks = compare.get("programs", {}).get(f"dev0-{program.name}", {}).get("checks", [])
        verdicts = " ".join(f"{c['check']}={verdict_letter(c)}" for c in checks) or "-"
        lines.append(f"{program.name:<20} {builds:<16} {short_cell(steps, ('run', 'dev0', program.name), n):<40} "
                     f"{short_cell(steps, ('run', 'asis', program.name), n):<14} {verdicts}")
    return lines


def fact_lines(steps: dict[tuple[str, str, str], dict[str, Any]], report: Path) -> list[str]:
    """Return the findings, JIT errors, unpack_to_dest facts, asis failure kinds, and probe lines."""
    out = []
    for (_step, variant, name), record in sorted(steps.items()):
        for text in record.get("findings", [])[:2]:
            out.append(f"finding {variant} {name}: {text}"[:200])
        for text in record.get("jit_errors", [])[:2]:
            out.append(f"jit {variant} {name}: {text}"[:200])
        for text in record.get("diagnostics", {}).get("first_errors", [])[:2]:
            out.append(f"build error {variant} {name}: {text}"[:200])
    unpack = {n: r["unpack_to_dest"] for (s, v, n), r in steps.items()
              if s == "run" and v == "dev0" and "unpack_to_dest" in r}
    flagged = [n for n, f in unpack.items() if f["issue18_path_possible"]]
    out.append(f"unpack_to_dest: {len(unpack)} dev0 runs read; DST_ACCUM_MODE values "
               f"{sorted({m for f in unpack.values() for m in f['dst_accum_mode']})}; unpack formats "
               f"{sorted({x for f in unpack.values() for x in f['unpack_src_format'] + f['unpack_dst_format']})}; "
               f"issue #18 path possible in: {' '.join(flagged) or 'none'}")
    asis = [r for (s, v, _n), r in steps.items() if s == "run" and v == "asis" and "exit_code" in r]
    exits = sorted({str(r["exit_code"]) for r in asis})
    kinds = sorted({k for r in asis for k in r.get("stderr_kind") or []})
    written = sum(1 for r in asis for v in r["outputs"].values() if isinstance(v, dict))
    out.append(f"asis runs (device id 6): {len(asis)} ran; exit statuses {' '.join(exits) or '-'}; stderr kinds "
               f"{' '.join(kinds) or '-'}; output files written {written}")
    probe_path = report / "device-probe.json"
    probe = json.loads(probe_path.read_text()) if probe_path.is_file() else {"skipped": "not recorded"}
    if "skipped" in probe:
        out.append(printable(f"probe: skipped ({probe['skipped']})"))
    else:
        out.append(f"probe: ok={probe['ok']}; /dev entries {len(probe['dev_entries'])}, unexpected "
                   f"{probe['unexpected_dev'] or 'none'}; /sys directories checked {probe['sys_checked']}, not "
                   f"empty {probe['sys_nonempty'] or 'none'}; /proc/bus/pci {probe.get('proc_bus_pci', '-')} "
                   "(recorded, not gated)")
    return out


def cmd_summarize(args: argparse.Namespace) -> int:
    """Print the findings and write report/summary.txt."""
    report = Path(args.run) / "report"
    steps = load_steps(report)
    compare = json.loads((report / "compare.json").read_text()) if (report / "compare.json").is_file() else {}
    device = (report / "device.txt").read_text().strip() if (report / "device.txt").is_file() else "-"
    lines = [f"== device: {device}", f"   every wall time in report/steps is {WALL_LABEL}", f"   {DEV0_NOTE}",
             *table_lines(steps, compare), *fact_lines(steps, report)]
    if compare:
        fails = [f"{k} {c['check']}{' (absent)' if 'absent' in c else ''}" for k, v in compare["programs"].items()
                 for c in v["checks"] if not c["pass"]]
        lines.append(f"reference checks: {sum(len(v['checks']) for v in compare['programs'].values())} run, "
                     f"failed: {', '.join(fails) or 'none'}; sign table = make_sign_diagonal(128, 0): "
                     f"{compare.get('sign_table_equals_make_sign_diagonal_128_seed0')}")
        for variant, rows in compare.get("committed_dumps", {}).items():
            lines.append(f"{variant}-baseline vs the codec's committed dumps: " + " ".join(
                f"{row['file']}={'same' if row['identical'] else str(row['differing_bytes']) + 'B'}" for row in rows))
    else:
        lines.append("reference checks: not run (report/compare.json absent)")
    write_ascii(report / "summary.txt", "\n".join(lines) + "\n")
    print("\n".join(line[:200] for line in lines[:60]))
    return 0


def parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    top = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = top.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--src", required=True)
    prep.add_argument("--report", required=True)
    copy = sub.add_parser("copy")
    copy.add_argument("--src", required=True)
    copy.add_argument("--report", required=True)
    copy.add_argument("--copy", required=True)
    batch = sub.add_parser("batch")
    batch.add_argument("--run", required=True)
    batch.add_argument("--toolchains", required=True)
    batch.add_argument("--runs-root", required=True)
    batch.add_argument("--budget-s", type=float, required=True)
    batch.add_argument("--wall-scale", type=float, default=1.0)
    batch.add_argument("--stop-file", action="append", default=[])
    for name in ("compare", "summarize"):
        sub.add_parser(name).add_argument("--run", required=True)
    return top


def main(argv: list[str]) -> int:
    """Dispatch to the subcommand."""
    args = parser().parse_args(argv)
    if args.command == "batch" and not (0 < args.wall_scale <= 1 and args.budget_s > 0):
        print("p53: --wall-scale must be in (0, 1] and --budget-s positive")
        return 2
    commands = {"prepare": cmd_prepare, "copy": cmd_copy, "batch": cmd_batch, "compare": cmd_compare,
                "summarize": cmd_summarize}
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
