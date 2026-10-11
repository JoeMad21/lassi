"""Stand-ins for the local dry run of the P5.3 batch (plans/spikes/p5-tier-b/dryrun.sh); never used on alpha01.

Two roles, both standard library only:

- clang: replaces the pinned clang++-20 (the dry run's copy of toolchains/tt-metal.pin names a script that calls
  this). `--version` prints the pin's EXPECT_VERSION. A build checks that the command holds the pinned words
  (-std=c++20, -idirafter ., an include under tt-metal@5280a9cf, -o main), names the program by its host source,
  reads the variant from the device-id literal, prints one warning for the baseline, fails with a clang-style error
  when the faults file says "build-error", and otherwise writes `main`, a launcher of this file's run role.
- run: emulates a codec program's file behavior on its inputs, with the reference math of tier_b.py (bf16 by
  truncation): writes the dumps the program writes, a JIT descriptor header under $TT_METAL_CACHE, and a timing
  line. "asis" (device id 6) aborts after the reads and writes the real program does before opening the device.
  Faults: "ub" prints a ttsim finding and exits 1, "hang" sleeps, "jit-error" prints a kernel build failure and
  aborts.

It holds no codec or tt-metal text: constants are read from the codec checkout the dry run names.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tier_b  # noqa: E402

HOSTS = {"turboquant_host.cpp": "baseline", "turboquant_multi_tile.cpp": "mt-rotation",
         "turboquant_multi_tile_pq.cpp": "mt-polarquant", "turboquant_multi_tile_ir.cpp": "mt-inverse-rotation",
         "turboquant_multi_tile_resid.cpp": "mt-residual", "turboquant_multi_tile_proj.cpp": "mt-qjl-projection",
         "turboquant_multi_tile_rnorm.cpp": "mt-rnorm"}
LAUNCHER = """#!/usr/bin/python3
import runpy, sys
sys.argv = [{standin!r}, "run", {program!r}, {variant!r}, {fault!r}, {codec!r}] + sys.argv[1:]
runpy.run_path({standin!r}, run_name="__main__")
"""


def expect_version() -> str:
    """Return EXPECT_VERSION from the dry run's copy of toolchains/tt-metal.pin."""
    for line in (HERE.parents[2] / "toolchains" / "tt-metal.pin").read_text().splitlines():
        if line.startswith("EXPECT_VERSION="):
            return line.split("=", 1)[1].strip('"')
    raise SystemExit("no EXPECT_VERSION")


def cmd_clang(args: argparse.Namespace, rest: list[str]) -> int:
    """Act as clang++-20 for ttmetal-host: --version, or a 'build' that writes the launcher `main`."""
    if "--version" in rest:
        print(expect_version())
        return 0
    words = " ".join(rest)
    for needed in ("-std=c++20", "-idirafter .", "tt-metal@5280a9cf", "-o main"):
        if needed not in words:
            print(f"clang++-20: error: the dry-run stand-in expected '{needed}' in the command", file=sys.stderr)
            return 1
    sources = [w for w in rest if w.endswith(".cpp")]
    if len(sources) != 1 or Path(sources[0]).name not in HOSTS:
        print(f"clang++-20: error: unexpected sources {sources}", file=sys.stderr)
        return 1
    program = HOSTS[Path(sources[0]).name]
    text = Path(sources[0]).read_text(encoding="utf-8")
    variant = "asis" if tier_b.DEVICE_ASIS in text else "dev0"
    fault = json.loads(Path(args.faults).read_text()).get(f"{variant}:{program}", "")
    if fault == "build-error":
        print(f"{sources[0]}:57:5: error: use of undeclared identifier 'p53_dry_fault'", file=sys.stderr)
        print("1 error generated.", file=sys.stderr)
        return 1
    if program == "baseline":
        print(f"{sources[0]}:338:14: warning: unused variable 'grid' [-Wunused-variable]", file=sys.stderr)
    Path("main").write_text(LAUNCHER.format(program=program, variant=variant, fault=fault, codec=args.codec,
                                            standin=str(Path(__file__).resolve())))
    Path("main").chmod(0o755)
    return 0


def host_vectors(n: int, d: int, seed: int = 42) -> list[float]:
    """Return the baseline host's normalized test vectors, flat, with float32 arithmetic emulated."""
    f32 = lambda x: struct.unpack("<f", struct.pack("<f", x))[0]  # noqa: E731
    state, out = seed, []
    for _ in range(n):
        row, nm = [], 0.0
        for _ in range(d):
            state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            bits = ((state >> 33) & 0x7FFFFF) | 0x3F800000
            value = f32(struct.unpack("<f", struct.pack("<I", bits))[0] - 1.5)
            row.append(value)
            nm = f32(nm + f32(value * value))
        nm = f32(math.sqrt(nm))
        out += [f32(v / nm) for v in row]
    return out


def rows_of(values: list[float]) -> list[list[float]]:
    """Return a flat N x 128 list as rows."""
    return [values[r * tier_b.DIM:(r + 1) * tier_b.DIM] for r in range(tier_b.NUM_VECTORS)]


def write_descriptor() -> None:
    """Write one JIT descriptor header under $TT_METAL_CACHE in the pin's format (Float16_b CBs, no dest acc)."""
    cache = Path(os.environ.get("TT_METAL_CACHE", "@ttsim-missing")) / "tt-metal-cache-dry" / "kernels" / "k" / "0"
    cache.mkdir(parents=True, exist_ok=True)
    formats = ",".join(["5", "5"] + ["255"] * 30) + ","
    (cache / "chlkc_unpack_data_format.h").write_text(
        f"constexpr std::int32_t unpack_src_format[32] = {{\n    {formats}\n}};\n"
        f"constexpr std::int32_t unpack_dst_format[32] = {{\n    {formats}\n}};\n")
    (cache / "chlkc_dst_accum_mode.h").write_text("constexpr bool DST_ACCUM_MODE = false;\n")


def baseline(codec: Path) -> None:
    """Write the baseline's dumps from its own generated input."""
    text = tier_b.read_text(codec / "include" / "codebooks.h")
    sign = tier_b.header_numbers(text, "SIGN_DIAGONAL_d128")
    cents = tier_b.header_numbers(text, "CODEBOOK_CENTROIDS_b4_d128")
    bounds = tier_b.header_numbers(text, "CODEBOOK_BOUNDARIES_b4_d128")
    x = tier_b.bf16_values(tier_b.bf16_bytes(host_vectors(tier_b.NUM_VECTORS, tier_b.DIM)))
    z = tier_b.bf16_values(tier_b.bf16_bytes([v for row in tier_b.wht_rows(
        [[s * v for s, v in zip(sign, row, strict=True)] for row in rows_of(x)]) for v in row]))
    idx = [sum(1 for b in bounds if v > b) for v in z]
    cent = tier_b.bf16_values(tier_b.bf16_bytes([cents[i] for i in idx]))
    xhat = [s * v for row in tier_b.wht_rows(rows_of(cent)) for s, v in zip(sign, row, strict=True)]
    S = tier_b.lfsr_rows(tier_b.DIM, tier_b.DIM, 0xCAFEBABE)
    qjl, out = b"", b""
    for r in range(tier_b.NUM_VECTORS):
        res = [a - b for a, b in zip(x[r * 128:(r + 1) * 128], xhat[r * 128:(r + 1) * 128], strict=True)]
        norm = math.sqrt(sum(v * v for v in res))
        bits = [sum(s * v for s, v in zip(row, res, strict=True)) / (norm + 1e-12) > 0 for row in S]
        packed = bytes(sum(1 << k for k in range(8) if bits[8 * j + k]) for j in range(16))
        record = packed + tier_b.bf16_bytes([norm])
        pq = bytes(idx[r * 128 + 2 * j] | idx[r * 128 + 2 * j + 1] << 4 for j in range(64))
        qjl += record
        out += pq + record + bytes(14)
    Path("dump_rotated.bin").write_bytes(tier_b.bf16_bytes(z))
    pairs = bytes(idx[j] | idx[j + 1] << 4 for j in range(0, len(idx), 2))
    Path("dump_quant_indices.bin").write_bytes(pairs)
    Path("dump_quant_centroids.bin").write_bytes(tier_b.bf16_bytes(cent))
    Path("dump_qjl.bin").write_bytes(qjl)
    Path("dump_output.bin").write_bytes(out)


def stage(program: str, codec: Path, input_path: str) -> None:
    """Write a multi_tile stage's dump from its inputs."""
    have = {n: Path(n).read_bytes() for n in ("dump_input.bin", "dump_quant_centroids.bin", "dump_xhat_tile.bin",
                                              "dump_rotated_tile.bin", "dump_r_tile.bin") if Path(n).is_file()}
    if program == "mt-inverse-rotation":
        have["dump_quant_centroids.bin"] = Path(input_path).read_bytes()
    text = tier_b.read_text(codec / "include" / "codebooks.h")
    if program == "mt-polarquant":
        cents = tier_b.header_numbers(text, "CODEBOOK_CENTROIDS_b4_d128")
        bounds = tier_b.header_numbers(text, "CODEBOOK_BOUNDARIES_b4_d128")
        z = tier_b.bf16_values(have["dump_rotated_tile.bin"])
        picked = [cents[sum(1 for b in bounds if v >= b)] for v in z]
        Path("dump_centroids_tile.bin").write_bytes(tier_b.bf16_bytes(picked))
        return
    if program in ("mt-qjl-projection", "mt-rnorm"):
        r = rows_of(tier_b.bf16_values(have["dump_r_tile.bin"]))
        if program == "mt-rnorm":
            norms = [math.sqrt(sum(v * v for v in row)) for row in r]
            Path("dump_rnorm_tile.bin").write_bytes(tier_b.bf16_bytes(norms))
            return
        S = tier_b.lfsr_rows(tier_b.DIM, tier_b.DIM, 0xCAFEBABE)
        b = [sum(s * v for s, v in zip(srow, row, strict=True)) for row in r for srow in S]
        b16 = tier_b.bf16_bytes(b)
        Path("dump_b_tile.bin").write_bytes(b16)
        signs = [not (h >> 15) for h in struct.unpack(f"<{len(b)}H", b16)]
        Path("dump_bits_tile.bin").write_bytes(bytes(sum(1 << k for k in range(8) if signs[8 * j + k])
                                                     for j in range(len(signs) // 8)))
        return
    name = {"mt-rotation": "dump_rotated_tile.bin", "mt-inverse-rotation": "dump_xhat_tile.bin",
            "mt-residual": "dump_r_tile.bin"}[program]
    have.pop(name, None)
    Path(name).write_bytes(tier_b.synthesize(name, codec, have))


def cmd_run(args: argparse.Namespace, rest: list[str]) -> int:
    """Emulate a codec program (see the module docstring)."""
    input_path = rest[rest.index("--input") + 1] if "--input" in rest else "dump_quant_centroids.bin"
    program = tier_b.PROGRAMS[[p.name for p in tier_b.PROGRAMS].index(args.program)]
    for name in program.inputs:
        if not Path(name).is_file():
            print(f"cannot open {name}", file=sys.stderr)
            os.abort()
    if args.program == "baseline":
        Path("dump_input.bin").write_bytes(tier_b.bf16_bytes(host_vectors(tier_b.NUM_VECTORS, tier_b.DIM)))
    if args.variant == "asis":
        print("terminate called after throwing an instance of 'std::runtime_error' (dry-run stand-in: TT_FATAL "
              "device 6 is not a chip of this cluster)", file=sys.stderr, flush=True)
        os.abort()
    write_descriptor()
    if args.fault == "ub":
        print("[4489] ERROR: UndefinedBehavior: tensix_unpacr: unpack_to_dst=0 (dry-run stand-in)", flush=True)
        return 1
    if args.fault == "hang":
        time.sleep(3600)
    if args.fault == "jit-error":
        print("trisc0 build failed. Log: multi_tile/kernels/x_compute.cpp:70:5: error: 'p53_dry_fault' was not "
              "declared in this scope\n", flush=True)
        os.abort()
    if args.program == "baseline":
        baseline(Path(args.codec))
    else:
        stage(args.program, Path(args.codec), input_path)
    print("  Dispatch+execute time: 0.1 ms (dry-run stand-in)")
    return 0


def main(argv: list[str]) -> int:
    """Dispatch by position, so no compiler or program argument is read as this file's own option:
    `clang <faults file> <codec dir> <clang args...>` or `run <program> <variant> <fault> <codec dir> <args...>`."""
    if argv[:1] == ["clang"] and len(argv) >= 3:
        return cmd_clang(argparse.Namespace(faults=argv[1], codec=argv[2]), argv[3:])
    if argv[:1] == ["run"] and len(argv) >= 5:
        names = ("program", "variant", "fault", "codec")
        return cmd_run(argparse.Namespace(**dict(zip(names, argv[1:5], strict=True))), argv[5:])
    print("usage: dry_standin.py clang FAULTS CODEC ARGS... | run PROGRAM VARIANT FAULT CODEC ARGS...")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
