"""P17.1 spike: read the gfx942 device code object out of a HIP object's .hip_fatbin section, on the workstation.

/opt/rocm/core-7.12/lib/llvm/bin on alpha01 holds no llvm-readelf (the batch's hip.json lists it under tools_missing),
so the batch could not list the sections of the object it compiled. This script reads the object pulled from the
batch's run directory (`rx pull --path lassi-runs/p17-frameworks/<rx id>/raw`, raw/hip-compile-work/probe.o) with the
standard library only: the host ELF's section table, the clang offload bundle in .hip_fatbin (magic, entry count, then
each entry's offset, size, and id), and each device entry's ELF header (e_machine and e_flags). Usage:

    python plans/spikes/p17-frameworks/inspect_bundle.py <probe.o>

It prints one line per bundle entry. EM_AMDGPU is 224, and the low byte of e_flags (EF_AMDGPU_MACH) is 0x4c for
gfx942 (LLVM's ELF.h, EF_AMDGPU_MACH_AMDGCN_GFX942).
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"
EM_AMDGPU = 224
MACH_GFX942 = 0x4C


def sections(data: bytes) -> dict[str, tuple[int, int]]:
    """Return {name: (offset, size)} from a little-endian ELF64 file's section headers."""
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    headers = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize) for i in range(shnum)]
    strtab = headers[shstrndx][4]
    out = {}
    for header in headers:
        name_end = data.index(b"\0", strtab + header[0])
        out[data[strtab + header[0]:name_end].decode("ascii")] = (header[4], header[5])
    return out


def entries(bundle: bytes) -> list[tuple[str, bytes]]:
    """Return (id, contents) for each entry of a clang offload bundle."""
    if not bundle.startswith(MAGIC):
        raise SystemExit("no clang offload bundle magic at the start of .hip_fatbin")
    count, = struct.unpack_from("<Q", bundle, len(MAGIC))
    at = len(MAGIC) + 8
    out = []
    for _ in range(count):
        offset, size, id_size = struct.unpack_from("<QQQ", bundle, at)
        at += 24
        out.append((bundle[at:at + id_size].decode("ascii"), bundle[offset:offset + size]))
        at += id_size
    return out


def main(path: str) -> int:
    """Print the bundle entries of the object at `path` and each device entry's ELF machine and flags."""
    data = Path(path).read_bytes()
    found = sections(data)
    if ".hip_fatbin" not in found:
        print("no .hip_fatbin section")
        return 1
    offset, size = found[".hip_fatbin"]
    print(f".hip_fatbin: offset {offset:#x}, {size} bytes")
    for ident, body in entries(data[offset:offset + size]):
        if body[:4] != b"\x7fELF":
            print(f"entry {ident}: {len(body)} bytes, not an ELF (host entries are empty)")
            continue
        machine, = struct.unpack_from("<H", body, 0x12)
        flags, = struct.unpack_from("<I", body, 0x30)
        gfx942 = machine == EM_AMDGPU and flags & 0xFF == MACH_GFX942
        print(f"entry {ident}: {len(body)} bytes, ELF e_machine {machine}, e_flags {flags:#x}, gfx942 code: {gfx942}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
