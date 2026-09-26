"""Harness: the control plane's side of the harness I/O contract (bible Harness Contract, Frontend Rules).

- lassi.harness.lassi_io: the lassi_io file format (one named array per
  binary file), its reader and writer, and the seeded generator of held-out
  input files. Programs read and write the same files through the C and C++
  header assets/harness/c/lassi_io.h.
"""

from lassi.harness.lassi_io import (
    DTYPES,
    MAGIC,
    VERSION,
    DType,
    LassiArray,
    LassiIOError,
    Pcg32,
    generate_inputs,
    read_array,
    stream_of,
    write_array,
)

__all__ = [
    "DTYPES",
    "MAGIC",
    "VERSION",
    "DType",
    "LassiArray",
    "LassiIOError",
    "Pcg32",
    "generate_inputs",
    "read_array",
    "stream_of",
    "write_array",
]
