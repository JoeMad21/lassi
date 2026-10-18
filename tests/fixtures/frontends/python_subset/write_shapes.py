"""Outputs of rank 0 and rank 2, and a write inside a loop (a Python-subset frontend fixture)."""

import sys

from lassi_io import write_array


def main() -> None:
    s = memoryview(bytearray(1 * 8)).cast("d")
    g = memoryview(bytearray(6 * 4)).cast("i")
    write_array(sys.argv[1], "norm", "f64", (), s)
    write_array(sys.argv[2], "grid", "i32", (2, 3), g)
    for k in range(2):
        g[k] = k
        write_array(sys.argv[3], "last", "i32", (6,), g)
