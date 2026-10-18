"""A whole program: inputs and outputs through the binding, and the entry guard (a Python-subset frontend fixture)."""

import sys
from typing import Annotated

from lassi_io import read_array, write_array


def double(n: int, x: Annotated[memoryview, "f64"], y: Annotated[memoryview, "f64"]) -> None:
    for i in range(n):
        y[i] = 2.0 * x[i]


def main() -> None:
    x: Annotated[memoryview, "f64"] = read_array(sys.argv[1]).data
    n = len(x)
    y = memoryview(bytearray(n * 8)).cast("d")
    double(n, x, y)
    write_array(sys.argv[2], "y", "f64", (n,), y)


if __name__ == "__main__":
    main()
