"""A prange loop with a start and a step, and a range loop that reuses its variable's name."""

from typing import Annotated

from lassi_io import prange


def stripes(n: int, out: Annotated[memoryview, "f64"]) -> None:
    for i in range(n):
        out[i] = 0.0
    for i in prange(1, n, 2):
        out[i] = 1.0
