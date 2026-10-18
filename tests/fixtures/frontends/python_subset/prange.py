"""A prange loop from the binding, with locals private to each iteration (a Python-subset frontend fixture)."""

from typing import Annotated

from lassi_io import prange


def row_sums(n: int, a: Annotated[memoryview, "f64"], out: Annotated[memoryview, "f64"]) -> None:
    for i in prange(n):
        s = 0.0
        for j in range(n):
            s += a[i * n + j]
        out[i] = s
