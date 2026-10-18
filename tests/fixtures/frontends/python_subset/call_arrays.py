"""Calls with array arguments, an int argument to a float parameter, and a call inside a loop."""

from typing import Annotated


def fill(n: int, v: float, out: Annotated[memoryview, "f64"]) -> None:
    for i in range(n):
        out[i] = v


def first(a: Annotated[memoryview, "f64"]) -> float:
    return a[0]


def run(n: int, a: Annotated[memoryview, "f64"]) -> float:
    fill(n, 2, a)
    t = 0.0
    for i in range(n):
        t += first(a) * i
    return t
