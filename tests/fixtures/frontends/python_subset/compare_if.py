"""Comparisons, if, elif, and else, and, or, and not, and a scalar local (a Python-subset frontend fixture)."""


def classify(x: float, lo: float, hi: float, strict: bool) -> int:
    kind = 0
    if x < lo or x > hi:
        kind = 2
    elif strict and not x == lo:
        kind = 1
    else:
        kind = -1
    return kind


def is_small(n: int) -> bool:
    return n <= 9
