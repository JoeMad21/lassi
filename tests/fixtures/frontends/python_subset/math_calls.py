"""Math intrinsics, abs(), float(), and int() (a Python-subset frontend fixture)."""

import math


def norm(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)


def wave(t: float, k: int) -> float:
    return math.exp(-t) * math.cos(t) + abs(math.sin(t)) - math.log(float(k))


def whole(x: float) -> int:
    return int(x) + abs(-3)
