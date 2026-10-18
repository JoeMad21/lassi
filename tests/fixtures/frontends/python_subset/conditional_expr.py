"""A conditional expression (a Python-subset frontend fixture)."""


def inside(zr: float, zi: float) -> int:
    return 1 if zr * zr + zi * zi <= 4.0 else 0
