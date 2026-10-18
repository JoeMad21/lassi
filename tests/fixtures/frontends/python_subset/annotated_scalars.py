"""Annotated scalar assignments, and a bool local in its own slot (a Python-subset frontend fixture)."""


def clamp_count(n: int, x: float) -> int:
    limit: int = 10
    scale: float = 0.5
    over: bool = x * scale > 1.0
    count: int = n
    if over:
        count = limit
    return count
