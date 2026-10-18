"""Float arithmetic, unary minus, and an int operand promoted to float (a Python-subset frontend fixture)."""


def blend(a: float, b: float, k: int) -> float:
    return -a / (b - 1.5) * k
