"""Int arithmetic: floor division and modulo with Python's rounding, bit operations, and true division."""


def mix(a: int, b: int) -> int:
    return (a + b) * (a - b) // 3 % b


def bits(a: int, b: int) -> int:
    return (a & b) | (a ^ b) << 2 >> 1


def ratio(a: int, b: int) -> float:
    return a / b
