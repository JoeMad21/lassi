"""Functions over float scalars, a return, and a call between them (a Python-subset frontend fixture)."""


def axpy(a: float, x: float, y: float) -> float:
    return a * x + y


def twice(x: float) -> float:
    return axpy(2.0, x, 0.0)
