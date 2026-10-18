"""Array allocation in the binding's form, zero-filled as bytearray is (a Python-subset frontend fixture)."""


def ramp(n: int) -> None:
    y = memoryview(bytearray(n * 8)).cast("d")
    c = memoryview(bytearray(16 * 4)).cast("i")
    for i in range(n):
        y[i] = 0.5 * i
    c[0] = len(y)
