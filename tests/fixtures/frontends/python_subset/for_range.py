"""Loops over range(stop), range(start, stop), and range(start, stop, step), with augmented assignment."""


def total(n: int) -> int:
    s = 0
    for i in range(n):
        s += i
    for j in range(1, n):
        s -= j
    for k in range(0, n, 3):
        s += k * 2
    return s
