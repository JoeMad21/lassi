"""Another name for an array, plain and annotated (a Python-subset frontend fixture)."""

from typing import Annotated


def mirror(a: Annotated[memoryview, "i64"], n: int) -> int:
    b = a
    c: Annotated[memoryview, "i64"] = b
    b[0] = n
    return c[0] + len(b)
