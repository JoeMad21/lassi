"""Array loads and stores with element conversions, len(), and augmented element assignment."""

from typing import Annotated


def scale(x: Annotated[memoryview, "f32"], k: Annotated[memoryview, "i32"], m: Annotated[memoryview, "u8"]) -> None:
    for i in range(len(x)):
        x[i] = x[i] * k[i]
        m[i] += 1
