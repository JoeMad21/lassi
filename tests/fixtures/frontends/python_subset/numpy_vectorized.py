"""A NumPy vectorized expression assigned into an array (a Python-subset frontend fixture)."""

from typing import Annotated

import numpy as np


def saxpy_sqrt(
    a: float, xs: Annotated[memoryview, "f64"], ys: Annotated[memoryview, "f64"], zs: Annotated[memoryview, "f64"]
) -> None:
    x = np.asarray(xs)
    y = np.asarray(ys)
    z = np.asarray(zs)
    z[:] = a * x + np.sqrt(y)
