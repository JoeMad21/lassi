"""Compare a probe's outputs with a numpy reference (plans/spikes/ttmlir-runtime.md, 2026-10-30).

    <runtime venv>/bin/python -I compare.py <case> <workdir>

Reads the inputs and outputs with the repository's standard-library binding
(assets/harness/python/lassi_io.py), computes the reference with numpy, and
prints one JSON object: the output's dtype and shape, and how it differs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HARNESS = Path(__file__).resolve().parents[3] / "assets" / "harness" / "python"
sys.path.insert(0, str(HARNESS))
import lassi_io  # noqa: E402  (the harness binding, placed on the path above)

FORMATS = {"f32": np.float32, "i32": np.int32, "u32": np.uint32, "u8": np.uint8}


def load(path: Path) -> np.ndarray:
    """Return a lassi_io file as a numpy array of its dtype and shape; bf16 comes back widened to float32 exactly."""
    array = lassi_io.read_array(path)
    if array.dtype == "bf16":
        bits = np.frombuffer(bytes(array.data), dtype=np.uint16).astype(np.uint32) << 16
        return bits.view(np.float32).reshape(array.shape)
    return np.frombuffer(bytes(array.data), dtype=FORMATS[array.dtype]).reshape(array.shape)


def main() -> int:
    """Compare one case and print the JSON."""
    case, workdir = sys.argv[1], Path(sys.argv[2])
    inputs = workdir / "@inputs"
    if case.startswith("add"):
        # The f32 sum of the inputs as read (a bf16 input widened exactly); a bf16 output differs by its rounding.
        expected = (load(inputs / "a.lassiio") + load(inputs / "b.lassiio")).astype(np.float32)
        got = load(workdir / "c.lassiio")
    else:
        expected = load(inputs / "counts0.lassiio").astype(np.int64)
        np.add.at(expected, load(inputs / "index.lassiio").astype(np.int64), load(inputs / "ones.lassiio"))
        got = load(workdir / "counts.lassiio").astype(np.int64)
    same_shape = got.size == expected.size
    diff = np.abs(got.reshape(-1).astype(np.float64) - expected.reshape(-1).astype(np.float64)) if same_shape else None
    print(json.dumps({
        "numpy": np.__version__,
        "dtype": str(got.dtype),
        "shape": list(got.shape),
        "elements": int(got.size),
        "reference_sum": float(expected.sum()),
        "output_sum": float(got.sum()),
        "max_abs": None if diff is None else float(diff.max()),
        "mismatches": None if diff is None else int(np.count_nonzero(diff)),
        "first_values": got.reshape(-1)[:8].tolist(),
        "first_reference": expected.reshape(-1)[:8].tolist(),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
