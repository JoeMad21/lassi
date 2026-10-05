"""Oracles: compare a run with the reference and return alignment in [0, 1] (bible Oracles, Oracle interface).

Importing this package registers the Oracles `stdout_mask` (with its
passfail choice) and `binary_io` in the default registry. See
lassi.oracles.stdout_mask for the stdout rules and the mask files under
assets/harness/masks/, and lassi.oracles.binary_io for the rules over
lassi_io output files.
"""

from lassi.oracles.binary_io import FROM_BASELINE, BinaryIOOracle, compare_arrays, pearson
from lassi.oracles.stdout_mask import (
    MASK_MARKER,
    MASKS_ROOT,
    StdoutMaskOracle,
    alignment,
    load_masks,
    mask_stdout,
    passfail_score,
    stdout_mask_score,
)

__all__ = [
    "FROM_BASELINE",
    "MASKS_ROOT",
    "MASK_MARKER",
    "BinaryIOOracle",
    "StdoutMaskOracle",
    "alignment",
    "compare_arrays",
    "load_masks",
    "mask_stdout",
    "passfail_score",
    "pearson",
    "stdout_mask_score",
]
