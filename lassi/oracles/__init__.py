"""Oracles: compare a run with the reference and return alignment in [0, 1] (bible Oracles, Oracle interface).

Importing this package registers the Oracle `stdout_mask` (with its passfail
choice) in the default registry. See lassi.oracles.stdout_mask for the rules
and the mask files under assets/harness/masks/.
"""

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
    "MASKS_ROOT",
    "MASK_MARKER",
    "StdoutMaskOracle",
    "alignment",
    "load_masks",
    "mask_stdout",
    "passfail_score",
    "stdout_mask_score",
]
