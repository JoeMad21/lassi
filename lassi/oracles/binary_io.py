"""The binary_io oracle: compare a run's output files with the reference's (bible Oracles, binary_io row).

Programs write their outputs as lassi_io files, one named array per file
(bible Harness Contract; lassi.harness.lassi_io). This oracle reads both
runs' output files, matches arrays by the name in each file's header (never
by file name), and gives each output an OutputStats (lassi.core.record).

Rules [DESIGN, task P4.4]:

- Float dtypes (f16, bf16, f32, f64): a NaN equals a NaN in the same
  position, whatever its bits, and an infinity equals an infinity of the
  same sign in the same position; any other pairing with a NaN or an
  infinity fails the output, with a note naming the first such element
  and, when there are more, how many. NaNs and infinities never enter the
  statistics.
  Over the other pairs (both elements finite):
  - max_abs is the largest absolute difference, computed in double; a
    difference that overflows double (f64 only) leaves max_abs None and
    fails the output with a note.
  - max_ulp is the largest ULP distance: the count of representable values
    of the dtype between the two elements, in the dtype's own format (bf16
    in bf16 units, f16 in f16 units), with -0 and +0 one point, so the
    smallest positive and negative subnormals are 2 apart.
  - pcc is 1.0 whenever the two sides are equal element for element (-0
    equals +0). Otherwise it is 0.0 when either side is constant over the
    compared elements, and else the Pearson correlation computed in double,
    clamped to [-1, 1]. With no finite pair (an empty array, or only
    matched NaNs and infinities) pcc is 1.0, max_abs 0.0, and max_ulp 0.
  The output passes when nothing failed it and its metric meets the
  threshold (lassi.core.tolerance within: pcc at least it, max_abs and ulp
  at most it).
- Integer dtypes (i8, u8, i32, u32, i64): exact match counts. The output
  passes only when every element is equal, whatever the metric and
  threshold; max_abs is the exact largest difference as a float, pcc is
  computed in double as for floats, and max_ulp is None.
- A dtype or a shape mismatch fails the output with a note naming it; its
  pcc, max_abs, and max_ulp are None.
- Output files: every file the reference run wrote must be a lassi_io file
  whose array name no other file of that run repeats, and there must be at
  least one; anything else is a reference problem (reference_problem), and
  comparing against such files raises ValueError. On the candidate side,
  an array the candidate lacks is a missing output, one the reference lacks
  an extra output, an array name two candidate files hold is a repeated
  output, and a file that is not a lassi_io file is an unreadable output
  named by its file path; each fails with a note, and nothing raises.
- per_input is 1.0 when every output passes, else 0.0 (alignment).

The recipe section is `oracle: {kind: binary_io, metric: <pcc | max_abs |
ulp>, threshold: <number | from_baseline>}`; both keys are required choices
with no default. A number follows lassi.core.tolerance (pcc in [-1, 1];
max_abs and ulp finite and >= 0; exact match is max_abs 0). With
`from_baseline`, each output's threshold is that output's value of the
metric in the references' agreement the baseline recorded
(Trial.reference_agreement), bound with with_baseline. A float output
the agreement gives no value for fails with a note; an integer output
passes only on exact equality, whatever the agreement holds.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path

from lassi.core.capabilities import ALIGNS_OUTPUT_FILES
from lassi.core.interfaces import RunResult
from lassi.core.record import Alignment, OutputStats
from lassi.core.registry import register
from lassi.core.tolerance import METRICS, STAT_FIELDS, Tolerance, metric_problem, threshold_problem, within
from lassi.harness.lassi_io import DTYPES, LassiArray, read_array

# The threshold value that takes each output's threshold from the references' agreement.
FROM_BASELINE = "from_baseline"
# The names the notes give the two sides of a comparison, unless a caller names them.
SIDES = ("reference", "candidate")

# Float dtypes: the struct code of a value (bf16 is widened to f32) and of its bits, and the bit width.
_FLOAT_VALUE = {"f16": "e", "bf16": "f", "f32": "f", "f64": "d"}
_FLOAT_BITS = {"f16": ("H", 16), "bf16": ("H", 16), "f32": ("I", 32), "f64": ("Q", 64)}
# Integer dtypes: the struct code of a value.
_INT_VALUE = {"i8": "b", "u8": "B", "i32": "i", "u32": "I", "i64": "q"}


# ---------------------------------------------------------------------------
# Elements


def _values(array: LassiArray) -> tuple[float, ...] | tuple[int, ...]:
    """Return the array's elements in C order: floats in double, ints as ints (a bf16 is the f32 of its bits)."""
    count = len(array.data) // DTYPES[array.dtype].itemsize
    if array.dtype == "bf16":
        widened = bytearray(2 * len(array.data))
        widened[2::4] = array.data[0::2]
        widened[3::4] = array.data[1::2]
        return struct.unpack(f"<{count}f", widened)
    code = _FLOAT_VALUE.get(array.dtype) or _INT_VALUE[array.dtype]
    return struct.unpack(f"<{count}{code}", array.data)


def _keys(array: LassiArray) -> list[int]:
    """Return each float element's position on the dtype's number line: bits, or minus the magnitude bits if negative.

    -0 and +0 both map to 0, so the ULP distance of two elements is the
    difference of their keys.
    """
    code, width = _FLOAT_BITS[array.dtype]
    count = len(array.data) * 8 // width
    sign = 1 << (width - 1)
    return [-(bits & (sign - 1)) if bits & sign else bits for bits in struct.unpack(f"<{count}{code}", array.data)]


def _scaled(values: Sequence[float]) -> list[float]:
    """Return `values` scaled by the power of two that puts the largest magnitude in [0.5, 1); exact unless tiny."""
    largest = max(abs(value) for value in values)
    if largest == 0:
        return list(values)
    exponent = math.frexp(largest)[1]
    return [math.ldexp(value, -exponent) for value in values]


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Return the Pearson correlation of two equal-length sequences, in double, clamped to [-1, 1].

    It is exactly 1.0 whenever the sides are equal element for element (no
    element included), so rounding never scores identical outputs below 1.
    Otherwise it is 0.0 when either side is constant, and else the
    correlation, each side first scaled by a power of two, which leaves the
    correlation unchanged and keeps every square and sum finite.
    """
    if list(xs) == list(ys):
        return 1.0
    if len(set(xs)) == 1 or len(set(ys)) == 1:
        return 0.0
    sx, sy = _scaled(xs), _scaled(ys)
    mean_x, mean_y = math.fsum(sx) / len(sx), math.fsum(sy) / len(sy)
    dx = [value - mean_x for value in sx]
    dy = [value - mean_y for value in sy]
    sxx, syy = math.fsum(d * d for d in dx), math.fsum(d * d for d in dy)
    if sxx == 0 or syy == 0:
        return 0.0
    value = math.fsum(a * b for a, b in zip(dx, dy, strict=True)) / math.sqrt(sxx * syy)
    return min(1.0, max(-1.0, value))


# ---------------------------------------------------------------------------
# One output


def compare_arrays(
    reference: LassiArray,
    candidate: LassiArray,
    *,
    metric: str,
    threshold: float,
    sides: tuple[str, str] = SIDES,
) -> OutputStats:
    """Return the OutputStats of `candidate` against `reference` under `metric` and `threshold` (module rules).

    The stats carry the reference's array name. `sides` names the two runs
    in the notes. Raises ValueError for a metric outside METRICS or a
    threshold lassi.core.tolerance refuses.
    """
    problem = metric_problem(metric) or threshold_problem(metric, threshold)
    if problem:
        raise ValueError(problem)
    return _compared(reference, candidate, metric, threshold, sides)


def _compared(
    reference: LassiArray, candidate: LassiArray, metric: str, threshold: float | None, sides: tuple[str, str]
) -> OutputStats:
    """Return the OutputStats of one output; a None threshold fails a float output with a note."""
    name = reference.name
    for part in ("dtype", "shape"):
        ours, theirs = getattr(reference, part), getattr(candidate, part)
        if ours != theirs:
            note = f"{part} mismatch: {_shown(ours)} in the {sides[0]}, {_shown(theirs)} in the {sides[1]}"
            return _not_compared(name, note)
    if reference.dtype in _INT_VALUE:
        return _integer_stats(reference, candidate, sides)
    return _float_stats(reference, candidate, metric, threshold, sides)


def _shown(value: object) -> str:
    """Return a dtype as its name and a shape as a tuple, for a note."""
    return str(tuple(value)) if isinstance(value, tuple) else str(value)


def _not_compared(name: str, note: str) -> OutputStats:
    """Return the failed OutputStats of an output whose statistics were not computed."""
    return OutputStats(name=name, pcc=None, max_abs=None, max_ulp=None, passed=False, note=note)


def _integer_stats(reference: LassiArray, candidate: LassiArray, sides: tuple[str, str]) -> OutputStats:
    """Return the OutputStats of an integer output: it passes only on exact equality."""
    xs, ys = _values(reference), _values(candidate)
    differ = [index for index, (x, y) in enumerate(zip(xs, ys, strict=True)) if x != y]
    note = None
    if differ:
        first = differ[0]
        note = (
            f"{len(differ)} element(s) differ and an integer output must match exactly; the first, element {first} "
            f"in C order, is {xs[first]} in the {sides[0]} and {ys[first]} in the {sides[1]}"
        )
    max_abs = float(max((abs(x - y) for x, y in zip(xs, ys, strict=True)), default=0))
    pcc = pearson([float(x) for x in xs], [float(y) for y in ys])
    return OutputStats(name=reference.name, pcc=pcc, max_abs=max_abs, max_ulp=None, passed=not differ, note=note)


def _float_stats(
    reference: LassiArray, candidate: LassiArray, metric: str, threshold: float | None, sides: tuple[str, str]
) -> OutputStats:
    """Return the OutputStats of a float output under the NaN, infinity, and -0 rules of the module docstring."""
    xs, ys = _values(reference), _values(candidate)
    x_keys, y_keys = _keys(reference), _keys(candidate)
    pairs_x: list[float] = []
    pairs_y: list[float] = []
    max_ulp, unmatched = 0, []
    for index, (x, y) in enumerate(zip(xs, ys, strict=True)):
        if math.isfinite(x) and math.isfinite(y):
            pairs_x.append(x)
            pairs_y.append(y)
            max_ulp = max(max_ulp, abs(x_keys[index] - y_keys[index]))
        elif not ((math.isnan(x) and math.isnan(y)) or (math.isinf(x) and x == y)):
            unmatched.append(index)
    max_abs: float | None = max((abs(x - y) for x, y in zip(pairs_x, pairs_y, strict=True)), default=0.0)
    notes = []
    if unmatched:
        notes.append(_special_note(unmatched, xs, ys, sides))
    if max_abs is not None and not math.isfinite(max_abs):
        max_abs = None
        notes.append("the largest difference overflows a double, so max_abs is not recorded")
    pcc = pearson(pairs_x, pairs_y)
    value = {"pcc": pcc, "max_abs": max_abs, "ulp": max_ulp}[metric]
    if not notes:
        notes += _threshold_notes(reference.name, metric, value, threshold)
    note = "; ".join(notes) or None
    return OutputStats(name=reference.name, pcc=pcc, max_abs=max_abs, max_ulp=max_ulp, passed=not notes, note=note)


def _special_note(unmatched: Sequence[int], xs: Sequence[float], ys: Sequence[float], sides: tuple[str, str]) -> str:
    """Return the note of an output whose NaN or infinity elements are not matched on the other side."""
    first = unmatched[0]
    x, y = xs[first], ys[first]
    kind = "NaN" if math.isnan(x) or math.isnan(y) else "infinity"
    note = f"unmatched {kind} at element {first} in C order: {x!r} in the {sides[0]}, {y!r} in the {sides[1]}"
    if len(unmatched) > 1:
        note += f" ({len(unmatched)} elements hold an unmatched NaN or infinity)"
    return note


def _threshold_notes(name: str, metric: str, value: float | None, threshold: float | None) -> list[str]:
    """Return the note of a float output that misses its threshold or has none, else no note."""
    if threshold is None:
        return [f"no {metric} threshold: the references' agreement records no {metric} value for output {name!r}"]
    if value is None or within(metric, value, threshold):
        return []
    where = "below" if metric == "pcc" else "above"
    return [f"{STAT_FIELDS[metric]} {value!r} is {where} the {metric} threshold {threshold!r}"]


# ---------------------------------------------------------------------------
# Output files


def _read_side(
    files: Mapping[str, Path], side: str
) -> tuple[dict[str, LassiArray], dict[str, str], list[tuple[str, str]]]:
    """Read one run's output files: arrays by name, a note per repeated name, and (file, note) per unreadable file.

    Files are read in sorted path order; an array name a later file repeats
    keeps the first file's array and gets a note naming both files.
    """
    arrays: dict[str, LassiArray] = {}
    owners: dict[str, str] = {}
    repeated: dict[str, str] = {}
    unreadable: list[tuple[str, str]] = []
    for file in sorted(files):
        path = Path(files[file])
        try:
            array = read_array(path)
        except (OSError, ValueError) as error:
            reason = str(error).removeprefix(f"{path}: ")
            unreadable.append((file, f"the {side}'s output file {file!r} is not a readable lassi_io file: {reason}"))
            continue
        name = array.name
        if name in arrays:
            repeated[name] = f"the {side} wrote the array {name!r} in two files, {owners[name]!r} and {file!r}"
            continue
        arrays[name], owners[name] = array, file
    return arrays, repeated, unreadable


def _reference_arrays(files: Mapping[str, Path], side: str) -> dict[str, LassiArray]:
    """Return the reference run's arrays by name; ValueError when its files cannot serve as a reference."""
    arrays, repeated, unreadable = _read_side(files, side)
    problems = [*(note for _, note in unreadable), *repeated.values()]
    if problems:
        raise ValueError(problems[0])
    if not arrays:
        raise ValueError(f"the {side} wrote no output file, so there is nothing to compare against")
    return arrays


@register("Oracle", "binary_io")
class BinaryIOOracle:
    """Compares a run's lassi_io output files with the reference's, output by output (see the module rules).

    Built from the recipe section `oracle: {kind: binary_io, metric: ...,
    threshold: ...}` as BinaryIOOracle(metric=..., threshold=...).
    """

    name = "binary_io"
    capabilities = frozenset({ALIGNS_OUTPUT_FILES})
    config_keys = frozenset({"metric", "threshold"})

    def __init__(self, *, metric: str | None = None, threshold: float | str | None = None) -> None:
        """Keep the recipe's metric and threshold, both required choices; ValueError names a bad or missing one."""
        if metric is None:
            choices = ", ".join(METRICS)
            raise ValueError(f"oracle.metric is a required choice with no default: set it to one of {choices}")
        problem = metric_problem(metric)
        if problem:
            raise ValueError(f"oracle.metric: {problem}")
        if threshold is None:
            raise ValueError(f"oracle.threshold is a required choice with no default: set a number or {FROM_BASELINE}")
        if not (isinstance(threshold, str) and threshold == FROM_BASELINE):
            problem = threshold_problem(metric, threshold)
            if problem:
                raise ValueError(f"oracle.threshold: {problem}, or {FROM_BASELINE}")
        self.metric = metric
        self.threshold = threshold
        self._agreement: dict[str, float | int | None] | None = None

    @property
    def agreement_setting(self) -> str | None:
        """Return "oracle.threshold from_baseline" when thresholds come from the references' agreement, else None."""
        return f"oracle.threshold {FROM_BASELINE}" if self.threshold == FROM_BASELINE else None

    def with_baseline(self, agreement: Sequence[OutputStats] | None) -> BinaryIOOracle:
        """Return a copy whose threshold per output is that output's metric value in `agreement`.

        An oracle with a number for its threshold returns an unchanged copy.
        With from_baseline, an agreement of None (not recorded) raises
        ValueError naming from_baseline.
        """
        bound = BinaryIOOracle(metric=self.metric, threshold=self.threshold)
        if self.threshold != FROM_BASELINE:
            return bound
        if agreement is None:
            raise ValueError(
                f"oracle.threshold {FROM_BASELINE} reads each output's threshold from the references' agreement, "
                "which the baseline records only with fixes.baseline_both on for an item that declares a tolerance; "
                "none was recorded"
            )
        bound._agreement = {entry.name: getattr(entry, STAT_FIELDS[self.metric]) for entry in agreement}
        return bound

    def with_tolerance(self, tolerance: Tolerance) -> BinaryIOOracle:
        """Return an oracle that judges every output against `tolerance`, its metric and threshold."""
        return BinaryIOOracle(metric=tolerance.metric, threshold=tolerance.threshold)

    def describe(self) -> str:
        """Return a one-line description: the metric and the threshold each output must meet."""
        bound = "at least" if self.metric == "pcc" else "at most"
        if self.threshold == FROM_BASELINE:
            return f"{self.name}: {self.metric} {bound} the references' agreement, per output ({FROM_BASELINE})"
        return f"{self.name}: {self.metric} {bound} {self.threshold!r}, per output"

    def reference_problem(self, files: Mapping[str, Path], side: str = SIDES[0]) -> str | None:
        """Return why a reference run's output files cannot be compared against, or None when they can.

        Every file must be a readable lassi_io file, no two may hold one
        array name, and there must be at least one. `side` names the run in
        the message.
        """
        try:
            _reference_arrays(files, side)
        except ValueError as error:
            return str(error)
        return None

    def compare(
        self,
        reference_files: Mapping[str, Path],
        candidate_files: Mapping[str, Path],
        *,
        sides: tuple[str, str] = SIDES,
    ) -> list[OutputStats]:
        """Return one OutputStats per output found on either side: the reference's names, then extra and unreadable.

        Both mappings hold relative path -> file, as RunResult.output_files
        does. Raises ValueError naming from_baseline for an unbound
        from_baseline oracle, and ValueError when the reference files cannot
        serve (reference_problem); a candidate problem only fails outputs.
        """
        if self.threshold == FROM_BASELINE and self._agreement is None:
            raise ValueError(
                f"oracle.threshold {FROM_BASELINE} needs the references' agreement: bind it with with_baseline first"
            )
        reference = _reference_arrays(reference_files, sides[0])
        candidate, repeated, unreadable = _read_side(candidate_files, sides[1])
        stats = []
        for name in sorted(reference):
            if name in repeated:
                stats.append(_not_compared(name, repeated[name]))
            elif name not in candidate:
                stats.append(_not_compared(name, f"missing output: the {sides[1]} wrote no array named {name!r}"))
            else:
                stats.append(_compared(reference[name], candidate[name], self.metric, self._threshold(name), sides))
        for name in sorted(set(candidate) - set(reference)):
            note = repeated.get(name, f"extra output: the {sides[0]} wrote no array named {name!r}")
            stats.append(_not_compared(name, note))
        return stats + [_not_compared(file, note) for file, note in unreadable]

    def alignment(
        self,
        reference_files: Mapping[str, Path],
        candidate_files: Mapping[str, Path],
        *,
        sides: tuple[str, str] = SIDES,
    ) -> Alignment:
        """Return Alignment(per_input=[v], mean=v, outputs=<the stats>): v is 1.0 when every output passes, else 0.0."""
        stats = self.compare(reference_files, candidate_files, sides=sides)
        value = 1.0 if all(entry.passed for entry in stats) else 0.0
        return Alignment(per_input=[value], mean=value, outputs=stats)

    def align(self, reference: RunResult, candidate: RunResult) -> float:
        """Return the candidate's per-input value against the reference, from the two runs' output files."""
        mean = self.alignment(reference.output_files, candidate.output_files).mean
        return 0.0 if mean is None else mean

    def _threshold(self, name: str) -> float | None:
        """Return the threshold of output `name`: the recipe's number, or its value in the bound agreement."""
        if self._agreement is None:
            return float(self.threshold)
        return self._agreement.get(name)

