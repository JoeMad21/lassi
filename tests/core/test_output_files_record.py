"""Tests for output files in the Result Record and the binary store beside the text store (task P4.4).

Bible: Result Record (RunInfo, Alignment, Trial.reference_run, end_reason,
Storage), Oracles (binary_io row). Plan: plans/p4-ttsim.md, P4.4 and the
planning decision "binary_io (P4.4)".

The contract these tests fix:

- lassi.core.store.BlobStore(root) keeps bytes once by their sha256 at
  `<root>/blobs/<sha[:2]>/<sha>`, beside the text store's `<root>/texts/`.
  put(data) returns the sha256 hex digest and writes the file unless it is
  there; get(sha) returns the bytes, raises KeyError for a hash it does not
  hold and ValueError for a malformed hash or for stored bytes that no
  longer match their hash; `sha in store` says whether it holds the file.
  Stages keep a run's output files in BlobStore(<the text store's root>),
  that is `<run>/blobs/`.
- RunInfo gains `outputs: dict[str, str] | None = None`: each output file
  the run wrote (its relative path, as RunResult.output_files names it) ->
  the sha256 of its bytes in the binary store. None means not recorded; a
  value that is not 64 lowercase hex characters raises ValueError. The
  existing outputs_ref field is left as it is.
- lassi.core.record.OutputStats records one output's statistics: name, pcc,
  max_abs, max_ulp, passed, note (any other field has a default).
  Alignment gains `outputs: list[OutputStats] | None = None` (the oracle's
  statistics per output; None for an oracle that gives none, as stdout_mask),
  and Trial gains `reference_agreement: list[OutputStats] | None = None`
  (the source reference's agreement with the target reference per output,
  measured by the baseline; None when not measured). All round-trip through
  JSON.
- END_REASONS gains `baseline-disagree`: the references of a pair disagree
  beyond the item's declared tolerance, which ends the trial at the
  baseline.
- A trial.json written before these fields existed loads unchanged, with
  each of them None.

Every value here is SYNTHETIC and written for these tests; none is a
measurement.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from lassi.core import record as record_module
from lassi.core import store as store_module
from lassi.core.interfaces import Sampling
from lassi.core.record import (
    Alignment,
    Attempt,
    BenchItem,
    EndReason,
    Final,
    ModelInfo,
    Provenance,
    RunInfo,
    Trial,
    from_json,
    make_trial_id,
    to_json,
)
from lassi.core.store import TextStore, read_trial, trial_dir, write_trial

SAMPLING = Sampling(temperature=0.2, top_p=0.9, max_tokens=4096)
# A commit id for synthetic provenance; not a commit of this repository.
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
SUITE = "binio-fixture"
TRIAL_ID = make_trial_id("binio-record", "scripted-fixture", SUITE, "cpu-tt", "vadd", 1)
OUTPUT_BYTES = b"SYNTHETIC output bytes: not a real lassi_io file\x00\x01\x02\xff"
OTHER_BYTES = b"SYNTHETIC other output bytes\x10\x20"
DISAGREE = "baseline-disagree"


# ---------------------------------------------------------------------------
# Names this task adds, looked up so a missing one fails its test with a clear message


def blob_store(root: Path) -> Any:
    """Return lassi.core.store.BlobStore(root); fail the test clearly while the class is missing."""
    cls = getattr(store_module, "BlobStore", None)
    if cls is None:
        pytest.fail("lassi.core.store has no BlobStore; task P4.4 adds the binary store beside the text store")
    return cls(root)


def require_field(cls: type, name: str) -> None:
    """Fail the test clearly while the record class `cls` has no field `name`."""
    if name not in {spec.name for spec in dataclasses.fields(cls)}:
        pytest.fail(f"{cls.__name__} has no {name} field; task P4.4 adds it to the Result Record")


def output_stats(**fields: Any) -> Any:
    """Return lassi.core.record.OutputStats(**fields); fail the test clearly while the class is missing."""
    cls = getattr(record_module, "OutputStats", None)
    if cls is None:
        pytest.fail("lassi.core.record has no OutputStats; task P4.4 adds the per-output statistics record")
    return cls(**fields)


def sha(data: bytes) -> str:
    """Return the sha256 hex digest of `data`."""
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# A SYNTHETIC trial that sets every new field


def run_info(store: TextStore, outputs: dict[str, str]) -> RunInfo:
    """Return a SYNTHETIC clean RunInfo that recorded `outputs`; the wall time is a fixture value."""
    require_field(RunInfo, "outputs")
    return RunInfo(
        exit_code=0,
        hang=False,
        wall_s=1.25,
        stdout_ref=store.put("SYNTHETIC stdout\n"),
        outputs=outputs,
        stdout_truncated=False,
        stderr_truncated=False,
        workdir_incomplete=False,
    )


def full_trial(store: TextStore) -> Trial:
    """Return a SYNTHETIC trial whose reference run, agreement, attempt run, and alignment set every new field."""
    require_field(Alignment, "outputs")
    require_field(Trial, "reference_agreement")
    agreement = [output_stats(name="c", pcc=0.96875, max_abs=0.001953125, max_ulp=17, passed=True, note=None)]
    stats = [
        output_stats(name="c", pcc=0.75, max_abs=0.125, max_ulp=4096, passed=False, note="SYNTHETIC: past it"),
        output_stats(name="d", pcc=None, max_abs=None, max_ulp=None, passed=False, note="SYNTHETIC: missing"),
    ]
    attempt = Attempt(
        index=0,
        prompt_ref=store.put("SYNTHETIC prompt\n"),
        response_text="SYNTHETIC reply",
        files={"host.cpp": "int main() { return 0; }\n"},
        stage_reached="S5",
        run=run_info(store, {"c.lassiio": sha(OTHER_BYTES)}),
        alignment=Alignment(per_input=[0.0], mean=0.0, outputs=stats),
    )
    return Trial(
        trial_id=TRIAL_ID,
        recipe_hash="0123456789abcdef" * 4,
        provenance=Provenance(
            commit=FAKE_COMMIT, dirty=False, device="SYNTHETIC device", sdk=None, date="2026-09-25T00:00:00+00:00"
        ),
        bench_item=BenchItem(suite=SUITE, item="vadd", split="eval", direction="cpu-tt"),
        model=ModelInfo(backend="scripted", id="scripted-fixture", sampling=SAMPLING),
        reference_run=run_info(store, {"c.lassiio": sha(OUTPUT_BYTES)}),
        reference_agreement=agreement,
        requests=[],
        attempts=[attempt],
        final=Final(stage_reached="S5", alignment=0.0, corrections=0),
    )


# ---------------------------------------------------------------------------
# The binary store


def test_the_binary_store_keeps_bytes_once_by_sha256_beside_the_text_store(tmp_path: Path) -> None:
    blobs = blob_store(tmp_path)
    digest = blobs.put(OUTPUT_BYTES)
    assert digest == sha(OUTPUT_BYTES)
    path = tmp_path / "blobs" / digest[:2] / digest
    assert path.read_bytes() == OUTPUT_BYTES
    assert blobs.put(OUTPUT_BYTES) == digest, "a second put of the same bytes names the same file"
    assert [entry for entry in (tmp_path / "blobs").rglob("*") if entry.is_file()] == [path]
    assert not (tmp_path / "texts").exists(), "bytes never enter the text store"
    empty = blobs.put(b"")
    assert empty == sha(b"") and blobs.get(empty) == b""


def test_the_binary_store_returns_its_bytes_and_says_what_it_holds(tmp_path: Path) -> None:
    blobs = blob_store(tmp_path)
    digest = blobs.put(OUTPUT_BYTES)
    assert blobs.get(digest) == OUTPUT_BYTES
    assert digest in blobs
    assert sha(OTHER_BYTES) not in blobs
    with pytest.raises(KeyError):
        blobs.get(sha(OTHER_BYTES))
    with pytest.raises(ValueError):
        blobs.get("not-a-sha256")


def test_the_binary_store_refuses_bytes_that_no_longer_match_their_hash(tmp_path: Path) -> None:
    blobs = blob_store(tmp_path)
    digest = blobs.put(OUTPUT_BYTES)
    (tmp_path / "blobs" / digest[:2] / digest).write_bytes(OTHER_BYTES)
    with pytest.raises(ValueError, match=digest):
        blobs.get(digest)


# ---------------------------------------------------------------------------
# The record


def test_run_info_records_output_files_by_hash_and_defaults_to_not_recorded() -> None:
    require_field(RunInfo, "outputs")
    assert RunInfo().outputs is None
    info = RunInfo(exit_code=0, hang=False, outputs={"c.lassiio": sha(OUTPUT_BYTES), "sub/d.lassiio": sha(b"")})
    assert info.outputs == {"c.lassiio": sha(OUTPUT_BYTES), "sub/d.lassiio": sha(b"")}
    assert RunInfo(outputs={}).outputs == {}, "a run that wrote no file records an empty mapping"


@pytest.mark.parametrize("digest", ["not-a-sha256", "ABCDEF" + "0" * 58, "0" * 63], ids=["word", "upper", "short"])
def test_run_info_refuses_an_output_hash_that_is_not_a_sha256(digest: str) -> None:
    require_field(RunInfo, "outputs")
    with pytest.raises(ValueError, match="outputs"):
        RunInfo(outputs={"c.lassiio": digest})


def test_the_new_fields_default_to_not_measured(tmp_path: Path) -> None:
    require_field(Alignment, "outputs")
    require_field(Trial, "reference_agreement")
    assert Alignment().outputs is None
    trial = full_trial(TextStore(tmp_path / "store"))
    bare = Trial(
        trial_id=trial.trial_id,
        recipe_hash=trial.recipe_hash,
        provenance=trial.provenance,
        bench_item=trial.bench_item,
        model=trial.model,
    )
    assert bare.reference_agreement is None


def test_output_statistics_and_output_hashes_round_trip_through_json(tmp_path: Path) -> None:
    trial = full_trial(TextStore(tmp_path / "store"))
    again = from_json(Trial, to_json(trial))
    assert again == trial
    assert again.reference_run.outputs == {"c.lassiio": sha(OUTPUT_BYTES)}
    assert [entry.name for entry in again.reference_agreement] == ["c"]
    assert [entry.name for entry in again.attempts[0].alignment.outputs] == ["c", "d"]
    assert isinstance(again.attempts[0].alignment.outputs[0], record_module.OutputStats)


def test_baseline_disagree_is_a_fixed_end_code() -> None:
    assert DISAGREE in record_module.END_REASONS
    reason = EndReason(code=DISAGREE, message="SYNTHETIC: the references disagree on c past the item's tolerance")
    assert reason.code == DISAGREE


def test_an_older_trial_json_without_the_new_fields_loads_unchanged_with_them_null(tmp_path: Path) -> None:
    store = TextStore(tmp_path / "run")
    trial = full_trial(store)
    out = write_trial(trial, tmp_path / "run", store)
    json_path = out / "trial.json"
    data = json.loads(json_path.read_text(encoding="ascii"))
    del data["reference_agreement"]
    del data["reference_run"]["outputs"]
    for attempt in data["attempts"]:
        del attempt["run"]["outputs"]
        del attempt["alignment"]["outputs"]
    json_path.write_bytes((json.dumps(data, indent=2) + "\n").encode("ascii"))
    loaded = read_trial(trial_dir(tmp_path / "run", TRIAL_ID), store)
    older = dataclasses.replace(
        trial,
        reference_agreement=None,
        reference_run=dataclasses.replace(trial.reference_run, outputs=None),
        attempts=[
            dataclasses.replace(
                attempt,
                run=dataclasses.replace(attempt.run, outputs=None),
                alignment=dataclasses.replace(attempt.alignment, outputs=None),
            )
            for attempt in trial.attempts
        ],
    )
    assert loaded == older
