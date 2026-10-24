"""Tests for the prepare step, which fills $HF_HOME/hub for a train recipe's base (2026-10-23; lassi.train.prepare).

Bible: Training Module (Compute), Agent Rules 7, 10, and 17; the owner's
direction of 2026-10-23 (downloads on alpha01 happen only in the owner's
window, as a separate, explicit command).

The contract these tests fix:

- plan(base) names what a base needs from the Hub: a Hub base's repo at its
  revision with MODEL_PATTERNS (the config, the tokenizer, and the
  safetensors weights), a scratch base's tokenizer repo at its pinned
  revision with TOKENIZER_PATTERNS, and nothing for a fixture.
- prepare(recipe_path, download=...) loads the train recipe, refuses an
  unset or empty HF_HOME (Agent Rule 7), calls the download function once
  with repo_id, revision, cache_dir $HF_HOME/hub, and allow_patterns, and
  then requires the snapshot check of lassi.train.bases to pass, so a
  download that left no snapshot at the pinned revision is an error. A
  dry run calls nothing and returns the plan. The default download is
  huggingface_hub's snapshot_download, imported only when it runs.
- tools/prepare_train.py runs it from the command line, prints the plan or
  the snapshot directory, and exits 0, or 2 with a message on a refusal.

The download is a fake here: nothing reaches a network. No value in this
module is a measurement.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from train_fakes import REPO, write_recipe
from trl_smoke import smoke_recipe, train_module

import lassi.train  # noqa: F401  (registers the trl trainer)

HUB_REPO = "SYNTHETIC-org/tiny-model"
REVISION = "0123456789abcdef0123456789abcdef01234567"
TOOL = REPO / "tools" / "prepare_train.py"


def prepare() -> Any:
    """Import lassi.train.prepare."""
    return train_module("prepare")


class FakeDownload:
    """Records each call; writes a snapshot at the requested revision unless told to leave none."""

    def __init__(self, write: bool = True) -> None:
        """Start with no calls; `write` False leaves the cache empty."""
        self.calls: list[dict[str, Any]] = []
        self.write = write

    def __call__(self, **kwargs: Any) -> str:
        """Record the call and, when writing, place a SYNTHETIC config and tokenizer at the snapshot."""
        self.calls.append(kwargs)
        owner, name = kwargs["repo_id"].split("/")
        directory = Path(kwargs["cache_dir"]) / f"models--{owner}--{name}" / "snapshots" / kwargs["revision"]
        if self.write:
            directory.mkdir(parents=True)
            (directory / "config.json").write_bytes(b"{}\n")
            (directory / "tokenizer.json").write_bytes(b"{}\n")
        return str(directory)


def hub_recipe(tmp_path: Path) -> Path:
    """Write the sft smoke recipe on the SYNTHETIC Hub base; return its path."""
    return write_recipe(tmp_path / "recipes", smoke_recipe("sft", "lora", base_model=f"{HUB_REPO}@{REVISION}"))


def test_plan_names_the_repo_revision_and_patterns_of_each_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = prepare()
    bases = train_module("bases")
    hub = module.plan(bases.parse_base(f"{HUB_REPO}@{REVISION}"))
    assert hub == (HUB_REPO, REVISION, module.MODEL_PATTERNS)
    assert "*.safetensors" in module.MODEL_PATTERNS and "*.json" in module.MODEL_PATTERNS
    assert module.plan(bases.parse_base("fixture:tiny-causal-lm")) is None
    monkeypatch.setattr(bases, "SCRATCH_DIR", tmp_path / "scratch")
    directory = tmp_path / "scratch" / "tiny"
    directory.mkdir(parents=True)
    (directory / "config.json").write_bytes(b"{}\n")
    pin = {"repo": "SYNTHETIC-org/tok", "revision": REVISION}
    (directory / "tokenizer-pin.json").write_bytes(json.dumps(pin).encode("ascii"))
    assert module.plan(bases.parse_base("scratch:tiny")) == ("SYNTHETIC-org/tok", REVISION, module.TOKENIZER_PATTERNS)
    assert not any(pattern.endswith(".safetensors") for pattern in module.TOKENIZER_PATTERNS)


def test_prepare_downloads_once_into_hf_home_then_checks_the_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    download = FakeDownload()
    found = prepare().prepare(hub_recipe(tmp_path), download=download)
    assert download.calls == [{"repo_id": HUB_REPO, "revision": REVISION, "cache_dir": str(tmp_path / "hf" / "hub"),
                               "allow_patterns": list(prepare().MODEL_PATTERNS)}]
    assert found == tmp_path / "hf" / "hub" / "models--SYNTHETIC-org--tiny-model" / "snapshots" / REVISION


def test_a_download_that_leaves_no_snapshot_is_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    with pytest.raises(ValueError, match=REVISION):
        prepare().prepare(hub_recipe(tmp_path), download=FakeDownload(write=False))


@pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
def test_prepare_refuses_an_unset_hf_home(value: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if value is None:
        monkeypatch.delenv("HF_HOME", raising=False)
    else:
        monkeypatch.setenv("HF_HOME", value)
    download = FakeDownload()
    with pytest.raises(ValueError, match="HF_HOME"):
        prepare().prepare(hub_recipe(tmp_path), download=download)
    assert download.calls == []


def test_a_dry_run_and_a_fixture_download_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    download = FakeDownload()
    assert prepare().prepare(hub_recipe(tmp_path), download=download, dry_run=True) is None
    fixture = write_recipe(tmp_path / "recipes", smoke_recipe("sft", "full"), name="fixture")
    assert prepare().prepare(fixture, download=download) is None
    assert download.calls == []


def test_the_module_imports_no_hub_library_until_a_download_runs() -> None:
    source = Path(prepare().__file__).read_bytes().decode("ascii")
    head = source.split("\ndef ", 1)[0]
    assert "huggingface_hub" not in head.split('"""', 2)[-1], "huggingface_hub is imported inside a function only"


def tool() -> Any:
    """Import tools/prepare_train.py as the other tool tests import theirs."""
    spec = importlib.util.spec_from_file_location("prepare_train_under_test", TOOL)
    assert spec is not None and spec.loader is not None, f"missing {TOOL}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_tool_prints_the_plan_on_a_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                               capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert tool().main([str(hub_recipe(tmp_path)), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert HUB_REPO in out and REVISION in out and "dry run" in out


def test_the_tool_exits_2_on_a_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                       capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("HF_HOME", raising=False)
    assert tool().main([str(hub_recipe(tmp_path))]) == 2
    assert "HF_HOME" in capsys.readouterr().err
