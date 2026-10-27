"""Tests for the ttxla Trainer: sft on a Tenstorrent card through the tt-xla wheel, with fakes (task P17.38).

Bible: Training Module (Weight Modes, Compute, Safeguards), Component
Interfaces (Trainer), Toolchain Pins, Agent Rules 1, 5, 6, 7, 9, and 10;
plans/spikes/tt-training.md, Finding 3, steps 2 and 3 (the design), and its
test list; OQ-100 (a) with (s2); OQ-101; OQ-102.

The contract these tests fix (lassi.train.ttxla_trainer):

- Registered as Trainer "ttxla": capability takes_device; method sft alone
  (no reward, so RL never touches silicon); weight modes full, lora, and
  prompt_tuning; data sources synthetic and corpus; packages, the tt-xla
  venv's distributions, whose versions the layer reads from the venv's
  site-packages (package_paths(), a staticmethod), never LASSI's own
  environment; config keys steps, seed, batch_size, learning_rate,
  max_length, and timeout_s, each required with no default, and
  gradient_checkpointing, chat_format, and virtual_tokens.
- framework() reads the install under $LASSI_TOOLCHAINS/<PREFIX_NAME of
  toolchains/tt-xla.pin>: its record's first line names tt-xla, the wheel's
  version, and its sha256, and the venv's pjrt-plugin-tt metadata, read by
  path, has the pinned version and a Summary naming the pinned tt-xla,
  tt-mlir, and tt-metal commits; a wheel naming another tt-metal commit is
  refused. It returns FrameworkBuild(pjrt-plugin-tt, <version>, None, None).
- Building refuses, naming the key: a device that is not one Tenstorrent
  card, each setting out of range, an unset HF_HOME, and an unset or
  relative LASSI_SCRATCH. It imports no framework and reads no file.
- check() refuses what it does not carry out: a base outside the three
  kinds or missing files or snapshot, a scratch base with an adapter, lora
  without lora.r or with targets other than all-linear, virtual_tokens
  unless the weights are prompt_tuning (then required), and a record whose
  keys are not prompt and completion, or whose values are not strings.
- train() writes the worker's job file, environment, and command
  deterministically under output/worker/, runs the worker once with the
  venv's interpreter (`python -I`), never retried, and on success writes
  output/checkpoint-<steps>/'s records through lassi.train.checkpoint
  (non-finite losses written as null and named). A worker refusal (status
  3) raises a ValueError, and any other status from 1 to 127 a
  RuntimeError, both silicon.CleanEnding (the card left in a known state);
  a hang past trainer.timeout_s, a death by signal, or a status outside 0
  to 127 raises silicon.CardEnding naming it, after the worker is killed.

The worker here is a fake (ttxla_fakes.worker_script) run with the test
interpreter; the venv, the card, and the host are SYNTHETIC. No value here
is a measurement.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import ttxla_fakes as fakes
from silicon_fakes import LOCK, MARKER, Rig
from train_fakes import REPO, Setup, provenance

from lassi.core.devices import DeviceUnavailable, FrameworkBuild
from lassi.core.interfaces import TrainData, TrainJob
from lassi.core.record import DeviceRecord
from lassi.core.registry import DEFAULT_REGISTRY, Registry
from lassi.core.runner import RunError

TRAINER = "ttxla"
FIXTURE_BASE = "fixture:tiny-causal-lm"
CONFIG: dict[str, Any] = {
    "device": {"kind": "tenstorrent", "indices": [0]},
    "steps": 3,
    "seed": 7,
    "batch_size": 1,
    "learning_rate": 0.0002,
    "max_length": 64,
    "timeout_s": 600,
    "network": "none",
}
CONFIG_KEYS = frozenset({
    "steps", "seed", "batch_size", "learning_rate", "max_length", "timeout_s", "gradient_checkpointing",
    "chat_format", "virtual_tokens", "network",
})
RECORDS = (
    {"prompt": "SYNTHETIC sft prompt 1: say abc.", "completion": " abc"},
    {"prompt": "SYNTHETIC sft prompt 2: say def.", "completion": " def"},
)


def ttxla() -> ModuleType:
    """Import lassi.train.ttxla_trainer, failing the test clearly while it does not exist (task P17.38)."""
    try:
        return importlib.import_module("lassi.train.ttxla_trainer")
    except ModuleNotFoundError as error:
        pytest.fail(f"lassi.train.ttxla_trainer does not exist yet (task P17.38): {error}")


def silicon() -> ModuleType:
    """Import lassi.train.silicon."""
    try:
        return importlib.import_module("lassi.train.silicon")
    except ModuleNotFoundError as error:
        pytest.fail(f"lassi.train.silicon does not exist yet (task P17.38): {error}")


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Set HF_HOME, LASSI_SCRATCH, and LASSI_TOOLCHAINS under the test directory, with the SYNTHETIC venv."""
    paths = {"hf_home": tmp_path / "hf", "scratch": tmp_path, "toolchains": tmp_path / "toolchains"}
    paths["hf_home"].mkdir()
    monkeypatch.setenv("HF_HOME", str(paths["hf_home"]))
    monkeypatch.setenv("LASSI_SCRATCH", str(paths["scratch"]))
    monkeypatch.setenv("LASSI_TOOLCHAINS", str(paths["toolchains"]))
    paths["prefix"] = fakes.write_venv(paths["toolchains"])
    monkeypatch.setattr(ttxla(), "network_namespace_problem", lambda: None)
    return paths


def build(**changes: Any) -> Any:
    """Build the trainer from CONFIG with `changes` (None removes a key)."""
    config = {**CONFIG, **changes}
    return ttxla().TtxlaTrainer(**{key: value for key, value in config.items() if value is not None})


def data(records: tuple[dict[str, Any], ...] = RECORDS) -> TrainData:
    """Return synthetic TrainData holding `records`."""
    return TrainData(source="synthetic", split_hash="0" * 64, records=tuple(records), suite=None, items=())


def recipe(**changes: Any) -> dict[str, Any]:
    """Return a resolved train recipe mapping for the ttxla trainer, with top-level `changes`."""
    base: dict[str, Any] = {
        "base_model": FIXTURE_BASE, "method": "sft", "weights": "full",
        "trainer": {"kind": TRAINER, **CONFIG}, "data": {"synthetic": "sft-smoke.jsonl"},
    }
    base.update(changes)
    return {key: value for key, value in base.items() if value is not None}


# ---------------------------------------------------------------------------
# Registration and declarations


def test_ttxla_is_registered_with_its_declarations() -> None:
    module = ttxla()
    cls = module.TtxlaTrainer
    assert DEFAULT_REGISTRY.get("Trainer", TRAINER).factory is cls
    assert set(cls.capabilities) == {"takes_device"}
    assert tuple(cls.methods) == ("sft",), "no reward method: RL rewards never execute on silicon"
    assert tuple(cls.weight_modes) == ("full", "lora", "prompt_tuning")
    assert tuple(cls.data_sources) == ("synthetic", "corpus")
    assert frozenset(cls.config_keys) == CONFIG_KEYS
    assert {"pjrt-plugin-tt", "torch", "torch-xla", "transformers", "peft"} <= set(cls.packages)
    assert isinstance(vars(cls)["framework"], staticmethod)
    assert isinstance(vars(cls)["package_paths"], staticmethod)


# ---------------------------------------------------------------------------
# The framework build and the pins, read from the venv by path


def test_framework_reads_the_pinned_wheel_from_the_venv(env: dict[str, Path]) -> None:
    pin = fakes.pin()
    assert ttxla().TtxlaTrainer.framework() == FrameworkBuild("pjrt-plugin-tt", pin["VERSION"], None, None)
    (site,) = ttxla().TtxlaTrainer.package_paths()
    assert Path(site) == fakes.site_packages(env["prefix"] / pin["VENV_DIR"], pin["PYTHON_MINOR"])


def test_framework_refuses_a_wheel_naming_another_tt_metal_commit(tmp_path: Path, env: dict[str, Path]) -> None:
    other = tmp_path / "other-toolchains"
    fakes.write_venv(other, summary=fakes.OTHER_SUMMARY)
    os.environ["LASSI_TOOLCHAINS"] = str(other)
    with pytest.raises(ValueError) as refused:
        ttxla().TtxlaTrainer.framework()
    message = str(refused.value)
    assert "tt-metal" in message and "5280a9cfb00998fd49667a29523d03aee905c129" in message, message


def test_framework_refuses_another_wheel_version_and_a_missing_record(tmp_path: Path, env: dict[str, Path]) -> None:
    fakes.write_venv(tmp_path / "v", version="0.9.0.dev20260224001247")
    os.environ["LASSI_TOOLCHAINS"] = str(tmp_path / "v")
    with pytest.raises(ValueError, match="0.9.0.dev20260224001247"):
        ttxla().TtxlaTrainer.framework()
    fakes.write_venv(tmp_path / "r", record=False)
    os.environ["LASSI_TOOLCHAINS"] = str(tmp_path / "r")
    with pytest.raises(ValueError, match="toolchains/tt-xla.sh"):
        ttxla().TtxlaTrainer.framework()


def test_framework_needs_lassi_toolchains(env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LASSI_TOOLCHAINS")
    with pytest.raises(ValueError, match="LASSI_TOOLCHAINS"):
        ttxla().TtxlaTrainer.framework()


# ---------------------------------------------------------------------------
# Building: every setting checked, naming its key; no framework imported


@pytest.mark.parametrize(
    ("changes", "named"),
    [
        pytest.param({"device": {"kind": "cpu"}}, "trainer.device", id="cpu"),
        pytest.param({"device": {"kind": "tenstorrent", "indices": [0, 1]}}, "trainer.device", id="two-cards"),
        pytest.param({"steps": 0}, "trainer.steps", id="steps"),
        pytest.param({"seed": -1}, "trainer.seed", id="seed"),
        pytest.param({"batch_size": True}, "trainer.batch_size", id="batch-bool"),
        pytest.param({"learning_rate": 0}, "trainer.learning_rate", id="rate"),
        pytest.param({"max_length": None}, "trainer.max_length", id="no-max-length"),
        pytest.param({"timeout_s": None}, "trainer.timeout_s", id="no-timeout"),
        pytest.param({"timeout_s": 10}, "trainer.timeout_s", id="short-timeout"),
        pytest.param({"gradient_checkpointing": "yes"}, "trainer.gradient_checkpointing", id="checkpointing"),
        pytest.param({"chat_format": 1}, "trainer.chat_format", id="chat-format"),
        pytest.param({"virtual_tokens": 0}, "trainer.virtual_tokens", id="virtual-tokens"),
        pytest.param({"network": None}, "trainer.network", id="no-network"),
        pytest.param({"network": "host"}, "trainer.network", id="host-network"),
    ],
)
def test_building_refuses_a_bad_setting_naming_it(env: dict[str, Path], changes: dict[str, Any], named: str) -> None:
    with pytest.raises(ValueError) as refused:
        build(**changes)
    assert named in str(refused.value), str(refused.value)


@pytest.mark.parametrize("name", ["HF_HOME", "LASSI_SCRATCH"])
def test_building_needs_hf_home_and_an_absolute_lassi_scratch(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.delenv(name)
    with pytest.raises(ValueError, match=name):
        build()
    monkeypatch.setenv(name, "relative/path")
    if name == "LASSI_SCRATCH":
        with pytest.raises(ValueError, match=name):
            build()


def test_building_imports_no_framework(env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("torch", "torch_xla", "transformers", "peft"):
        monkeypatch.setitem(sys.modules, name, None)
    trainer = build(gradient_checkpointing=True, chat_format=False)
    assert (trainer.card, trainer.max_length, trainer.timeout_s) == (0, 64, 600)


# ---------------------------------------------------------------------------
# check(): what the trainer does not carry out is refused before any directory


@pytest.mark.parametrize(
    ("changes", "named"),
    [
        pytest.param({"base_model": "SYNTHETIC/no-revision"}, "revision", id="hub-without-revision"),
        pytest.param({"base_model": "SYNTHETIC/model@" + "a" * 40}, "prepare", id="hub-not-cached"),
        pytest.param({"base_model": "scratch:qwen3-77m", "weights": "lora", "lora": {"r": 8, "targets": "all-linear"}},
                     "full weights only", id="scratch-adapter"),
        pytest.param({"weights": "lora"}, "lora", id="lora-missing"),
        pytest.param({"weights": "lora", "lora": {"targets": "all-linear"}}, "lora.r", id="lora-without-r"),
        pytest.param({"weights": "lora", "lora": {"r": 8, "targets": "q_proj"}}, "lora.targets", id="targets"),
        pytest.param({"lora": {"r": 8, "targets": "all-linear"}}, "lora", id="lora-on-full"),
        pytest.param({"weights": "prompt_tuning"}, "trainer.virtual_tokens", id="prompt-without-tokens"),
    ],
)
def test_check_refuses_what_it_does_not_carry_out(
    env: dict[str, Path], changes: dict[str, Any], named: str
) -> None:
    with pytest.raises(ValueError) as refused:
        build().check(recipe(**changes), data())
    assert named in str(refused.value), str(refused.value)


def test_check_refuses_virtual_tokens_off_prompt_tuning_and_bad_records(env: dict[str, Path]) -> None:
    with pytest.raises(ValueError, match="trainer.virtual_tokens"):
        build(virtual_tokens=8).check(recipe(), data())
    with pytest.raises(ValueError, match="record 2"):
        build().check(recipe(), data((RECORDS[0], {"prompt": "SYNTHETIC", "target": "x"})))
    with pytest.raises(ValueError, match="record 1"):
        build().check(recipe(), data(({"prompt": "SYNTHETIC", "completion": 3},)))
    build(virtual_tokens=8).check(recipe(weights="prompt_tuning"), data())
    build().check(recipe(weights="lora", lora={"r": 8, "targets": "all-linear"}), data())


# ---------------------------------------------------------------------------
# The worker's job, environment, and command, built deterministically


def job_for(trainer: Any, tmp_path: Path, **changes: Any) -> dict[str, Any]:
    """Return the job the trainer writes for the fixture base, under <tmp>/out/worker."""
    work = tmp_path / "out" / "worker"
    return trainer.worker_job(recipe(**changes), data(), work, tmp_path / "out" / "checkpoint-3")


def test_the_job_names_the_resolved_base_the_settings_and_the_records(env: dict[str, Path], tmp_path: Path) -> None:
    trainer = build(gradient_checkpointing=True)
    job = job_for(trainer, tmp_path, weights="lora", lora={"r": 8, "targets": "all-linear"})
    fixture = REPO / "tests" / "fixtures" / "train" / "tiny-causal-lm"
    assert job["base"] == {"kind": "fixture", "path": str(fixture), "tokenizer": None}
    assert job["backend"] == "xla", "the trainer always runs the worker on the card"
    expected = {"weights": "lora", "lora_r": 8, "virtual_tokens": None, "steps": 3, "seed": 7, "batch_size": 1,
                "learning_rate": 0.0002, "max_length": 64, "gradient_checkpointing": True, "chat_format": False,
                "source": "data.synthetic", "dtype": "bfloat16", "network": "none"}
    assert {key: job[key] for key in expected} == expected
    assert job["records"] == [dict(record) for record in RECORDS]
    work = tmp_path / "out" / "worker"
    assert job["checkpoint"] == str(tmp_path / "out" / "checkpoint-3")
    for key, name in (("steps_file", "steps.jsonl"), ("result_file", "result.json"), ("stage_file", "stage")):
        assert job[key] == str(work / name)
    assert job == job_for(build(gradient_checkpointing=True), tmp_path, weights="lora",
                          lora={"r": 8, "targets": "all-linear"}), "deterministic"


def test_a_hub_base_is_read_from_its_snapshot_and_a_scratch_base_with_its_tokenizer(
    env: dict[str, Path], tmp_path: Path
) -> None:
    revision = "b" * 40
    snapshot = env["hf_home"] / "hub" / "models--SYNTHETIC--model" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_bytes(b"{}")
    job = job_for(build(), tmp_path, base_model=f"SYNTHETIC/model@{revision}")
    assert job["base"] == {"kind": "hub", "path": str(snapshot), "tokenizer": None}
    tokenizer = env["hf_home"] / "hub" / "models--Qwen--Qwen3-0.6B" / "snapshots" / (
        "c1899de289a04d12100db370d81485cdf75e47ca")
    tokenizer.mkdir(parents=True)
    (tokenizer / "tokenizer.json").write_bytes(b"{}")
    scratch = job_for(build(), tmp_path, base_model="scratch:qwen3-77m")
    assert scratch["base"] == {"kind": "scratch", "path": str(REPO / "assets" / "train" / "scratch" / "qwen3-77m"),
                               "tokenizer": str(tokenizer)}


def test_the_worker_environment_is_built_from_nothing(
    env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SYNTHETIC_CALLER_VARIABLE", "must not pass")
    monkeypatch.setenv("TT_METAL_RUNTIME_ROOT", "/SYNTHETIC/must/not/pass")
    work = tmp_path / "out" / "worker"
    found = build().worker_environment(work)
    venv = env["prefix"] / fakes.pin()["VENV_DIR"]
    assert found["TT_VISIBLE_DEVICES"] == "0"
    assert found["PJRT_DEVICE"] == "TT"
    assert found["TT_METAL_CACHE"] == str(env["scratch"] / "ttxla" / "cache")
    assert found["TT_METAL_LOGS_PATH"] == str(work / "logs")
    assert found["TMPDIR"] == str(work / "tmp")
    assert found["HOME"] == str(env["scratch"] / "ttxla" / "home")
    assert found["HF_HOME"] == str(env["hf_home"])
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY",
                 "DO_NOT_TRACK"):
        assert found[name] == "1", name
    assert found["PATH"].split(os.pathsep)[0] == str(venv / "bin")
    assert "SYNTHETIC_CALLER_VARIABLE" not in found
    assert "TT_METAL_RUNTIME_ROOT" not in found, "the wheel finds its own tt-metal; the pinned sfpi is linked there"
    profilers = [name for name in found if "PROFILER" in name or name.startswith("TRACY")]
    assert profilers == [], "the device profiler stays off and Tracy is left as the wheel builds it"
    assert all(isinstance(value, str) and value.isascii() for value in found.values())


def test_the_worker_command_runs_the_venv_python_isolated_in_the_network_namespace() -> None:
    namespace = ["/usr/bin/unshare", "--user", "--map-root-user", "--net", "--"]
    command = ttxla().worker_command(Path("/v/bin/python"), Path("/w/worker.py"), Path("/o/job.json"),
                                     ["ionice", "-c", "2", "-n", "7", "nice", "-n", "10"], namespace)
    assert command == ["ionice", "-c", "2", "-n", "7", "nice", "-n", "10", *namespace, str(Path("/v/bin/python")),
                       "-I", str(Path("/w/worker.py")), str(Path("/o/job.json"))]


def test_the_network_prefix_is_unshare_by_path_with_a_user_and_a_network_namespace() -> None:
    found = ttxla().network_prefix(posix=True, which=lambda name: f"/usr/bin/{name}")
    assert found == ["/usr/bin/unshare", "--user", "--map-root-user", "--net", "--"]
    with pytest.raises(FileNotFoundError, match="unshare"):
        ttxla().network_prefix(posix=True, which=lambda name: None)
    with pytest.raises(FileNotFoundError, match="unshare"):
        ttxla().network_prefix(posix=False)


def test_check_refuses_a_host_where_the_namespace_cannot_be_made(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ttxla(), "network_namespace_problem", lambda: "SYNTHETIC: unshare: write failed")
    with pytest.raises(ValueError) as refused:
        build().check(recipe(), data())
    message = str(refused.value)
    assert "trainer.network" in message and "SYNTHETIC: unshare: write failed" in message, message


# ---------------------------------------------------------------------------
# train(): the worker once, its ending read as the executor reads a program's


def device_record() -> DeviceRecord:
    """Return the SYNTHETIC device record of card 0 a train job carries."""
    return DeviceRecord(
        key="trainer.device", kind="tenstorrent", indices=[0], name="SYNTHETIC n300 card 0: serial S0", count=8,
        memory_bytes=None, driver="SYNTHETIC-2.8.0", runtime=None, framework="pjrt-plugin-tt",
        framework_version="0.9.0.dev20260223001148",
    )


def train_job(tmp_path: Path, **changes: Any) -> TrainJob:
    """Return a TrainJob on the fixture base with an empty out_dir and a provenance holding the copied keys."""
    out_dir = tmp_path / "train" / "output"
    out_dir.mkdir(parents=True)
    manifest = {"train_id": "t1", "trainer": TRAINER, "recipe_hash": "0" * 64, "commit": None, "dirty": None,
                "device_records": [], "driver": None, "framework_pins": {}, "data": {}}
    return TrainJob(recipe=recipe(**changes), recipe_yaml="SYNTHETIC: recipe\n", data=data(),
                    device=device_record(), provenance=manifest, out_dir=out_dir)


@pytest.fixture
def worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Return a function that puts the fake worker of a mode in place of the real one, run with this interpreter."""

    def use(mode: str) -> Path:
        path = fakes.write_worker(tmp_path / "fake-workers", mode)
        monkeypatch.setattr(ttxla(), "WORKER", path)
        monkeypatch.setattr(ttxla(), "venv_python", lambda: Path(sys.executable))
        unshare = fakes.write_unshare(tmp_path / "fake-workers")
        monkeypatch.setattr(ttxla(), "network_prefix", lambda: [
            sys.executable, str(unshare), "--user", "--map-root-user", "--net", "--"])
        return path

    return use


def test_a_complete_worker_leaves_the_checkpoint_with_its_records(
    env: dict[str, Path], tmp_path: Path, worker: Any
) -> None:
    script = worker("complete")
    job = train_job(tmp_path)
    result = build().train(job)
    assert (result.steps, tuple(result.checkpoints)) == (3, ("checkpoint-3",))
    checkpoint = job.out_dir / "checkpoint-3"
    lines = [json.loads(line) for line in (checkpoint / "steps.jsonl").read_text(encoding="ascii").splitlines()]
    assert [line["step"] for line in lines] == [1, 2, 3]
    assert lines[1]["loss"] is None and lines[1]["nonfinite"] == ["loss"], "never bridged (Agent Rule 1)"
    assert math.isclose(lines[2]["loss"], 1 / 3)
    records = json.loads((checkpoint / "checkpoint.json").read_text(encoding="ascii"))
    own = records["trainer_records"]
    assert (own["method"], own["weights"], own["global_step"], own["peft"]) == ("sft", "full", 3, None)
    assert own["base_model"]["name"] == FIXTURE_BASE and own["worker"]["device"] == "SYNTHETIC xla:0"
    assert own["settings"]["max_length"] == 64 and own["settings"]["timeout_s"] == 600
    assert own["settings"]["network"] == "none"
    assert (checkpoint / "recipe.resolved.yaml").read_text(encoding="ascii") == "SYNTHETIC: recipe\n"
    seen = fakes.seen(job.out_dir / "worker")
    assert seen["argv"] == [str(script), str(job.out_dir / "worker" / "job.json")]
    namespace = fakes.unshare_record(tmp_path / "fake-workers")
    assert namespace["flags"] == ["--user", "--map-root-user", "--net"], "the worker runs in its own network namespace"
    assert namespace["command"] == [sys.executable, "-I", str(script), str(job.out_dir / "worker" / "job.json")]
    assert seen["environment"]["TT_VISIBLE_DEVICES"] == "0"
    written = json.loads((job.out_dir / "worker" / "job.json").read_text(encoding="ascii"))
    assert written == build().worker_job(job.recipe, job.data, job.out_dir / "worker", checkpoint)


def test_a_refusing_worker_raises_a_clean_value_error(env: dict[str, Path], tmp_path: Path, worker: Any) -> None:
    worker("refuse")
    with pytest.raises(ValueError) as refused:
        build().train(train_job(tmp_path))
    assert isinstance(refused.value, silicon().CleanEnding)
    assert "record 2" in str(refused.value) and "trainer.max_length" in str(refused.value)


def test_a_failing_worker_raises_a_clean_runtime_error_with_its_log(
    env: dict[str, Path], tmp_path: Path, worker: Any
) -> None:
    worker("error")
    with pytest.raises(RuntimeError) as failed:
        build().train(train_job(tmp_path))
    assert isinstance(failed.value, silicon().CleanEnding)
    message = str(failed.value)
    assert "status 1" in message and "the worker failed" in message and "checked" in message, message


def test_a_crashing_worker_raises_a_card_ending(env: dict[str, Path], tmp_path: Path, worker: Any) -> None:
    worker("crash")
    with pytest.raises(silicon().CardEnding) as ended:
        build().train(train_job(tmp_path))
    assert "139" in ended.value.ending, ended.value.ending
    assert not isinstance(ended.value, silicon().CleanEnding)


def test_a_hanging_worker_is_killed_once_and_raises_a_card_ending(
    env: dict[str, Path], tmp_path: Path, worker: Any
) -> None:
    worker("hang")
    trainer = build()
    trainer.timeout_s = 1
    with pytest.raises(silicon().CardEnding) as ended:
        trainer.train(train_job(tmp_path))
    assert "hang" in ended.value.ending and "trainer.timeout_s" in ended.value.ending, ended.value.ending


# ---------------------------------------------------------------------------
# Through the layer: `lassi train --silicon` with the real trainer, a fake worker, and a SYNTHETIC host


def layer_recipe(setup: Setup) -> Path:
    """Write the fixture-base recipe of the ttxla trainer on card 0 and return its path."""
    return setup.write({"base_model": FIXTURE_BASE, "method": "sft", "weights": "full",
                        "trainer": {"kind": TRAINER, **CONFIG}, "data": {"synthetic": "sft-smoke.jsonl"}})


def ttxla_registry() -> Registry:
    """Return a registry holding the real ttxla trainer alone."""
    registry = Registry()
    registry.register("Trainer", TRAINER, ttxla().TtxlaTrainer)
    return registry


def test_lassi_train_silicon_runs_the_trainer_and_records_the_venv_pins(
    env: dict[str, Path], tmp_path: Path, worker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker("complete")
    setup = Setup(tmp_path / "layer")
    rig = Rig(setup, monkeypatch)
    path = layer_recipe(setup)
    options = setup.options(registry=ttxla_registry(), probes=rig.probes(), silicon=True,
                            silicon_host=rig.host_seams(), synthetic_dir=REPO / "tests" / "fixtures" / "train")
    train_dir = importlib.import_module("lassi.train.run").run_training(path, options)
    record = provenance(train_dir)
    assert record["status"] == "complete" and record["steps"] == 3
    assert record["framework_pins"]["build"]["name"] == "pjrt-plugin-tt"
    packages = record["framework_pins"]["packages"]
    assert packages["pjrt-plugin-tt"] == fakes.pin()["VERSION"] and packages["torch"] == "2.9.0+cpu"
    assert packages["jax"] is None, "a distribution the venv lacks is null, never LASSI's own version"
    assert record["network"] == "none"
    device = record["device_records"][0]
    assert device["kind"] == "tenstorrent" and rig.fakes.serial(0) in device["name"]
    assert set(record["checkpoint_sha256"]) == {"checkpoint-3"}
    assert not (rig.locks / MARKER).exists() and not rig.flock.held(rig.locks / LOCK)


def test_the_layer_refuses_a_ttxla_run_without_the_flag_or_a_card(
    env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class NoCards:
        """A tenstorrent probe on a host without cards."""

        kind = "tenstorrent"

        def probe(self) -> Any:
            """Refuse, as the real probe does on such a host."""
            raise DeviceUnavailable("SYNTHETIC: /dev/tenstorrent does not exist on this host")

    setup = Setup(tmp_path / "layer")
    rig = Rig(setup, monkeypatch)
    path = layer_recipe(setup)
    run = importlib.import_module("lassi.train.run")
    fixtures = REPO / "tests" / "fixtures" / "train"
    with pytest.raises(RunError, match="--silicon"):
        run.run_training(path, setup.options(registry=ttxla_registry(), probes=rig.probes(), synthetic_dir=fixtures))
    with pytest.raises(RunError, match="/dev/tenstorrent"):
        run.run_training(path, setup.options(registry=ttxla_registry(), probes={"tenstorrent": NoCards()},
                                             silicon=True, synthetic_dir=fixtures))
    assert not setup.runs_root.exists() and not rig.locks.exists()
