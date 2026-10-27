"""Tests for `lassi train --silicon`: training on a Tenstorrent card, started by a human (task P17.38).

Bible: Agent Rule 9 as OQ-100 (a) amends it ("silicon runs are evaluation
runs, or training runs of tracked code, that a human starts"; RL rewards
never execute on silicon); OQ-100 (s2): one human start may run a sequence
of runs, each its own process, stopping at the first failure or hang;
OQ-075 (2a) and (3a), the tt_silicon executor's lock and quarantine marker,
which a training run shares; Training Module, Safeguards.

The contract these tests fix (lassi.train.run, lassi.train.silicon,
lassi.cli):

- A train recipe whose trainer.device is of kind tenstorrent is refused
  unless TrainOptions.silicon is True, which only `lassi train <recipe>
  --silicon` sets (no recipe key or variable stands in for it), before any
  directory exists, naming --silicon and Agent Rule 9. On such a device the
  method must be sft (RL rewards never execute on silicon), trainer.network
  must be none, with no default (the training worker runs in its own empty
  network namespace: the training stack's Tracy client listens and
  broadcasts from library load; plans/spikes/tt-perf.md), which
  provenance.json records under network (null for a run whose trainer
  section names none), and $LASSI_SCRATCH must be set, since the lock lives
  there.
- With the flag, once every other refusal has passed and before any
  directory exists, the run takes the one lock of every card,
  $LASSI_SCRATCH/locks/tt-silicon.lock (refused at once while another
  process holds it, never a wait), refuses while any quarantine marker is
  there, and polls the cards until no other account holds them (the
  executor's poll), refusing after its limit. It holds the lock until the
  final provenance.json is written.
- Right before train(job) it writes the quarantine marker
  tt-silicon.quarantine naming the ending "not recorded", the run, and the
  card's serial and PCI address. train() is called once, never retried.
  The marker is removed when train() returns or raises an exception that
  says the card was left in a known state (silicon.CleanEnding); it stays,
  rewritten to name the ending, after silicon.CardEnding (a hang, a kill,
  or a crash the trainer saw) and after any other exception.
- A cpu device needs no flag and takes no lock, with or without it.

Every component is a fake: the train layer's fake Trainer (train_fakes)
with a SYNTHETIC pjrt-plugin-tt build, the real tenstorrent probe on the
tt_silicon tests' SYNTHETIC host tree, and their fake flock, owners, and
clock. No card, node, or process is touched. No value here is a
measurement.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from silicon_fakes import LOCK, MARKER, TT_DEVICE, Rig, silicon
from train_fakes import Log, Setup, fake_trainer, provenance, recipe_data

from lassi import cli
from lassi.core.registry import Registry
from lassi.core.runner import RunError

TT_BUILD = ("pjrt-plugin-tt", "0.0.0.dev-synthetic", None, None)


def trainer_registry(log: Log, behave: Callable[[Any], Any] | None = None, look: Callable[[Any], None] | None = None,
                     methods: frozenset[str] = frozenset({"sft", "dpo"})) -> Any:
    """Return a registry with the fake Trainer on a SYNTHETIC pjrt-plugin-tt build; `behave` replaces its train()."""
    base = fake_trainer(log, framework=TT_BUILD, methods=methods)

    def train(self: Any, job: Any) -> Any:
        if look is not None:
            look(job)
        if behave is None:
            return base.train(self, job)
        log.events.append("train")
        return behave(job)

    registry = Registry()
    registry.register("Trainer", "fake", type("SiliconFake", (base,), {"train": train}))
    return registry


@pytest.fixture
def setup(tmp_path: Path) -> Setup:
    """Return the test's directories and log."""
    return Setup(tmp_path)


@pytest.fixture
def rig(setup: Setup, monkeypatch: pytest.MonkeyPatch) -> Rig:
    """Return the SYNTHETIC host and fakes of a silicon run."""
    return Rig(setup, monkeypatch)


# ---------------------------------------------------------------------------
# The human start


def test_a_tenstorrent_device_without_the_flag_is_refused_before_any_directory(setup: Setup, rig: Rig) -> None:
    with pytest.raises(RunError) as refused:
        rig.run(trainer_registry(setup.log), flag=False)
    message = str(refused.value)
    assert "--silicon" in message and "Agent Rule 9" in message, message
    assert not setup.runs_root.exists() and not rig.locks.exists()
    assert "build" not in setup.log.events and "train" not in setup.log.events


def test_only_sft_trains_on_silicon(setup: Setup, rig: Rig) -> None:
    with pytest.raises(RunError) as refused:
        rig.run(trainer_registry(setup.log), method="dpo")
    message = str(refused.value)
    assert "sft" in message and "Agent Rule 9" in message, message
    assert not setup.runs_root.exists() and "train" not in setup.log.events


@pytest.mark.parametrize("network", [None, "host", "bridge"], ids=["missing", "host", "bridge"])
def test_a_silicon_run_must_name_network_none(setup: Setup, rig: Rig, network: str | None) -> None:
    trainer = {"kind": "fake", "device": TT_DEVICE, "steps": 2}
    if network is not None:
        trainer["network"] = network
    with pytest.raises(RunError) as refused:
        rig.run(trainer_registry(setup.log), trainer=trainer)
    message = str(refused.value)
    assert "trainer.network" in message and "'none'" in message and "tt-perf" in message, message
    assert not setup.runs_root.exists() and not rig.locks.exists() and "train" not in setup.log.events


def test_provenance_records_the_network_setting(setup: Setup, rig: Rig) -> None:
    train_dir = rig.run(trainer_registry(setup.log))
    assert provenance(train_dir)["network"] == "none"


def test_a_silicon_run_needs_lassi_scratch_for_its_lock(
    setup: Setup, rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LASSI_SCRATCH")
    with pytest.raises(RunError) as refused:
        rig.run(trainer_registry(setup.log))
    assert "LASSI_SCRATCH" in str(refused.value)
    assert not setup.runs_root.exists() and "train" not in setup.log.events


@pytest.mark.parametrize("flag", [False, True], ids=["without-flag", "with-flag"])
def test_a_cpu_run_takes_no_lock_with_or_without_the_flag(setup: Setup, rig: Rig, flag: bool) -> None:
    registry = trainer_registry(setup.log)
    data = recipe_data(trainer={"kind": "fake", "device": {"kind": "cpu"}, "steps": 2})
    train_dir = setup.run(data, registry=registry, probes=rig.probes(), silicon=flag, silicon_host=rig.host_seams())
    assert provenance(train_dir)["status"] == "complete"
    assert provenance(train_dir)["network"] is None, "a trainer section without network records null"
    assert not rig.locks.exists() and rig.flock.calls == []


def test_the_command_line_passes_the_flag_and_no_variable_stands_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LASSI_SILICON", "1")
    seen: list[Any] = []
    monkeypatch.setattr(cli, "run_training", lambda path, options: seen.append(options))
    recipe = str(tmp_path / "train.yaml")
    assert cli.main(["--graphics", "off", "train", recipe, "--silicon"]) == 0
    assert cli.main(["--graphics", "off", "train", recipe]) == 0
    assert [options.silicon for options in seen] == [True, False]


# ---------------------------------------------------------------------------
# The lock, the quarantine, and the poll, all before any directory


def test_the_run_holds_the_lock_and_the_marker_while_it_trains_and_releases_both(setup: Setup, rig: Rig) -> None:
    train_dir = rig.run(trainer_registry(setup.log, look=rig.watching()))
    (seen,) = rig.seen
    assert seen["lock_held"], "the lock is held while train() runs"
    marker = seen["marker"]
    assert marker is not None and "not recorded" in marker, marker
    assert rig.fakes.serial(0) in marker and rig.fakes.address(0) in marker, marker
    assert setup.train_dir().name in marker, "the marker names the train run"
    assert not rig.flock.held(rig.locks / LOCK) and not (rig.locks / MARKER).exists()
    record = provenance(train_dir)
    assert record["status"] == "complete"
    (device,) = record["device_records"]
    assert device["kind"] == "tenstorrent" and rig.fakes.serial(0) in device["name"]
    assert device["framework"] == "pjrt-plugin-tt"


def test_a_second_holder_is_refused_at_once_before_any_directory(setup: Setup, rig: Rig) -> None:
    rig.locks.mkdir()
    fd = os.open(rig.locks / LOCK, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o600)
    try:
        os.write(fd, b"4242\n")
        rig.flock(fd, rig.fakes.LOCK_EX | rig.fakes.LOCK_NB)
        with pytest.raises(RunError) as refused:
            rig.run(trainer_registry(setup.log))
    finally:
        os.close(fd)
    message = str(refused.value)
    assert LOCK in message and "4242" in message, message
    assert not setup.runs_root.exists() and "train" not in setup.log.events
    assert rig.clock.sleeps == [], "a held lock is refused at once, never waited for"


def test_a_quarantine_marker_refuses_the_run_before_any_directory(setup: Setup, rig: Rig) -> None:
    rig.locks.mkdir()
    (rig.locks / "owner-quarantine-note").write_bytes(b"SYNTHETIC: a human stopped silicon runs\n")
    with pytest.raises(RunError) as refused:
        rig.run(trainer_registry(setup.log))
    assert "owner-quarantine-note" in str(refused.value)
    assert not setup.runs_root.exists() and "train" not in setup.log.events
    assert not rig.flock.held(rig.locks / LOCK), "the lock is released after the refusal"


def test_busy_cards_are_polled_then_refused_before_any_directory(setup: Setup, rig: Rig) -> None:
    rig.host.at("/dev/hugepages-1G/other-tenant").write_bytes(b"")
    rig.owners.uids["dev/hugepages-1G/other-tenant"] = rig.fakes.OTHER_UID
    with pytest.raises(RunError) as refused:
        rig.run(trainer_registry(setup.log))
    assert "other-tenant" in str(refused.value)
    assert rig.clock.sleeps, "the cards are polled before the refusal"
    assert not setup.runs_root.exists() and "train" not in setup.log.events


# ---------------------------------------------------------------------------
# How train() ended settles the marker; nothing is retried


def test_a_clean_ending_removes_the_marker_and_records_the_failure(setup: Setup, rig: Rig) -> None:
    clean = silicon().CleanEnding

    class Refused(clean, ValueError):
        """A SYNTHETIC refusal the worker gave before it touched the card."""

    def refuse(job: Any) -> Any:
        raise Refused("SYNTHETIC: a record is longer than trainer.max_length")

    with pytest.raises(Refused):
        rig.run(trainer_registry(setup.log, behave=refuse))
    assert not (rig.locks / MARKER).exists()
    assert provenance(setup.train_dir())["status"] == "failed"
    assert setup.log.events.count("train") == 1 and not rig.flock.held(rig.locks / LOCK)


def test_a_card_ending_keeps_the_marker_naming_it(setup: Setup, rig: Rig) -> None:
    card_ending = silicon().CardEnding

    def hang(job: Any) -> Any:
        raise card_ending("SYNTHETIC: the worker hung", "a hang at the wall limit (SYNTHETIC)")

    with pytest.raises(card_ending):
        rig.run(trainer_registry(setup.log, behave=hang))
    text = (rig.locks / MARKER).read_text(encoding="ascii")
    assert "a hang at the wall limit (SYNTHETIC)" in text and "not recorded" not in text, text
    assert rig.fakes.serial(0) in text
    assert provenance(setup.train_dir())["status"] == "failed"
    assert setup.log.events.count("train") == 1, "nothing is retried after a hang"


def test_any_other_error_keeps_the_marker_naming_it(setup: Setup, rig: Rig) -> None:
    def crash(job: Any) -> Any:
        raise RuntimeError("SYNTHETIC: something failed while the card was held")

    with pytest.raises(RuntimeError):
        rig.run(trainer_registry(setup.log, behave=crash))
    text = (rig.locks / MARKER).read_text(encoding="ascii")
    assert "RuntimeError" in text, text
    assert not rig.flock.held(rig.locks / LOCK)
    assert setup.log.events.count("train") == 1


def test_a_later_run_is_refused_while_the_marker_stays(setup: Setup, rig: Rig) -> None:
    def crash(job: Any) -> Any:
        raise RuntimeError("SYNTHETIC")

    with pytest.raises(RuntimeError):
        rig.run(trainer_registry(setup.log, behave=crash))
    with pytest.raises(RunError) as refused:
        setup.run(recipe_data(trainer={"kind": "fake", "device": TT_DEVICE, "steps": 2, "network": "none"}),
                  registry=trainer_registry(setup.log), probes=rig.probes(), silicon=True,
                  silicon_host=rig.host_seams(), train_id="t2")
    assert MARKER in str(refused.value)
    assert not setup.train_dir("t2").exists()
