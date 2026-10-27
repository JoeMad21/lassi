"""The guard of `lassi train --silicon`: a training run on a Tenstorrent card, started by a human (task P17.38).

Bible: Agent Rule 9 as OQ-100 (a) amends it ("silicon runs are evaluation
runs, or training runs of tracked code, that a human starts"; RL rewards
never execute on silicon); OQ-100 (s2); OQ-075 (2a), (3a), and (5), the
tt_silicon executor's lock, quarantine marker, and poll, which a training
run shares (lassi.executors.tt_silicon card_lock, quarantine_markers, and
wait_for_cards); Training Module, Safeguards.

A train recipe whose trainer.device names a kind in SILICON_KINDS is a
silicon run. lassi.train.run refuses it, before any directory exists,
unless TrainOptions.silicon is True (`lassi train --silicon`, the human
start no recipe can set), unless its method is sft (SILICON_METHODS: no
reward, so RL never touches silicon), and unless trainer.network is none
(SILICON_NETWORK, no default: the worker runs in its own empty network
namespace, since the training stack's Tracy client listens and broadcasts
from library load; plans/spikes/tt-perf.md), which provenance.json records. card_guard() builds its CardGuard,
refusing an unset or relative $LASSI_SCRATCH, which holds the lock.

CardGuard.held() is entered after every other refusal and before any
directory exists, and left after the final provenance.json: it takes the
one lock of every card (never a wait: another holder is refused at once),
refuses while any quarantine marker exists under $LASSI_SCRATCH/locks, and
polls until no other account holds the cards (the executor's poll, for
every card of the host, since which chips the framework opens is not
known), each refusal a RunError. CardGuard.run(call) then writes the
quarantine marker naming the ending "not recorded" (so a kill of this
process or a host crash leaves it), calls the trainer's train() once,
never retried, and settles the marker by how it ended:

- train() returned, or raised a CleanEnding (the trainer's word that the
  card was left in a known state: its worker never started, or exited with
  a status from 0 to 127 without a hang or a kill, as the executor reads a
  clean ending): the marker is removed;
- it raised a CardEnding (a hang, a kill, or a crash the trainer saw): the
  marker is rewritten naming CardEnding.ending;
- it raised anything else (an interrupt, an error of the trainer itself):
  the marker is rewritten naming the exception's type.

The marker names the train run, the trainer, and each card's serial and PCI
address, read from sysfs (lassi.executors.devices tenstorrent_inventory).
While it exists every later silicon run, evaluation or training, is
refused until a human checks the cards and removes it. Device files are
read only through lassi.executors (Design Principle 9).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from lassi.core.devices import DeviceUnavailable
from lassi.core.record import DeviceRecord
from lassi.core.runner import RunError
from lassi.executors import devices as host_devices
from lassi.executors import tt_silicon
from lassi.executors.sandbox import SandboxUnavailableError
from lassi.executors.ttsim import printable

# The device kinds a training run reaches silicon through, the methods it may run there (no reward), and the one
# trainer.network a run there may name: none, its worker in its own empty network namespace.
SILICON_KINDS = ("tenstorrent",)
SILICON_METHODS = ("sft",)
SILICON_NETWORK = "none"
# The ending a marker names from before train() is called until its ending replaces it or a clean one removes it.
UNRECORDED = (
    "not recorded: this marker was written before the training run's worker could start on the card, and the "
    "process that wrote it ended before it recorded how the run ended (a kill of that process, or a host crash)"
)
T = TypeVar("T")


class CleanEnding(Exception):
    """A trainer's error that leaves the card in a known state: its worker never started or exited cleanly.

    A mixin: a trainer raises, for example, a class deriving from this and
    ValueError for a record its worker refused; the guard then removes the
    quarantine marker.
    """


class CardEnding(RuntimeError):
    """A trainer's run that left the card in an unknown state: a hang, a kill, or a crash; `ending` says which."""

    def __init__(self, message: str, ending: str) -> None:
        """Keep the message and the ending the quarantine marker names."""
        super().__init__(message)
        self.ending = ending


@dataclass(frozen=True)
class SiliconHost:
    """The host seams of the guard: the root of every host path, the access check, flock, the uid, owners, clock.

    None means the real host's: fcntl.flock (imported when the lock is
    taken) and os.getuid().
    """

    root: Path = host_devices.HOST_ROOT
    access: Callable[[Path], bool] = host_devices._accessible
    flock: Callable[[int, int], object] | None = None
    uid: int | None = None
    owner: Callable[[Path], int] = tt_silicon._owner
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep


def is_silicon(device: DeviceRecord) -> bool:
    """Return True when the device record's kind reaches silicon."""
    return device.kind in SILICON_KINDS


def check_start(where: str, kind: str, method: Any, network: Any, silicon: bool) -> None:
    """Refuse a silicon device without the human start, or with a method other than sft or a network other than none.

    Each refusal is a RunError naming why. trainer.network has no default:
    the training stack's Tracy client listens on TCP and broadcasts on UDP
    from library load (plans/spikes/tt-perf.md), so the trainer names
    network none and runs its worker with no network.
    """
    if kind not in SILICON_KINDS:
        return
    if not silicon:
        raise RunError(
            f"{where}: trainer.device is a {kind} card, and training runs on silicon are runs a human starts (Agent "
            "Rule 9, as OQ-100 (a) amends it): start this run with `lassi train <recipe> --silicon`, which no recipe "
            "can set and no unattended session passes"
        )
    if method not in SILICON_METHODS:
        raise RunError(
            f"{where}: method {method!r} on a {kind} card: only {', '.join(SILICON_METHODS)} trains on silicon, since "
            "RL rewards never execute on silicon (Agent Rule 9)"
        )
    if network != SILICON_NETWORK:
        raise RunError(
            f"{where}: trainer.network is {network!r} on a {kind} card; it must be {SILICON_NETWORK!r}, with no "
            "default: the training worker runs in its own empty network namespace, since the training stack's "
            "Tracy client listens on TCP and broadcasts on UDP from library load (plans/spikes/tt-perf.md)"
        )


def card_guard(device: DeviceRecord, train_id: str, trainer: str, host: SiliconHost | None) -> CardGuard | None:
    """Return the guard of a silicon device's run, None for any other device; RunError without $LASSI_SCRATCH."""
    if not is_silicon(device):
        return None
    value = os.environ.get("LASSI_SCRATCH", "")
    if not value or not Path(value).is_absolute():
        raise RunError(
            "LASSI_SCRATCH is unset or not absolute; a training run on a card holds the cards' lock and quarantine "
            "marker under $LASSI_SCRATCH/locks, as the tt_silicon executor does"
        )
    return CardGuard(device, train_id, trainer, SiliconHost() if host is None else host, Path(value))


class CardGuard:
    """The lock, the quarantine, and the poll of one training run on a card (see the module docstring)."""

    def __init__(self, device: DeviceRecord, train_id: str, trainer: str, host: SiliconHost, scratch: Path) -> None:
        """Keep the run's identity and seams, and read its cards' serials and PCI addresses for the marker."""
        self.device, self.train_id, self.trainer, self.host = device, train_id, trainer, host
        self.locks = scratch / tt_silicon.LOCKS_DIR
        self.marker = self.locks / tt_silicon.QUARANTINE_FILE
        self.card_lines = _card_lines(device, host)

    @contextmanager
    def held(self) -> Iterator[None]:
        """Hold the lock for the block, after the quarantine check and the poll; each refusal a RunError."""
        with ExitStack() as stack:
            try:
                stack.enter_context(tt_silicon.card_lock(self.locks, self.host.flock))
            except SandboxUnavailableError as error:
                raise RunError(f"{error}; no train directory was made") from None
            self._check_free()
            yield

    def _check_free(self) -> None:
        """Refuse while a quarantine marker exists, then poll the cards; RunError naming why, before any directory."""
        markers = tt_silicon.quarantine_markers(self.locks)
        if markers:
            raise RunError(
                f"the Tenstorrent cards are quarantined: {', '.join(str(path) for path in markers)} exists (a silicon "
                "run ended in a way that leaves them in an unknown state, as the file says); a human checks every "
                "card and removes it before another silicon run (OQ-075 (3a)); no train directory was made"
            )
        uid = os.getuid() if self.host.uid is None else self.host.uid
        needed = tt_silicon.PAGES_PER_CARD * (self.device.count or 1)
        try:
            tt_silicon.wait_for_cards(self.host.root, uid, needed, self.host.owner, self.host.clock, self.host.sleep)
        except SandboxUnavailableError as error:
            raise RunError(f"{error}; no train directory was made") from None

    def run(self, call: Callable[[], T]) -> T:
        """Write the marker, call `call` once, and settle the marker by how it ended (the module docstring)."""
        self._write([UNRECORDED])
        try:
            result = call()
        except CleanEnding:
            with suppress(OSError):
                self.marker.unlink(missing_ok=True)
            raise
        except CardEnding as error:
            with suppress(OSError):
                self._write([error.ending])
            raise
        except BaseException as error:
            with suppress(OSError):
                self._write([f"the run ended by {type(error).__name__} while the trainer held the card: {error}"])
            raise
        self.marker.unlink(missing_ok=True)
        return result

    def _write(self, endings: list[str]) -> None:
        """Write the quarantine marker naming the endings, the run, and its cards (OQ-075 (3a))."""
        written = datetime.now(timezone.utc).isoformat(timespec="seconds")
        lines = [
            f"tt_silicon quarantine (OQ-075 (3a)), written {written} by pid {os.getpid()}: lassi train --silicon, "
            f"train id {self.train_id}, trainer {self.trainer}",
            *(f"ending: {ending}" for ending in endings),
            *self.card_lines,
            "Every later silicon run, lassi run --silicon or lassi train --silicon, is refused until a human checks "
            "these cards and removes this file.",
        ]
        self.locks.mkdir(mode=0o700, parents=True, exist_ok=True)
        text = "".join(f"{printable(line)}\n" for line in lines)
        self.marker.write_bytes(text.encode("ascii"))


def _card_lines(device: DeviceRecord, host: SiliconHost) -> list[str]:
    """Return one marker line per card the device names: its serial and PCI address, or that they were not read."""
    try:
        cards = host_devices.tenstorrent_inventory(host.root, host.access).cards
    except (DeviceUnavailable, OSError):
        cards = {}
    lines = []
    for index in device.indices:
        card = cards.get(index)
        if card is None:
            lines.append(f"card {index}: serial and PCI address not read")
        else:
            lines.append(f"card {index}: serial {card.serial}, PCI {card.address or 'not read'}")
    return lines
