"""The progress hook: the events the runner and the stages send to an in-process observer (task P4.8).

Bible Component Interfaces (contract rules) and Readability Standards,
Terminal Presentation. The runner writes trial.json only when a trial ends,
so a live view of a trial in progress (the inference table,
lassi.present.live) needs the trial as it changes. An observer is any
callable that takes one ProgressEvent; `lassi run` passes one with graphics
on, through RunOptions.observer, and the runner hands it to every stage
through RunContext.observer.

The events, in the order a trial sends them (EVENT_KINDS):

- trial-start (TRIAL_START): the runner, once per trial, after the trial is
  built and before its first stage, with `run_id`, `number`, `count`, and
  `source` set;
- stage-start (STAGE_START): the runner, before each stage it runs, with
  `stage`, the stage's registered name;
- request-sent (REQUEST_SENT): a stage, before each model call, once its
  messages are in the text store and before the reply is recorded, with
  `stage`, the stage the Request is recorded under;
- attempt (ATTEMPT): a stage, after generate or a correction appends an
  attempt and after compile_loop builds or run_loop runs the last attempt,
  with `stage`;
- trial-end (TRIAL_END): the runner, once per trial, after the trial is
  scored and written, carrying the trial as written.

Every event carries `trial`, the Trial as it stands at that moment. A Trial
is frozen and the stages never change one in place, so an event's trial
never changes afterwards; an observer reads it and never changes it.

An observer writes nothing the run keeps, and nothing depends on it. The
runner wraps the one it is given in a GuardedObserver: an observer that
raises an Exception is dropped for the rest of the run, with one line on
stderr naming the exception's class when stderr can take it
(stderr_notice), and the run goes on as if none had been passed, so it
changes no record, file, or exit status. When a write or flush to stderr
fails with OSError (a pipe whose reader has gone), stderr is pointed at
the null device (mute_stderr), so the bytes left in its buffer cannot fail
the flush at interpreter exit and change the process's exit status.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from lassi.core.record import Trial

TRIAL_START = "trial-start"
STAGE_START = "stage-start"
REQUEST_SENT = "request-sent"
ATTEMPT = "attempt"
TRIAL_END = "trial-end"
EVENT_KINDS = (TRIAL_START, STAGE_START, REQUEST_SENT, ATTEMPT, TRIAL_END)


@dataclass(frozen=True)
class ProgressEvent:
    """One progress event: its kind (one of EVENT_KINDS) and the trial as it stands.

    `stage` is the registered stage name on stage-start, request-sent, and
    attempt events. The runner sets the other four on each trial-start
    event: `run_id`, the run directory's name; `number`, the trial's 1-based
    place in the run; `count`, the run's trial count; and `source`, the
    item's files in the direction's source language (file name -> text, as
    Suite.source_files reads them), None when they cannot be read. Each
    defaults to None. Any other kind raises ValueError.
    """

    kind: str
    trial: Trial
    stage: str | None = None
    run_id: str | None = None
    number: int | None = None
    count: int | None = None
    source: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        """Refuse a kind that is not one of EVENT_KINDS."""
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"ProgressEvent.kind must be one of {', '.join(EVENT_KINDS)}, not {self.kind!r}")


Observer = Callable[[ProgressEvent], None]
"""What the runner and the stages send events to: any callable that takes one ProgressEvent."""


class GuardedObserver:
    """Passes each event to an observer until it raises; from then on the observer gets none.

    The exception is swallowed (an Exception, never a KeyboardInterrupt or
    SystemExit) and one line on stderr names its class when stderr can take
    it (stderr_notice, which never raises), so an observer can never stop a
    run or change what it writes, even when its failure is a broken stderr.
    """

    def __init__(self, observer: Observer) -> None:
        """Keep the observer."""
        self._observer: Observer | None = observer

    @property
    def dropped(self) -> bool:
        """True once the observer has raised; it gets no more events."""
        return self._observer is None

    def __call__(self, event: ProgressEvent) -> None:
        """Pass `event` to the observer, unless it was dropped; drop it when it raises."""
        observer = self._observer
        if observer is None:
            return
        try:
            observer(event)
        except Exception as error:  # an observer's failure never reaches the run
            self._observer = None
            name = type(error).__name__
            stderr_notice(f"lassi: the progress observer raised {name}; it gets no more events in this run")


def stderr_notice(text: str) -> None:
    """Write `text` as one line on stderr when stderr can take it.

    A missing stderr (None), or one that raises OSError (a broken pipe) or
    ValueError (a closed stream) on the write or the flush, is passed over,
    so a notice never raises: the failure it reports is often stderr itself.
    After an OSError, mute_stderr points stderr at the null device.
    """
    stream = sys.stderr
    if stream is None:
        return
    try:
        stream.write(text + "\n")
        stream.flush()
    except OSError:
        mute_stderr(stream)
    except ValueError:
        pass


def mute_stderr(stream: object) -> None:
    """Point the process's stderr at the null device after a write or flush to `stream`, when it is sys.stderr, failed.

    The bytes of a failed write stay in the stream's buffer, and Python
    flushes sys.stderr again when the interpreter exits; if that flush fails
    too, the process exits 120 whatever the command returned. So the file
    descriptor under sys.stderr is pointed at os.devnull (os.dup2, as
    Python's signal documentation advises for SIGPIPE), and that flush and
    every later write to stderr go nowhere and succeed. A `stream` other
    than sys.stderr, or one with no file descriptor (a test's StringIO), is
    left alone. Never raises.
    """
    if stream is None or stream is not sys.stderr:
        return
    try:
        descriptor = stream.fileno()  # type: ignore[attr-defined]
        null = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(null, descriptor)
        finally:
            os.close(null)
    except Exception:  # no file descriptor, no null device, or a failed dup2: stderr stays as it is
        pass


def notify(observer: Observer | None, kind: str, trial: Trial, stage: str | None = None) -> None:
    """Send `observer` an event of `kind` with `trial` and `stage`; do nothing when there is no observer."""
    if observer is not None:
        observer(ProgressEvent(kind=kind, trial=trial, stage=stage))
