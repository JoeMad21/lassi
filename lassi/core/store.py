"""The text store, the binary store, and the per-trial files under a run tree.

Large texts (prompts, responses, stdout) are stored once by their sha256 and
referenced from records by TextRef (bible Result Record, Storage); run
output files are stored once by their sha256 in the binary store beside it
(BlobStore) and referenced from RunInfo.outputs by hash. Each trial
directory under the run tree holds `trial.json`, the record with every
response text replaced by its reference, and `trial.md`, the human-readable
view (bible Readability Standards, Trial row). Files are written with LF
newlines on every OS.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from lassi.core.record import TextRef, Trial, from_dict, json_text, parse_trial_id, to_dict
from lassi.core.trial_md import render_trial_md

TRIAL_JSON = "trial.json"
TRIAL_MD = "trial.md"

_SHA256 = re.compile(r"[0-9a-f]{64}")


def sha256_text(text: str) -> str:
    """Return the sha256 hex digest of the UTF-8 bytes of `text`."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TextNotFoundError(KeyError):
    """Raised when the text store holds no text for a hash."""


class TextStore:
    """Texts stored once by hash at `<root>/texts/<sha[:2]>/<sha>.txt`, as their exact UTF-8 bytes."""

    def __init__(self, root: Path) -> None:
        """Use `root` as the store directory; it is created on the first put."""
        self.root = Path(root)

    def put(self, text: str) -> TextRef:
        """Store `text` unless it is already stored, and return its reference.

        Concurrent puts of one text all succeed: when another writer stores the
        file first, this write may fail (os.replace does on Windows while the
        file is being replaced or read), and the file already stored is kept.
        """
        data = text.encode("utf-8")
        sha = hashlib.sha256(data).hexdigest()
        ref = TextRef(sha256=sha, path=_text_path(sha))
        target = self.root / ref.path
        if not target.exists():
            try:
                _write_atomic(target, data)
            except OSError:
                if not target.is_file():
                    raise
        return ref

    def get(self, ref: TextRef | str) -> str:
        """Return the text for a reference or sha256 hex string.

        Raises TextNotFoundError when the text is not stored (no file at its
        path), and ValueError for a malformed hash, for a reference whose path
        is not the store path of its hash, or when the stored bytes no longer
        match the hash.
        """
        sha = _sha_of(ref)
        path = self._path(sha)
        if not path.is_file():
            raise TextNotFoundError(sha)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(f"stored text {sha} does not match its hash")
        return data.decode("utf-8")

    def __contains__(self, ref: TextRef | str) -> bool:
        """Return True when a file is stored for a reference or sha256 hex string.

        A malformed hash or a reference whose path is not the store path of its
        hash is never contained. Only the file's presence is checked; get()
        also checks its bytes against the hash.
        """
        try:
            sha = _sha_of(ref)
        except ValueError:
            return False
        return self._path(sha).is_file()

    def _path(self, sha: str) -> Path:
        """Return the file that holds the text with this hash."""
        return self.root / _text_path(sha)


class BlobStore:
    """Bytes stored once by hash at `<root>/blobs/<sha[:2]>/<sha>`, beside the text store's `<root>/texts/`.

    The binary store keeps the output files of runs (RunInfo.outputs, bible
    Result Record, Storage). Stages use BlobStore(<the text store's root>),
    so a run's blobs sit at `<run>/blobs/`.
    """

    def __init__(self, root: Path) -> None:
        """Use `root` as the store directory; `blobs/` under it is created on the first put."""
        self.root = Path(root)

    def put(self, data: bytes) -> str:
        """Store `data` unless it is already stored, and return its sha256 hex digest.

        Concurrent puts of one blob all succeed, as for TextStore.put.
        """
        raw = bytes(data)
        sha = hashlib.sha256(raw).hexdigest()
        target = self.root / _blob_path(sha)
        if not target.exists():
            try:
                _write_atomic(target, raw)
            except OSError:
                if not target.is_file():
                    raise
        return sha

    def get(self, sha: str) -> bytes:
        """Return the bytes stored for a sha256 hex string.

        Raises KeyError when none are stored, and ValueError for a malformed
        hash or when the stored bytes no longer match the hash.
        """
        return self.path(sha).read_bytes()

    def path(self, sha: str) -> Path:
        """Return the file that holds the blob `sha`, after checking its bytes against the hash.

        Raises KeyError when no file is stored, and ValueError for a
        malformed hash or bytes that no longer match it.
        """
        if not isinstance(sha, str) or not _SHA256.fullmatch(sha):
            raise ValueError(f"not a sha256 hex string: {sha!r}")
        path = self.root / _blob_path(sha)
        if not path.is_file():
            raise KeyError(sha)
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError(f"stored blob {sha} does not match its hash")
        return path

    def __contains__(self, sha: object) -> bool:
        """Return True when a file is stored for a sha256 hex string; a malformed hash is never contained."""
        if not isinstance(sha, str) or not _SHA256.fullmatch(sha):
            return False
        return (self.root / _blob_path(sha)).is_file()


def _blob_path(sha: str) -> str:
    """Return the store path of the blob with this hash: `blobs/<sha[:2]>/<sha>`."""
    return f"blobs/{sha[:2]}/{sha}"


def _text_path(sha: str) -> str:
    """Return the store path of the text with this hash: `texts/<sha[:2]>/<sha>.txt`."""
    return f"texts/{sha[:2]}/{sha}.txt"


def _sha_of(ref: TextRef | str) -> str:
    """Return the sha256 a reference or hex string names.

    Raises ValueError for a malformed hash, and for a reference whose path is
    not the store path of its hash, so a reference never names a file other
    than the one its text is read from.
    """
    if isinstance(ref, TextRef):
        if ref.path != _text_path(ref.sha256):
            raise ValueError(f"text reference path {ref.path!r} is not the store path {_text_path(ref.sha256)!r}")
        return ref.sha256
    if not isinstance(ref, str) or not _SHA256.fullmatch(ref):
        raise ValueError(f"not a sha256 hex string: {ref!r}")
    return ref


def _write_atomic(target: Path, data: bytes) -> None:
    """Write bytes to `target` through a temporary file in the same directory and os.replace."""
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=target.parent, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(handle, "wb") as temp_file:
            temp_file.write(data)
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)


def trial_dir(run_root: Path, trial_id: str) -> Path:
    """Return the directory of a trial under the run tree: one level per trial_id segment."""
    parse_trial_id(trial_id)
    return Path(run_root).joinpath(*trial_id.split("/"))


def write_trial(trial: Trial, run_root: Path, store: TextStore) -> Path:
    """Write trial.json and trial.md for `trial` under `run_root` and return the trial directory.

    In trial.json each attempt's response_text is replaced by the reference of
    the text in `store`; everything else is the record as to_dict gives it.
    Both files are built before either is written, and each is replaced
    through a temporary file, so an error while building them (such as
    TextNotFoundError for a prompt missing from `store`) leaves the directory
    as it was. Raises ValueError, before writing anything, when the directory
    already holds a different trial (see _check_same_trial).
    """
    out = trial_dir(run_root, trial.trial_id)
    _check_same_trial(out / TRIAL_JSON, trial.trial_id)
    md_text = render_trial_md(trial, store)
    data = to_dict(trial)
    for attempt_data, attempt in zip(data["attempts"], trial.attempts, strict=True):
        attempt_data["response_text"] = to_dict(store.put(attempt.response_text))
    json_bytes = json_text(data).encode("ascii")
    _write_atomic(out / TRIAL_JSON, json_bytes)
    _write_atomic(out / TRIAL_MD, md_text.encode("ascii"))
    return out


def _check_same_trial(json_path: Path, trial_id: str) -> None:
    """Raise ValueError when `json_path` holds the record of a trial other than `trial_id`.

    Ids that differ only by letter case or a trailing '.' share one directory
    on Windows, so one trial could silently overwrite another. A missing file,
    or one whose content is not a record, is not another trial and may be
    replaced; any other error reading it (such as a permission or sharing
    error) propagates, since the file may hold another trial.
    """
    try:
        existing = json.loads(json_path.read_text(encoding="utf-8"))["trial_id"]
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        return
    if existing != trial_id:
        raise ValueError(f"{json_path} holds trial {existing!r}, not {trial_id!r}; refusing to overwrite it")


def read_trial(path: Path, store: TextStore) -> Trial:
    """Read a Trial from a trial directory or its trial.json, resolving response texts through `store`.

    A record the file does not hold, such as a trial.json written before a
    required field existed, raises ValueError naming the file and the field.
    """
    path = Path(path)
    json_path = path / TRIAL_JSON if path.is_dir() else path
    data = json.loads(json_path.read_text(encoding="utf-8"))
    attempts = data.get("attempts", []) if isinstance(data, dict) else None
    if not isinstance(attempts, list) or not all(isinstance(item, dict) for item in attempts):
        raise ValueError(f"{json_path}: expected a trial object whose attempts are a list of objects")
    for position, attempt_data in enumerate(attempts):
        where = f"{json_path}: attempts[{position}].response_text"
        attempt_data["response_text"] = _stored_text(attempt_data.get("response_text"), store, where)
    try:
        return from_dict(Trial, data)
    except ValueError as error:
        raise ValueError(f"{json_path}: {error}") from error


def _stored_text(stored: object, store: TextStore, where: str) -> str:
    """Return the text a trial.json reference names; its path must be the store path of its hash."""
    if not isinstance(stored, dict):
        raise ValueError(f"{where} must be a text reference, got {stored!r}")
    ref = from_dict(TextRef, stored)
    try:
        return store.get(ref)
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from error
