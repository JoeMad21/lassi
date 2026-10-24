"""The base models a train recipe names at base_model: three kinds, read and checked without a framework (2026-10-23).

Bible: Training Module (Weight Modes, Compute), Agent Rules 7 and 10; OQ-076
(the owner's techniques: full training, frozen weights, LoRA, and a model of
our own).

- `fixture:<name>` (task P17.9): a directory under tests/fixtures/train/
  (lassi.train.data SYNTHETIC_DIR) holding FIXTURE_FILES; the model is
  built from its config with random weights under the trainer's seed, with
  its tokenizer from tokenizer.json. Nothing is downloaded.
- `scratch:<name>`, the from-scratch baseline (the owner's "training our own
  model"): a directory under SCRATCH_DIR, assets/train/scratch/, holding
  SCRATCH_FILES: config.json, a model config written here, and
  tokenizer-pin.json, {"repo": "<owner>/<name>", "revision": "<40 hex>"},
  the Hub tokenizer it trains with. The model is built from the config with
  random weights under the seed, exactly as a fixture is; only the
  tokenizer's files come from the Hub cache.
- `<owner>/<name>@<revision>`: a Hub base at a full commit id (Agent Rule
  10; a Hub id without one is refused). Its weights, config, and tokenizer
  load from the Hub cache, in float32, so full weights train with float32
  master weights under TRL's bf16 mixed precision.

The Hub cache is $HF_HOME/hub (Agent Rule 7), laid out as huggingface_hub
lays it out: models--<owner>--<name>/snapshots/<revision>/ (snapshot_dir).
Nothing here downloads: a snapshot the cache lacks is refused by
check_base, naming the separate prepare step, tools/prepare_train.py
(lassi.train.prepare), which runs only when a person starts it. Every load
passes the revision, cache_dir, local_files_only=True, and
trust_remote_code=False, and the trl trainer sets the hub library offline
first (lassi.llm.hf_local hub_offline).

base_record is the checkpoint's record of the base: a fixture's is
{name, files, seed} as in task P17.9; a scratch base adds kind, its own
files' sha256, and the tokenizer's repo, revision, and snapshot files; a
Hub base holds kind, repo, revision, the sha256 of every snapshot file, and
the seed. This module imports only the standard library and lassi at
module level; load_base is handed the framework modules.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from lassi.core.digest import file_sha256
from lassi.train import data as train_data

REPO = Path(__file__).resolve().parents[2]
# Where scratch:<name> names a directory, and the files a fixture and a scratch directory hold.
SCRATCH_DIR = REPO / "assets" / "train" / "scratch"
FIXTURE_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")
SCRATCH_FILES = ("config.json", "tokenizer-pin.json")
TOKENIZER_PIN = "tokenizer-pin.json"
# The separate, explicit step that fills the Hub cache; check_base names it when a snapshot is missing.
PREPARE = "uv run --extra cpu tools/prepare_train.py <train.yaml>"
_PLAIN = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_REPO_ID = re.compile(rf"{_PLAIN}/{_PLAIN}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_LOCAL = re.compile(rf"(fixture|scratch):({_PLAIN})")


@dataclass(frozen=True)
class Base:
    """A parsed base_model: its kind (fixture, scratch, or hub), the value as written, and where it is read from.

    `path` is the fixture or scratch directory (None for hub); `repo` and
    `revision` name the Hub base, or for a scratch base its tokenizer once
    read (None until then and for a fixture).
    """

    kind: str
    name: str
    path: Path | None
    repo: str | None = None
    revision: str | None = None


def parse_base(value: Any) -> Base:
    """Return the Base base_model names; ValueError naming base_model for any value outside the three kinds."""
    if not isinstance(value, str):
        raise ValueError(f"base_model must be a string, not {value!r}")
    local = _LOCAL.fullmatch(value)
    if local is not None:
        root = train_data.SYNTHETIC_DIR if local.group(1) == "fixture" else SCRATCH_DIR
        return Base(kind=local.group(1), name=value, path=root / local.group(2))
    repo, at, revision = value.partition("@")
    if _REPO_ID.fullmatch(repo) and not at:
        raise ValueError(
            f"base_model {value!r}: a Hub base needs a revision pin (Agent Rule 10); write "
            "<owner>/<name>@<the 40-character commit id>"
        )
    if _REPO_ID.fullmatch(repo) and at:
        if not _REVISION.fullmatch(revision):
            raise ValueError(f"base_model {value!r}: the revision must be a full commit id, 40 lowercase hex digits")
        return Base(kind="hub", name=value, path=None, repo=repo, revision=revision)
    raise ValueError(
        f"base_model {value!r} is none of fixture:<name> (under {train_data.SYNTHETIC_DIR}), scratch:<name> "
        f"(under {SCRATCH_DIR}), or <owner>/<name>@<revision>"
    )


def snapshot_dir(hf_home: str, repo: str, revision: str) -> Path:
    """Return $HF_HOME/hub/models--<owner>--<name>/snapshots/<revision>, the Hub cache's snapshot directory."""
    return Path(hf_home) / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / revision


def read_tokenizer_pin(path: Path) -> tuple[str, str]:
    """Return (repo, revision) of a scratch base's tokenizer-pin.json; ValueError naming the file otherwise."""
    try:
        document = json.loads(path.read_bytes().decode("ascii"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"{path} cannot be read as an ASCII JSON tokenizer pin: {error}") from None
    if not isinstance(document, dict) or set(document) != {"repo", "revision"}:
        raise ValueError(f"{path} must be exactly {{repo, revision}}, a Hub tokenizer at a commit, not {document!r}")
    repo, revision = document["repo"], document["revision"]
    if not isinstance(repo, str) or not _REPO_ID.fullmatch(repo):
        raise ValueError(f"{path}: repo must be a Hub repository id <owner>/<name>, not {repo!r}")
    if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
        raise ValueError(f"{path}: revision must be a full commit id, 40 lowercase hex digits, not {revision!r}")
    return repo, revision


def check_base(base: Base, hf_home: str) -> Base:
    """Refuse a base whose files or Hub snapshot are missing, naming base_model; return it with its pins read.

    A scratch base comes back with its tokenizer's repo and revision. A
    missing snapshot names the prepare step; nothing is downloaded here.
    """
    if base.kind in ("fixture", "scratch"):
        files = FIXTURE_FILES if base.kind == "fixture" else SCRATCH_FILES
        assert base.path is not None
        if base.path.resolve().parent != base.path.parent.resolve():
            raise ValueError(f"base_model {base.name!r} names a path outside {base.path.parent}")
        missing = [file for file in files if not (base.path / file).is_file()]
        if missing:
            raise ValueError(f"base_model {base.name!r}: {base.path} lacks {', '.join(missing)}")
        if base.kind == "fixture":
            return base
        repo, revision = read_tokenizer_pin(base.path / TOKENIZER_PIN)
        base = Base(kind="scratch", name=base.name, path=base.path, repo=repo, revision=revision)
    assert base.repo is not None and base.revision is not None
    snapshot = snapshot_dir(hf_home, base.repo, base.revision)
    needed = "tokenizer.json" if base.kind == "scratch" else "config.json"
    if not (snapshot / needed).is_file():
        what = "its tokenizer" if base.kind == "scratch" else "it"
        raise ValueError(
            f"base_model {base.name!r}: $HF_HOME/hub holds no snapshot of {base.repo} at {base.revision} "
            f"({snapshot}); download {what} first with the separate prepare step, {PREPARE}, which never runs "
            "inside a training run"
        )
    return base


def base_record(base: Base, hf_home: str, seed: int) -> dict[str, Any]:
    """Return the checkpoint's record of the base: its files' sha256, its pins, and the seed (the module docstring)."""
    if base.kind == "fixture":
        assert base.path is not None
        return {"name": base.name, "files": {file: file_sha256(base.path / file) for file in FIXTURE_FILES},
                "seed": seed}
    assert base.repo is not None and base.revision is not None
    snapshot = {"repo": base.repo, "revision": base.revision,
                "files": snapshot_files(snapshot_dir(hf_home, base.repo, base.revision))}
    if base.kind == "hub":
        return {"name": base.name, "kind": "hub", **snapshot, "seed": seed}
    assert base.path is not None
    return {"name": base.name, "kind": "scratch",
            "files": {file: file_sha256(base.path / file) for file in SCRATCH_FILES}, "tokenizer": snapshot,
            "seed": seed}


def snapshot_files(snapshot: Path) -> dict[str, str]:
    """Return {relative POSIX path: sha256} of every file of a Hub snapshot, in path order.

    The Hub cache keeps a snapshot's files as symbolic links into its blobs
    directory on Linux (copies where links are not allowed), so each link is
    followed and the content it names is hashed.
    """
    found: dict[str, str] = {}
    for directory, _, names in os.walk(snapshot):
        for name in names:
            path = Path(directory) / name
            if path.is_file():
                found[path.relative_to(snapshot).as_posix()] = file_sha256(path)
    return dict(sorted(found.items()))


def load_base(transformers: ModuleType, torch: ModuleType, base: Base, seed: int, hf_home: str) -> tuple[Any, Any]:
    """Return (model, tokenizer) of a checked base; the seed is set before any weight is made or loaded."""
    if base.kind == "fixture":
        return _load_fixture(transformers, base, seed)
    assert base.repo is not None and base.revision is not None
    options = _hub_options(base.revision, hf_home)
    tokenizer = transformers.AutoTokenizer.from_pretrained(base.repo, **options)
    transformers.set_seed(seed)
    if base.kind == "hub":
        model = transformers.AutoModelForCausalLM.from_pretrained(base.repo, dtype=torch.float32, **options)
        return model, tokenizer
    assert base.path is not None
    config = transformers.AutoConfig.from_pretrained(str(base.path), local_files_only=True)
    vocabulary = getattr(config, "vocab_size", None)
    if not isinstance(vocabulary, int) or len(tokenizer) > vocabulary:
        raise ValueError(f"base_model {base.name!r}: the tokenizer has {len(tokenizer)} entries, more than the "
                         f"config's vocab_size {vocabulary}")
    return transformers.AutoModelForCausalLM.from_config(config), tokenizer


def _hub_options(revision: str, hf_home: str) -> Mapping[str, Any]:
    """Return the keyword arguments of every Hub-cache load: the revision, $HF_HOME/hub, offline, no remote code."""
    return {"revision": revision, "cache_dir": str(Path(hf_home) / "hub"), "local_files_only": True,
            "trust_remote_code": False}


def _load_fixture(transformers: ModuleType, base: Base, seed: int) -> tuple[Any, Any]:
    """Return a fixture base built from its config with random weights under the seed, and its tokenizer."""
    assert base.path is not None
    config = transformers.AutoConfig.from_pretrained(str(base.path), local_files_only=True)
    settings = json.loads((base.path / "tokenizer_config.json").read_bytes().decode("ascii"))
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_file=str(base.path / "tokenizer.json"),
        eos_token=settings["eos_token"],
        pad_token=settings["pad_token"],
        model_max_length=settings["model_max_length"],
    )
    transformers.set_seed(seed)
    return transformers.AutoModelForCausalLM.from_config(config), tokenizer
