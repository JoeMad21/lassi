"""The prepare step: download a train recipe's base into $HF_HOME/hub, separately and explicitly (2026-10-23).

Bible: Training Module (Compute), Agent Rules 7 and 10; the owner's
direction of 2026-10-23: on alpha01 a download happens only in the owner's
window, as its own command, never inside a training run.

lassi train never downloads (lassi.train.bases): a Hub base, and a scratch
base's tokenizer, load from $HF_HOME/hub only, and check() refuses a
snapshot the cache lacks. This step fills the cache. plan(base) names what
a base needs: a Hub base's repo at its pinned revision with MODEL_PATTERNS
(the config files, the tokenizer, and the safetensors weights; no other
weight format, so no pickled file is fetched), a scratch base's tokenizer
repo at the revision its tokenizer-pin.json pins with TOKENIZER_PATTERNS,
and nothing for a fixture. prepare(recipe) loads the train recipe
(lassi.core.recipe load_train_recipe; nothing is built), refuses an unset
HF_HOME, calls the download function once, and then runs the same
snapshot check as lassi train, so a download that left no snapshot at the
pinned revision is a ValueError. The default download is huggingface_hub's
snapshot_download, imported only when it runs; it writes under
cache_dir=$HF_HOME/hub. tools/prepare_train.py is its command line; each
recipe under projects/train/ notes its base's download size.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lassi.core.recipe import load_train_recipe
from lassi.train import bases

# What a Hub base needs: the config files and tokenizer (JSON), the BPE merges, and the safetensors weights.
MODEL_PATTERNS = ("*.json", "merges.txt", "*.safetensors")
# What a scratch base needs of its tokenizer repo: the tokenizer files, and config.json, which names the tokenizer's
# model type for transformers' AutoTokenizer.
TOKENIZER_PATTERNS = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json")

Download = Callable[..., Any]


def plan(base: bases.Base) -> tuple[str, str, tuple[str, ...]] | None:
    """Return (repo, revision, patterns) a base needs from the Hub, or None for a fixture, which needs nothing."""
    if base.kind == "fixture":
        return None
    if base.kind == "hub":
        assert base.repo is not None and base.revision is not None
        return base.repo, base.revision, MODEL_PATTERNS
    assert base.path is not None
    repo, revision = bases.read_tokenizer_pin(base.path / bases.TOKENIZER_PIN)
    return repo, revision, TOKENIZER_PATTERNS


def prepare(recipe_path: Path, *, download: Download | None = None, dry_run: bool = False) -> Path | None:
    """Download the recipe's base into $HF_HOME/hub and return the snapshot directory; None when nothing is fetched.

    Raises ValueError for an unset HF_HOME, a base_model outside the three
    kinds, and a snapshot that is still missing after the download.
    """
    recipe = load_train_recipe(Path(recipe_path))
    hf_home = os.environ.get("HF_HOME", "")
    if not hf_home:
        raise ValueError("HF_HOME is not set or is empty; the prepare step downloads into $HF_HOME/hub only "
                         "(Agent Rule 7)")
    base = bases.parse_base(recipe.data["base_model"])
    wanted = plan(base)
    if wanted is None or dry_run:
        return None
    repo, revision, patterns = wanted
    get = _snapshot_download if download is None else download
    get(repo_id=repo, revision=revision, cache_dir=str(Path(hf_home) / "hub"), allow_patterns=list(patterns))
    bases.check_base(base, hf_home)
    return bases.snapshot_dir(hf_home, repo, revision)


def _snapshot_download(**kwargs: Any) -> Any:
    """Run huggingface_hub's snapshot_download with `kwargs`; the library is imported only here."""
    return importlib.import_module("huggingface_hub").snapshot_download(**kwargs)
