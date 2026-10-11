"""The provenance hash chain of a corpus entry (task P5.11).

This is v0's manifest with parent-child hashes moved into lassi/corpus/
(bible Review Of v0, the Keep row; Corpus Pipeline, Record: provenance
hashes; plans/spikes/p5-v0-corpus.md, Results 5, the migration map's
manifest row). v0 recorded each snapshot's sha256 beside its parent's
(tools/parse_snapshots.py:71 and :82-88 at 3ccd280). Here a record's chain
starts with the source (lassi.corpus.record SOURCE_STEP) and adds one link
per step's output, source to raised IR to normalized IR to lowered program,
so a record names the chain that produced it: each link holds its step's
name, the sha256 of that output's bytes, and its parent's sha256 (None for
the first link).

Every hash is over bytes. v0 hashed source text decoded with replacement
characters (tools/parse_snapshots.py:92), which differs from the file's
sha256 for a source that is not UTF-8; hash_chain takes bytes only and
refuses a str, so no caller hashes decoded text.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, kw_only=True)
class HashLink:
    """One output in a provenance chain: its step, the sha256 of its bytes, and its parent's sha256.

    `step` is a non-empty string; `sha256` and `parent` are 64 lowercase
    hex characters, and `parent` is None for the first link. Any other value
    raises ValueError.
    """

    step: str
    sha256: str
    parent: str | None

    def __post_init__(self) -> None:
        """Check the step and both hashes."""
        if not isinstance(self.step, str) or not self.step:
            raise ValueError(f"HashLink.step must be a non-empty string, got {self.step!r}")
        if not isinstance(self.sha256, str) or not _SHA256.fullmatch(self.sha256):
            raise ValueError(f"HashLink.sha256 must be 64 lowercase hex characters, got {self.sha256!r}")
        if self.parent is not None and not (isinstance(self.parent, str) and _SHA256.fullmatch(self.parent)):
            raise ValueError(f"HashLink.parent must be 64 lowercase hex characters or None, got {self.parent!r}")


def hash_chain(outputs: Iterable[tuple[str, bytes]]) -> tuple[HashLink, ...]:
    """Return one HashLink per (step, bytes) output, in order, each the child of the one before.

    Each link's sha256 is hashlib's sha256 of that output's bytes and its
    parent the previous link's sha256; the first link's parent is None. An
    output that is not bytes, a str among them, raises TypeError naming its
    step; an empty step raises ValueError.
    """
    links: list[HashLink] = []
    parent: str | None = None
    for step, data in outputs:
        if not isinstance(data, bytes):
            raise TypeError(f"the {step!r} output must be bytes, got {type(data).__name__}; hash bytes, never text")
        link = HashLink(step=step, sha256=hashlib.sha256(data).hexdigest(), parent=parent)
        links.append(link)
        parent = link.sha256
    return tuple(links)


def check_chain(links: Sequence[HashLink]) -> None:
    """Return when `links` form one chain; raise ValueError otherwise.

    One chain: at least one link, every item a HashLink, the first link's
    parent None, and each later link's parent the sha256 of the link before
    it. The message names the step of the first link that breaks the rule.
    """
    if not links:
        raise ValueError("the provenance chain holds no link")
    previous: str | None = None
    for index, link in enumerate(links):
        if not isinstance(link, HashLink):
            raise ValueError(f"provenance link {index} must be a HashLink, got {link!r}")
        if link.parent != previous:
            wanted = "no parent, as the first link" if previous is None else f"parent {previous}, the link before it"
            raise ValueError(
                f"the provenance chain breaks at step {link.step!r}: it needs {wanted}, got parent {link.parent}"
            )
        previous = link.sha256
