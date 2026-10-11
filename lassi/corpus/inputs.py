"""Corpus input through the bench registry (task P5.11; bible Benchmark Suites, split rules; Agent Rule 5).

The corpus reads a bench item only through the bench registry
(lassi.bench.registry, Suite.item), never from a manifest or a source tree
of its own. harvest_item reads an item for purpose harvest, which the
registry refuses for an eval or unassigned item with EvalSplitError, so
harvest takes train items only. verification_item reads an item for purpose
eval, which the registry gives for every split; a record keeps its item's
split, so the round trips of an eval or unassigned item go to a
verification table (lassi.corpus.tables write_verification), which no
pair-set writer or training reader accepts.
"""

from __future__ import annotations

from lassi.bench.registry import Suite, SuiteItem


def harvest_item(suite: Suite, name: str) -> SuiteItem:
    """Return item `name` of `suite` for corpus harvest, through Suite.item(name, purpose="harvest").

    An eval or unassigned item raises EvalSplitError with the registry's own
    words (Agent Rule 5), and an unknown item ValueError.
    """
    return suite.item(name, purpose="harvest")


def verification_item(suite: Suite, name: str) -> SuiteItem:
    """Return item `name` of `suite` for evaluation, through Suite.item(name, purpose="eval").

    The registry gives an item of any split for evaluation; an unknown item
    raises ValueError. The records of an eval or unassigned item's round
    trips go to a verification table, never to the corpus or a pair set.
    """
    return suite.item(name, purpose="eval")
