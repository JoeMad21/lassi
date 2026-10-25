"""A strict reader for the judges' small YAML files: the rubric schemas and the model family table (task P15.1).

read_yaml(path, error) reads one file once, which must be plain ASCII
(printable characters, line feeds, and tabs; lassi.core.interfaces
is_plain_ascii), and parses it with a safe loader that refuses a repeated
mapping key, so a key (a model id, a schema key) written twice is an
error, never a silent last-wins. It returns the document and the bytes it
read, so a caller hashes exactly what it parsed. Every refusal is the
caller's error class, a ValueError, naming the file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from lassi.core.interfaces import is_plain_ascii


class _StrictLoader(yaml.SafeLoader):
    """A safe YAML loader that refuses a repeated mapping key."""

    def construct_mapping(self, node: yaml.Node, deep: bool = False) -> dict[object, object]:
        """Construct a mapping as SafeLoader does; raise ConstructorError when any key appears twice."""
        mapping = super().construct_mapping(node, deep=deep)
        seen: set[object] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                problem = f"found a repeated key {key!r}"
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark, problem, key_node.start_mark
                )
            seen.add(key)
        return mapping


def read_yaml(path: Path, error: type[ValueError]) -> tuple[Any, bytes]:
    """Return the YAML document of the plain ASCII file at `path` and its bytes; raise `error` naming the file.

    A file that cannot be read, is not plain ASCII, or is not one YAML
    document with distinct keys in every mapping is refused.
    """
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise error(f"{path} cannot be read ({exc.strerror or exc})") from None
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError:
        text = None
    if text is None or not is_plain_ascii(text):
        raise error(f"{path} is not plain ASCII (printable characters, line feeds, and tabs)")
    try:
        document = yaml.load(text, Loader=_StrictLoader)  # a SafeLoader subclass: no arbitrary objects
    except (yaml.YAMLError, TypeError) as exc:
        problem = " ".join(str(exc).split())
        raise error(f"{path} is not valid YAML with distinct keys: {problem}") from None
    return document, data
