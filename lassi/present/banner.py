"""The LASSI-DF banner that starts every `lassi` command when graphics are on (bible, Terminal Presentation).

BANNER is plain ASCII art, at most 80 columns wide, with a caption line that
names LASSI-DF; it holds no escape sequence, so print_banner writes the same
text to a terminal, a pipe, or a file.
"""

from __future__ import annotations

from typing import TextIO

BANNER = "\n".join(
    (
        r" _        _    ____ ____ ___      ____  _____",
        r"| |      / \  / ___/ ___|_ _|    |  _ \|  ___|",
        r"| |     / _ \ \___ \___ \| |_____| | | | |_",
        r"| |___ / ___ \ ___) |__) | |_____| |_| |  _|",
        r"|_____/_/   \_\____/____/___|    |____/|_|",
        "",
        "LASSI-DF: languages <-> dataflow accelerators through an MLIR hub",
    )
)


def print_banner(stream: TextIO) -> None:
    """Write BANNER to `stream` once, followed by a blank line, and flush the stream."""
    stream.write(BANNER + "\n\n")
    stream.flush()
