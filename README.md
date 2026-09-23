# LASSI

One codebase that reproduces LASSI, reproduces LASSI-EE, and builds LASSI-DF: LLM-driven, self-correcting code translation between major languages and dataflow accelerators through an MLIR hub.

- Project reference: `docs/BIBLE.md`
- Contributor and agent instructions: `AGENTS.md`
- State of work: `plans/STATUS.md`

## Setup

1. Install git, uv, and ssh. On Windows also set `git config --global core.autocrlf false` and `git config --global core.longpaths true`.
2. `git config core.hooksPath .githooks`
3. Place the owner's pattern list at `~/.config/lassi/text-policy.txt`.
4. `uv sync` and `uv run tools/check_setup.py`.

Remote work on the build host goes through `uv run tools/rx.py` (see AGENTS.md, Remote Execution).
