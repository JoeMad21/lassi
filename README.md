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

## Terminal Graphics (planned)

`lassi` commands will be able to show a LASSI-DF banner and live tables of inference and training in the terminal. Graphics are optional: the first interactive command asks whether to show them and offers to save the answer as your preset, `lassi settings graphics on|off` changes it, and `--graphics on|off` or `LASSI_GRAPHICS` overrides it for one command. Nothing is asked without an interactive terminal, and graphics never change records or results. Design: `docs/BIBLE.md`, Readability Standards, Terminal Presentation.

## Devices And Frameworks (planned)

LASSI is built to serve its reproduction arms on Furiosa RNGD cards (P3) and to run Tenstorrent code on the ttsim simulator (P4). Phase P17 prepares the same pipeline for standard ML frameworks on other devices: models served by vLLM, SGLang, llama.cpp, or Ollama, or run in-process by Hugging Face Transformers on PyTorch; candidate programs run on NVIDIA and AMD GPUs; and training on PyTorch with standard libraries. Recipes will name every device, and each run will record it. The GPU half waits for a GPU host. Plan: `plans/p17-portable.md`; design: `docs/BIBLE.md`, Build Roadmap.
