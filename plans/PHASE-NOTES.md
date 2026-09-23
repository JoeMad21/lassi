# Phase Notes

Repository-specific hints for planning. The bible stays authoritative; these notes only say what already exists and where the traps are.

## All Phases

- Python 3.10 through uv (`uv sync`, `uv run`); the pipeline control plane targets 3.10 to match the Furiosa SDK environment.
- Anything that needs a compiler, simulator, or SDK runs on alpha01 through `tools/rx.py`. Pure Python logic and unit tests run locally.
- Long builds are `rx job start --big`; plan tasks so local work continues while a job runs.
- Check scratch space with `rx doctor` before any build; storage was reported exhausted on 2026-09-17.
- Keep plans lean: cite bible sections instead of copying them.

## P0 Core

- Already provided: AGENTS.md, docs/BIBLE.md, `.githooks/`, `.github/workflows/text-policy.yml`, `tools/check_text_policy.py`, `tools/policy_canary.py`, `tools/status.py`, `tools/rx.py`, `tools/server/`, and tests under `tests/tools/`. P0 verifies and extends them; it does not rewrite them.
- Text-policy gate evidence: `uv run tools/policy_canary.py local` (both commits must be blocked), then `push`, `status` (the CI run on branch p0-canary must fail), and `cleanup`. The CI half needs OQ-007.
- Compile-only HeCBench needs nvcc or nvc++ on alpha01. Spike first (`rx exec -- 'command -v nvcc nvc++; ls /usr/local /opt'`). If absent, install user-space: the CUDA runfile in toolkit-only mode and the NVHPC tarball, both under `$LASSI_TOOLCHAINS`, each with a pin file. Record the result in the bible (Execution Backends).
- Sandbox executor: spike which unprivileged isolation works on alpha01 without sudo (bubblewrap, `unshare -rn`, Apptainer, `systemd-run --user`). Record the choice with evidence; if none isolates the network, that is an owner-queue item.
- Mock LLM backend: returns the reference target for a bench item; it is the P0 gate driver and the P1 dry-run driver.
- `trial.md` and the Result record follow the bible's Result Record and Readability Standards exactly.
- `lassi/core/files.py` exists since P0.4 with `render_file_blocks`, `FILE_MARKER`, `language_for`, and `fence_for`; P0.5 adds the parser there. Split on "\n" only (not `splitlines`), close a block only on a fence at least as long as the opening one, and add a round trip from `render_file_blocks` through the parser (four-backtick fence, CRLF, form feed). `_check_path` is stricter than "reject absolute and .." (it refuses `./main.cu`); decide explicitly whether model output goes through it.
- Toolchains (since P0.5): `lassi/toolchains` registers `nvcc-sm80` and `nvcpp-cc80`. Their default executables are `nvcc` and `nvc++` from PATH; P0.7 or the runner must pass the pinned `$LASSI_TOOLCHAINS/<name>@<pin>` path as `executable` (Agent Rule 10). The stderr fixtures in `tests/toolchains/fixtures/` are hand-written PLACEHOLDERs; replace them with captures from alpha01 once P0.7 pins the compilers (check the NVC++-F abort suffix and the nvc++ link lines). build() expects a fresh workdir per attempt and refuses model paths that start with '-' or '@' or claim the reserved names `main` and `compile.stderr`. `subprocess_runner` passes the parent environment through, so the runner must clear or record variables that change a compile silently (NVCC_PREPEND_FLAGS, NVCC_APPEND_FLAGS, CPATH, and the like). Faithful LASSI feeds raw compiler output back to the model while the Toolchain contract keeps raw stderr as an attachment only; P1 needs a faithful toggle that reads the `compile.stderr` attachment.
- Pinned compilers (P0.7): `$LASSI_TOOLCHAINS/cuda@12.6.3/bin/nvcc` and `$LASSI_TOOLCHAINS/nvhpc@24.11/Linux_x86_64/24.11/compilers/bin/nvc++`. alpha01 has no GPU driver, so the nvcpp-cc80 build must run with `NVHPC_CUDA_HOME=$LASSI_TOOLCHAINS/cuda@12.6.3` (see toolchains/nvhpc.pin); without it nvc++ looks for CUDA 11.8 and fails.
- CUDA install (P0.7): until OQ-010 is answered, do not rerun the runfile path of `toolchains/cuda.sh` (for example after deleting `cuda@12.6.3`); it writes a transient log to /tmp on the root filesystem.
- Runner (P0.11): import `lassi.llm` when the runner or CLI module loads, never lazily, so the backends register in the real DEFAULT_REGISTRY. A backend declaring `needs_reference` (the mock) must get the bench item's reference target through `with_reference` before each trial. `Sampling.max_tokens` is required but `projects/base.yaml` has none: fail loudly when a recipe gives no value, never pick one. The recipe `model` section holds only backend and id; HTTP settings (base_url, api_key_env, timeout_s) keep their defaults until a recipe key for them is designed, and a recipe may only ever name the key's environment variable.

## P1 Faithful LASSI

- Prefer `third_party/LASSI` as a git submodule pinned at 74b4681: upstream stays untouched and its text stays out of the policy scan. If it must be vendored, pattern hits inside it go to the owner queue; never edit upstream text.
- HeCBench sources are pinned by commit in `assets/bench/`; `entropy` needs `reference.h` from pinned HeCBench (bible, upstream quirks).
- `assets/bench/lassi-hecbench-10.yaml` (since P0.8) pins HeCBench master of 2026-09-23 and lists only `layout`. Before adding the other nine apps, compare the pin with the sources upstream LASSI copied (its 20 `*_main` files) and re-pin if they differ; `uv run tools/fetch_bench.py <manifest>` on alpha01 fetches the listed directories under `$LASSI_SCRATCH/bench/`.
- The notebook replay gate needs recorded responses; capture them as fixtures under `tests/`.
- The mock tags `.cu` blocks as cuda; under the faithful fence quirk that tag becomes "uda". Decide whether mock dry runs go through faithful fence stripping, and record the choice.
- The bible's lassi-df recipe binds `model.backend: furiosa`, which is not registered. furiosa-llm speaks the OpenAI API, so decide between an alias of openai_compat and a separate backend before P3.

## P2 Scoring

- The gate is an owner review: prepare a packet (one full run, score components per trial) and add a review item; set the phase GATE-OWNER and move to P4.

## P4 ttsim Execution

- An existing tt-metal checkout is at `/mnt/nvme10/joseph_ufl/tt-metal`. Record its commit before reuse; build a pinned copy under `$LASSI_TOOLCHAINS` if it does not match the tt-mlir pin.
- ttsim needs `TT_METAL_SIMULATOR`, the SoC descriptor beside the library, slow dispatch, and single chip (bible, Execution Backends). Never open silicon; the gate refuses device commands.
- Check every reference kernel against the Wormhole `unpack_to_dest` issue before it enters a suite.

## P5 IR Levels

- Polygeist and tt-mlir each pin an LLVM; never mix pins in one module (bible, Toolchain Pins). Each LLVM build is a big job of several hours; run one at a time.
- Migrate mlir-corpus-pipeline v0 per the bible's Review Of v0. First spike: compare the alpha01 copy with GitHub commit 3ccd280 (bible question 8).

## P11 Dataflow Dialect

- Before P5 is done, only design notes: op set, type system, lowering sketch, as proposals in the owner queue (bible question 7 is a choice).

## P12 Language Frontends

- rustup and dotnet install without root: rustup with CARGO_HOME and RUSTUP_HOME in scratch (the gate sets both), dotnet through dotnet-install.sh with `--install-dir $LASSI_TOOLCHAINS/dotnet@<ver>`.
- Publish each subset frontend's accepted subset in its docstring and module docs (bible, Frontend Rules).
