# P0.19 CUDA 12.6.3 from the redistributable archives

Source: `provenance.json` in this directory, the rx run record. Every value
below comes from it, from the run's output, or from comparing the run's
fixture capture with `tests/toolchains/fixtures/`.

## Run

- rx id: `20260923-220925-desktop-8r113ei-p0-core-6db9` (state done, rc 0),
  host alpha01 (Ubuntu 22.04.5 LTS), 2026-09-23.
- Commit `7d8d3d5` on branch p0-core, clean tree: `dirty` is false, and
  `git status --short | wc -l` printed 0 in the slot.
- Command: record the start time, `uv sync --quiet`, check the tree,
  `bash toolchains/cuda.sh`, `uv run python tools/capture_toolchain_fixtures.py`,
  `LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -m remote -rfEs`, then
  `find /tmp /var/tmp -xdev -user $(id -un) -newermt <start>`.
- Pin: toolchains/cuda.pin at 7d8d3d5. It holds four archives from
  redistrib_12.6.3.json (manifest sha256 9c598598...): cuda_nvcc 12.6.85,
  cuda_cudart 12.6.77, cuda_cccl 12.6.77, and cuda_cuobjdump 12.6.77, each
  with its sha256. nvhpc 24.11 is unchanged.
- Device: none. Nothing ran on an accelerator; the compiles ran on the host
  CPU.

## Result

- `bash toolchains/cuda.sh` found the pinned toolkit already installed, from
  the exploratory install job recorded in plans/spikes/p0-cuda-redist.md. It
  printed nvcc's banner, ending "Cuda compilation tools, release 12.6,
  V12.6.85".
- The fixture capture's manifest records commit 7d8d3d5, dirty false, and
  snapshot_of null. All 16 scenarios kept their exit status and diagnostic
  count. The 12 byte-stable scenarios are byte-identical to
  tests/toolchains/fixtures/.
- Remote suite: 63 passed, 2483 deselected, pytest status 0. That covers the
  P0.16 sandbox, the P0.20 compile sandbox with the real pinned nvcc and
  nvc++, and the fixture recapture test.
- The root filesystem check printed no file.

## Left for the owner

The runfile tree from P0.7 was moved aside, not deleted, to
`$LASSI_TOOLCHAINS/cuda@12.6.3.runfile-20260923-214325` (about 7.0G). Agents
may not delete it; plans/OWNER-QUEUE.md asks the owner to remove it.
