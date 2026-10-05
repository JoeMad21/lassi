# P4.13: the Tier A suite's remote tests

[MEASURED] provenance.json (rx run 20261005-122628-desktop-8r113ei-detached-5e72f4e8-56d0):

- commit: 5e72f4e (5e72f4e861c16235d17a206a07e303bab2003687), clean tree: `dirty` false and `snapshot_of` null in provenance.json, and the slot's `git status --short | wc -l` printed 0 (output.log, first line); run from the detached clean worktree, slot desktop-8r113ei-detached-5e72f4e8;
- host: alpha01, Ubuntu 22.04.5 LTS, Python 3.10.12; start 2026-10-05T12:26:28-07:00, end 2026-10-05T12:27:38-07:00; rc 0;
- pins: toolchains/tt-metal.pin (tt-metal 5280a9cf, PREFIX_NAME tt-metal@5280a9cf, EXECUTABLE /usr/bin/clang++-20), toolchains/ttsim.pin, and toolchains/gcc.pin; provenance.json's pins holds the first 4000 characters of each pin file, as the gate records it;
- scratch root free: 449.1 GB before and 440.8 GB after (the cause of the difference was not examined); root filesystem free 244.3 GB (provenance.json);
- devices: ttsim, a simulator, for every TT run, with tt-metal 5280a9cf, and the host CPU for every native run. Every ttsim pass is provisional until a silicon check (ttsim Facts), and no simulator time here is performance.

Command (provenance.json, cmd):

```
uv sync --quiet && git status --short | wc -l && uv run tools/fetch_bench.py assets/bench/tt-pairs-v0.yaml && LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -p no:cacheprovider -m remote tests/bench/test_tt_pairs_remote.py
```

## Results

- fetch_bench printed that $LASSI_SCRATCH/bench/tt-pairs-v0@5280a9cfb00998fd49667a29523d03aee905c129 already holds tt-pairs-v0 at that commit and exited 0, so every listed kernel was present with its manifest sha256 (output.log, second line).
- With LASSI_REQUIRE_SANDBOX=1, so that none could skip, pytest printed `5 passed in 67.66s (0:01:07)` (output.log). The time is pytest's own line, not a measurement.
- The five tests are test_an_items_references_run_clean_agree_and_pass_the_guard for loopback, eltwise_binary, eltwise_sfpu, matmul_single_core, and matmul_multi_core. What each checks is stated in tests/bench/test_tt_pairs_remote.py's module docstring: the fetch; the TT reference built with ttmetal-host and run on ttsim with exit status 0, no hang, no undefined behavior, no gap, and no jit-stage error; the C++ counterpart built with gcc-native and run natively with exit status 0 and no hang; every output passing the binary_io statistics under the item's declared tolerance (loopback max_abs 0, eltwise_binary max_abs 0.01, eltwise_sfpu max_abs 0.05, both matmuls pcc 0.97), with exactly the declared outputs on both sides; and the CPU -> TT guard reading the TT host program host_compute False, loopback with exactly the data-movement-only tag and the compute items with no guard diagnostic.
- The tests assert the tolerances but print no statistic, so this run records no agreement value. The pass is a correctness check of the pairs at the pin, not a measurement.
