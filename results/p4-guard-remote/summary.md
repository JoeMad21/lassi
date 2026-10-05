# P4.12: the CPU -> TT guard's remote tests

[MEASURED] provenance.json (rx run 20261004-220812-desktop-8r113ei-detached-9e8dc504-a6b1):

- commit: 9e8dc50 (9e8dc504e2eb1e78e96194243946b22be74e1c34), clean tree: `dirty` false and `snapshot_of` null in provenance.json, and the slot's `git status --short | wc -l` printed 0 (output.log, first line); run from the detached clean worktree, slot desktop-8r113ei-detached-9e8dc504;
- host: alpha01, Ubuntu 22.04.5 LTS, Python 3.10.12; start 2026-10-04T22:08:12-07:00, end 2026-10-04T22:08:45-07:00; rc 0;
- toolchain pin: toolchains/tt-metal.pin, VERSION 5280a9cf (tt-metal 5280a9cfb00998fd49667a29523d03aee905c129), PREFIX_NAME tt-metal@5280a9cf, EXECUTABLE /usr/bin/clang++-20; provenance.json's pins holds the first 4000 characters of each pin file, as the gate records it;
- scratch root free: 430.6 GB before and 430.6 GB after; root filesystem free 247.5 GB (provenance.json);
- device: none. The tests read host text and compile programs only; no built program ran, and neither the kernel JIT nor ttsim started.

Command (provenance.json, cmd):

```
uv sync --quiet && git status --short | wc -l && LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -p no:cacheprovider -m remote tests/toolchains/test_ttmetal_guard_remote.py
```

## Results

With LASSI_REQUIRE_SANDBOX=1, so that none could skip, pytest printed `10 passed in 32.22s` (output.log). The time is pytest's own line, not a measurement. What each test checks is stated in tests/toolchains/test_ttmetal_guard_remote.py's module docstring:

| Test | Cases |
| --- | --- |
| test_an_upstream_example_reads_as_the_pinned_api_says | add_2_integers_in_riscv, eltwise_binary, eltwise_sfpu, loopback, matmul_multi_core, matmul_single_core |
| test_an_acceptance_program_builds_at_the_pin | clean_offload, host_loop_output, data_movement_only, compute_beside_host_loop |

- The six pinned examples' host sources, read from the tree by tools/ttsim_smoke.py example_sources, went through read_host_compute: add_2_integers_in_riscv and loopback carry the data-movement-only tag exactly once, the four compute examples carry none, none reads host_compute true, and none has an unreadable-text or reader-failed note. This ties the identifier sets (plans/spikes/p4-guard-identifiers.md) to the pinned examples.
- The four acceptance programs under tests/toolchains/fixtures/host_compute/ built with ttmetal-host in the compile sandbox, with assets/harness/c/lassi_io.h as the harness file: no error diagnostic, and the artifact `main` in place. Their kernel placeholders were placed, not compiled.
- The pass is a correctness check of the guard and the fixtures at the pin, not a measurement.
