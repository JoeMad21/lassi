# P4.11: the ttsim executor's remote tests and smoke runs

[MEASURED] provenance.json (rx job 20260930-172230-p411-evidence-92e0):

- commit: b9fc0d3 (b9fc0d3b3cd7b98c6aefaea9224686d8af69df13), clean tree (provenance.json dirty false, snapshot_of
  null; commit.txt: dirty_paths 0), run in the detached clean worktree slot desktop-8r113ei-detached-b9fc0d3b;
- host: alpha01, Ubuntu 22.04.5 LTS, nproc 256; start 2026-09-30T17:22:30-07:00, end 17:26:25; job rc 0;
  scratch free 557.4 GB before and after;
- toolchain pins: toolchains/tt-metal.pin (tt-metal 5280a9cfb00998fd49667a29523d03aee905c129, host compiler
  /usr/bin/clang++-20, Ubuntu clang 20.1.8) and toolchains/ttsim.pin (ttsim v1.3.4, libttsim_wh.so sha256
  7b10aa05a5297c4a28f274e39526bfe6c69373661d1b4d4e47c4257e44b79507, SoC descriptor sha256
  24fd3dfae80435a7d9113e255d6d6af9cdf83f6ae3b1c385e1d48a24795ccf49). provenance.json's pins holds the first 4000
  characters of each pin file; tt-metal.pin is 7668 bytes, so its EXPECT_VERSION is in each summary-*.json pins,
  not in provenance.json, and each summary-*.json records the pairs the executor read;
- device: "ttsim v1.3.4 (libttsim_wh.so, a virtual Wormhole) on the host CPU, with tt-metal 5280a9cf: a
  simulator, not silicon" (each summary's device). Every wall time below is simulator wall time: exploratory,
  sizing only, never performance. Every pass is provisional until confirmed on silicon (bible, ttsim Facts).

Command (provenance.json, cmd): with LASSI_REQUIRE_SANDBOX=1, `uv run pytest -q -p no:cacheprovider -m remote
tests/executors/test_ttsim_remote.py tests/executors/test_sandbox_remote.py`, then
`uv run python tools/ttsim_smoke.py metal_example_<example>` for each of the six examples, P4.G's form of the name.
Each smoke's own summary.json was copied here as summary-<example>.json and its output as smoke-<example>.txt.

## Remote tests

remote-tests.txt: 50 passed in 153.82 s, remote_rc=0. They include the ttsim executor's four remote tests
(tests/executors/test_ttsim_remote.py: a program the executor runs sees no Tenstorrent device node; the gate's
example runs clean; a kernel seeded with a 4-byte load from an address 2 mod 4 reads as undefined behavior with a
NonContractualBehavior diagnostic; a seeded hang under a 30 s limit reads as a hang with a Watcher note naming the
kernel) and the sandbox's remote tests under the new CPU-time cap, among them a program keeping its CPUs busy until
the wall limit, which reads as a hang.

## Smoke runs

Each through the ttsim executor in the sandbox with Limits cpus 16, memory 8192 MiB, wall 120 s, tasks 256, and a
256 MiB workdir disk cap (summary limits); each built with 0 build diagnostics.

| Example | Verdict | Exit | Hang | sim_ub | sim_gap | Own success line | PCC | Wall (s, simulator) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| add_2_integers_in_riscv | pass | 0 | false | false | none | Success: Result is 21 | | 1.63 |
| loopback | pass | 0 | false | false | none | Test Passed | | 1.68 |
| eltwise_binary | pass | 0 | false | false | none | Test Passed | | 1.86 |
| eltwise_sfpu | pass | 0 | false | false | none | Test Passed | | 1.73 |
| matmul_single_core | pass | 0 | false | false | none | Test Passed | 0.98210174 | 3.95 |
| matmul_multi_core | pass | 0 | false | false | none | Test Passed | 0.9999929 | 3.19 |

Every summary has findings [] and reasons []. The two PCC values exceed the examples' bound of 0.97. Each run is
one run. Both PCC values differ from task P4.9's (results/p4-ttsim-runtime): matmul_single_core's from 0.9806516
and matmul_multi_core's from 0.99999344; the cause was not examined. The smoke trees stay on alpha01 under
$LASSI_RUNS_ROOT/ttsim-smoke/ (the paths are in smoke-<example>.txt).
