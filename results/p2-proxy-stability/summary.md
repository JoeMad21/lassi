# Stability of the reference output under the CPU proxy (P2.5)

[MEASURED] provenance.json:
- run: rx job 20260924-133017-p2-proxy-3a3a, run directory p2-proxy-stability on alpha01;
- commit: 207e4dd (clean tree);
- recipe: tests/fixtures/recipes/p2-proxy-mock.yaml, hash 6ca943f75972;
- date: 2026-09-24, 20:30 to 20:39 UTC;
- toolchain pins: nvhpc 24.11 and cuda 12.6.3;
- device: the host CPU through the sandboxed native executor, 16 CPUs per run. It uses the -mp=multicore proxy (Harness Contract), which checks outputs, never runtime.
- The job log ends with `lassi_rc=0`.

Result, for 10 items (CUDA to OpenMP) with 3 trials each; each trial ran the OpenMP reference once and a mock candidate once:
- All 30 trials reached S5.
- For every item, the three masked reference stdouts agree with each other and with the three masked candidate stdouts.
- Alignment means are 1.0 for 9 items. dense-embedding scores 0.0 because its reference prints FAIL under the proxy (one OpenMP team), so passfail cannot pass. That is a limit of the proxy, not an instability.
- The mock candidate is the reference program again, so these passes check the scoring path and are never model results.

Details and method: plans/spikes/p2-proxy-stability.md. No wall time is reported; the proxy checks outputs, never runtime.
