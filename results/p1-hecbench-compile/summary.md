# P1.2 reference compile of lassi-hecbench-10

[MEASURED] provenance.json:
- run: rx 20260924-002513-desktop-8r113ei-p1-faithful-e7ba;
- commit: a815c45 (clean tree);
- host: alpha01, 2026-09-24;
- toolchain pins: cuda.pin (12.6.3) and nvhpc.pin (24.11), both recorded in provenance.json;
- device: none (compile only; no GPU is used or claimed).

Command: `LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -s -m remote -p no:cacheprovider tests/bench/test_hecbench10.py`. It printed 1 passed and pytest_rc=0.

Result:
- tools/fetch_bench.py found `/mnt/nvme10/joseph_ufl/bench/lassi-hecbench-10@692cba32c5744f6ef024cca59f65e9488edba8bf` already at the pin. An earlier exploratory run from a dirty tree had fetched it. The "already holds" check compares every listed file's sha256 with the manifest.
- In the compile sandbox, 20 of 20 reference programs compiled:
  - the 10 CUDA mains with nvcc-sm80;
  - the 10 OpenMP mains with nvcpp-cc80;
  - entropy with reference.h as a harness file in both languages.
- The test also checks each model-facing file's git blob id against upstream LASSI's `*_main` file (Decision Log, 2026-09-24).

Limits: this is compile-only. No program was run, and nothing here is a performance number.
