# CPU run check of the OpenMP references under the multicore proxy

[MEASURED] provenance.json:
- run: rx 20260924-083314-demo-proxy-d805;
- commit: 7c5cad7 (clean tree);
- host: alpha01, 2026-09-24;
- toolchain pins: nvhpc.pin (24.11) and cuda.pin (12.6.3), recorded in provenance.json;
- device: the host CPU. This is the bible's CUDA to OpenMP proxy without a GPU (Harness Contract): `nvc++ -mp=multicore`, so target regions run on the host.

Command: `LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -s -m remote -p no:cacheprovider tests/bench/test_multicore_proxy.py`, which printed 1 passed and pytest_rc=0.

Result, for the 10 OpenMP reference mains of lassi-hecbench-10 at HeCBench 692cba3:
- 10 of 10 compiled with Toolchain nvcpp-multicore in the compile sandbox.
- 10 of 10 exited 0 through the native executor in the sandbox. The limits were 180 s wall, 32768 MB, and 16 CPUs, and OMP_NUM_THREADS was 16.
- Of the 7 apps whose OpenMP program prints PASS or FAIL, 6 printed PASS: atomicCost, colorwheel, entropy, jacobi, layout, and matrix-rotate.
- dense-embedding exited 0 but printed no PASS. The proxy creates one OpenMP team, and the program assigns work by team number, so only the first batch is written. See plans/spikes/demo-multicore-proxy.md (DEMO.2 section, exploratory diagnosis).

Limits: the proxy checks outputs, never runtime. The wall_s values in the log are not performance numbers, and none is reported here.
