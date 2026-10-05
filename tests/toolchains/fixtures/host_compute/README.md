# CPU -> TT guard fixtures

Each folder here is one hand-written TT host program for the CPU -> TT
host-compute guard (task P4.12; lassi.toolchains.ttmetal_guard): `main.cpp`,
the host program, and `kernels/*.cpp`, placeholders for the kernels the host
program names. Every file is SYNTHETIC: it starts with the line
`// SYNTHETIC: written for tests/toolchains/test_ttmetal_guard.py; not tt-metal source`,
and it uses only tt-metal API names (OQ-018 practice; plans/p4-ttsim.md), with
no upstream text.

The guard reads only host text: `main.cpp` and the files it includes. No
host file includes a kernel placeholder, so the guard never reads them, and
the host compiler never builds them (a C++ file under `kernels/` is a kernel
source). They exist so that the programs build as the toolchain builds a
model's reply.

The four programs are the acceptance cases of plans/p4-ttsim.md, P4.12. Each
reads its inputs with `lassi_io_read` and writes its output with
`lassi_io_write` from the harness header `assets/harness/c/lassi_io.h`, which
a build gets as a harness file. Each must build with `ttmetal-host` at the
pinned tt-metal (the remote test
tests/toolchains/test_ttmetal_guard_remote.py builds them in the compile
sandbox; nothing runs them). The device buffers carry bf16: each program
converts its inputs with a small helper, to_bf16, and the three that convert
the device result back use a second, from_bf16; host_loop_output reads the
result back but never converts it, since its output comes from the inputs.

| Case | What the host program does | Expected reading |
| --- | --- | --- |
| `clean_offload/` | Reads a and b, stages them as bf16, creates two data-movement kernels and one compute kernel (ComputeConfig), launches, reads the result back into `res`, converts it into `c`, and writes `c` | host_compute False, no diagnostic |
| `host_loop_output/` | The same kernels and launch, and the read back, but `c[i] = af[i] + bf[i]` from the input data before the write | host_compute True, one parse-stage note `guard-host-compute` at the `lassi_io_write` line, naming the first `lassi_io_read` line and the write line |
| `data_movement_only/` | Reads a, creates one data-movement copy kernel and no compute kernel config, reads the result back into `res`, converts it into `c`, and writes `c` | host_compute False, one parse-stage warning `guard-data-movement-only` at the `CreateKernel` line |
| `compute_beside_host_loop/` | `clean_offload` plus a host golden `golden[i] = af[i] + bf[i]`, a check of `c` against it, and a printf | host_compute False, no diagnostic |

The rule cases (rules K, T, and H, the limits, and robustness) are inline
strings in tests/toolchains/test_ttmetal_guard.py and
tests/toolchains/test_cxx_scan.py. No value in these files is a measurement:
nothing here ran on a device or on a simulator.
