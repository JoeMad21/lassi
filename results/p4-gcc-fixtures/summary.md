# P4.5: gcc-native stderr fixture capture

[MEASURED] provenance.json (rx run 20260925-195420-desktop-8r113ei-detached-94368a41-e825) and tests/toolchains/fixtures/gcc/captures.json (the capture tool's manifest, copied byte for byte from the run's manifest.json):

- commit: 94368a41ba63c71a68773daecda83d184d543f0e (94368a4), clean tree (`dirty` false and `snapshot_of` null in both files), run from a detached clean worktree;
- host: alpha01, Ubuntu 22.04.5 LTS; start 2026-09-25 19:54:20, end 19:54:26 (UTC-07:00); rc 0;
- toolchain pin: toolchains/gcc.pin (its text is in provenance.json under pins): g++-12 12.3.0 at /usr/bin/g++-12, FLAGS `-O3 -fopenmp`;
- device: none for the six captures (compile only). Of the four remote tests, one built a test program and ran it natively in the sandbox on alpha01's host CPU, one built a program that fails to compile, and two ran only `g++ --version`; the pass is a correctness check, not a measurement, and no value here is performance.

Command (provenance.json, cmd): `uv sync --quiet`, the clean-tree count, then `uv run python tools/capture_toolchain_fixtures.py --fixtures tests/toolchains/fixtures/gcc` and `LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -p no:cacheprovider -m remote tests/toolchains/test_gcc_native_remote.py`, the run passing only when both return 0. The output printed the clean-tree count 0, one line per case, `4 passed`, and `capture_rc=0 remote_rc=0`.

## The pinned compiler

- The capture's own version check (captures.json): first line "g++-12 (Ubuntu 12.3.0-1ubuntu1~22.04.3) 12.3.0", exit status 0, executable /usr/bin/g++-12, pins {"gcc": "12.3.0"}, environment PATH, LANG=C, and LC_ALL=C only.
- The pin's EXPECT_VERSION came from rx 20260925-192127-exec-fcd6, which ran `/usr/bin/g++-12 --version | head -1` and printed "g++-12 (Ubuntu 12.3.0-1ubuntu1~22.04.3) 12.3.0"; the same exec showed /usr/bin/g++-12 linking to x86_64-linux-gnu-g++-12.

## Cases

Each case built with `/usr/bin/g++-12 -O3 -fopenmp -o main main.cpp`. The diagnostic count is what the parser at 94368a4 gave on alpha01; each equals the length of the list derived by hand from the stderr in tests/toolchains/fixtures/gcc/<case>.json.

| Case | Exit status | Diagnostics | stderr bytes |
| --- | --- | --- | --- |
| gcc_clean | 0 | 0 | 0 |
| gcc_header_error | 1 | 1 | 222 |
| gcc_note_chain | 1 | 5 | 710 |
| gcc_syntax_error | 1 | 2 | 277 |
| gcc_undeclared_identifier | 1 | 1 | 208 |
| gcc_wall_warning | 0 | 1 | 155 |

The six .stderr fixtures are byte-identical to the pulled captures and match the sha256 and size in captures.json; all are plain ASCII.
