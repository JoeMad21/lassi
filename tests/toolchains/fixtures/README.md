# Toolchain stderr fixtures

Status: PLACEHOLDER. Every `.stderr` file here is hand-written, not captured
from a compiler run. Each one follows the stderr format the P0.5 parsers
assume for the CUDA 12 EDG front end, host GCC through `-Xcompiler -Wall`, the
nvcc driver, ptxas, and the linker for nvcc; and for the NVHPC EDG front end,
the NVC++ backend, `-Minfo` output, and the linker for nvc++. These formats are
unverified assumptions until real output is captured; the version token in the
NVC++ summary lines is PLACEHOLDER, not a real release. Message wording is
illustrative; the line formats are what the tests rely on.
GCC quotes are plain ASCII, as GCC prints them under the C locale.

Each file is raw stderr for one scenario, with no comment or header, stored
with LF line endings. The sources that the EDG echo and caret lines point into
live in tests/toolchains/test_diagnostics.py, next to the exact Diagnostic
list each file must parse into. No value in these files is a measurement.

| File | Scenario |
| --- | --- |
| nvcc_undefined_identifier.stderr | EDG error, source echo and caret, error summary line |
| nvcc_warning_177.stderr | EDG warning #177-D in a header, echo and caret, Remark line |
| nvcc_host_gcc_warning.stderr | Host GCC warning with a `[-W...]` flag, `In function` line, GCC echo |
| nvcc_fatal.stderr | nvcc driver fatal error |
| nvcc_ptxas_error.stderr | ptxas error |
| nvcc_linker_error.stderr | Linker undefined reference and the collect2 line |
| nvcc_clean.stderr | Clean build (empty stderr) |
| nvcpp_edg_error.stderr | EDG error, source echo and caret, error summary line |
| nvcpp_edg_warning.stderr | EDG warning with a `[tag]`, Remark line, then `-Minfo` output |
| nvcpp_backend_error.stderr | NVC++-S backend error with file and line, `-Minfo` lines, summary line |
| nvcpp_fatal_abort.stderr | NVC++-F abort without file, summary line |
| nvcpp_minfo_clean.stderr | `-Minfo` output only, which yields no diagnostics |
| nvcpp_linker_error.stderr | Linker undefined reference through the nvc++ link step |
| nvcpp_missing_include.stderr | EDG catastrophic error for a missing header, echo and caret, summary lines |

These files will be replaced by stderr captured on the build host (`rx run`)
once P0.7 pins the CUDA and NVHPC compilers, as the plan's P0.5 Remote note
says; the expected Diagnostic lists then change with them.
