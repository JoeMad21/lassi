# Demo run: CUDA to OpenMP with the generated code run on the CPU

Provenance (provenance.json, from the job):
- run: rx job 20260924-093127-demo-rngd-cpu-1ff6, run directory demo-rngd-cpu-1 on alpha01;
- commit: 2f052e6 (clean tree);
- recipe: projects/lassi-demo/rngd-cpu.yaml;
- date: 2026-09-24, 16:31 to 17:19 UTC;
- exit status: the job log ends with `lassi_rc=0`, the status of `lassi run` itself;
- toolchain pins: nvhpc 24.11 (the nvcpp-multicore proxy preset) and cuda 12.6.3;
- device: the host CPU, through the sandboxed native executor. This is the bible's CUDA to OpenMP proxy (nvc++ -mp=multicore), which checks outputs, never runtime.
- The model: the same demo model and server as results/demo-rngd-run, rx job 20260924-082436-rngd-serve-9c91.

The recipe is not faithful: it builds with the proxy instead of upstream's GPU offload flags, and it caps the correction loop at 5. The correction prompts still carry upstream's compiler text (-mp=gpu).

[MEASURED] results, 10 trials (10 apps, CUDA to OpenMP, one run each):
- 5 of 10 reached a compiling translation: bsearch, entropy, layout, matrix-rotate, and randomAccess.
- 3 of 10 ran clean (exit 0), each scored against the reference program's run by stdout_mask with passfail:
  - bsearch (first try) implemented one of the reference's four search kernels, so its output differs.
  - entropy (after 2 corrections) printed PASS but left out the reference's timing lines, so its output differs.
  - matrix-rotate (first try) printed FAIL, its own validator's verdict, so its output differs.
- None of the 10 matched the reference output exactly.
- layout compiled, then its runs ended by signal (exit statuses 139 and 134) until the cap.
- randomAccess compiled on attempt 2, then every run exited with status 137 (killed) until the cap.
- 7 of 10 stopped at the correction cap of 5.

Limits:
- This is one sample of one demo model.
- Run times are not reported; the proxy checks outputs, never runtime.
