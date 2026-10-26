# Polygeist (cgeist) install on alpha01 (task P5.5)

Provenance: provenance.json, rx 20261026-002209-p5-polygeist-6360 (a big job; commit
00bd90f65580b693b5dd5af2dece292dbc5e1206, dirty false; alpha01; 2026-10-26 00:22:09 to 00:30:43 on alpha01's clock,
03:22:09 to 03:30:43 EDT; rc 0; scratch free 131.5 GB before, 127.3 GB after). The command was
`LASSI_ALLOWED_ROOT=/mnt/nvme10/joseph_ufl LASSI_STOP_GIB=345 bash toolchains/polygeist.sh`. Pins:
toolchains/polygeist.pin at 00bd90f; provenance.json's pins holds the first 4000 characters of each pin file, as the
gate records it. The stop line of 345 GiB is this job's own bound under the owner's lift of the scratch cap (OQ-074,
2026-10-23, recorded on p9-ee, p15-judges, and p16-adversarial), which names no figure; du of the scratch root was
293.1 GiB at the start. Values below come from the job's printed log (rx pull's output.log for this id, 5224 lines,
not committed), except TARGETS, read from toolchains/polygeist.pin at 00bd90f.

Device: none. The job built on alpha01's CPU: the build ran its TableGen code generators, and the script ran the five
tools with --version; nothing opened a device.

Host: the install runs on alpha01 under the owner's direction of 2026-10-26 in the working session ("If they don't
require the hardware accelerators, you can build them now."), not the local route plans/p5-ir.md first named.

## Results [MEASURED, alpha01, rx 20261026-002209-p5-polygeist-6360, 2026-10-26]

- Installed polygeist@77c04bb2 at Polygeist 77c04bb2a7a2406ca9480bcc9e729b07d2c8d077 in
  /mnt/nvme10/joseph_ufl/toolchains/polygeist@77c04bb2, with llvm-project at its gitlink
  26eb4285b56edd8c897642078d91f16ff0fd3472, both unchanged after the build.
- TARGETS cgeist, polygeist-opt, mlir-opt, mlir-translate, clang, and LLVMFrontendOpenACC (OQ-063, read as option
  (a)); a fresh configure and build of 4311 steps with 32 jobs; every pinned setting found in CMakeCache.txt.
- Host tools: /usr/local/bin/cmake 3.31.6, /usr/bin/ninja 1.10.1, clang-17 and clang++-17 Ubuntu 17.0.6,
  /usr/bin/python3.10 3.10.12, git 2.34.1. The log holds 67 compiler warning lines and no error.
- --version: cgeist, polygeist-opt, mlir-opt, and mlir-translate print "LLVM version 18.0.0git"; clang prints
  "clang version 18.0.0 (https://github.com/llvm/llvm-project.git 26eb4285b56edd8c897642078d91f16ff0fd3472)", and its
  builtin headers are in build/lib/clang/18/include.
- The configure fetched no source (lassi-fetched-sources.txt reads `none`, as plans/spikes/p5-mlir-pins.md Results 5
  expects with Polymer off); the line is copied into toolchains/polygeist-fetched-sources.txt.
- Space: tt-mlir's install was complete, so the plan was this stack alone, 20080231 KiB (PROJECTED) against 307358540
  KiB used under the scratch root and a stop line of 361758720 KiB. The install takes 4159440 KiB (du); the scratch
  root read 311518444 KiB after.
- Wall time from the script's stamps: 07:22:09Z to 07:30:35Z, 8 minutes 26 seconds.
- The root-filesystem check passed: no file this account owns under /tmp, /var/tmp, or /opt changed during the job;
  the check (find -xdev -user) skipped 43 unreadable paths (job log).

## Not measured

- Peak memory (the script does not read it).
- The remote tests of P5.5's Accept (a small C and C++ program raised through cgeist, lowered, built, and run in the
  sandbox) and the exploratory read of which Tier A references cgeist raises; both follow.
