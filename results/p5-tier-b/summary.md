# P5.3: the Tier B codec's host programs at the joint pin on ttsim

[MEASURED] provenance.json (rx job 20261010-184351-p53-tier-b-e64d):

- commit: e505af7 (e505af741be22360f814ee52a0c28033d9a3d45a), clean tree (provenance.json dirty false, snapshot_of
  null; batch.txt: "dirty=false, changed paths 0"), run in the existing slot desktop-8r113ei-p5-ir, approved by the
  owner in the working session of 2026-10-10 (Agent Rule 17; plans/OWNER-QUEUE.md, OQ-053);
- host: alpha01; start 2026-10-10T18:43:51-07:00, end 18:46:39; job rc 0; the batch reports "done in 161s, status 0" (the job's output log, not in results/);
  scratch free 321.7 GB before and after;
- the codec: https://github.com/JoeMad21/TurboQuant-Tenstorrent.git at 81817eb09076d17b29491c6646b54c6927cac0bd
  (batch.txt), fetched once into the run directory; its license state (license.txt): no license file in the tree,
  and the README names a LICENSE.txt the tree lacks;
- toolchain pins: toolchains/tt-metal.pin (tt-metal 5280a9cfb00998fd49667a29523d03aee905c129) and toolchains/ttsim.pin
  (ttsim v1.3.4, libttsim_wh.so sha256 7b10aa05a5297c4a28f274e39526bfe6c69373661d1b4d4e47c4257e44b79507, re-hashed by
  the batch); provenance.json's pins holds the first 4000 characters of each pin file;
- device: "ttsim v1.3.4 (libttsim_wh.so, a virtual Wormhole) on the host CPU, with tt-metal 5280a9cf: a simulator, not
  silicon" (device.txt). Every pass is provisional until confirmed on silicon; every wall time is simulator wall time,
  sizing only, never performance. No Tenstorrent or RNGD device was opened.

Command (provenance.json, cmd): `bash plans/spikes/p5-tier-b/run.sh` as an rx job with --timeout 7200.
plans/spikes/p5-tier-b.md gives the design, the audits, and the reading of these files.

## Results

"dev0" below is a changed copy of each program, with create_unit_mesh(6) set to create_unit_mesh(0) in the run
directory only (ttsim presents one chip), never the codec's program; "asis" is the program as written (summary.txt,
compare.txt, compare.json "variants").

| Program (programs.tsv) | Stage | Build asis / dev0 | dev0 run: exit, hang, UB, gap, JIT errors, outputs | asis run: exit | dev0 reference checks (compare.txt) |
| --- | --- | --- | --- | --- | --- |
| baseline | WHT, PolarQuant, QJL (baseline) | ok / ok | 0, false, false, none, 0, 6 of 6 | 134 | rotation, indices, centroids, qjl_bits, qjl_rnorm: all PASS |
| mt-rotation | WHT (multi_tile, stage 0) | ok / ok | 0, false, false, none, 0, 1 of 1 | 134 | mt_rotation PASS |
| mt-polarquant | PolarQuant (multi_tile, stage 1) | ok / ok | 0, false, false, none, 0, 1 of 1 | 134 | mt_polarquant PASS |
| mt-inverse-rotation | inverse WHT (multi_tile, stage 2a) | ok / ok | 0, false, false, none, 0, 1 of 1 | 134 | mt_inverse_rotation PASS |
| mt-residual | residual (multi_tile, stage 2b1) | ok / ok | 0, false, false, none, 0, 1 of 1 | 134 | mt_residual PASS |
| mt-qjl-projection | QJL projection (multi_tile, stage 2b2) | ok / ok | 0, false, false, none, 0, 2 of 2 | 134 | mt_qjl_projection PASS |
| mt-rnorm | r_norm (multi_tile, stage 2c) | ok / ok | 0, false, false, none, 0, 1 of 1 | 134 | mt_rnorm PASS |

The reference checks (compare.txt; NumPy 2.2.6; each against the codec's own bound, cited in the spike): 11 run, none
failed. Measured: baseline rotation max_abs 0.00192165; indices match 1.0; centroids max_abs 0.00090341; qjl_bits
match 1.0; qjl_rnorm max_rel 0.00486621; mt_rotation max_abs 0.00381117; mt_polarquant max_abs 0.0015383;
mt_inverse_rotation max_abs 0.00245726; mt_residual max_abs 6.104e-05; mt_qjl_projection overall 0.99902344, strict
1.0, b_max_abs 0.00305176; mt_rnorm max_rel 0.00460621. The kernels' sign table equals the reference's
make_sign_diagonal(128, 0).

Against the codec's committed dumps (summary.txt): the dev0 baseline's dump_input.bin, dump_quant_indices.bin,
dump_quant_centroids.bin, dump_qjl.bin, and dump_output.bin are each the same as the committed file; the asis
baseline's dump_input.bin is the same (compare.json, committed_dumps; the host writes its input before it opens the
device, from reading host/turboquant_host.cpp).

The asis runs (summary.txt): all 7 exited 134 with a tt-fatal terminate on stderr, as the spike predicted for a program
that asks for chip 6 on a one-chip simulator; one output file was written (the baseline's input dump).

unpack_to_dest (summary.txt; PHASE-NOTES P4): the 7 dev0 runs' descriptors set DST_ACCUM_MODE false, with unpack
formats 5 and 10; issue #18's path is possible in none.

## Host safety

- The device probe through the sandbox (device-probe.json) passed: /dev held only the 13 expected entries, the 33
  /sys directories it checks were empty, and /proc/bus/pci was present with 161 entries, readable in the sandbox as a
  known limit that opens nothing (lassi/executors/sandbox.py), recorded and not gated.
- checks.txt: 0 paths this account changed under /tmp, /var/tmp, and /dev/shm; 0 paths changed in the pinned trees;
  0 top entries changed in the default JIT cache root under the shared home; /tmp/tt-metal-cache absent; scratch du
  102,689,076 KiB before and 102,741,484 KiB after; the run directory 51,436 KiB against its 1,048,576 KiB start gate.
- By design the batch deletes no file it did not create (plans/spikes/p5-tier-b/run.sh header); the alpha01 copy
  comparison did not run, since the read-only search found no copy (rx 20261010-184325-exec-4116).
