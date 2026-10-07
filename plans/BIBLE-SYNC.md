# Bible edits awaiting the master copy

AGENTS.md, Authority: when the master copy is unreachable, each mirror edit is appended here, exact, for the owner. Apply them to the master in order, then clear an entry once it is mirrored.

## 2026-10-07: the owner's directions on server access and the MI300Xs (mirror at master revision 266)

1. Agent Rules, rule 17

Old:

````
17. No model inference on alpha01 without the owner's approval in the working session. File uploads and compiler work need none; serving a model (RNGD included), loading one in process (hf_local), and any training run wait for the owner's yes, asked with what runs, for how long, and what it touches.
````

New:

````
17. No access to alpha01 without the owner's approval in the working session: no rx command of any kind (run, job, exec, doctor, pull, or sync), read-only probes, file uploads, and compiler work included, and no model inference (serving, RNGD included, or loading one in process) or training there. Ask once per batch of server steps, with what runs, for how long, and what it touches, and wait for the owner's yes.
````

2. Environment State, the AMD MI300X row, last cell

Old:

````
its GPU half has no host: the owner said on 2026-10-06 that the MI300Xs will not be granted and to focus on Tenstorrent and CPU (OQ-040) |
````

New:

````
its GPU half has no host for now: the owner said on 2026-10-06 that there is no MI300X access and to focus on Tenstorrent and CPU, and on 2026-10-07 that access may come in the future (OQ-040) |
````

3. Build Roadmap, the work-order sentence

Old:

````
the GPU half runs only if a GPU host is granted, and none is planned (OQ-038; OQ-040, 2026-10-06).
````

New:

````
the GPU half runs only once a GPU host is granted; none is planned now, and MI300X access may come later (OQ-038; OQ-040, 2026-10-06 and 2026-10-07).
````

4. Risks And Questions, the no-GPU-host row, last cell

Old:

````
the owner said on 2026-10-06 that the MI300Xs will not be granted, so no GPU host is planned and these stay unmeasured |
````

New:

````
the owner said on 2026-10-06 that there is no MI300X access for now and on 2026-10-07 that it may come in the future, so these stay unmeasured until a GPU host is granted |
````

5. Decision Log, the count sentence

Old:

````
One hundred and thirty-one decisions have been made: twenty-six on 2026-09-22, twenty on 2026-09-23, thirty-six on 2026-09-24, seven on 2026-09-25, sixteen on 2026-09-26, five on 2026-09-28, three on 2026-09-30, two on 2026-10-04, three on 2026-10-05, twelve on 2026-10-06, and one on 2026-10-07;
````

New:

````
One hundred and thirty-three decisions have been made: twenty-six on 2026-09-22, twenty on 2026-09-23, thirty-six on 2026-09-24, seven on 2026-09-25, sixteen on 2026-09-26, five on 2026-09-28, three on 2026-09-30, two on 2026-10-04, three on 2026-10-05, twelve on 2026-10-06, and three on 2026-10-07;
````

6. Decision Log, two new rows at the top of the table

Old:

````
| Date | Decision | Rationale |
| --- | --- | --- |
````

New:

````
| Date | Decision | Rationale |
| --- | --- | --- |
| 2026-10-07 | MI300X access may come later (OQ-040; Environment State, Build Roadmap, Risks And Questions; plans/STATUS.md, plans/p17-portable.md). The owner said in the working session: "to be clear, we may in the future have access to the MI300Xs." The 2026-10-06 answer means no MI300X access now, not never: P17.12, P17.13, P17.G, P8, and P9 stay BLOCKED until the owner grants access, and OQ-040 reopens then | The 2026-10-06 entry recorded the answer as the MI300Xs not being granted; the owner's clarification narrows it to the present |
| 2026-10-07 | Agent Rule 17 widened: no access to alpha01 at all without the owner's approval in the working session (the owner, opening the session: "Keep going, you do not have access to the server without my approval."; Agent Rules; AGENTS.md, Remote Execution). Every rx command, read-only probes, uploads, and compiles included, waits for the owner's yes, asked once per batch of server steps with what runs, for how long, and what it touches; a task whose server steps lack it goes OWNER. The read-only probes task P5.1 ran on 2026-10-06 came before this widening | The rule of 2026-10-06 held only inference and training; the owner now holds every server action |
````

## 2026-10-07: task P5.1, the pins of P5's two MLIR stacks (after the entry above)

1. Toolchain Pins, a new bullet before the pin-files bullet

Old:

````
- Pin files live in `toolchains/<name>.pin` and record commit, build flags, and install path.
````

New:

````
- MLIR stacks [DESIGN] (task P5.1, plans/spikes/p5-mlir-pins.md): two LLVMs, so CPU and TT modules stay apart, each verified by its own toolchain. Polygeist is pinned by the P5 plan's rule, the Polygeist main commit whose llvm-project submodule commit is nearest tt-mlir's LLVM 4efe170d in llvm-project's history (fewest commits apart, the newer on a tie), unless upstream CI, issues, or source show a commit building against 4efe170d itself: 77c04bb2a7a2406ca9480bcc9e729b07d2c8d077 (2024-07-31, the head of llvm/Polygeist main on 2026-10-07), whose llvm-project submodule commit is 26eb4285b56edd8c897642078d91f16ff0fd3472 (2023-09-28, LLVM 18.0.0git), an ancestor of 4efe170d and 78,791 commits behind it. Every main commit since 2023-10-06 carries that submodule commit, so the tie goes to the head, and no Polygeist branch or pull request updated since 4efe170d's date carries 4efe170d. Polygeist does not build against 4efe170d as it stands: it includes clang/Parse/ParseDiagnostic.h and mlir/Support/MathExtras.h, which llvm-project removed before 4efe170d, and its maintainers support only the submodule's commit (issue 433). Its LLVM builds clang and mlir for the host target. tt-mlir stays at the joint pin with its LLVM 4efe170d858eb54432f520abb4e7f0086236748b, to which its env/CMakeLists.txt applies env/patches/affine-allow-symbol-vars.patch (git blob 2ffdae14d2bcb8244189d8eb044dd093642fa436), and with flatbuffers fb9afbafc7dfe226b9db54d4923bfb8839635274, whose flatc tt-mlir's build needs; that LLVM builds mlir and lld, no clang. The TT raiser links the clang of Polygeist's LLVM, since the build host lacks the clang development libraries (libclang-17-dev, libclang-20-dev, libclang-cpp17-dev, and libclang-cpp20-dev are not installed; rx 20261006-214936-exec-a225). [OPEN] until tasks P5.4 and P5.5 install them: the build configurations, and the space, which P5.1 projects for both stacks together, built in place, at about 24 to 43 GiB with static libraries, or about 13 to 28 GiB with shared libraries and only the needed targets (PROJECTED), against 17.1 GiB left under the 115G stop line, 5G under the 120G scratch cap (Host Facts), at du 102,646,204 KiB on 2026-10-06 (rx 20261006-214855-exec-9935); neither range fits, so both installs wait for the owner (OQ-049). A pin change is a Decision Log entry.
- Pin files live in `toolchains/<name>.pin` and record commit, build flags, and install path.
````

2. Decision Log, the count sentence

Old:

````
One hundred and thirty-three decisions have been made: twenty-six on 2026-09-22, twenty on 2026-09-23, thirty-six on 2026-09-24, seven on 2026-09-25, sixteen on 2026-09-26, five on 2026-09-28, three on 2026-09-30, two on 2026-10-04, three on 2026-10-05, twelve on 2026-10-06, and three on 2026-10-07;
````

New:

````
One hundred and thirty-four decisions have been made: twenty-six on 2026-09-22, twenty on 2026-09-23, thirty-six on 2026-09-24, seven on 2026-09-25, sixteen on 2026-09-26, five on 2026-09-28, three on 2026-09-30, two on 2026-10-04, three on 2026-10-05, twelve on 2026-10-06, and four on 2026-10-07;
````

3. Decision Log, a new row at the top of the table

Old:

````
| Date | Decision | Rationale |
| --- | --- | --- |
````

New:

````
| Date | Decision | Rationale |
| --- | --- | --- |
| 2026-10-07 | The pins of P5's two MLIR stacks (task P5.1; Toolchain Pins): Polygeist 77c04bb2a7a2406ca9480bcc9e729b07d2c8d077 with its llvm-project submodule commit 26eb4285b56edd8c897642078d91f16ff0fd3472, chosen by the P5 plan's rule (the Polygeist main commit whose submodule commit is nearest tt-mlir's LLVM 4efe170d in llvm-project's history, the newer on a tie, unless one is shown building against 4efe170d itself), so two LLVMs, with CPU and TT modules kept apart; tt-mlir at the joint pin with LLVM 4efe170d, tt-mlir's affine-allow-symbol-vars patch, and flatbuffers fb9afbaf; and the TT raiser linking the clang of Polygeist's LLVM. The build configurations and the space for them stay [OPEN], and tasks P5.4 and P5.5 stay held for the owner (OQ-049) | Polygeist's main has not moved its submodule since 2023-10-06; of the 51 submodule commits its main has carried that llvm-project holds, its head's 26eb4285 is the nearest, an ancestor 78,791 commits behind 4efe170d. No Polygeist branch or pull request updated since 4efe170d's date carries 4efe170d, Polygeist includes two headers llvm-project removed before it, and its maintainers support only the submodule's commit, so one LLVM cannot serve both stacks. tt-mlir's LLVM builds mlir and lld only and the host has no clang development libraries, while Polygeist's LLVM builds clang anyway, so the raiser needs no LLVM of its own. PROJECTED from upstream build figures and source sizes, both stacks together, built in place, take about 24 to 43 GiB with static libraries, or about 13 to 28 GiB with shared libraries and only the needed targets, against 17.1 GiB left under the 115G stop line at du 102,646,204 KiB (rx 20261006-214855-exec-9935); the shared-library range fits only at its low end, so neither projected range fits under the stop line before the owner's answer. plans/spikes/p5-mlir-pins.md records the commands, outputs, and sources |
````

## 2026-10-07: task P5.2, the v0 corpus pipeline (after the entries above)

1. 1. Review Of v0, the opening paragraph

Old:

````
Reviewed at [JoeMad21/mlir-corpus-pipeline](https://github.com/JoeMad21/mlir-corpus-pipeline) commit 3ccd280. [OPEN] The alpha01 copy may be ahead and gets the same review once agents have access.
````

New:

````
Reviewed at [JoeMad21/mlir-corpus-pipeline](https://github.com/JoeMad21/mlir-corpus-pipeline) commit 3ccd280. The alpha01 copy got the same review (task P5.2, plans/spikes/p5-v0-corpus.md; rx 20261007-123046-exec-6b45 and 20261007-123121-exec-795a): it is at `/mnt/nvme10/joseph_ufl/mlir-corpus-pipeline` on main at 3ccd280, with no tracked file changed and no commit GitHub lacks, so the review stands, with the Keep row corrected below. Of its 38 untracked files, 37 are a LoRA fine-tuning experiment over v0's Parquet (training/, configs/, three scripts, and the adapter run runs/ir-completion) and one is a stray empty file; none is migrated: the experiment's pass_transform task is the pass imitation the captured-chain row reclassifies, its ir_completion task trains on syntax pretraining data, its signal extraction is the string heuristic the boilerplate row replaces, and its split by random benchmark put apps that are LASSI eval items in its training pool (Agent Rule 5). v0's ClangIR LLVM build and HeCBench clone are not at v0's paths under `/mnt/nvme10/joseph_ufl` (rx 20261007-123031-exec-a546) and were not found by bounded finds, so v0 cannot be rerun as configured, and its output stays there as it is. Task P5.11 moves the Keep row's pieces from GitHub by P5.2's migration map. Of v0's 480 Parquet entries (derived from the manifests by build_parquet.py's rule: one per benchmark and pass, round trip ok, from 91 HeCBench apps), 451 from 85 apps can stand as cir-snapshots, pending the per-op generic-form test; 27 entries of 5 apps that are eval items of lassi-hecbench-10 or lassi-ee-22, and 2 truncated entries, are left out and counted. None is a whole module in generic form by a first-line test (rx 20261007-123334-exec-e456); the per-op test, task P5.11's refusal, applies before any catalogue reads them, and the catalogue waits for the first phase that reads syntax-pretraining data.
````

2. 2. Review Of v0, the Keep row

Old:

````
| Pinned LLVM, manifest with parent-child hashes, exit-code taxonomy, Policy B, Hive Parquet with provenance | Sound | Keep; move into `lassi/corpus/` |
````

New:

````
| Manifest with parent-child hashes, exit-code taxonomy, Policy B, Hive Parquet with provenance | Sound, with two findings (task P5.2): Policy B deletes a snapshot that fails, and the Parquet's dedup key, benchmark and pass name, would keep only the first of a pass repeated in one run | Keep; move into `lassi/corpus/` by P5.2's migration map: failures as records, never deletions, and dedup by structural hash |
| LLVM built by `scripts/build-llvm.sh` | Not pinned: a shallow clone of llvm-project HEAD (LLVM_COMMIT is HEAD, with a TODO to pin), whose commit no output read records; the build is not at its install path on alpha01 (task P5.2) | Replaced by P5's own pins (Toolchain Pins, MLIR stacks); v0's ClangIR chain is not rebuilt |
````

3. 3. Review Of v0, the defines row

Old:

````
| `-Ddfloat=float -Ddlong=int` | Changes program semantics | Record per entry; exclude from pair sets unless built with Makefile values |
````

New:

````
| `-Ddfloat=float -Ddlong=int` | Changes program semantics where a source uses the two names. Of v0's apps only adv-cuda's build changed: its Makefile sets dfloat=double, v0's flags float; axhelm-cuda uses the names, but its Makefile sets the same values as v0's flags (task P5.2; HeCBench at 692cba3 and 8b88057) | Record per entry; exclude from pair sets unless built with Makefile values |
````

4. 4. Host Facts, the existing-assets bullet

Old:

````
the v0 MLIR corpus, a ClangIR LLVM build, and CUDA 12.6.3 headers under the corpus pipeline's project root; and a complete CUDA 12.6 toolkit (directory `cuda-12.6.3`, nvcc V12.6.85, not on PATH) at `/mnt/nvme10/joseph_ufl/cuda-12.6.3` [MEASURED 2026-09-23].
````

New:

````
the v0 MLIR corpus (`corpus-alpha01`, 1,157,684 KiB, and `corpus-parquet`, 6,664 KiB) and the corpus pipeline's copy (`mlir-corpus-pipeline`, 300,984 KiB) under its project root, `/mnt/nvme10/joseph_ufl` (du on 2026-10-07, rx 20261007-123205-exec-0778; v0's ClangIR LLVM build and HeCBench clone are not at v0's paths there, rx 20261007-123031-exec-a546; task P5.2); and a complete CUDA 12.6 toolkit (directory `cuda-12.6.3`, nvcc V12.6.85, not on PATH) at `/mnt/nvme10/joseph_ufl/cuda-12.6.3` [MEASURED 2026-09-23], the CUDA_HOME of v0's scripts (task P5.2).
````

5. 5. Risks And Questions, question 8

Old:

````
8. Is the alpha01 copy of the corpus pipeline ahead of GitHub commit 3ccd280?
````

New:

````
8. Is the alpha01 copy of the corpus pipeline ahead of GitHub commit 3ccd280? Answered 2026-10-07 (task P5.2): no; it is at 3ccd280 with no tracked change and no commit GitHub lacks, plus the untracked files of a fine-tuning experiment that is not migrated; see Review Of v0 and plans/spikes/p5-v0-corpus.md.
````

6. 6. Decision Log, the count sentence and a new top row

Old:

````
One hundred and thirty-four decisions have been made: twenty-six on 2026-09-22, twenty on 2026-09-23, thirty-six on 2026-09-24, seven on 2026-09-25, sixteen on 2026-09-26, five on 2026-09-28, three on 2026-09-30, two on 2026-10-04, three on 2026-10-05, twelve on 2026-10-06, and four on 2026-10-07; add new entries at the top, newest first.

| Date | Decision | Rationale |
| --- | --- | --- |
````

New:

````
One hundred and thirty-five decisions have been made: twenty-six on 2026-09-22, twenty on 2026-09-23, thirty-six on 2026-09-24, seven on 2026-09-25, sixteen on 2026-09-26, five on 2026-09-28, three on 2026-09-30, two on 2026-10-04, three on 2026-10-05, twelve on 2026-10-06, and five on 2026-10-07; add new entries at the top, newest first.

| Date | Decision | Rationale |
| --- | --- | --- |
| 2026-10-07 | The v0 corpus pipeline on alpha01 (task P5.2; Review Of v0, Risks And Questions question 8, Host Facts): the alpha01 copy is GitHub commit 3ccd280 with no tracked change, so the review at 3ccd280 stands, with its Keep row corrected, and its [OPEN] closes; the copy's untracked fine-tuning experiment is not migrated; the LLVM v0's script builds was never pinned and is replaced by P5's pins; task P5.11 moves the manifest's hash chain, the exit-code taxonomy, Policy B (as records, not deletions), and the Hive Parquet layout from GitHub by plans/spikes/p5-v0-corpus.md's migration map; v0's output stays on alpha01, and of its 480 Parquet entries (derived) 451 from 85 apps can stand as cir-snapshots pending the per-op generic-form test, with 27 entries of 5 apps that are eval items of lassi-hecbench-10 or lassi-ee-22 and 2 truncated entries left out and counted | Read-only probes under OQ-050 (the spike's Base names those that went past it) found the copy at 3ccd280, 0 commits ahead or behind, no tracked diff, and 38 untracked files (rx 20261007-123046-exec-6b45, 20261007-123121-exec-795a). The ClangIR build and the HeCBench clone are not at v0's paths (rx 20261007-123031-exec-a546) and were not found by bounded finds, so v0's output is fixed: 291 runs of 100 benchmarks, 1,364 snapshot files; the 480-entry count, derived by build_parquet.py's rule from the manifests' statuses, is consistent with the README's 141 MB after the 2 MiB cap (rx 20261007-123822-exec-f923). Agent Rule 5 keeps eval items' entries out of any training data and Agent Rule 11 keeps truncated and generic IR out; the fine-tuning's split held out random benchmarks, so its training pool held eval apps |
````
