# Spike P5.2: the v0 corpus pipeline on alpha01 (question 8)

- Task: P5.2 (plans/p5-ir.md, planning decision "v0 migration (P5.2, P5.11)"). Bible: Review Of v0, Corpus Pipeline, Benchmark Suites (cir-snapshots row; split rules), Risks And Questions (question 8), Host Facts (existing assets), Agent Rules 5 and 11.
- Date: 2026-10-07 on the workstation. Read-only probes of alpha01 ran from 12:30 to 12:38 on 2026-10-07 by the alpha01 clock (UTC-07:00), that is 19:30 to 19:38 UTC; each rx id carries its start time. GitHub was read the same day with `gh api` (authenticated, read-only) and from clones on the workstation.
- Base: branch p5-ir at c4d5f95 with task P5.1's changes uncommitted in the working tree; they were committed during the spike as eb9ebd9, and the bible edits below are exact against eb9ebd9's docs/BIBLE.md. The owner approved on 2026-10-07 under Agent Rule 17 (OQ-050, option (a)) one `uv run tools/rx.py doctor` and about 10 to 15 read-only `uv run tools/rx.py exec` probes: a bounded find, git state, du, file counts and sizes, and first lines of v0's outputs. One doctor and 14 exec probes ran, each bounded by `--timeout`, from the scratch root without a checkout; no rx run, job, pull, sync, or slot command ran, and nothing was deleted. Seven probes went past that approval: rx 20261007-123449-exec-53cd wrote on alpha01 (a leftover `mktemp -d` created one empty directory, Host state), and rx 20261007-123334-exec-e456, 20261007-123525-exec-d345, 20261007-123537-exec-7065, 20261007-123610-exec-0783, 20261007-123756-exec-b195, and 20261007-123822-exec-f923 read more than first lines (Method). Separately, rx 20261007-123046-exec-6b45 printed a GitHub access token (Host state). All of it was put to the owner in the working session, who ratified the reads on 2026-10-07 (OQ-051, option (a)), so the figures that rest on them stand; the directory and the token wait for the owner's actions (OQ-051).
- Device: none. No probe opened a device or named a device tool.
- Labels: (host) marks a value read on alpha01 by a read-only rx exec, cited by its rx id; (source) a value read from v0's source at GitHub commit 3ccd280, cited as path:line; (metadata) a value from the GitHub API; (derived) a count computed from host readings by the rule stated beside it, not read from the file it describes. Nothing here is [MEASURED], since no run came from a clean commit. Figures not read are PROJECTED where they appear.
- Output conventions: outputs are verbatim except where `[... trimmed ...]` or `[redacted]` says otherwise. No benchmark source, IR text beyond one module name, or model output enters this file.

## Question

Bible question 8 asks whether the alpha01 copy of the corpus pipeline is ahead of GitHub commit 3ccd280, and the Review Of v0 section holds an [OPEN]: the copy gets the same review once agents have access. P5.2's acceptance asks, with commands, outputs, and sources:

1. Where the alpha01 copy lives, its git state, and its differences from 3ccd280, file by file.
2. Each difference reviewed against the Review Of v0 table, with an action.
3. An inventory of v0's outputs: entry counts and sizes, each entry's source app, entries cut by the 2 MB whale truncation, entries in generic form, entries built with `-Ddfloat=float -Ddlong=int`, the LLVM and CUDA trees v0 uses with sizes, and the ClangIR LLVM build's commit and whether any P5 build can use it.
4. Each entry's source app matched against every eval or unassigned item of every suite, with counts per suite.
5. A migration map for the Keep row of Review Of v0.
6. What of v0's output can stand as cir-snapshots under Agent Rules 5 and 11.

Why it matters: P5.11 migrates what the map keeps into lassi/corpus/ and needs a remote step only if a kept piece exists only on alpha01; the cir-snapshots catalogue, deferred to the first phase that reads syntax-pretraining data, starts from the counts and exclusions here; OQ-049 waits for any space this spike finds.

Classification: factual. Every part is answered from v0's source at 3ccd280, GitHub metadata, HeCBench's source on GitHub, the LASSI-EE paper's application table, and read-only probes of alpha01.

## Method

GitHub side first, before any probe:

- `gh api repos/JoeMad21/mlir-corpus-pipeline` and its branches and commits (metadata): public, default branch main, one branch, eight commits from 83d7b35 to 3ccd280, all on 2026-07-08; pushed_at 2026-07-08T22:03:53Z. main is 3ccd280283b05379c585e55a0400d9106577e45c.
- A clone at 3ccd280 on the workstation: nine tracked files, .gitignore, README.md, env/setup.sh, scripts/build-llvm.sh, scripts/lower_all.sh, scripts/lower_one.sh, tools/build_parquet.py, tools/mlir_to_python.py, and tools/parse_snapshots.py, each read in full. env/setup.sh sets PROJECT_ROOT to /mnt/nvme10/joseph_ufl, LLVM_INSTALL to $PROJECT_ROOT/llvm-cir, and CUDA_HOME to $PROJECT_ROOT/cuda-12.6.3; the README's layout adds llvm-project-src/, HeCBench/, and corpus-alpha01/ there, and build_parquet's usage writes corpus-parquet/. So the probe list targeted those names.
- The suites' items: assets/bench/lassi-hecbench-10.yaml (10 items, all split eval); lassi-ee-22, which has no manifest yet, from the bible's Benchmark Suites row ("20 HeCBench apps, XSBench, miniMDock", eval only) and the paper's Table I (arXiv:2505.02184 v3); assets/bench/tt-pairs-v0.yaml (Tier A, five tt-metal programming examples, unassigned). Tier B is planned and Tier C not built.
- HeCBench's CUDA sources, for the defines: a sparse, blob-filtered checkout of src/*-cuda (sources, headers, and Makefiles) of zjin-lcf/HeCBench at the LASSI pin 692cba32c5744f6ef024cca59f65e9488edba8bf (2026-04-02) and at 8b88057a1d7005c3853bb504fac1c41401b1bba1 (2026-07-08T16:00:39Z), the newest master commit before v0's first batch (19:48 UTC that day), searched for the words dfloat and dlong. v0's own HeCBench clone is not at its path (Results 3), so its commit is not known; these two bracket it (Results 3).

Probe plan, fixed after the GitHub side, each probe's worst-case output counted before it ran and kept under 120 lines (`| head -N` on every list, names folded onto long lines): one `rx doctor`; then read-only `rx exec` probes in this order: locate the copy and v0's trees by a bounded find; the copy's git state; its untracked and ignored files; the untracked training configuration and stats; the corpus's sizes, run directories, and benchmark names; snapshot sizes, whales, and first lines; module lines, defines, batch statuses, and Parquet partitions; per-app counts; the manifests' round-trip statuses. Fourteen exec probes ran, each under a minute. Scans used find with -maxdepth and -printf for sizes, `head -n` of the first lines of each file, and sort, uniq, awk, sed, and comm to count. Six probes read past OQ-050's "first lines of v0's outputs", all read-only, none approved beforehand: rx 20261007-123334-exec-e456 read the seven batches' summary.csv files whole, and each snapshot's lines up to its first line that is not an alias definition; rx 20261007-123537-exec-7065 read one entry's main.ll whole for a compiler banner, and rx 20261007-123525-exec-d345 read whole, for the same, the main.ll files matching `ac*-cuda-2026070*-13*`; rx 20261007-123610-exec-0783 read every def and class line of three training modules; rx 20261007-123756-exec-b195 read the 252 run manifest.json files whole in three passes, the first of which sent its output to /dev/null, and rx 20261007-123822-exec-f923 read them whole again and ran stat on the 599 snapshot files they mark ok. A seventh, rx 20261007-123449-exec-53cd, wrote (Host state). They are put to the owner for ratification after the fact (Owner queue).

## Host state

- `uv run tools/rx.py doctor` (no rx id), 2026-10-07 at about 12:30 on the alpha01 clock: stop false, scratch_free_gb 393.7, root_free_gb 221.5, nproc 256, mem_avail_gb 2983.6, load 28.0 / 39.3 / 45.6, config max_big_jobs 1 and max_build_jobs 32, devices_enabled rngd true and tt_silicon, rocm_gpu, and nvidia_gpu false, running_jobs empty, seven slots.
- A probe wrote to alpha01 by mistake, which OQ-050 did not allow. rx 20261007-123449-exec-53cd's command began with a leftover `mktemp -d -p $R/tmp`, which created one empty directory, /mnt/nvme10/joseph_ufl/tmp/tmp.TM9Wfi9pr3, at 2026-10-07T12:34:49-07:00, 4,096 bytes, no entries (read in rx 20261007-123525-exec-d345). Nothing else was written. Agents never delete under the scratch root, so it waits for the owner's removal (Owner queue).
- rx 20261007-123046-exec-6b45 printed `git remote -v` of the copy. Its URL embeds a GitHub access token in clear text. The value is not recorded in this file, but the gate's run record of that id on alpha01 and the working session's transcripts on the workstation hold it, so redaction cannot close the exposure: the token has to be revoked on GitHub and dropped from the copy's remote URL, both the owner's to do or approve (Owner queue).

## Results

### 1. Where the copy lives, its git state, and its differences from 3ccd280

rx 20261007-123031-exec-a546 (host): a find to depth 4 of the scratch root and /home/joseph_ufl, the gate's run and worktree trees left out, for lower_one.sh, build_parquet.py, parse_snapshots.py, and names starting mlir-corpus-pipeline:

```
2026-10-07T12:30:31-07:00
--- find copies
/mnt/nvme10/joseph_ufl/mlir-corpus-pipeline
/mnt/nvme10/joseph_ufl/mlir-corpus-pipeline/scripts/lower_one.sh
/mnt/nvme10/joseph_ufl/mlir-corpus-pipeline/tools/parse_snapshots.py
/mnt/nvme10/joseph_ufl/mlir-corpus-pipeline/tools/build_parquet.py
--- top names
[... trimmed: 83 names, among them corpus-alpha01, corpus-parquet, cuda-12.6.3, and mlir-corpus-pipeline, and no llvm-cir, llvm-project-src, or HeCBench ...]
--- tmp builds
--- end
```

One copy, at /mnt/nvme10/joseph_ufl/mlir-corpus-pipeline. /tmp holds no llvm-build-* tree.

rx 20261007-123046-exec-6b45 (host), every git command with `--no-optional-locks`:

```
--- git
3ccd280283b05379c585e55a0400d9106577e45c
* main                3ccd280 [origin/main] Fix extract_signal line budget: clip within functions so large functions cannot blow the budget
  remotes/origin/main 3ccd280 Fix extract_signal line budget: clip within functions so large functions cannot blow the budget
origin	https://JoeMad21:[redacted]@github.com/JoeMad21/mlir-corpus-pipeline.git (fetch)
origin	https://JoeMad21:[redacted]@github.com/JoeMad21/mlir-corpus-pipeline.git (push)
--- has 3ccd280
commit
--- ahead behind
0	0
--- status
?? =3.1.0
?? configs/base.yaml
[... trimmed: 23 more untracked paths, listed below ...]
--- diff stat vs 3ccd280
--- log
3ccd280 2026-07-08T15:03:52-07:00 Fix extract_signal line budget: clip within functions so large functions cannot blow t
[... trimmed: the seven older commits, 83d7b35 to dd03f19, the same as GitHub's ...]
```

`git log --oneline --all --not 3ccd280` printed nothing. rx 20261007-123121-exec-795a (host) counted the status lines and listed the rest:

```
--- status classes
     15 !!
     38 ??
```

Git state: branch main at 3ccd280, tracking origin/main, 0 commits ahead and 0 behind, no commit outside 3ccd280's history, and no tracked file changed (`git diff --stat 3ccd280` is empty; status shows no tracked change). The answer to question 8 is no: the copy is not ahead of GitHub. Its differences from 3ccd280 are 38 untracked files and 15 ignored ones, none of them on GitHub (rx 20261007-123121-exec-795a, 20261007-123153-exec-b9bc):

| Path | Size, date (alpha01 clock) | What it is |
| --- | --- | --- |
| `=3.1.0` | 0 bytes, 2026-07-08T14:55:31 | An empty file, the trace of an unquoted shell redirection such as `pip install x>=3.1.0` |
| configs/base.yaml, configs/ir_completion.yaml, configs/pass_transform.yaml | 2,080, 622, and 517 bytes; 59, 24, and 23 lines; 2026-07-16 | Configurations of a LoRA fine-tuning of a 1.5B-parameter code model (Qwen/Qwen2.5-Coder-1.5B-Instruct) on the CPU over corpus-parquet: data.corpus_path /mnt/nvme10/joseph_ufl/corpus-parquet, exclude_truncated true, eval_fraction 0.1 of benchmarks with split_seed 0, signal extraction with max_lines 400, tasks pass_transform and ir_completion, LoRA r 16, alpha 32 |
| scripts/train.sh, scripts/evaluate.sh, scripts/install-deps-training.sh | 729, 794, and 1,619 bytes; 21, 21, and 54 lines; 2026-07-16 | Drivers of that fine-tuning |
| training/ (README.md, \_\_init\_\_.py, collate.py, config.py, data.py, evaluate.py, merge_adapter.py, modeling.py, signal.py, tasks.py, train.py, tests/\_\_init\_\_.py, tests/smoke_test.py, tests/synthetic_corpus.py) | 14 files; 188 KiB by du, which also counts the ignored training/\_\_pycache\_\_; 2026-07-16 | The fine-tuning package: "LoRA fine-tuning over the ClangIR corpus" (training/README.md, 108 lines; its first 70 lines and the three modules' def and class lines read in rx 20261007-123610-exec-0783). data.py splits by benchmark (`group_split`), signal.py isolates user functions by symbol-name tests (`_is_library_symbol`), tasks.py builds pass_transform (parent IR and pass name to child IR) and ir_completion (causal LM over extracted IR). Its README refers to a TRAINING_HANDOFF.md at the repository root, which is not in the copy |
| runs/ir-completion/ (17 files: adapter_model.safetensors, adapter_config.json, tokenizer files, train_stats.json, resolved_config.yaml, training_args.bin, README.md, and checkpoint-7/ with optimizer.pt, scheduler.pt, rng_state.pth, and trainer_state.json) | 300,188 KiB | The ir_completion run's LoRA adapter and its checkpoint |
| ir.log (ignored by `*.log`) | 9,197 bytes, 2026-07-16T16:09:00 | The run's log, not read |
| 14 files under tools/\_\_pycache\_\_ and training/\_\_pycache\_\_ (ignored) | not read | Python bytecode |

The adapter run's train_stats.json (rx 20261007-123449-exec-53cd, first 31 lines):

```
{
  "data": {
    "rows_total": 478,
    "rows_train": 431,
    "rows_eval": 47,
    "examples_train": 100,
    "examples_eval": 47,
    [... trimmed: encoded and truncated counts, 100 and 47 ...]
    "eval_groups": [
      "adv-cuda",
      "aes-cuda",
      "aligned-types-cuda",
      "asta-cuda",
      "atomicReduction-cuda",
      "attentionMergeState-cuda",
      "bh-cuda",
      "bincount-cuda",
      "channelShuffle-cuda"
    ],
    "task": "ir_completion"
  },
  [... trimmed: eval_loss 0.425, epoch 1.0 ...]
```

rows_total 478 is consistent with the Parquet's 480 entries (derived, Results 3) less its 2 truncated ones, if exclude_truncated drops them before rows_total is counted (not read). None of the five eval-matched apps of Results 4 is among the nine eval groups, so their 27 entries were in the training pool (the run used 100 of its 431 training rows; which ones was not read).

### 2. Each difference reviewed against the Review Of v0 table

The tracked files are byte for byte 3ccd280, so the review at 3ccd280 stands for them, with its Keep row corrected: re-reading them for the migration map (Results 5) adds two findings to the Keep row and moves its LLVM into a row of its own (Proposed bible edit). The untracked files are reviewed here:

| Difference | Review Of v0 row it meets | Action |
| --- | --- | --- |
| `=3.1.0` | none | Drop: an empty artifact |
| training/, configs/, scripts/train.sh, evaluate.sh, install-deps-training.sh | Captured chain (reclassify as syntax pretraining data): its pass_transform task trains a model to imitate deterministic passes, which the row rejects as translation material; its ir_completion task trains on syntax pretraining data, which cir-snapshots already names and whose catalogue waits for the first phase that reads syntax-pretraining data. Standard-library boilerplate (filter in IR): signal.py and the probe tools/mlir_to_python.py find user code by symbol-name strings, the heuristic the row replaces with `symbol-dce` from user roots or `loc` filtering. Agent Rule 5: its split holds out a random 10% of benchmarks at training time, while LASSI assigns splits per item before any training, and eval items never train | Drop: not migrated. P17's trainer interface (lassi/train/) and P7 replace the training code; the IR filter is P5.6's Normalize |
| runs/ir-completion/ (adapter and checkpoint) | Agent Rule 5: its training pool held the entries of five apps that are LASSI eval items (Results 4) | Drop: never used by LASSI; a space candidate for the owner (OQ-049 Response) |
| ir.log, \_\_pycache\_\_ files | none | Drop: logs and bytecode |

### 3. Inventory of v0's outputs

Trees v0 uses, with sizes (host):

| Tree | Role in v0 (source) | At this path (rx 20261007-123031-exec-a546) | Size |
| --- | --- | --- | --- |
| /mnt/nvme10/joseph_ufl/mlir-corpus-pipeline | the copy | present | 300,984 KiB, of which runs/ 300,188 (rx 20261007-123205-exec-0778, 20261007-123121-exec-795a) |
| /mnt/nvme10/joseph_ufl/corpus-alpha01 | lower_one.sh and lower_all.sh output (scripts/lower_one.sh:14, scripts/lower_all.sh:14) | present | 1,157,684 KiB (rx 20261007-123205-exec-0778) |
| /mnt/nvme10/joseph_ufl/corpus-parquet | build_parquet.py output | present: 91 partitions, 91 files, 6,247,748 bytes, all written 2026-07-08T13:54 | 6,664 KiB (rx 20261007-123205-exec-0778, 20261007-123334-exec-e456) |
| /mnt/nvme10/joseph_ufl/llvm-cir | LLVM_INSTALL, the ClangIR LLVM install (env/setup.sh:14; scripts/build-llvm.sh:89) | absent | none |
| /mnt/nvme10/joseph_ufl/llvm-project-src | its source, a shallow clone (scripts/build-llvm.sh:30, :66) | absent | none |
| /tmp/llvm-build-PID | its build tree, removed by the script after install (scripts/build-llvm.sh:35, :137) | absent | none |
| /mnt/nvme10/joseph_ufl/HeCBench | the benchmark sources (README, One-time setup, step 6; scripts/lower_all.sh:9) | absent | none |
| /mnt/nvme10/joseph_ufl/cuda-12.6.3 | CUDA_HOME (env/setup.sh:56), passed as `--cuda-path` by scripts/lower_one.sh:98, as the clang line of the oldest run's driver.log shows (rx 20261007-123419-exec-9672) | present | 7,300,128 KiB on 2026-10-06 (rx 20261006-215953-exec-0b88, task P5.1) |

The absence rests first on rx 20261007-123031-exec-a546: its top-level listing of /mnt/nvme10/joseph_ufl (Results 1) holds no llvm-cir, llvm-project-src, or HeCBench, the paths v0 configures (env/setup.sh:13-14, scripts/build-llvm.sh:30, scripts/lower_all.sh:9), and `ls -1d /tmp/llvm-build-*` printed nothing. Two bounded finds, as they ran, printed nothing. rx 20261007-123046-exec-6b45 searched the scratch root only, to depth 5, for the names cir-opt, HeCBench (exact case), llvm-project*, llvm-cir*, and build-llvm.out, with lassi-runs/, lassi-wt/, and .cache/ left out. rx 20261007-123121-exec-795a searched the scratch root and /home/joseph_ufl to depth 7 for cir-opt, hecbench* (any case), build-llvm.out, and the exact name llvm-cir, with lassi-runs/, lassi-wt/, .cache/, toolchains/, and amd/ left out. So llvm-project* and llvm-cir* were searched only to depth 5 of the scratch root, and nothing outside those two roots and /tmp was searched. The build and the clone are not at v0's paths and were not found by these finds.

The ClangIR LLVM build's commit: none was pinned. scripts/build-llvm.sh:56 sets `LLVM_COMMIT="HEAD"  # TODO: pin a 40-char SHA after first successful build`, :66 clones llvm-project with `--depth=1`; a rerun with the clone present runs `git fetch origin` and then `git checkout HEAD`, which stays at the commit already checked out (:58-62); the commit was printed only to the build log (:69), build-llvm.out, which neither find found; and the one entry's main.ll read for a compiler banner holds none (rx 20261007-123537-exec-7065; it is x86 assembly, `.att_syntax` on its first line, since scripts/lower_one.sh:115-116 passes -S without -emit-llvm), and neither do the main.ll files rx 20261007-123525-exec-d345 read. So the commit is recorded nowhere read, and the build is not at its install path. No P5 build can use it: it is not at its path, its commit is unknown, and P5's two LLVMs are pinned at 26eb4285 (Polygeist) and 4efe170d (tt-mlir) by task P5.1 (Toolchain Pins, MLIR stacks). Without that clang and the HeCBench clone at their paths, v0 cannot be rerun as configured; its output on alpha01 is what there is.

The corpus, rx 20261007-123205-exec-0778 (host):

```
--- top kinds
other: batch-runs
other: reverse-cuda-v0
run dirs: 291
--- distinct names
100 names, 100 with more than one run
accuracy-cuda ace-cuda adam-cuda adamw-cuda addBiasQKV-cuda addBiasResidualLayerNorm-cuda adjacent-cuda adv-cuda aes-cuda affine-cuda aidw-cuda aligned-types-cuda all-pairs-distance-cuda allreduce-cuda amgmk-cuda ans-cuda aobench-cuda aop-cuda asmooth-cuda assert-cuda asta-cuda atan2-cuda atomicAggregate-cuda atomicCAS-cuda atomicCost-cuda atomicIntrinsics-cuda atomicPerf-cuda
atomicReduction-cuda atomicSystemWide-cuda attention-cuda attentionMergeState-cuda attentionMultiHead-cuda attentionMultiHeadKVCache-cuda attention-paged-cuda axhelm-cuda axpby-cuda babelstream-cuda background-subtract-cuda backprop-cuda base64e-cuda bezier-surface-cuda bfs-cuda bgmv-cuda bh-cuda bicgstab-cuda bilateral-cuda bincount-cuda binomial-cuda bitcracker-cuda bitonic-sort-cuda
bitpacking-cuda bitpermute-cuda black-scholes-cuda blas-dot-cuda blas-fp8gemm-cuda blas-gemmBatched-cuda blas-gemm-cuda blas-gemmEx2-cuda blas-gemmEx-cuda blas-gemmStridedBatched-cuda blockAccess-cuda blockexchange-cuda blockScan-cuda bm3d-cuda bmf-cuda bn-cuda bonds-cuda boxfilter-cuda bscan-cuda bsearch-cuda bspline-vgh-cuda bsw-cuda b+tree-cuda btree-cuda burger-cuda bwt-cuda car-cuda
cbsfil-cuda cc-cuda ccl-cuda ccs-cuda ccsd-trpdrv-cuda ced-cuda cfd-cuda chacha20-cuda channelShuffle-cuda channelSum-cuda che-cuda chemv-cuda chi2-cuda clenergy-cuda clink-cuda clock-cuda cm-cuda cmembench-cuda cmp-cuda cobahh-cuda collision-cuda colorwheel-cuda reverse-cuda
--- batch runs
batch-20260708-124804
[... trimmed: six more, to batch-20260708-155315 ...]
```

An entry, for this spike, is one snapshot: one `.mlir` file a run's snapshots/ directory holds, the module after one ClangIR pass. Each run directory is `<benchmark>-<YYYYMMDD-HHMMSS>`, and its source app is the benchmark name less `-cuda`; each module's name confirms it, for example `@"/mnt/nvme10/joseph_ufl/HeCBench/src/atomicCost-cuda/main.cu"` (rx 20261007-123537-exec-7065), and every one of the 1,374 module lines starts `module @"/mnt/nvme10/j` (rx 20261007-123334-exec-e456). Of the 100 names above, 99 sort from accuracy-cuda to colorwheel-cuda, and the hundredth is reverse-cuda, the README's example. scripts/lower_all.sh:47-55 takes the first --limit directories of HeCBench's src/ in sorted order, and a driver_error (exit 1) leaves no run directory, since every exit 1 of scripts/lower_one.sh (:10-63) comes before :66-68 creates it, so a benchmark that failed that way left no name. At 8b88057, 104 -cuda directories sort up to colorwheel-cuda: v0's 99, and blas-fp4gemm-cuda, blas-groupgemm-cuda, blas-mxfp6gemm-cuda, blas-mxfp8gemm-cuda, and braycurtis-cuda, which have no v0 run; at 692cba3, 82 do, and 17 of v0's 99 are not there (source). v0's HeCBench commit is unknown, so the suites are matched against these names exactly (Results 4), not by position. reverse-cuda-v0 is an earlier hand run (10 snapshots, 867,583 bytes, and 5 more under snapshots-changeonly/, rx 20261007-123525-exec-d345). All 291 runs are from 2026-07-08.

Counts and sizes (host, rx 20261007-123308-exec-37fd and 20261007-123525-exec-d345):

```
--- snapshot sizes: files bytes over2MiB
1374 493576299 10
le64K 68 le1M 1284 le2M 12 le8M 2 gt8M 8
--- over 2MiB by run name
      8 bm3d-cuda
      2 bn-cuda
--- manifests
253
--- passes
    249 cir-cxxabi-lowering
    249 cir-canonicalize
    248 cir-flat-to-llvm
    244 cir-hoist-allocas
    230 cir-flatten-cfg
    109 cir-lowering-prepare
     32 cir-eh-abi-lowering
     11 cir-goto-solver
      1 omp-mark-declare-target
      1 cir-target-lowering
```

```
--- timestamped runs only: files bytes over2MiB; pairs pre1354 names; pairs all names
1364 492708716 10
521 91
521 91
```

So the 291 timestamped runs hold 1,364 snapshot files, 492,708,716 bytes, from 91 of the 100 benchmarks, in 252 manifests (derived: rx 20261007-123308-exec-37fd's 253 less reverse-cuda-v0's); the counts above that include 1,374 files add reverse-cuda-v0's 10. The 91 names with snapshots are the Parquet's 91 partition names (rx 20261007-123449-exec-53cd compared the two lists and found only reverse-cuda-v0 in one and not the other); the other 9 never produced a snapshot. Which 9 those are was not printed: naming them needs one more read-only listing of corpus-parquet, which this pass did not run, so a later match uses all 100 names, a superset. The size classes, the over-2 MiB list, and the pass counts above count all 1,374.

The batches (rx 20261007-123334-exec-e456), from each summary.csv:

```
batch-20260708-124804 rows=0
batch-20260708-124811 rows=20 driver_error=1 roundtrip_fail=1 compile_fail=7 success=11
batch-20260708-125648 rows=20 roundtrip_fail=1 compile_fail=8 success=11
batch-20260708-130016 rows=20 roundtrip_fail=2 compile_fail=2 success=16
batch-20260708-130213 rows=100 driver_error=1 roundtrip_fail=19 compile_fail=13 success=67
batch-20260708-131253 rows=100 driver_error=1 compile_fail=9 partial_kept=19 success=71
batch-20260708-155315 rows=20 compile_fail=1 partial_kept=2 success=17
```

batch-20260708-131253 is the README's "100-benchmark HeCBench-CUDA run" (71 success, 19 partial_kept, 9 compile_fail, 1 driver_error).

Round-trip statuses (rx 20261007-123756-exec-b195 and 20261007-123822-exec-f923, host; derived where marked): 115 of the 252 manifests (derived: 252 less the 137 without one) record a roundtrip_status per snapshot, 599 ok and 51 failed, and 137, from runs of the older script, record none, so build_parquet.py skips their snapshots (tools/build_parquet.py:81). All 599 ok files are present, 182,036,397 bytes. The Parquet's entries (derived, by build_parquet.py's rule at tools/build_parquet.py:80-112, :121-127: snapshots with status ok, the newest run per benchmark and pass, runs before the Parquet's write at 13:54): 480 entries from 91 benchmarks, 154,422,155 bytes before the cap. With the cap, the two entries over 2 MiB, bm3d-cuda's cir-hoist-allocas (14,047,896 bytes, run 20260708-131307) and bn-cuda's cir-flat-to-llvm (3,674,207 bytes, run 20260708-131308), become 2,097,152 bytes each, which gives 140,894,356 bytes of IR text, the README's "141 MB of IR text" on "the reference 91-benchmark corpus"; and the fine-tuning's rows_total, 478, is 480 less those 2. The derivation is consistent with both figures, though the 141 MB match is to the README's rounding, and the 478 match assumes rows_total is counted after exclude_truncated, which was not read (only training/data.py's def and class lines were); so it stands for the Parquet's rows, which were not read directly.

Whale truncation: 10 snapshot files are over 2 MiB (2,097,152 bytes), 8 of bm3d-cuda's (13.4 to 16.0 MB, runs 20260708-130227 and 20260708-131307) and 2 of bn-cuda's (3,674,207 bytes each, runs 20260708-130227 and 20260708-131308) (rx 20261007-123419-exec-9672). The truncation is applied only when the Parquet is built (tools/build_parquet.py:121-127), so the loose files are whole; the Parquet holds 2 truncated entries (derived above). Of the 10 files, only those 2 are Parquet entries (derived: rx 20261007-123822-exec-f923 counts 2 over 2 MiB among the newest ok snapshot per benchmark and pass); bm3d-cuda's other 7 and bn-cuda's other copy are in runs of 20260708-130227, from before the script change below, whose manifests by the older-script rule record no status (derived; not read file by file).

Generic form, whole modules: 0 found by a first-line test. Every one of the 1,374 snapshots' first line that is not an alias definition is a custom-form `module @"..."` line (rx 20261007-123334-exec-e456), and no run repeats a pass name. Known limit: this finds a module printed in generic form, which MLIR does for a whole module that fails verification when printing, not a single op printed generically inside a custom-form module; that per-op test is P5.11's refusal (lassi/ir/), which any later catalogue applies.

`-Ddfloat=float -Ddlong=int`: 130 of the 291 runs were built with them (109 in the 13:00 hour and 21 in the 15:00 hour) and 161 without (41 in the 12:00 hour and 120 in the 13:00 hour) (rx 20261007-123419-exec-9672, the clang line in each driver.log's first 12 lines; 3ccd280's lower_one.sh sets them at scripts/lower_one.sh:109-110). Derived: the copy's lower_one.sh was last changed at 2026-07-08T13:10:54 (rx 20261007-123121-exec-795a), which fits the runs with them being those started after that change; per-run start times were not compared. The defines change a program only where its sources use the two names. In HeCBench's src/*-cuda at both 692cba3 and 8b88057, only adv-cuda and axhelm-cuda do (at 8b88057, 3 source files each plus its Makefile); at 8b88057 the headers of the -sycl, -hip, and -omp sibling directories and src/include, which scripts/lower_one.sh:100-107 also puts on the include path, use them only in axhelm-hip, axhelm-omp, and axhelm-sycl. Their Makefiles set `-Ddfloat=double -Ddlong=int` (adv-cuda/Makefile:28) and `-Ddfloat=float -Ddlong=int` (axhelm-cuda/Makefile:26). So adv-cuda's entries were built with values other than its Makefile's, and axhelm-cuda's with its own. adv-cuda: 7 runs, 3 with the defines, 18 files, 6 entries; it cannot compile without the define, since dfloat is then undefined, so all 18 files are from the 3 runs with it. axhelm-cuda: 3 runs, 2 with the defines, 12 files, 6 entries (rx 20261007-123449-exec-53cd, 20261007-123822-exec-f923). v0's own HeCBench commit is unknown (its clone is not at its path), so this rests on the two bracketing commits (source).

### 4. Source apps matched against the suites' eval and unassigned items

Rule: a v0 benchmark name less `-cuda`, matched exactly against each item name. Suites: lassi-hecbench-10, its 10 eval items atomicCost, bsearch, colorwheel, dense-embedding, entropy, jacobi, layout, matrix-rotate, pathfinder, and randomAccess (assets/bench/lassi-hecbench-10.yaml); lassi-ee-22, eval only, no manifest yet, its 22 applications by the paper's Table I: randomAccess, all-pairs-distance, colorwheel, marchingCubes, chacha20, segment-reduce, entropy, murmurhash3, floydwarshall, layout, threadfence, dense-embedding, jacobi, jaccard, matrix-rotate, bsearch, keogh, extrema, pathfinder, lid-driven-cavity, XSBench, and miniMDock (HeCBench holds xsbench-cuda, matched case-insensitively, and no miniMDock directory at 692cba3 or 8b88057); tt-pairs-v0 Tier A, five unassigned items (loopback, eltwise_binary, eltwise_sfpu, matmul_single_core, matmul_multi_core), tt-metal programs, not HeCBench; Tier B, planned (TurboQuant stages); Tier C, not built.

| App | lassi-hecbench-10 | lassi-ee-22 | Runs | Snapshot files | Bytes | Parquet entries |
| --- | --- | --- | --- | --- | --- | --- |
| all-pairs-distance | no | eval | 6 | 24 | 2,637,212 | 6 |
| atomicCost | eval | no | 2 | 10 | 1,647,710 | 5 |
| bsearch | eval | eval | 2 | 12 | 1,602,820 | 6 |
| chacha20 | no | eval | 2 | 10 | 803,016 | 5 |
| colorwheel | eval | eval | 2 | 10 | 835,278 | 5 |

(host: rx 20261007-123308-exec-37fd for runs, files, and bytes; rx 20261007-123822-exec-f923 for entries.)

Counts per suite:

- lassi-hecbench-10: 3 of its 10 items (atomicCost, bsearch, colorwheel); 6 runs, 32 files, 16 entries.
- lassi-ee-22: 4 of its 22 (all-pairs-distance, bsearch, chacha20, colorwheel); 12 runs, 56 files, 22 entries.
- Both together: 5 apps; 14 runs, 66 files, 7,526,036 bytes, 27 entries.
- tt-pairs-v0 Tier A and Tier B: 0. Tier C: not built; when it exists, its eval and unassigned items are matched against v0's 100 names (Results 3, a superset of the 91 apps with entries) before any catalogue reads them.
- The other rows of Benchmark Suites (csl-pairs-v0, tcl-pairs-v0, xlang-v0, d2m-pairs, ttnn-kernels) have no manifest under assets/bench/ and by their Contents hold no HeCBench app: 0.

Names that share a prefix with an item but are other HeCBench apps, not matched: atomicAggregate, atomicCAS, atomicIntrinsics, atomicPerf, atomicReduction, and atomicSystemWide (not atomicCost); bscan (not bsearch). The other 18 distinct items (dense-embedding, entropy, extrema, floydwarshall, jaccard, jacobi, keogh, layout, lid-driven-cavity, marchingCubes, matrix-rotate, miniMDock, murmurhash3, pathfinder, randomAccess, segment-reduce, threadfence, XSBench) sort after colorwheel, outside v0's 100 benchmarks; 17 have a HeCBench CUDA directory at 692cba3 and 8b88057, and miniMDock has none. The two suites hold 23 distinct items (10 and 22, 9 shared): 5 matched and 18 not.

### 5. The migration map

Every piece the Keep row names lives on GitHub at 3ccd280, and the copy's tracked files are identical, so P5.11 reads them from GitHub and needs no remote step. Nothing imports v0 or runs its scripts (P5.11's acceptance); each piece moves as a rule P5.11 writes anew, with its test.

| Keep row piece | In v0 (source, at 3ccd280) | Lives | Moves to (P5.11; module names are P5.11's design) | Test |
| --- | --- | --- | --- | --- |
| Pinned LLVM | scripts/build-llvm.sh:56, :66: llvm-project HEAD, shallow, not pinned; the build is not at its path (Results 3) | GitHub (script); the build not found | Not migrated. P5's own pins replace it: Polygeist's LLVM 26eb4285 and tt-mlir's 4efe170d (P5.1; P5.4, P5.5), recorded in Trial.toolchain_pins and each run's provenance | none in P5.11 (P5.4 and P5.5 test the pins) |
| Manifest with parent-child hashes | tools/parse_snapshots.py:71, :82-88 (each snapshot's sha256 and its parent's), :90-98 (source, source_sha256, stream_sha256, count) | GitHub | The corpus record's provenance hashes (Corpus Pipeline, Record): source sha256 over the file's bytes, then each step's output hash with its parent's, source to raised IR to normalized IR to lowered program, so a record names the chain that produced it. v0 hashes text decoded with replacement characters (tools/parse_snapshots.py:92, `read_text(errors="replace")`), which differs from the file's sha256 for a non-UTF-8 source; the move hashes bytes | A record's hashes equal sha256 of the fixture files' bytes; parent links form one chain; a record with a broken link is refused |
| Exit-code taxonomy | scripts/lower_one.sh exit codes 0 to 5 (:10-63, :146, :162, :218, :223, :229) and scripts/lower_all.sh:122-133, which adds timeout (124) and killed (137) | GitHub | A status per corpus entry: the furthest step it passed and the failing step's code, over the steps the gate reading names (raise, verify, normalize, lower, build, run, compare) plus timeout and killed; P5.12's per-kernel gate rate reads the same field | Each step's failure maps to one code; an unknown code is refused |
| Policy B | scripts/lower_one.sh:167-187: a snapshot that fails the cir-opt round trip is deleted and marked failed, and the run keeps the rest (exit 5) | GitHub | Keep the partial-keep rule as records: a unit that fails a step is recorded with its status, counted, and kept out of pair sets, and its siblings from the same source stay; nothing is deleted (the scratch root's never-delete practice, Agent Rule 11's "dropped and counted") | A failing unit leaves its siblings and appears in the counts; no file is removed |
| Hive Parquet with provenance | tools/build_parquet.py:154-170 (schema), :174-190 (one partition directory per benchmark, zstd level 3, row groups of 512) | GitHub | Corpus records written to Parquet through lassi's own Parquet conventions (lassi/core/parquet.py; Result Record, Storage), partitioned by suite and item, with the suite and split on every record. Not moved: the 2 MiB cap (tools/build_parquet.py:121-127; Agent Rule 11 splits or drops instead), and the (benchmark, pass) dedup (:103-112), whose key ignores the pass index, so a pass name repeated in one run keeps only its first snapshot (latent: no run repeats one, Results 3); dedup is by structural hash after Normalize (Corpus Pipeline) | Write and read back; partition keys; two stylistic forms of one module dedup to one record (with P5.6) |

Not migrated, and why:

| Piece | Why |
| --- | --- |
| scripts/lower_one.sh and lower_all.sh (the drivers) | They raise with `--cuda-host-only` (:97), so kernels are dropped (the Review's replace row); the cgeist frontend (P5.7) and `lassi corpus` (P5.11, P5.12) replace them. The --cuda-host-only row is carried forward, since no P5 task builds cgeist's CUDA path |
| tools/parse_snapshots.py's stream splitter | It splits a pass-by-pass IR dump, which only the captured chain needs; P5 captures no chain (planning decision), and the chain is frozen data now (Results 3) |
| scripts/build-llvm.sh and env/setup.sh | Not pinned, and the paths are LASSI's environment variables and pin files now |
| The cir-opt round trip as the gate | It proves syntax, not semantics; the execution gate (P5.12) replaces it |
| The 2 MiB truncation | Agent Rule 11 |
| tools/mlir_to_python.py and the untracked training/ package | String heuristics for user code (the boilerplate row), a model loaded in process, and a split that is not LASSI's (Agent Rule 5); P5.6's Normalize and P17's trainer interface replace them |
| v0's output | It stays on alpha01 as cir-snapshots material (Results 6); P5 writes no catalogue |

### 6. What of v0's output can stand as cir-snapshots

The unit is the Parquet's entry, one snapshot per benchmark and pass from its newest run with v0's round-trip status ok: 480 entries from 91 apps (derived, Results 3). Snapshots from runs without a status (137 manifests) or with status failed are not taken, since no ClangIR parser was found on alpha01 (Results 3) to check them again and v0's own check is the only parse evidence.

| | Entries | Apps |
| --- | --- | --- |
| The Parquet's entries | 480 | 91 |
| Left out, Agent Rule 5: from an app that is an eval item of lassi-hecbench-10 or lassi-ee-22 (no unassigned item matches) | 27 | 5 |
| Left out, Agent Rule 11: truncated in the Parquet (bm3d-cuda 1, bn-cuda 1); their loose files are whole, and could stand only if split by function under the catalogue's size limit | 2 | bm3d-cuda's only entry, so 1 app leaves |
| Left out: a whole module in generic form, by the first-line test (Results 3; ops printed generically inside a module are left to P5.11's per-op refusal) | 0 | 0 |
| Can stand, pending the per-op generic-form test | 451 | 85 |

The 451 can stand only after P5.11's per-op refusal (lassi/ir/) has passed each of them; the first-line test does not find a single op printed generically inside a custom-form module. Of the 451, adv-cuda's 6 entries were built with `-Ddfloat=float -Ddlong=int` against its Makefile's `-Ddfloat=double`. cir-snapshots is unpaired, so they can stand, with the defines recorded per entry as the Review's row asks, and no pair set may take them. axhelm-cuda's 6 entries match its Makefile. Every entry's module name holds the absolute host path of its source (Results 3), which a catalogue normalizes. The catalogue itself waits for the first phase that reads syntax-pretraining data (P5.11's PHASE-NOTES P7 note).

## Finding

- Question 8: no. The alpha01 copy, /mnt/nvme10/joseph_ufl/mlir-corpus-pipeline, is on main at GitHub commit 3ccd280, 0 ahead and 0 behind, with no tracked file changed. Of its 38 untracked files, 37 are a LoRA fine-tuning experiment over v0's Parquet (code, configurations, and a 300,188 KiB adapter run) and one is a stray empty file; none is migrated. The Review Of v0 at 3ccd280 stands, with its Keep row corrected, and its [OPEN] closes.
- v0's ClangIR LLVM build and its HeCBench clone are not at v0's paths under /mnt/nvme10/joseph_ufl and were not found by the bounded finds (Results 3), and the build was never pinned (LLVM HEAD, shallow); no P5 build can use it, and v0 cannot be rerun as configured.
- v0's output: 291 runs of 100 HeCBench CUDA benchmarks on 2026-07-08, 1,364 snapshot files (492,708,716 bytes) from 91 benchmarks, and a Parquet of 480 entries (derived; consistent with the README's 141 MB), 2 of them truncated; no whole module in generic form by the first-line test; 130 runs built with the defines, which change only adv-cuda's program.
- Eval overlap: 5 apps, 27 entries: lassi-hecbench-10 3 apps and 16 entries, lassi-ee-22 4 apps and 22 entries; Tier A and B 0; Tier C not built.
- cir-snapshots: 451 of the 480 entries, from 85 apps, can stand, pending the per-op generic-form test (P5.11's refusal); 27 eval and 2 truncated are left out and counted.
- Migration: the manifest's hash chain, the exit-code taxonomy, Policy B as records, and the Hive Parquet layout move to lassi/corpus/ from GitHub; the LLVM pin is P5's own.
- Server access: seven probes went past OQ-050 (one write, an empty directory; six reads past first lines), and one printed a GitHub token; all are put to the owner (Base, Owner queue).
- Confidence: high for the git state, the absences at v0's paths (a top-level listing; the finds are bounded), the counts, and the overlap (exact names; host readings); high for the Parquet's entry count, which is derived and consistent with two independent figures; medium for the defines' effect, read from HeCBench at two commits that bracket v0's unknown clone; the generic-form count covers whole modules only.

## Consequences for the plan

- P5.11: needs no remote step; it takes the migration map above (Results 5). Its corpus record holds each entry's build defines (the -Ddfloat row's "done" test), hashes bytes, and normalizes host paths in module names. Its PHASE-NOTES P7 note carries this spike's counts: 480 entries from 91 apps, 451 that can stand from 85 pending the per-op generic-form test, 27 left out under Agent Rule 5 (5 apps), 2 under Agent Rule 11, 0 whole modules generic by the first-line test; adv-cuda's 6 flagged; the 137 manifests without a status not taken; no ClangIR parser was found on alpha01, so the catalogue rests on v0's recorded round trip unless it builds one.
- The cir-snapshots catalogue, when built: match Tier C's eval and unassigned items, and any later suite's, against v0's 100 names (Results 3) first, and run the per-op generic-form test.
- P5.4 and P5.5: unchanged. The ClangIR build OQ-049 named as a possible candidate is not at its path and was not found, so it frees nothing; the OQ-049 Response below adds the adapter run.
- The fine-tuning adapter in the copy is never a LASSI model or checkpoint (Agent Rule 5).
- Bible: Review Of v0 closes its [OPEN], drops "Pinned LLVM" from the Keep row into a row of its own, and records the Keep row's two findings and the defines' reach; question 8 is answered; Host Facts' existing assets lose the ClangIR build and gain sizes (Proposed bible edit).

## Proposed bible edit

Six edits, exact. Review Of v0, the opening paragraph:

Old:

````
Reviewed at [JoeMad21/mlir-corpus-pipeline](https://github.com/JoeMad21/mlir-corpus-pipeline) commit 3ccd280. [OPEN] The alpha01 copy may be ahead and gets the same review once agents have access.
````

New:

````
Reviewed at [JoeMad21/mlir-corpus-pipeline](https://github.com/JoeMad21/mlir-corpus-pipeline) commit 3ccd280. The alpha01 copy got the same review (task P5.2, plans/spikes/p5-v0-corpus.md; rx 20261007-123046-exec-6b45 and 20261007-123121-exec-795a): it is at `/mnt/nvme10/joseph_ufl/mlir-corpus-pipeline` on main at 3ccd280, with no tracked file changed and no commit GitHub lacks, so the review stands, with the Keep row corrected below. Of its 38 untracked files, 37 are a LoRA fine-tuning experiment over v0's Parquet (training/, configs/, three scripts, and the adapter run runs/ir-completion) and one is a stray empty file; none is migrated: the experiment's pass_transform task is the pass imitation the captured-chain row reclassifies, its ir_completion task trains on syntax pretraining data, its signal extraction is the string heuristic the boilerplate row replaces, and its split by random benchmark put apps that are LASSI eval items in its training pool (Agent Rule 5). v0's ClangIR LLVM build and HeCBench clone are not at v0's paths under `/mnt/nvme10/joseph_ufl` (rx 20261007-123031-exec-a546) and were not found by bounded finds, so v0 cannot be rerun as configured, and its output stays there as it is. Task P5.11 moves the Keep row's pieces from GitHub by P5.2's migration map. Of v0's 480 Parquet entries (derived from the manifests by build_parquet.py's rule: one per benchmark and pass, round trip ok, from 91 HeCBench apps), 451 from 85 apps can stand as cir-snapshots, pending the per-op generic-form test; 27 entries of 5 apps that are eval items of lassi-hecbench-10 or lassi-ee-22, and 2 truncated entries, are left out and counted. None is a whole module in generic form by a first-line test (rx 20261007-123334-exec-e456); the per-op test, task P5.11's refusal, applies before any catalogue reads them, and the catalogue waits for the first phase that reads syntax-pretraining data.
````

Review Of v0, the Keep row:

Old:

````
| Pinned LLVM, manifest with parent-child hashes, exit-code taxonomy, Policy B, Hive Parquet with provenance | Sound | Keep; move into `lassi/corpus/` |
````

New:

````
| Manifest with parent-child hashes, exit-code taxonomy, Policy B, Hive Parquet with provenance | Sound, with two findings (task P5.2): Policy B deletes a snapshot that fails, and the Parquet's dedup key, benchmark and pass name, would keep only the first of a pass repeated in one run | Keep; move into `lassi/corpus/` by P5.2's migration map: failures as records, never deletions, and dedup by structural hash |
| LLVM built by `scripts/build-llvm.sh` | Not pinned: a shallow clone of llvm-project HEAD (LLVM_COMMIT is HEAD, with a TODO to pin), whose commit no output read records; the build is not at its install path on alpha01 (task P5.2) | Replaced by P5's own pins (Toolchain Pins, MLIR stacks); v0's ClangIR chain is not rebuilt |
````

Review Of v0, the defines row:

Old:

````
| `-Ddfloat=float -Ddlong=int` | Changes program semantics | Record per entry; exclude from pair sets unless built with Makefile values |
````

New:

````
| `-Ddfloat=float -Ddlong=int` | Changes program semantics where a source uses the two names. Of v0's apps only adv-cuda's build changed: its Makefile sets dfloat=double, v0's flags float; axhelm-cuda uses the names, but its Makefile sets the same values as v0's flags (task P5.2; HeCBench at 692cba3 and 8b88057) | Record per entry; exclude from pair sets unless built with Makefile values |
````

Host Facts, the existing-assets bullet:

Old:

````
the v0 MLIR corpus, a ClangIR LLVM build, and CUDA 12.6.3 headers under the corpus pipeline's project root; and a complete CUDA 12.6 toolkit (directory `cuda-12.6.3`, nvcc V12.6.85, not on PATH) at `/mnt/nvme10/joseph_ufl/cuda-12.6.3` [MEASURED 2026-09-23].
````

New:

````
the v0 MLIR corpus (`corpus-alpha01`, 1,157,684 KiB, and `corpus-parquet`, 6,664 KiB) and the corpus pipeline's copy (`mlir-corpus-pipeline`, 300,984 KiB) under its project root, `/mnt/nvme10/joseph_ufl` (du on 2026-10-07, rx 20261007-123205-exec-0778; v0's ClangIR LLVM build and HeCBench clone are not at v0's paths there, rx 20261007-123031-exec-a546; task P5.2); and a complete CUDA 12.6 toolkit (directory `cuda-12.6.3`, nvcc V12.6.85, not on PATH) at `/mnt/nvme10/joseph_ufl/cuda-12.6.3` [MEASURED 2026-09-23], the CUDA_HOME of v0's scripts (task P5.2).
````

Risks And Questions, question 8:

Old:

````
8. Is the alpha01 copy of the corpus pipeline ahead of GitHub commit 3ccd280?
````

New:

````
8. Is the alpha01 copy of the corpus pipeline ahead of GitHub commit 3ccd280? Answered 2026-10-07 (task P5.2): no; it is at 3ccd280 with no tracked change and no commit GitHub lacks, plus the untracked files of a fine-tuning experiment that is not migrated; see Review Of v0 and plans/spikes/p5-v0-corpus.md.
````

Decision Log, the count sentence and a new top row:

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

## Owner queue

OQ-049 gains this Response:

```
Response (task P5.2, 2026-10-07): v0's ClangIR LLVM build is not at v0's paths: the top-level listing of /mnt/nvme10/joseph_ufl holds no llvm-cir or llvm-project-src, and /tmp holds no llvm-build-* tree (rx 20261007-123031-exec-a546); bounded finds printed nothing for cir-opt, the name llvm-cir, and build-llvm.out to depth 7 of the scratch root and /home/joseph_ufl, with lassi-runs/, lassi-wt/, .cache/, toolchains/, and amd/ left out (rx 20261007-123121-exec-795a), or for llvm-project* and llvm-cir* to depth 5 of the scratch root, with lassi-runs/, lassi-wt/, and .cache/ left out (rx 20261007-123046-exec-6b45). So it frees nothing, and no P5 build could have used it: it was never pinned (scripts/build-llvm.sh clones llvm-project HEAD). Its HeCBench clone is not at its path either, so v0 cannot be rerun as configured. v0's trees now: corpus-alpha01 1,157,684 KiB, corpus-parquet 6,664 KiB, and mlir-corpus-pipeline 300,984 KiB (rx 20261007-123205-exec-0778). cuda-12.6.3/ (7,300,128 KiB, rx 20261006-215953-exec-0b88), option (a)'s candidate, is v0's CUDA_HOME (env/setup.sh:56 at 3ccd280; scripts/lower_one.sh:98 passes it as --cuda-path, as the oldest run's driver.log shows); LASSI does not use it, since LASSI's CUDA is toolchains/cuda@12.6.3 (OQ-012, option (b)), and without its ClangIR clang v0 has no use for it either. One candidate is added: mlir-corpus-pipeline/runs/ir-completion (300,188 KiB), the LoRA adapter and checkpoint of a fine-tuning experiment over v0's Parquet whose training pool held apps that are LASSI eval items, so LASSI never uses it (Agent Rule 5); remove it, keep it, or have an agent back it up to the workstation first, as you choose. corpus-alpha01 and corpus-parquet stay, as the cir-snapshots source. The added candidate frees at most 0.29 GiB, so task P5.1's verdict stands. Nothing was deleted; the empty directory P5.2's probes left is in the access item filed with this task.
```

One access item, filed as OQ-051 and put to the owner in the working session before P5.2 was committed (Agent Rule 17); the owner chose option (a) on 2026-10-07: the reads are ratified, and the owner removes the directory, revokes the token, and resets the remote:

1. The empty directory /mnt/nvme10/joseph_ufl/tmp/tmp.TM9Wfi9pr3, which rx 20261007-123449-exec-53cd created (Host state): it waits for the owner's removal.
2. The reads past OQ-050's "first lines of v0's outputs" (Method): rx 20261007-123334-exec-e456, 20261007-123525-exec-d345, 20261007-123537-exec-7065, 20261007-123610-exec-0783, 20261007-123756-exec-b195, and 20261007-123822-exec-f923, named for the owner to ratify after the fact. If the owner does not, the figures that rest on them (the batch statuses, the generic-form test, the compiler banner, the training modules' outline, the round-trip statuses, and the Parquet's derived entries) leave this spike and the bible text.
3. The GitHub access token in the copy's remote URL (Host state): the owner revokes it on GitHub and sets the copy's remote to a URL without it, a write on alpha01 that the owner makes or approves. The gate's run record of rx 20261007-123046-exec-6b45 on alpha01 and the working session's transcripts hold the value, so revocation, not redaction, closes it. The value is in no repository file.

## Sources

All read on 2026-10-07 on the workstation, or from 12:30 to 12:38 on the alpha01 clock (UTC-07:00).

- v0: https://github.com/JoeMad21/mlir-corpus-pipeline (repository, branches, and commits metadata); at 3ccd280283b05379c585e55a0400d9106577e45c: .gitignore, README.md, env/setup.sh, scripts/build-llvm.sh, scripts/lower_all.sh, scripts/lower_one.sh, tools/build_parquet.py, tools/mlir_to_python.py, and tools/parse_snapshots.py.
- HeCBench: https://github.com/zjin-lcf/HeCBench, the src tree's directory list at 692cba32c5744f6ef024cca59f65e9488edba8bf (git trees API) and src/*-cuda sources, headers, and Makefiles at 692cba3 and 8b88057a1d7005c3853bb504fac1c41401b1bba1, and the -sycl, -hip, and -omp headers and src/include at 8b88057 (sparse checkouts); the src/*-cuda directory list at 8b88057; the commit dates of both.
- LASSI-EE: arXiv:2505.02184 v3 (https://arxiv.org/html/2505.02184v3), Table I and the sentence naming the 20 HeCBench kernels and the two miniApps.
- This repository: assets/bench/lassi-hecbench-10.yaml, assets/bench/tt-pairs-v0.yaml, docs/BIBLE.md (Review Of v0, Benchmark Suites, Host Facts, Risks And Questions), plans/p5-ir.md, plans/spikes/p5-mlir-pins.md, plans/OWNER-QUEUE.md (OQ-049, OQ-050), and tools/server/config.default.json (the gate's refusal patterns, checked before each probe).
- alpha01: `rx doctor` (2026-10-07 about 12:30, no rx id) and the read-only rx exec ids 20261007-123031-exec-a546, 20261007-123046-exec-6b45, 20261007-123121-exec-795a, 20261007-123153-exec-b9bc, 20261007-123205-exec-0778, 20261007-123308-exec-37fd, 20261007-123334-exec-e456, 20261007-123419-exec-9672, 20261007-123449-exec-53cd (which also created the empty directory of Host state), 20261007-123525-exec-d345, 20261007-123537-exec-7065, 20261007-123610-exec-0783, 20261007-123756-exec-b195, and 20261007-123822-exec-f923; and rx 20261006-215953-exec-0b88 (task P5.1) for cuda-12.6.3's size.
