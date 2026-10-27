# Spike: bale as LASSI benchmark data (owner request of 2026-10-26)

- Request (the owner, 2026-10-26): "I want to add the BALE benchmark (not really a benchmark but you get the idea)
  into the dataset."
- Date: 2026-10-26, on the workstation, Git Bash.
- Base: this worktree, C:/dev/lassi-p17, branch p16-adversarial at 3d4e0e988c444b82f153dcf5cff3f749bc8efe30
  (bible mirror at master revision 435). Built as task P17.40 (assets/bench/bale-v0.yaml), which differs from
  the sketch below: no shared bale_graph.h (each triangle program builds its own graph), and omp builds with
  gcc-openmp, which tags an omp candidate with no OpenMP construct.
- What ran: reads only. The GitHub REST API through `gh api` (repository, commits, branches, tags, releases, tree,
  forks, compare); one `git clone --no-checkout` of the upstream repository into a session scratch directory outside
  the repository, whose blobs were read with `git cat-file blob` (nothing was built or run from it); `sha256sum` and
  `grep` over those blobs; two web searches; one read of Debian's openmpi packaging. No rx command ran; nothing
  touched alpha01, the LASSI WSL distro, or a device. Facts about alpha01 come from earlier spikes and the bible,
  with their own rx ids.
- Device: none.
- Labels: every claim carries one of read (seen in the cited file, line, or command output), inferred (reasoned from
  read facts; the reasoning is given), or unknown (not established here). Nothing here is [MEASURED]; sizes,
  counts, and line estimates for work not yet done are PROJECTED.
- Citations: `B:<path>:<line>` is github.com/jdevinney/bale at 1b8f673b56645b2bd74a8af6213462bbc9d559fe; `L:<path>:<line>`
  is this repository at 3d4e0e9; `bible:<line>` is docs/BIBLE.md at 3d4e0e9. No bale program text is quoted
  beyond identifiers and short idioms (OQ-018 practice).

## Question

What would it take to add bale to LASSI's benchmark data? First, which "bale" is meant, under what license, at which
commit, and what it holds: each app, its languages and parallel models, whether a serial version exists, its
inputs, its outputs, and how its own checks verify them, and what building it needs against what LASSI's hosts
have. Then, mapped onto LASSI: which apps can become suite items, in which translation directions, with what
held-out inputs, outputs, and tolerances; whether each item's references come from bale itself or need programs
written here, and how much; which files a model writes and which are support; how splits would be assigned; and a
sketch of a manifest, assets/bench/bale-v0.yaml, for two or three apps.

Why it matters: no task exists yet. The owner's request adds a suite, which changes a project's scope (AGENTS.md,
Authority), so the phase or tasks that would build it wait on the owner's answers below: which apps and directions
first, which splits (Agent Rule 5 makes an eval label irrevocable), and whether bale's text may be copied into this
public repository (OQ-018 precedent).

## Classification

Mixed. Identity, license, commit, layout, the apps, and the fit to LASSI's harness are factual and are answered
here by reading. Whether and how to add bale (scope, directions, splits, copy or fetch) is the owner's choice:
section 12 drafts the owner-queue item. No [OPEN] bible item covers bale, so no bible edit is due now; section 13
drafts the edit that would follow the answer.

## Summary

- bale is github.com/jdevinney/bale (read): "a vehicle for discussion" for parallel programming productivity whose
  README says "bale is not a collection of benchmarks" (B:README.md:3, :15), which matches the owner's words. No
  other project named bale or BALE fits better (section 1).
- License: an Institute for Defense Analyses copyright with BSD-3-Clause-style terms plus a DFARS reproduction
  clause (B:LICENSE:1-31, read). It allows use and a tracked copy, verbatim or modified, if the notice, conditions,
  and disclaimer are kept (B:LICENSE:9-18, read). One header carries Knuth's own notice instead (section 2).
- Pin: the default branch master at 1b8f673b56645b2bd74a8af6213462bbc9d559fe, committed 2022-06-26T10:47:11-07:00
  ("typo"); every source file dates from the first commit of bale 3.0 on 2020-10-23; no tags or releases (read).
- Apps (read): histo, ig (index gather), randperm, transpose_matrix, permute_matrix, triangle, toposort, sssp, and
  sparse_matrix_io in bale_classic (UPC, or C with OpenSHMEM 1.4, each with AGP, exstack, exstack2, and conveyor
  models), and a serial C version of all but sparse_matrix_io, plus unionfind, in src/other_serial/C, built with
  plain gcc. Chapel and Rust cousins cover subsets.
- Fit (inferred): no bale program can be a LASSI reference as written. Each draws its own inputs from a seed with
  Knuth's generator (graph apps may read a Matrix Market file), writes no result file, reports to stderr, and
  checks itself or not at all; the parallel models
  need UPC or OpenSHMEM and a multi-PE launcher, which no LASSI executor has. Every item therefore needs programs
  written here on the lassi_io harness, following bale's serial kernels, as tt-pairs-v0 Tier A and adv-seeded-v0
  do.
- Apps that fit binary_io with a unique, exact answer (inferred): histo, ig, triangle, transpose_matrix,
  permute_matrix, sssp, and unionfind (with a canonical labeling). toposort and randperm have many correct answers
  and need an oracle that checks a property, which LASSI lacks. sparse_matrix_io is distributed file I/O and does
  not fit.
- Directions runnable with today's toolchains (inferred): serial C++ <-> OpenMP C++ on the host CPU (cpp <-> omp),
  both built by gcc-native, whose pinned flags already hold -fopenmp, and run by the native executor. C as a
  source language (bale's own) waits for P12.6's C toolchain. CUDA would be compile-only (no GPU host), and TT
  directions for these irregular apps are research-grade.
- Recommendation (section 12): v0 = histo, ig, and triangle in cpp <-> omp, programs written here, with a static
  check that tags an omp candidate holding no OpenMP construct; every item unassigned until training exists; fetch
  only, no bale text tracked.

## 1. Identity: which bale

### Commands and outputs (trimmed)

```
$ gh api repos/jdevinney/bale --jq '{full_name, description, default_branch, license: .license, created_at,
    pushed_at, updated_at, stargazers_count, archived, size, language, homepage}'
{"archived":false,"created_at":"2020-10-23T00:53:27Z","default_branch":"master","description":null,
 "full_name":"jdevinney/bale","homepage":null,"language":"C","license":{"key":"other","name":"Other",
 "spdx_id":"NOASSERTION","url":null},"pushed_at":"2022-06-26T17:47:15Z","size":2694,"stargazers_count":29,
 "updated_at":"2026-10-12T17:22:33Z"}

$ gh api repos/jdevinney/bale/commits/master --jq '{sha, date: .commit.committer.date, author: .commit.author.name,
    message: .commit.message}'
{"author":"John R. Gilbert","date":"2022-06-26T17:47:11Z","message":"typo",
 "sha":"1b8f673b56645b2bd74a8af6213462bbc9d559fe"}

$ gh api repos/jdevinney/bale/branches --jq '.[] | {name, sha: .commit.sha}'
{"name":"master","sha":"1b8f673b56645b2bd74a8af6213462bbc9d559fe"}
$ gh api repos/jdevinney/bale/tags        -> (empty)
$ gh api repos/jdevinney/bale/releases    -> (empty)

$ gh api "repos/jdevinney/bale/git/trees/1b8f673b56645b2bd74a8af6213462bbc9d559fe?recursive=1" --jq '.truncated, (.tree|length)'
false
435

$ git -c core.autocrlf=false clone --quiet --no-checkout https://github.com/jdevinney/bale.git bale
$ git -c core.autocrlf=false checkout --quiet 1b8f673b56645b2bd74a8af6213462bbc9d559fe
error: invalid path 'example_matrices/kron_1:3x4x5.mm'
$ git rev-list --count HEAD
5
$ git log --format='%h %cI %s' --stat   (trimmed to the files each commit changed)
1b8f673 2022-06-26T10:47:11-07:00 typo                     docs/README.md
a72284f 2022-06-26T10:46:12-07:00 fix dates                docs/README.md
3668b57 2022-06-26T10:18:39-07:00 added Saint-Girons talk  README.md, docs/Bale-StGirons-Final.pdf, docs/README.md
bca9e07 2020-10-23T09:25:07-04:00 Update README.md         README.md
2b2498d 2020-10-23T09:02:27-04:00 first commit of bale-3.0 (every source file, LICENSE, example_matrices, ...)
```

The Windows checkout fails on one file whose name holds a colon; the 357 other blobs of the 358 were written out
with `git cat-file blob <commit>:<path>` for reading (read).

### Finding

- bale is github.com/jdevinney/bale (read). Its README calls it "first and foremost, *a vehicle for discussion* for
  parallel programming productivity" and says "bale is not a collection of benchmarks" (B:README.md:3, :15), the
  owner's "not really a benchmark". bale_classic's README repeats both (B:src/bale_classic/README.md:17, :31).
- Version history (read, B:README.md:45-68): 1.0.0 May 2018, 2.0.0 Dec 2018 (transpose, randperm, permute_matrix),
  2.1.0 Sep 2019, 3.0.0 Oct 2020 (serial C, Rust, parallel Rust, and Chapel cousins; SSSP; geometric graphs).
  Contact bale@super.org (B:README.md:72).
- Other candidates (read, 2026-10-26): two web searches ("BALE benchmark code translation parallel"; a quoted
  search for the README's phrase) found no benchmark or project named BALE; GitHub repository search for "bale"
  returns unrelated projects (balena, a messenger VPN). The forks seen through the API are forks of this
  repository: bradcray/bale, hplp/bale-apps, and kaushikravichandran/bale-gpu are 0 commits ahead of master;
  dpang314/bale is 1 ahead; singhalshubh/bale is 59 ahead, its changed files being src/bale_classic/convey and two
  perlmutter.patch files (`gh api repos/jdevinney/bale/compare/master...<owner>:master`). None fits better.

## 2. License and copying

- Text (read, B:LICENSE:1-31): "Copyright (c) 2020, Institute for Defense Analyses", a DFARS 252.227-7013 and
  252.227-7014 reproduction clause for the U.S. Government (:3-5), "All rights reserved" (:7), then the three
  conditions of the BSD 3-clause license (:9-18: keep the notice, conditions, and disclaimer in source copies;
  reproduce them in binary distributions; no use of the holder's name to endorse) and its disclaimer (:20-31).
  GitHub's detector reports "Other" (NOASSERTION), which is inferred to come from the extra DFARS lines.
- B:src/other_parallel/Rust/convey/LICENSE is byte-identical to it; B:src/bale_classic/convey/LICENSE is the same text
  with `//` comment markers (cmp, read).
- Every .c and .h file of src/other_serial/C carries an IDA header pointing at the top-level LICENSE (for example
  B:src/other_serial/C/histo.c:1-9), except knuth_rng_double_2019.h (grep, read).
- B:src/other_serial/C/knuth_rng_double_2019.h:1-2 carries Knuth's own notice: public domain and freely copyable
  "AS LONG AS YOU MAKE ABSOLUTELY NO CHANGES". Whether bale's copy already differs from Knuth's published file is
  unknown (not compared). No item proposed here needs this generator (section 7), so the question does not arise.
- Our use (inferred): reading, fetching at a pinned commit, writing our own programs that follow bale's algorithms,
  and running them are all within the license. A tracked copy, verbatim or modified, is allowed if each copied
  source keeps IDA's notice, the conditions, and the disclaimer (B:LICENSE:11-12). Whether to track any bale text
  is a policy choice for this public repository (OQ-018 kept upstream LASSI text out; Tier A and the HeCBench
  suites name upstream files by path and sha256 only, bible:663, bible:677).
- ASCII (read): grep for bytes above 0x7F finds only B:src/bale_classic/README.md among the files read; no file of
  src/other_serial/C has a CR line ending. The text-policy checker's plain-ASCII rule would accept a copy of the
  serial C files (inferred; its owner pattern list was not run against them).
- A full vendored copy could not be checked out on Windows as it stands: B:example_matrices/kron_1:3x4x5.mm has a
  colon in its name (the checkout error above, read).

## 3. Pin

- master, 1b8f673b56645b2bd74a8af6213462bbc9d559fe, committed 2022-06-26T10:47:11-07:00 (17:47:11Z) by John R.
  Gilbert, message "typo" (read). It is the only branch; there are no tags or releases (read).
- The last three commits change only README.md and docs/ (read, the stat above). Every source file is as committed
  in 2b2498d on 2020-10-23, "first commit of bale-3.0" (read). The repository is not archived; last push
  2022-06-26 (read).
- Repository size 2,694 KB by the API (read). tools/fetch_bench.py fetches by commit sha into a sparse, shallow
  checkout of only the listed paths (L:tools/fetch_bench.py:194-210), so the colon-named file is never fetched
  unless a manifest names it (inferred), and on Linux the name is legal anyway.
- sha256 of the blobs at the pin (read; `git cat-file blob` piped to `sha256sum` on the workstation):

```
8391f5a020ff67d6d3bcb55558bec37b3a97ee4f6176d8cbc7d605dd53c3ccf2  src/other_serial/C/histo.c
2cb08618a08eddb9979ca27c5cb5b259325ca34d766eae34f0a9ef961c2144d0  src/other_serial/C/ig.c
07b3d5ba0858fb0667a75b6044e5e7c14741b0c45600582881d7815886a4237f  src/other_serial/C/triangle.c
c5be537e2bb8f80cf3c2167a5081a64d00bccb9219a1459ee0d47ae3224245c3  src/other_serial/C/sssp.c
a2d811ce4a73adf291e00e9d6c57392096ca687f2ac75ffb420f14704bf2eafc  src/other_serial/C/sssp_dijsktra.c
6f806951ac2d64c9e69be552eab1813aad7ebe50168b91008cce75341f9c7ebf  src/other_serial/C/transpose_matrix.c
24dc98a9209fe5da6c325f147fcb44c64ea2054a2eac398af663fcb0ff157443  src/other_serial/C/permute_matrix.c
1e9bbafd2bc944c001195b2f10aa70bc6ef8253314dfc9841a6944deec9cbeb2  src/other_serial/C/toposort.c
ed2116e527bebdcd39f53f779f5cb3f9b3c9d8f5a805e1ac35ce056fb4f8ee4a  src/other_serial/C/unionfind.c
fac775d6d40565102ce0d347420b40121fa8dfe8576b01f022cf542e4e829686  src/other_serial/C/randperm.c
b9c9e54ff74f071d2a0e03c34be21b8dff1442671881877db49475175118768c  src/other_serial/C/spmat_utils.c
beacbb5e96a4988de43c26facfacaf111f48c997225ebac6cf14fd016f9add95  src/other_serial/C/spmat_utils.h
b6a7f0c2a84390f28d8882bd92a4f54dc3d56429e52c7321c110031707421903  src/other_serial/C/std_options.c
2b08090f83778dcd050291a9abda0e37ede03466fb094ea3ec9e0d3b3b1c2269  src/other_serial/C/std_options.h
c818586aa8aae21b35c7009b32e532d85921558749c596af3898bee08c885853  src/other_serial/C/default_app_opts.h
c72cf83124f00361cf0ad697a4677bfdb74f33c2e23ecf7a0c91e71466684e23  src/other_serial/C/knuth_rng_double_2019.h
1968cb88865ed364ffc74ac07efc2e50476ada707c1367e0c4232da7467788fd  LICENSE
f6c9cbce6c75ce511d16dbd73624635e68a4ba9a0bbcefe863d9328873ad6226  README.md
```

## 4. Layout (read, the tree at the pin)

- `src/bale_classic/`: the original bale, "implemented in UPC or C and SHMEM" (B:README.md:19). libgetput (remote
  gets, puts, and atomics over UPC or SHMEM), exstack (exstack and exstack2), convey (conveyors), spmat (the
  distributed sparse matrix library, where permute, transpose, and randperm live), std_options, and apps/ with one
  directory per app (B:src/bale_classic/README.md:51-58). Autotools per subpackage, a make_bale script, run_apps.py,
  pytest tests, and Dockerfiles for Clang UPC, GNU UPC, OpenMPI OSHMEM, and Sandia OpenSHMEM
  (B:src/bale_classic/INSTALL.md:21-33, :36-54).
- `src/other_serial/C/`: "A textbook, C, implementation of the apps in bale", self-contained, built by one Makefile
  with gcc (B:src/other_serial/C/README.md:13, :29-30, :39-50; B:src/other_serial/C/Makefile:16-19). 181,845 bytes
  of .c and .h (wc, read).
- `src/other_serial/Rust/`: "a partially complete project" (B:src/other_serial/Rust/README.md:1): sparsemat,
  toposort, delta_stepping crates, depending on clap 2, rand 0.7, and regex 1 from crates.io
  (B:src/other_serial/Rust/toposort/Cargo.toml:23-27; .../sparsemat/Cargo.toml:23-25).
- `src/other_parallel/Rust/`: histo, ig, permute, randperm, toposort, triangle, delta_stepping on Rust conveyors over
  SHMEM, depending on a sibling checkout of convey_private outside this repository (B:src/other_parallel/Rust/README.md:14-38;
  B:src/other_parallel/Rust/Cargo.toml:22-26). Launched with oshrun.
- `src/other_parallel/Chapel/`: histo, ig, spmat, topo, triangle in Chapel, mostly the AGP style
  (B:src/other_parallel/Chapel/README.md:3-9).
- `example_matrices/` (Matrix Market and .dst files of 100 rows, one toposort input), `docs/` (two PDFs), `images/`.
- CI: a GitHub Actions workflow builds the OSHMEM and GNU UPC Docker images and runs pytest in them
  (B:.github/workflows/dockerimage.yml:18-37).

## 5. The apps

Models in bale_classic (read, B:src/bale_classic/apps/README.md:15): AGP (PGAS atomics, gets, and puts), exstack,
exstack2, and conveyor; some apps add alternates. Every serial app takes the standard options (seed `-s`, default
18181; models mask `-M`; dump `-D`; json `-j`) (B:src/other_serial/C/std_options.c:20-44), and the graph apps take
the graph options (`-N` rows, `-e` edge probability or `-z` nonzeros per row, default 10, `-F` Erdos-Renyi, `-G`
geometric, `-K` Kronecker, `-d`, `-l`, `-w`, `-f` Matrix Market file) (B:src/other_serial/C/std_options.c:69-138).
Inputs are generated inside each program from the seed with Knuth's lagged-Fibonacci generator
(B:src/other_serial/C/spmat_utils.h:167-176; spmat_utils.c:39-72). Reports go to stderr (std_options.c:326-349).

| App | bale_classic models (UPC or C+SHMEM) | Other versions | Serial C | Inputs in serial C (defaults) | What it computes and keeps | Its own check (serial C) |
| --- | --- | --- | --- | --- | --- | --- |
| histo | AGP, exstack, exstack2, conveyor; 4 alternates (apps/histo_src/) | Chapel; parallel Rust | histo.c | N updates 100000, table 2^20 (default_app_opts.h:13-14); index[i] = rand_int64(T) (histo.c:184-187) | counts[index[i]] += 1, three ways: generic, buffered, sorted (histo.c:30-127); counts not written | Adds -num_runs per update and requires every count to be 0; prints FAILED to stderr; exit status = error count (histo.c:216-232) |
| ig | AGP, exstack, exstack2, conveyor; 4 alternates | Chapel; parallel Rust | ig.c | N requests 100000, table 2^20 (default_app_opts.h:16-17); table[i] = -i, index from rand_int64 (ig.c:191-196) | tgt[i] = table[index[i]], generic and buffered (ig.c:57-132) | tgt[i] == -index[i] after each model; errors to stdout; exit = errors (ig.c:32-47, :224-227) |
| randperm | AGP, exstack, exstack2, conveyor (spmat/spmat_*.upc rand_permp_*) | parallel Rust | randperm.c | N 1000000 (default_app_opts.h:21) | A uniform random permutation: Fisher-Yates, dart throwing, or sort by random key (randperm.c:27-137) | is_perm only; exit 1 on failure (randperm.c:36-39, :85-88, :130-133) |
| transpose_matrix | AGP, exstack, exstack2, conveyor (spmat/spmat_*.upc) | Chapel spmat | transpose_matrix.c | Graph, 10000 rows (default_app_opts.h:25) | Transpose of a CSR matrix (spmat_utils.c:467-508); result discarded (transpose_matrix.c:37) | None ("TODO", transpose_matrix.c:104); bale_classic checks the transpose of the transpose (apps/transpose_matrix_src/transpose_matrix.upc:123-126) |
| permute_matrix | AGP, exstack, exstack2, conveyor | parallel Rust; Chapel spmat | permute_matrix.c | Graph, 10000 rows (default_app_opts.h:19); perms rand_perm(seed+1), (seed+2) (permute_matrix.c:89-90) | Row and column permutation of a CSR matrix (spmat_utils.c:429-456); discarded | None (permute_matrix.c:106); bale_classic compares models (apps/permute_matrix_src/permute_matrix.upc:145-150) |
| triangle | AGP, exstack, exstack2, conveyor; 4 alternates | Chapel; parallel Rust | triangle.c | Undirected, no loops, 50000 rows (default_app_opts.h:27; triangle.c:108-109) | Triangle count, L .& (L*L) and L .& (L*U) (triangle.c:29-66, :143-160); printed to stderr | Prints the known count for Kronecker graphs but compares nothing (triangle.c:132-137); bale_classic compares models and the Kronecker answer (apps/triangle_src/triangle.upc:226-229) |
| toposort | AGP, exstack, exstack2, conveyor; 3 alternates | Chapel; serial and parallel Rust | toposort.c | Morally upper-triangular matrix from a random unit upper-triangular one and permutations with fixed seeds 1234 and 5678 (toposort.c:60-98); 500 rows (:271) | Row and column permutations making it upper triangular, by a queue or by loops (toposort.c:108-231); "The answer need not be unique" (B:src/bale_classic/apps/topo_src/README.md:20) | Property: both are permutations and the permuted matrix is upper triangular; exit 1 (toposort.c:31-50, :312-315) |
| sssp | Bellman-Ford (exstack, exstack2, conveyor; AGP alternate) and delta-stepping (exstack, exstack2, conveyor) | serial and parallel Rust delta_stepping | sssp.c, sssp_dijsktra.c, sssp_bellmanford.c, sssp_delta.c | Directed, weighted, no loops (sssp.c:100-102), 10000 rows (default_app_opts.h:23); weights rand_double in [0, 1) (spmat_utils.c:1519-1523); source V0 = 0 (sssp.c:66) | Distances from V0, INFINITY when unreachable, by Dijkstra (heap), delta-stepping, Bellman-Ford (sssp.c:120-145; sssp_dijsktra.c:234-253) | Compares each algorithm with the first by L2 norm < 1e-8, but only labels the timing line " (compares)"; never fails (sssp.c:33-44, :146-155) |
| sparse_matrix_io | AGP and exstack only (apps/sparse_matrix_io_src/README.md:36) | none | none | A distributed sparse matrix dataset on disk | Write in BLOCK and read into CYCLIC layout (apps/sparse_matrix_io_src/README.md:5-36) | (not read) |
| unionfind | none: a placeholder for a future app (B:src/bale_classic/apps/union_find/README.md:5-18) | none | unionfind.c | Undirected, no loops, 10000 rows (default_app_opts.h:29; unionfind.c:220-224) | Number of connected components, by a "bad" union and a rank union (unionfind.c:119-172) | None; prints the count (unionfind.c:253) |

Paths in the table without a prefix are under B:src/other_serial/C/ unless they start with apps/ or spmat/, which
are under B:src/bale_classic/. All read.

Quirks found while reading (read where cited; the effect is inferred from C semantics and was not run). They
matter because LASSI records upstream quirks rather than fixing them silently (Agent Rule 4):

1. toposort keeps the last column of a row as `rowtrck & 0xFFFF`, 16 bits of a 32-bit column sum
   (toposort.c:142, :207), so a matrix of 65536 or more columns gets wrong pivots. bale's default (500) and its
   tests (at most 3332 rows, B:src/other_serial/C/tests/test_all.py:31-37) stay below.
2. randperm's sort model compares random keys in [0, 1) by returning their double difference as an int
   (randperm.c:100-104), which is 0 for every pair, so the sort shuffles nothing; with glibc's qsort the result
   is presumably the identity (inferred), and is_perm still passes.
3. histo's sort comparator returns a uint64 difference as an int (histo.c:99-102), implementation-defined; the
   counts do not depend on the order (inferred).
4. read_matrix_mm sets only the offset after the last nonempty row (spmat_utils.c:414), so trailing empty rows keep
   offset 0 from calloc (:832), and it accepts an "integer" field and then asserts it is real (:321-333, :363,
   :399).
5. The edge comparators return int64 differences as int (spmat_utils.c:268-289), wrong past 2^31 rows.
6. toposort's input permutations ignore `-s` (fixed seeds 1234 and 5678, toposort.c:76-77).
7. Reports: the input-matrix summary prints columns as rows (std_options.c:184-185); the header prints the year
   plus 1990 and a 0-based month (:290, :297); triangle's two model labels are swapped (triangle.c:148, :152);
   random_graph prints to stdout (spmat_utils.c:869, :887-890).
8. The serial pytest file never runs transpose_matrix with sizes, since it tests `app == 'transpose'`
   (tests/test_all.py:20, :38); every test only asserts exit status 0 (:74-78).
9. sssp's ALTERNATE model equals ALL_Models (8), so it never runs (sssp.c:92, :120).

## 6. Build needs and what the hosts have

- bale_classic (read): UPC (Cray UPC, GNU UPC, Clang UPC) or OpenSHMEM 1.4 (OpenMPI OSHMEM with UCX, Cray SHMEM,
  Cray openshmemX), argp from glibc, autoconf 2.69, automake 1.13.4; optional python3, pytest, doxygen
  (B:src/bale_classic/README.md:60-87). `make_bale` needs `-u` or `-s` (B:src/bale_classic/INSTALL.md:46); for SHMEM
  `CC=oshcc` (:14). Known problems: OSHMEM 4.0.3 with UCX on an SMP (exstack2 and conveyors) and Sandia OpenSHMEM
  1.4.5 on an SMP (conveyors) (B:src/bale_classic/README.md:72-75). Runs are multi-PE (oshrun, upcrun).
- Parallel Rust (read): convey_private, an external repository, plus a SHMEM install named by SHMEM_PATH
  (B:src/other_parallel/Rust/README.md:14-25).
- Serial C (read): gcc with -Wall, -lm, and argp (B:src/other_serial/C/Makefile:16-19; B:src/other_serial/C/README.md:34, :64-68).
- alpha01 (read from plans/spikes/p4-tt-pins.md:184-199, rx 20260924-185239-exec-2d3a and 20260924-185249-exec-1540,
  2026-09-24; bible:1218): gcc-12 and g++-12 12.3.0, glibc 2.35, clang-17 and clang-20, cmake, ninja, Python 3.10.12;
  openmpi-bin and libopenmpi-dev 4.1.2 and openmpi-ulfm 5.0.7 at /opt/openmpi-v5.0.7-ulfm. Whether that openmpi
  ships OpenSHMEM (oshcc, oshrun) is unknown: Debian's packaging installs those tools only when the build produced
  oshmem_info (sources.debian.org openmpi 4.1.4-3 debian/rules:225-245, read 2026-10-26; Debian bookworm, not
  Ubuntu 22.04's 4.1.2, whose file list could not be read: packages.ubuntu.com returned HTTP 500 on 2026-10-26).
  A UPC compiler, Chapel, autoconf, and automake on alpha01: unknown (no record). No probe was run (Agent Rule 17).
- Workstation (read, bible:1208, :1219; L:plans/STATUS.md:132-134): LASSI builds and runs only in the LASSI WSL
  distro (Ubuntu 22.04, mirroring alpha01), which waits for the owner to run install batch 1 (P5.15 OWNER); the
  sandbox there needs P5.16. Until then nothing native runs locally.
- LASSI toolchains (read): gcc-native compiles only .cpp, .cc, and .cxx files with `g++-12 -O3 -fopenmp`, and "A .c
  file is not a source here" (L:lassi/toolchains/gcc.py:1-16; L:toolchains/gcc.pin). A C toolchain is P12.6's,
  READY and not built (L:plans/STATUS.md:141; L:plans/p12-frontends.md:78-90; bible:543 proposes gcc-12 at
  /usr/bin/gcc-12). P12.6's Rust builds offline with only rayon vendored (bible:543), so bale's Rust crates (clap,
  rand, regex) would not build as they stand (inferred). Uncommitted P9.15 work on branch p9-ee in C:/dev/lassi adds
  a support-entry key compile_as (c++ only); bale's serial C would not compile as C++ as written, for example
  `malloc` assigned to an `int64_t *` without a cast (spmat_utils.c:97) (inferred).
- So (inferred): building bale's own programs needs nothing LASSI lacks for the serial C (gcc, argp), but they
  would not be LASSI references anyway (section 7.1); building bale_classic needs UPC or OpenSHMEM and a multi-PE
  launcher, which no LASSI executor or install batch has.

## 7. Mapping onto LASSI

### 7.1 What an item needs (read)

- A suite manifest under assets/bench/ with per-item split (train, eval, or unassigned), languages (dir, files,
  optional tracked, support, reference), and optional run_args, passfail, support, tolerance, inputs, outputs,
  seed, contract, and unpack_to_dest (L:lassi/bench/registry.py:12-98, :118-140; bible:651, :661, :665).
- An Oracle: stdout_mask or passfail for stdout (bible:766-777), or binary_io over lassi_io output files with
  pcc, max_abs, or ulp; integer outputs always match exactly; an infinity matches an infinity of the same sign
  (bible:780-790).
- Held-out inputs: generate_inputs entries (name, dtype, shape, dist uniform or integers, lo, hi) drawn by PCG from
  the item's seed, at most 2^20 elements each, never shown in a prompt (bible:667, :750-751;
  L:lassi/harness/contract.py:104). A rank-0 (scalar) entry is allowed (L:lassi/harness/lassi_io.py:228-241;
  L:assets/harness/c/lassi_io.h:22, :362-388). A contract bounds each element alone and cannot relate inputs, such
  as an index below another input's length (bible:752).
- Programs read inputs and write outputs through lassi_io.h (bible:745-748); every output file a run creates is an
  output (bible:753).
- Against that, bale's programs (section 5) take sizes and a seed on the command line and generate their inputs
  with Knuth's generator (the graph apps can instead read a Matrix Market file, `-f`); they write no result file
  (only optional debug dumps, `-D`, and a JSON timing record, `-j`; std_options.c:24-25, :53-65), report on
  stderr, and print almost nothing on stdout. With run_args and stdout_mask they would compare nothing useful;
  their exit status is not an oracle (inferred). Hence every item's references are programs written here.

### 7.2 Per-app fit

"Reference source" says whether the item's programs come from bale or are written here. Line estimates are
PROJECTED, from the size of bale's serial kernels and of the Tier A and adv-seeded programs, not counted from code.
Bare bale file names here and in the notes below are under B:src/other_serial/C/; topo_src/ is under
B:src/bale_classic/apps/.

| App | Item? | Directions (first in bold) | Held-out inputs | Outputs | Tolerance | Reference source and size |
| --- | --- | --- | --- | --- | --- | --- |
| histo | Yes | **cpp -> omp, omp -> cpp**; later c -> cpp, c -> omp (P12.6); cuda compile-only; TT research-grade | table_size i64 scalar (constant), index i64[100000] in [0, T) | counts i64[T] | max_abs 0 (exact, as bale's check) | Written here: cpp about 50 lines, omp about 55 (atomic adds, bale's AGP analog) |
| ig | Yes | as histo | table i64[2^20] drawn (bale uses -i), index i64[100000] in [0, 2^20) | tgt i64[100000] | max_abs 0 | Written here: about 50 and 50 |
| triangle | Yes | **cpp -> omp, omp -> cpp**; later c | num_vertices i64 scalar, edge_u and edge_v i64[m] in [0, n) | triangles i64 scalar | max_abs 0 | Written here: about 70 and 75, plus a shared tracked header bale_graph.h (COO to a tidy lower-triangular CSR) of about 150 |
| sssp | Yes (v1) | cpp -> omp, omp -> cpp | num_vertices, source i64 scalars; src, dst i64[m]; weight f64[m] uniform [0, 1) | dist f64[n], inf when unreachable | max_abs 1e-8 (bale's L2 < 1e-8 check bounds each element) or 0 (inferred to be exact, below) | Written here: Dijkstra with a heap as bale's sssp_dijsktra_heap, about 150; omp Bellman-Ford about 110; header extended for weights |
| transpose_matrix | Yes (v1) | cpp -> omp, omp -> cpp | as triangle, plus value f64[m] | offset i64[n+1], nonzero i64[nnz], value f64[nnz] | max_abs 0 (values move, not computed) | Written here: about 60 and 80 |
| permute_matrix | Yes (v1) | as transpose | as transpose, plus rkey and ckey f64[n] that the header turns into permutations by a stable argsort | offset, nonzero (rows not re-sorted, as bale's, spmat_utils.c:444-452) | max_abs 0 | Written here: about 70 and 80 |
| unionfind | Yes (v1), serial-only in bale | cpp -> omp (label propagation), omp -> cpp | as triangle | num_components i64 scalar; label i64[n], the smallest vertex of each component (our addition; bale keeps only the count) | max_abs 0 | Written here: about 90 and 100 |
| toposort | Later | serial only (c or cpp -> rust, csharp, python after P12.6) with exact match, or any direction with a property oracle | A unit upper-triangular matrix and two permutations from keys | rperm, cperm | not unique (topo_src/README.md:20): binary_io would fail a correct parallel candidate | Needs a new Oracle component that runs a tracked checker, as bale's check_result does (toposort.c:31-50); not proposed for v0 |
| randperm | Later or never | none useful under binary_io | - | A random permutation | Exact only by reproducing bale's generator and algorithm; bale checks is_perm only | Same property-oracle need; the sort model's quirk (section 5, item 2) |
| sparse_matrix_io | No | - | - | Files in bale's dataset layout | - | Distributed file I/O; no serial version; outside the harness contract (inferred) |

Notes on the table (inferred unless cited):

- Sizes follow bale's defaults where the 2^20 cap allows: histo and ig use 2^20 and 100000 (default_app_opts.h:13-17);
  triangle could use bale's 50000 rows with about 500000 candidate pairs (about 10 nonzeros per row of L, from
  nz_per_row 10, std_options.c:105, and spmat_utils.c:1128-1147), about 1333 triangles expected for a uniform random
  graph of average degree 20 (PROJECTED, n^3 p^3 / 6 with n p = 20). Outputs are not capped, and the largest, histo's
  counts, is 8 MiB, inside the sandbox's 256 MiB workdir (bible:795).
- Graph inputs: generate_inputs draws independent arrays, so it cannot draw a CSR or a permutation (bible:751). The
  proposal keeps the harness unchanged: endpoints are drawn as arrays and bale_graph.h, a tracked header written
  here (C99 and C++17, like lassi_io.h), builds the graph deterministically (drop u == v, store (max, min), drop
  repeats, sort rows). That replaces bale's Erdos-Renyi generator (spmat_utils.c:1458-1529) with our construction;
  the kernel run on the graph is bale's. Alternatives: a new generate_inputs dist that draws graphs (a [DESIGN]
  change to Held-out input files, bible:751), or fixed Matrix Market inputs as support files (not seed-derived,
  outside the Harness Contract as written).
- sssp exactness: every correct algorithm computes the same fixed point d(w) = min over v of fl(d(v) + c(v, w)),
  since rounding is monotonic, so Dijkstra, Bellman-Ford, and delta-stepping agree bit for bit (inferred, not run).
  bale's own check tolerates an L2 difference below 1e-8 (sssp.c:151). A tolerance of max_abs 1e-8 follows bale;
  0 would follow the inference. binary_io matches same-signed infinities (bible:784), so unreachable vertices align.
- bale's parallel models map onto OpenMP only in the AGP style (atomics, gets, puts); exstack, exstack2, and
  conveyors aggregate network messages and have no shared-memory analog worth translating (inferred from
  B:src/bale_classic/README.md:37-41 and the app READMEs).
- A cpp -> omp candidate that returns the serial program unchanged would pass binary_io, since both builds use
  `-fopenmp` and the oracle reads outputs only. Serial -> parallel items need either a static check that tags (not
  fails) an omp candidate with no OpenMP construct, a new component behind the Toolchain capability pattern of
  host_compute_guard (L:lassi/core/capabilities.py:78-81; Agent Rule 3), or a reported speedup; neither exists
  (grep, read). The native executor runs OpenMP with OMP_NUM_THREADS at the run's CPU count, 16 (bible:798, :804).
- TT: histo and ig are random scatter and gather, which on Tensix are data-movement work; the CPU -> TT guard would
  tag such programs data-movement-only (bible:757), and Tier A's model of upstream kernels does not apply, since
  bale has no TT kernels: every kernel would be written here (inferred). Research-grade; not first.
- CUDA: compile-only without a GPU host (bible:700, :1206). HIP the same (bible:703).
- Overlap with eval items (inferred): histo is the random-update pattern of randomAccess, and sssp is a shortest-path
  problem like floydwarshall; randomAccess is an eval item of lassi-hecbench-10 (L:assets/bench/lassi-hecbench-10.yaml:133-145)
  and lassi-ee-22, floydwarshall of lassi-ee-22 (branch p9-ee). Agent Rule 5 forbids training on eval items, not on
  similar programs, so this matters only if bale items are assigned train.

### 7.3 Files: what the model writes and what is support

- Model-written (inferred from the Tier A pattern, bible:655): the whole target program, one file per item and
  language, `<lang>/<item>.cpp`, at the path the manifest lists; for graph items it includes bale_graph.h and
  calls it, as the source program does.
- Support, written into every build directory and never writable by the model (L:lassi/toolchains/_base.py:371-378,
  :433-449): lassi_io.h (tracked), bale_graph.h (tracked, graph items), and, if the fetch form below is chosen,
  bale's serial C file for the app (upstream, path and sha256), which gcc-native never compiles, since a .c file is
  not a source and harness files are never sources (L:lassi/toolchains/gcc.py:8-11; _base.py:428).
- Prompts: a .h or .c support file is neither a kernel file nor a host file (L:lassi/core/stages.py:784, :788,
  :1251-1290), so the model sees only the source program's calls to bale_graph.h, as it sees lassi_io.h calls today
  (inferred). Language names reach templates raw (stages.py:2519), and p0-smoke says "Translate the program below
  from $source_language to $target_language" (L:assets/prompts/p0-smoke/generate.txt:1), so "omp" needs a template
  set that says what it means: C++ with OpenMP on the host CPU (inferred).
- Recipes bind a toolchain per language name (L:projects/adversary-seeded/control-fuzzer.yaml:14-20), so a bale
  recipe binds `cpp: gcc-native` and `omp: gcc-native` (inferred). lassi-hecbench-10's `omp` means OpenMP offload
  under nvc++, so the shared name is per suite and per recipe; a different name (for example `cpu_omp`) avoids the
  ambiguity at the cost of a longer direction name (a choice for the plan).

### 7.4 Splits (Agent Rule 5; a choice)

- A split is assigned per item before any training, and an assigned split never changes; unassigned decides
  nothing and is refused to training, prompt tuning, and corpus harvest like eval (bible:651;
  L:lassi/bench/registry.py:118-122, :281-295).
- Precedents (read): tt-pairs-v0 every item unassigned until P7 (OQ-025 (d)); xlang-v0 recommended the same
  (OQ-057, splits (a)); tt-pairs-v0 named a development set (OQ-083 (b)), whose split value P17.31 is to add.
- "Into the dataset" can mean evaluation items or training data (inferred). Training data comes only from train
  items: the corpus reads bench items through harvest_item, train only (bible:628).
- bale is public since 2018 (1.0.0, B:README.md:47) and on GitHub since 2020-10-23 (read), so its algorithms are
  likely in models' pretraining data; as eval items they measure less than private items would, as OQ-025 noted
  for Tier A (inferred). The programs written here are new text, which helps somewhat.
- Section 12 lists the options; the recommendation is every item unassigned until training exists.

### 7.5 Copy or fetch (a choice)

- Fetch only (Tier A and HeCBench practice, OQ-018): the manifest pins jdevinney/bale by repo and commit, and each
  app's serial C file is a support entry `{upstream: <path>, sha256: <hex>}`, which tools/fetch_bench.py fetches
  in a sparse, shallow checkout by sha and checks (L:tools/fetch_bench.py:16-36, :194-250); the programs written
  here follow bale's kernels in their own code. Consequence: a run needs the bench root and a fetch on the build
  host, which needs network there once (inferred). A variant drops the upstream entries and repo and commit,
  making a tracked suite like adv-seeded-v0 that needs no fetch (bible:675; L:lassi/bench/registry.py:464-485),
  with the pin recorded only in comments.
- Verbatim kernels: copy bale's kernel functions into the tracked programs with IDA's notice, the conditions, and
  the disclaimer (allowed, B:LICENSE:9-18). Consequence: closest fidelity; bale text published in this repository.
- Vendor the serial C tree (181,845 bytes): allowed for IDA's files; Knuth's header must stay unchanged; the
  colon-named example matrix cannot be checked out on Windows; not needed by any item (inferred).

### 7.6 Agent Rules touched

- Rule 1: nothing here is a measurement; the line counts, sizes, and the triangle count are PROJECTED.
- Rule 4: no faithful recipe reproduces a bale pipeline, so no faithful toggles arise (inferred). The programs
  written here follow bale's serial algorithms, record every deviation (inputs from generate_inputs, outputs as
  lassi_io files, our graph construction, unionfind's labels) in the manifest comments, and keep inputs where
  bale's quirks do not fire (toposort below 65536 rows) rather than fixing them silently.
- Rule 5: splits as above; unassigned or eval items never reach training, prompt tuning, or harvest. A bale prompt
  set built while reading bale items' outcomes would be prompt tuning on them unless the owner names them a
  development set (OQ-083 precedent).
- Rule 10: the manifest pins bale's commit; gcc.pin pins the compiler; each run records its recipe and pins.
  Adding the suite is a Decision Log entry.
- Rule 11: only if bale items become corpus entries (train items only): their IR is committed in custom assembly
  and never truncated (bible:628). Under unassigned splits nothing is harvested.
- Rules 6, 7, 17: candidates run only in the sandbox (native executor); fetched sources live under
  $LASSI_SCRATCH/bench (L:lassi/bench/registry.py:412-423); this spike touched no host.

### 7.7 Manifest sketch (text only; not a file)

```yaml
# bale-v0 (sketch): three apps of bale (github.com/jdevinney/bale at 1b8f673; IDA, BSD-3-Clause-style license),
# re-hosted on the LASSI harness. Each item pairs two programs written here, tracked under
# assets/bench/bale-v0/<item>/: cpp/<item>.cpp, a serial C++ program following the generic serial kernel of bale's
# src/other_serial/C/<app>.c, and omp/<item>.cpp, the same computation with OpenMP on the host CPU, both built by
# gcc-native (g++-12 -O3 -fopenmp; toolchains/gcc.pin). Both read held-out inputs and write outputs through
# lassi_io.h. No bale text is tracked: each item names bale's serial C file by path and sha256 as a support entry,
# fetched and checked by tools/fetch_bench.py and never compiled (a .c file is not a gcc-native source). Per item:
# split stays unassigned until the owner assigns it; tolerance is exact, as bale's own checks are for these apps.
suite: bale-v0
repo: https://github.com/jdevinney/bale.git
commit: 1b8f673b56645b2bd74a8af6213462bbc9d559fe
items:
  histo:
    split: unassigned
    seed: 4101
    # counts[index[i]] += 1 over all updates: bale's generic serial loop (src/other_serial/C/histo.c:30-43). bale draws
    # index with Knuth's generator (histo.c:184-187); here generate_inputs draws it. Sizes are bale's defaults, 2**20
    # bins and 100000 updates (default_app_opts.h:13-14). table_size is a constant scalar: a contract cannot bound
    # index by another input.
    inputs:
      - {name: table_size, dtype: i64, shape: [], dist: integers, lo: 1048576, hi: 1048577}
      - {name: index, dtype: i64, shape: [100000], dist: integers, lo: 0, hi: 1048576}
    outputs: [counts]  # i64 [1048576] on both sides
    tolerance: {metric: max_abs, threshold: 0}  # bale's check is exact (histo.c:216-227)
    languages:
      cpp:
        dir: assets/bench/bale-v0/histo/cpp
        tracked: true
        files: [histo.cpp]
        support:
          lassi_io.h: {tracked: assets/harness/c/lassi_io.h}
          upstream/histo.c:
            upstream: src/other_serial/C/histo.c
            sha256: 8391f5a020ff67d6d3bcb55558bec37b3a97ee4f6176d8cbc7d605dd53c3ccf2
      omp:
        dir: assets/bench/bale-v0/histo/omp
        tracked: true
        files: [histo.cpp]
        support:
          lassi_io.h: {tracked: assets/harness/c/lassi_io.h}
          upstream/histo.c:
            upstream: src/other_serial/C/histo.c
            sha256: 8391f5a020ff67d6d3bcb55558bec37b3a97ee4f6176d8cbc7d605dd53c3ccf2
  ig:
    split: unassigned
    seed: 4102
    # tgt[i] = table[index[i]]: bale's generic serial gather (src/other_serial/C/ig.c:57-69). bale fills table with
    # minus each index so that its check is tgt[i] == -index[i] (ig.c:32-47, :191-192); here table is drawn too, so a
    # candidate that returns -index cannot pass. Sizes are bale's defaults (default_app_opts.h:16-17); table holds
    # 2**20 elements, the cap per input.
    inputs:
      - {name: table, dtype: i64, shape: [1048576], dist: integers, lo: -1099511627776, hi: 1099511627776}
      - {name: index, dtype: i64, shape: [100000], dist: integers, lo: 0, hi: 1048576}
    outputs: [tgt]  # i64 [100000]
    tolerance: {metric: max_abs, threshold: 0}
    languages:
      cpp:
        dir: assets/bench/bale-v0/ig/cpp
        tracked: true
        files: [ig.cpp]
        support:
          lassi_io.h: {tracked: assets/harness/c/lassi_io.h}
          upstream/ig.c:
            upstream: src/other_serial/C/ig.c
            sha256: 2cb08618a08eddb9979ca27c5cb5b259325ca34d766eae34f0a9ef961c2144d0
      omp:
        dir: assets/bench/bale-v0/ig/omp
        tracked: true
        files: [ig.cpp]
        support:
          lassi_io.h: {tracked: assets/harness/c/lassi_io.h}
          upstream/ig.c:
            upstream: src/other_serial/C/ig.c
            sha256: 2cb08618a08eddb9979ca27c5cb5b259325ca34d766eae34f0a9ef961c2144d0
  triangle:
    split: unassigned
    seed: 4103
    # Triangles of an undirected graph: for each nonzero (u, v) of its strictly lower-triangular adjacency L, the
    # size of the intersection of rows u and v of L, bale's L .& (L * L) form (src/other_serial/C/triangle.c:29-66,
    # called on (L, L) at :147-149), which needs sorted rows (:37-43). bale builds L with its Erdos-Renyi generator
    # (spmat_utils.c:1458-1529); here bale_graph.h, a header written here, builds it from the drawn pairs: a pair
    # with u == v is dropped, the rest stored as (max, min), repeats dropped, rows sorted. 50000 vertices as bale's
    # default (default_app_opts.h:27) and 500000 pairs, about bale's 10 nonzeros per row of L (std_options.c:105).
    inputs:
      - {name: num_vertices, dtype: i64, shape: [], dist: integers, lo: 50000, hi: 50001}
      - {name: edge_u, dtype: i64, shape: [500000], dist: integers, lo: 0, hi: 50000}
      - {name: edge_v, dtype: i64, shape: [500000], dist: integers, lo: 0, hi: 50000}
    outputs: [triangles]  # i64 scalar (shape [])
    tolerance: {metric: max_abs, threshold: 0}
    languages:
      cpp:
        dir: assets/bench/bale-v0/triangle/cpp
        tracked: true
        files: [triangle.cpp]
        support:
          lassi_io.h: {tracked: assets/harness/c/lassi_io.h}
          bale_graph.h: {tracked: assets/bench/bale-v0/common/bale_graph.h}
          upstream/triangle.c:
            upstream: src/other_serial/C/triangle.c
            sha256: 07b3d5ba0858fb0667a75b6044e5e7c14741b0c45600582881d7815886a4237f
      omp:
        dir: assets/bench/bale-v0/triangle/omp
        tracked: true
        files: [triangle.cpp]
        support:
          lassi_io.h: {tracked: assets/harness/c/lassi_io.h}
          bale_graph.h: {tracked: assets/bench/bale-v0/common/bale_graph.h}
          upstream/triangle.c:
            upstream: src/other_serial/C/triangle.c
            sha256: 07b3d5ba0858fb0667a75b6044e5e7c14741b0c45600582881d7815886a4237f
```

Notes on the sketch (inferred from the registry's rules, L:lassi/bench/registry.py:426-701, not loaded):

- Every key it uses exists today; the seeds are arbitrary distinct choices; no `contract` key, so each item's
  contract is its held-out spec, since widening table_size or num_vertices would need a relation the contract
  cannot state (bible:752).
- Dropping the three `upstream/...` entries and the repo and commit lines gives the tracked-suite variant.
- A recipe would read `bench: {suite: bale-v0, split: unassigned}`, `directions: [{source: cpp, target: omp},
  {source: omp, target: cpp}]`, `toolchain: {cpp: gcc-native, omp: gcc-native}`, `executor: {kind: native}`,
  `oracle: {kind: binary_io, metric: max_abs, threshold: 0}`, and, being non-faithful, `fixes.baseline_both` on so
  the two references must agree before any model call (bible:788), plus a prompt set that names the languages.
- Adding any manifest with held-out inputs requires recording its digests in tests/bench/test_held_out_contract.py,
  which walks every manifest (L:tests/bench/test_held_out_contract.py:84-97).

### 7.8 Work this implies (PROJECTED, for planning)

1. Windows half, no host needed: the manifest, six programs, bale_graph.h, a prompt set for cpp <-> omp, registry
   and program tests (as tests/bench/test_tt_pairs.py and test_held_out_contract.py do for their suites), the
   OpenMP-construct tag if chosen, and the bible edits (section 13). Roughly 400 to 500 lines of programs and header
   plus tests (PROJECTED).
2. Host half, in the LASSI distro after P5.15 and P5.16 (or alpha01 in an owner window): fetch, build every
   reference, baseline agreement on every item, and a mock-LLM dry run over both directions, the CPU analog of the
   LASSI-DF acceptance criteria (bible:1116); results under results/ with the local provenance writer (bible:1222).
3. Later: v1 apps; a `c` language after P12.6; a property-checking Oracle for toposort and randperm; TT or CUDA
   directions if wanted.

## 8. Unknowns

- Whether alpha01's openmpi 4.1.2 ships OpenSHMEM, and whether alpha01 or the LASSI distro has a UPC compiler,
  Chapel, autoconf, or automake (no record; no probe, Agent Rule 17).
- Whether bale's Knuth header differs from Knuth's published file (not compared; no item needs it).
- What glibc's qsort returns for randperm's all-equal comparator (inferred identity; not run).
- Whether the sssp references agree bit for bit (inferred; not run).
- Whether gcc-native accepts the three items' programs and the header without new flags (inferred yes: C++17 and
  OpenMP; not built).
- Which language name the owner prefers for host OpenMP (`omp` or another), and whether "dataset" meant evaluation
  items, training data, or both.

## 9. Finding and confidence

bale is github.com/jdevinney/bale at 1b8f673b56645b2bd74a8af6213462bbc9d559fe (2022-06-26; sources from bale 3.0,
2020-10-23), under IDA's BSD-3-Clause-style license with a DFARS clause, which allows our use and a tracked copy
with the notice kept; none of its programs reads held-out inputs or writes outputs LASSI's oracles read, so each
item needs programs written here that follow bale's serial kernels. histo, ig, and triangle fit binary_io exactly
and can run in cpp <-> omp on the host CPU with today's toolchains; sssp, transpose_matrix, permute_matrix, and
unionfind fit next; toposort and randperm need a property oracle; sparse_matrix_io does not fit.

Confidence: high for identity, license text, commit, layout, and the apps' code (read at the pin); medium for the
fit and the effort (inferred from the registry, harness, and bale's code; nothing built or run); unknown for host
packages beyond the earlier spikes.

## 10. Consequences for the plan

- No task changes state now; nothing is blocked today. The suite waits on the owner-queue item below.
- After the answer: a phase plan (a new Build Roadmap row, or tasks in an existing phase, as the owner chooses)
  per AGENTS.md Phase Planning, its Windows half runnable now, its host half after P5.15 and P5.16.
- If the owner wants bale as training data, the corpus path also needs P5's frontend and execution gate (P5.7,
  P5.12) and train splits; under unassigned splits it stays evaluation only.
- bible: a Benchmark Suites row, question 10 of Risks And Questions, and a Decision Log entry (section 13), only
  after the answer.

## 11. Sources

- GitHub REST API, read 2026-10-26: https://api.github.com/repos/jdevinney/bale (and /commits/master, /branches,
  /tags, /releases, /git/trees/1b8f673b56645b2bd74a8af6213462bbc9d559fe?recursive=1, /forks, /compare/master...<fork>:master).
- Repository at the pin: https://github.com/jdevinney/bale/tree/1b8f673b56645b2bd74a8af6213462bbc9d559fe (README.md,
  LICENSE, src/bale_classic/README.md, INSTALL.md, apps/*/README.md, apps/*_src/*.upc, spmat/*.upc,
  src/other_serial/C/*, src/other_serial/Rust, src/other_parallel/Rust, src/other_parallel/Chapel,
  .github/workflows/dockerimage.yml), read 2026-10-26 from blobs of a local clone.
- Debian openmpi 4.1.4-3 packaging: https://sources.debian.org/data/main/o/openmpi/4.1.4-3/debian/rules, lines
  225-245, read 2026-10-26.
- https://packages.ubuntu.com/jammy/amd64/openmpi-bin/filelist: HTTP 500 on 2026-10-26 (not read).
- Web searches, 2026-10-26: "BALE benchmark code translation parallel"; a quoted search for bale's README phrase.
  Neither found another BALE.
- This repository at 3d4e0e9: docs/BIBLE.md (lines cited above), AGENTS.md, lassi/bench/registry.py,
  tools/fetch_bench.py, lassi/harness/lassi_io.py, lassi/harness/contract.py, assets/harness/c/lassi_io.h,
  lassi/toolchains/gcc.py, lassi/toolchains/_base.py, lassi/core/stages.py, lassi/core/capabilities.py,
  toolchains/gcc.pin, assets/bench/tt-pairs-v0.yaml, assets/bench/lassi-hecbench-10.yaml,
  assets/bench/adv-seeded-v0.yaml, projects/adversary-seeded/control-fuzzer.yaml, assets/prompts/p0-smoke/generate.txt,
  tests/bench/test_held_out_contract.py, plans/STATUS.md, plans/p12-frontends.md, plans/spikes/p4-tt-pins.md,
  plans/OWNER-QUEUE.md (OQ-018, OQ-025, OQ-057, OQ-083).
- Branch p9-ee in C:/dev/lassi (read 2026-10-26): assets/bench/lassi-ee-22.yaml at HEAD (item names), and the
  uncommitted P9.15 compile_as change in lassi/bench/registry.py.

## 12. Owner-queue draft

```
## OQ-<next> bale: Which Apps And Directions First, Their Splits, And Copy Or Fetch
State: OPEN
Kind: decision
Blocks: none yet (no bale task exists; the plan that builds the suite waits on this answer)
Evidence: plans/spikes/bale.md; docs/BIBLE.md (Benchmark Suites: the table, split rules, Item tolerance, manifest
keys, Held-out inputs, Input contracts; Harness Contract; Oracles, binary_io; Agent Rules 3, 4, 5, 10, 11); OQ-018
(no upstream text tracked); OQ-025 (d) and OQ-057 (splits unassigned until training exists); OQ-083 (b)
(development sets)
Question: You asked on 2026-10-26 to add bale to the dataset. bale is github.com/jdevinney/bale ("bale is not a
collection of benchmarks", README.md:15), last commit 1b8f673 of 2022-06-26, under the Institute for Defense
Analyses' BSD-3-Clause-style license, which allows use and copies with its notice kept. It holds ten apps: nine in
bale_classic, parallel in UPC or C with OpenSHMEM, and serial C versions of all but sparse_matrix_io, plus
unionfind. No bale program can be a LASSI reference as written: each generates its own inputs from a seed (or
reads a Matrix Market file), writes no result file, reports on stderr, and checks itself or not at all, and the
parallel ones need UPC or OpenSHMEM
and a multi-PE launcher that LASSI lacks. So every item needs programs written here on the lassi_io harness that
follow bale's serial kernels, as Tier A's do. Which apps and directions first, with which splits, and may bale's
text be copied into this public repository?
Options:
- Scope: (a) v0: histo, ig, and triangle, each as a serial C++ program and a host OpenMP C++ program written here
  (directions cpp -> omp and omp -> cpp, both built by the pinned g++-12 with -fopenmp and run by the native
  executor in the sandbox), exact outputs under binary_io, a shared graph header for triangle, a prompt set for
  these directions, and a static check that tags (never fails) an omp candidate with no OpenMP construct, since
  returning the serial program unchanged would otherwise pass; then v1: sssp, transpose_matrix, permute_matrix,
  and unionfind on the same header, and a c language (bale's own) once P12.6 registers a C toolchain; toposort and
  randperm wait for an oracle that checks a property, since many answers are correct; sparse_matrix_io is left
  out; TT and CUDA directions later. Consequences: no new toolchain or harness change; about 400 to 500 lines of
  programs and header plus tests (PROJECTED); the Windows half can start now, and every run waits for the LASSI
  distro (P5.15, P5.16) or an alpha01 window. (b) All nine serial apps at once: the seven that fit exactly, and
  toposort and randperm through a property-checking Oracle built first. Consequences: a new Oracle component and
  its bible design before any item runs; about three times the first step (PROJECTED). (c) A pilot of histo and ig only. Consequences: the smallest step
  and no graph header; no graph app until a second answer. (d) bale's own parallel models as references.
  Consequences: a UPC compiler or OpenSHMEM in an install batch, a multi-PE executor, and still a harness, since no
  bale program reads lassi_io inputs; not recommended.
- Splits: (a) every item unassigned until training exists, as tt-pairs-v0 and xlang-v0: evaluation runs allowed,
  training, prompt tuning, and corpus harvest refused; you assign splits before any training run reads the suite.
  (b) Assign now by app, some train and some eval. Consequences: an eval label is permanent; histo repeats the
  random-update pattern of randomAccess, an eval item of both HeCBench suites, so histo as train is a judgment
  call. (c) Every item eval: bale never feeds training or the corpus. (d) Every item train: bale feeds the corpus
  and training, and no bale item measures generalization; bale has been public since 2018, so it is likely in
  pretraining data either way. (e) A development set, as tt-pairs-v0 became (OQ-083 (b)), once P17.31 adds the split
  value: prompt and feature work steered by bale outcomes allowed, never a held-out result.
- Copy or fetch: (a) Fetch only: no bale text tracked; the programs written here follow bale's kernels in their own
  code; the manifest pins jdevinney/bale at 1b8f673 and names each app's serial C file by path and sha256,
  fetched and checked by tools/fetch_bench.py and never compiled, as Tier A names its kernels. Consequence: a fetch
  on the build host before the first run. A variant records the pin only in comments, as a tracked suite with
  nothing to fetch. (b) Copy bale's kernel functions verbatim into the tracked programs with IDA's notice and the
  license text kept, plus a NOTICE file. Consequences: closest fidelity; bale's text public in this repository
  under its license, which OQ-018 avoided for LASSI's own upstream text. (c) Vendor bale's serial C directory
  (about 182 KB). Consequences: as (b), more text; one Knuth header must stay unchanged; not needed by any item.
Recommendation: Scope (a), splits (a), copy or fetch (a). Together they need no new toolchain, harness change, or
host install, decide nothing that cannot be undone, keep upstream text out of the repository as for every other
suite, and start with the three apps whose answers are unique and exact; v1 and the c language follow without
another design change.
Answer:
```

## 13. Bible edit after the answer (draft; not made now)

Under the recommendation, the plan's first task would make these edits, with the owner's answer cited:

- Benchmark Suites, a row after xlang-v0:

```
| bale-v0 | Apps of [bale](https://github.com/jdevinney/bale) at 1b8f673 (histo, ig, triangle): serial C++ and host OpenMP C++ programs written here that follow bale's serial C kernels, on held-out inputs through lassi_io.h; bale's files named by path and sha256 only | Hand-written (cpp <-> omp) | Every item unassigned until the owner assigns it (OQ-<next>) |
```

- Risks And Questions, question 10: add bale-v0 to the list of suites whose item splits are open.
- Decision Log, at the top, with the count sentence updated:

```
| <date> | bale-v0, a suite of bale's apps (github.com/jdevinney/bale at 1b8f673; IDA BSD-3-Clause-style license) re-hosted on the LASSI harness: histo, ig, and triangle as serial C++ and host OpenMP C++ programs written here, exact under binary_io, directions cpp -> omp and omp -> cpp, every item unassigned, no bale text tracked (OQ-<next>, the owner's answer of <date>; plans/spikes/bale.md; Benchmark Suites) | No bale program reads held-out inputs or writes lassi_io outputs, and its parallel models need UPC or OpenSHMEM, so the references are written here; these three apps have unique, exact answers and run with the pinned g++ alone |
```
