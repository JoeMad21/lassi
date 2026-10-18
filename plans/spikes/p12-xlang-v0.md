# Spike P12.2: xlang-v0 candidates from the Computer Language Benchmarks Game

- Task: P12.2 (plans/p12-frontends.md, "P12.2 Spike: xlang-v0 candidates (documentation only)", its Constraints, and
  "Decisions taken at planning": the gate reading and the frontend pin rule). Bible: Benchmark Suites (the xlang-v0
  row; split rules; Item tolerance; Held-out inputs), Language Frontends (its table; Frontend Rules), Harness
  Contract, Oracles (binary_io), Risks And Questions (question 10), Agent Rules 4 and 5. Owner item: OQ-057, which
  this spike answers as a Response.
- Date: 2026-10-18 on the workstation, Git Bash.
- Base: branch p12-frontends at 8b30baa (8b30baa4280a11e469f9d7debc734bce112732fc).
- Documentation only; no upstream text tracked. The reads were the salsa.debian.org GitLab API (project, branches,
  commits, tree, and files JSON), raw files at a named commit, the live site's pages, Polygeist source and tests at
  77c04bb2, and a few language and compiler manuals. One file was downloaded whole: the Benchmarks Game's source zip
  (1,991,840 bytes), into a session scratch directory outside the repository, read in memory by a scratch script
  under `python -I` (names, sizes, dates, hashes, and member text for reading constructs). Nothing was extracted
  into the repository, installed, built, or run, and no upstream program ran. No rx command ran; nothing touched
  alpha01, the LASSI distro, or a device; the repository was only read, apart from this file.
- Device: none.
- Labels: nothing here is [MEASURED]. Upstream times are the site's figures on its own machine (Ubuntu 24.04, an
  i5-3330), not LASSI's. Cost and agreement figures are PROJECTED with their basis; sizes not yet fixed are
  PLACEHOLDER. A value read from source at a commit is cited as file:line.
- Agent Rule 5 and OQ-018: no upstream program text appears here. Constructs are described in our own words; quotes
  are limited to identifiers, short idioms, and format specifiers. Every member is named by its path inside the
  zip and, where read, its sha256.
- Notation: `Z` is public/download/benchmarksgame-sourcecode.zip at commit 40296663; a member is `<bench>/<file>`.
  `BG raw` is https://salsa.debian.org/benchmarksgame-team/benchmarksgame/-/raw/40296663ed350d5fe4a6ab5e367bab61cb77c219/.
  `PG` is Polygeist at 77c04bb2a7a2406ca9480bcc9e729b07d2c8d077 (P5.5's provisional pin), read under
  https://raw.githubusercontent.com/llvm/Polygeist/77c04bb2a7a2406ca9480bcc9e729b07d2c8d077/. Source tags in
  brackets are local to the numbered section that uses them; section 10 lists them by section.
- The six areas (source; n-body; spectral-norm; mandelbrot; fannkuch-redux; the other six benchmarks) were read
  separately. Where they overlap or disagree, section 7 says so.

## Question

Which Benchmarks Game programs can xlang-v0 hold, and how, so that OQ-057's owner answer rests on evidence: the
source repository, a commit, and its license; per benchmark, which of xlang-v0's languages (C, C++, C#, Rust,
Python) and Fortran have programs, their output form, and the constructs each uses, measured against the published
subsets (Frontend Rules; the Python-subset, Rust, and C# rows) and cgeist's C-like subset (P5.7); and a proposed item
map for each holding option of OQ-057, with held-out inputs and tolerances.

Why it matters: P12.11 builds assets/bench/xlang-v0.yaml from OQ-057's answer, and P12.12 and P12.G run the gate over
it. The gate reading (plans/p12-frontends.md, Decisions taken at planning) needs every program to raise to a hub
module tagged structured (scf, memref, affine, arith), to read held-out inputs and write arrays through its
language's lassi_io binding, and to be judged by binary_io (bible, Oracles) against its own source build within the
item's tolerance.

Classification: factual for every read (repository, commit, license, programs, constructs, output forms). The
holding, the items, the splits, and Fortran's place are owner choices (OQ-057). This spike adds a Response to
OQ-057; it proposes no bible edit, and nothing enters a manifest.

## Summary

- Source: salsa.debian.org/benchmarksgame-team/benchmarksgame, one branch (master), newest commit
  40296663ed350d5fe4a6ab5e367bab61cb77c219 (2025-03-07, "word smith"), still the newest on 2026-10-18. The README
  calls the project "STOPPED". Program source is not tracked as separate files: it is one zip, `Z` (1,991,840
  bytes, sha256 aabcf6726cdc14f0f45b99e5daba48584f94bbb48883fd3711a1d040474d1cb4), plus highlighted program pages.
  The live site is that commit's public/ tree.
- License: a 3-clause BSD text, in LICENSE.md and in every benchmark directory of the zip; the site's license page
  titles it "Revised BSD license", and every program page links it. It covers the programs by placement and notice;
  CONTRIBUTING.md states no grant from contributors.
- Programs: every one of the 10 benchmarks has programs in all six languages (k-nucleotide's single Fortran program
  fails to link). The zip holds 304 program files in these languages; 303 are shown on the site, and the extra
  nbody.flang-6.flang is a byte copy of nbody.ifc-6.ifc. The site builds Fortran with Intel ifx; its gfortran pages
  are stale.
- Output: every upstream program reads N from argv or FASTA from stdin and writes text (mandelbrot a PBM stream) to
  stdout; none reads held-out inputs or writes a lassi_io array, so binary_io cannot judge any of them as written.
- Subsets: for n-body, spectral-norm, mandelbrot, and fannkuch-redux, no upstream program read (the simplest
  single-threaded one per language) fits its frontend's subset as written; the I/O alone puts each out. C's
  numeric cores come closest under cgeist. No Rust program for spectral-norm, mandelbrot, or fannkuch-redux is
  single-threaded. Fortran reaches only fir, then llvm, at 26eb4285, so no Fortran program reaches structured.
- Items (a): programs written here from the description, plus this spike's own-words reading of the upstream
  algorithm where the description defers to a program (n-body, spectral-norm, and mandelbrot; fannkuch-redux's
  description is complete), with one operation order over flat arrays and counted loops. They fit every published
  subset and cgeist's once P12.4, P12.8, and P12.9 publish those constructs (function calls with array arguments,
  array allocation, and while loops, or a bounded fallback), and the C and C++ programs reach structured only if
  P5.7's lassi_io.h route keeps the header's stdio calls out of the raised module (section 7). Proposed: n-body
  (max_abs 1e-8, PROJECTED and a choice, since the description's check covers only its two printed energies; seeds
  refused whose minimum pair separation over the run falls below a stated bound), spectral-norm (max_abs 0, ulp
  fallback), mandelbrot and fannkuch-redux (exact; integer outputs), each with seeded held-out inputs that avoid every
  published size. fasta is an optional fifth item; reverse-complement is possible but not recommended for v0;
  k-nucleotide, pidigits, regex-redux, and binary-trees are not items.
- Items (b): every program would need stdout_mask and low modules allowed, a zip-member fetch, and held-out inputs
  only as run_args sizes. spectral-norm, mandelbrot (at a power of two such as 512), fannkuch-redux, fasta, pidigits,
  and binary-trees could be items, three of them held out in name only; n-body is not one unless the oracle gains a
  numeric tolerance, and k-nucleotide, reverse-complement, and regex-redux are not unless the harness can feed stdin.
- Items (c): (a) now; later corpus of at most C, C++, and Fortran, low or partly low, minus any benchmark that is an
  eval or unassigned item (Agent Rule 5).
- Fortran: no structured route at the pin, so the evidence supports Fortran (a).

## 1. Source: repository, commit, layout, license, benchmarks, and programs

### Commands and outputs (trimmed)

Read-only, on the workstation in Git Bash, into a session scratch directory outside the repository.

- `GET https://salsa.debian.org/api/v4/projects/benchmarksgame-team%2Fbenchmarksgame`: id 19610, default_branch
  master, created_at 2018-04-01T21:50:06Z, last_activity_at 2026-06-15T20:49:32Z, description "Website that shows
  toy-program performance measurements for ~24 language implementations." [S2]
- `GET .../projects/19610/repository/branches`: one branch, master, at 40296663ed350d5fe4a6ab5e367bab61cb77c219 [S3].
- `GET .../repository/commits?ref_name=master&per_page=5`, newest first [S4]:

  ```
  40296663 2025-03-07T09:52:04-08:00 "word smith"
  b40635c4 2025-03-06 "charts & done"
  28fe5cdc 2025-03-05 "pidigits"
  b613e88a 2025-03-03 "word smith"
  6c33fee6 2025-03-03 "lua"
  ```

- `GET .../commits?ref_name=master&since=2025-03-08T00:00:00Z` returned `[]`: 40296663 is the newest commit on
  2026-10-18; the project's 2026-06-15 activity date is not a commit [S4].
- `GET .../repository/tree?ref=40296663...` for root, bencher, bencher/programs/*, public, public/download,
  public/description, public/performance, public/fastest, and public/program (16 pages, 1,589 entries) [S5].
- `curl -sI <raw zip URL>`: content-length 1991840. The files API gave x-gitlab-content-sha256 aabcf672...1cb4 and
  last commit b40635c4 (2025-03-06) [S6].
- `curl -sSL -o <scratch>/benchmarksgame-sourcecode.zip BG raw/public/download/benchmarksgame-sourcecode.zip`, then
  `sha256sum`: aabcf6726cdc14f0f45b99e5daba48584f94bbb48883fd3711a1d040474d1cb4; `git hash-object`:
  05648fc9a0d6220ad9427b99f3f6eee39eb2759f, the blob id the tree lists for that path [S5, S6].
- A scratch script under `python -I` over `zipfile.infolist()` (names, sizes, dates; nothing extracted): 1,106
  entries; top-level directories binarytrees, fannkuchredux, fasta, knucleotide, mandelbrot, nbody, nbodyinprocess,
  pidigits, regexredux, revcomp, spectralnorm, samplefannkuchredux, samplenbody, samplespectralnorm; top-level files
  README, LICENSE, JuliaCI-BaseBenchmarks-LICENSE; entry dates 2018-04-26 to 2025-03-05; 304 program files with the
  six extensions: .gcc 58, .gpp 61, .csharp 56, .rust 61, .python3 43, .ifc 24, .flang 1 [S6].
- Four files from the pages host (performance/nbody.html, description/nbody.html, license.html, and the zip) are
  byte-identical to public/ at 40296663, which .gitlab-ci.yml publishes from master [S1, S5].
- Raw reads at the commit: the 11 performance pages (10 benchmarks and nbodyinprocess), the 10 description pages,
  how-programs-are-measured.html, and the log sections of 27 program pages [S7, S8, S9].

### Repository, commit, and layout

- Repository: https://salsa.debian.org/benchmarksgame-team/benchmarksgame; the site's index links it as "project
  repo" [S1, S2]. Commit 40296663ed350d5fe4a6ab5e367bab61cb77c219 on master, its only branch [S3, S4]. The README
  calls the project "STOPPED", updated "once or twice a year" [S10]; site pages carry the release label 25.03 [S7].
- Root at the commit: `.gitlab/`, `bencher/`, `mybenchmarks/`, `public/`, `.gitlab-ci.yml`, `CONTRIBUTING.md`,
  `JuliaCI-BaseBenchmarks-LICENSE`, `LICENSE.md`, `README.md` [S5].
- Program source is the zip blob `Z`; the README points to it as the program source [S5, S6, S10]. Inside it, program
  #1 is `<bench>/<bench>.<ext>` and program #n is `<bench>/<bench>.<ext>-<n>.<ext>` (for example `nbody/nbody.gcc`
  is "C gcc" #1 and `nbody/nbody.gcc-9.gcc` is #9). Each benchmark directory also holds a `LICENSE`, a reference
  output `<N>_out`, and for stdin benchmarks a `<bench>-input1000.txt`. The top-level README marks the 10 benchmarks
  ACTIVE and five "obsolete,removed" (chameneosredux, fastaredux, meteor, regexdna, threadring) [S6].
- One source extension feeds several site language ids, as the build logs show (`cp nbody.csharpaot Program.cs`,
  `mv nbody.ifx nbody.f90`): `.gcc` builds as gcc, clang, and icx; `.csharp` as csharpcore (JIT) and csharpaot
  (NativeAOT); `.ifc` as ifc and ifx, and earlier gfortran; `.gpp`, `.rust`, and `.python3` have one id each [S6, S9].
- Site tree: public/description/<bench>.html, public/performance/<bench>.html, public/program/<bench>-<langid>-<n>.html
  (1,589 pages, each with source and build log), and the check files public/download/<bench>-output.txt and
  <bench>-input.txt [S5].
- Some program pages are stale, with a page but no zip member; among them fasta-gcc-9, spectralnorm-gcc-1,
  nbody-rust-8, mandelbrot-rust-8, fannkuchredux-rust-3, and pidigits-python3-2 [S5, S9], and the pages sections 2 to
  4 name: nbody-gpp-8, spectralnorm-gpp-8, spectralnorm-csharpcore-1, mandelbrot-gpp-8, and mandelbrot's Fortran #8
  pages. Every gfortran page is dated 2024-03-01. "Shown" below therefore means linked from a performance page [S5,
  S9].
- `bencher/programs/` holds the measurement tool's samples, not the site's programs: Java, PHP, and old-extension
  `.python` files for nbody, spectralnorm, fannkuchredux, and the obsolete regexdna [S5]. So at the commit the C, C++,
  C#, Rust, Python 3, and Fortran programs exist only as zip members and as rendered pages.

### License

- LICENSE.md: a 3-clause BSD text with no title, copyright 2004-2008 Brent Fulgham and 2005-2024 Isaac Gouy. Source
  redistributions keep the notice, conditions, and disclaimer; binary redistributions reproduce them in their
  documentation; the names "The Computer Language Benchmarks Game", "The Benchmarks Game", and "The Computer
  Language Shootout Benchmarks", and the contributors' names, may not endorse derived products; "AS IS", no
  warranty [S11]. public/license.html is the same license for 2005-2025, under the heading "Revised BSD license"
  (its h1), the name this spike uses [S12]; bencher/LICENSE and bencher/license.txt cover the bencher tool [S13].
- Inside the zip: the top-level LICENSE is the same text (2005-2024); every benchmark directory has its own LICENSE
  (2005-2019 in most, 2005-2021 in spectralnorm); JuliaCI-BaseBenchmarks-LICENSE puts five Julia programs under MIT
  "Expat" (none in the six languages); `pidigits/gmp.so` is a prebuilt shared library (648,360 B) that the BSD notice
  does not name [S6].
- Coverage of the programs: by placement and notice, not by explicit grant. Each benchmark directory carries the BSD
  LICENSE; each program page's footer links "3-Clause BSD License" to ../license.html [S9]; all 304 program files
  name the Benchmarks Game or the Shootout in their header and none names another license (a keyword scan;
  `fannkuchredux/fannkuchredux.gpp-4.gpp` points to the license page); CONTRIBUTING.md sets no license term [S14].

### Benchmarks and description pages

The site lists 10 benchmarks [S1]; `nbodyinprocess` is an n-body measurement variant with in-process timing, not an
11th benchmark [S8, S15].

| Benchmark (dir) | Input | Check size and tool | Size measured | Output form (stdout) |
| --- | --- | --- | --- | --- |
| fannkuch-redux (fannkuchredux) | n, permutation length (argv) | N=7, diff, 24 B | 12 | checksum line, then max-flips line (integers) |
| n-body (nbody) | step count (argv) | N=1000, `ndiff -abserr 1.0e-8`, 26 B | 50000000 | 2 lines, energy before and after, 9 decimals |
| spectral-norm (spectralnorm) | matrix order N (argv) | N=100, diff, 12 B | 5500 | 1 line, 9 decimals |
| mandelbrot | N for an N-by-N bitmap over [-1.5-i, 0.5+i] (argv) | N=200, cmp, 5,011 B | 16000 | PBM "P4" header, then packed bits |
| fasta | sequence-length scale N (argv) | N=1000, diff, 10,245 B | 25000000 | FASTA, 3 records (ONE, TWO, THREE), 60-column lines |
| k-nucleotide (knucleotide) | FASTA on stdin (fasta's output) | fasta N=1000 input, diff, 242 B | knucleotide-input25000000.txt | 1-mer and 2-mer percentage tables (3 decimals), then 5 counts |
| reverse-complement (revcomp) | FASTA on stdin | fasta N=1000 input, diff, 10,245 B | revcomp-input100000001.txt | FASTA |
| binary-trees (binarytrees) | max depth N (argv) | N=10, diff, 223 B | 21 | 6 lines of depths, tree counts, integer checks |
| pidigits | digit count N (argv) | N=30, diff, 45 B (the zip's own check is 27_out) | 10000 | 10 digits per line, tab, ":" and running total |
| regex-redux (regexredux) | FASTA on stdin | fasta N=1000 input, diff, 262 B | regexredux-input5000000.txt | 9 pattern-count lines, blank line, 3 lengths |

Each row's check size and tool come from its description page [S7], the measured size from the description and the
program pages' command lines [S9]. The zip's `<N>_out` files have the same git blob ids as the public/download
`*-output.txt` files, except pidigits (27_out against an N=30 file) [S5, S6]. What a description defines without
pointing to a program [S7]: fannkuch-redux gives the flip rule, checksum, and permutation order; spectral-norm gives
the matrix entries and the output; mandelbrot gives the region, bitmap size, and PBM format but not the iteration
limit or escape test; n-body defines its integrator only by reference to "the Java program" and gives no initial
bodies.

### Programs per benchmark and language at 40296663

Program numbers in the zip; the shown set (linked from public/performance/<bench>.html) equals the zip set for every
benchmark and language, except `nbody/nbody.flang-6.flang`, not shown, whose sha256 equals nbody.ifc-6.ifc's
[S6, S8]. The performance pages label the languages C gcc, C++ g++, C# naot (csharpaot), Rust, Python 3, and Intel
Fortran (ifx); csharpcore, clang, and icx are linked only from nbodyinprocess.html and public/fastest/; gfortran from
no performance page [S5, S8]. Statuses are the performance page's own markers [S8].

| Benchmark | C (.gcc) | C++ (.gpp) | C# (.csharp) | Rust (.rust) | Python 3 (.python3) | Fortran (.ifc; site: ifx) |
| --- | --- | --- | --- | --- | --- | --- |
| binarytrees | 1,2,3,5 | 1-7,9 | 1,2,3,6 (Empty),7 | 1-5 | 1-5 | 1,2 |
| fannkuchredux | 1-6,8 | 1-7 | 2,3,5,6,8,9 | 2,4,5,6 | 4,6,8 | 1,3 |
| fasta | 1-8 (4 Empty) | 1-7,9 | 1,5,8,9 | 1-7,9 | 1,2,3,5,8 | 1,3,4 |
| knucleotide | 1 | 1,2,3 | 1,4,6,7,8 | 1-7,9 | 1,3,4,8,9 | 1 (Empty; 183 B file) |
| mandelbrot | 1,3-9 | 0-7,9 | 1,2,3,5,7,8,9 | 1-7 | 2,3,5,7,8 | 4,5,6,7 |
| nbody | 1-9 | 0-7,9 | 0-4,6-9 | 1-7,9 | 1,2,8 | 1-6 (4 Bad Output); plus nbody.flang-6.flang, not shown |
| pidigits | 1,2 | 4,5 | 4,5,6,7 | 1-4 | 3,4,5 | 4 |
| regexredux | 2,3,4,5 | 1-6 (1,2,4,5 Empty) | 1,2,4,5 | 1-7 | 1,2 | 1 |
| revcomp | 2,4,5,6,7,9 | 1,2,3,5,6 (3 and 6 Bad Output) | 1,3,5,6,7,9 (9 Bad Output) | 1,2,3 | 1-6 | 1,2 |
| spectralnorm | 2-8 (2 Empty; AVX2-under-ivybridge build error) | 1,5,6,7 | 2,3,5,8 | 1-7 (7 Empty) | 4,6,7,8 | 2,3 |
| nbodyinprocess (variant) | 1,8 | none | 1,8 | none | 1,8 | none |

Upstream build toolchains, from program-page logs (upstream's facts, not LASSI's) [S9, S15]: gcc and g++ 14.2.0
(Ubuntu 14.2.0-4ubuntu2), `-pipe -O3 -fomit-frame-pointer -march=ivybridge`, with `-fopenmp`, `-pthread`, `-lgmp`, or
`-lpcre` per program; .NET SDK 9.0.100, net9.0, PublishAot true for naot; rustc 1.84.1 (LLVM 19.1.5), `-C
opt-level=3 -C target-cpu=ivybridge -C codegen-units=1`; CPython 3.13.0 with `-OO` after a pyright check; ifx
2024.1.0, `-O3 -march=ivybridge -ipo -static`; gfortran 13.2.0 on the 2024-03-01 pages only; host Ubuntu 24.04 on an
i5-3330. LASSI's proposed pins differ: python 3.10 (batch 1), .NET SDK 10.0.402, a rustc nightly, and gcc-12
(plans/spikes/p12-frontend-pins.md).

### Output form and binary_io

As written, no upstream program can be judged by binary_io: each writes stdout only, and under binary_io a reference
run must leave at least one readable lassi_io output file, or the trial ends at baseline-run (bible, Oracles, Rules
for binary_io, the references bullet). The upstream checks are stdout comparisons, which stdout_mask fits, except that
n-body's absolute error of 1.0e-8 has no form in stdout_mask's exact match [S7]. If the same quantities were written
as lassi_io arrays (bible, Oracles, the float and integer dtype bullets):

| Benchmark | Quantities | Comparison under binary_io |
| --- | --- | --- |
| n-body | f64 energies, or the final body state | float (max_abs, ulp, pcc); upstream's check is abserr 1.0e-8 |
| spectral-norm | f64 | float |
| k-nucleotide | percentages (f64) and counts (i64) | float for the percentages, exact for the counts |
| mandelbrot | packed bits or per-pixel values (u8) | integer, exact only; its floats stay internal |
| fannkuch-redux | checksum and max flips (i64) | exact |
| binary-trees | check values (i64) | exact |
| pidigits | digits | exact |
| regex-redux | counts and lengths | exact |
| fasta, reverse-complement | sequence bytes (u8) | exact; byte equality, the same as stdout equality |

## 2. n-body

Sources for this section: [R] the repository and commit; [D] description/nbody.html (byte-identical, 4,259 bytes,
to the commit's copy); [Z] the zip (nbody/ holds 119 members, among them LICENSE and 1000_out); [P:<id>] program
pages nbody-<id>.html; [O] public/download/nbody-output.txt; [T] public/data/data.csv; [L] the licenses; [CM], [CC],
[CH] PG's tools/cgeist/Lib/clang-mlir.cc, CGCall.cc, and clang-mlir.h; [G] GCC 12.3.0 Optimize Options. Commands
beyond section 1: `.../repository/tree?path=bencher/programs/nbody&ref=40296663...` (nbody.java, nbody.java-2 to -5,
nbody.php-3.php, nbody.python, nbody.python-2.python); raw program pages (MAKE and toolchain lines); `grep '^nbody,'`
over [T] (elapsed times); raw PG files.

What the description asks [D]: model the Jovian planets' orbits with "the same simple symplectic-integrator",
deferring the algorithm to the Java program; check with `ndiff -abserr 1.0e-8` at N = 1000; measure at 50000000. In
our words: a central body and four planets; first set the central body's velocity so total momentum is zero; each
step kicks every pair i < j (both velocities change by the pairwise inverse-square term times dt), then drifts every
position by dt times its velocity; the energy is kinetic plus pairwise potential. Every program prints the energy
before and after N steps at 9 decimals (`%.9f` in C, C++, Python; `{0:f9}` in C#; `{:.9}` in Rust; `f12.9` in
Fortran); at N = 1000 the lines are -0.169075164 and -0.169087605 ([O]; nbody/1000_out).

Programs [Z, P]: C gcc #1 to #9 (all in the zip); C++ #0 to #9 (9 members; page gpp-8 has none); C# #0 to #4 and #6
to #9 (all 9); Rust #1 to #9 (8 members; page rust-8 has none); Python 3 #1, #2, #8; Fortran ifx #1 to #6 (25.03
pages) and gfortran #1 to #6 (older pages) from nbody.ifc and nbody.ifc-2 to -6, plus nbody.flang-6.flang with no
page. A grep over the 45 members found no threads, OpenMP, Parallel, rayon, tasks, async, or do concurrent. SIMD
intrinsics or vector types appear in C gcc-4, -5, -9; C++ gpp-0, -4, -7 (gpp-2 asks for vectorization by pragma);
C# csharp-4, -6, -7, -9; Rust rust-5, -6, -7, -9. None of the six read uses bignum or recursion.

Programs read (the simplest single-threaded one per language; elapsed is the site's at N = 50,000,000 [T]):

| Language | Program | Member, lines | sha256 | Site toolchain [P] | Elapsed s |
| --- | --- | --- | --- | --- | --- |
| C | gcc #1 | nbody/nbody.gcc, 141 | 6c7a44f4f242b8aafb2e72c1814f67c5568b3815abe0a19a005fe20645a03676 | gcc 14.2.0 `-O3 -fomit-frame-pointer -march=ivybridge` | 5.230 |
| C++ | g++ #3 | nbody/nbody.gpp-3.gpp, 173 | b22f8afa7e51850fefdbbf7bf094e630a72a6e9b65f99eff9744f144f18488c3 | g++ 14.2.0, same flags | 5.328 |
| C# | #1 (csharpcore-1) | nbody/nbody.csharp, 171 | d28e9a0888e363c8a9142e5533536fb7fc90b4d570f704f5f095fc3f9f6b3fb2 | .NET SDK 9.0.100, `dotnet build -r linux-x64 -c Release` | 7.677 |
| Rust | #1 | nbody/nbody.rust, 159 | 1097288b773ff0666c453ad91aa9c04024cff3e12e548565d10435ca4e1980d3 | rustc 1.84.1 (LLVM 19.1.5), `-C opt-level=3 -C target-cpu=ivybridge -C codegen-units=1` | 5.518 |
| Python 3 | #8 | nbody/nbody.python3-8.python3, 136 | b0adb5671e270b37d1f398bfb8e77ad15bdc2d5898315a0d7e844ae5297088ab | Python 3.13.0 `-OO` | 529.378 |
| Fortran | ifx #1 (gfortran #1 on the older page) | nbody/nbody.ifc, 149 | ca109da051d637a9410e744500d891019a0c5c396c70540db6524d8a85908183 | ifx 2024.1.0; gfortran 13.2.0 `-O3 -march=ivybridge` | 4.547 (ifx) |

Why these: C #1 is the plainest (C #7, sha256 609a2c599fcbe8c81a248f1088e52a4b0d3e6b50609192fcdae4d065acabbd26, keeps
one global array per axis and no struct, closest to the subsets); C++ #3 has no templates, operator overloading, or
SIMD (#1 and #6 use templates, #5 and #9 overload operators, #2 has a pragma); C# #1 has plain classes (#0 uses
tuples and Span, #3 an array of pair objects); Rust #1 is the shortest (#2 to #4 run 204 to 239 lines); Python #8
uses plain index loops (#1 and #2 use a dict of tuples of lists, destructuring targets, and `** (-1.5)` and `**
0.5`); Fortran #1 uses plain do loops (#4 uses whole-array expressions, norm2, dot_product, sum, and reshape, and
reads its local `pe` before assigning it, nbody.ifc-4.ifc:54, :59, :62, a quirk faithful mode would keep, Agent
Rule 4).

Constructs (identifiers only):

- C gcc #1: nested `for` over pairs (j from i + 1) and a step loop in main; `struct planet` of seven doubles
  (:16-20) in an initialized global array of five (:87), passed by pointer; element pointers (`&(bodies[i])`,
  `b->x`, :27-40); `sqrt` (:33), `atoi(argv[1])` (:131), `printf("%.9f\n")` (:135, :138); macro constants; no heap,
  tuple, closure, iterator, SIMD, or thread.
- C++ g++ #3: `class Body` (:11-80) with seven doubles, a member-initializer constructor, `constexpr static const
  double` members, static factories returning Body by value, and a member function setting the velocity; `class
  NBodySystem` (:82-158) with an int and a `Body bodies[5]` member, a constructor that fills and offsets, and member
  functions energy and advance; the same pair loops; `std::sqrt`, `std::atoi(argv[1])` (:162), `std::printf` (:166,
  :170); no heap (automatic system object), templates, SIMD, or threads.
- C# #1: classes NBody (:9), NBodySystem (:24), and Body (:100) with seven public doubles, static factories that
  `new` a Body (:110-158), and OffsetMomentum returning `this` (:167); `Body[]` of references (:28); `for` over pairs
  and `foreach` over objects (:39, :69); `Math.Sqrt` (:56, :91), `Int32.Parse(args[0])` defaulting to 10000 (:12),
  `Console.WriteLine("{0:f9}")` (:16, :19); no tuples, LINQ, unsafe, SIMD, or threads.
- Rust #1: `static BODIES: [Planet; 5]` (:12) of a `#[derive(Clone, Copy)] struct Planet` (:61-66); `&mut [Planet;
  5]` and `&mut [Planet]`; `loop` with `match` on an Option (:71-75); `iter_mut` (:76), `iter` with `next` and
  `clone` (:103-110, :125); a generic `shift_mut_ref<'a, T>` using `std::mem::replace` and `split_at_mut` (:153-159);
  closures over `std::env::args_os` (:137-140); `println!("{:.9}")` (:144, :148), `f64::sqrt`; heap only for the
  argument String. It drifts body i right after body i's pair loop (:94-96), which by reading gives the same values
  as a separate drift loop.
- Python 3 #8: `class Body` with `__init__` setting seven attributes (:14-22), a list of Body objects (:77), `range`
  loops, a tuple assignment, `dx**2` (a pow call, :58), `math.sqrt` (:48, :59), `print("%.9f" % ...)` (:127, :130),
  `int(sys.argv[1])` (:136).
- Fortran #1: a derived type `body` of seven `real*8` components (:19; a nonstandard spelling); named constants built
  with structure constructors (:22-55) and an array constructor `(/ ... /)` (:58); assumed-shape dummies with
  `size()`; internal procedures after `contains` (:72), with offsetMomentum's index `i` the main program's by host
  association (:12, :81); `getarg` (:60), a vendor extension (#4 uses the standard `get_command_argument`; whether
  flang at 26eb4285 accepts `getarg` was not read), then an internal list-directed read (:61); `write(*,'(f12.9)')`
  (:65, :70).

Expression order differs among the upstream programs, so their f64 results are not bitwise equal in principle: mag is
`dt/(d*d*d)` in C, C++, C#, and Fortran but `dt/(d2*sqrt(d2))` in Rust and Python; the kinetic term is `(v.v)*m/2.0`
in Rust and `0.5*m*(v.v)` elsewhere; Python squares through pow. The site's 1e-8 check on 9-decimal text absorbs
these.

Subsets: cgeist's C-like subset has no published list (P5.7), so it was read in PG's source, not run. Loops, scalar
arithmetic, and arrays of scalars map to scf or affine, arith, and memref; `sqrt` becomes `math.sqrt` (CGCall.cc:
544-557); under `memref-abi`, on by default (clang-mlir.cc:57), a struct whose fields share one type becomes a
memref of that type (:5558-5563, :5640), an array of them gains a dimension (:5664-5672), and `getMLIRType` merges by
default (clang-mlir.h:140-141); a mixed struct, or one with LLVM-type fields, becomes an LLVM struct (:5617-5621); a
variadic call such as printf becomes `llvm.call` (CGCall.cc:1666-1688); a global becomes `memref.global`
(clang-mlir.cc:4924) whose initializer survives only if cgeist folds it, else a warning "warning not initializing
global" (:4950-4976). The member-access and pointer-parameter paths were not read.

| Language | File | Fits which subsets | What falls outside |
| --- | --- | --- | --- |
| C | nbody/nbody.gcc | cgeist C-like: the numeric core, by source (loops, f64 arithmetic, `math.sqrt`, the struct as memref<7xf64>, bodies as memref<5x7xf64>) | `printf` (llvm.call); `atoi` of argv (text input, not lassi_io); the initialized global (kept only if it folds; not read); text output |
| C++ | nbody/nbody.gpp-3.gpp | none | constructors, static factories, member functions; NBodySystem (an int beside a Body array) becomes an LLVM struct by source (Body alone would be a memref; clang-mlir.cc:182-206, :5640); `std::printf`, `std::atoi`, text output |
| C# | nbody/nbody.csharp | Frontend Rules: `for` loops, double arithmetic, `Math.Sqrt` only | classes, constructors, factories, `return this`, arrays of references, `foreach` over objects, `Int32.Parse`, composite formatting |
| Rust | nbody/nbody.rust | Frontend Rules: range loops, f64 arithmetic, `sqrt` only | the struct array and slices of structs, iterator methods, `loop`/`match` on Option, the generic with a lifetime, `mem::replace`, `split_at_mut`, closures, `println!` |
| Python 3 | nbody/nbody.python3-8.python3 | Frontend Rules: range loops, float arithmetic, `math.sqrt` only | the class and attributes, the list of objects, the tuple assignment, `**2`, `%` formatting, `sys.argv` and `int` |
| Fortran | nbody/nbody.ifc | none (no structured route at the pin) | FIR leaves only through cf and llvm at 26eb4285 (bible Fortran row; p12-frontend-pins.md, Flang finding 3); also the derived type, constructors, assumed-shape dummies, host association, `getarg`, the internal read, the formatted write |

Confidence: high for the commit, hashes, and lines; the timings are the site's; medium for the cgeist mappings (read,
not run; member access, pointer parameters, and initializer folding not read to the end).

## 3. spectral-norm

Sources for this section: [B1] the API (commits, branches, tree at the commit); [B2] description/spectralnorm.html
(last changed in 6c33fee6, 2025-03-03); [B3] LICENSE.md; [B4] public/download/spectralnorm-output.txt; [B5]
bencher/makefiles/my.linux.ini; [B6] the zip; [B7] program pages; [B8] performance/spectralnorm.html; [G1], [G2] PG's
clang-mlir.cc and CGCall.cc; [P1] the Python 3.12 What's New; [N1] the numpy.sum manual page. Commands beyond
section 1: tree reads of bencher/programs/spectralnorm (java x3, php x3, python-6, python-7) and
mybenchmarks/websites/linux/code (the same set); `od -c` of [B4] (`1.274219991\n`, 12 bytes); raw program pages
spectralnorm-{gcc-1..8, gpp-1,5..8, csharpcore-1,2,3,5,8, rust-1..7, python3-4,6,7,8, gfortran-2,3, ifx-2,3}, tags
stripped; whitespace-insensitive diffs of each chosen page against its zip member (0 differing lines for gcc-8,
gpp-1, csharpcore-8, python3-8, rust-4, and gfortran-2 against ifc-2); `commits?path=` for dates.

What the description asks [B2]: the spectral norm of the infinite matrix A with entries a11=1, a12=1/2, a21=1/3,
a13=1/4, a22=1/5, a31=1/6, and so on, by the same algorithm as the other programs, in 4 separate functions "like
the Java program"; check at N=100 and measure at N=5500. With 0-based indices the entry is 1/((i+j)(i+j+1)/2 + i + 1)
(derived from the listed entries; all six programs below use it). All six start from the all-ones vector, apply AtA
20 times (10 passes of two), and output sqrt((u.v)/(v.v)). Output: one line, 9 decimals; 1.274219991 at N=100 [B4],
1.274224153 at 5500 [B7]. Published sizes: 100 and 5500 [B2]; `spectralnorm = 1000 2000 3000` in the bencher
sample [B5]; program defaults 100 (C #8, C# #8, Python 3 #8, Rust #4) and 2000 (C++).

Programs (site pages against zip members) [B1, B6]: C pages gcc #1 to #8, members #2 to #8; C++ pages g++ #1 and #5
to #8, members `spectralnorm.gpp`, -5, -6, -7; C# pages #1, 2, 3, 5, 8, members csharp-2, -3, -5, -8; Rust #1 to #7,
all seven; Python 3 #4, 6, 7, 8, all four; Fortran gfortran and ifx #2 and #3, members ifc-2 and ifc-3.

| Language | Member (site page) | Bytes | sha256 | Why this one | Site build |
| --- | --- | --- | --- | --- | --- |
| C | spectralnorm/spectralnorm.gcc-8.gcc (gcc #8) | 1101 | 5ae8e0c4d2cb97b744d682e4e974a22b727b5209ecc3293d4d1581ccf86d7574 | single-threaded; #1 is not in the zip; #2 to #7 use OpenMP with SSE/AVX intrinsics or `vector_size` types | gcc 14.2.0 -O3 -march=ivybridge, no -fopenmp; 5.41 s at 5500 |
| C++ | spectralnorm/spectralnorm.gpp (g++ #1) | 1355 | 2842f77e9fcbdb31f4b722005348c76735b52671ccd99794f2cb780484b127e2 | single-threaded; #5 to #8 use OpenMP with SSE2 or `vector_size`, plus templates | g++ 14.2.0 -O3 -march=ivybridge; 5.34 s |
| C# | spectralnorm/spectralnorm.csharp-8.csharp (csharpcore #8) | 1373 | 2f8d281d9a93d5b0c134f3c985e8927dff88ace06d0c08fb8f26bef95aa693e6 | single-threaded; #2 Thread with a custom barrier, #3 Task.Run with Barrier, #5 Parallel.For with Vector128/Sse2 and unsafe; #1 not in the zip | .NET SDK 9.0.100, net9.0, ImplicitUsings |
| Rust | spectralnorm/spectralnorm.rust-4.rust (#4) | 2939 | c33712a6b2c264c937c9fbca390d0643f51058ce05d1fc843c1b1cebba63478d | no Rust program is single-threaded: #1 to #3 spawn std::thread over raw-pointer chunks with an unsafe Send wrapper; #4 to #7 use rayon (#6 adds std::arch SSE, #7 generic_array and numeric_array); #4 is the shortest with no unsafe or SIMD type | rustc 1.84.1 (LLVM 19.1.5), opt-level=3, target-cpu=ivybridge; 0.72 s elapsed, 2.85 s CPU |
| Python 3 | spectralnorm/spectralnorm.python3-8.python3 (#8) | 914 | 008630d2ec8f650d813004a250970305b51ac6d2773cbdd84f059e02f398894b | single-threaded; #4 and #7 use multiprocessing Pool(4); #6 adds with builtin sum() over comprehensions | Python 3.13.0 `-OO`; about 6 min at 5500 |
| Fortran | spectralnorm/spectralnorm.ifc-2.ifc (gfortran #2, ifx #2) | 1960 | 874cec6adbc5ca9c1dfc0f0ba908ae09f4d4ef87efb76360d62577b83a802f95 | single-threaded; #3 uses OpenMP | gfortran 13.2.0 -O3 -march=ivybridge -fopenmp (no directives in the program) |

The C, C#, and Python 3 files call themselves transliterations of one C original and, by reading, do the same
floating-point operations in the same order.

Constructs (from [B6, B7]): none of the six uses SIMD intrinsics, bignum, or recursion.

- C #8: nested `for` with int indices; stack VLAs of double sized by N; `const double u[]` parameters; an int
  denominator, a reciprocal times u[j], sequential sums; atoi, sqrt, printf `%0.9f`.
- C++: nested `for` with an int index against unsigned `size()`; `std::vector<double>` by reference; `std::fill`;
  cout with `setprecision(10)`. Quirk: the A and At routines add into their output without clearing it, and main
  clears u but never v, so from the second pass on v carries the previous pass into its sum; the printed value
  still matches C's to 10 digits [B7], but u and v differ from C's (derived). Faithful mode would keep this (Agent
  Rule 4).
- C# #8: nested `for`; `double[]` with `new double[N]` per AtA call; one class whose four routines are instance
  methods called from static Main; Int32.Parse, Math.Sqrt, Console.WriteLine `{0:f9}`; relies on implicit usings (no
  using directive); the same order as C.
- Rust #4: rayon `par_iter_mut` with `for_each` over output pairs, an inner loop over `iter().enumerate()`;
  `Vec<[f64; 2]>`, slices, `[usize; 2]` values, tuple destructuring; a generic routine taking the entry function as
  `F: Fn(...) + Sync`, the transposed case an argument-swapping closure; `zip`, `map`, `fold`; `vec!`, `assert!`
  rejecting odd n, `#[inline(never)]`; env::args with parse, and_then, unwrap_or; println! `{:.9}`. Numerics: a
  64-bit usize denominator, u[j] divided by it (one rounding where C has two), each row summed in two lanes (even j,
  odd j) added at the end.
- Python 3 #8: nested `for` over range; lists whose elements start as int and become float; true division (a float
  denominator, exact here); sys.argv, int, math.sqrt, `%.9f`; a main guard; the same order as C.
- Fortran #2: nested 1-based do loops; `real(dp)` allocatable arrays, assumed-shape dummies, array-valued function
  results and their temporaries; one module of contained pure functions; get_command_argument, an internal
  list-directed read into a 6-character buffer, size, dot_product (order left to the compiler), sqrt,
  selected_real_kind; write `(f0.9)`.

| Language | File | Fits which subsets | What falls outside |
| --- | --- | --- | --- |
| C | spectralnorm/spectralnorm.gcc-8.gcc | cgeist C-like: loops, arrays, arithmetic; VLAs become `memref.alloca` with a dynamic size (clang-mlir.cc:535-545); sqrt becomes `math.sqrt` (CGCall.cc:544-555) | printf is variadic, so `llvm.call` (CGCall.cc:1666) with its format string as `llvm.mlir.global` (clang-mlir.cc:730-734, 4984-4996); whether that module counts as structured is P5.7's tag rule; argv and atoi handling not read; no lassi_io input or output |
| C++ | spectralnorm/spectralnorm.gpp | none | std::vector, iterators, `std::fill`, iostream are library C++ beyond the C-like subset (ClangIR, out of P12, covers the rest; whether cgeist at 77c04bb2 raises libstdc++ templates is unmeasured); text output |
| C# | spectralnorm/spectralnorm.csharp-8.csharp | Frontend Rules: loops over `double[]`, arithmetic, Math.Sqrt | not named: instance methods, object creation, `new double[]`, calls with array arguments; outside: argument parsing, console formatting; no Parallel.For |
| Rust | spectralnorm/spectralnorm.rust-4.rust | Frontend Rules: the rayon parallel-for (whether `par_iter_mut` is covered by the rule's `par_iter` is P12.8's call), loops, arithmetic, sqrt | not named: other closures, generics with trait bounds, fixed arrays and tuples as values, iterator adapters, macros; outside: argument parsing, println!, the panic path |
| Python 3 | spectralnorm/spectralnorm.python3-8.python3 | Frontend Rules: loops, arithmetic, math.sqrt | lists instead of arrays, with an element type that changes from int to float; argv; printing; no prange |
| Fortran | spectralnorm/spectralnorm.ifc-2.ifc | none (fir, then llvm, at the pin; at best low through LLVM import, P12.7, once flang exists, P12.10, OWNER on OQ-063) | no published Fortran subset; text output |

Cross-language behavior of the upstream programs: none can be judged by binary_io as written (one text line each;
under stdout_mask the mask list would be empty, since none prints a timing line). C, C++, C#, and Fortran compute
(i+j)(i+j+1) in 32-bit int, below 2^31 only while N <= 23171 (derived); above that C and C++ overflow (undefined
behavior), C# wraps, Rust #4 uses 64-bit usize, and Python is unbounded. Rust divides where the others multiply by a
reciprocal and sums in two lanes, and C++ carries the accumulation quirk, so the programs agree at 9 decimals but not
bitwise in their arrays (derived; nothing run). Rust #4 to #6 accept only even n (spectralnorm.rust-4.rust:24,
rust-5:61, rust-6:63 assert it). Rust #7 has no even-n assert: it rounds n up to a multiple of 4 without a message
(rust-7:33-34), asserts n <= 23170 (:37), computes its denominators in i32 lanes (:102), and allocates n/4 four-lane
vectors (:39-41), so at any n that is not a multiple of 4 it prints the norm of a larger order. Python #4 and #6 add
with builtin sum(), which since CPython 3.12 uses Neumaier summation [P1], so on the site's 3.13 their rounding
differs from #8's left-to-right order; batch 1's pinned 3.10 predates the change.

Confidence: high for the upstream facts (named commit, sha256); medium for the subset judgments (P12.4, P12.8, and
P12.9 have not published beyond the bible's one line; cgeist beyond the lines read is unmeasured).

## 4. mandelbrot

Sources for this section: [M1] the repository and commit; [M2] description/mandelbrot.html (byte-identical, 4,264
bytes, to the commit's copy); [M3] public/download/mandelbrot-output.txt (5,011 bytes, sha256
97610473750700638fc63d13cfa49d339b67c18e7f26b3f9c9acb61e746472d5: an 11-byte header, P4 then 200 200, and 200 rows of
25 bytes; mandelbrot-output.pbm is byte-identical); [M4] the zip (135 members under mandelbrot/); [M5] program pages
for gcc-2, gcc-8, gpp-3, gpp-8, csharpcore-2, csharpcore-8, rust-5, python3-3, python3-8, gfortran-5, ifx-5, ifc-5;
[M6] public/measurements/gcc.html; [M7] the licenses; [M8] the netpbm PBM page; [M9] MathWorld, the description's
Background link; [M10] GCC 14.2 x86 options; [M11] GCC 14.2 optimize options; [M12] PG's
tools/cgeist/Test/Verification/; [M13] libstdc++ releases/gcc-12.3.0 include/std/complex. Each chosen page's source
equals its zip member with whitespace removed; the gcc-2 page shows gcc-8's source; the ifc-5 page holds only a PHP
parse error (167 bytes).

The benchmark [M2]: plot the set over [-1.5-i, 0.5+i] on an N-by-N bitmap and write it as binary PBM; check with cmp
at N = 200 against [M3]; measure at N = 16000 (the site measures 1,000 and 16,000 [M6]; gcc #8 at 16,000 shows 25.18
cpu s and 25.171 s elapsed, one core). The description states neither the iteration count nor the escape test; the
programs use 50 iterations and |z|^2 > 4 from z = 0 (gcc-8:14-15, 26, 29), while MathWorld gives radius 2 from z_0 =
C [M9]. PBM rows are Width bits packed 8 to a byte, most significant first, 1 black, with "don't care" fill bits
[M8]. All three published sizes are multiples of 8, so no published check reaches a row's fill bits.

Programs per language (members): C 8 (mandelbrot.gcc, gcc-3 to gcc-9; only gcc-8 is single-threaded, the others use
pthreads or OpenMP, several with `vector_size` types or SSE2); C++ 9 (mandelbrot.gpp, gpp-0, gpp-2 to gpp-7, gpp-9;
the site's gpp-8 has no member; single-threaded: gpp-2, gpp-3, gpp-5; the others use OpenMP, std::thread, and AVX or
SSE); C# 7 (mandelbrot.csharp, csharp-2, 3, 5, 7, 8, 9; single-threaded: csharp-2 and csharp-8; csharp-3 and -5
start Threads; the rest use `Parallel.For` with Vector or Vector256); Rust 7 (mandelbrot.rust, rust-2 to rust-7; none
single-threaded: std::thread, a futures_cpupool, or rayon); Python 3 5 (python3-2, 3, 5, 7, 8; single-threaded:
python3-3 and python3-8); Fortran 4 (ifc-4 to ifc-7; ifc-5 has no OpenMP directive though the site builds it with
-fopenmp; the site's #8 Fortran pages have no member). "Simplest" means the fewest constructs beyond counted loops and
scalar arithmetic, then the fewest bytes.

| Language | Member (site page) | Bytes | sha256 | Constructs (file:line) |
| --- | --- | --- | --- | --- |
| C | mandelbrot/mandelbrot.gcc-8.gcc (gcc-8; also gcc-2, clang-8) | 1303 | c785a5031762cb7818d4b3ca595dbce90c68279763aebd5625c0ef4956943246 | one main (10-57); N from atoi (18); header by printf (20); nested `for` with double counters (15, 22, 24); an inner loop with a two-part condition that ends early (29-35); double arithmetic (31-34); a char accumulator with shift and or (13, 37-38), padded by an int-modulo shift (50); putc (44, 51); no arrays, structs, heap, recursion, threads, SIMD, or bignum |
| C++ | mandelbrot/mandelbrot.gpp-3.gpp | 1121 | f09c079e5d619a9797ebc81dea60407f21f176ca76bdc0e0c7f7281b273008ac | atoi (19); iostream (sync_with_stdio, std::cout, operator<<, std::endl; 18, 21, 34, 36); `std::complex<double>` with overloaded * and +, std::norm (10, 26, 30, 32); the alternative token `and` (29); nested int `for` (23-24); early exit (29-30); char accumulators (14, 32, 35) |
| C# | mandelbrot/mandelbrot.csharp-2.csharp (csharpcore-2, csharpaot-2) | 1355 | 9d4b7ec19252fa6d38cdc435f853afb1d2e58388a73726a852d1b91c1d68c622 | a class with static Main (10-12); Int32.Parse, default 100 (14-16); Console.WriteLine with a composite format (22-23); a Stream from Console.OpenStandardOutput with WriteByte (24, 49, 55); nested int `for` (26, 31); a do-while with break and a pre-decrement condition (38-46); records escaped bits and inverts with XOR 0xff, so fill bits come out 1 (49, 55) |
| Rust | mandelbrot/mandelbrot.rust-5.rust | 2016 | 1547d36fd3d91369d803c427c13293bffb8b0c16de0ad423f7382d7131554c18 | crates rayon, num_traits, generic_array, numeric_array (8-16); 8-lane f64 NumericArray with overloaded arithmetic (19, 25-38); escape test every 5 steps (31-43); closures in `all` and `fold` (40, 46-48); Option combinators on args (52-55); N rounded down to a multiple of 8 (57); coordinates as x times a precomputed 2/N (59, 63, 75); heap Vecs (61, 72); `into_par_iter().map().collect()` into a Vec of Vecs (72-78); println! (70); stdout lock and write_all (66-68, 80-82) |
| Python 3 | mandelbrot/mandelbrot.python3-8.python3 | 1414 | 4fd93dceb9dab51b11ed25821ba93c346b7296b93c2bf73f69a6726d06645601 | chained assignments (12-16, 26); int of sys.argv (18); an f-string header via sys.stdout.write (20); nested `for` over range (22, 24); a fixed 50-step loop with an if guard, no break (29-34); int shift and or (37-38, 50); bytes() to sys.stdout.buffer (43, 51); no arrays, NumPy, or complex (python3-3, 1026 bytes, sha256 26b2be1b..., is shorter but uses complex, `abs(z) >= 2`, and for-else) |
| Fortran | mandelbrot/mandelbrot.ifc-5.ifc (gfortran-5, ifx-5) | 1943 | ca89f92a519751dca38eef276d1aa073cc5aaafbc7d1f605e8b6909b7142fe83 | kind parameters (12-18); `complex(kind=dp)` with cmplx, `**2`, conjg, real (21, 34, 38-39); do loops with exit (32-43); a logical flag (22, 36-44); `integer(kind=1)` bytes with ior and ishft (14-16, 46, 49, 61), mod (61); getarg with an internal list-directed read (24-25); formatted writes with advance='no' and char() (30-31, 55, 62) |

Subsets judged against: the Frontend Rules line (BIBLE.md:532) plus the binding's calls (P12.4, P12.8, P12.9), and
cgeist's C-like subset (BIBLE.md:516; P5.7) judged from cgeist's tests at PG [M12]: cout.cpp (std::cout) and
vector.cpp (GCC `vector_size`) are XFAIL; simpcomplex.cpp:1-2 and float_real_to_complex.c:1-2, 11-17 handle complex
only under --struct-abi=0 as llvm.struct, the default path saying "we dont support this yet"; fscanf.c:22-44 turns a
stdio call into llvm.call with llvm.mlir.addressof and polygeist.memref2pointer beside scf, affine, and memref.

| Language | File | Fits which subsets | What falls outside |
| --- | --- | --- | --- |
| C | mandelbrot/mandelbrot.gcc-8.gcc | cgeist C-like: it is C and raises; a clean structured module is not expected (not run) | double loop counters give no integer induction variable (no scf.for or affine.for); the early exit is a while form; printf, putc, atoi become llvm.call with string globals and polygeist.memref2pointer; no arrays, no lassi_io |
| C++ | mandelbrot/mandelbrot.gpp-3.gpp | none | iostream (cout.cpp XFAIL); std::complex is a class template over `__complex__ double` [M13], handled only as llvm.struct under --struct-abi=0; operator overloading; no lassi_io |
| C# | mandelbrot/mandelbrot.csharp-2.csharp | Frontend Rules: the kernel's multiply, add, compare, XOR, and shifts | Console and Stream I/O, Int32.Parse, composite formatting; the do-while with break is not a loop over an array; no arrays |
| Rust | mandelbrot/mandelbrot.rust-5.rust | Frontend Rules: the inner arithmetic | a parallel map with a closure and a nested-Vec collect (the rule names `par_iter`); closures in all and fold; iterator adaptors and Option combinators; println!, stdout lock and write; lane arrays with overloaded operators; three crates outside P12.1's vendored rayon closure, so no offline build |
| Python 3 | mandelbrot/mandelbrot.python3-8.python3 | Frontend Rules: the loop nest over range, the guarded fixed-count loop, scalar float arithmetic (mostly in) | sys.argv, the f-string header, sys.stdout writes, bytes(); whether shifts on unbounded ints count as subset arithmetic, and at what width, is P12.4's call; no arrays |
| Fortran | mandelbrot/mandelbrot.ifc-5.ifc | none (no published Fortran subset; fir, then llvm, at the pin, BIBLE.md:519; OQ-064) | complex arithmetic, getarg, formatted I/O |

Upstream outputs differ across languages (from reading, not runs): rust-5 (also rust-2:59, rust-7:57) rounds N down
to a multiple of 8 (rust-5:57), so at N = 257 it writes a 256-by-256 image with another header; csharp-2 writes 1s
into a row's fill bits (55) where gcc-8 and python3-8 write 0s (gcc-8:50; python3-8:50), both legal PBM but
different bytes; rust-5 computes coordinates as x times (2/N) where gcc-8, csharp-2, and python3-8 divide (2x) by N,
so the last bit can differ and move boundary pixels at sizes other than 200 (an inference); python3-3 tests |z| >= 2
through abs. So the bible's "identical outputs" (BIBLE.md:593) rests on the site's N = 200 cmp check, not on how the
programs are built.

Confidence: high for file facts, constructs, threads and SIMD, and the Rust rounding and C# fill-bit differences;
medium for subset fits (subsets unpublished in code; cgeist judged from its tests); boundary-pixel movement is an
inference.

## 5. fannkuch-redux

Sources for this section: [D] description/fannkuchredux.html (site header 25.03; sha256 of the fetched file
bd9265b28893996018f1a09d7376de11c8b93ba4238449c6af9e5bded768b07b); [O] download/fannkuchredux-output.txt; [X]
public/download/fannkuchredux-extra.txt (the permutation order for N = 7); [P] performance/fannkuchredux.html; [C]
commits and branches; [T] the tree of bencher/programs/fannkuchredux (8 blobs: three Java, three PHP, python-4,
python-6); [Z] the zip (the files API's content_sha256 matches, last_commit_id b40635c4); [L] LICENSE.md; [Pg]
program pages fannkuchredux-gcc-8, -gpp-3, -csharpaot-8, -rust-4, -python3-8, -ifx-1; [PG] PG's clang-mlir.cc (sha256
cea53e24...), CGStmt.cc (8d7426bc...), and Test/Verification whileset.c, capture.cpp, struct.cpp, vector.cpp; [OEIS]
A000375 (Topswops). Members were read in memory by a scratch reader under `uv run python -I`; nothing extracted,
built, or run.

What the description asks [D]: for each permutation of 1..n, repeatedly reverse the first k elements, k being the
first element, until the first is 1, counting flips; add the count to a checksum at even permutation index and
subtract it at odd; report the checksum and the maximum. The permutation order is fixed: rotate the first n elements
left n times, recursing on the first n-1 at each rotation; a count[] array tracks rotations per level, and the next
permutation comes from a rotation at level 1 that carries upward when a level's count passes its level number; index
to permutation is count[k] = (index mod (k+1)!) / k!, from k = n-1 down to 1. N = 7 checks output (228, then
"Pfannkuchen(7) = 16" [O]); N = 12 is for performance (every page shows 3968050 and 65 [Pg]); [X] lists the
permutations of 1..7 in order.

Programs per language [Z]: C 7 (single-threaded by [P]: gcc-1, -3, -4, -8; gcc-4 uses SSE); C++ 7 (gpp-3, gpp-7;
gpp-7 uses SSE); C# 6 (csharp-2, csharp-8); Rust 4 (none: every one uses about 4x CPU over elapsed, #6 3.81/15.05,
#4 6.94/27.12, #5 7.41/28.89, #2 9.19/36.04; #2 std::thread, #4 and #5 rayon, #6 rayon with SSE); Python 3: 3
(python3-6, python3-8); Fortran 2 (ifc #1). Single-threaded means CPU seconds match elapsed in [P]. The site's
csharpaot-8 is member csharp-8 and ifx-1 is member ifc (byte identity not checked). C gcc-8, C# csharp-8, and Python
3 #8 call themselves transliterations of one Scala program by the same contributor.

| Language | Member | Bytes | sha256 | Site build [Pg] |
| --- | --- | --- | --- | --- |
| C | fannkuchredux/fannkuchredux.gcc-8.gcc | 1837 | 65c8d4b7b5697e1ee1ccff75aec8fb66ae97811fbd96b02a0ef46941c49dea97 | gcc 14.2.0 -O3 -march=ivybridge |
| C++ | fannkuchredux/fannkuchredux.gpp-3.gpp | 1413 | fccf786c37f0ad3939f77727cacc2ce629aed85b6e0963eeab5ae8cb4c12d06f | g++ 14.2.0 -O3 |
| C# | fannkuchredux/fannkuchredux.csharp-8.csharp | 2014 | d9d7a87488a1be0dd06c8823d53310104e82b5bca4c85af1b2c013607d0ed5ff | .NET SDK 9.0.100, PublishAot, ImplicitUsings |
| Rust | fannkuchredux/fannkuchredux.rust-4.rust | 5586 | 367a0a0187b3ee6744f9a455a170a5b7f0307046a4dd245d5f8678e89ccf2e33 | rustc 1.84.1 called directly, -L to a prebuilt rayon |
| Python 3 | fannkuchredux/fannkuchredux.python3-8.python3 | 1585 | fad3eec38b9b516c4c0e8b49b2466d8f689d912f08b3d72bc558b051721fb2af | CPython 3.13.0 -OO |
| Fortran | fannkuchredux/fannkuchredux.ifc | 2023 | 9b664a0078e800c9f0c60fd16127fa90f51b196cbd711e5a184e6ef6d06161a8 | ifx 2024.1.0 -qopenmp (no OpenMP directive in the program) |

None of the six uses recursion, bignum, or floating point; every value is a small integer; each reads N from argv
and prints two text lines. No frontend source exists on this branch (lassi/frontends/ is absent), so verdicts are
against the Frontend Rules sentence; "not named" means the frontend task must publish or refuse the construct.

- C gcc-8: one function and main; three int VLAs sized by n; while loops and one counted for; the flip bound read
  from the array; swaps through a temporary; a bool flag loop; an early return from two loops deep that also prints;
  parity by `& 0x1`; atoi, printf. In cgeist at PG: VLAs become memref.alloca with a dynamic size
  (clang-mlir.cc:536-545); while loops carry keepRunning and noBreak flags in memref allocas (CGStmt.cc:779-830), and
  whileset.c expects a while(1)-with-break to raise to scf.while; a return inside loops stores false into every
  enclosing loop's flags (CGStmt.cc:1195); string literals become llvm.mlir.global (clang-mlir.cc:733, :4984-4993).
- C++ gpp-3: u8 elements through a typedef; fixed 16-element arrays; helpers with pointer plus count, a bool return,
  while with break; a two-int struct returned by value (in reach: a same-typed struct becomes a memref,
  clang-mlir.cc:5617-5640); std::generate with a lambda capturing a local by reference (capture.cpp raises a
  capturing lambda through !llvm.struct with llvm.store, llvm.getelementptr, and polygeist.memref2pointer);
  std::copy, std::reverse, std::max on raw pointers (instantiations from the host libstdc++, not read); a do-while
  whose condition uses the comma operator; atoi, printf.
- C# csharp-8: a static class and method; `new int[n]` (GC heap); while loops and one for; a bool flag loop; an early
  return from a nested loop with Console.WriteLine; `& 0x1`; Int32.Parse over String[] args; Console.Write with a
  composite format; no `using` line (ImplicitUsings).
- Rust rust-4: rayon `into_par_iter` over block numbers, `map` with a closure, `reduce` with identity and combiner
  closures into an (i32, i32) tuple; [i32; 16] and [u32; 16] arrays; `iter_mut().enumerate()`, `(1..n).rev()`;
  copy_from_slice, reverse on a subslice, mem::replace, cmp::min and cmp::max; `loop` with a return from the closure;
  while loops; if-expressions; `as` casts; argv through Option combinators; println!.
- Python 3 #8: a def with an int parameter; lists as `[0] * n`; for over range; while loops, some on one line with
  semicolons; chained assignment; `& 0x1`; a bool flag loop; an early return with print; module-level `from sys
  import argv` and int(); print with `%`.
- Fortran ifc: a main program with an internal subroutine; automatic zero-based arrays sized by an argument;
  whole-array assignment; `do while (.true.)` with exit and return; counted do loops; ishft, max, modulo;
  declaration-initialized locals (implicitly SAVE; harmless, run once); get_command_argument; internal formatted read
  and write with an I2 read (N at most two digits); trim, adjustl, concatenation.

| Language | File | Fits which subsets | What falls outside |
| --- | --- | --- | --- |
| C | fannkuchredux/fannkuchredux.gcc-8.gcc | cgeist C-like: loops, VLAs, early return, by source | only the text I/O: printf's string globals put llvm ops in the module (P5.7's rule decides the tag); argv parsing and text output are outside the gate's I/O contract |
| C++ | fannkuchredux/fannkuchredux.gpp-3.gpp | cgeist C-like in part (the struct) | the lambda; the `<algorithm>` templates; printf |
| C# | fannkuchredux/fannkuchredux.csharp-8.csharp | Frontend Rules: loops, int arithmetic, ifs | not named: array allocation, the early return; outside: Console I/O, string parsing, implicit usings; the NativeAOT-LLVM fallback is not served at the pins (BIBLE.md :523; OQ-064), so refused |
| Rust | fannkuchredux/fannkuchredux.rust-4.rust | Frontend Rules: arrays, arithmetic, while loops | rayon as a reduction (the rule names `par_iter` as a parallel-for); closures, tuples, iterator adaptors, slice methods, mem::replace, Option combinators, parse, println!; the LLVM IR fallback is not served (BIBLE.md :522; OQ-064), so refused |
| Python 3 | fannkuchredux/fannkuchredux.python3-8.python3 | Frontend Rules: loops, int arithmetic, ifs | lists (neither NumPy nor the binding's arrays); the early return; print and `%`; argv; module-level code; Numba is out of P12 |
| Fortran | fannkuchredux/fannkuchredux.ifc | none (low at the pin, BIBLE.md :519; OQ-064 (3)) | character handling and formatted I/O |

Verdict: none fits its subset together with the lassi_io contract. Setting I/O aside, C gcc-8 is closest, C#
csharp-8 next (it needs only array allocation), C++ gpp-3 adds a lambda and STL algorithms, Python 3 #8 needs lists,
Rust has no program without rayon or threads, and Fortran is low at the pin. Upper limits in the programs: gpp-3
refuses n > 16, the Fortran program reads at most two digits, and rust-4's u32 factorials overflow at 13.

Confidence: high for the commit, hashes, and contents; medium for the subset verdicts and the cgeist behavior (read
from source and tests, not run).

## 6. The other benchmarks: fasta, k-nucleotide, reverse-complement, pidigits, regex-redux, binary-trees

Basis: program facts here were first read from the rendered program pages on the live site, through a fetch that
returns a summary. The source check then read each of the 36 members named below in memory from `Z` (2026-10-18,
under `python -I`, nothing extracted), recorded its sha256 (in the tables), and confirmed against the member the
constructs each row reports, by a keyword and line search for the libraries, calls, and types named (lambdas,
tuples, and OpenMP pragmas by their syntax); every row held, and knucleotide/knucleotide.ifc is 183 bytes of
comments. Section 1 confirms that the live site is the commit's public/ tree; as a check, the commit's raw
performance/pidigits.html lists the same 16 rows for the six languages as the live page [S4, S5]. Program counts are
the rows each performance page listed on 2026-10-18 [P1-P6]; member paths below follow section 1's naming rule
(csharpaot reads `.csharp`, ifx reads `.ifc`). cgeist's behavior on these programs is an expectation, not a
measurement; P5.7 measures cgeist's reach. Source tags: S1 to S5, D1 to D7, O1 to O6, P1 to P6, G1 to G36, and Z
(section 10).

All six print text to stdout and none writes arrays [O1-O6]. k-nucleotide, reverse-complement, and regex-redux read
fasta's text output from stdin [D2, D3, D5]; held-out inputs come only from generate_inputs draws (Harness Contract),
so a program written here would read code arrays, not FASTA text.

| Benchmark | C gcc | C++ g++ | C# naot | Rust | Python 3 | Fortran ifx | Output form | Upstream programs depend on |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fasta | 8 (#4 Empty) | 8 | 4 | 8 | 5 | 3 | 3 FASTA records, 60-column lines | strings, stdio, an LCG; threads in C# #1 |
| k-nucleotide | 1 | 3 | 5 | 8 | 5 | 1 (fails to link) | frequency lines (3 decimals), then count-tab-key lines | hash maps (required), threads, sorting, number formatting |
| reverse-complement | 6 | 5 (#3, #6 Bad Output) | 6 (#9 Bad Output) | 3 | 6 | 2 | FASTA records | growing buffered I/O, a lookup table; threads in C #2 and Rust #2 |
| pidigits | 2 | 2 | 4 | 4 | 3 | 1 | 10 digits, a tab, a colon, a running count | arbitrary-precision integers |
| regex-redux | 4 | 6 (#1, #2, #4, #5 Empty) | 4 | 7 | 2 | 1 | pattern-space-count lines, then 3 lengths | regex engines, strings, threads |
| binary-trees | 4 | 8 | 5 (#6 Empty) | 5 | 5 | 2 | phrase-tab-check lines | pointer-linked nodes, recursion, per-node allocation or library pools, threads |

### fasta

Output: three FASTA records, ONE (uppercase ACGT copied cyclically from a fixed sequence), TWO (lowercase acgt with
uppercase IUB codes), THREE (lowercase acgt), in 60-column lines; at N = 1000 about 34, 50, and 84 lines, which fits
section lengths 2N, 3N, and 5N (an inference from the line counts) [O1]. Rules [D1]: N from the command line; one
section copies a fixed sequence cyclically, two pick symbols by weight from cumulative probabilities drawn with a
naive LCG (modulus 139968, multiplier 3877, increment 29573, seed 42); each pick takes a new draw (no caching,
scaled lookup, or optimized LCG); linear or binary search.

| Language | File (page; member) | sha256 | Fits which subsets | What falls outside |
| --- | --- | --- | --- | --- |
| C | fasta-gcc-1; fasta/fasta.gcc | 628b37687acb8ae008116ed7dbbaf001308ee46de5367c60eb4743301de596ff | none as written | (char, double) struct arrays, static RNG state, malloc, printf and puts; stdio and struct arrays expected to leave cgeist's structured dialects [G1] |
| C++ | fasta-gpp-1; fasta/fasta.gpp | 56a0ea5221ad0a75efe9819f2a9d31e6d30bef69e1320b522fe82b1d724e4dca | none | std::vector, function objects with operator(), a function template, lambdas in std::partial_sum and std::count_if [G2] |
| C# | fasta-csharpaot-1; fasta/fasta.csharp | 721adde490094329dcf0c9aff71e7ac5b9ab82a76b3cfb4af8518e5e3475484c | none | multi-threaded: a Task.Run producer, BlockingCollection, a generic iterator over Func delegates, LINQ, MemoryStream [G3] |
| Rust | fasta-rust-1; fasta/fasta.rust | 560295dd3349aebb3ece8c747fb4656178362ca7a2b0375cb3024779eda135a4 | none | structs, a custom Iterator, code generic over io::Write, closures, a (u32, u8) lookup table (so it appears to compare integer state against integer thresholds where C #1 compares doubles) [G4] |
| Python 3 | fasta-python3-1; fasta/fasta.python3 | f8bc35bb3b1c35fed1db20010aaa2abbdcce524e99822ae4a1238c2014e89899 | none | a generator, bisect, strings, print [G5] |
| Fortran | fasta-ifx-1; fasta/fasta.ifc | 297593b87e9a063a3639e6fcab8b6eb24d01c61557d9cde90aae3889b5e80124 | none (low at the pin) | formatted writes; a linear scan [G6] |

A program written here can be an item: the LCG is integer arithmetic whose largest intermediate, 139967 x 3877 +
29573 = 542681632, fits in i32, so Python's ints agree with the typed subsets; the weighted pick can be a counted
loop (the number of cumulative bounds at or below the draw, clamped to the last index; no break, no while); the
repeat section is modular indexing; each section writes a u8 array of ASCII codes; one LCG state runs through TWO
and THREE, so the program is sequential and exercises loop-carried state (scf.for with iter_args), no parallel-for.

### k-nucleotide

Output: 4 single-symbol and 16 two-symbol lines (key, space, percentage to three decimals, by descending frequency
then ascending key), then 5 count lines for the named 3-, 4-, 6-, 12-, and 18-mers [O2, D2]. Rules: a hash table of
k-mer counts, built in or from a library (khash and CK_HT named for C), updated per reading frame, starting small and
growing; no custom hash table; 2-bit key packing allowed [D2].

| Language | File (page; member) | sha256 | Fits which subsets | What falls outside |
| --- | --- | --- | --- | --- |
| C | knucleotide-gcc-1; knucleotide/knucleotide.gcc | 25e2d15a0c436f814254ede1bc87342428f28636f6168634db88b63243a15105 | none | khash (a third-party header via -IInclude), OpenMP sections, fgets, a realloc-grown buffer, qsort, snprintf [G7] |
| C++ | knucleotide-gpp-1; knucleotide/knucleotide.gpp | f681de4594a73681f9674d40ff10192a40ac6a13a316996ae821861cb9d62531 | none | __gnu_pbds cc_hash_table (a GNU extension), std::async, -lpthread, iostream formatting [G8] |
| C# | knucleotide-csharpaot-1; knucleotide/knucleotide.csharp | 0bbc696804904edae8c11d1957bf0152947b89d795a06411d9da19694e57daad | none | DictionarySlim (Microsoft.Collections.Extensions), Task.Run, Parallel.ForEach, LINQ, StringBuilder [G9] |
| Rust | knucleotide-rust-1; knucleotide/knucleotide.rust | 6e36ebd098878896465b1714284c51751976ab2d0a226c8af1b28c6f921d0f64 | none | std HashMap, std::thread::spawn sharing through Arc [G10] |
| Python 3 | knucleotide-python3-1; knucleotide/knucleotide.python3 | e8da061e093461b7c2f27d488924e40e89fe66ec2d58996d490c66e6712e4fc4 | none | a dict (one process) [G11] |
| Fortran | knucleotide-ifx-1; knucleotide/knucleotide.ifc | 1f0b5d781d2059351e97b175047962a59a1e042b8666cf97d06b4fec29ebc51c | none | fails to link (undefined MAIN__); the page notes no Fortran library hash table was found [G12, P2] |

Not an item: a program written here with dense 1- and 2-mer histograms and 2-bit packed comparisons for the five
named k-mers fits the subsets but removes the hash table the description requires and forbids replacing; a dense
18-mer table would need 4^18 (about 6.9e10) counters.

### reverse-complement

Output: FASTA records (three in the check file), uppercase, 60-column lines [O3]. Rules [D3]: read stdin line by line
with buffered reads, growing the data one read at a time; reverse and complement each sequence through the
description's IUPAC table; write id, description, and sequence.

| Language | File (page; member) | sha256 | Fits which subsets | What falls outside |
| --- | --- | --- | --- | --- |
| C | revcomp-gcc-2; revcomp/revcomp.gcc-2.gcc | 71c5bc72d1478e869b80631e450a2d828ecd3f2f0303e9240a74a64f953ee072 | none | pthreads, read() into a realloc-doubled buffer, a 128-entry table, an in-place two-pointer swap, one write() [G13] |
| C++ | revcomp-gpp-1; revcomp/revcomp.gpp | ca1cfd89c0cb141e50e8d734d7f2b9d96f08432f334edb714c6e198278464b91 | none | single-threaded; getline, a 25-entry table, reverse iterators with std::transform, substr chunks [G14] |
| C# | revcomp-csharpaot-1; revcomp/revcomp.csharp | 0ec7b01c68c6dea4754dacfa6690b911eb9f01c0ee8d31030604261d05a7c59a | none | single-threaded; BinaryReader blocks in a List<byte[]>, a 256-entry table, a Stream extension method [G15] |
| Rust | revcomp-rust-2; revcomp/revcomp.rust-2.rust | a000f1ff5b10f92c9ac7ff49081888f87807634a96529dfb2834c52ba15e4a29 | none | rayon::join, read_until, a 256-entry table, a swap from both ends [G16] |
| Python 3 | revcomp-python3-1; revcomp/revcomp.python3 | 0750e816ae6271b5e188b3af5c46e34858c344e0a741bdfc40ffeb5f43fecb88 | none | bytes.maketrans and translate, slice reversal [G17] |
| Fortran | revcomp-ifx-1; revcomp/revcomp.ifc | a5f4d7fdde900887100ec01043a064668112888ab52fafb4262606eace8e376a | none (low at the pin) | stream I/O, an integer(1) buffer grown with move_alloc [G18] |

A program written here keeps only out[i] = comp[in[n-1-i]] over a code array: a gather every subset holds, a fit for
parallel-for, exact u8 output; but it drops the I/O rules that define the benchmark and is only a few lines.

### pidigits

Output: lines of 10 digits, a tab, a colon, and the running count [O4]; the C++, C#, and Rust programs read pad a
final partial line with spaces [G20, G21, G22]. Rules: the first N digits of pi by the step-by-step spigot of pages 4,
6, and 7 of Gibbons' "Unbounded Spigot Algorithms for the Digits of Pi", with arbitrary-precision arithmetic; the
page calls the library choice contentious; N = 30 checks, N = 10000 measures [D4].

| Language | File (page; member) | sha256 | Fits which subsets | What falls outside |
| --- | --- | --- | --- | --- |
| C | pidigits-gcc-1; pidigits/pidigits.gcc | f719dddd30aad201aa3ce385b80d48ef3befce29b22c3a12a5aa7e7d743f948e | none | GMP mpz, -lgmp: opaque external calls on library structs [G19] |
| C++ | pidigits-gpp-4; pidigits/pidigits.gpp-4.gpp | e9ea172b053a24a2fc804a0232af6b242b1fbfedc143500a66355446bb251d98 | none | gmpxx mpz_class, -lgmp -lgmpxx [G20] |
| C# | pidigits-csharpaot-7; pidigits/pidigits.csharp-7.csharp | f09491d004bdcf6c211e05831225b36644095f0fb05617a668a2628168e1dc57 | none | System.Numerics.BigInteger through a lazy iterator [G21] |
| Rust | pidigits-rust-4; pidigits/pidigits.rust-4.rust | fb6b33b1bf74612f8ce78d19e70f10f41650b02b72e4c73d2cb0480508fbef65 | none | the rug crate over GMP, integer feature only [G22] |
| Python 3 | pidigits-python3-4; pidigits/pidigits.python3-4.python3 | cdfff9f6ad7154f1b7020ff7fae2027653cd113a4913074431a7e9af7f800795 | none | built-in int as bignum [G23] |
| Fortran | pidigits-ifx-4; pidigits/pidigits.ifc-4.ifc | 44c96d2d0200d69fc6fdad556fbf0191791bed9948e6033648a439c5ba1701c2 | none | GMP through an ISO_C_BINDING module [G24] |

All single-threaded [G19-G24]. Not an item: no subset or hub dialect has arbitrary-precision integers; a hand-written
limb-array bignum in five languages, with operands that grow every step, needs while loops and runtime growth the
subsets may not hold, keeps no upstream program's method, and could vary only N.

### regex-redux

Output: nine lines of a pattern, a space, and its count; a blank line; the input length, the length after headers
and newlines are stripped, and the length after the substitutions [O5, D5]. Rules [D5]: read all of stdin; one
match-replace removes headers and newlines; count matches of nine alternations (an 8-mer and its reverse complement
with one character class); apply five substitutions in order; use the language's regex, built in or from a library.

| Language | File (page; member) | sha256 | Fits which subsets | What falls outside |
| --- | --- | --- | --- | --- |
| C | regexredux-gcc-2; regexredux/regexredux.gcc-2.gcc | eac8ab538739b2344ab1253b8b0f7c04525b0af135281c0f39d6d67568997562 | none | PCRE (pcre.h, -lpcre) with JIT through pcre_study; fread into a realloc-grown buffer [G25] |
| C++ | regexredux-gpp-3; regexredux/regexredux.gpp-3.gpp | 5a7096dbafe2cba77b27212b94f07b55396a06012c6498c0a8a9b945945dac51 | none | Boost.Regex (-lboost_regex), OpenMP sections and a parallel for [G26] |
| C# | regexredux-csharpaot-1; regexredux/regexredux.csharp | 00b33bd284452bad4c43ee47d689713c1940dc8bbfb328327acf96878cd99e6a | none | Regex with RegexOptions.Compiled [G27] |
| Rust | regexredux-rust-1; regexredux/regexredux.rust | 9e826f38b182de0bff08d52f56277054cd0b3141e3e0fa538f5b42b314bf2a43 | none | the regex crate (bytes::Regex), a thread per pattern [G28] |
| Python 3 | regexredux-python3-1; regexredux/regexredux.python3 | 955633de90f6d4d45d81aec246ea2c62ef02d9d3bf87c9115017e494b2b0b86b | none | re with multiprocessing.Pool [G29] |
| Fortran | regexredux-ifx-1; regexredux/regexredux.ifc | 99607f260506bd283bef05f7e6f76f90dc0bb1bbbc8f37b7bc24b5c5502abcf6 | none | PCRE2 through C interop, JIT, OpenMP [G30] |

Not an item: regex engines, strings, and length-changing substitutions are outside every subset and the hub; hand-coded
matchers for fixed patterns would be a different algorithm from the one the description requires.

### binary-trees

Output: a stretch-tree line, one line per depth group (tree count, depth, check), and a long-lived-tree line, each
with a tab before an integer "check:" field; N = 10 checks, N = 21 measures [O6, D6]. Rules [D6]: build perfect
binary trees bottom-up with at least as many allocations as the reference C program, leaves allocated like interior
nodes; the default GC where there is one, else per-node allocation or a library pool; no custom arenas, pools, or
free lists; a node type with its own functions.

| Language | File (page; member) | sha256 | Fits which subsets | What falls outside |
| --- | --- | --- | --- | --- |
| C | binarytrees-gcc-1; binarytrees/binarytrees.gcc | 730e4c677c1913d48d81535890070c6ca04b9768fc5edccb7600d1d251ea15ea | none (at best low: pointer structs expected to raise to llvm-dialect pointers, not memref) | malloc and free of a two-pointer struct, recursion [G31] |
| C++ | binarytrees-gpp-1; binarytrees/binarytrees.gpp | d20a2fcc8fe13af6f438e56bc2ebfb550a893ac4ef30a9ae7f1dd7d8a5be6c9f | none | boost::object_pool, std::async [G32] |
| C# | binarytrees-csharpaot-1; binarytrees/binarytrees.csharp | 756d56532a8ba4f64281743a9aa5ce68a6a1d25962e722c80c334cab62232fca | none | a class node, recursion, Task.Run [G33] |
| Rust | binarytrees-rust-1; binarytrees/binarytrees.rust | 2bb471318d836aa874fecbcf2dd3da00c6bedee70ec20c141f9ca02bc25abe07 | none | typed_arena with Option references, std::thread::spawn [G34] |
| Python 3 | binarytrees-python3-1; binarytrees/binarytrees.python3 | 4eba41e47c256a732446527a2889eaf7e88bb959ce250905eb5d98fa48fb6b20 | none | nested tuples, recursion, multiprocessing.Pool [G35] |
| Fortran | binarytrees-ifx-1; binarytrees/binarytrees.ifc | 724ee1831f64df5ed3db49b275caafa7af2b2bb2f52bfdaa7a409f8608310dcc | none | a derived type with pointer components, allocate and deallocate, recursion [G36] |

Not an item: an array-encoded tree breaks the allocation rules, and since every tree is perfect and the check counts
nodes [G31], each check value depends on depth alone (an inference), so held-out inputs could vary only the depth
and the outputs would not test the translated computation.

Confidence for this section: high on output forms and rules (description pages and raw output files at the commit);
high on the libraries, calls, and types each row names (confirmed against the zip members, with their sha256);
medium-high on structural readings such as recursion depth and loop forms, which the keyword and line search
checked only by their syntax; medium on cgeist's expected tags
(unmeasured until P5.7) and on fasta's fit (it depends on the subsets P12.4, P12.8, and P12.9 publish: counted
reductions, integer modulo, and a runtime-sized output if `n` is held out).

## 7. Where the areas agree, overlap, or disagree

- Agree: the commit (40296663), the zip's size and sha256, the BSD license, that the C, C++, C#, Rust, Python 3, and
  Fortran sources exist at the commit only as zip members and rendered pages, and that no upstream program writes a
  lassi_io array. The per-benchmark program counts of sections 2 to 6 match section 1's table; their extra site pages
  without members (nbody gpp-8 and rust-8, spectralnorm gcc-1, gpp-8, and csharp #1, mandelbrot gpp-8 and Fortran
  #8) are among the stale pages section 1 lists.
- Disagree, rayon: section 6 lists rayon among the libraries outside P12.1's proposed pins. P12.1's spike proposes
  rayon 1.12.0 with its dependency crates vendored offline (plans/spikes/p12-frontend-pins.md, Summary; section 4
  cites its lines 258-259). The pins spike holds; the rest of section 6's list (GMP, gmpxx, rug, PCRE, PCRE2,
  Boost.Regex, Boost.Pool, khash, __gnu_pbds, the regex crate, typed_arena, Microsoft.Collections.Extensions) stands,
  as do the three non-vendored crates of mandelbrot's rust-5 (num_traits, generic_array, numeric_array).
- Differ, scalar inputs: n-body and fannkuch-redux propose rank-0 inputs (shape []), and spectral-norm a rank-0
  `norm` output; mandelbrot and fasta propose one-element arrays (shape [1]), so that every binding indexes them like
  any other array and no subset needs a scalar case. Both are legal (generate_inputs accepts rank 0). P12.11 picks
  one convention; P12.3 and P12.6's bindings must serve the one chosen (n-body's section asks for ranks 0 and 2).
- Differ, float tolerance policy: n-body proposes max_abs 1e-8 (PROJECTED and a choice: the description's ndiff
  check covers only its two printed energies at N = 1000), with max_abs 0 as the strict alternative if P12.11
  measures bitwise agreement; spectral-norm proposes max_abs 0 first, with an ulp 65536
  fallback only for a measured, traced, permitted reordering. Both expect bitwise agreement under a shared operation
  order (PROJECTED). The difference is a choice of default, not of fact; P12.11's pairwise measurement settles it.
- Differ, FMA contraction: mandelbrot proposes `-ffp-contract=off` for the item's C and C++ source builds so a later
  `-march` change cannot fuse; n-body and spectral-norm rely on gcc.pin's FLAGS (`-O3 -fopenmp`, no `-march`) leaving
  FMA off on baseline x86-64 (GCC fuses only on FMA targets; FMA starts at haswell, not ivybridge). The same flag
  would serve all three float-bearing items; whether an item may carry its own compiler flags is P12.11's question,
  and a gcc.pin question to the owner if not. None of the areas read whether RyuJIT, rustc, or the CPU target
  contracts multiply-adds.
- Overlap, the harness route: n-body notes that lassi_io.h's refusal path calls `fprintf`
  (assets/harness/c/lassi_io.h:111, :116), which becomes `llvm.call` by CGCall.cc:1666-1688, so every C and C++
  program that includes the header gains llvm ops unless P5.7's route keeps the header's functions external;
  spectral-norm, mandelbrot, and fannkuch-redux name the same P5.7 route as the remaining cgeist risk. It applies to
  every item.
- Open reading, description constants: the n-body item avoids upstream data on purpose (random states replace the
  planet constants). fasta's proposed programs would carry the description's two tables and repeat sequence as
  constants. Whether constants taken from a description page count as upstream text under OQ-018 is a reading the
  areas did not settle; it is the owner's if fasta is taken.
- Basis: section 6's program facts first came from a read of rendered pages; the source check then
  read its 36 members from the zip, recorded their sha256, and confirmed each row's constructs against them. Every
  area now names its members with sha256 from `Z`.

## 8. Proposed item map

Every option shares these facts: P12.11 picks each seed, never shown in a prompt; nothing here enters a manifest; the
splits are OQ-057's separate question, and no area found evidence that bears on them beyond the Agent Rule 5 reading
under (c) below.

### Items (a): programs written here from the descriptions

Common to the four items: languages c, cpp, csharp, rust, python (the xlang-v0 row); one program per language, tracked
under assets/bench/xlang-v0/<item>/<lang>/, written from the description, plus this spike's own-words reading of the
upstream algorithm where the description defers to a program, carrying no upstream text. That reading applies to three
items. n-body: its page gives no bodies and defers the integrator to "the Java program" (description/nbody.html;
section 2), so the momentum offset, kick, drift, and energy steps below come from reading the programs. spectral-norm:
its page gives the matrix entries and the four functions; the power iteration (10 AtA passes from a start vector, then
the norm) comes from the programs. mandelbrot: its page gives the region, the bitmap size, and PBM; the iteration
limit and the guarded escape loop come from the programs (gcc-8:14-15, 26, 29). fannkuch-redux's description is
complete (flip rule, checksum, permutation order, and index-to-permutation rule). OQ-057's option (a) is worded "from
each benchmark's published description", so answering (a) accepts this reading under it. Each main reads its inputs as
`@inputs/<name>.lassiio` in manifest order through its binding, writes its outputs through it in order, and prints
nothing (Held-out inputs paragraph, BIBLE.md:618). Every language performs the same operations in the same order. Per
language: C and C++ (C++17, C-like: no classes, templates, containers, iostreams, lambdas, or exceptions) over
pointers from `lassi_io_array` with lassi_io.h; C# a static class of static methods over arrays with the C# binding;
Rust functions over slices with the binding crate (rayon vendored offline, P12.1); Python module functions over the
binding's array type (P12.3), with prange from the binding where a parallel-for is proposed. Fortran: under OQ-057
Fortran (a), P12.10 writes its own programs outside the gate; under Fortran (b), each item gains a Fortran program,
which reaches only low at the pin until OQ-064 is answered, so the strict reading fails for Fortran.

#### n-body

- Structure: read the six inputs; N = P + 1 with P from planet_mass's dimension (never a literal); flat arrays x[N*3],
  v[N*3], m[N], body 0 the central body at the origin at rest. Momentum offset: per axis p = sum of m_k * v_k over
  ascending k, then v_0 = -p / m_0. Energy: from 0.0, for ascending i add `0.5 * m_i * ((vx*vx + vy*vy) + vz*vz)`,
  then for each j > i ascending subtract `(m_i * m_j) / sqrt(d2)`. Repeat `steps` times: kick over pairs i < j (i
  outer, j inner, ascending): d = x_i - x_j per axis, `d2 = (dx*dx + dy*dy) + dz*dz`, `mag = dt / (d2 * sqrt(d2))`,
  `v_i -= d * (m_j * mag)` and `v_j += d * (m_i * mag)` per axis; then drift `x += dt * v` per body and axis. Compute
  e1 the same way; write energy = [e0, e1], pos_final, vel_final.
- Constructs: counted loops with bounds from input dimensions; f64 and i32 scalars; 1-D f64 arrays indexed i*3 + k;
  `+ - * /`, sqrt, the binding's calls; nothing else. Per language: C and C++ `sqrt` from math.h; C# `Math.Sqrt`;
  Rust `for i in 0..n`, `as usize` for steps, `f64::sqrt`; Python `for i in range(n)`, `math.sqrt`.
- Parallel-for: the drift loop writes only its own body, so prange, Parallel.For, rayon par_iter, or OpenMP could
  carry it without changing values; the kick loop cannot (iteration i writes body j for every j > i), nor any sum.
  Proposed: all sequential in v0; P12.11 may add the parallel drift loop once the frontends publish their forms.
- Inputs (generate_inputs entries):

  | name | dtype | shape | dist | lo | hi | reason |
  | --- | --- | --- | --- | --- | --- | --- |
  | central_mass | f64 | [] | uniform | 30.0 | 50.0 | brackets 4*pi^2 (about 39.48), the central mass in AU-and-year units with G = 1 |
  | planet_mass | f64 | [8] | uniform | 0.0004 | 0.04 | about 1e-5 to 1e-3 of the central mass; Jupiter's is about 1e-3 by the programs' constants |
  | planet_pos | f64 | [8, 3] | uniform | -30.0 | 30.0 | AU; the upstream planets reach about 26 AU |
  | planet_vel | f64 | [8, 3] | uniform | -2.0 | 2.0 | AU per year; Jupiter's speed is about 2.8 by the programs' constants |
  | steps | i32 | [] | integers | 2000 | 6000 | neither the site's 1000 nor 50,000,000 |
  | dt | f64 | [] | uniform | 0.005 | 0.02 | not fixed at 0.01 |

- Outputs: `energy` f64 [2], `pos_final` f64 [9, 3], `vel_final` f64 [9, 3].
- Why: the site publishes the N = 1000 output, so a program could print it without computing; N = 50,000,000 took
  Python #8 529.378 s on the site's machine, too long for a gate that runs each program twice. PROJECTED cost at the
  site's Python rate (about 9.4e5 pair updates per second, another machine): at most 6000 x 36 = 216,000 pair
  updates, about 0.23 s, well inside the 30 s attempt-run floor (RUN_WALL_FLOOR_S,
  lassi/core/stages.py:393; P12.12 sets the gate's limit). A random state replaces the planet constants, so no
  program carries upstream data, and a program that hardcodes five bodies, the central mass, 0.01, or the published
  energies fails. The largest input has 24 elements, far under the 2^20 cap.
- Close passes: the integrator has no softening and a fixed dt, and these draws let a planet pass close to the
  central body or to another planet, where one-ulp differences grow fast; the initial separations alone do not show
  it. Proposed: for each drawn seed, P12.11 computes the minimum pair separation over the whole run (every pair i < j
  after every step, in its reference build or a Python model of the same steps) and refuses the seed if that minimum
  falls below 3.0 AU (PLACEHOLDER), drawing the next; the manifest comment records the minimum and the number of
  seeds refused. At 3.0 AU from a central mass below 50, the pair's dynamical time sqrt(r^3 / M) is about 0.73 years,
  so a step below 0.02 stays under about 3% of it (derived). P12.11 fixes the bound by measuring how far a one-ulp
  change to the drawn state moves the outputs for seeds near it; the refusal rate is unmeasured, and if most seeds
  are refused, the position draw changes (P12.11's proposal). This spike proposes refusal, not a narrower velocity
  draw: generate_inputs draws each element uniformly, so no velocity range makes orbits near circular, and a range
  nearer zero lowers each planet's angular momentum and brings it closer to the central body, not farther (derived).
- Tolerance: `{metric: max_abs, threshold: 1.0e-8}` on energy, pos_final, and vel_final, PROJECTED and a choice. The
  description's `ndiff -abserr 1.0e-8` checks only the two 9-decimal energies of its fixed Jovian system at N = 1000,
  "to check your program output has the correct format" [D]; nothing upstream checks positions, velocities, or a
  drawn state, so the threshold here is this spike's choice, not the description's check. A shared order with
  correctly rounded sqrt is expected to agree bit for bit (PROJECTED; P12.11's pair test measures it); the margin is
  meant to cover what a toolchain may change (GCC contracts multiply-adds by default on targets with native FMA, [G];
  gcc.pin's FLAGS leave that off on baseline x86-64, but a changed flag or the CPU target's flags, P5.7, could turn it
  on), and whether 1e-8 covers that rests on the separation bound above, since a close pass amplifies one-ulp
  differences; P12.11 measures the margin (a one-ulp perturbation, or a contracted build) before it fixes the
  threshold. Not ulp (a coordinate near zero gives a huge ulp distance for a tiny absolute one) and not pcc (blind to
  an offset on a two-element array). If P12.11 measures bitwise agreement in every pair and round trip, max_abs 0 is
  the strict alternative.

#### spectral-norm

- Structure: read `u0` (f64, rank 1, length n); allocate u (a copy of u0), v, and scratch t, each f64[n]. Four
  functions as the description asks: the denominator d(i,j) = (i+j)(i+j+1)/2 + i + 1 in 64-bit integers (a Python
  int), exact; A times x into y; At times x into y using d(j,i); AtA times x into y through t. A times x: per row i,
  one accumulator from 0.0, for j from 0 to n-1 in order add x[j] / double(d(i,j)), then y[i] takes it; the row loop
  is a parallel-for where the subset names one, and each row's sum stays inside one iteration so it cannot be
  reordered. Run 10 passes of v = AtA u, then u = AtA v (fixed, not an input). vBv and vv as sequential sums over
  ascending i; norm = sqrt(vBv / vv). Write u and v (f64 [n]) and norm (f64, rank 0); exit 0.
- Per language: C over `double *` with `int64_t` indices, scratch from malloc (a typed malloc becomes `memref.alloc`,
  clang-mlir.cc:3958-3984), no VLA, no printf; C++ the same as C++17 (the C and C++ programs will differ very little,
  which the item should record); C# with Parallel.For and a lambda body, long denominators, Math.Sqrt; Rust over
  `&[f64]` and `&mut [f64]`, `vec![0.0; n]`, the row loop as `par_iter_mut().enumerate().for_each` (the only
  closure), u64 denominators, `f64::sqrt`; Python with prange for the row loop, an inner `for j in range(n)` with a
  float accumulator, int denominators, math.sqrt, and no `np.dot`, `np.sum`, `@` (NumPy's add.reduce uses partial
  pairwise summation, [N1]) or builtin sum() ([P1]).
- Subset gap: the Frontend Rules line names neither function definitions and calls with array arguments nor array
  allocation (malloc, `new double[n]`, `vec!`, a NumPy constructor). The four functions need the first and the
  scratch arrays the second; P12.4, P12.8, and P12.9 should publish both, or the item inlines the four functions and
  departs from the description.
- Inputs: `[{name: u0, dtype: f64, shape: [521], dist: uniform, lo: 0.5, hi: 1.5}]`; outputs `[u, v, norm]`.
- Why: 521 is not a published size or default (100, 5500, 1000, 2000, 3000); it is odd and prime, so no split into 2,
  4, or 8 lanes or rayon pairs comes out even, and a program that assumes even n (as upstream Rust #4 to #6 do) or a
  multiple of 4 (as Rust #7 does, rounding n up) fails; it is far inside N <= 23171 and the 2^20 cap. PROJECTED
  CPython cost: 40 n^2 = 10,857,640 inner steps at the site's Python #8 rate (about 0.3 us each, from about 6 min for
  40 x 5500^2 = 1.21e9) is about 3 s, possibly several times more if NumPy arrays are indexed element by element; the
  limit is the 30 s run floor, and n = 257 costs a quarter as much if P12.11 measures a run near it. u0 is random
  rather than all ones because the norm barely depends on the start after 20 products, so u and v show whether a
  program read its input; the positive bounds keep every term positive (no cancellation), which the fallback bound
  relies on.
- Tolerance: `{metric: max_abs, threshold: 0}`. Every build performs the same binary64 operations in the same order:
  exact integer denominators below 2^53, one correctly rounded division plus one addition per term (no multiply-add
  to contract), sequential sums, correctly rounded sqrt; so the five builds and their round trips should agree bit
  for bit, and a difference marks a build that reordered or contracted, which the gate should report. Conditions: no
  FMA contraction in the two dot products (gcc.pin has no -march; the CPU target's flags are P5.7's), and no lowering
  that turns a sequential reduction into a parallel or vector one (not read; P12.11's pairwise check measures it).
  Fallback, if a nonzero difference is measured and traced to a permitted construct: `{metric: ulp, threshold:
  65536}` (PROJECTED, derived, never from_baseline): with all terms positive, a sum of n terms in any order has
  relative error at most about n*u (u = 2^-53); 40 products in sequence add, so two orders differ by at most about 2
  x 40 x n x u = 4.6e-12 relative at n = 521, at most about 42,000 ulps, rounded up to 2^16. It still catches real
  faults: a dropped AtA product changes values by about 1.62 (the largest eigenvalue of AtA, the square of
  1.274224153), and a swap of A and At fails too. Not pcc (ignores scale; gives 0 for the rank-0 norm unless exact),
  and not a max_abs above 0 (u and v grow by about 1.62^20, or 1.6e4, so any absolute threshold would depend on n and
  the seed).

#### mandelbrot

- Structure: main reads the five inputs and allocates both outputs zeroed (in main only, through the binding or the
  plain array constructor), calls the kernel, and writes both outputs. The kernel takes (size, max_iter, re_min,
  im_min, span, bitmap, iters), arrays flat and row-major. For each row y in [0, size): ci = (span * y) / size +
  im_min (ints to f64 first, then multiply, divide, add). For each byte b in [0, (size + 7) / 8): acc = 0; for each
  bit j in [0, 8): x = 8b + j, inside = 0; if x < size: cr = (span * x) / size + re_min, zr = zi = tr = ti = 0.0,
  count = 0; for k in [0, max_iter): if tr + ti <= 4.0 then zi = (2.0 * zr) * zi + ci, zr = (tr - ti) + cr, tr =
  zr * zr, ti = zi * zi, count += 1; then inside = 1 if tr + ti <= 4.0, else 0, and iters[y * size + x] = count;
  after each bit acc = acc * 2 + inside; after the 8 bits bitmap[y * ((size + 7) / 8) + b] = acc.
- Constructs: counted loops with constant step, if, i32 and binary64 arithmetic, int-to-f64 conversion, array loads
  and stores; never break, while, early return, complex types, structs, formatting, or library math; bounds affine in
  size ((size + 7) floordiv 8 included), so cgeist can reach scf or affine, memref, and arith. The row loop is the
  parallel-for site: Python uses prange from the binding; C# Parallel.For and Rust a rayon form only if P12.9 and
  P12.8 publish one that writes disjoint rows of two arrays (in Rust, par_chunks_mut with zip or similar, beyond the
  bare `par_iter` the rule names), else a counted loop; C and C++ a counted loop.
- Inputs (each a one-element array of shape [1]; section 7 on rank 0):

  | name | dtype | dist | lo | hi | gives |
  | --- | --- | --- | --- | --- | --- |
  | size | i32 | integers | 257 | 258 | 257 |
  | max_iter | i32 | integers | 64 | 65 | 64 |
  | re_min | f64 | uniform | -1.55 | -1.45 | drawn from the seed |
  | im_min | f64 | uniform | -1.05 | -0.95 | drawn from the seed |
  | span | f64 | uniform | 1.9 | 2.1 | drawn from the seed |

- Outputs: `bitmap` u8 [size, (size + 7) / 8], the PBM P4 raster without its text header (which size alone
  determines), most significant bit first, 1 in the set, fill bits 0; `iters` i32 [size, size], the updates each
  pixel took.
- Why: 257 = 32 x 8 + 1 is none of 200, 1,000, or 16,000, and every row ends in a byte with one pixel and seven fill
  bits, which no published size reaches and where upstream programs disagree. size^2 x max_iter is about 4.2M guarded
  steps, a few seconds under CPython (PROJECTED) and far below the 30 s floor natively. max_iter 64 is not the
  programs' 50, so a program carrying the upstream constant fails `iters` and boundary bits; the description states
  no count. The window is drawn near the site's, so no published image matches and the boundary falls on other
  pixels; re stays in [-1.55, 0.65) and im in [-1.05, 1.15), so |c| < 1.94 and the radius-2 test keeps its meaning. A
  pixel stops updating once |z|^2 > 4, so every value stays finite (|z|^2 < 36) and binary_io's NaN and infinity
  rules never apply. Five input elements in all.
- Conformance check for P12.11's tests (not a held-out input): at size 200, max_iter 50, re_min -1.5, im_min -1.0,
  span 2.0, the structure performs gcc-8's operations exactly ((2x)/N + (-1.5) equals (2x)/N - 1.5 in binary64, and
  the guarded updates match gcc-8:29-35), so `bitmap` should equal the 5,000 bytes after [M3]'s 11-byte header,
  fetched by path and sha256 and never tracked (OQ-018). An inference from reading; P12.11's remote test confirms it.
- Tolerance: `{metric: max_abs, threshold: 0}`, which binary_io enforces for integer outputs anyway (BIBLE.md:728).
  Exact is reachable: the same correctly rounded binary64 operations in the same order, exact conversions (size <
  2^53), exact doubling, no libm call. Risk: contracting `(2zr)zi + ci` or the coordinate expression into an FMA, or
  fast-math reassociation, flips boundary pixels; GCC fuses only on FMA targets and gcc.pin has no -march, but the
  item's C and C++ source builds should pass `-ffp-contract=off`, and the CPU target must emit arith ops with no
  fastmath `contract` flag. A flipped pixel is a defect to fix with a flag or in a frontend, not slack; a float copy
  of the bits compared by pcc below 1 would hide the defects the gate exists to find and is not recommended.

#### fannkuch-redux

- Structure: read `n` (i32, rank 0), `block_len` (i64, rank 0), `starts` (i64 [B]); fill a 16-entry i64 factorial
  table. For each block b (a counted loop, serial in v0): build the starting permutation from starts[b] as [D] gives
  it (identity 0..n-1, rem = starts[b]; for k from n-1 down to 1: d = rem / k!, rem = rem mod k!, count[k] = d, rotate
  positions 0..k left by d through a scratch copy); then block_len times (counted): copy the permutation into a work
  array; while work[0] != 0, reverse positions 0..work[0] with a counted loop of swaps and count the flip; add the
  flips to the block's checksum when starts[b] + t is even and subtract when odd; keep the block's maximum; advance (i
  = 1, rotate positions 0..1, count[1] += 1; while count[i] > i: count[i] = 0, i += 1, rotate positions 0..i left by
  one, count[i] += 1). Write `checksum` (i64 [B]) then `maxflips` (i32 [B]).
- Constructs: counted loops and while loops with one comparison as the condition; no break, continue, or return in
  any loop (one exit each, which keeps cgeist's flag memrefs out and eases P12.8's MIR structurizer); if/else; fixed
  16-element local i32 and i64 arrays (`new int[16]` in C#); integer `+ - * / % &` on non-negative operands only, so
  truncating and flooring division agree; the binding's calls; nothing else. C++ is the C program in the C-like
  subset; Rust writes counted loops as while loops with a counter unless P12.8 publishes Range loops, and P12.8 must
  accept MIR's bounds-check assertions (no overflow at these sizes); Python allocates scratch with P12.4's published
  constructor, and every value stays below 2^31. The block loop is the natural later parallel-for: each block writes
  only its own elements, so no reduction construct is needed.
- Fallback if a subset publishes no while loop: the flip loop becomes a guarded counted loop of at most 64 steps
  (the maximum at n = 10 is 38 flips, [OEIS]) and the advance a counted loop over levels with a carry flag; neither
  order nor results change.
- Inputs:

  ```
  inputs:
    - {name: n, dtype: i32, shape: [], dist: integers, lo: 10, hi: 11}
    - {name: block_len, dtype: i64, shape: [], dist: integers, lo: 2048, hi: 2049}
    - {name: starts, dtype: i64, shape: [64], dist: integers, lo: 0, hi: 3626752}
  outputs: [checksum, maxflips]
  ```

- Why: the site publishes N = 7 (228, 16) and N = 12 (3968050, 65), and A000375 lists the maximum for every n up to
  21, so any whole-range item has a known answer; with seeded block starts all 128 output values depend on the seed.
  n = 10 is neither published size, gives 10! = 3,628,800 positions, and has a maximum of 38. n and block_len are
  one-value ranges because run_args cannot sit beside inputs (BIBLE.md:618), and generate_inputs accepts any lo < hi
  and rank 0 (lassi/harness/lassi_io.py:478-493, :212-225). hi = 10! - 2048 (exclusive) keeps every block's last
  advance at index 10! - 1 or below, so no program wraps and no guard is needed. Random start parities fail a
  candidate that signs by block-local position instead of global index in about half the blocks. PROJECTED run time:
  131,072 permutations (3.6% of 10!), under a second to a few seconds for the CPython build (scaled from the site's
  Python 3 #8 at N = 12, 40 min for 479,001,600 permutations, about 5 us each, CPython 3.13.0 with lists; Python 3.10
  and the binding's arrays unmeasured), milliseconds natively; both under the 30 s floor. 64 elements, far below
  MAX_INPUT_ELEMENTS (lassi/bench/registry.py:90).
- Tolerance: `{metric: max_abs, threshold: 0}`: integer outputs, matched exactly whatever the metric (BIBLE.md:728);
  no floating point, so any difference between languages or between a source build and its round trip is a bug.
- Test anchors for P12.11 (published sizes, not held-out inputs): n = 7, block_len = 5040, starts = [0] gives 228 and
  16 [O]; blocks tiling [0, 5040) give checksums summing to 228 and a largest maximum of 16; the order from index 0
  matches [X] (its first seven lines checked by hand against [D]'s rule; nothing run).

#### Optional and excluded items under (a)

- fasta (optional fifth item; the owner decides): inputs `seed` (i32, shape [1], integers in [0, 139968)) and `n`
  (i32, shape [1], integers in [1000, 100001); PLACEHOLDER); the description's two tables and repeat sequence as
  constants (see section 7 on OQ-018), f64 throughout; outputs `one`, `two`, `three` (u8, lengths 2n, 3n, 5n;
  PLACEHOLDER until P12.11 reads the multipliers from the description). Tolerance `{metric: max_abs, threshold: 0}`,
  enforced by binary_io's integer rule anyway; it also catches any reassociation or fast-math in a build or the CPU
  lowering, since either could move a pick across a bound, so no build may use -ffast-math or the like. If a subset
  lacks runtime-sized allocation, `n` becomes a program constant (proposed 100000, PLACEHOLDER) and only `seed` is
  held out.
- reverse-complement: possible as a small integer parallel-for item (input `seq` u8 [n], n at most 1048576, integers
  in [0, K) indexing the description's code alphabet, K PLACEHOLDER until P12.11 counts it; output `rc` u8, same
  shape; LF bytes at every 60th output position; max_abs 0), but not recommended for v0: it drops the I/O rules that
  define the benchmark and is only a few lines. A candidate for a later suite.
- Not items while the subsets stay as the Frontend Rules state them: k-nucleotide (needs the hash table its
  description requires), pidigits (arbitrary precision), regex-redux (regex engines), binary-trees (pointer trees and
  allocation rules; outputs depend on depth alone).

#### Subset needs the (a) items place on P12.3, P12.4, P12.6, P12.8, and P12.9

- Bindings: f64 and i32 arrays of rank 0 (or one-element arrays, section 7) and ranks 1 and 2; i64, u8, and i32
  outputs.
- Subsets: index loops whose bounds come from dimensions; flat f64 arrays and scalar locals; augmented assignment;
  sqrt; i32-to-index and int-to-f64 conversion in mixed expressions; function definitions and calls with array
  arguments; array allocation; while loops with one comparison (or the bounded fallback); fixed-length local i32 and
  i64 arrays; i64 `/ % &`; MIR bounds-check assertions (P12.8); and, for later variants, a parallel-for that writes
  disjoint rows of two arrays (P12.8, P12.9).
- P5.7 and P12.12: how lassi_io calls in a C main are carried through (section 7, the harness route) and how the tag
  rule reads them; the CPU target emits no `contract` fastmath flag.

### Items (b): upstream programs fetched by zip member and sha256

- Pin: one pin covers every program: `Z` at 40296663, path public/download/benchmarksgame-sourcecode.zip, sha256
  aabcf6726cdc14f0f45b99e5daba48584f94bbb48883fd3711a1d040474d1cb4, each program addressed by its member path; P12.11
  records per-member hashes (those read are in sections 2 to 6). Nothing need be tracked (OQ-018); if text ever were
  tracked, the BSD notice goes with it.
- Fetch: tools/fetch_bench.py checks out listed repository paths, and these sources are not tree files at the commit,
  so (b) needs a member-extraction step that checks the zip's sha256 and each member's (a tools change), or a source
  taken from a rendered page (not a source form).
- Oracle and gate: no upstream program writes a lassi_io file, so binary_io cannot judge any of them; each needs
  stdout_mask (an empty mask list, since none prints timing lines) and the narrowed gate reading OQ-057 names (low
  modules allowed, stdout_mask in place of binary_io).
- Held-out inputs: only sizes through run_args (run_args never sit beside held-out inputs, BIBLE.md:618). No item key
  feeds a program's stdin (the item keys, lassi/bench/registry.py:106), and generate_inputs draws arrays, not FASTA
  text, so the stdin benchmarks (k-nucleotide, reverse-complement, regex-redux) have no input route. Every program
  can be run by anyone at any size, so a held-out size hides only the published outputs.
- Tolerance: an exact stdout match under stdout_mask with an empty mask list, for every item below. Per benchmark:

  | Benchmark | Under (b) | run_args size | Tolerance | Basis or reason |
  | --- | --- | --- | --- | --- |
  | n-body | not an item unless the oracle gains a numeric tolerance | none (a step count other than 1000 and 50,000,000 if it does, PLACEHOLDER) | none: the description's abserr 1.0e-8 on the two printed energies has no form in stdout_mask's exact match | its programs differ in expression order (section 2), so exact text equality across languages is not shown |
  | spectral-norm | item | n = 522 | exact stdout | not 100, 5500, 1000, 2000, or 3000; even, as Rust #4 to #6 require, so it suits the proposed member, rust-4; not a multiple of 4, so it does not suit #7, which would print the norm for n = 524; below 23171; the programs agree only at the printed precision (the C++ quirk, Rust's lanes; section 3), which suits stdout_mask, and that they agree at 522 is an inference |
  | mandelbrot | item | N = 512 (or 4096) | exact stdout | a power of two that is no published size (200, 1,000, 16,000): 2/N is exact, so rust-5's x times (2/N) equals (2x)/N in binary64; no row has fill bits, so csharp-2's 1s never appear; rust-5's rounding down to a multiple of 8 (rust-5:57, VLEN 8 at :22; rust-2:59, rust-7:57) does nothing. This removes the two differences section 4 found (derived); that the programs then agree is an inference. 512 costs at most about 13.1M escape-loop steps at the programs' 50 iterations, about three times the (a) item's 4.2M, and 4096 64 times that (PROJECTED) |
  | fannkuch-redux | item, held out in name only | N = 10 | exact stdout | not 7 or 12; inside gpp-3's n <= 16, the Fortran program's two digits, and rust-4's u32 factorials (overflow at 13); A000375 has the maximum |
  | fasta | item | N = 100000 (PLACEHOLDER) | exact stdout | not 1000 or 25,000,000; about 1 MB of stdout, scaled from 10,245 bytes at N = 1000 (PROJECTED); that the languages agree at other sizes is an inference, since Rust #1 appears to compare integer thresholds where C #1 compares doubles (section 6) |
  | k-nucleotide | not an item unless the harness can feed stdin | none | none | reads fasta's output on stdin; its Fortran program fails to link |
  | reverse-complement | not an item unless the harness can feed stdin | none | none | reads FASTA on stdin |
  | regex-redux | not an item unless the harness can feed stdin | none | none | reads FASTA on stdin; needs PCRE, PCRE2, Boost.Regex, and the regex crate |
  | pidigits | item, held out in name only | N = 2000 (PLACEHOLDER) | exact stdout | not 30, 27 (the zip's own check), or 10000; a multiple of 10, so no final partial line, whose padding section 6 read only in the C++, C#, and Rust programs; pi's digits are published; needs GMP, gmpxx, and rug |
  | binary-trees | item, held out in name only | N = 16 (PLACEHOLDER) | exact stdout | not 10 or 21; each check value depends on depth alone (an inference, section 6); needs Boost.Pool and typed_arena |
- Program limits: fannkuch-redux gpp-3 refuses n > 16, its Fortran reads at most two digits, and rust-4's u32
  factorials overflow at 13; spectral-norm's 32-bit programs disagree above N = 23171.
- Expected furthest step at the pins (from reading, not runs): C through cgeist a module with llvm ops beside scf and
  memref (printf, putc, atoi as llvm.call; P5.7's rule tags it), and low or refused for `vector_size`, intrinsics,
  pthreads, iostream, complex, and pointer trees; C++ low or refused (lambdas, templates, iostream, std::complex, an
  LLVM struct for NBodySystem); Rust refused (the nightly's LLVM 23.1.4 IR is not readable at 26eb4285, BIBLE.md:522);
  C# refused (NativeAOT-LLVM gives no Linux LLVM IR, BIBLE.md:523); Python refused (I/O, lists, classes; Numba is out
  of P12); Fortran low through LLVM import once flang is pinned (OQ-063, OQ-064).
- Toolchains: upstream builds use CPython 3.13.0, .NET SDK 9.0.100 with NativeAOT, rustc 1.84.1, and gcc 14.2.0
  -march=ivybridge; LASSI's proposed pins are python 3.10.12 with numpy 2.2.6, .NET SDK 10.0.402, a rustc nightly
  (nightly-2026-10-17), and gcc-12. Libraries the upstream programs need that no pin covers: GMP, gmpxx, rug, PCRE,
  PCRE2, Boost.Regex, Boost.Pool, khash, __gnu_pbds, the regex crate, typed_arena, Microsoft.Collections.Extensions,
  and mandelbrot rust-5's num_traits, generic_array, and numeric_array; each would join an install batch.
- Members proposed per benchmark (the simplest single-threaded program per language, else the simplest):

  | Benchmark | C | C++ | C# | Rust | Python 3 | Fortran |
  | --- | --- | --- | --- | --- | --- | --- |
  | n-body | nbody/nbody.gcc | nbody/nbody.gpp-3.gpp | nbody/nbody.csharp | nbody/nbody.rust | nbody/nbody.python3-8.python3 | nbody/nbody.ifc |
  | spectral-norm | spectralnorm/spectralnorm.gcc-8.gcc | spectralnorm/spectralnorm.gpp | spectralnorm/spectralnorm.csharp-8.csharp | spectralnorm/spectralnorm.rust-4.rust (rayon) | spectralnorm/spectralnorm.python3-8.python3 | spectralnorm/spectralnorm.ifc-2.ifc |
  | mandelbrot | mandelbrot/mandelbrot.gcc-8.gcc | mandelbrot/mandelbrot.gpp-3.gpp | mandelbrot/mandelbrot.csharp-2.csharp | mandelbrot/mandelbrot.rust-5.rust (rayon and three crates) | mandelbrot/mandelbrot.python3-8.python3 | mandelbrot/mandelbrot.ifc-5.ifc |
  | fannkuch-redux | fannkuchredux/fannkuchredux.gcc-8.gcc | fannkuchredux/fannkuchredux.gpp-3.gpp | fannkuchredux/fannkuchredux.csharp-8.csharp | fannkuchredux/fannkuchredux.rust-4.rust (rayon) | fannkuchredux/fannkuchredux.python3-8.python3 | fannkuchredux/fannkuchredux.ifc |
  | fasta, pidigits, binary-trees | the members section 6 names, with their sha256 | as section 6 | as section 6 | as section 6 | as section 6 | as section 6 |

  k-nucleotide, reverse-complement, and regex-redux take members only if the harness gains a stdin route, then the
  ones section 6 names (k-nucleotide's Fortran program fails to link). None of these reaches the strict gate reading;
  the Fortran column applies only under Fortran (b).

### Items (c): (a) as the gate's items now, (b) later as an unpaired low-level corpus

- Now: the (a) items above, unchanged.
- Later (a later phase): upstream members by path and sha256 inside `Z`, with the zip-member fetch step of (b),
  compared by stdout_mask with an empty mask list at (b)'s sizes (spectral-norm 522; mandelbrot a power of two that is
  no published size, such as 512, where agreement is an inference). Besides the members above, mandelbrot's nearest
  raisable ones are mandelbrot/mandelbrot.gcc-9.gcc (3482 bytes, sha256
  0c8bcfcbd20e0328896ef50863d61cb3a109920d6ad3d7f17057c28d23122ca4: OpenMP parallel-for, a malloc'd pixel array,
  fprintf and fwrite, N*N/8 bytes so N a multiple of 8; the only candidate for cgeist's OpenMP row, omp and scf) and
  mandelbrot/mandelbrot.gpp-5.gpp (2385 bytes, sha256
  efdfc5765c85208ad201c2892ae7bbf8a77e7c7574f363364fd10188f2b346b7: a C-like kernel with std::vector, copy, fill, and
  max).
- What it would hold at the pins: at most C, C++, and Fortran, low or partly low (C with printf as llvm.call and
  globals kept only if their initializers fold; C++ with LLVM structs and llvm.call; Fortran through LLVM import if
  flang is pinned). Rust, C#, and Python have no route. Low modules stay syntax data until raised to structured
  (Frontend Rules). A text suite judged by stdout_mask is not what the xlang-v0 row names; holding one there would be
  a bible edit.
- Agent Rule 5: while a benchmark is an xlang-v0 eval or unassigned item, harvesting its upstream programs would be
  corpus harvest on the same task, so the later corpus should leave out every such benchmark unless the owner rules
  otherwise, and any harvest needs a train split the owner assigns.

## 9. Response for OQ-057

OQ-057 carries this text as one line, after its Recommendation line.

> Response (task P12.2, 2026-10-18): The Benchmarks Game's source is
> salsa.debian.org/benchmarksgame-team/benchmarksgame at commit 40296663 (2025-03-07, still its newest on 2026-10-18),
> where every program sits in one zip, public/download/benchmarksgame-sourcecode.zip (sha256 aabcf672...1cb4), under a
> 3-clause BSD license (the site's "Revised BSD license") in each benchmark directory, with no explicit grant from
> contributors. All 10 benchmarks have programs in C, C++, C#, Rust, Python 3, and Fortran (Intel ifx; k-nucleotide's
> one Fortran program fails to link), but each takes N from argv or FASTA from stdin and prints text, so none reads
> held-out inputs or writes a lassi_io array and binary_io cannot judge any as written; and none of the simplest
> single-threaded programs read for n-body, spectral-norm, mandelbrot, and fannkuch-redux fits its frontend's subset
> (no Rust program for the last three is single-threaded; C's numeric cores come closest under cgeist; Fortran reaches
> only low at the pin). Under items (a), or (c), which is (a) for now, four items hold as programs written here from
> the description, plus this spike's own-words reading of the upstream algorithm where the description defers to a
> program (n-body's integrator and energy, spectral-norm's power iteration, mandelbrot's iteration limit and escape
> test; fannkuch-redux's description is complete), a reading that answering (a) accepts under its words "from each
> benchmark's published description". Each uses one operation order over flat arrays and counted loops and fits the
> published subsets once P12.4, P12.8, and P12.9 publish function calls with array arguments, array allocation, and
> while loops (or a bounded fallback), and the C and C++ programs reach structured only if P5.7's lassi_io.h route
> keeps the header's stdio calls out of the raised module. The items: n-body (a seeded central body and 8 planets,
> steps in [2000, 6000), dt in [0.005, 0.02), refusing seeds whose minimum pair separation over the run falls below a
> stated bound; outputs energy, pos_final, vel_final; max_abs 1e-8, PROJECTED and a choice, since the description's
> ndiff check covers only the two printed energies of its fixed system), spectral-norm (u0 f64[521]; outputs u, v,
> norm; max_abs 0, with an ulp 65536 fallback only for a measured, permitted reordering), mandelbrot (size 257,
> max_iter 64, a seeded window; outputs bitmap u8 and iters i32; exact, with -ffp-contract=off for C and C++ and a
> conformance check against the site's N = 200 image), and fannkuch-redux (n 10, block_len 2048, 64 seeded block
> starts; outputs checksum and maxflips; exact). fasta can be an optional fifth integer item; reverse-complement is
> possible but not recommended for v0; k-nucleotide, pidigits, regex-redux, and binary-trees are not items, since
> their descriptions require hash tables, arbitrary precision, regex engines, or allocated pointer trees. Under (b),
> the gate reading would need narrowing (exact stdout under stdout_mask in place of binary_io, low modules allowed,
> held-out inputs only as run_args sizes, a zip-member fetch in tools/fetch_bench.py, and libraries such as GMP and
> PCRE in install batches); then spectral-norm (n 522), mandelbrot (a power of two such as 512, which removes its
> programs' coordinate and fill-bit differences; agreement there is an inference), fannkuch-redux (10), fasta,
> pidigits, and binary-trees could be items, the last two and fannkuch-redux held out in name only, while n-body needs
> a numeric tolerance stdout_mask lacks and k-nucleotide, reverse-complement, and regex-redux a stdin route the
> harness lacks; even so, at the pins only C and C++ reach a hub module (low or partly low), Fortran low once flang is
> pinned, with Rust, C#, and Python refused. Under (c), the later corpus would hold at most C, C++, and Fortran, low
> or partly low, and while a benchmark is an eval or unassigned item its upstream programs stay out of any harvest
> (Agent Rule 5). No Fortran program reaches structured at 26eb4285, so the evidence supports Fortran (a). The
> recommendation stands: items (a), or (c), which is (a) for now; splits (a); Fortran (a). Evidence:
> plans/spikes/p12-xlang-v0.md (sections 8 and 9).

## Confidence

- High: the repository, commit, layout, program lists, statuses, and member hashes (API reads, the zip's own index,
  page reads at the commit, the zip's git blob id, the live site matching the repository, scripted comparisons); the
  constructs of the programs read from zip members, section 6's included (confirmed against its 36 members by a
  keyword and line search); output forms and description rules.
- Medium-high: license coverage (BSD-3 beside every program and linked from every page, but no contributor grant);
  section 6's structural readings beyond the names the search checked.
- Medium: every subset verdict (P12.4, P12.8, and P12.9 have published nothing beyond the bible's one line) and every
  cgeist mapping (read in source and tests at 77c04bb2, not run; member access, pointer parameters, initializer
  folding, and argv handling not read to the end).
- PROJECTED: every run-time figure, every claim of bitwise agreement under a shared operation order, and n-body's
  1e-8 threshold (a choice); P12.11's pairwise remote tests measure them. n-body's 3.0 AU separation bound and every
  size marked PLACEHOLDER are not yet fixed.
- Inference, to be checked by P12.11: mandelbrot's boundary-pixel movement under coordinate rounding and its size-200
  conformance; that the upstream programs agree at (b)'s sizes (mandelbrot at a power of two, spectral-norm at 522,
  fasta); the binary-trees check depending on depth alone; fasta's section lengths.

## Consequences for the plan

- OQ-057: this spike's section 9 goes to OQ-057 as its Response; P12.11 stays OWNER until the answer.
- P12.11 under (a) or (c): the four items with the structures, inputs, outputs, and tolerances of section 8; the
  scalar convention and float tolerance default of section 7; the mandelbrot conformance check and the fannkuch-redux
  anchors as tests; item-level `-ffp-contract=off` for C and C++, or a gcc.pin question to the owner if items cannot
  carry flags; for n-body, the whole-run minimum pair separation of each drawn seed, seeds below the bound refused,
  and the bound and the 1e-8 threshold fixed by measuring a one-ulp perturbation; fasta only if the owner takes it,
  with its constants read against OQ-018.
- P12.3 and P12.6: the binding ranks and dtypes of section 8.
- P12.4, P12.8, P12.9: publish the subset needs of section 8, or the items fall back (inlined functions, bounded
  counted loops).
- P5.7 and P12.12: the lassi_io.h route (its fprintf path) and the tag rule for llvm ops beside structured dialects;
  every C and C++ item reaches structured only if that route keeps the header's stdio calls out of the raised module;
  no fastmath `contract` from the CPU target.
- Under (b), or (c)'s later corpus: a zip-member fetch step in tools/fetch_bench.py, stdout_mask, and install-batch
  items for the libraries listed in section 8; n-body only with a numeric tolerance, and k-nucleotide,
  reverse-complement, and regex-redux only with a stdin route, both changes the harness does not have.
- Bible: no edit from this spike; P12.11 makes the xlang-v0 and Item tolerance edits once OQ-057 is answered.

## 10. Sources

All read 2026-10-18. Tags are local to their section.

Section 1 (source):

- [S1] https://benchmarksgame-team.pages.debian.net/benchmarksgame/ (index: benchmark list, "project repo", "license").
- [S2] https://salsa.debian.org/api/v4/projects/benchmarksgame-team%2Fbenchmarksgame
- [S3] https://salsa.debian.org/api/v4/projects/benchmarksgame-team%2Fbenchmarksgame/repository/branches
- [S4] https://salsa.debian.org/api/v4/projects/19610/repository/commits?ref_name=master (with `per_page=5`, and with
  `since=2025-03-08T00:00:00Z`).
- [S5] https://salsa.debian.org/api/v4/projects/19610/repository/tree?ref=40296663ed350d5fe4a6ab5e367bab61cb77c219
  (root, bencher, bencher/programs/*, public, public/download, public/description, public/performance,
  public/fastest, public/program pages 1-16), and BG raw/.gitlab-ci.yml.
- [S6] BG raw/public/download/benchmarksgame-sourcecode.zip (index, LICENSE and README entries, `_out` files, hashes).
- [S7] BG raw/public/description/<bench>.html for the 10 benchmarks (same bytes as
  https://benchmarksgame-team.pages.debian.net/benchmarksgame/description/nbody.html).
- [S8] BG raw/public/performance/<bench>.html for the 10 benchmarks and nbodyinprocess.
- [S9] BG raw/public/program/<page>.html, log sections: nbody-{gcc,gpp,csharpaot,csharpcore,rust,python3,ifx,gfortran}-1;
  {binarytrees,fannkuchredux,fasta,knucleotide,mandelbrot,pidigits}-gcc-1; {spectralnorm,regexredux,revcomp}-gcc-2;
  spectralnorm-gcc-3; spectralnorm-python3-8; {binarytrees,fasta,knucleotide,mandelbrot,pidigits,regexredux,revcomp,
  spectralnorm}-rust-1; fannkuchredux-rust-2; revcomp-python3-1.
- [S10] BG raw/README.md. [S11] BG raw/LICENSE.md. [S12] https://benchmarksgame-team.pages.debian.net/benchmarksgame/license.html
  (same bytes as public/license.html at the commit). [S13] BG raw/bencher/LICENSE and BG raw/bencher/license.txt.
  [S14] BG raw/CONTRIBUTING.md. [S15] BG raw/public/how-programs-are-measured.html and
  BG raw/public/description/nbodyinprocess.html.

Section 2 (n-body):

- [R] https://salsa.debian.org/benchmarksgame-team/benchmarksgame; https://salsa.debian.org/api/v4/projects/19610/repository/commits
  and .../repository/branches.
- [D] https://benchmarksgame-team.pages.debian.net/benchmarksgame/description/nbody.html (byte-identical to
  BG raw/public/description/nbody.html).
- [Z] BG raw/public/download/benchmarksgame-sourcecode.zip, and
  https://salsa.debian.org/api/v4/projects/19610/repository/files/public%2Fdownload%2Fbenchmarksgame-sourcecode.zip?ref=40296663ed350d5fe4a6ab5e367bab61cb77c219
  (headers).
- [P:<id>] BG raw/public/program/nbody-<id>.html. [O] BG raw/public/download/nbody-output.txt. [T]
  BG raw/public/data/data.csv. [L] BG raw/LICENSE.md and the nbody/LICENSE member of [Z]. Tree:
  .../repository/tree?path=bencher/programs/nbody&ref=40296663ed350d5fe4a6ab5e367bab61cb77c219.
- [CM], [CC], [CH] PG tools/cgeist/Lib/clang-mlir.cc, tools/cgeist/Lib/CGCall.cc, tools/cgeist/Lib/clang-mlir.h.
- [G] https://gcc.gnu.org/onlinedocs/gcc-12.3.0/gcc/Optimize-Options.html

Section 3 (spectral-norm):

- [B1] https://salsa.debian.org/api/v4/projects/19610/repository/commits, /branches, and
  /tree?ref=40296663ed350d5fe4a6ab5e367bab61cb77c219 (paths bencher/programs/spectralnorm, public/program,
  mybenchmarks/websites/linux/code), and `commits?path=`.
- [B2] BG raw/public/description/spectralnorm.html (live:
  https://benchmarksgame-team.pages.debian.net/benchmarksgame/description/spectralnorm.html).
- [B3] BG raw/LICENSE.md. [B4] BG raw/public/download/spectralnorm-output.txt. [B5]
  BG raw/bencher/makefiles/my.linux.ini. [B6] BG raw/public/download/benchmarksgame-sourcecode.zip.
- [B7] BG raw/public/program/spectralnorm-{gcc-8,gpp-1,csharpcore-8,rust-4,python3-8,gfortran-2}.html and the other
  pages listed in section 3. [B8] BG raw/public/performance/spectralnorm.html.
- [G1] PG tools/cgeist/Lib/clang-mlir.cc. [G2] PG tools/cgeist/Lib/CGCall.cc. Also PG tools/cgeist/driver.cc.
- [P1] https://docs.python.org/3/whatsnew/3.12.html (Other Language Changes: sum() uses Neumaier summation).
- [N1] https://numpy.org/doc/stable/reference/generated/numpy.sum.html (NumPy v2.5 manual, Notes: partial pairwise
  summation; the proposed pin is numpy 2.2.6, P12.1).

Section 4 (mandelbrot):

- [M1] https://salsa.debian.org/benchmarksgame-team/benchmarksgame/-/commit/40296663ed350d5fe4a6ab5e367bab61cb77c219;
  https://salsa.debian.org/api/v4/projects/19610/repository/branches; .../repository/tree?path=bencher/programs&ref=40296663ed350d5fe4a6ab5e367bab61cb77c219.
- [M2] https://benchmarksgame-team.pages.debian.net/benchmarksgame/description/mandelbrot.html and
  BG raw/public/description/mandelbrot.html.
- [M3] BG raw/public/download/mandelbrot-output.txt (and mandelbrot-output.pbm). [M4]
  BG raw/public/download/benchmarksgame-sourcecode.zip.
- [M5] BG raw/public/program/mandelbrot-<impl>-<n>.html for gcc-2, gcc-8, gpp-3, gpp-8, csharpcore-2, csharpcore-8,
  rust-5, python3-3, python3-8, gfortran-5, ifx-5, ifc-5.
- [M6] BG raw/public/measurements/gcc.html. [M7] the zip's LICENSE, mandelbrot/LICENSE, and BG raw/LICENSE.md.
- [M8] https://netpbm.sourceforge.net/doc/pbm.html. [M9] https://mathworld.wolfram.com/MandelbrotSet.html.
- [M10] https://gcc.gnu.org/onlinedocs/gcc-14.2.0/gcc/x86-Options.html. [M11]
  https://gcc.gnu.org/onlinedocs/gcc-14.2.0/gcc/Optimize-Options.html.
- [M12] https://github.com/llvm/Polygeist/tree/77c04bb2a7a2406ca9480bcc9e729b07d2c8d077/tools/cgeist/Test/Verification
  (cout.cpp, vector.cpp, simpcomplex.cpp, float_real_to_complex.c, fscanf.c).
- [M13] https://github.com/gcc-mirror/gcc/blob/releases/gcc-12.3.0/libstdc++-v3/include/std/complex (:1229-1369).

Section 5 (fannkuch-redux):

- [D] https://benchmarksgame-team.pages.debian.net/benchmarksgame/description/fannkuchredux.html and
  BG raw/public/description/fannkuchredux.html.
- [O] https://benchmarksgame-team.pages.debian.net/benchmarksgame/download/fannkuchredux-output.txt. [X]
  BG raw/public/download/fannkuchredux-extra.txt.
- [P] https://benchmarksgame-team.pages.debian.net/benchmarksgame/performance/fannkuchredux.html.
- [C] https://salsa.debian.org/api/v4/projects/benchmarksgame-team%2Fbenchmarksgame/repository/commits?per_page=3 and
  .../repository/branches. [T]
  https://salsa.debian.org/api/v4/projects/benchmarksgame-team%2Fbenchmarksgame/repository/tree?ref=40296663ed350d5fe4a6ab5e367bab61cb77c219&path=bencher/programs/fannkuchredux.
- [Z] BG raw/public/download/benchmarksgame-sourcecode.zip and
  .../repository/files/public%2Fdownload%2Fbenchmarksgame-sourcecode.zip?ref=40296663ed350d5fe4a6ab5e367bab61cb77c219.
- [L] BG raw/LICENSE.md.
- [Pg] https://benchmarksgame-team.pages.debian.net/benchmarksgame/program/<id>.html for fannkuchredux-gcc-8, -gpp-3,
  -csharpaot-8, -rust-4, -python3-8, -ifx-1.
- [PG] PG tools/cgeist/Lib/clang-mlir.cc, tools/cgeist/Lib/CGStmt.cc, and tools/cgeist/Test/Verification/whileset.c,
  capture.cpp, struct.cpp, vector.cpp.
- [OEIS] https://oeis.org/A000375

Section 6 (the other benchmarks):

- S1: https://salsa.debian.org/api/v4/projects/19610/repository/commits?ref_name=master&per_page=2 and
  https://salsa.debian.org/api/v4/projects/benchmarksgame-team%2Fbenchmarksgame.
- S2: https://salsa.debian.org/api/v4/projects/19610/repository/tree?path=public/description&per_page=100
- S3: https://salsa.debian.org/api/v4/projects/19610/repository/tree?path=bencher/programs&per_page=100;
  .../tree?path=bencher/programs/regexdna&ref=40296663ed350d5fe4a6ab5e367bab61cb77c219&per_page=100;
  .../tree?path=public/download&per_page=100.
- S4: BG raw/public/performance/pidigits.html. S5:
  https://benchmarksgame-team.pages.debian.net/benchmarksgame/performance/pidigits.html.
- Z: BG raw/public/download/benchmarksgame-sourcecode.zip (sha256 aabcf672...1cb4), its 36 members named in the
  section 6 tables, read in memory for sha256 and the keyword and line search (source check, 2026-10-18).
- D1-D6: https://benchmarksgame-team.pages.debian.net/benchmarksgame/description/{fasta,knucleotide,revcomp,pidigits,regexredux,binarytrees}.html.
  D7: BG raw/public/description/nbodyinprocess.html.
- O1-O6: BG raw/public/download/{fasta,knucleotide,revcomp,pidigits,regexredux,binarytrees}-output.txt (form only).
- P1-P6: https://benchmarksgame-team.pages.debian.net/benchmarksgame/performance/{fasta,knucleotide,revcomp,pidigits,regexredux,binarytrees}.html.
- G1-G36, under https://benchmarksgame-team.pages.debian.net/benchmarksgame/program/: G1-G6 fasta-{gcc-1, gpp-1,
  csharpaot-1, rust-1, python3-1, ifx-1}; G7-G12 knucleotide-{gcc-1, gpp-1, csharpaot-1, rust-1, python3-1, ifx-1};
  G13-G18 revcomp-{gcc-2, gpp-1, csharpaot-1, rust-2, python3-1, ifx-1}; G19-G24 pidigits-{gcc-1, gpp-4, csharpaot-7,
  rust-4, python3-4, ifx-4}; G25-G30 regexredux-{gcc-2, gpp-3, csharpaot-1, rust-1, python3-1, ifx-1}; G31-G36
  binarytrees-{gcc-1, gpp-1, csharpaot-1, rust-1, python3-1, ifx-1}.

Repository (read at 8b30baa): docs/BIBLE.md (Language Frontends rows :513-524 and Frontend Rules :528-534; Benchmark
Suites, the xlang-v0 row :593, Item tolerance :612, Held-out inputs :618; Harness Contract; Oracles, Rules for
binary_io :723-733); plans/p12-frontends.md (Constraints; Decisions taken at planning; P12.2, P12.3, P12.4, P12.8,
P12.9, P12.11); plans/p5-ir.md (P5.7); plans/OWNER-QUEUE.md (OQ-057); plans/spikes/p12-frontend-pins.md (Summary;
Flang finding 3; Proposed pins); toolchains/gcc.pin (FLAGS, :12); assets/harness/c/lassi_io.h (:111, :116);
lassi/harness/lassi_io.py (:212-225, :478-493); lassi/bench/registry.py (:90).
