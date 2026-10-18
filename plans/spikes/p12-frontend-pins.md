# Spike P12.1: frontend toolchain pins, paths, and budget

- Task: P12.1 (plans/p12-frontends.md, "P12.1 Spike: frontend toolchain pins, paths, and budget (documentation
  only)", its Constraints, and "Decisions taken at planning", the frontend pin rule). Bible: Language Frontends (its
  table; Frontend Rules), Toolchain Pins (MLIR stacks; host compilers; CUDA archives), IR Strategy (cpu-mlir and
  llvm-ir rows), Sandbox, Host Facts (Workstation scratch and approvals), Risks And Questions (new frontend tooling;
  toolchain churn; questions 9 and 13), Agent Rules 7 and 10.
- Date: 2026-10-17 on the workstation, Git Bash. The Date headers of the Rust CDN, PyPI, and archive.ubuntu.com read
  2026-10-18 02:40 UTC, the evening of 2026-10-17 local time.
- Base: branch p12-frontends at 9cbd9bd (9cbd9bd01760b5f6cd0f077726609bab51ee3039).
- Review additions, 2026-10-18 (about 04:35 UTC, 00:35 local), after the review of the first draft: the DLTI syntax at
  LLVM 23 (section 6.B), flang-23's dependencies against batch 1 (Flang finding 5), the job-pool scope (Flang budget),
  promote-to-affine as a partial start (Flang finding 3), and the nightly's date rule (R1). Their reads are curl of
  raw files at named commits and of two package indexes, plus `git show` of the stash; each is dated where it
  appears. Nothing was installed, built, or run.
- Documentation only; nothing installed or built. The reads were curl and web fetches of pages, manifests, checksum
  files, package index and API JSON, and source files at named commits; `gh api` metadata; HTTP HEAD requests; and
  small HTTP range reads (64 KiB of each Rust archive's xz index, the last 400,000 bytes of the numpy wheel, and the
  gzip size trailer and first 10 bytes of the .NET SDK archive). No archive larger than a few MB was downloaded
  whole. No rx command ran; nothing touched alpha01, the LASSI distro, or a device; the repository was only read.
- Deviations, both in session scratch directories outside the repository: one `uv run` in a scratch directory that
  held a stray pyproject.toml created a .venv there; and one `uv run python` call ran in the scratch folder that held
  xDSL's downloaded pyproject.toml, so uv tried to build xDSL as an editable package and failed at setuptools-scm
  version detection before any xDSL code ran (only setuptools and setuptools-scm from PyPI ran, in uv's cache). That
  scratch .venv was deleted, and later Python ran with `--no-project` from another folder. Nothing in the repository,
  the distro, or uv's Python installs changed.
- Device: none.
- Labels: a value from source at a pinned commit is cited as path:line; a value from a package index, release
  manifest, checksum file, the GitHub API, or HTTP headers names its URL. Every size, time, or memory figure that was
  not read is PROJECTED with its basis. Nothing here is [MEASURED]; values quoted from earlier spikes keep their own
  tags. Lists of limits are known cases, not complete.
- Notation: `LL:` is llvm-project at 26eb4285b56edd8c897642078d91f16ff0fd3472, read as
  https://raw.githubusercontent.com/llvm/llvm-project/26eb4285b56edd8c897642078d91f16ff0fd3472/<path>. `PG:` is
  Polygeist at 77c04bb2a7a2406ca9480bcc9e729b07d2c8d077 (https://github.com/llvm/Polygeist). `X73:` and `X23:` are
  xDSL v0.73.0 (tag commit a2544953390717dc5502ce385e0d337359d43639) and v0.23.0 (tag commit
  d1f33aac8cedd81790796713ee790b3a7cc0888f), https://github.com/xdslproject/xdsl. sandbox.py and native.py are
  lassi/executors/ at the base commit.
- P5.5's pin: it exists only in the local stash `stash@{0}`, "On p5-ir: P5.5 WIP (owner suspended 2026-10-13): tests
  and an unaudited pin, script, and sources list". Its files are untracked in the stash, so they sit in its third
  parent, 275520371cf5d3d00009a1e5228657e801886e76: `git show 'stash@{0}^3:toolchains/polygeist.pin'` reads it, and
  `git show stash@{0}:toolchains/polygeist.pin` finds nothing. "Stash pin :N" is a line of that file, and "stash
  polygeist.sh" is the script beside it. Both are provisional and unaudited throughout.
- Output conventions: outputs are trimmed where marked. No upstream file content beyond names, versions, hashes, and
  short quoted lines enters this file (OQ-018 practice).

## Frontend pin rule (fixed at planning, before any candidate was read)

The rule, verbatim from plans/p12-frontends.md, Decisions taken at planning:

> Frontend pin rule (P12.1), fixed before any candidate is read. The hub's verifier and lowering are the cpu-mlir
> level's tools at Polygeist's LLVM, llvm-project 26eb4285 (Toolchain Pins, MLIR stacks; P5.5, P5.6). A frontend
> from llvm-project (flang; LLVM import's mlir-translate) is taken at that commit. A frontend outside llvm-project
> prints hub text that this LLVM parses, and its own toolchain takes the newest release on the spike's date that
> serves it: the newest rustc nightly whose rustc_public exposes what the numeric subset reads; the newest .NET LTS
> SDK with a Roslyn the frontend needs; the newest numpy with cp310 wheels. A source build's compiler that the
> project does not install, such as C's, is a host compiler pinned by version and path, as gcc.pin pins g++
> (Toolchain Pins). A path the rule cannot serve (for example, rustc's LLVM IR read by the pinned mlir-translate) is
> recorded with its cause, and a pin outside the rule, such as a third LLVM, goes to the owner queue.

The spike's date is 2026-10-17.

## Question

What P12's install batch (P12.5) and its frontend and toolchain tasks (P12.6 to P12.10, P12.13, P12.14) build on, as
P12.1's acceptance asks: by the rule, a pin for the Rust MIR frontend's nightly with rayon, the .NET SDK with Roslyn,
numpy, the C compiler, and flang, each with its publisher's hash; what each build and native run needs against the
sandbox; what LLVM import at 26eb4285 reads and writes; the facts OQ-058 and OQ-059 wait on; and the budget against
the 145G stop line beside P5's projection.

Why it matters: P12.5 writes its install script from these pins; P12.6 changes the sandbox only for a need named here;
P12.7 to P12.10 take the routes and refusals found here; P12.13 and P12.14 wait on OQ-058 and OQ-059; each pin outside
the rule and each P5.5 pin change is an owner decision.

Classification: factual for every read. The paths the rule cannot serve, the pins outside it, the P5.5 pin change,
and the scope of a raise from LLVM IR are owner choices (Owner-queue candidates).

## Summary

- Rust: nightly-2026-10-17, rustc 1.101.0-nightly (5d8fec066 2026-10-16) on LLVM 23.1.4, with rustc, cargo, rust-std,
  rustc-dev, llvm-tools-preview, and rust-src, and rayon 1.12.0 with its five dependency crates vendored offline.
  rustc_public at 5d8fec06 exposes the bodies, locals, places, blocks, terminators, rvalues, types, and constants the
  subset reads. The sandbox misses two needs, a defined CARGO_HOME and a rayon thread bound (P12.6 [DESIGN] changes).
  rustc's LLVM IR is not readable by mlir-translate at 26eb4285 (LLVM 23.1.4 against 18.0.0git), so the Rust low
  fallback is a path the rule cannot serve.
- .NET: SDK 10.0.402 (LTS 10.0, runtime 10.0.13, C# 14.0, Roslyn 5.9.0) and Microsoft.CodeAnalysis.CSharp 5.9.0 with
  its two dependencies in an offline feed, each with its publisher's SHA-512. The compile sandbox cannot run the SDK
  command line (no writable CLI home; no DOTNET_* name is allowed), so C# builds call csc.dll through the pinned host
  by path, and runs go through a script that execs the pinned host. A thread bound is unmet. NativeAOT-LLVM targets
  WebAssembly only, so it gives no Linux LLVM IR.
- Python and C: numpy 2.2.6, the newest with cp310 wheels, in a pip-less venv of batch 1's /usr/bin/python3.10;
  programs run through a `-B` launcher with no PYTHONPATH and no sandbox change, but no component provides that
  launcher yet. Batch 1's g++-12 brings gcc-12 12.3.0-1ubuntu1~22.04.3 at /usr/bin/gcc-12 by an exact Depends, so C
  gets a host pin, gcc-c, and a new toolchain class that puts `-lm` after the sources. Numba is not needed.
- Flang: at 26eb4285, built standalone against P5.5's build tree once LLVMFrontendOpenACC is built in that tree,
  through P5.5's TARGETS or by `ninja LLVMFrontendOpenACC` from flang's own script (a P5.5 pin change either way, by
  the plan's rule). Its driver links executables against a static runtime the sandbox serves unchanged. At the pin FIR
  lowers only to cf and llvm (HLFIR is experimental), so Fortran reaches the hub only as low. apt.llvm.org's jammy
  flang packages (17 to 23) are a third LLVM, outside the rule.
- LLVM import: `mlir-translate --import-llvm` at 26eb4285 writes the LLVM dialect only. `lift-cf-to-scf` exists but
  walks func.func only, and no pass at 26eb4285, in Polygeist 77c04bb2, or at llvmorg-23.1.3 raises the LLVM dialect
  to core dialects, so imported modules are low. Its producer under the rule is Polygeist's clang; the nightly's rustc
  (LLVM 23.1.4) and NativeAOT-LLVM (LLVM 18.1.3, WebAssembly only) are newer than the reader.
- OQ-058: cgeist's CPU route for CUDA (`--cuda-lower --cpuify=distribute -S`) is built unconditionally at 77c04bb2
  and, by source, needs neither POLYGEIST_ENABLE_CUDA nor a target beyond host, so P5.5's stashed pin serves it
  unchanged; clang 18.0.0git reads cuda@12.6.3 as newer than supported, a warning. Three runtime unknowns remain.
- OQ-059: xDSL 0.73.0 (2026-10-09, Python >=3.10) targets MLIR 22.1.2 and rejects what 26eb4285 prints for
  scf.parallel and dlti; 0.23.0, whose scf matches, lacks most affine custom forms and dlti. Neither parses the
  pinned hub text. Releases before 0.23.0 were not read; that they fail too is inferred from 0.23.0's lack of affine
  custom forms and dlti.
- Budget (PROJECTED): under the rule P12 adds 4.71 to 8.33 GiB to the scratch root, up to 9.09 GiB with the
  conditional items; beside P5's 28 to 46 GiB that is 32.71 to 55.09 GiB, under the 145G stop line. Flang's fallback,
  an in-tree build of its own, makes P12's part 14.71 to 26.08 GiB (42.71 to 72.08 GiB with P5). The one LLVM-scale
  build is flang's. Its peak memory at 8 compiles, 10 to 32 GiB, passes OQ-056's 20G job cap; pools of 4 compiles
  and 1 link (6 to 20 GiB) put its upper end at the cap, so its pin proposes 3 compiles and 1 link (5 to 17 GiB),
  which still needs the owner's 24GB .wslconfig figure (OQ-056) in place, since WSL's default is 15.6 GiB.
- Owner-queue candidates: one P5.5 pin change (LLVMFrontendOpenACC); one pin outside the rule offered as an
  alternative (apt.llvm.org flang), with xDSL's candidates going into OQ-059; three paths the rule cannot serve (Rust
  LLVM IR, C# through NativeAOT-LLVM, structured Fortran); and one scope choice (a raiser from LLVM IR to core
  dialects). The rest are conditional. Filed 2026-10-17 as OQ-063 (flang's route: candidates 1 and 3), OQ-064 (the
  three paths: candidates 4 to 6), and OQ-065 (the raiser: candidate 7).

## 1. Rust: rustc nightly, rustc_public, rayon, cargo offline

Reads, all 2026-10-17: the channel manifests and their `.sha256` files, the per-archive `.sha256` files, crates.io
index and API JSON, source at named commits, and one 64 KiB HTTP range read of each archive's xz index. Paths are
cited as read: rust-lang/rust at 5d8fec0669a64607db9ccbf663e7c23125bf4748 unless the finding names cargo (rust-lang/
cargo 3960fa69bb617e3d93c4ddc087c1b32bf12ad907), rustup (1.29.1), rayon (tag v1.12.0,
c9ced185ae3508246a9eb70c8407a1199bb1b77f), or rust-lang/llvm-project (915c669d232e4cad467ba95ff041041e2e7c3f69).
src/tools/rust-installer/install-template.sh at 5d8fec06 was read again on 2026-10-17 for its libdir default (R6).

### Findings

R1. Nightly choice.
- The manifest https://static.rust-lang.org/dist/2026-10-17/channel-rust-nightly.toml (923,466 bytes, Last-Modified
  2026-10-17 01:11:37 GMT) says `date = "2026-10-17"`. At line 26265 it gives rustc
  `1.101.0-nightly (5d8fec066 2026-10-16)`, git_commit_hash 5d8fec0669a64607db9ccbf663e7c23125bf4748. The GitHub API
  dates that commit 2026-10-16T12:32:57Z ("Auto merge of #164430").
- The undated channel-rust-nightly.toml already says `date = "2026-10-18"` (rustc 8006091eb, Last-Modified
  2026-10-18 01:11:36 GMT). The proposed pin takes the nightly whose date label is the spike's date, 2026-10-17. Read
  by local time, 2026-10-18 would also qualify, since it was published on the evening of 2026-10-17 local time. P12.1
  settles this tie-break, not an owner item, by the channel's date label, a UTC date (docs/BIBLE.md, Toolchain Pins and
  the Decision Log entry of 2026-10-17): nightly-2026-10-18 does not qualify, and its rustc_public was not read.

R2. Components for x86_64-unknown-linux-gnu, all `available = true` in the 2026-10-17 manifest: rustc (line 26471),
cargo (211), rust-std (26187), rustc-dev (26935), llvm-tools-preview (1355), and rust-src (target `"*"`, 25417).
`[renames.llvm-tools] to = "llvm-tools-preview"` is at lines 27460-27461. Each per-archive `<url>.sha256` file at the
same path matches the manifest's xz_hash (fetched for all six).

R3. rustc's LLVM is 23.1.4.
- Submodule src/llvm-project at 5d8fec06 is rust-lang/llvm-project 915c669d232e4cad467ba95ff041041e2e7c3f69 (GitHub
  contents API), on branch `rustc/23.1-2026-07-22` (rust .gitmodules:25-29).
- cmake/Modules/LLVMVersion.cmake:3-14 at 915c669d sets major 23, minor 1, patch 4, no suffix.
- The dist builder builds this submodule; its configure args name no external llvm-config
  (src/ci/docker/host-x86_64/dist-x86_64-linux/Dockerfile:87-101).
- Cargo is rust-lang/cargo 3960fa69bb617e3d93c4ddc087c1b32bf12ad907 (src/tools/cargo submodule; committed
  2026-10-12T19:29:05Z).

R4. rustc_public at 5d8fec06 exposes what the numeric subset reads (compiler/rustc_public, crate version
`0.1.0-preview`, Cargo.toml:3). It is not on crates.io: the API and the index both return 404.
- Bodies: `CrateItem::body()` and `expect_body()` (src/lib.rs:215-225), `all_local_items()` (lib.rs:301),
  `entry_fn()` (lib.rs:276), and the monomorphized `Instance::body()`, `Instance::resolve()`, and `resolve_closure()`
  (src/mir/mono.rs:55, 126, 149). The bodies are `tcx.instance_mir`, which is optimized MIR
  (rustc_public_bridge/src/context/impls.rs:142-143; builder.rs:27-46). So the MIR shape follows the rustc flags, and
  the frontend must fix them for byte-identical re-raises.
- Locals: `Body::locals()`, `arg_locals()`, `ret_local()`, `local_decl()` (src/mir/body.rs:77-125), and
  `LocalDecl { ty, span, mutability }` (body.rs:177-181).
- Places: `Place { local, projection }` (body.rs:729-732), with `ProjectionElem` Deref, Field, Index, ConstantIndex,
  Subslice, Downcast, and OpaqueCast (body.rs:841-905).
- Basic blocks: `Body.blocks: Vec<BasicBlock>` (body.rs:15-16) and `BasicBlock { statements, terminator }`
  (body.rs:184-186).
- Terminators: `TerminatorKind` Goto, SwitchInt, Return, Drop, Call, Assert, and others (body.rs:204-243);
  `Call { func, args, destination, target, unwind }` is at body.rs:221-227.
- Statements and rvalues: `StatementKind::Assign` (body.rs:527-538); `Rvalue` BinaryOp, CheckedBinaryOp, UnaryOp,
  Cast, Use, Ref, Aggregate, Len (body.rs:542-647); `BinOp` (body.rs:405-428).
- Types: `Ty::kind()` and `Ty::layout()` (src/ty/tys.rs:96-102). `RigidTy` has Bool, Int, Uint, Float, Adt, Array,
  Slice, RawPtr, Ref, FnDef, Closure, Tuple (tys.rs:558-581). `FloatTy` has F16, F32, F64, F128 (tys.rs:643-648).
- Constants: `Allocation::read_uint`, `read_int`, `read_bool` (tys.rs:1053-1071). There is no float reader, so
  floats are read as bits.
- Shipped in rustc-dev: rustc_driver_impl depends on rustc_public (compiler/rustc_driver_impl/Cargo.toml:33), so
  rustc_public is built into librustc_driver; rustc-dev ships the compiler crates' target libs plus the sources
  (src/bootstrap/src/core/build_steps/dist.rs:977-1000).

R5. How a driver uses rustc_public.
- The crate needs `#![feature(rustc_private)]` and `extern crate rustc_driver; extern crate rustc_interface; extern
  crate rustc_middle; #[macro_use] extern crate rustc_public;`. It calls `run!(args, callback)` with rustc-style args
  (rustc_internal/mod.rs:86-137, macro at 129). tests/ui-fulldeps/rustc_public/check_binop.rs:8-16 and 81-86 do
  exactly this; walking `TerminatorKind::Call`, then `RigidTy::FnDef`, then `Instance::resolve` collects callees
  there (check_binop.rs:28-74).
- The callback runs in `after_analysis`: returning `ControlFlow::Break` stops the compile (`Compilation::Stop`), and
  `Continue` goes on to codegen and link (rustc_internal/mod.rs:223-247). The frontend should stop.
- The driver finds its sysroot from the loaded librustc_driver through dladdr
  (compiler/rustc_session/src/filesearch.rs:277-345), so it needs no `--sysroot`.
- The guide (https://rust-lang.github.io/rustc_public/initial.html, lines 187-213 of the HTML) says to install
  `rustc-dev` and `llvm-tools`; its example toolchain lists `["llvm-tools", "rustc-dev", "rust-src"]`. Why
  llvm-tools: dist.rs:2746-2750 copies libLLVM.so into lib/rustlib/<target>/lib "as a dependency of rustc-dev to
  support the inherited -lLLVM". llvm-tools ships no MLIR tool (LLVM_TOOLS, dist.rs:51-67). rust-src is not needed to
  build; it is the guide's recommendation (inference).
- To see rayon and the harness crate, the driver can run as cargo's `build.rustc-workspace-wrapper`. Cargo then
  passes the rustc path as the first argument and builds the dependencies with the same rustc (cargo
  doc/book/src/reference/config.md:517-529).

R6. rpath, not LD_LIBRARY_PATH.
- The sandbox never passes LD_LIBRARY_PATH (sandbox.py:405-425).
- rustc's own binaries get `-Wl,-rpath,$ORIGIN/../lib`. Bootstrap passes that explicitly because `-C rpath` writes
  paths relative to the output location (src/bootstrap/src/core/builder/cargo.rs:321-368;
  rustc_codegen_ssa/src/back/rpath.rs:46-80).
- So the MIR driver links with an absolute `-C link-args=-Wl,-rpath,<toolchain lib>`, whose directory follows the
  install route (R8), with `P=$LASSI_TOOLCHAINS/rustc@nightly-2026-10-17` written out as an absolute path:
  - Route B, recommended: install.sh puts every `lib/*` file under its libdir (install-template.sh:565-567), whose
    default is `<prefix>/lib` (:808), so the rpath is `$P/lib`.
  - Route A: `$P/rustup/toolchains/nightly-2026-10-17-x86_64-unknown-linux-gnu/lib`.
- The sandbox mounts the toolchains root back at its own path (sandbox.py:739-748), so an absolute path under
  $LASSI_TOOLCHAINS resolves inside.
- libLLVM sits in the same lib directory (dist.rs:607-612) and is found through librustc_driver's own
  `$ORIGIN/../lib` runpath (inference from cargo.rs:363; not measured).

R7. Cargo offline with vendored crates.
- Home: only CARGO_HOME, else `home_dir()/.cargo`, sets cargo's home (cargo crates/home/src/env.rs:67-78;
  src/context/mod.rs:2175-2182; doc/book/src/guide/cargo-home.md:5-7; environment-variables.md:17-22). std's
  `home_dir` falls back to `getpwuid_r` when HOME is unset (library/std/src/env.rs:661-667). Compiles get no HOME and
  no CARGO_HOME (sandbox.py:1597-1602, 1649-1651, 414-425). So cargo's home is the passwd entry of the sandbox uid,
  which is 0 in the user namespace (sandbox.py:112-114): likely /root/.cargo on the read-only host (inference, not
  measured).
- Writes: cargo takes `.package-cache` and `.package-cache-mutate` locks under its home (src/util/cache_lock.rs:29-39,
  459-462) but tolerates a read-only home. Exclusive locks fall back to shared on EACCES or EROFS
  (cache_lock.rs:281-318, 543-556); a shared-lock failure is only a trace warning (cache_lock.rs:218-236); the
  last-use database save is skipped when nothing was recorded, and its failure is a warning
  (src/workspace/global_cache_tracker.rs:104-115, 1579-1621, 1813-1823); auto-gc is off offline
  (src/workspace/gc.rs:52-59). So a writable CARGO_HOME is not required. But cargo still reads
  `$CARGO_HOME/config.toml` (config.md:12-22), so an undefined home lets a host file outside the pin into a build.
- Vendoring and source replacement: doc/book/src/commands/cargo-vendor.md:14-24 ("Cargo treats vendored sources as
  read-only") and reference/source-replacement.md:59-69 and 147-149 (`.cargo-checksum.json`). Offline mode is
  `net.offline` (config.md:998-1007). The rustc path is `build.rustc` (config.md:501-506); without RUSTC or
  build.rustc, and without RUSTUP_TOOLCHAIN, cargo runs `rustc` from PATH (context/mod.rs:1808-1887).
- rustup at run time: rustup's installation docs say RUSTUP_HOME and CARGO_HOME must always be set when the proxies
  are used (https://rust-lang.github.io/rustup/installation/index.html). Compiles therefore call the toolchain's
  bin/cargo and bin/rustc by path, never the proxies.

R8. One prefix holds rustup's home and the toolchain.
- Route A, rustup: at install time, outside the sandbox, set `RUSTUP_HOME=$P/rustup` and `CARGO_HOME=$P/cargo` with
  `P=$LASSI_TOOLCHAINS/rustc@nightly-2026-10-17`, then run
  `rustup-init -y --no-modify-path --default-toolchain nightly-2026-10-17 --profile minimal --component
  rustc-dev,rust-src,llvm-tools` (rustup 1.29.1 src/cli/setup_mode.rs:36-66). The toolchain lands in
  `$P/rustup/toolchains/nightly-2026-10-17-x86_64-unknown-linux-gnu`. rustup checks each archive's sha256 against the
  manifest (src/dist/download.rs:84-119), checks the manifest against its `.sha256` (download.rs:143, 233), and
  stores the manifest at `lib/rustlib/multirust-channel-manifest.toml` (src/dist/prefix.rs:7-14).
- Route B, no rustup: curl each archive, check it against the manifest hash, and run its
  `install.sh --prefix=$P --components=... --disable-ldconfig`. With ldconfig disabled nothing is written outside the
  prefix (src/tools/rust-installer/install-template.sh:656-693, 791, 804, 815). B keeps each download with its hash,
  which P12.5's acceptance asks for.
- Recommendation: B. It installs the same pin and archives as route A, so it needs no owner decision.

R9. Linker. rustc runs `cc` by default (rustc_codegen_ssa/src/back/link.rs:1859-1875). jammy's gcc-12 package ships
/usr/bin/gcc-12 but no /usr/bin/cc or /usr/bin/gcc (https://packages.ubuntu.com/jammy/amd64/gcc-12/filelist). So the
build config sets `target.x86_64-unknown-linux-gnu.linker` to the C pin's compiler by path (config.md:1483-1489).
Whether `cc` exists in LASSI-Ubuntu-22.04 has not been measured.

R10. rayon 1.12.0 is the newest release (crates.io API max_stable_version; published 2026-04-14; rust_version 1.80;
MIT OR Apache-2.0). Its normal-dependency closure from the sparse index (https://index.crates.io/) is six crates;
their `cksum` values (sha256 of the .crate) are in the pin table. wasm_sync and loom are optional and off, and
dev-dependencies are not resolved. The .crate files come from https://static.crates.io/crates
(index.crates.io/config.json). rayon-core 1.13.0 has `links = "rayon-core"` and a build.rs (rayon-core/Cargo.toml:3-7
at tag v1.12.0).

R11. rayon's thread count. rayon uses RAYON_NUM_THREADS, else `available_parallelism()`
(rayon-core/src/lib.rs:458-486 at c9ced185), which counts the host CPUs unless an affinity mask or cgroup quota
limits it (https://doc.rust-lang.org/nightly/std/thread/fn.available_parallelism.html). The sandbox sets neither. WSL
sees 16 CPUs [MEASURED 2026-10-15, workstation; plans/spikes/local-host.md:110, 139]. The CPU-time cap is
`cpus x (wall + 3)` seconds per process (sandbox.py:115-121), so a rayon program running 16 threads under
`Limits.cpus` of 1 or 2 reaches that cap before its wall limit, the failure DEMO.2 fixed for OpenMP
(native.py:21-25, 54-66).

R12. rustc's LLVM IR cannot be read by mlir-translate at 26eb4285.
- `--import-llvm` parses with that commit's `llvm::parseIR` (LL:mlir/lib/Target/LLVMIR/ConvertFromLLVMIR.cpp:35-53).
  26eb4285 is LLVM 18.0.0git (LL:llvm/CMakeLists.txt:11-21), committed 2023-09-28. LLVM promises no textual IR
  compatibility and only old-to-new bitcode compatibility (https://llvm.org/docs/DeveloperPolicy.html, IR Backwards
  Compatibility).
- Three constructs rustc emits that 26eb4285 rejects:
  - Every array index, at any opt level, emits `getelementptr inbounds nuw` (rustc_codegen_ssa/src/mir/place.rs:281-298,
    line 295; rustc_codegen_llvm/src/builder.rs:992-1009). 26eb4285's `parseGetElementPtr` accepts only `inbounds`
    (LL:llvm/lib/AsmParser/LLParser.cpp:7836-7841); rustc's LLVM accepts `nusw` and `nuw` (its LLParser.cpp:9291-9295).
  - Every bool immediate emits `trunc nuw` (builder.rs:660-663 and 1016-1026). 26eb4285's casts take no flags
    (LL:llvm/lib/AsmParser/LLParser.cpp:6384-6397, 7162-7170).
  - Under -O, parameters get `captures(none)` (rustc_codegen_llvm/src/abi.rs:54-58 and 80-105;
    compiler/rustc_llvm/llvm-wrapper/RustWrapper.cpp:432-434). Only `nocapture` exists at 26eb4285
    (LL:llvm/include/llvm/IR/Attributes.td:146, against `captures` at rustc LLVM's Attributes.td:187).
- Even LLVM 18.1.8's lexer has `disjoint` and `nneg` (LLLexer.cpp:567, 569), which 26eb4285's lacks, so an
  LLVM-18-era rustc is not shown readable either.
- The path is not served. Cause: LLVM IR is not forward compatible; LLVM 23.1.4 output cannot be read by an LLVM
  18.0.0git parser.

### Proposed pin (Rust)

Archive URLs are https://static.rust-lang.org/dist/2026-10-17/<file>; crate URLs are
https://static.crates.io/crates/<name>/<name>-<version>.crate.

| Item | Pin | File | Publisher hash (where read) |
| --- | --- | --- | --- |
| Channel manifest | nightly-2026-10-17 | channel-rust-nightly.toml | sha256 5e87d004b989db293416a5595c0564bef052dca8d773f2c95cd165ee2ddce6db (same URL + `.sha256`; matched the fetched file; an `.asc`, 801 bytes, is also published) |
| rustc | 1.101.0-nightly (5d8fec066 2026-10-16), LLVM 23.1.4 | rustc-nightly-x86_64-unknown-linux-gnu.tar.xz | sha256 368aaf0bdb0ebb5b06adef78129540d7fae28189ce1758f07660d9132c3894ac (manifest xz_hash, line 26476; `.sha256` file) |
| cargo | 3960fa69b | cargo-nightly-x86_64-unknown-linux-gnu.tar.xz | sha256 cd98b02d00bc519d4dbf439d0d125688ebe81c2129ab5dd83a53a960cf57d95e (line 216) |
| rust-std | same | rust-std-nightly-x86_64-unknown-linux-gnu.tar.xz | sha256 f072cbc3d3bcb0668decdab02f203eff447d45dc05dab3d00c8876f695718417 (line 26192) |
| rustc-dev | same | rustc-dev-nightly-x86_64-unknown-linux-gnu.tar.xz | sha256 4f1adfbb84a9eaf7b024280ca664b3d08b08e593738208dff5c929de0ae738e7 (line 26940) |
| llvm-tools-preview | same | llvm-tools-nightly-x86_64-unknown-linux-gnu.tar.xz | sha256 631dfc3e1dbb0243c02e60ae4954018b2ee6e159ea40271c24d91d9360b93c1c (line 1360) |
| rust-src | same | rust-src-nightly.tar.xz | sha256 dc1f489de3053060387a96afa12d75378d998aa3351cfda6691151d05b8182af (line 25422) |
| rustup-init (route A only) | 1.29.1 (release-stable.toml) | https://static.rust-lang.org/rustup/archive/1.29.1/x86_64-unknown-linux-gnu/rustup-init | sha256 dda7234360b7f578ca8b0ddcb80145646fa61a67c1720a5abc7051b35c9fcb71 (same URL + `.sha256`) |
| rayon | 1.12.0 | rayon-1.12.0.crate | sha256 fb39b166781f92d482534ef4b4b1b2568f42613b53e5b6c160e24cfbfa30926d (index cksum and API checksum) |
| rayon-core | 1.13.0 | rayon-core-1.13.0.crate | sha256 22e18b0f0062d30d4230b2e85ff77fdfe4326feb054b9783a3460d8435c8ab91 (index cksum) |
| either | 1.19.0 | either-1.19.0.crate | sha256 0e9c71c2167ca323c882b99918929403426e2373ea17242ff5653e0d5e1058be (index cksum) |
| crossbeam-deque | 0.8.8 | crossbeam-deque-0.8.8.crate | sha256 622f3fc73690be383c7214310406f28a90e6edeadc3cea882f9d71e495b9711a (index cksum) |
| crossbeam-epoch | 0.9.21 | crossbeam-epoch-0.9.21.crate | sha256 dc74980687109a3b14c72fd458107bf0baa1da1a1a805e178d15501ba9b86d9d (index cksum) |
| crossbeam-utils | 0.8.23 | crossbeam-utils-0.8.23.crate | sha256 a31eee39dddec8330830986fcd7625edb5a24ec90ea038215273bbc3adb08ac6 (index cksum) |

Pin layout:
- PREFIX_NAME `rustc@nightly-2026-10-17`.
- The vendored crates go in `$P/vendor`, written by `cargo vendor --locked --versioned-dirs` from a committed
  Cargo.lock that carries the six checksums; cargo then checks `.cargo-checksum.json` against them.
- EXPECT_VERSION `rustc 1.101.0-nightly (5d8fec066 2026-10-16)`, PROJECTED from the manifest's version string; the
  install record reads the real banner.
- Recorded LLVM: 23.1.4 at rust-lang/llvm-project 915c669d.
- The MIR driver's rpath (P12.8): `-Wl,-rpath,$P/lib` under route B (R6); under route A it would be
  `$P/rustup/toolchains/nightly-2026-10-17-x86_64-unknown-linux-gnu/lib`.

### Build and native-run needs against the sandbox (Rust)

| Need | Met? | How, or what is missing |
| --- | --- | --- |
| cargo and rustc in a compile | Yes | Call by path under the exposed toolchains root; put that bin on the compile PATH (sandbox.py:322-325); `build.rustc` set by absolute path. No RUSTUP_HOME needed. |
| A defined CARGO_HOME | No | Not in ENVIRONMENT_NAMES, and HOME is refused (sandbox.py:414-425, 1597-1602). Cargo still works through its read-only fallbacks (R7), but its home falls to the passwd entry and a host `$CARGO_HOME/config.toml` could leak into a build. P12.6 [DESIGN]: add CARGO_HOME, set by the rustc Toolchain to an empty `@`-prefixed directory under the build dir (as COMPILE_TMPDIR is made) or to an empty read-only directory in the pin's prefix. |
| Writable CARGO_HOME | Not required | R7 (cache_lock.rs:281-318, 543-556; global_cache_tracker.rs:1579-1621). |
| Vendored source | Yes | `[source.crates-io] replace-with`, a directory source at `$P/vendor` (absolute, read-only), `[net] offline = true`, and `--frozen`, all in `<build>/.cargo/config.toml` (cwd search, config.md:12-22). P12.6 must keep model files from writing `.cargo/`. |
| Linker | Yes, if set | `target.x86_64-unknown-linux-gnu.linker = "/usr/bin/gcc-12"` (R9). |
| Parallelism in compiles | Yes, if set | `build.jobs = 2` to match COMPILE_CPUS (sandbox.py:449-452); cargo otherwise runs 16 jobs. |
| MIR driver finds librustc_driver | Yes, by rpath | Absolute `-Wl,-rpath,$P/lib` to the toolchain lib (route B; R6). LD_LIBRARY_PATH stays banned; no sandbox change. |
| Driver sysroot | Yes | Found through dladdr (filesearch.rs:277-345). |
| Driver sees dependency rlibs | Yes | Build them in the same cargo build through `build.rustc-workspace-wrapper` (R5); target dir in the build dir. |
| Native run of a Rust program | Yes | Static std, with glibc and libgcc_s at system paths (PROJECTED, not measured). Needs no HOME (native.py:54-66). |
| rayon thread bound in native runs | No | RAYON_NUM_THREADS is not in ENVIRONMENT_NAMES or `_program_environment` (R11). P12.6 [DESIGN]: add RAYON_NUM_THREADS=<Limits.cpus> beside OMP_NUM_THREADS. |
| Network | Not needed in the sandbox | Vendoring runs once, in P12.5's batch, with CARGO_HOME under `$P`. |

### Budget (Rust)

| Item | Download | Installed | Wall time at 8 jobs | Peak memory |
| --- | --- | --- | --- | --- |
| Six nightly archives (.tar.xz) | 298,228,180 B, 284.4 MiB (read, Content-Length: rustc 85,422,644; cargo 10,954,560; rust-std 31,606,644; rustc-dev 122,990,396; llvm-tools 41,266,956; rust-src 5,986,980) | Tar contents 1,610,757,632 B, 1.50 GiB (read from each xz index: 404.7, 35.2, 161.7, 653.0, 206.7, and 74.9 MiB; each index summed to the exact file size); installed about the same, PROJECTED | PROJECTED 1 to 5 min, assuming 1 to 5 MB/s for the 298 MB (bandwidth not measured) plus xz decode; network and xz bound, so jobs do not matter | PROJECTED under 1 GiB (xz decode), an allowance with no basis read |
| Manifest | 923,466 B (read) | 0.9 MiB | seconds | - |
| rustup-init (route A only) | 21,113,232 B (read) | PROJECTED about 20 MiB | seconds | - |
| Route B staging | - | PROJECTED peak about 3.3 GiB (archives, extracted tree, and install), about 1.8 GiB kept | - | - |
| rayon closure | 398,787 B (read, sum of crate_size) | PROJECTED 2 to 4 MiB vendored | seconds | - |
| MIR driver build (P12.8) | 0 (no crates.io dependencies planned) | PROJECTED 50 to 300 MiB target dir, an allowance with no basis read | PROJECTED 1 to 3 min, an allowance with no basis read | PROJECTED 1 to 3 GiB, an allowance with no basis read; one crate linked dynamically against librustc_driver, not LLVM-scale |
| Per-program compile (rayon rebuilt in each build dir) | 0 | PROJECTED 50 to 150 MiB, an allowance with no basis read; capped by COMPILE_DISK_MB 2048 | PROJECTED 15 to 60 s at `-j 2`, an allowance with no basis read | PROJECTED under 1.5 GiB, an allowance with no basis read; capped by COMPILE_MEMORY_MB 8192 |

Rust total: PROJECTED about 1.9 to 2.2 GiB. Rust has no LLVM-scale build.

## 2. .NET: C# Roslyn frontend, C# native path, NativeAOT-LLVM

Rule applied (plans/p12-frontends.md:27): the newest .NET LTS SDK whose Roslyn the frontend needs. The frontend
prints hub text that mlir-opt at llvm-project 26eb4285 (LLVM 18.0.0git) parses. Every source line below is cited at a
pinned commit; everything was read 2026-10-17.

### Commands (trimmed)

- `curl https://builds.dotnet.microsoft.com/dotnet/release-metadata/releases-index.json`: 6988 bytes,
  byte-identical to the dotnetcli.blob.core.windows.net copy.
- `curl https://builds.dotnet.microsoft.com/dotnet/release-metadata/10.0/releases.json`: 1100901 bytes, parsed with
  `uv run --no-project python -I`.
- `curl -I https://builds.dotnet.microsoft.com/dotnet/Sdk/10.0.402/dotnet-sdk-10.0.402-linux-x64.tar.gz`:
  Content-Length 240279509, Last-Modified 2026-10-13.
- `curl -r 240279505-240279508` on the same archive (the gzip size trailer, ISIZE): 646399488. `curl -r 0-9`:
  `1f 8b 08`.
- `curl https://builds.dotnet.microsoft.com/dotnet/checksums/10.0.13-sha.txt`: 58847 bytes.
- Release notes at dotnet/core 50e60dcd: `release-notes/10.0/10.0.13/10.0.13.md` and
  `release-notes/10.0/os-packages.json`.
- NuGet: flat-container `index.json`, `registration5-gz-semver2` leaves, `catalog0` leaves, and the `.nuspec` files
  for Microsoft.CodeAnalysis.CSharp, .Common, .Analyzers, Microsoft.Net.Compilers.Toolset,
  Microsoft.NETCore.App.Runtime.linux-x64, and the ILCompiler packages.
- `gh api` for tags and commits in dotnet/dotnet, dotnet/sdk, dotnet/runtime, and dotnet/runtimelab. Raw source read
  at SDK d037e5f8 (tag v10.0.402), VMR a8dd0577 (tag v10.0.402), runtime 6aa72daa (tag v10.0.13), install-scripts
  5147e323, and runtimelab 527a5657.
- `curl https://dot.net/v1/dotnet-install.sh`: a 301 redirect to builds.dotnet.microsoft.com, 63563 bytes,
  byte-identical to install-scripts 5147e323 `src/dotnet-install.sh`.
- The dotnet-experimental feed's flat2 `index.json` for Microsoft.DotNet.ILCompiler.LLVM. A HEAD request for a
  package was refused with 405, so its size was not read.
- `releases.ubuntu.com/22.04.5/ubuntu-22.04.5-wsl-amd64.manifest`: the package list of batch 1's image.

### Findings

1. Newest LTS. releases-index.json lists 11.0 as "sts", "go-live", 11.0.0-rc.2. 10.0 is "lts", "active",
   latest-release 10.0.13, latest-sdk 10.0.402, dated 2026-10-13, EOL 2028-11-14.
2. The 10.0.13 release (releases.json, `releases[0]`) has two SDKs, 10.0.402 and 10.0.113, both with runtime 10.0.13
   and csharp-version 14.0. linux-x64 URL:
   https://builds.dotnet.microsoft.com/dotnet/Sdk/10.0.402/dotnet-sdk-10.0.402-linux-x64.tar.gz. The publisher's
   SHA-512 (full value in the pin table) is the same in releases.json and in 10.0.13-sha.txt line 275. releases.json
   gives no size; the HTTP size is 240279509 bytes.
3. Roslyn in SDK 10.0.402. The VMR source-manifest at the tag names roslyn 02b35215 (`src/source-manifest.json:41-43`).
   `src/roslyn/eng/Versions.props:7-10` gives 5.9.0 with prerelease label 1. dotnet/sdk
   `eng/Version.Details.xml:87,95,103` names Microsoft.Net.Compilers.Toolset, Microsoft.CodeAnalysis, and
   Microsoft.CodeAnalysis.CSharp at 5.9.0-1.26466.105.
4. Microsoft.CodeAnalysis.CSharp on NuGet. The newest stable is 5.9.0; the flat-container index ends at "5.6.0",
   "5.9.0". Published 2026-08-17, MIT licence, repository commit 35d9211b on release/insiders; packageSize 12210215,
   SHA-512 base64 `pjIhVe62OXX1os6tkUW0lJ3eTmveLnq0YojCJoztOjEpE3ho6UFAm8gaHpM/tTLPutoWh6OQyDg52zAUalN2IQ==` (catalog
   leaf https://api.nuget.org/v3/catalog0/data/2026.08.17.16.59.10/microsoft.codeanalysis.csharp.5.9.0.json). It is
   the same 5.9 minor as the SDK's compiler but a different build (NuGet 5.9.0 against the VMR's 5.9.0-1.26466.105).
   Known case, not complete: servicing differences between the frontend's parser and csc are possible.
5. net10.0 dependency closure (from the nuspec files): CSharp 5.9.0 -> Common `[5.9.0]` and Analyzers
   `5.9.0-1.26328.17`, both `exclude="Build,Analyzers"`; Common 5.9.0 -> Analyzers `5.9.0-1.26328.17`. NuGet resolves
   the lowest applicable version, 5.9.0-1.26328.17, which exists on nuget.org (published 2026-08-18) and sorts below
   5.9.0. The netstandard2.0 group adds eight System.* packages, which a net10.0 tool does not need.
6. SDK layout. `src/Layout/redist/targets/GenerateLayout.targets:44-60` copies the toolset's `tasks/netcore/**` to
   `sdk/<ver>/Roslyn/`. The toolset package holds `tasks/netcore/bincore/csc.dll` (catalog leaf of
   Microsoft.Net.Compilers.Toolset 5.9.0), so csc is at `sdk/10.0.402/Roslyn/bincore/csc.dll`.
   `RestoreLayout.targets:72-91` bundles `packs/Microsoft.NETCore.App.Ref` and `packs/Microsoft.NETCore.App.Host.<rid>`.
   Runtime packs are bundled only when `BundleRuntimePacks` is set (`:87-89`), which happens only for source-only
   builds (`src/Layout/Directory.Build.props:53`), so a self-contained publish needs Microsoft.NETCore.App.Runtime.
   linux-x64 10.0.13 in the feed. The ref-pack directory version is not settled: dotnet/sdk `Version.Details.xml:41`
   says 10.0.12, while the release package list says 10.0.13 (10.0.13.md:457). P12.5 reads the directory name after
   extraction.
7. dotnet-install.sh (install-scripts 5147e323). Options include `-i|--install-dir` (:1681), `-v|--version` (:1669),
   `--no-path` (:1713), and `--keep-zip` and `--zip-path` (:1752-1758). Without `--install-dir` it installs to
   `$DOTNET_INSTALL_DIR` or else `$HOME/.dotnet` (:924-927). Temp files go to `${TMPDIR:-/tmp}/dotnet.XXXXXXXXX`
   (:1640). It checks only the file size after download (:594, :1059); a grep for sha512, sha256, or checksum finds
   nothing, so the script does not check the publisher's hash. Its sha256,
   082f7685e156738a1b2e2ed8381a621870d4ce8e8c59278034556f05c186eb2e, was computed here from the fetched copy; it is not
   a publisher hash.
8. CLI home (SDK d037e5f8). `src/Common/CliFolderPathCalculatorCore.cs:22-35`: `DOTNET_CLI_HOME`, else
   `Environment.GetFolderPath(UserProfile)`; runtime 6aa72daa `PersistedFiles.Unix.cs:49-96`: `HOME`, else
   `getpwuid_r(geteuid())`. `Program.cs:150-158` builds the sentinels from that folder, and first-run configuration
   then writes `<home>/.dotnet/<ver>.dotnetFirstUseSentinel` (`:223`; `FirstTimeUseNoticeSentinel.cs:11,36-39`;
   `IFileExtensions.cs:11-27`, which retries on IOException with no catch). `dotnet --version` returns earlier, at
   `Program.cs:175-178`. First-run variables are read at `Program.cs:191-197`: DOTNET_GENERATE_ASPNET_CERTIFICATE,
   DOTNET_CLI_TELEMETRY_OPTOUT, DOTNET_ADD_GLOBAL_TOOLS_TO_PATH, DOTNET_NOLOGO, DOTNET_SKIP_WORKLOAD_INTEGRITY_CHECK.
   Telemetry is on only if not opted out and the sentinel exists (`Telemetry.cs:47-48`). NuGet signature verification
   is on by default on Linux (`NuGetSignatureVerificationEnabler.cs:19-48`). NuGet (VMR
   `src/nuget-client/.../NuGetEnvironment.cs:292-296`) takes its home from `DOTNET_CLI_HOME`, else UserProfile; its
   scratch folder is `NUGET_SCRATCH`, else `Path.GetTempPath()/NuGetScratch<user>` on Linux (`:40-52`).
9. Docs.
   - https://learn.microsoft.com/en-us/dotnet/core/tools/dotnet-environment-variables (git 6b94bd8a): DOTNET_ROOT is
     "used only when running apps via generated executables (apphosts)"; NUGET_PACKAGES defaults to
     `~/.nuget/packages`; DOTNET_SKIP_FIRST_TIME_EXPERIENCE is "no longer supported in .NET Core 3.0 and later".
   - https://learn.microsoft.com/en-us/nuget/consume-packages/managing-the-global-packages-and-cache-folders (git
     2bc45989): for the global packages folder, "The environment variable takes precedence over the configuration
     setting." The NuGet temp folder on Linux defaults to `/tmp/NuGetScratch<username>` when TMPDIR is unset.
   - https://learn.microsoft.com/en-us/dotnet/core/project-sdk/msbuild-props (git eb76160a): `AppHostDotNetSearch`
     and `AppHostRelativeDotNet` arrived in .NET 9 and apply on publish only; `InvariantGlobalization` is documented
     there.
   - https://learn.microsoft.com/en-us/dotnet/core/tools/dotnet-build (git 4bce4d81): `--disable-build-servers` exists
     since the .NET 7 SDK; a build starts a background download of workload advertising manifests.
10. Host and app runs. hostfxr runs `dotnet [exec] <dll>` without the SDK (runtime
    `src/native/corehost/fxr/fx_muxer.cpp:577-591`). The .NET 10 packages for Ubuntu 22.04 (dotnet/core 50e60dcd
    `release-notes/10.0/os-packages.json`) are libc6, libgcc-s1, ca-certificates, libssl3, libstdc++6, libicu70,
    tzdata, and libgssapi-krb5-2; all are in batch 1's image (manifest lines 24, 134, 185, 208, 215, 317, 318, 528).
    `Environment.ProcessorCount` takes `PROCESSOR_COUNT` first (`clrconfigvalues.h:462`; `utilcode/util.cpp:1001-1010`),
    else the affinity mask (`pal/src/misc/sysinfo.cpp:152-197`) and the CPU quota.
11. NativeAOT-LLVM (runtimelab feature/NativeAOT-LLVM, head 527a5657, 2026-09-22; earlier commits 2026-07-07,
    2026-07-05, 2026-06-22).
    - `README.md:5`: "the only supported target is WebAssembly ... not investing work into them at present."
    - `docs/using-nativeaot/compiling.md:3`: it compiles on Windows x64 and Linux x64; `:56`: RIDs are `browser-wasm`
      or `wasi-wasm`; `:25-26,30`: packages are floating `10.0.0-*` from the dotnet-experimental feed.
    - LLVM is `llvmorg-18.1.3` with `-DLLVM_TARGETS_TO_BUILD=WebAssembly`
      (`eng/pipelines/runtimelab/install-llvm.ps1:33,73`). Emscripten is 3.1.56 (`install-emscripten.ps1:17-19`); the
      WASI SDK is 29 (`install-wasi-sdk.ps1:6`).
    - The newest feed package is 10.0.0-rc.1.26357.1. Its date, 2026-07-07, is a reading of Arcade's date code. No
      package exists for the 2026-09-22 head.

### Proposed pins (.NET)

NuGet package URLs follow https://api.nuget.org/v3-flatcontainer/<id>/<version>/<id>.<version>.nupkg, with the id in
lower case.

| Pin | Version or commit | URL | Publisher hash (where read) |
| --- | --- | --- | --- |
| `toolchains/dotnet.pin` (NAME=dotnet, PREFIX_NAME=dotnet@10.0.402, RUNTIME_VERSION=10.0.13) | SDK 10.0.402 | https://builds.dotnet.microsoft.com/dotnet/Sdk/10.0.402/dotnet-sdk-10.0.402-linux-x64.tar.gz | SHA-512 `1904a98dda2db34ba78ba3c4fb1b224455329af91c341d83547843c1839e112b04ae699ee92fa2449a8f62249196a938723078c0c60b9f65c88539d5687729c2` (10.0 releases.json; 10.0.13-sha.txt:275) |
| Feed: Microsoft.CodeAnalysis.CSharp | 5.9.0 | https://api.nuget.org/v3-flatcontainer/microsoft.codeanalysis.csharp/5.9.0/microsoft.codeanalysis.csharp.5.9.0.nupkg | SHA-512 base64 `pjIhVe62...UalN2IQ==` (full value in finding 4), hex `a6322155eeb63975f5a2cead9145b4949dde4e6bde2e7ab46288c2268ced3a3129137868e941409bc81a1e933fb532cfbada1687a390c83839db30146a537621` (catalog leaf 2026.08.17.16.59.10) |
| Feed: Microsoft.CodeAnalysis.Common | 5.9.0 | same pattern | SHA-512 base64 `zJKrcLzk...Qp6fRw==`, hex `cc92ab70bce4e7996b7c2c1dfd5369322b5ac50a8fba2a7aef8bdabd59e2e447d59ebb22aaea52ef106a96d2a2dfb545f733dde2067a2dc814ba8070429e9f47` (same catalog page) |
| Feed: Microsoft.CodeAnalysis.Analyzers | 5.9.0-1.26328.17 | same pattern | SHA-512 base64 `i6S7Z6J2...2WUW8w==`, hex `8ba4bb67a276bb93528b2ae3c9e0b2141e90889ca4c0c31bb5ff30ba91c8b159326ce5b8f1173b5d93bb8c35e5f0587229c5b43e40fd9380825d6e90d96516f3` (catalog 2026.08.18.17.14.14) |
| Only if a self-contained publish is chosen (not recommended): Microsoft.NETCore.App.Runtime.linux-x64 | 10.0.13 | same pattern | SHA-512 hex `fd1124825bcd8641d3698abeebf615636ae395e0b380d860276ffb6c4be9cbedfc4b292d661d7d426366d36eb57eb5401afd8e58ce42b713bdacad51e89631f2` (catalog 2026.10.13.18.30.11) |
| `toolchains/csharp-roslyn.pin` | the source tree hash of `frontends/csharp_roslyn/`, built by dotnet@10.0.402 | none | none (built here) |

The hex values are decoded from the publisher's base64; they are not computed from the files. The base64 values of
Common and Analyzers are shortened here; the catalog leaves hold them whole. P12.5 adds the sha256 values.

Install: P12.5 downloads the archive, checks the SHA-512, then runs `tar -xzf` into
`$LASSI_TOOLCHAINS/dotnet@10.0.402`, since dotnet-install.sh checks only the size (finding 7). If the script is used
anyway: `dotnet-install.sh --version 10.0.402 --install-dir "$LASSI_TOOLCHAINS/dotnet@10.0.402" --no-path --keep-zip
--zip-path <scratch file>`, then check the kept archive's hash.

EXPECT_VERSION: `10.0.402` from `dotnet --version` (10.0.13.md:30-33), or `dotnet --list-sdks`, which hostfxr answers
without loading the SDK. For the compile route, the version line printed by `csc -version` (its exact form not read).

### What each step needs, against the sandbox (.NET)

Sandbox facts used: the compile environment is exactly the ENVIRONMENT_NAMES names (sandbox.py:414-425, with
lassi/toolchains/pins.py:47 adding only NVHPC_CUDA_HOME). HOME is refused (sandbox.py:1597-1599). The program runs as
uid 0 of its user namespace (:20, :112-114) on a host that is read-only (step 3). The toolchains root is re-exposed
read-only (:739-748). There is no network. The build dir has a 2 GiB cap with 4096 inodes, and memory is 8 GiB
(:443-449). Native runs get PATH, LANG, TMPDIR, and OMP_NUM_THREADS only, and argv[0] must be an artifact in the runs
root (native.py:56, 250-261).

- Install and frontend build (outside the sandbox, through longjob):
  - The host profile exports `NUGET_PACKAGES=$root/.nuget/packages` and `DOTNET_CLI_HOME=$root/.dotnet-home`
    (tools/local/host-profile.sh:128-129). Because the environment variable wins over nuget.config (finding 9),
    nuget.config alone would extract packages into the folder the plan forbids. The build must set `NUGET_PACKAGES`
    (and `DOTNET_CLI_HOME`) to a folder under the build dir itself.
  - Also set `DOTNET_CLI_TELEMETRY_OPTOUT=1`, `DOTNET_NOLOGO=1`, `DOTNET_CLI_WORKLOAD_UPDATE_NOTIFY_DISABLE=1`,
    `DOTNET_SKIP_WORKLOAD_INTEGRITY_CHECK=1`, and `DOTNET_ADD_GLOBAL_TOOLS_TO_PATH=0`.
  - Use nuget.config with `<clear/>` and one local folder source, and pass `--configfile`, `--disable-build-servers`,
    and `-m:$LASSI_JOBS`. Without `--disable-build-servers`, compiler and MSBuild server processes could outlive the
    build and keep the scope alive (an inference).
  - The NuGet temp folder follows TMPDIR, which the profile sets, so nothing goes in /tmp.
  - Signature verification is on by default, and its revocation check goes online unless
    `NUGET_CERT_REVOCATION_MODE=offline` is set (NuGet CLI environment variables page, git 2bc45989).
- Frontend run (compile sandbox): met. argv is
  `$LASSI_TOOLCHAINS/dotnet@10.0.402/dotnet $LASSI_TOOLCHAINS/csharp-roslyn@<hash>/<tool>.dll ...`. The host finds the
  runtime next to itself, so no DOTNET_ROOT is needed. Reference assemblies come from the pin's
  `packs/Microsoft.NETCore.App.Ref/<ver>/ref/net10.0/`. ICU is met by libicu70, or set `InvariantGlobalization` in
  the build. Not met: an apphost started without the muxer, which needs DOTNET_ROOT (not on the allowlist) or a
  system install location (a system setting, refused). The .NET 9+ `AppHostRelativeDotNet` option would work, since
  both trees sit under one toolchains root.
- C# program build (compile sandbox):
  - `dotnet build` or `dotnet publish`: not met. `DOTNET_CLI_HOME`, `DOTNET_CLI_TELEMETRY_OPTOUT`, `DOTNET_NOLOGO`,
    `DOTNET_SKIP_WORKLOAD_INTEGRITY_CHECK`, `DOTNET_CLI_WORKLOAD_UPDATE_NOTIFY_DISABLE`, and `NUGET_PACKAGES` are not
    on the allowlist. Without them the CLI home becomes `/root` (getpwuid_r of uid 0) on a read-only mount, so the
    first-run sentinel write is expected to fail (inferred from source, not run).
  - Recommended and met: `<dotnet> exec <dotnet>/sdk/10.0.402/Roslyn/bincore/csc.dll -noconfig -nologo -optimize+
    -r:<ref pack>/*.dll -out:prog.dll prog.cs <C# binding>`. The wrapper writes `prog.runtimeconfig.json`
    (Microsoft.NETCore.App 10.0.13, optionally `System.Globalization.Invariant`). No CLI, MSBuild, or NuGet is
    involved.
- C# native run:
  - A framework-dependent apphost: not met (it needs DOTNET_ROOT). A self-contained output is met but costs about
    100 MB per trial dir and needs the SDK command line to build.
  - Recommended and met: a script artifact in the workdir,
    `exec $LASSI_TOOLCHAINS/dotnet@10.0.402/dotnet "<dir>/prog.dll" "$@"`.
  - Not met: a thread bound. The sandbox sets neither CPU affinity nor a CPU quota (OQ-011), so ProcessorCount is the
    host's CPU count and `Parallel.For` sizes its pool from it, as in the OMP_NUM_THREADS case (DEMO.2, TasksMax).
    `DOTNET_PROCESSOR_COUNT` is not on the allowlist (the DOTNET_ prefix is standard CLRConfig practice, not read
    here). A build-time cap exists in runtimeconfig (`System.Threading.ThreadPool.MaxThreads`, threading docs git
    028e3437), but it is fixed at build time rather than taken from Limits.cpus. Adding `DOTNET_PROCESSOR_COUNT`
    would be a Sandbox [DESIGN] edit with a Decision Log entry in the C# native-path task, as OMP_NUM_THREADS was.
- NativeAOT-LLVM: not met for any LASSI route. It needs a third LLVM, network packages from a prerelease feed, and
  Emscripten or the WASI SDK, and it emits wasm32 code.

### Budget (.NET)

| Item | Download | Installed | Wall time at 8 jobs | Peak memory |
| --- | --- | --- | --- | --- |
| SDK 10.0.402 | 240279509 B, 229.1 MiB (read, Content-Length) | 646399488 B, 616.5 MiB of uncompressed tar (read from the gzip size trailer, assuming one gzip member); du PROJECTED at 0.62 to 0.70 GiB (block rounding; file count not read) | PROJECTED 1 to 5 min, assuming 1 to 4 MB/s plus extraction | PROJECTED under 0.5 GiB (tar); not LLVM-scale |
| Offline feed, 3 packages | 19273255 B (read, packageSize) | 19273255 B in the feed | PROJECTED seconds | negligible |
| Frontend restore and publish | 0 (offline) | Packages folder 62725559 B extracted plus 19.3 MB of nupkg, about 78 MiB (read, sums of package entries); publish output PROJECTED at 12 to 25 MiB (the net10.0 Roslyn dlls are 7079208 B and 3103056 B, read; satellites and the tool's own dll come on top) | PROJECTED 0.5 to 2 min, an allowance with no basis read | PROJECTED 1 to 2 GiB, an allowance with no basis read |
| Runtime pack (self-contained only) | 40101183 B (read) | 100754355 B (read) | not applicable | not applicable |
| Per-program csc in the sandbox | none | PROJECTED under 1 MiB, an allowance with no basis read | PROJECTED 1 to 5 s, an allowance with no basis read | PROJECTED 0.2 to 0.5 GiB, an allowance with no basis read; capped by the 8 GiB compile limit |
| NativeAOT-LLVM | not proposed; size not read (the feed refused HEAD with 405) | not applicable | not applicable | a source build of its LLVM 18.1.3 would be LLVM-scale; not projected |

.NET total: download about 0.25 GiB; installed PROJECTED 0.73 to 0.82 GiB, the sums of the rows' ends (SDK 0.62 to
0.70, feed 0.018, packages folder 0.076, publish output 0.012 to 0.024), and up to 1.04 GiB with the kept SDK archive
(240,279,509 B, 0.22 GiB); no LLVM-scale build.

Confidence: high for versions, hashes, sizes, and NativeAOT-LLVM's targets (all read from publishers or pinned
source); medium for the sandbox behaviour, which is inferred from source and not run; P12.5 and the distro tests
measure it.

## 3. Python and C

Everything below comes from package indexes, release pages, docs, and upstream source at named tags, read 2026-10-17.
The reading scripts ran as `uv run --no-project python -I <script> <file>` from a session scratch directory outside
the repository, with each download in its own directory.

### Commands and outputs (trimmed; long output lines wrapped)

1. `curl -sS -o numpy.json https://pypi.org/pypi/numpy/json` (3,729,087 bytes), then a reader of its cp310 files:

   ```
   info.version: 2.5.4 requires_python: >=3.12
   == 2.2.6
     numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl 16801050
     fc7b73d02efb0e18c000e9ad8b83480dfcd5dfd11065997ed4c6747470ae8915 2025-05-17T21:29:27.675703Z
     requires_python= >=3.10 yanked= False
   final releases after 2.2.6 : 18
     2.3.0 requires_python= ['>=3.11'] ... 2.4.6 ['>=3.11'] ; 2.5.0 ... 2.5.4 ['>=3.12']  (no cp310 tag in any)
   ```

2. `curl -sS -o numpy-2.2.6.json https://pypi.org/pypi/numpy/2.2.6/json`, then a reader of that version:

   ```
   requires_python: >=3.10   requires_dist: None   yanked: False   vulnerabilities: []
   url: https://files.pythonhosted.org/packages/b4/63/3de6a34ad7ad6646ac7d2f55ebc6ad439dbbf9c4370017c50cf403fb19b5/
        numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
   size: 16801050  sha256 fc7b73d02efb0e18c000e9ad8b83480dfcd5dfd11065997ed4c6747470ae8915
   blake2b_256 b4633de6a34ad7ad6646ac7d2f55ebc6ad439dbbf9c4370017c50cf403fb19b5  md5 8f382b9ca6770db600edd5ea2447a925
   ```

3. Hash cross-check: `curl -sSL https://github.com/numpy/numpy/releases/tag/v2.2.6 | grep -o <the sha256 above>`
   found it twice, under that page's "Checksums" heading.
4. The wheel's zip central directory only, by a range request (400,000 of the 16,801,050 bytes):
   `curl -sS -r -400000 <url above>` returned `HTTP/1.1 206`, `Content-Range: bytes 16401050-16801049/16801050`; a
   reader of that tail printed:

   ```
   entries: 1102  files: 1102  uncompressed bytes: 58606290 (55.9 MiB)
     numpy: 28.9 MiB   numpy.libs: 26.8 MiB   dist-info: 0.2 MiB
     largest: numpy.libs/libscipy_openblas64_-56d6093b.so 25021457
   .py files: 491, 8989606 bytes
   ```

5. `curl -sS https://packages.ubuntu.com/jammy-updates/g++-12`: "Package: g++-12 (12.3.0-1ubuntu1~22.04.3)
   [universe]", with "dep: gcc-12 (= 12.3.0-1ubuntu1~22.04.3)".
6. `curl -sS https://packages.ubuntu.com/jammy-updates/amd64/gcc-12/filelist` lists `/usr/bin/gcc-12` and
   `/usr/bin/x86_64-linux-gnu-gcc-12`, among others.
7. `curl http://archive.ubuntu.com/ubuntu/dists/jammy-updates/{main,universe}/binary-amd64/Packages.gz` (4,757,462
   and 1,627,428 bytes; Last-Modified 2026-10-18 00:58:05 GMT), then an awk read of each stanza:

   ```
   g++-12 12.3.0-1ubuntu1~22.04.3 [universe] Depends: gcc-12-base (= ...), gcc-12 (= 12.3.0-1ubuntu1~22.04.3),
          libstdc++-12-dev (= ...) ...
          Size 12185324 SHA256 c4123ea47a1d422ec43fb0918f670cfeb7bef1f26e6bf372fe15099fd42bbfcd
          (same as batch 1's PACKAGES line)
   gcc-12 12.3.0-1ubuntu1~22.04.3 [main] Installed-Size 70069  Depends: cpp-12 (= ...), libgcc-12-dev (= ...),
          binutils (>= 2.38), ...
          Filename pool/main/g/gcc-12/gcc-12_12.3.0-1ubuntu1~22.04.3_amd64.deb Size 21740140
          SHA256 f1ce0ee798bd9c3d30c5602962521fc7faf309795fc6596de88df1cfc4732d55
   cpp-12 Size 10833634 Installed-Size 34554 SHA256 58a71c973aaaa24e2194ac061c362118e9aab3f2c926c92f6ecc70f82ac78c76
   libgcc-12-dev Size 2617904 Installed-Size 14449
          SHA256 a9933d9949219f3a0598bf48390091f45c03c036fe736de38f33b7c60eddca0e
   libstdc++-12-dev [universe] Depends: ... libc6-dev (>= 2.13-0ubuntu6)
   python3.10-venv 3.10.12-1~22.04.18 [universe] Depends: python3.10 (= ...), python3-pip-whl,
          python3-setuptools-whl, python3.10-distutils
   ```

8. Upstream source through `curl https://raw.githubusercontent.com/...`: numpy `v2.2.6`
   (`requirements/ci_requirements.txt:4`, `pyproject.toml:19`); OpenBLAS `v0.3.29` (`driver/others/memory.c:453-497`,
   `driver/others/openblas_env.c:75-88`); CPython `v3.10.12` (`Lib/site.py:493-540`); GCC through the gcc-mirror/gcc
   GitHub mirror of gcc.gnu.org's git, tag `releases/gcc-12.3.0` (`gcc/cp/g++spec.cc:35-36, 127, 408-413`,
   `gcc/c/gccspec.cc:28-30`, `gcc/gcc.cc:8783`, `gcc/c-family/c-opts.cc:257-258`).

### Findings

Python:
- P1. numpy 2.2.6 is the pin under the rule. It is the newest numpy with cp310 manylinux x86_64 wheels (uploaded
  2025-05-17). Every later final release requires Python 3.11 or newer, and the newest, 2.5.4, requires 3.12 or newer
  (commands 1 and 2). Its wheel is `numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`,
  16,801,050 bytes, sha256 `fc7b73d02efb0e18c000e9ad8b83480dfcd5dfd11065997ed4c6747470ae8915`, read from
  https://pypi.org/pypi/numpy/2.2.6/json and matched on https://github.com/numpy/numpy/releases/tag/v2.2.6. It
  declares no requirements, so the offline install needs no other wheel, and PyPI lists no vulnerabilities for it.
  The manylinux_2_17 tag needs glibc 2.17 or newer, and jammy has 2.35: alpha01's `ldd --version` printed
  `ldd (Ubuntu GLIBC 2.35-0ubuntu3.15) 2.35` [MEASURED 2026-09-23, alpha01, rx 20260923-041453-exec-1aa9;
  plans/spikes/p0-nvcc.md:498].
- P2. Its OpenBLAS thread count follows OMP_NUM_THREADS. The wheel bundles `libscipy_openblas64_` (command 4), and
  numpy pins `scipy-openblas64==0.3.29.0.0` (numpy v2.2.6 requirements/ci_requirements.txt:4); that the wheel carries
  this exact build is an inference. OpenBLAS 0.3.29 takes its thread count from OPENBLAS_NUM_THREADS, then
  GOTO_NUM_THREADS, then OMP_NUM_THREADS, and caps it at the CPU count (memory.c:466-488; openblas_env.c:75-88).
  Programs start with `env -i`, so the native executor's OMP_NUM_THREADS=<Limits.cpus> (native.py:56-66) is what
  bounds OpenBLAS.
- P3. The venv works from a symlinked interpreter inside the sandbox. CPython 3.10's site.py looks for pyvenv.cfg next
  to `sys.executable` and one level up, without resolving symlinks; it then puts the venv's site-packages on sys.path,
  and with system site-packages off it sets `ENABLE_USER_SITE = False` (Lib/site.py:493-540 at v3.10.12). So no HOME
  and no PYTHONPATH are needed. Ubuntu's 3.10.12-1~22.04.18 carries Debian patches to site.py; that the venv branch is
  unchanged is assumed, not read. `python3.10 -m venv --without-pip` skips ensurepip
  (https://docs.python.org/3.10/library/venv.html). python3.10-venv's Depends bring the pip and setuptools wheels
  anyway (command 7).
- P4. sys.path[0] is the script's directory (https://docs.python.org/3.10/library/sys.html#sys.path), so the harness
  module `lassi_io.py` copied beside the program imports with no PYTHONPATH. `-I` must not be used, because it drops
  the script's directory from sys.path (https://docs.python.org/3.10/using/cmdline.html#cmdoption-I).
- P5. `-B` keeps bytecode out of the workdir: it stops `.pyc` writes on import (cmdline.html#cmdoption-B). Linux takes
  one optional argument on a `#!` line of up to 255 characters since Linux 5.1
  (https://man7.org/linux/man-pages/man2/execve.2.html). By default uv compiles no bytecode at install;
  `--compile-bytecode` does (https://docs.astral.sh/uv/reference/cli/, the current docs, not pinned to 0.12.17).
- P6. Numba is not needed. P12.3's binding exports prange as range, and subset programs run under the pinned python
  with no Numba (plans/p12-frontends.md:30). The Numba fallback is outside P12's scope, because llvmlite would bring a
  third LLVM (plans/p12-frontends.md:7).
- P7. Python 3.10 reached upstream end of life on 2026-10-01, and 3.10.22 was its final security release
  (https://peps.python.org/pep-0619/ and https://devguide.python.org/versions/). numpy 2.2.6 is therefore the last
  numpy for this interpreter. This changes no pin.

C:
- C1. g++-12 brings gcc-12 at an exact version. g++-12 12.3.0-1ubuntu1~22.04.3 (jammy-updates, universe) depends on
  `gcc-12 (= 12.3.0-1ubuntu1~22.04.3)`. gcc-12 is in jammy-updates main and installs `/usr/bin/gcc-12`, a link to
  `x86_64-linux-gnu-gcc-12` (commands 5 to 7). So batch 1's pinned g++-12 brings gcc-12 at exactly that version. Its
  .deb is 21,740,140 bytes, sha256 `f1ce0ee798bd9c3d30c5602962521fc7faf309795fc6596de88df1cfc4732d55`, from the
  Packages index named in command 7; apt checks it against the signed index.
- C2. The rest of the C toolchain comes the same way. cc1 is in cpp-12, and libgomp in libgcc-12-dev and libgomp1.
  gcc-12 itself does not depend on libc6-dev, but g++-12 pulls it in through libstdc++-12-dev. Batch 1 keeps every
  dependency .deb under `<root>/install/batch1/debs` (tools/local/install-batch1.sh:26-27) and logs `dpkg-query -W`;
  that log is where the installed gcc-12 version becomes a record once the owner runs the batch.
- C3. The version banner is PROJECTED: `gcc-12 --version` should print `gcc-12 (Ubuntu 12.3.0-1ubuntu1~22.04.3)
  12.3.0`. Basis: GCC prints the invoked name, then the package version, then the version (gcc/gcc.cc:8783 at
  releases/gcc-12.3.0); on alpha01, `/usr/bin/gcc` from the same build printed `gcc (Ubuntu 12.3.0-1ubuntu1~22.04.3)
  12.3.0` [MEASURED 2026-09-23, alpha01, rx 20260923-041453-exec-1aa9; plans/spikes/p0-nvcc.md:478]; and gcc.pin records the g++-12 banner
  (toolchains/gcc.pin:13-16). gcc.pin's EXPECT_VERSION fits gcc-12 as well.
- C4. The C driver adds no libm. g++'s driver adds `-lm` (gcc/cp/g++spec.cc:35-36, 127, 408-413); the C driver's
  lang_specific_driver adds nothing (gcc/c/gccspec.cc:28-30). So C builds need `-lm` after the sources. GCC 12's
  default C dialect is gnu17 (c-opts.cc:257-258), which covers `assets/harness/c/lassi_io.h`, written for C99 and
  C++17 (its line 3).
- C5. gcc-native does not take C. Its SOURCE_SUFFIXES are `.cpp .cc .cxx` (lassi/toolchains/gcc.py:8-11, 108), so a
  C-only build fails with "no source file to compile" (lassi/toolchains/_base.py:475-478). C needs a new Toolchain
  class behind the existing interface (Agent Rule 3) with its own pin. The bible's native row still lists
  `g++ -O3 -fopenmp` for C and C++ (docs/BIBLE.md:647).

### Proposed pins (Python and C; proposed, not installed)

`toolchains/python.pin`:

| Key | Value |
| --- | --- |
| NAME | `python` |
| VERSION | `3.10.12-numpy2.2.6` (so a Trial's recorded VERSION names both) |
| PREFIX_NAME | `python@3.10.12-numpy2.2.6` |
| INTERPRETER | `/usr/bin/python3.10` (batch 1's `python3.10=3.10.12-1~22.04.18`) |
| NUMPY_URL | the files.pythonhosted.org URL in command 2 |
| NUMPY_SIZE | `16801050` |
| NUMPY_SHA256 | `fc7b73d02efb0e18c000e9ad8b83480dfcd5dfd11065997ed4c6747470ae8915` (PyPI JSON; the v2.2.6 release page) |
| Class binary | `bin/python3` (a class PIN_BIN) |
| EXPECT_VERSION | `Python 3.10.12`, PROJECTED from the P5.5 stash's `PYTHON_EXPECT` for `/usr/bin/python3.10` |
| VERSION_ARGS (optional) | prints `numpy.__version__` as well |

Proposed install for P12.5: `/usr/bin/python3.10 -m venv --without-pip $LASSI_TOOLCHAINS/python@3.10.12-numpy2.2.6`,
then an offline `uv pip install --python <prefix>/bin/python3 --no-index --require-hashes --compile-bytecode` from the
checked wheel. The uv cache stays in the scratch root and is used only at install time.

`toolchains/gcc-c.pin`, a host compiler pinned the way gcc.pin pins g++ (no PREFIX_NAME, no install script):

| Key | Value |
| --- | --- |
| NAME | `gcc-c` |
| VERSION | `12.3.0` |
| EXECUTABLE | `/usr/bin/gcc-12` |
| FLAGS | `-O3 -fopenmp` (the native row's) |
| LIBS | `-lm`, placed after the sources |
| EXPECT_VERSION | `(Ubuntu 12.3.0-1ubuntu1~22.04.3) 12.3.0` (C3, PROJECTED) |
| Toolchain class | `gcc-c-native`: SOURCE_SUFFIXES `(".c",)`, command `gcc-12 -O3 -fopenmp -o main <sources> -lm` |

### Needs against the sandbox (Python and C)

| Need | Met? | Source |
| --- | --- | --- |
| Venv interpreter `$LASSI_TOOLCHAINS/python@.../bin/python3`, a link to `/usr/bin/python3.10` | Met. The toolchains root is read-only in runs and compiles, and /usr is not hidden. | native.py:250-257; sandbox.py:82-85, 285-290 |
| Venv detection with no HOME | Met: pyvenv.cfg is found from the unresolved path, and user site is off. | site.py:493-540 |
| PYTHONPATH | None needed; it is not on ENVIRONMENT_NAMES anyway. | sandbox.py:414-425; P4 |
| A way to start the interpreter: the executor runs `[artifact, *inputs]` with no interpreter slot | Not met by any current component. The sandbox needs no change: the artifact must be an executable launcher using the resolved toolchains path, `#!<prefix>/bin/python3 -B`, or a `/bin/sh` wrapper that runs `exec <prefix>/bin/python3 -B <program>.py "$@"`. No Python toolchain exists in lassi/toolchains. | native.py:207-261 |
| No bytecode in the workdir | Not met through the environment. Importing `lassi_io.py` would write `__pycache__/lassi_io.cpython-310.pyc` into the writable workdir, and it would show up in `RunResult.output_files`. PYTHONDONTWRITEBYTECODE cannot be set: it is not on ENVIRONMENT_NAMES and the executor's environment is fixed. `-B` on the launcher solves it. | native.py:56, 209-213, 262-263; sandbox.py:414-425 |
| numpy's own `.pyc` files | Writes fail quietly on the read-only venv, and each import compiles again. Met by `--compile-bytecode` at install. | P5 |
| OpenBLAS thread bound | Met through OMP_NUM_THREADS. | P2 |
| Writable caches or TMPDIR | None needed. TMPDIR=/tmp is a private tmpfs. | native.py:56 |
| Version check in the compile sandbox | Met, with the interpreter visible. | lassi/toolchains/runner.py:232-237 |
| C compile: gcc-12, cc1, as and ld, libc headers, libgomp | Met: all under /usr, as g++-12 is today. | sandbox.py:199-213, 322-325; C1, C2 |
| C run: libgomp.so.1, libm, libc | Met: /usr/lib/x86_64-linux-gnu is visible. | sandbox.py:285-290 |
| C sources accepted | Not met by a component: gcc-native takes no `.c` file. Needs `gcc-c-native`. | C5 |

### Budget (Python and C)

| Item | Download | Installed (scratch root) | Wall time at 8 jobs | Peak memory |
| --- | --- | --- | --- | --- |
| numpy 2.2.6 wheel | 16,801,050 B (read, PyPI) | 58,606,290 B unpacked (read, the wheel's central directory); bytecode adds 5 to 10 MiB, PROJECTED (8,989,606 B of `.py` in 491 files, with `.pyc` at about 0.5 to 1.1 times the source) | PROJECTED under 1 min: one 17 MB fetch and an unpack, no build | PROJECTED under 1 GiB, an allowance with no basis read: uv unpacks and compiles bytecode; not LLVM-scale |
| Venv without pip | 0 (batch 1's python3.10-venv) | PROJECTED under 1 MiB: pyvenv.cfg, links, activate scripts | PROJECTED seconds | n/a |
| gcc-12, cpp-12, libgcc-12-dev | 0 for P12; batch 1 already fetches 21,740,140 + 10,833,634 + 2,617,904 B (read, index) and keeps the .debs in its own count | 0 in the scratch root; they land in the distro's /usr: 70,069 + 34,554 + 14,449 KiB (read, Installed-Size) | 0, no P12 step | n/a |

Python and C total: about 17 MB downloaded; about 0.08 GiB in the scratch root including the kept wheel, PROJECTED.

Confidence: high for the index and source reads; medium for the gcc-12 banner and for the claim that bytecode would
land in `output_files`, which come from reading source, not from a run.

## 4. Flang at llvm-project 26eb4285

`LL:` paths below are flang's, clang's, or llvm's sources at 26eb4285. Paths without a prefix are in the directory of
the last named file.

### Findings

1. A standalone build exists. When flang is the top-level source directory it sets FLANG_STANDALONE_BUILD
   (LL:flang/CMakeLists.txt:36-44), and then needs LLVM, Clang, and MLIR CMake packages, plus mlir-tblgen in
   LLVM_TOOLS_BINARY_DIR (:73-74, :92-96, :116-125).
   - It refuses LLVM_ENABLE_EH (:16-22). In standalone mode FLANG_INCLUDE_TESTS defaults to ON (:167-169).
   - The documented standalone build points LLVM_DIR, CLANG_DIR, and MLIR_DIR at a base build tree
     (`$ROOTDIR/build/lib/cmake/{llvm,clang,mlir}`) and passes LLVM_ENABLE_ASSERTIONS=ON
     (LL:flang/docs/GettingStarted.md:158-173). That base build is a full `ninja` of an in-tree clang;mlir;flang;openmp
     build (GettingStarted.md:72-87, :143-144). compiler-rt is needed only for 16-bit float (:36-37).
   - In-tree, flang requires mlir and clang and is not part of "all" (LL:llvm/CMakeLists.txt:120-121, :136-143).
2. P5.5's tree against that need.
   - LLVM_CMAKE_FLAGS: `-G Ninja`, Release, `LLVM_ENABLE_PROJECTS=clang;mlir`, `LLVM_TARGETS_TO_BUILD=host`,
     assertions ON, `LLVM_INCLUDE_TESTS=OFF`. TARGETS: `cgeist polygeist-opt mlir-opt mlir-translate clang`. Nothing
     is installed; the tree is built in place, with a depth-1 full checkout at src/llvm-project, so flang/ is present
     (stash pin :55, :63; polygeist.sh:1-26). The script builds every target in one call,
     `cmake --build ... --target <TARGETS> --parallel $LASSI_JOBS` (stash polygeist.sh:367; read 2026-10-18).
   - The build tree's CMake packages serve a standalone configure, and EH stays off (the default).
   - Gap, medium confidence: flang's parser includes `llvm/Frontend/OpenACC/ACC.h.inc` and `ACC.inc`
     (LL:flang/include/flang/Parser/parse-tree.h:28, :4116). Those headers come from LLVM's `acc_gen` target
     (LL:llvm/include/llvm/Frontend/OpenACC/CMakeLists.txt:1-4). flang's libraries also link LLVMFrontendOpenACC
     (Parser :30-32, Semantics :57-61, flangFrontend :46-56; LL:flang/lib/*/CMakeLists.txt). In a standalone build
     acc_gen is only an empty placeholder target (LL:llvm/cmake/modules/LLVMConfig.cmake.in:159-161), so the headers
     must already exist in P5.5's tree.
   - None of the libraries read in the closure of TARGETS names FrontendOpenACC or acc_gen: the clang driver tool
     (LL:clang/tools/driver/CMakeLists.txt:1-46); clangDriver, clangBasic, clangCodeGen, clangParse, clangSema, and
     clangAST (each CMakeLists.txt:1-30); mlir-opt :5-69 and mlir-translate :1-23; MLIRTargetLLVMIRExport
     (LL:mlir/lib/Target/LLVMIR/CMakeLists.txt:18-42); MLIROpenACCDialect, which takes ACC.td through mlir-tblgen
     instead (LL:mlir/include/mlir/Dialect/OpenACC/CMakeLists.txt:1-3); and MLIROpenACCToLLVMIRTranslation.
   - Every other LLVM component and MLIR library flang names (Passes, Extensions, IRReader, Target, FrontendOpenMP,
     the MLIR dialect, conversion, extension, and translation libraries, MLIROptLib) is linked by clang, mlir-opt, or
     mlir-translate.
   - The closure was not read in full. A configure followed by `ninja -t query` settles it.
3. Lowering at the pin.
   - HLFIR is opt-in: `-flang-experimental-hlfir` is "Use HLFIR lowering (experimental)"
     (LL:clang/include/clang/Driver/Options.td:6131-6133; CompilerInvocation.cpp:1030-1034).
   - `-fc1 -emit-fir` prints FIR right after lowering; with HLFIR on, it first runs the HLFIR-to-FIR pipeline.
     `-emit-mlir` is an alias of `-emit-fir`, and `-emit-hlfir` prints HLFIR (Options.td:6273-6278;
     FrontendActions.cpp:675-696, :1117-1130; CLOptions.inc:239-254).
   - The only exits from FIR: `cfg-conversion`, where fir.do_loop, fir.if, and fir.iterate_while become branches
     (Transforms/Passes.td:148-166); and `fir-to-llvm-ir` (CodeGen/CGPasses.td:19-26). The default pipeline runs
     cfg-conversion, then scf-to-cf, then fir-to-llvm-ir (CLOptions.inc:226-227, :294-326). `promote-to-affine`
     (Passes.td:46-70, read again 2026-10-18) lifts affine-legal fir.do_loop and fir.if to affine.for and affine.if,
     and fir.load and fir.store to affine.load and affine.store, adding "a dummy `fir.convert` cast to adapt values
     of type `!fir.ref<!fir.array>` to `memref`", so FIR casts stay in the module; `demote-affine` reverses the
     memory ops because "The `memref` type is not compatible with the Fortran runtime" (:72-85).
   - No FIR-to-scf or FIR-to-memref source exists at the pin: the GitHub contents of flang/lib/Optimizer/Transforms at
     26eb4285 list 21 files, none of them (`gh api repos/llvm/llvm-project/contents/...?ref=26eb4285...`).
   - Tools: fir-opt registers affine, fir, hlfir, acc, omp, scf, arith, cf, func, vector, math, complex, DLTI, fircg,
     and llvm (InitFIR.h:30-38; fir-opt.cpp:32-45). tco has `-emit-fir` (parse and print) and `-code-gen-llvm`, and by
     default runs the full pipeline to LLVM IR (tco.cpp:48-59, :110-128). Polygeist's mlir-opt links only MLIR's own
     dialect libraries (mlir-opt CMakeLists:5-69), so FIR text cannot parse there (inferred, not run).
   - A structured hub module from Fortran would need a new FIR-to-core conversion that is not upstream at the pin:
     fir.do_loop and fir.if to scf; fir.alloca, load, store, array_coor, and box to memref with layouts; fir.convert to
     arith; runtime calls kept as external func.call. promote-to-affine is a partial starting point for the
     affine-legal loops, conditions, loads, and stores, but not for its fir.convert casts or the rest of FIR.
   - The pin's own route is low: `flang-new -fc1 -emit-llvm`, then `mlir-translate --import-llvm` at the same LLVM.
     That importer reads text or bitcode through llvm::parseIR (LL:mlir/lib/Target/LLVMIR/ConvertFromLLVMIR.cpp:34-41);
     `lift-cf-to-scf` then follows (LL:mlir/include/mlir/Conversion/Passes.td:297), which is P12.7's route. The
     alternative is fir-opt's cfg-conversion plus fir-to-llvm-ir.
   - Upstream added the missing passes later (gh API commits on main and contents at tags): `fir-to-scf`, first commit
     953302eb9856, 2025-05-25, present at llvmorg-21.1.8 and llvmorg-22.1.8; `fir-to-memref`, first commit
     2698d156644b, 2026-01-14, present at llvmorg-23.1.3 and at 0cd28174a1f6 (Passes.td there :79-91 and :218-225),
     but not at llvmorg-22.1.8.
4. Executables and runtime.
   - The driver is `flang-new` (FlangDriver.md:18-23; flang-driver CMakeLists:14). It links executables with no
     experimental flag: Options.td at the pin, and at llvmorg-17.0.6, has no `flang-experimental-exec`.
   - The link line adds `-L<driver dir>/../lib -lFortran_main -lFortranRuntime -lFortranDecimal -lm`
     (LL:clang/lib/Driver/ToolChains/CommonArgs.cpp:971-998; Gnu.cpp:575-583). Building flang-new also builds those
     three libraries (flang-driver CMakeLists:18-25). They are static unless BUILD_SHARED_LIBS is on; Fortran_main is
     always static (AddFlang.cmake:54-65; runtime/FortranMain/CMakeLists.txt:1-3). The runtime avoids a libstdc++.so
     dependency (runtime/CMakeLists.txt:80-82). Fortran_main's `main` calls `_QQmain` (Fortran_main.c:12-23).
   - Intrinsic modules are found at `<bin>/../include/flang` (CompilerInvocation.cpp:657-664) and built by the
     `module_files` target with the new flang-new (f18 CMakeLists:7-23, :57-71). The ppc64le modules need no PPC
     target, because `-fsyntax-only` creates no TargetMachine (FrontendActions.cpp:127, :206-286).
   - `--version` prints `getClangToolFullVersion("flang-new")` (Driver.cpp:1947-1949).
5. apt.llvm.org jammy (outside the rule).
   - Suites jammy-13 to jammy-23, plus the snapshot suite (https://apt.llvm.org/jammy/dists/). Each amd64 Packages
     index's sha256 matched its Release file. Release dates: 2026-02-23 for 17 to 21, 2026-06-13 for 22, 2026-10-14
     for 23, 2026-10-15 for the snapshot. The Release signatures were not checked.
   - Batch 1 already adds the jammy-17 and jammy-20 suites, with clang-17 and clang-20
     (tools/local/install-batch1.sh:60-63, :88-91).

   | Suite | Package | Version | .deb bytes | Installed KiB | SHA256 (Packages index) |
   | --- | --- | --- | --- | --- | --- |
   | jammy-17 | flang-17 | 1:17.0.6~++20231209124227+6009708b4367-1~exp1~20231209124336.77 | 24,029,922 | 131,631 | 3d6e616480461ebd7e1d2f337b9d8eb1f227aa12a44c795f4deffcc6fb473bf2 |
   | jammy-17 | libflang-17-dev | same | 21,533,850 | 714,582 | 595d1488124fbada85f124fec101e8632be758fe87c2998a544de9932bc91940 |
   | jammy-18 | flang-18 | 1:18.1.8~++20240731024944+3b5b5c1ec4a3-1~exp1~20240731145000.144 | 26,142,048 | 142,019 | fe7bc07ab5b069f420505d8d9700e01bee1d9b9b8f39a3a89d42d73ac2326a75 |
   | jammy-18 | libflang-18-dev | same | 23,822,302 | 800,881 | 94a2861ed6d1b5aca729812ffea65e48a9869aabd05f328d7bfd12f7cb7ce0db |
   | jammy-20 | flang-20 | 1:20.1.8~++20250708082409+6fb913d3e2ec-1~exp1~20250708202428.132 | 16,443,066 | 90,626 | 7b69643de551a751073171f9faa38b761fef226cad29044533d58adf8ad9bcd1 |
   | jammy-20 | libflang-20-dev | same | 25,379,358 | 774,292 | cb812e821c631b229a900fbb500ec61fa828b97ffef78105d83d13a0a911f84e |
   | jammy-20 | libmlir-20 (a flang-20 dependency) | same | 13,346,602 | 73,708 | f5fa084fa1e849e2d1846b5863a2c8e842163ac26ad78756728735542b133752 |
   | jammy-22 | flang-22 | 1:22.1.8~++20260613092327+e80beda6e255-1~exp1~20260613092437.81 | 18,920,788 | 109,850 | ec4e546145f38a57f28b9b905f7f9df912473243e2d8d4da9dd023eda8445f02 |
   | jammy-22 | libflang-22-dev | same | 26,912,036 | 805,714 | 5df48ea74a1fc541560e35a09ffad6188a26215dc15626b63b3a7f2a040ab279 |
   | jammy-23 | flang-23 | 1:23.1.4~++20261014054259+0cd28174a1f6-1~exp1~20261014054309.84 (release/23.x snapshot; the newest tag is 23.1.3) | 20,262,506 | 116,082 | 52ec629d7e67a2785594d715e8d65e863bb762f923e32084a2504d06a30d293c |
   | jammy-23 | libflang-23-dev | same | 30,223,240 | 887,021 | f469aee1c5df9b1340983606e999e9988dd2dc761f091144064fc03f0b4a5682 |

   - Index URLs: https://apt.llvm.org/jammy/dists/llvm-toolchain-jammy-NN/main/binary-amd64/Packages. Index sha256
     values, each matched by its suite's Release file
     (https://apt.llvm.org/jammy/dists/llvm-toolchain-jammy-NN/Release):
     - jammy-17 `3bb8a452e70c315eaf117faea35dcb05720368d5bec52d5b68632f6fe135312f`
     - jammy-18 `380ae90b5df8029aca2e30302427949f0a384d4970b156d843eeb0cea7fc4da2`
     - jammy-20 `8a0e8e25adff657cca9169991c2f4357d45bb12edf6c44491a2bd7c10fda31df`
     - jammy-22 `ad09b4e29325c30fcbfe8c035b390ae66351b083858b3c7df27fd5f64d485dc9`
     - jammy-23 `cc04cdba571b0bf2adb8a50534ef343a31f9efb2c02676d38b65f06efa716e99`
   - Releases 19 and 21 exist too: 19.1.7 and 21.1.8.
   - Dependencies (each stanza's Depends, same Packages files): flang-N needs libllvmN, libclang-cppN, libstdc++-11-dev,
     and libgcc-11-dev; 20, 22, and 23 also need libmlir-N, and 22 and 23 libquadmath0. flang-20's line is
     `libmlir-20 (>= 1:20.1.8~++20250708082409+6fb913d3e2ec)`. Batch 1's clang-17 and clang-20 already bring libllvm and
     libclang-cpp at the same versions for 17 and 20, and both depend on libstdc++-11-dev and libgcc-11-dev too, but
     neither brings libmlir-20. So flang-17 needs nothing beyond batch 1 but itself and libflang-17-dev, and flang-20
     also needs libmlir-20.
   - flang-23, read again 2026-10-18 from the jammy-23 index (sha256
     cc04cdba571b0bf2adb8a50534ef343a31f9efb2c02676d38b65f06efa716e99, unchanged): `Depends: libc6
     (>= 2.34), libclang-cpp23 (>= 1:23.1.4~++20261014054259+0cd28174a1f6), libgcc-s1 (>= 3.0), libllvm23, libmlir-23
     (>= 1:23.1.4~++20261014054259+0cd28174a1f6), libquadmath0 (>= 4.6), libstdc++6 (>= 11), libstdc++-11-dev,
     libgcc-11-dev, libflang-23-dev (= <the same version>)` (trimmed). Batch 1 adds only the jammy-17 and jammy-20
     suites (tools/local/install-batch1.sh:63), so flang-23 needs the jammy-23 suite and libllvm23, libclang-cpp23, and
     libmlir-23 from it; their sizes were not read. Its libquadmath0 comes with batch 1: g++-12 depends on gcc-12 at
     its exact version (C1), gcc-12 on `libgcc-12-dev (= 12.3.0-1ubuntu1~22.04.3)`, and libgcc-12-dev on `libquadmath0
     (>= 12.3.0-1ubuntu1~22.04.3)` (http://archive.ubuntu.com/ubuntu/dists/jammy-updates/main/binary-amd64/Packages.gz,
     read 2026-10-18, sha256 38b4a31807366848a76294e3c9fe82f6d33d3f37bdc7adaf570331bdee929d7a). flang-23 and
     libflang-23-dev together are 50,485,746 bytes (the table).
   - The package file lists were not read (the .debs are 16 to 30 MB), so whether the runtime ships static or shared
     is unknown.
   - LLVM promises only bitcode back-compatibility: "The textual format is not backwards compatible" and "supports
     loading any bitcode since version 3.0" (LL:llvm/docs/DeveloperPolicy.rst:639-660). flang-17 is LLVM 17.x, an
     earlier version whose branch left main before the pin: the GitHub compare 26eb4285...llvmorg-17.0.6 reads
     "diverged", ahead_by 405 and behind_by 7136, with merge base d0b54bb50e51 (2023-07-25), and llvmorg-17.0.6 is
     dated 2023-11-28, after the pin (compare and commits API, read again 2026-10-17). So 17.0.6 carries 405 commits
     the pin lacks, and the policy covers flang-17's bitcode by version number only (inference, not shown). Output
     from flang-18 or later is not promised readable. flang-23's MLIR text is a known refusal at the pin as printed:
     its `dlti.dl_spec` module attribute (section 6.B).

### Proposed pin (Flang, by the planning rule)

- NAME=flang, VERSION=26eb4285, PREFIX_NAME=flang@26eb4285, LLVM_COMMIT=26eb4285b56edd8c897642078d91f16ff0fd3472
  (LLVM 18.0.0git), URL=https://github.com/llvm/llvm-project.git.
- Source: the flang/ tree of P5.5's depth-1 checkout, so there is no second fetch.
- Publisher hash: llvm-project publishes none for a commit. The integrity check is the commit id plus flang's git tree
  id 61b4657087e745864f78103478cdfaacb9ddc4dc (3,078 blobs, 22,008,225 bytes; read from the GitHub git trees API at
  the pin).
- Build: standalone in its own prefix, against `polygeist@77c04bb2/build/lib/cmake/{llvm,clang,mlir}`. This needs one
  P5.5 pin change: `LLVMFrontendOpenACC` built in P5.5's tree (its DEPENDS builds acc_gen), either added to P5.5's
  TARGETS or built by `ninja LLVMFrontendOpenACC` in that tree from flang's own script (candidate 1 (a) or (d);
  OQ-063).
- Flags: `-G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON` (to match P5.5's ABI; GettingStarted.md:166);
  `-DFLANG_INCLUDE_TESTS=OFF -DLLVM_PARALLEL_COMPILE_JOBS=3 -DLLVM_PARALLEL_LINK_JOBS=1`
  (HandleLLVMOptions.cmake:37-60), which give a PROJECTED peak of 5 to 17 GiB (basis 1; 4 compiles would give 6 to 20
  GiB, at OQ-056's 20G job cap); `-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF` and the system-registry equivalent; clang-17
  and clang++-17, as P5.5 uses.
- TARGETS: `flang-new fir-opt tco module_files`. EXPECT_VERSION: `flang-new version 18.0.0`, PROJECTED: `--version`
  prints `getClangToolFullVersion("flang-new")` (LL:clang/lib/Driver/Driver.cpp:1947-1949), and CLANG_VERSION leaves
  out LLVM_VERSION_SUFFIX (LL:clang/CMakeLists.txt:293-294); the build's banner is read when P12.10 builds it.

### Build and native-run needs against the sandbox (Flang)

Sources: sandbox.py:199-221, :322-325, :405-425, :445-452, :1575-1630; native.py:50-66, :250-259.

| Need | Met? |
| --- | --- |
| flang-new, its `-fc1` re-exec, fir-opt, and tco under $LASSI_TOOLCHAINS (exposed read-only) | Met |
| Intrinsic .mod files at `<bin>/../include/flang` and runtime archives at `<bin>/../lib` inside the standalone build tree (LL:flang/CMakeLists.txt:159-163, :220) | Met (read-only is enough) |
| User .mod files and objects written to the build dir; driver temp files in TMPDIR | Met (private TMPDIR) |
| Linker and crt objects found through the compile's caller PATH and /usr/lib/gcc (system paths stay visible) | Met, provided binutils and libgcc-dev are in the distro (via g++-12's Debian dependencies; not read) |
| Environment variables, HOME | No flang variable is needed, so ENVIRONMENT_NAMES does not change; no HOME use found in the files read |
| Compile limits (8192 MiB, 2048 MiB disk, 2 CPUs, 600 s) | PROJECTED enough for small programs |
| Native runs: static runtime, so NEEDED is libc and libm only | Met with PATH=SANDBOX_PATH and no loader variables |
| CPU target (P5.7) linking a raised module | Met: needs `-L<flang>/lib -lFortran_main -lFortranRuntime -lFortranDecimal -lm`, and the module must define `_QQmain` and `_QQEnvironmentDefaults` |
| Binding's C shim | Met: compiled with P5.5's clang (same commit) |

Needs not met (known cases, not complete): none on the source route. On the apt route, if its runtime ships shared
only, a program would need /usr/lib/llvm-N/lib at run time: the driver adds `-L` but no rpath (CommonArgs.cpp:984-998),
and native runs take no LD_LIBRARY_PATH. The fix would be an rpath or a [DESIGN] sandbox change. The same applies to
a flang built with BUILD_SHARED_LIBS.

### Budget (Flang; LASSI_JOBS 8, the proposed pools of 3 compiles and 1 link, 145G stop line)

| Item | Download | On disk | Wall time | Peak memory | Basis |
| --- | --- | --- | --- | --- | --- |
| P5.5 TARGETS + LLVMFrontendOpenACC, or its `ninja` target in P5.5's tree (candidate 1 (d)) | 0 | under 0.01 GiB, PROJECTED | under 1 min, PROJECTED | negligible | one source file, ACC.cpp, and two generated headers (LL:llvm/lib/Frontend/OpenACC/CMakeLists.txt:1-12) |
| (a) flang standalone, in place | 0 | 2 to 5 GiB, PROJECTED | 27 min to 1.8 h at the proposed 3 compiles and 1 link; 21 to 86 min at 4 compiles; 12 to 51 min at 8 compiles; PROJECTED | 5 to 17 GiB at 3 compiles and 1 link; 6 to 20 GiB at 4 and 1, whose upper end sits at OQ-056's 20G job cap; 10 to 32 GiB at 8 compiles; PROJECTED | basis 1 below |
| (b) own in-tree clang;mlir;flang at 26eb4285 | 0 if it reads P5.5's source; else a depth-1 checkout of 1.6 to 2.1 GiB on disk, PROJECTED, P5's row for the same checkout (plans/spikes/p5-mlir-pins.md:289, from the GitHub tree API's 1,513,866,872 bytes in 134,659 files at 26eb4285, :274); its download size was not read | 12 to 22 GiB, PROJECTED | about 1.5 to 4.3 h, PROJECTED, for two build steps (basis 2): the local 1 to 2.5 h for clang's and MLIR's targets at 8 jobs, plus (a)'s 27 min to 1.8 h for flang's at 3 compiles and 1 link; longer if the pools of 3 and 1 cover the whole build (not projected) | flang's step: 5 to 17 GiB, PROJECTED from basis 1. The clang and MLIR step at 8 jobs: not projected, since no figure for the LLVM, clang, and MLIR units or for the static clang and mlir links was read, and plans/local-plan.md 3.2 gives none for Polygeist's build either. With pools of 3 and 1 over the whole build: 5 to 17 GiB, if those links stay within basis 1's 2 to 8 GiB per link | P5's 10 to 17 GiB in-tree row (plans/spikes/p5-mlir-pins.md, Results 5) and the local 1 to 2.5 h (plans/local-plan.md 3.2), plus (a) |
| (c) flang added to P5.5's configure | 0 | +2 to 5 GiB, PROJECTED | depends on the pools (basis 2), PROJECTED: +12 to 51 min at P5.5's 8 compiles; +27 min to 1.8 h for a second build call at 3 compiles and 1 link; with pools of 3 and 1 in P5.5's configure, P5.5's whole build slows to 3 compiles (not projected) | 10 to 32 GiB at P5.5's 8 compiles, past OQ-056's 20G job cap; 5 to 17 GiB for the second build call, or with the pools in P5.5's configure if P5.5's own links stay within 2 to 8 GiB each; PROJECTED | (a); basis 2 |
| apt flang-17 / flang-20 (outside the rule) | 45,563,772 / 55,169,026 bytes, read (flang-N and libflang-N-dev, and for 20 libmlir-20) | 846,213 / 938,626 KiB, read (Installed-Size), in the distro's /usr; batch 1's clang-17 and clang-20 bring the other dependencies | minutes, PROJECTED | n/a | Packages index |

Basis 1, for (a):
- About 285 C++ translation units (flang lib 219 files, 6,661,778 bytes; runtime 58; tools 8; git tree). At 15 to
  60 s each, an allowance with no basis read for flang's own units, that is 4,275 to 17,100 s (1.2 to 4.8 h) of
  summed compile time. For comparison only: an earlier, non-LASSI clang;mlir LLVM build in WSL, at llvm-project
  60efb5c0a3f5 (llvmorg-23-init-10166), averaged 4.8 s per Ninja step (33,468 s of summed step time over 6,967
  steps), a (log) figure from its .ninja_log, not a LASSI measurement (plans/spikes/local-host.md:216-225). Add 3
  static links of 1 to 5 min each, run one at a time.
- Wall time, assuming a full compile pool: the summed compile time divided by the pool, plus 3 to 15 min of links.
  At 3 compiles that is 23.8 to 95 min plus links, about 27 to 110 min (1.8 h); at 4 compiles, 17.8 to 71.3 min
  plus links, about 21 to 86 min; at 8 compiles, 8.9 to 35.6 min plus links, about 12 to 51 min.
- Size scale: libflang-18-dev's Installed-Size is 800,881 KiB, and objects plus archives count twice. The 2020
  out-of-tree build was 651 MB (https://discourse.llvm.org/t/comparing-build-times-for-flang/57199, 2020-12-11).
- Memory: 1 to 3 GiB per compile and 2 to 8 GiB per link. Bases: "1GB per thread ... not unreasonable for flang"
  (https://lists.macports.org/pipermail/macports-users/2025-March/053432.html, 2025-03-07); about 5GB for one debug
  TU, "2Gb+" for parse-tree files, about 8Gb per link
  (https://lists.llvm.org/pipermail/flang-dev/2019-November/000066.html); FLANG_PARALLEL_COMPILE_JOBS was added only on
  2024-06-18 because "Some of the flang sources require large amounts of
  memory" (https://llvm.googlesource.com/llvm-project/+/083c683969b2436afdc45becadc955841f5f4d31%5E%21), and it is not
  in the pin.
- Peak memory, the compile pool times 1 to 3 GiB plus one link of 2 to 8 GiB: 10 to 32 GiB at 8 compiles; 6 to 20
  GiB at 4 compiles, whose upper end sits at OQ-056's 20G job cap with no margin; 5 to 17 GiB at 3 compiles.

Basis 2, the pools' scope, for (b) and (c) (read 2026-10-18):
- LLVM_PARALLEL_COMPILE_JOBS adds one global Ninja pool and sets CMAKE_JOB_POOL_COMPILE to it, and
  LLVM_PARALLEL_LINK_JOBS does the same for links (LL:llvm/cmake/modules/HandleLLVMOptions.cmake:37-46, :48-61), so a
  pool set in a configure covers every target that configure builds. The flang-only FLANG_PARALLEL_COMPILE_JOBS is
  not in the pin (basis 1). In (a) the standalone configure builds flang alone, so its pools cover flang only.
- (b) is a configure of its own, so its script can build in two steps: clang's and MLIR's targets at 8 jobs, then
  flang's at 3 compiles and 1 link. Its time and flang's peak assume those steps.
- (c) puts flang in P5.5's configure, whose script builds every target in one call at LASSI_JOBS (stash
  polygeist.sh:367). So flang's units compile at P5.5's 8 jobs (12 to 51 min, a peak of 10 to 32 GiB, past OQ-056's
  20G job cap), or pools of 3 and 1 in P5.5's configure slow LLVM, clang, MLIR, and Polygeist to 3 compiles as well,
  or P5.5's script adds a second build call for flang's targets at 3 compiles (27 min to 1.8 h, 5 to 17 GiB), a
  further change to that script.

Against the stop line: P5's 28 to 46 GiB plus (a) gives 30 to 51 GiB, and plus (b) 40 to 68 GiB (P5's 28 to 46 plus
(b)'s 12 to 22); with the rest of P12, the Totals give 42.71 to 72.08 GiB. All are under 145G. Memory: at 8 compile
jobs the upper end of (a)'s peak passes OQ-056's 20G job cap, and pools of 4 compiles and 1 link put it at the cap,
so the pin proposes LLVM_PARALLEL_COMPILE_JOBS=3 and LLVM_PARALLEL_LINK_JOBS=1 (5 to 17 GiB). Even 17 GiB is above
WSL's default 15.6 GiB of memory with 4 GiB of swap [MEASURED 2026-10-15, workstation; plans/spikes/local-host.md:139],
and the long-job runner turns swap off for a job (MemorySwapMax=0; docs/BIBLE.md, Decision Log, 2026-10-17), so the
build waits for the owner's memory=24GB in .wslconfig (OQ-056, option (a)) to be in place. Confidence in this budget
is low.

## 5. LLVM import at llvm-project 26eb4285

Classification: factual for every read below. Nothing was run: only curl of raw files at named commits, GitHub API
metadata, static.rust-lang.org manifests, and HEAD requests.

### Commands (all `curl -sS -f`, saved to a session scratch directory)

- llvm-project raw files at 26eb4285 (`LL:`): mlir/lib/Target/LLVMIR/{ModuleImport.cpp, ConvertFromLLVMIR.cpp,
  CMakeLists.txt}; mlir/include/mlir/Conversion/Passes.td; mlir/lib/Conversion/ControlFlowToSCF/ControlFlowToSCF.cpp;
  mlir/include/mlir/Transforms/{CFGToSCF.h, Passes.td}; mlir/include/mlir/Dialect/SCF/Transforms/Passes.td;
  mlir/include/mlir/Dialect/LLVMIR/Transforms/Passes.td; mlir/include/mlir/{InitAllTranslations.h, InitAllPasses.h};
  mlir/tools/mlir-translate/{mlir-translate.cpp, CMakeLists.txt}; mlir/tools/mlir-opt/mlir-opt.cpp;
  llvm/CMakeLists.txt, llvm/lib/IRReader/IRReader.cpp, llvm/docs/{DeveloperPolicy.rst, LangRef.rst}.
  mlir/lib/Dialect/SCF/Transforms/UpliftWhileToFor.cpp returned 404 (it returns 200 at llvmorg-23.1.3).
- llvm-project GitHub API: `/repos/llvm/llvm-project/commits/26eb4285...`;
  `/contents/mlir/include/mlir/Conversion?ref=26eb4285...` and `?ref=llvmorg-23.1.3`;
  `/contents/mlir/include/mlir/Transforms?ref=26eb4285...`; `/compare/26eb4285...llvmorg-18.1.3?per_page=1`;
  `/releases?per_page=15`; raw `.github/workflows/release-binaries.yml` and `clang/cmake/caches/Release.cmake` at
  llvmorg-23.1.3.
- Polygeist raw files at 77c04bb2 (`PG:`): include/polygeist/Passes/Passes.td, lib/polygeist/Passes/CanonicalizeFor.cpp.
- Rust: `https://static.rust-lang.org/dist/2026-10-17/channel-rust-nightly.toml` and
  `.../channel-rust-nightly-git-commit-hash.txt`;
  `/repos/rust-lang/rust/contents/src/llvm-project?ref=5d8fec0669a64607db9ccbf663e7c23125bf4748` and `?ref=1.77.2`; raw
  `.gitmodules`, `RELEASES.md`, and `compiler/rustc_codegen_llvm/src/abi.rs` at 5d8fec06;
  rust-lang/llvm-project at 915c669d (`cmake/Modules/LLVMVersion.cmake`, `llvm/docs/LangRef.md`) and at d1e9093d
  (`llvm/CMakeLists.txt`); `https://static.rust-lang.org/dist/channel-rust-1.77.2.toml`, and HEAD requests for its
  three archives.
- NativeAOT-LLVM: `/repos/dotnet/runtimelab/branches/feature/NativeAOT-LLVM` (head
  527a565752062b20bb9693eea6c884b34ecce7bf, 2026-09-22); raw README.md, eng/Versions.props,
  eng/pipelines/runtimelab/install-llvm.ps1, and src/coreclr/jit/CMakeLists.txt at that head; raw
  llvm/docs/LangRef.rst at llvmorg-18.1.3.
- https://llvm.org/docs/DeveloperPolicy.html.
- Repository: `git stash list`, then `git show 'stash@{0}^3:toolchains/polygeist.pin'` and
  `git show 'stash@{0}^3:toolchains/polygeist.sh'`.

### Findings

1. The reader. 26eb4285 is dated 2023-09-28 (commits API) and is LLVM 18.0.0git (LL:llvm/CMakeLists.txt:11-26).
   `import-llvm` is registered at LL:mlir/lib/Target/LLVMIR/ConvertFromLLVMIR.cpp:34-35; registerAllTranslations
   calls that registration (InitAllTranslations.h:28-31), and mlir-translate's main runs it (mlir-translate.cpp:38).
   The libraries are linked through MLIR_TRANSLATION_LIBS (mlir-translate CMakeLists.txt:6, :15;
   mlir/lib/Target/LLVMIR/CMakeLists.txt:62-87). It parses with llvm::parseIR (ConvertFromLLVMIR.cpp:40-42), which
   reads bitcode or text (IRReader.cpp:69-89; the isBitcode check is at :75), then verifies the module (:50) and
   translates data layout, comdats, metadata, globals, and functions (ModuleImport.cpp:1893-1925).
2. What the import writes: the LLVM dialect only. Each function becomes llvm.func (ModuleImport.cpp:1694); br becomes
   llvm.br or llvm.cond_br (:1287-1300), and switch becomes llvm.switch (:1302-1335; the file uses
   `using namespace mlir::LLVM`, :44); a PHI becomes a block argument (:1336-1340), and loop metadata a loop
   annotation (:1886-1890). Refusals, known cases, not complete: blockaddress (:1091-1092), and any unhandled
   instruction, intrinsic, constant, or global (:1501, :1267, :1094, :550-557); unknown metadata only warns (:595). So
   every imported module is `low` (docs/BIBLE.md:529): it holds no func, cf, arith, or memref ops.
3. `lift-cf-to-scf` is present (LiftControlFlowToSCFPass, LL:mlir/include/mlir/Conversion/Passes.td:297-321). It uses
   the generic algorithm in CFGToSCF.h:135-154, and mlir-opt registers it (mlir-opt.cpp:266; InitAllPasses.h:60).
   - It walks func.func only (ControlFlowToSCF.cpp:170).
   - It lifts only cf.cond_br (to scf.if) and cf.switch (to scf.index_switch) (:36-69); any other branch op fails the
     pass (:71-73).
   - Loops become do-while scf.while: the whole body goes in the "before" region, and the condition is an i32 flag
     truncated to i1 (:85-117). It never produces scf.for. No upstream while-to-for pass exists at this commit:
     UpliftWhileToFor.cpp is absent, and the SCF passes offer only scf-for-to-while (SCF Passes.td:120).
   - Undefined values become ub.poison (:129-133; ub is a dependent dialect, Passes.td:319), and an infinite loop's
     unreachable end becomes func.return of poison (:135-156). ub is not a hub dialect (docs/BIBLE.md:510).
   - A region with more than one kind of return keeps one cf.switch (Passes.td:304-307; CFGToSCF.h:141-144).
   - On an imported module the pass does nothing: there is no func.func to walk.
4. No pass turns the LLVM dialect into core dialects. At 26eb4285, every entry of mlir/include/mlir/Conversion goes
   toward LLVM, SPIR-V, or similar, with ControlFlowToSCF the only lift; no pass in Conversion/Passes.td names
   LLVMFuncOp. The LLVM-dialect passes do other things (LLVMIR/Transforms/Passes.td:14-69). Polygeist 77c04bb2 has no
   such pass either (PG:include/polygeist/Passes/Passes.td:7-246). At llvmorg-23.1.3 the Conversion directory adds
   RaiseWasm, for the wasmssa dialect, and still has no LLVM-dialect raise.
5. What a structured module from LLVM IR would take (design drawn from the reads; nothing was run): (a) import, which
   exists; (b) mem2reg and sroa (Transforms/Passes.td:332, :389) on the LLVM dialect, without having read whether
   every alloca promotes; (c) a new LASSI raiser, a component under Agent Rule 3: llvm.func to func.func, llvm
   branches to cf, integer and float ops to arith, intrinsics to math, and loads, stores, and getelementptr on
   !llvm.ptr to memref, the hard step, since opaque pointers carry no element type and array shapes must be recovered
   from GEP arithmetic; (d) lift-cf-to-scf; (e) Polygeist's canonicalize-scf-for
   (PG:include/polygeist/Passes/Passes.td:171), whose MoveWhileToFor (CanonicalizeFor.cpp:894) matches an arith.cmpi and
   arith.addi induction (:662-725, :903) only
   when the before region holds just the comparison (:702-709), while the lift's do-while form puts the whole body
   there, so whether it matches lifted loops is not established by reading; (f) remove ub.poison, or extend the
   hub's dialect list; (g) optionally, raise-scf-to-affine (Polygeist Passes.td:162). Steps (c) and (f) are new work.
6. Whose LLVM IR the reader can take.

   | Producer | Its LLVM | Covered by policy | Evidence |
   | --- | --- | --- | --- |
   | Polygeist's clang at 26eb4285 (stash pin :63, :88) | 18.0.0git, the same commit | Yes, text or bitcode | Same LLVM |
   | rustc nightly of 2026-10-17: 1.101.0-nightly (5d8fec066 2026-10-16), from the manifest | 23.1.4: rust-lang/llvm-project 915c669d, branch rustc/23.1-2026-07-22 (LLVMVersion.cmake:3-11); Rust has used LLVM 23 since 1.99.0 (RELEASES.md:139) | No | below |
   | NativeAOT-LLVM at runtimelab 527a5657 | llvmorg-18.1.3 (install-llvm.ps1:33), built with LLVM_TARGETS_TO_BUILD=WebAssembly (:73); the JIT links core and bitwriter (jit CMakeLists.txt:95) | No | below |
   | Outside the rule: rustc 1.77.2 stable (2024-04-09) | 17.0.6: rust-lang/llvm-project d1e9093d, Rust's own fork, not upstream llvmorg-17.0.6 (llvm/CMakeLists.txt:12-18); Rust 1.78.0 moved to LLVM 18 (RELEASES.md:3007) | Bitcode only, and by version number only (inference, not shown): LLVM 17.x is an earlier version whose branch left main before the pin, and d1e9093d was not compared with 26eb4285 | DeveloperPolicy.rst:652; low modules only |

   - rustc nightly: rustc writes `captures(none)` attributes (abi.rs:54-57, :100; LangRef.md:1562) and `range(...)`
     attributes (abi.rs:493-506). Neither exists at 26eb4285, whose LangRef has only `nocapture` (LangRef.rst:1295).
     LLVM 23's LangRef also has samesign (:13141), getelementptr nusw (:12205), #dbg_value records (:7205), and zext
     nneg (:12519), none of which exist at 26eb4285. rustc's minimum external LLVM is 21 (RELEASES.md:433), so rustc
     cannot be built against Polygeist's LLVM.
   - NativeAOT-LLVM: WebAssembly is its only supported target (README.md:5). 18.1.3 is 11,510 commits ahead of
     26eb4285 and 0 behind (compare API). 18.1.3's LangRef has zext nneg (:11287) and or disjoint (:10073), both
     absent at 26eb4285. It produces no Linux x86-64 IR.
   - Other LLVM languages: the LLVM versions of Swift, Zig, and Julia were not read. Each is covered only if its LLVM
     is 26eb4285's or an earlier version, through bitcode, and an earlier release branch only by version number
     (inference).
   - LLVM's compatibility policy (LL:llvm/docs/DeveloperPolicy.rst:639-666; the same words at
     https://llvm.org/docs/DeveloperPolicy.html): "The textual format is not backwards compatible. We don't change it
     too often, but there are no specific promises." "The current LLVM version supports loading any bitcode since
     version 3.0." "Newer releases can ignore features from older releases, but they cannot miscompile them." Debug
     metadata is dropped in upgrades. Nothing promises that an older LLVM reads a newer one's IR, in text or bitcode.
7. P5.5's stashed pin (provisional, unaudited): TARGETS="cgeist polygeist-opt mlir-opt mlir-translate clang" (:63), so
   mlir-translate is among the targets; MLIR_TRANSLATE=build/bin/mlir-translate (:87); EXPECT_VERSION="LLVM version
   18.0.0git" (:98), which check_tools checks for mlir-translate (stash polygeist.sh:426-432, :560). LLVM_CMAKE_FLAGS
   (:55) set neither BUILD_SHARED_LIBS nor LLVM_LINK_LLVM_DYLIB; both default to OFF (LL:llvm/CMakeLists.txt:475-476,
   :782), so the tools link statically. This area needs no P5.5 pin change.

### Proposed pin (LLVM import)

- No pin of its own and no download. It uses `mlir-translate --import-llvm`, plus mlir-opt and polygeist-opt, from
  polygeist@77c04bb2 with llvm-project 26eb4285b56edd8c897642078d91f16ff0fd3472
  (https://github.com/llvm/llvm-project.git), as P5.5's provisional pin builds them. No publisher hash applies: it is a
  git commit, checked against Polygeist's
  gitlink (stash pin :15-21). The frontend tags every module `low`. Proposed, not installed.
- Its producer under the rule: that stack's clang (build/bin/clang, LLVM_CLANG_EXPECT "clang version 18.0.0", stash
  pin :88, :99).
- Hashes for the options outside the rule:
  - LLVM-23.1.3-Linux-X64.tar.xz: sha256 ca9fe385a5d9e2d88132dc0e23a00e9f87f38fa75f67207b5e457d208e71d013, and
    llvm-project-23.1.3.src.tar.xz: sha256 c44186a7762ed28954be72e5ff6df9808e0779d4f1bf014ecc4e7e211d31ee34. Both are
    GitHub release-asset digests from https://api.github.com/repos/llvm/llvm-project/releases (tag llvmorg-23.1.3,
    published 2026-10-06), each with a .sig beside it. The Linux binaries are built on ubuntu-22.04 "for older glibc"
    (release-binaries.yml:19-22), and the release build includes mlir (Release.cmake:33). Whether the archive holds
    bin/mlir-translate was not read.
  - rustc 1.77.2 x86_64-unknown-linux-gnu archives, xz_hash from
    https://static.rust-lang.org/dist/channel-rust-1.77.2.toml: rustc
    7fa3779e7693825728e99639c9b34b9475bfc0dc4a31f730d1f2fbb6764e183c; rust-std
    23119121fae4b7163b9c2ff5cbaad03d08f80d55da011a025d6b17d27489df7f; cargo
    b57b050ee48123e05021aff43f84da4b10d1d777dd350dd54fbc14448cd3d2d8.

### What each step needs, against the sandbox (LLVM import)

| Step | Needs | Sandbox | Met |
| --- | --- | --- | --- |
| Polygeist clang `-S -emit-llvm` | The tool and its resource dir under build/lib/clang/18 (stash pin :59-61, :79-82), /usr/include, and input and output in the build dir | The toolchains root is exposed read-only, and the build dir is the one writable directory (sandbox.py:199-204, :1585-1586); paths outside the hidden roots stay readable (:285-290) | Met |
| `mlir-translate --import-llvm` | The binary by path under $LASSI_TOOLCHAINS/polygeist@77c04bb2/build/bin, .ll or .bc in and .mlir out in the build dir, and no loader path (static link) | ENVIRONMENT_NAMES (:414-425) holds PATH, LANG, LC_ALL, and TMPDIR, and refuses LD_LIBRARY_PATH (:410-413); limits are 8192 MiB of memory and 2048 MiB of disk (:445-448) | Met; memory for one program PROJECTED ample, not measured |
| mlir-opt (`lift-cf-to-scf`, mem2reg, sroa) and polygeist-opt | Same as the row above | Same | Met |
| Native build and run of a low module (`--mlir-to-llvmir`, clang, host ld) | P5.7's CPU target; the program's environment is PATH=SANDBOX_PATH, LANG, TMPDIR, and OMP_NUM_THREADS (native.py:56-66) | As for P5.7 | Met; nothing new |
| Native link of a rustc-derived module (only if the owner adds that route) | Rust std from the rustc pin's sysroot, linked statically or found through an rpath | The toolchains root is exposed, and loader variables are refused | PROJECTED met with a static std or an rpath; not read |
| Run of a NativeAOT-LLVM module | A WebAssembly runtime | The native executor runs only host executables | Not met: the output is wasm32 |

### Budget (LLVM import)

| Item | Download | Installed | Wall time at 8 jobs | Peak memory | Basis |
| --- | --- | --- | --- | --- | --- |
| LLVM import (mlir-translate, mlir-opt, polygeist-opt in P5.5's tree) | 0 | 0 beyond P5.5's 19.15 GiB (PROJECTED, stash pin :100-106) | 0 beyond P5.5's job | P5.5's | already in P5.5's TARGETS (:63) |
| LLVM-to-core raiser, if the owner places it in P12 | 0 (source in the repo) | 0.1 to 0.5 GiB, PROJECTED | a few minutes, PROJECTED | PROJECTED: one static link against MLIR's libraries, the same class as mlir-opt's link | the P5.8 raiser row, a placeholder (plans/spikes/p5-mlir-pins.md:294) |
| Outside the rule (a): LLVM-23.1.3-Linux-X64.tar.xz | 2,016,524,848 bytes (read) | 5 to 10 GiB, PROJECTED: an allowance of 2.5 to 5 times the archive size, not read | minutes; no build | not applicable | releases API |
| Outside the rule (b): mlir-translate and mlir-opt built from llvm-project-23.1.3.src.tar.xz | 179,220,548 bytes (read) | 4 to 9 GiB build tree, PROJECTED | 1 to 3 h, PROJECTED; LLVM-scale, unattended only after P5.4's first LLVM-scale build | Not projected: no basis read, and plans/local-plan.md 3.2 gives no figure for the Polygeist stack, the same class of build, either. The job runs under OQ-056's 20G cap; P5.4's first LLVM-scale build measures a per-job figure from which its pools would be set | tt-mlir's LLVM tree of 9 to 14 GiB static or 5 to 9 shared (p5-mlir-pins.md:282), less lld; its build time of 15 to 45 min at 32 jobs (:330), times 4 |
| Outside the rule (c): rustc, rust-std, and cargo 1.77.2 | 63,263,120 + 25,510,324 + 8,396,232 = 97,169,676 bytes (HEAD Content-Length) | 0.3 to 0.6 GiB, PROJECTED (an allowance of 3 to 6 times) | minutes | not applicable | manifest |

## 6. cgeist's CUDA path (OQ-058) and xDSL (OQ-059)

Classification: the facts are factual (source at pinned commits, PyPI JSON, NVIDIA's manifest, GitHub API); the two
questions stay owner choices (OQ-058, OQ-059). Sources, all read 2026-10-17: Polygeist at the pin (`PG:`);
llvm-project at Polygeist's LLVM (`LL:`); xDSL v0.73.0 (`X73:`) and v0.23.0 (`X23:`); PyPI
https://pypi.org/pypi/xdsl/json; NVIDIA's manifest
https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.6.3.json; GitHub API commit
lists and compares on llvm/llvm-project; the stashed P5.5 pin.

### A. cgeist's CUDA path at 77c04bb2

1. Flags (`PG:tools/cgeist/driver.cc`):

   | Flag | Line | What it does |
   | --- | --- | --- |
   | `--cuda-lower` | :98 | "Add parallel loops around cuda" |
   | `--cpuify=<m>` | :218 | `distribute`, `distribute` plus `mincut`, `omp`, or `continuation` (driver.cc:833-838; `PG:lib/polygeist/Passes/ParallelLoopDistribute.cpp:2706-2743`; an unknown value reaches `llvm_unreachable`) |
   | `--emit-cuda` | :101 | GPU code; refused without `POLYGEIST_ENABLE_CUDA` (:632-637) |
   | `--cuda-gpu-arch` | :194 | Passed to clang |
   | `--cuda-path` | :197 | Passed to clang |
   | `-nocudainc` | :203 | Passed to clang |
   | `-nocudalib` | :206 | Passed to clang |
   | `-resource-dir` | :224 | Passed to clang |
   | `--gpu-kernel-structure-mode` | :116-125 | Kernel structure handling |
   | `-x`, `--function`, `-S` | :235, :211, :146 | Language, function filter, stop at MLIR |

   - The clang arguments are forwarded at `PG:tools/cgeist/Lib/clang-mlir.cc:5965-5987`. Without `-resource-dir`,
     cgeist finds `<bin>/../lib/clang/18` from its own path (clang-mlir.cc:6027-6031).
   - Upstream CPU-path tests run exactly this line (`PG:tools/cgeist/Test/Verification/cugen.cu:1`, `cugen2.cu:1`,
     `cudashmem2reg.cu:1`): `cgeist %s --cuda-gpu-arch=sm_60 -nocudalib -nocudainc %resourcedir --function=*
     --cuda-lower --cpuify="distribute" -S`.
2. How kernels become gpu ops and then scf.parallel.
   - Frontend: a `<<<>>>` launch becomes `gpu.launch` around a call to the kernel
     (`PG:tools/cgeist/Lib/CGCall.cc:256-349`, op built at :338). `threadIdx` and the other built-ins become
     `gpu.thread_id`, `gpu.block_id`, `gpu.block_dim`,
     and `gpu.grid_dim` (clang-mlir.cc:1977-2093). `__shared__` gets memory space 5 (clang-mlir.cc:944). With
     `--cuda-lower` and `--cpuify`, `cudaMalloc` and `cudaFree` are handled in the frontend (clang-mlir.cc:1888-1953).
   - `parallel-lower` (`PG:include/polygeist/Passes/Passes.td:87-97`; `PG:lib/polygeist/Passes/ParallelLower.cpp`)
     inlines the kernel call, then replaces each `gpu.launch` with two nested `scf.parallel` loops, grid then block
     (:409-480); replaces the id ops with the loop induction variables (:585-596); and turns `__syncthreads`
     (`nvvm.barrier0`) into `polygeist.barrier` (:598-602). It wraps in `polygeist.gpu_wrapper` only when emitting GPU
     code: `wrapParallelOps = EmitGPU` (driver.cc:648, :744-745).
   - `convert-cudart-to-cpu` (Passes.td:58-64; ParallelLower.cpp:741-875) replaces `cudaMemcpy`, `cudaMemcpyAsync`,
     and `cudaMemcpyToSymbol` with `llvm.intr.memcpy`; `cudaMemset` with memset; `cudaMalloc` and `cudaMallocHost`
     with malloc; `cudaFree` and `cudaFreeHost` with free; and `cudaDeviceSynchronize`, `cudaThreadSynchronize`,
     `cudaGetLastError`, and `cudaPeekAtLastError` with 0. It runs only when `--cpuify` is given (driver.cc:747-749).
     Every other cudart call is left as a call.
   - `cpuify` (Passes.td:105-113; ParallelLoopDistribute.cpp:2649-2745): `distribute` splits `scf.parallel` around
     each barrier; `omp` replaces barriers with `omp.barrier`; `continuation` uses barrier-removal-continuation
     (driver.cc:833-835).
   - Result with `-S`: `scf.parallel` with an explicit `scf.yield` terminator, plus memref and arith ops
     (cugen.cu:30-33, cudashmem2reg.cu:31-46). The `gpu` dialect stays only without `--cuda-lower` (stream.cu:1, :21).
     The module also
     holds llvm ops and Polygeist ops (`polygeist.memref2pointer` from the cudart conversion), outside the cpu-mlir
     row's dialect list. All 20 Polygeist ops have no custom assembly format
     (`PG:include/polygeist/PolygeistOps.td:21-255`; the only `assemblyFormat` line, :236, is commented out), so they
     print in generic form, for example
     `"polygeist.stream2token"` (stream.cu:20).
3. CPU lowering.
   - cgeist's own executable route (no `-S`): `convert-scf-to-openmp` (driver.cc:967-968), linking with `-fopenmp`
     (:988). It also writes `/tmp/intermediate%%%%%%%.ll` (:1202).
   - P5's `-S` route leaves the lowering to P5.7's CPU target: `scf.parallel` lowers to `scf.for` through
     `convert-scf-to-cf` (`LL:mlir/lib/Conversion/SCFToControlFlow/SCFToControlFlow.cpp:210`, :471-476), and Polygeist
     ops lower through polygeist-opt's `convert-polygeist-to-llvm` (Passes.td:246). polygeist-opt is in the stashed
     pin's TARGETS.
   - Upstream CI evidence (not a measurement): lit excludes only directories named `CUDA` when `POLYGEIST_ENABLE_CUDA`
     is 0 (`PG:tools/cgeist/Test/lit.cfg:25-27`; lit.site.cfg.in:11); CI configures Polygeist without that flag
     (`PG:.github/workflows/build.yml:107`) and runs `check-cgeist` (:114); so the three CPU-path tests, which carry no
     XFAIL or REQUIRES line, run in a build with CUDA off. That build has all LLVM targets (build.yml:91 names none).
4. CUDA versions clang at 26eb4285 accepts, against cuda@12.6.3.
   - Clang knows CUDA 7.0 to 12.1: 11.8 is the newest fully supported, 12.1 is partially supported, and anything newer
     is `NEW`, "Too new. Issue a warning, but allow using it" (`LL:clang/include/clang/Basic/Cuda.h:20-46`;
     `LL:clang/lib/Basic/Cuda.cpp:30-43`).
   - The driver maps a `CUDA_VERSION` of 12020 or more to `NEW` (`LL:clang/lib/Driver/ToolChains/Cuda.cpp:77-81`, read
     from `<path>/include/cuda.h`, :204-205), then warns "CUDA version is newer than the latest partially supported
     version 12.1" (Cuda.cpp:113-125; `LL:clang/lib/Driver/Driver.cpp:790-794`;
     `LL:clang/include/clang/Basic/DiagnosticDriverKinds.td:93-95`). This is a warning, not an error.
   - The arch check passes: sm_80 needs at least 11.0, and its maximum is `NEW` (`LL:clang/lib/Basic/Cuda.cpp:201-202`,
     :232-233; driver Cuda.cpp:869-879). The runtime wrapper checks only a lower bound of 7.0
     (`LL:clang/lib/Headers/__clang_cuda_runtime_wrapper.h:55-59`).
   - 12.6.3's `CUDA_VERSION` = 12060 is PROJECTED (basis: NVIDIA's major*1000 + minor*10 encoding); the cudart archive
     was not fetched.
   - Clang added CUDA 12.2 and 12.3 at 631c6e834cb0 (2023-12-11), 12.4 at 2f956a35edb6, 12.5 at 8407779bb725, and 12.6
     at 30a06e8022e9 (2024-10-14), all after the pin. Their file lists touch version tables, NVPTX builtins, and PTX
     features, not `clang/lib/Headers`; no later commit to the CUDA wrapper headers names 12.2 to 12.6 in its subject.
     Whether clang 18.0.0git parses 12.6's headers without error is unverified.
   - Layout fit: the detector needs `<path>/include`, `<path>/bin`, and `<path>/nvvm/libdevice` unless `-nocudalib`
     (driver Cuda.cpp:190-200). cuda.pin's layout gives bin and nvvm at the prefix and an include link into
     `targets/x86_64-linux`. cuda.h ships in cuda_cudart, and crt headers, libdevice, and ptxas in cuda_nvcc (a
     secondary source: XLA's hermetic BUILD templates at openxla/xla 416df8ba99880855e536c483df6e9dcadce82931,
     `third_party/gpus/cuda/hermetic/cuda_cudart.BUILD.tpl:23,71` and `cuda_nvcc.BUILD.tpl:25,65,90`).
5. POLYGEIST_ENABLE_CUDA and LLVM targets, against the stashed pin (`-DPOLYGEIST_ENABLE_CUDA=0`,
   `LLVM_TARGETS_TO_BUILD=host`).
   - `POLYGEIST_ENABLE_CUDA=1` gates only GPU emission: the `--emit-cuda` refusal (driver.cc:632-637); the GPU pipeline
     under `#if POLYGEIST_ENABLE_GPU` (:900-933); cubin serialization (:1028 onward); and `find_package(CUDA)`,
     `enable_language(CUDA)`, the cubin pass, and libcuda (`PG:lib/polygeist/Passes/CMakeLists.txt:83-105`). The README
     says to add it "to enable the CUDA backend" (`PG:README.md:59`).
   - The CPU route (driver.cc:738-861) has no `#if`, and its passes are built unconditionally
     (Passes/CMakeLists.txt:1-24).
   - Clang's CUDA resource headers are copied whatever the targets (`LL:clang/lib/Headers/CMakeLists.txt:53-63`,
     :279-289, :362-365); only the ARM and AArch64 and the RISC-V headers depend on targets (:369, :402).
     README.md:33 asks to drop `LLVM_TARGETS_TO_BUILD=host` only "to enable compilation to cuda".
   - No line read makes the `-S` CPU route need NVPTX. But upstream never runs it host-only (CI builds all targets),
     and clang's own CUDA Sema tests require `nvptx-registered-target`
     (`LL:clang/test/SemaCUDA/function-overload.cu:1-2`). So the host-only device-side parse is unverified.
   - Result: no P5.5 pin change is needed by the source as read.

### B. xDSL

- Newest release on 2026-10-17: xdsl 0.73.0, uploaded 2026-10-09T19:54:24Z (PyPI JSON). `requires_python` is
  `>=3.10`; the classifiers name 3.10 to 3.14, and CI tests 3.10 to 3.14 (`X73:.github/workflows/ci-mlir.yml:23`).
  Runtime dependencies: `immutabledict<4.3.2`, `ordered-set==4.1`, `typing-extensions<5,>=4.13`. The repository's
  uv.lock already holds typing-extensions 4.16.0, and its `requires-python` is `>=3.10,<3.11`. It supports batch 1's
  Python 3.10.
- The MLIR version xDSL targets: README.md:61-63, "validated against a specific MLIR version ... 22.1.2";
  `docs/guides/mlir_interoperation.md:27`, tested with LLVM 22.1.2. CI's mlir-opt is nixpkgs `llvmPackages_22`
  (`X73:flake.nix:22-28`; flake.lock pins nixpkgs 39ad350a0602fa0a58a544344e3e9187526ea45c, lastModified 2026-10-07),
  which is 22.1.8 (`nixpkgs@39ad350a:pkgs/development/compilers/llvm/default.nix:29`).
- Known mismatches with what LLVM 26eb4285 prints (known cases, not complete):
  - scf.parallel terminator: at the pin the terminator is `scf.yield`, and `scf.yield` may sit in `scf.parallel`
    (`LL:mlir/include/mlir/Dialect/SCF/IR/SCFOps.td:788-794`, :1168-1170); cgeist prints it so (cugen.cu:32). In
    0.73.0 the terminator is `ReduceOp` (`X73:xdsl/dialects/scf.py:508`), and `scf.yield`'s allowed parents exclude
    ParallelOp (:179). MLIR changed this at 10056c821a56 (2023-12-20), 8,203 commits after 26eb4285. This affects
    cgeist's CUDA path and any prange lowered to `scf.parallel`.
  - DLTI syntax: cgeist sets a `dlti.dl_spec` module attribute (clang-mlir.cc, the `setAttr("dlti." ...)` call near
    :6081). The pin prints it as `#dlti.dl_spec<#dlti.dl_entry<key, value>, ...>`
    (`LL:mlir/lib/Dialect/DLTI/DLTI.cpp:95-101`, :313-316); 0.73.0 parses only `key = value` entries
    (`X73:xdsl/dialects/dlti.py:111-121`).
  - The same attribute from LLVM 23, the reverse direction (read 2026-10-18 at llvmorg-23.1.3, the newest tag; the
    jammy-23 snapshot 0cd28174a1f6 is a later commit on release/23.x and was not read for this). LLVM 23 prints the
    spec's entries as `key = value` (mlir/lib/Dialect/DLTI/DLTI.cpp:119-127, :443-444), and its parser takes that
    form or `#dlti.dl_entry` (:433-441). The pin's parser takes only a comma-separated list of attributes
    (`LL:mlir/lib/Dialect/DLTI/DLTI.cpp:295-311`), so a `key = value` entry is refused there. flang 23 sets the
    attribute on every module its lowering creates (flang/lib/Lower/Bridge.cpp:7182, through
    `fir::support::setMLIRDataLayout`, flang/lib/Optimizer/Support/DataLayout.cpp:22-28) and on FIR it reads
    (flang/lib/Frontend/FrontendActions.cpp:236-237); the 23.1.3 importer sets it on every module it imports
    (mlir/lib/Target/LLVMIR/ModuleImport.cpp:1130-1142, called with no condition at :3631). So MLIR text from
    flang-23 (OQ-063 (e), OQ-064 Fortran (c)) or from mlir-translate 23.1.x (OQ-064 Rust (c)) is a known refusal at
    18.0.0git as printed. A step that strips the attribute, or rewrites it in `#dlti.dl_entry` form, would come
    first, and the rest of the text is still not shown to parse (known cases, not complete).
  - Affine: 0.73.0's `affine.if` and `affine.parallel` have no custom format (`X73:xdsl/dialects/affine.py:444`,
    :463), and there is no `affine.max` (the op list at :921-930).
  - func, arith, memref, and builtin were not checked op by op.
- Nearest older release: 0.23.0 (2024-10-11, `>=3.10`), the newest release whose `scf.yield` may sit in
  `scf.parallel` and whose implicit terminator is Yield (`X23:xdsl/dialects/scf.py:166`, :473). Its README targets MLIR
  98e674c9 (2023-11-28), 5,828 commits after 26eb4285 and 2,375 before 10056c82 (GitHub compare). Its limits: only
  `affine.apply` has a custom format (`X23:xdsl/dialects/affine.py:74`, :91), so `affine.for`, `affine.load`, and
  `affine.store` print generic only; there is no `dlti.py` (HTTP 404); `math` is only in
  `xdsl/dialects/experimental/math.py`. 0.24.0 (c624080a58) moved to MLIR d401987f and to `ReduceOp`. Releases from
  0.25.0 to 0.72.x were not read; that none moves back to Yield is inferred from 0.73.0's `ReduceOp`.
- Result: neither 0.73.0 nor 0.23.0 parses all of LLVM 18.0.0git's hub text; 0.24.0 was read for its scf change
  only. Releases before 0.23.0 were not read; that they fail too is inferred from 0.23.0's lack of affine custom
  formats and of dlti.

### C. Proposed pins (CUDA path and xDSL)

- CUDA path: no new pin. polygeist@77c04bb2 as P5.5 builds it, unchanged, and cuda@12.6.3 as `toolchains/cuda.pin`
  holds it. The manifest, re-fetched: 49,142 bytes, sha256
  9c598598457a6463eb92889080c16b2b9dc04150e501b8bfc1536d403ba70aaf, Last-Modified 2024-11-20, identical to the pin. Its
  four archives, with sizes and sha256 from that manifest:

  | Archive | Version | Size (bytes) | sha256 |
  | --- | --- | --- | --- |
  | cuda_nvcc | 12.6.85 | 49,996,208 | 840deff234d9bef20d6856439c49881cb4f29423b214f9ecd2fa59b7ac323817 |
  | cuda_cudart | 12.6.77 | 1,126,072 | f74689258a60fd9c5bdfa7679458527a55e22442691ba678dcfaeffbf4391ef9 |
  | cuda_cccl | 12.6.77 | 934,952 | 9c3145ef01f73e50c0f5fcf923f0899c847f487c529817daa8f8b1a3ecf20925 |
  | cuda_cuobjdump | 12.6.77 | 219,068 | 54d0edf14284249baf8e2db1c26c71d079ad7afd1f41779e22f239ff9e8bf60f |

- Proposed raise line for P12.13: `cgeist <src>.cu --cuda-path=$LASSI_TOOLCHAINS/cuda@12.6.3 --cuda-gpu-arch=sm_80
  -nocudalib --cuda-lower --cpuify=distribute --function=* -S -o <out>.mlir`. sm_80 matches LASSI's CUDA builds;
  `-nocudalib` skips libdevice, which a `-S` parse does not use.
- xDSL: no pin under the planning rule (it cannot be served). Candidates for OQ-059, both from PyPI JSON:

  | Release | Wheel | Size (bytes) | sha256 | Tag commit |
  | --- | --- | --- | --- | --- |
  | 0.73.0 | `xdsl-0.73.0-py3-none-any.whl` | 4,579,150 | 6e32458c1ad31a65ea047d9696fe65b998943b211f0c1eedabf17e1e2ecc5b5c | a2544953 |
  | 0.23.0 | `xdsl-0.23.0-py3-none-any.whl` | 765,444 | 3b3c24be081964ab35b711aeff23f481668b3a123f0054482d1464d8175f30cc | d1f33aac |

  The 0.73.0 sdist's sha256 is a65edcca3a649a315e4826c43343fcbff070b60e0099b74de18a17a89f5ffdb2.

### D. Needs against the sandbox (CUDA path)

| Step | Need | What the sandbox gives | Met? |
| --- | --- | --- | --- |
| Raise (compile sandbox) | cgeist, and clang's resource directory `build/lib/clang/18` (cuda_wrappers, `__clang_cuda_*.h`) | Toolchains root exposed read-only (sandbox.py:199-205) | Met |
| Raise | CUDA install at `--cuda-path` | Under the toolchains root. The flag is required: the other places clang looks are ptxas on PATH (`SANDBOX_PATH` is the system dirs, sandbox.py:384) and `/usr/local/cuda*` and `/usr/lib/cuda` (driver Cuda.cpp:152-183). No variable is needed, and none is on ENVIRONMENT_NAMES (sandbox.py:405-424) | Met with the flag |
| Raise | Host libstdc++ and glibc headers (batch 1's g++-12) | Host is read-only, not hidden | Met; which GCC clang 18 picks is PROJECTED |
| Raise | No temp file with `-S`; private TMPDIR | `COMPILE_TMPDIR` (sandbox.py:438) | Met |
| Raise | No GPU | Private `/dev` | Met; the CPU route opens no device |
| Verify (P5.6) | Polygeist ops parse only in polygeist-opt; they print generic | polygeist-opt is in TARGETS | Met by the pin. Two open points for P5.6: cpu-mlir's verifier must use polygeist-opt or lower these ops first, and `NO_CUSTOM_FORM` must list the Polygeist ops, or `check_ir` refuses the modules as generic (Agent Rule 11). This applies to C too |
| Lower and build (P5.7) | `convert-scf-to-cf` (sequential); `convert-polygeist-to-llvm` | Toolchains root | Met. Not met for an OpenMP route: batch 1 names no libomp package (tools/local/install-batch1.sh:87-107); whether clang-17's dependencies pull one in was not checked |
| Native run | The program uses only malloc, free, memcpy, memset | The program environment is PATH, LANG, TMPDIR, `OMP_NUM_THREADS`; no HOME and no loader variables (native.py:51-62) | Met when every cudart call is one of the replaced set (ParallelLower.cpp:741-875). Not met for any other cudart call: it stays an external symbol, and libcudart would need libcuda and GPU device nodes, which the sandbox hides and P12 must not open; count these as refusals |
| cgeist without `-S` | Writes `/tmp/intermediate*.ll` (driver.cc:1202) | Private 64m `/tmp` (sandbox.py:772) | Met inside the sandbox; never run it outside |

### E. Budget (CUDA path and xDSL)

| Item | Download | Installed | Wall time at 8 jobs | Peak memory |
| --- | --- | --- | --- | --- |
| cuda@12.6.3, four archives (shared with P17.12's batch) | 52,276,300 bytes (read: the manifest sizes summed) | 233,316 KiB [MEASURED 2026-09-23, alpha01, rx 20260923-212629-exec-290a, per cuda.pin]; the same on the workstation is PROJECTED | PROJECTED minutes (about 50 MB plus xz extraction, no build) | n/a |
| Polygeist for the CUDA path | 0 | 0 beyond P5.5's 11.61 to 19.15 GiB (PROJECTED, plans/local-plan.md 3.2) | 0 | n/a |
| Conditional: Polygeist rebuild with NVPTX | no new sources | in place, PROJECTED a small addition (not read); as a second tree, up to another 11.61 to 19.15 GiB (PROJECTED) | PROJECTED 1 to 2.5 h (local-plan 3.2); a target change rewrites configuration headers that most of LLVM includes, so treat it as a full rebuild | Not projected: no basis read, and plans/local-plan.md 3.2's Polygeist row gives no figure either (it names the static clang and mlir-opt links with GNU ld as the largest risk); the job runs under OQ-056's 20G cap |
| xDSL 0.73.0 extra (only under OQ-059 (b) or (c)) | 4,579,150 bytes wheel plus immutabledict and ordered-set (sizes not read) | PROJECTED 15 to 25 MiB (a pure-Python wheel of 4.4 MiB; unpack ratio not read) | PROJECTED under a minute | n/a |
| One cgeist CUDA raise | n/a | n/a | PROJECTED seconds | PROJECTED within `COMPILE_MEMORY_MB` 8192 and `COMPILE_DISK_MB` 2048 (sandbox.py:445, :448). Basis: P0's exploratory nvcc compile of a CUDA app peaked at 216 MiB RSS (sandbox.py:439-442; exploratory, not a measurement) |

### Confidence (CUDA path and xDSL)

| Claim | Confidence |
| --- | --- |
| Flags, pipeline, and that CUDA is not gated | High |
| Host target is enough | Medium |
| CUDA 12.6 headers parse at the pin | Unknown (not read) |
| xDSL version facts and the three named mismatches | High |
| The mismatch list is complete | No (known cases, not complete) |

## Where the areas agree, overlap, or disagree

- rustc's LLVM IR (Rust R12; LLVM import finding 6): both find it unreadable at 26eb4285. The Rust area names the
  constructs from rustc's codegen source (`getelementptr inbounds nuw` on every array index, `trunc nuw` on every bool
  immediate, `captures(none)` under -O); the LLVM import area names them from LLVM 23's LangRef (`captures(none)`,
  `range(...)`, samesign, getelementptr nusw, #dbg_value records, zext nneg). No conflict; `captures(none)` is in
  both.
- Older-rustc options differ, without conflict: the Rust area's option is an LLVM-18-era nightly, "not shown readable"
  (LLVM 18.1.8's lexer has `disjoint` and `nneg`, which 26eb4285 lacks); the LLVM import area's is rustc 1.77.2
  stable on LLVM 17.0.6 (Rust's fork d1e9093d), an earlier LLVM version whose branch left main before the reader's
  commit (upstream 17.0.6 carries 405 commits the pin lacks; Flang finding 5), so the policy covers its bitcode by
  version number only (inference, not shown). Both give low modules only. The candidate below carries both.
- A third LLVM's mlir-translate: the Rust area's option (c) (rustc's LLVM 23.1) and the LLVM import area's option (b)
  (23.1.3 from the binary or from source) are one option; only the LLVM import area read 23.1.3's hashes and sizes.
- NativeAOT-LLVM (.NET finding 11; LLVM import finding 6): the same facts (WebAssembly only; LLVM 18.1.3). The .NET
  area offers "defer to a later phase" as (c), and the LLVM import area as (b); one option.
- The rpath (Rust R6) follows the install route R8 recommends: under route B, install.sh puts `lib/*` under
  `$P/lib` (install-template.sh:565-567, :808), so the MIR driver's rpath is `$P/lib`; route A's would be
  `$P/rustup/toolchains/nightly-2026-10-17-x86_64-unknown-linux-gnu/lib`. P12.8 uses the one for the route P12.5
  installs.
- P5.5's stash: the task brief's form `git show stash@{N}:toolchains/polygeist.pin` finds nothing, since the files are
  untracked in the stash; every area used `stash@{0}^3`.
- P5.5 pin changes: only the Flang area finds one (LLVMFrontendOpenACC). Rust, .NET, Python and C, LLVM import, and
  the CUDA path each find none for their own needs; the CUDA area adds a conditional one (NVPTX).
- Thread bounds: Rust (RAYON_NUM_THREADS) and .NET (DOTNET_PROCESSOR_COUNT) each name an unmet thread bound; Python's
  OpenBLAS is met through OMP_NUM_THREADS. The Rust area places its change in P12.6 as [DESIGN]; the .NET area places
  its change in "the C# native-path task", which in the plan is P12.6 (native toolchains and the native executor for
  C, Rust, C#, and Python, where a sandbox change is a named [DESIGN] change).
- P12.5's .NET install: the plan's P12.5 says "the .NET SDK by dotnet-install.sh into dotnet@<ver>"; the .NET area
  recommends a hash-checked `tar -xzf` instead, since dotnet-install.sh checks only the size (finding 7), with the
  script and `--keep-zip` as the alternative.
- Line citations of the sandbox differ slightly (ENVIRONMENT_NAMES as sandbox.py:405-425, :414-425, and :405-424; the
  native program environment as native.py:51-62, :54-66, and :56-66). At the base commit the ENVIRONMENT_NAMES comment
  is sandbox.py:405-413 and its definition :414-425; `_program_environment` is native.py:59-66, after its comment at
  :55. The citations point at the same code.
- Polygeist's budget: the LLVM import area cites "19.15 GiB (stash pin :100-106)" and the CUDA area "11.61 to 19.15
  GiB (plans/local-plan.md 3.2)": the two ends of one projection.
- The Fortran row of Language Frontends ("lowered to core dialects ... Available upstream") does not hold at the pin
  (Flang finding 3). P12.1 corrects it to what holds at the pin (2026-10-17); P12.10 updates its status from the code.

## Proposed pins

Proposed, not installed. Install locations follow the plan's Pins constraint: everything outside apt under
`$LASSI_TOOLCHAINS/<name>@<pin>`, host compilers and apt packages at their system paths.

| Toolchain | Pin | Source URL | Publisher hash and its kind | Install location |
| --- | --- | --- | --- | --- |
| rustc (MIR frontend, Rust builds) | nightly-2026-10-17: rustc 1.101.0-nightly (5d8fec066 2026-10-16), LLVM 23.1.4; rustc, cargo (3960fa69b), rust-std, rustc-dev, llvm-tools-preview, rust-src | https://static.rust-lang.org/dist/2026-10-17/channel-rust-nightly.toml and the six archives beside it | sha256 per archive (manifest xz_hash, matched by each `.sha256` file); manifest sha256 5e87d004b989db293416a5595c0564bef052dca8d773f2c95cd165ee2ddce6db (`.sha256`; `.asc` published) | `$LASSI_TOOLCHAINS/rustc@nightly-2026-10-17` (route B: install.sh `--prefix`; route A: `rustup/toolchains/nightly-2026-10-17-x86_64-unknown-linux-gnu` under it) |
| rustup-init (route A only) | 1.29.1 | https://static.rust-lang.org/rustup/archive/1.29.1/x86_64-unknown-linux-gnu/rustup-init | sha256 dda7234360b7f578ca8b0ddcb80145646fa61a67c1720a5abc7051b35c9fcb71 (`.sha256`) | used at install time only |
| rayon and its closure (vendored) | rayon 1.12.0, rayon-core 1.13.0, either 1.19.0, crossbeam-deque 0.8.8, crossbeam-epoch 0.9.21, crossbeam-utils 0.8.23 | https://static.crates.io/crates/<name>/<name>-<version>.crate | sha256 per crate (crates.io index cksum; Rust pin table) | `$LASSI_TOOLCHAINS/rustc@nightly-2026-10-17/vendor` |
| dotnet (C# frontend, C# builds and runs) | SDK 10.0.402, runtime 10.0.13, Roslyn 5.9.0 | https://builds.dotnet.microsoft.com/dotnet/Sdk/10.0.402/dotnet-sdk-10.0.402-linux-x64.tar.gz | SHA-512 1904a98d...87729c2 in full in the .NET pin table (releases.json; 10.0.13-sha.txt:275); P12.5 adds sha256 | `$LASSI_TOOLCHAINS/dotnet@10.0.402` |
| Offline NuGet feed | Microsoft.CodeAnalysis.CSharp 5.9.0, Microsoft.CodeAnalysis.Common 5.9.0, Microsoft.CodeAnalysis.Analyzers 5.9.0-1.26328.17 | https://api.nuget.org/v3-flatcontainer/<id>/<version>/<id>.<version>.nupkg | SHA-512 per package (nuget.org catalog leaves, base64; hex decoded in the .NET pin table); P12.5 adds sha256 | a feed directory under `$LASSI_TOOLCHAINS/dotnet@10.0.402` |
| csharp-roslyn (the frontend tool) | source tree hash of `frontends/csharp_roslyn/`, built by dotnet@10.0.402 | none (built here) | none | `$LASSI_TOOLCHAINS/csharp-roslyn@<hash>` |
| python (Python runs) | 3.10.12-numpy2.2.6: a pip-less venv of /usr/bin/python3.10 (3.10.12-1~22.04.18) with numpy 2.2.6 | https://files.pythonhosted.org/packages/b4/63/3de6a34ad7ad6646ac7d2f55ebc6ad439dbbf9c4370017c50cf403fb19b5/numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl | sha256 fc7b73d02efb0e18c000e9ad8b83480dfcd5dfd11065997ed4c6747470ae8915 (PyPI JSON; the v2.2.6 release page) | `$LASSI_TOOLCHAINS/python@3.10.12-numpy2.2.6` |
| gcc-c (C builds) | gcc-12 12.3.0-1ubuntu1~22.04.3, brought by batch 1's g++-12 | http://archive.ubuntu.com/ubuntu/dists/jammy-updates/main/binary-amd64/Packages.gz | .deb sha256 f1ce0ee798bd9c3d30c5602962521fc7faf309795fc6596de88df1cfc4732d55 (Packages index; apt checks it against the signed index) | system path `/usr/bin/gcc-12` |
| flang | llvm-project 26eb4285b56edd8c897642078d91f16ff0fd3472 (LLVM 18.0.0git), standalone against P5.5's build tree | https://github.com/llvm/llvm-project.git (P5.5's checkout) | none published for a commit; commit id plus flang's git tree id 61b4657087e745864f78103478cdfaacb9ddc4dc (GitHub git trees API) | `$LASSI_TOOLCHAINS/flang@26eb4285` |
| LLVM import | no pin of its own: mlir-translate, mlir-opt, polygeist-opt of polygeist@77c04bb2 | as P5.5 | none (git commit, checked against Polygeist's gitlink, stash pin :15-21) | `$LASSI_TOOLCHAINS/polygeist@77c04bb2/build/bin` |
| cgeist's CUDA path | no pin of its own: polygeist@77c04bb2 unchanged, cuda@12.6.3's headers by `--cuda-path` | https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.6.3.json | manifest sha256 9c598598457a6463eb92889080c16b2b9dc04150e501b8bfc1536d403ba70aaf; per-archive sha256 in the CUDA table | `$LASSI_TOOLCHAINS/cuda@12.6.3` |
| xDSL | none under the rule; candidates 0.73.0 and 0.23.0 go to OQ-059 | https://pypi.org/pypi/xdsl/json | wheel sha256 in the xDSL table (PyPI JSON) | an optional extra only under OQ-059 (b) or (c) |

EXPECT_VERSION values: rustc `rustc 1.101.0-nightly (5d8fec066 2026-10-16)` (PROJECTED from the manifest); dotnet
`10.0.402`; python `Python 3.10.12` (PROJECTED from the P5.5 stash's PYTHON_EXPECT); gcc-c
`(Ubuntu 12.3.0-1ubuntu1~22.04.3) 12.3.0` (PROJECTED); flang `flang-new version 18.0.0` (PROJECTED from
LL:clang/lib/Driver/Driver.cpp:1947-1949 and LL:clang/CMakeLists.txt:293-294).

## What builds and runs need against the sandbox

Sandbox facts the areas used (sandbox.py and native.py at the base commit): the compile environment is exactly the
ENVIRONMENT_NAMES names (sandbox.py:414-425: PATH, LANG, LC_ALL, TMPDIR, OMP_NUM_THREADS, the pins' prefix variables,
the TT_METAL names, and the GPU visibility names; lassi/toolchains/pins.py:47 adds only NVHPC_CUDA_HOME), with HOME,
LD_LIBRARY_PATH, and LD_PRELOAD never on it (:405-413, 1597-1602). The sandbox uid is 0 in its user namespace
(:112-114) on a read-only host; the toolchains root is mounted back read-only at its own path (:739-748); the build
dir is the one writable directory (:1585-1586), capped at 2048 MiB and 4096 inodes with 8192 MiB of memory and 2
CPUs (:443-452); /tmp is a private 64m tmpfs (:772); there is no network. Native runs get PATH, LANG, TMPDIR, and
OMP_NUM_THREADS=<Limits.cpus> only (native.py:54-66), argv[0] must be an artifact in the runs root (:250-261), and
each process has a CPU-time cap of `cpus x (wall + 3)` seconds (sandbox.py:115-121).

Needs the sandbox or the current components do not meet (known cases, not complete):

| Area | Need | Cause | Proposed handling |
| --- | --- | --- | --- |
| Rust | A defined CARGO_HOME in compiles | Not in ENVIRONMENT_NAMES; HOME refused; cargo's home falls to the passwd entry (likely /root/.cargo) and a host `$CARGO_HOME/config.toml` could leak into a build (R7) | P12.6 [DESIGN]: add CARGO_HOME, set to an empty `@`-prefixed directory under the build dir or an empty read-only directory in the pin's prefix. A writable CARGO_HOME is not required |
| Rust | A rayon thread bound in native runs | RAYON_NUM_THREADS is not in ENVIRONMENT_NAMES or `_program_environment`; 16 threads under `Limits.cpus` 1 or 2 reach the CPU-time cap (R11) | P12.6 [DESIGN]: add RAYON_NUM_THREADS=<Limits.cpus> beside OMP_NUM_THREADS |
| .NET | The SDK command line (`dotnet build`, `dotnet publish`) in compiles | No DOTNET_* name and no NUGET_PACKAGES on the allowlist; the CLI home becomes /root on a read-only mount, so the first-run sentinel write is expected to fail (inferred) | Compile with `csc.dll` through the pinned host by path (met); no sandbox change |
| .NET | A framework-dependent apphost run | Needs DOTNET_ROOT, which is not allowed, or a system install location | Run a script artifact that execs the pinned `dotnet` on `prog.dll` (met); `AppHostRelativeDotNet` would also work |
| .NET | A thread bound in native runs | No CPU affinity or quota (OQ-011); DOTNET_PROCESSOR_COUNT is not allowed | Option in P12.6: a Sandbox [DESIGN] edit adding DOTNET_PROCESSOR_COUNT, or a build-time `System.Threading.ThreadPool.MaxThreads` (fixed, not from Limits.cpus) |
| .NET | NativeAOT-LLVM for any route | wasm32 output, a third LLVM, a prerelease feed, Emscripten or the WASI SDK | Not served (owner-queue candidate 5) |
| .NET (outside the sandbox) | The frontend's build must not extract packages into the host profile's NUGET_PACKAGES | host-profile.sh:128-129 exports NUGET_PACKAGES and DOTNET_CLI_HOME into the scratch root; the environment variable wins over nuget.config | Set both under the build dir; `--disable-build-servers`; `NUGET_CERT_REVOCATION_MODE=offline` |
| Python | A way to start the interpreter | The executor runs `[artifact, *inputs]` with no interpreter slot; no Python toolchain exists in lassi/toolchains | An executable launcher artifact (`#!<prefix>/bin/python3 -B`, or a /bin/sh wrapper); no sandbox change |
| Python | No bytecode in the workdir | PYTHONDONTWRITEBYTECODE is not allowed; an import would write `__pycache__` into `output_files` | `-B` on the launcher; numpy's bytecode compiled at install |
| C | C sources accepted | gcc-native's SOURCE_SUFFIXES are C++ only | New `gcc-c-native` class with `-lm` after the sources |
| Flang (apt route only) | A shared runtime at run time | The driver adds `-L` but no rpath; native runs take no LD_LIBRARY_PATH | An rpath or a [DESIGN] sandbox change; none on the source route |
| LLVM import | Running a NativeAOT-LLVM module | wasm32 output; the native executor runs host executables only | Not served |
| CUDA path | Any cudart call outside the replaced set | It stays an external symbol; libcudart needs libcuda and GPU nodes, which the sandbox hides | Counted as a refusal |
| CUDA path | An OpenMP lowering route | Batch 1 names no libomp package (not checked whether clang-17's dependencies bring one) | Use the sequential `convert-scf-to-cf` route |
| CUDA path and C (P5.6) | Verifying modules that hold Polygeist ops | They print generic only and parse only in polygeist-opt | P5.6: verify with polygeist-opt or lower them first, and list them in `NO_CUSTOM_FORM`, else `check_ir` refuses them (Agent Rule 11) |

Needs met only if the build config sets them: the Rust linker (`/usr/bin/gcc-12` by path, since `cc` may be absent and
its presence in LASSI-Ubuntu-22.04 is unmeasured), `build.jobs = 2`, the vendored source and offline mode in
`<build>/.cargo/config.toml`, the MIR driver's absolute rpath (`$P/lib` under route B), and cuda's `--cuda-path`.

## Budget

PROJECTED unless a figure says read or [MEASURED]. "Installed" is in the scratch root unless it says otherwise; the
scratch root's du was not read for this spike, and P12.5 checks du and C:'s free space before its batch. Sums are
exact sums of the GiB figures shown.

| Area | Item | Download | Installed | Wall time at 8 jobs | Peak memory | Basis |
| --- | --- | --- | --- | --- | --- | --- |
| Rust | Six nightly archives, route B | 298,228,180 B, 284.4 MiB (read) | 1.50 GiB tar contents (read); about 1.8 GiB kept (archives and install), peak about 3.3 GiB during staging, PROJECTED | 1 to 5 min at an assumed 1 to 5 MB/s | under 1 GiB (an allowance) | Content-Length; xz indexes |
| Rust | rayon closure; MIR driver build; per-program compiles | 398,787 B (read) | 2 to 4 MiB; 50 to 300 MiB; 50 to 150 MiB per build dir | seconds; 1 to 3 min; 15 to 60 s at `-j 2` | -; 1 to 3 GiB; under 1.5 GiB | Rust budget; the MIR driver and per-program figures are allowances with no basis read |
| Rust | Area total | about 0.28 GiB | about 1.9 to 2.2 GiB | - | no LLVM-scale build | Rust budget |
| .NET | SDK 10.0.402 | 240,279,509 B (read) | 616.5 MiB of tar (read); du 0.62 to 0.70 GiB | 1 to 5 min | under 0.5 GiB | Content-Length; gzip trailer |
| .NET | Feed, restore, publish, per-program csc | 19,273,255 B (read) | feed 19.3 MB (read); packages about 78 MiB (read); publish 12 to 25 MiB | seconds to 2 min; 1 to 5 s per program | 1 to 2 GiB; 0.2 to 0.5 GiB | .NET budget; the restore, publish, and csc times and memory are allowances with no basis read |
| .NET | Area total | about 0.25 GiB | 0.73 to 0.82 GiB, the sums of its rows' ends; 1.04 GiB at the upper end with the kept SDK archive (0.22 GiB) | - | no LLVM-scale build | .NET budget |
| Python and C | numpy wheel, venv | 16,801,050 B (read) | about 0.08 GiB with the kept wheel | under 1 min | under 1 GiB (an allowance) | Python budget |
| Python and C | gcc-12, cpp-12, libgcc-12-dev | 0 for P12 (batch 1) | 0 in the scratch root; 119,072 KiB in the distro's /usr (read, Installed-Size) | 0 | n/a | jammy-updates index |
| Flang | P5.5 TARGETS + LLVMFrontendOpenACC, or its `ninja` target | 0 | under 0.01 GiB | under 1 min | negligible | Flang budget |
| Flang | (a) standalone build, in place | 0 | 2 to 5 GiB | 27 min to 1.8 h at the proposed 3 compiles and 1 link (12 to 51 min at 8 compiles) | 5 to 17 GiB at 3 compiles and 1 link; 6 to 20 GiB at 4 and 1, at the 20G cap; 10 to 32 GiB at 8 compiles | Flang basis 1 |
| LLVM import | Uses P5.5's tree | 0 | 0 | 0 | P5.5's | already in P5.5's TARGETS |
| CUDA path | Polygeist unchanged | 0 | 0 | 0 | n/a | CUDA budget |

Totals, GiB in the scratch root:

| Total | Low | High | Basis |
| --- | --- | --- | --- |
| P12 under the rule: Rust 1.9 to 2.2, .NET 0.73 to 1.04 (its rows' lower ends without the archive; their upper ends plus the 0.22 GiB SDK archive), Python and C 0.08, flang (a) 2 to 5, the OpenACC target 0 to 0.01, LLVM import 0 | 4.71 | 8.33 | rows above |
| Conditional items: cuda@12.6.3 0 to 0.23 (233,316 KiB [MEASURED 2026-09-23, alpha01, rx 20260923-212629-exec-290a, per cuda.pin]; 0 if P17.12's batch installs it), an LLVM-to-core raiser 0 to 0.5 (if the owner places it in P12), xDSL 0 to 0.03 (OQ-059 (b) or (c)) | 0 | 0.76 | CUDA and LLVM import budgets |
| P12 with the conditional items | 4.71 | 9.09 | sum |
| P5's toolchains | 28 | 46 | PROJECTED, plans/p5-ir.md, Constraints (plans/local-plan.md 3.2) |
| P5 and P12 together | 32.71 | 55.09 | sum; under the 145G stop line |
| P12 with flang's fallback (b), an in-tree build of its own (12 to 22 GiB in place of (a); no OpenACC target) | 14.71 | 26.08 | with the conditional items at the upper end |
| P5 and P12 with flang's fallback | 42.71 | 72.08 | sum; under the 145G stop line |

Transient: Rust's route B staging peaks about 1.5 GiB above its kept 1.8 GiB during P12.5's install only.

Outside the rule, additive only if the owner picks them (GiB in the scratch root unless noted): the LLVM 23.1.3 Linux
binary 5 to 10, plus 1.88 for its kept archive (2,016,524,848 B, read); or mlir-translate and mlir-opt from 23.1.3's
source, a 4 to 9 GiB build tree plus 0.17 for its archive (179,220,548 B, read), 1 to 3 h, LLVM-scale; rustc 1.77.2,
0.3 to 0.6 plus 0.10 for its archives (97,169,676 B, read); a second Polygeist tree for NVPTX, up to 11.61 to 19.15, 1
to 2.5 h; apt flang-17 or flang-20, 846,213 or 938,626 KiB (read; flang-20 with libmlir-20) in the distro's /usr, not
the scratch root. P5's upper end, P12's upper end with flang's fallback and every conditional item, the LLVM 23.1.3
binary with its archive, rustc 1.77.2 with its archives, and a second Polygeist tree together come to 103.81 GiB
(46 + 26.08 + 10 + 1.88 + 0.6 + 0.10 + 19.15), still under the stop line.

Memory: the one LLVM-scale build under the rule is flang's. At 8 compile jobs its PROJECTED peak, 10 to 32 GiB, passes
OQ-056's 20G job cap at the upper end, and pools of 4 compiles and 1 link (6 to 20 GiB) put the upper end at the cap
with no margin, so the pin proposes `LLVM_PARALLEL_COMPILE_JOBS=3` and `LLVM_PARALLEL_LINK_JOBS=1` (5 to 17 GiB, Flang
basis 1). 17 GiB is still above WSL's default 15.6 GiB with 4 GiB of swap [MEASURED 2026-10-15, workstation;
plans/spikes/local-host.md:139], and a long job runs with swap off, so the build needs the owner's memory=24GB in
.wslconfig (OQ-056) in place. Under the plan's Constraints it runs unattended only after P5.4's first LLVM-scale
build, else it waits for the owner's yes. Confidence in flang's budget is low. Outside the rule, the peaks of a 23.1.3
source build and of a Polygeist rebuild for NVPTX are not projected: no basis was read, and plans/local-plan.md 3.2
gives no figure for the Polygeist stack either.

## OQ-058 facts (for the Response)

Response (task P12.1, 2026-10-17): cgeist's CPU route for CUDA builds from P5.5's stashed pin as it stands. With
`--cuda-lower --cpuify=distribute -S`, kernels raise to `gpu.launch` and lower to nested `scf.parallel`, barriers are
distributed, and the replaced cudart calls become malloc, free, memcpy, and memset. None of it is gated by
`POLYGEIST_ENABLE_CUDA`, and the source names no LLVM target beyond host for it; only `--emit-cuda` needs those.
cuda@12.6.3's headers are accepted with a "newer than supported" warning (clang at 26eb4285 knows CUDA up to 12.1). So
the recommendation resolves to (a). Three runtime unknowns remain for P12.13's first run: (1) the host-only
device-side parse, which upstream CI never runs (it builds all targets); (2) whether 12.6's headers parse on clang
18.0.0git; (3) the generic-form Polygeist ops, which need polygeist-opt and `NO_CUSTOM_FORM` (P5.6). A cudart call
outside the replaced set stays external and is counted as a refusal; GPU emission (`--emit-cuda`) is not needed.
Proposed raise line: `cgeist <src>.cu --cuda-path=$LASSI_TOOLCHAINS/cuda@12.6.3 --cuda-gpu-arch=sm_80 -nocudalib
--cuda-lower --cpuify=distribute --function=* -S -o <out>.mlir`. All of this was read from source; nothing was built
or run. Evidence: plans/spikes/p12-frontend-pins.md (section 6).

## OQ-059 facts (for the Response)

Response (task P12.1, 2026-10-17): xDSL 0.73.0 (2026-10-09; Python >=3.10, so 3.10 works) targets MLIR 22.1.2. It
refuses what LLVM 26eb4285 prints for `scf.parallel` (an `scf.yield` terminator, which MLIR changed at
10056c821a56, 2023-12-20) and for `dlti.dl_spec` (the `#dlti.dl_entry` form), lacks custom forms for `affine.if` and
`affine.parallel`, and has no `affine.max`. 0.23.0 (2024-10-11), the newest release whose scf matches (0.24.0 moved
to `ReduceOp`; 0.25.0 to 0.72.x were not read), prints most affine ops generic only and has no dlti. Neither parses
the pinned hub text, so a pin of either is outside P12's rule; releases before 0.23.0 were not read, and that they
fail too is inferred from 0.23.0's lack of affine custom formats and of dlti. This supports (a). Known cases, not
complete: func, arith, memref, and builtin were not checked op by op. Wheels, if the owner pins one:
xdsl-0.73.0-py3-none-any.whl, 4,579,150 bytes, sha256
6e32458c1ad31a65ea047d9696fe65b998943b211f0c1eedabf17e1e2ecc5b5c; xdsl-0.23.0-py3-none-any.whl, 765,444 bytes, sha256
3b3c24be081964ab35b711aeff23f481668b3a123f0054482d1464d8175f30cc (PyPI JSON). Evidence:
plans/spikes/p12-frontend-pins.md (section 6).

## Owner-queue candidates

P5.5 pin changes:

1. TARGETS += LLVMFrontendOpenACC (Flang finding 2). Cause: flang's parser includes acc_gen's generated headers and
   its libraries link LLVMFrontendOpenACC, which P5.5's targets do not build, and a standalone build cannot make
   them. Timing: stash polygeist.sh records TARGETS and refuses a tree of another record (:94-127), so a TARGETS edit
   must land before P5.5's first install; option (d) need not. Options: (a) add it to TARGETS now: cheap, and P5.5 is
   still unaudited; (b) flang builds its own in-tree tree at 26eb4285: no P5.5 change, but +12 to 22 GiB and about
   1.5 to 4.3 h, LLVM-scale, in two build steps (clang and MLIR at 8 jobs, then flang at 3 compiles and 1 link; Flang
   basis 2); (c) add flang to P5.5's projects and targets: one tree, but it changes the cgeist stack's record, and
   since a pool covers P5.5's whole build and its script builds in one call, flang compiles at 8 jobs (a peak of 10 to
   32 GiB, past the 20G cap), or the whole build slows to 3 compiles, or the script gains a second build call
   (basis 2); (d) flang's own script runs `ninja LLVMFrontendOpenACC` in P5.5's build tree before its configure, as the
   stash pin leaves clangTooling to P5.8, "which builds it later in the same build tree" (stash polygeist.pin:61-62).
   P5.5's TARGETS and record stay as they are, so (d) can follow P5.5's first install, but by the plan's rule ("A
   target flang needs added to P5.5's tree is a P5.5 pin change", P12.1's Accept) it is still a P5.5 pin change, and
   P5.5's pin would have to name it, since the record does not. Recommendation: (a) if answered before P5.5's first
   install, else (d); (b) as the fallback if the standalone configure finds another gap.
2. Conditional, not filed now: NVPTX in LLVM_TARGETS_TO_BUILD (CUDA area A.5). Cause: the host-only device-side parse
   is untested upstream. File it only if P12.13's first run fails there; it means a full Polygeist rebuild.

Pins outside the rule:

3. apt.llvm.org flang, a third LLVM (Flang finding 5): flang-17 (17.0.6) shares batch 1's LLVM 17, an earlier
   version whose branch left main before the pin (17.0.6 carries 405 commits the pin lacks), so the policy covers its
   bitcode by version number only (inference, not shown); flang-20 (20.1.8) shares batch 1's LLVM 20 but also needs
   libmlir-20, which batch 1 does not bring, and its IR is not promised readable at the pin; flang-23 (the 23.1.4
   snapshot) is the only one with fir-to-scf and fir-to-memref, needs the jammy-23 suite and libllvm23,
   libclang-cpp23, and libmlir-23, which batch 1 does not bring (sizes not read; Flang finding 5), and its MLIR text
   is a known refusal at 18.0.0git as printed, through its `dlti.dl_spec` attribute, so a stripping step would come
   first (section 6.B). Recommendation: keep the rule (no apt flang) unless the owner wants structured Fortran
   (candidate 6). The packages go to the distro's /usr, and a shared-only runtime would need an
   rpath or a sandbox change.
- xDSL goes into OQ-059 as its Response, not as a new item: 0.73.0 and 0.23.0 are both outside the rule.
- Conditional, not filed now: a second, older CUDA header set (12.1 or below), if cuda@12.6.3's headers fail to parse
  on clang 18.0.0git. Cause: clang at the pin knows CUDA only up to 12.1. The alternative is `-nocudainc` with a
  minimal LASSI header, like upstream's `Test/Verification/Inputs/cuda.h`.
- None from Rust, .NET, or Python and C: the nightly, rayon 1.12.0, rustup 1.29.1, SDK 10.0.402, the Roslyn 5.9.0
  packages, numpy 2.2.6, and gcc-12 each follow the rule.

Paths the rule cannot serve (recorded with their causes; an item is needed only if the owner wants the path):

4. Rust's low fallback: `rustc --emit=llvm-ir` (or bitcode) read by the pinned mlir-translate (Rust R12; LLVM import
   finding 6). Cause: the nightly's LLVM is 23.1.4 and the reader's 18.0.0git; LLVM IR is not forward compatible, and
   rustc's text carries `getelementptr inbounds nuw`, `trunc nuw`, `captures(none)`, and `range(...)`, which the reader
   does not have; rustc needs an external LLVM of 21 or newer, so it cannot be built against Polygeist's LLVM. Options:
   (a) record it as unserved: Rust reaches the hub only through the MIR frontend, and rust-mir refuses out-of-subset
   code with a diagnostic naming both LLVM versions; (b) a second, older rustc for the fallback only, outside the rule:
   an LLVM-18-era nightly, not shown readable (26eb4285 predates LLVM 18.1's `disjoint` and `nneg`; it would need its
   own spike), or rustc 1.77.2 (LLVM 17.0.6 in Rust's fork d1e9093d, an earlier version whose branch left main before
   the pin), whose bitcode the policy covers by version number only (inference, not shown); either gives low modules
   only; (c) a third LLVM: mlir-translate at 23.1.x, from the 23.1.3 binary or source, outside the rule, LLVM-scale if
   built; it still gives only low modules, in MLIR 23's LLVM dialect, whose text is a known refusal at 18.0.0git as
   printed (its importer's `dlti.dl_spec` attribute, section 6.B), so a stripping step would come first, and the
   rest is still not promised to parse; and LLVM 23 has no raise either. Recommendation: (a); the gate needs
   structured modules, which neither (b) nor (c) gives without a new raiser.
5. C# through NativeAOT-LLVM (.NET finding 11; LLVM import finding 6). Cause: WebAssembly is its only target, so its IR
   is wasm32 and depends on the wasm NativeAOT runtime, with no native x86-64 run; its LLVM is 18.1.3, newer than the
   reader and a third LLVM; its packages are floating prereleases on dotnet-experimental, not nuget.org; it needs
   Emscripten 3.1.56 or WASI SDK 29. Options: (a) record the route as unavailable on Linux: C# reaches the hub only
   through the Roslyn subset, and other code is refused and counted; (b) pin NativeAOT-LLVM for wasm32 `low` syntax
   data only: a third LLVM, a prerelease feed, and a wasm toolchain in the install batch, with no native round trip;
   (c) leave the route open for a later phase. Recommendation: (a).
6. A structured hub module from Fortran (Flang finding 3). Cause: at 26eb4285 FIR lowers only to cf and llvm;
   core-dialect lowering arrived upstream in LLVM 21 (fir-to-scf) and 23 (fir-to-memref). Options: (a) Fortran is
   tagged low through LLVM IR import (P12.7) and stays outside the gate reading, consistent with OQ-057's Fortran
   option (a); (b) P12 writes a FIR-to-core conversion against the pinned flang libraries (new C++ work), with
   promote-to-affine as a partial starting point, behind its fir.convert casts (Flang finding 3); (c) a newer flang
   outside the rule, as in candidate 3, whose MLIR text needs the stripping step of section 6.B first.
   Recommendation: (a); tie it to OQ-057.
- GPU code emission (`--emit-cuda`): it needs `POLYGEIST_ENABLE_CUDA=1`, NVPTX, `MLIR_ENABLE_CUDA_RUNNER`, nvcc and
  libcuda at configure, and a GPU. P12 does not need it, so no item unless the owner wants GPU emission.
- The Numba fallback of the Python row: outside P12's scope; llvmlite's own LLVM would be a third LLVM. None filed.

Scope:

7. Turning LLVM IR into structured modules (LLVM import finding 5). Options: (a) in P12 the LLVM import frontend emits
   `low` modules only; (b) build a new LLVM-dialect-to-core raiser in P12 (steps (c) and (f) of finding 5, PROJECTED
   0.1 to 0.5 GiB); (c) give that raiser a later phase. Recommendation: (a); the gate reads cgeist for C and C++ and
   the subset frontends for the other languages, and the Frontend Rules allow low modules.

Not owner items: CARGO_HOME and RAYON_NUM_THREADS (and, as an option, DOTNET_PROCESSOR_COUNT) are P12.6 [DESIGN]
changes; the nightly's date tie-break (2026-10-17 by label, 2026-10-18 by local time) is settled by the channel's
date label, a UTC date (R1); Python 3.10's upstream end of life (P7) is recorded in Risks And Questions under
toolchain churn, and moving off 3.10 would be a pinned toolchain change that needs an owner decision (Ubuntu's own
support term for python3.10 was not read).

## Consequences for the plan

- P12.5: Rust by route B (curl, check against the manifest hash, `install.sh --prefix=$P --disable-ldconfig`), with
  the vendored crates written by `cargo vendor --locked --versioned-dirs` from a committed Cargo.lock and CARGO_HOME
  under `$P` at vendoring time; .NET by a SHA-512 check and `tar -xzf` (or dotnet-install.sh with `--keep-zip` and a
  check after), the three packages into a feed under the prefix, each with its SHA-512 and a computed sha256; numpy by
  `venv --without-pip` and an offline `uv pip install --no-index --require-hashes --compile-bytecode`; no flang unless
  the owner picks apt (candidate 3); cuda@12.6.3 if OQ-058 is answered (a) by then. P12.5 reads the ref-pack directory
  version (10.0.12 or 10.0.13) after extraction; under route B the MIR driver's rpath is `$P/lib` (R6).
- P12.6: the CARGO_HOME and RAYON_NUM_THREADS [DESIGN] changes, and DOTNET_PROCESSOR_COUNT if chosen; the Rust build
  config (linker by path, `build.jobs = 2`, vendored offline source, `build.rustc-workspace-wrapper`); C# compiles by
  `csc.dll` and runs by a script artifact; a Python launcher with `-B`; a `gcc-c-native` class and `gcc-c.pin`.
- P12.7: low modules only; `lift-cf-to-scf` does nothing on an imported module; the producer is Polygeist's clang; a
  refusal names the producer's LLVM version.
- P12.8: the driver stops after analysis, fixes its rustc flags for byte-identical re-raises, links with an absolute
  rpath, and has no low fallback under candidate 4 (a).
- P12.9: the Roslyn package (NuGet 5.9.0) and the SDK's csc (5.9.0-1.26466.105) are different builds of one minor; no
  NativeAOT-LLVM fallback under candidate 5 (a).
- P12.10: flang standalone after the P5.5 change (candidate 1, OQ-063), with job pools of 3 compiles and 1 link,
  once the owner's 24GB .wslconfig is in place; the low route through P12.7; the runtime link flags and `_QQmain` and
  `_QQEnvironmentDefaults` for the CPU target; the Fortran row's status updated from the code.
- P12.13: the raise line above; P5.6's two open points (polygeist-opt for verification, `NO_CUSTOM_FORM`).
- P12.14: as OQ-059 is answered.

## Proposed bible edits (from the areas; the main session merges them into one Decision Log entry)

Applied 2026-10-17 in docs/BIBLE.md as one Toolchain Pins bullet (P12's frontend toolchains, proposed), the Language
Frontends rows named below, and one Decision Log entry. The fragments stay as the areas proposed them, with the
owner-queue numbers filled in and the corrections of the source check applied.

The review's corrections, applied 2026-10-18 to the same 2026-10-17 text: the bullet tags its flang part "[OPEN] on
OQ-063, and until P12.10 builds it" and says the nightly pin follows the channel's date label, a UTC date (R1); the
Fortran row's status reads "to be built by P12.10 (OWNER on OQ-063)"; the hub sentence says xDSL parses the hub
dialects at its own MLIR version, not at 26eb4285 (OQ-059's Response); the Risks And Questions toolchain churn row
names Python 3.10's upstream end of life (P7) and that moving off it is a pin change; and the Decision Log entry
names those two sections, takes the date-label rule, marks the OpenACC gap medium confidence (Flang finding 2), says
rustc's IR carries constructs the 18.0.0git parser rejects (R12), and says NativeAOT-LLVM's IR is not promised
readable there (LLVM import finding 6 shows constructs in 18.1.3's LangRef that the pin lacks, but no NativeAOT-LLVM
output that uses them).

Toolchain Pins, new bullets (proposed, not installed):

- Rust: "Rust frontend pin [DESIGN, proposed, not installed] (P12.1, plans/spikes/p12-frontend-pins.md)", holding the
  nightly-2026-10-17 archives and components with their sha256, rustc's LLVM 23.1.4 (rust-lang/llvm-project 915c669d),
  the rayon 1.12.0 closure with its six checksums, the prefix layout `rustc@nightly-2026-10-17` with the vendored
  crates under it, and the linker by path, as in the Rust pin table.
- .NET: ".NET pin, proposed, not installed [DESIGN] (P12.1, plans/spikes/p12-frontend-pins.md, .NET), by the P12
  plan's frontend pin rule. The pin is the newest .NET LTS SDK on 2026-10-17: SDK 10.0.402, released 2026-10-13, with
  runtime 10.0.13, C# 14.0 and Roslyn 5.9.0. Its archive is dotnet-sdk-10.0.402-linux-x64.tar.gz, 240279509 bytes,
  SHA-512
  `1904a98dda2db34ba78ba3c4fb1b224455329af91c341d83547843c1839e112b04ae699ee92fa2449a8f62249196a938723078c0c60b9f65c88539d5687729c2`
  (10.0 releases.json; 10.0.13-sha.txt), and it installs under `$LASSI_TOOLCHAINS/dotnet@10.0.402`. An offline NuGet
  feed under that prefix holds Microsoft.CodeAnalysis.CSharp 5.9.0, Microsoft.CodeAnalysis.Common 5.9.0 and
  Microsoft.CodeAnalysis.Analyzers 5.9.0-1.26328.17; their SHA-512 values, as nuget.org's catalog publishes them, are
  in the spike. C# programs compile with the SDK's csc.dll through the pinned host, not with the SDK command line, and
  run through that host by path. The Roslyn frontend is framework-dependent and runs the same way. P12.5 adds the
  sha256 values."
- Python and C: "Python and C for native runs, proposed, not installed [DESIGN] (task P12.1;
  plans/spikes/p12-frontend-pins.md, Python and C): toolchains/python.pin, NAME python, VERSION 3.10.12-numpy2.2.6,
  PREFIX_NAME python@3.10.12-numpy2.2.6, a venv of batch 1's /usr/bin/python3.10 (3.10.12-1~22.04.18) made without
  pip, holding numpy 2.2.6, the newest numpy with cp310 wheels on 2026-10-17 (2.3.0 and later require Python 3.11 or
  later), from numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl, 16,801,050 bytes, sha256
  fc7b73d02efb0e18c000e9ad8b83480dfcd5dfd11065997ed4c6747470ae8915 (PyPI; the v2.2.6 release page), installed offline
  by hash with bytecode compiled; programs run as `<prefix>/bin/python3 -B` with no PYTHONPATH. toolchains/gcc-c.pin,
  NAME gcc-c, VERSION 12.3.0, EXECUTABLE /usr/bin/gcc-12 (gcc-12 12.3.0-1ubuntu1~22.04.3, which batch 1's g++-12
  brings by its exact Depends), FLAGS `-O3 -fopenmp` with `-lm` after the sources, EXPECT_VERSION
  `(Ubuntu 12.3.0-1ubuntu1~22.04.3) 12.3.0`, no PREFIX_NAME and no install script. Numba is not pinned." The native
  row (docs/BIBLE.md:647) changes when P12.12 builds these.
- Flang: "Flang [DESIGN, proposed, not installed] (task P12.1, plans/spikes/p12-frontend-pins.md): flang at
  llvm-project 26eb4285b56edd8c897642078d91f16ff0fd3472 (LLVM 18.0.0git, Polygeist's LLVM), by the frontend pin rule,
  prefix flang@26eb4285, built standalone from P5.5's checkout against P5.5's build tree (needs LLVMFrontendOpenACC
  built in P5.5's tree, OQ-063); Release, assertions on, tests off, LLVM_PARALLEL_COMPILE_JOBS=3,
  LLVM_PARALLEL_LINK_JOBS=1; targets flang-new, fir-opt, tco, module_files; EXPECT_VERSION "flang-new version 18.0.0"
  (PROJECTED).
  At this commit HLFIR is opt-in, FIR leaves only through cf and llvm, so Fortran modules are low (LLVM IR import);
  flang-new links the static FortranRuntime, FortranDecimal, and Fortran_main. Sizes and times PROJECTED. [OPEN] until
  P12.10 builds it." Applied in the bullet as "[OPEN] on OQ-063, and until P12.10 builds it".
- LLVM import: "LLVM import [DESIGN] (P12.1, proposed, not installed): no pin of its own. It uses the Polygeist stack's
  mlir-translate, mlir-opt, and polygeist-opt (P5.5's TARGETS). Its LLVM 18.0.0git reads LLVM IR from its own LLVM, or
  bitcode of an earlier LLVM version by LLVM's policy, so its producer under the frontend pin rule is that stack's clang. A pin change is a
  Decision Log entry."
- CUDA path and xDSL: "cgeist's CUDA path takes no pin of its own: polygeist@77c04bb2 as P5.5 builds it
  (`POLYGEIST_ENABLE_CUDA=0`, host target) and cuda@12.6.3's headers through `--cuda-path`. Clang at 26eb4285 knows
  CUDA up to 12.1 and reads 12.6 as newer than supported, with a warning (plans/spikes/p12-frontend-pins.md). xDSL
  stays unpinned pending OQ-059."

Language Frontends:
- The "Any LLVM language" row (docs/BIBLE.md:523): path "LLVM IR import (`mlir-translate --import-llvm` at Polygeist's
  LLVM, 26eb4285)"; level reached "llvm"; status "Available, low level: at 26eb4285 `lift-cf-to-scf` lifts cf
  branches in func.func only, and no pass raises an imported llvm.func to func and cf
  (plans/spikes/p12-frontend-pins.md)".
- The Rust row's note (:521): "llvm-import fallback not served at these pins: rustc's LLVM 23.1.4 IR is not readable by
  mlir-translate at 26eb4285 (LLVM 18.0.0git)". Applied as a fact at the pins; whether another reader serves it is
  OQ-064.
- The C# row (:522): applied likewise, since NativeAOT-LLVM gives no Linux LLVM IR (.NET finding 11; LLVM import
  finding 6); whether the route is pinned for wasm32 data or left for later is OQ-064.
- The Fortran row: applied now from Flang finding 3, as what holds at the pin (low through LLVM IR import); P12.10
  updates its status from the code, and structured Fortran is OQ-064.

Decision Log fragments, dated 2026-10-17, to merge into P12.1's single entry (and the count sentence):
- Rust: "P12.1: Rust pins proposed (nightly-2026-10-17, rayon 1.12.0); Rust LLVM-IR fallback recorded as not served,
  cause IR version skew | Frontend pin rule".
- .NET: ".NET pin proposed, not installed: SDK 10.0.402 (runtime 10.0.13, Roslyn 5.9.0) with
  Microsoft.CodeAnalysis.CSharp 5.9.0 in an offline feed; C# builds call csc through the pinned host | P12.1's frontend
  pin rule (newest .NET LTS SDK on the spike's date); the compile sandbox gives the SDK command line no writable CLI
  home".
- Python and C: "Proposed Python and C native pins (task P12.1; Toolchain Pins); P12.5 installs the python pin, and
  gcc-12 comes with batch 1 | The P12 frontend pin rule picks the newest numpy with cp310 wheels, 2.2.6 (PyPI JSON
  read 2026-10-17), and a host C compiler pinned by version and path. g++-12's `Depends: gcc-12 (=
  12.3.0-1ubuntu1~22.04.3)` in the jammy-updates index fixes gcc-12's version. GCC 12.3.0's C driver adds no libm
  (gcc/c/gccspec.cc), hence `-lm`."
- Flang: "Proposed Flang pin 26eb4285 (task P12.1; Toolchain Pins), standalone against P5.5's tree, pending the P5.5
  pin change for LLVMFrontendOpenACC (OQ-063) | The rule takes llvm-project frontends at Polygeist's LLVM; the
  standalone build reads P5.5's CMake packages, but flang's parser needs acc_gen's headers and LLVMFrontendOpenACC, which P5.5's targets do
  not build; no FIR-to-core lowering exists at the pin (fir-to-scf from LLVM 21, fir-to-memref from LLVM 23), so
  Fortran is low there; apt flang would be a third LLVM."
- LLVM import: "LLVM import (task P12.1; Language Frontends, Toolchain Pins): the stack's mlir-translate --import-llvm
  at 26eb4285, with no pin of its own, tagging modules low. Its producer is that stack's clang. The Any LLVM language
  row reaches llvm only. rustc's and NativeAOT-LLVM's LLVM IR are paths the rule cannot serve (OQ-064) | the import
  writes only the LLVM dialect (ModuleImport.cpp:1694, :1287-1340), and the lift walks func.func and cf branches only
  (ControlFlowToSCF.cpp:170, :36-73). No raise exists at 26eb4285, in Polygeist 77c04bb2, or at 23.1.3. LLVM promises
  only backward bitcode reading (DeveloperPolicy.rst:646-660). The nightly's LLVM is 23.1.4, and NativeAOT-LLVM's is
  18.1.3 for WebAssembly only."
- CUDA path and xDSL: "P12.1 read cgeist's CUDA path and xDSL's releases from source and indexes. No P5.5 pin change
  is needed for the CPU route. Neither xDSL 0.73.0 nor 0.23.0 parses LLVM 18.0.0git's hub text (earlier releases not
  read). Both findings were filed as Responses to OQ-058 and OQ-059."

## Confidence

- High: versions, publisher hashes, and sizes read from manifests, indexes, catalogs, and HTTP headers; rustc's LLVM
  and the IR constructs the reader rejects; NativeAOT-LLVM's targets; the import's output and the lift's reach; FIR's
  exits at the pin; cgeist's CUDA flags and pipeline and that the CPU route is not gated; xDSL's versions and the named
  mismatches; the DLTI syntax at the pin and at llvmorg-23.1.3, and the pools' scope.
- Medium: the sandbox behaviour of cargo, the dotnet CLI, and Python bytecode, inferred from source and not run; the
  OpenACC gap (the closure of P5.5's TARGETS was not read in full); that a host-only LLVM serves the CUDA parse; the
  gcc-12 banner.
- Low: every PROJECTED size, time, and memory figure, flang's most of all. P12.5's install record and P12.10's build
  measure them.
- Unknown: whether cuda@12.6.3's headers parse at clang 18.0.0git; whether the xDSL mismatch list is complete.

## Sources

All read 2026-10-17 on the workstation, except the review additions of 2026-10-18 listed last; each finding above
cites its file and line.

- Rust: https://static.rust-lang.org/dist/2026-10-17/channel-rust-nightly.toml, its `.sha256` and `.asc`, the six
  archives' `.sha256` files and Content-Length, and their xz indexes (64 KiB range reads); the undated
  channel-rust-nightly.toml; https://static.rust-lang.org/rustup/archive/1.29.1/x86_64-unknown-linux-gnu/rustup-init
  and its `.sha256`; https://static.rust-lang.org/dist/channel-rust-1.77.2.toml and HEAD requests for its archives;
  rust-lang/rust at 5d8fec0669a64607db9ccbf663e7c23125bf4748 (compiler/rustc_public, rustc_public_bridge,
  rustc_driver_impl, rustc_session, rustc_codegen_ssa, rustc_codegen_llvm, rustc_llvm, src/bootstrap, src/ci,
  src/tools/rust-installer, its install-template.sh read again 2026-10-17, library/std,
  tests/ui-fulldeps/rustc_public, .gitmodules, RELEASES.md) and its commit metadata; rust-lang/llvm-project at 915c669d and d1e9093d; rust-lang/cargo at 3960fa69; rustup 1.29.1;
  https://rust-lang.github.io/rustc_public/initial.html; https://rust-lang.github.io/rustup/installation/index.html;
  https://doc.rust-lang.org/nightly/std/thread/fn.available_parallelism.html; crates.io API and
  https://index.crates.io/ (rayon, rayon-core, either, crossbeam-deque, crossbeam-epoch, crossbeam-utils); rayon at
  c9ced185; https://packages.ubuntu.com/jammy/amd64/gcc-12/filelist.
- .NET: https://builds.dotnet.microsoft.com/dotnet/release-metadata/releases-index.json and 10.0/releases.json;
  https://builds.dotnet.microsoft.com/dotnet/checksums/10.0.13-sha.txt; the SDK archive's headers and range reads;
  dotnet/core 50e60dcd release notes; nuget.org flat-container, registration, and catalog leaves and nuspec files;
  dotnet/sdk d037e5f8, dotnet/dotnet a8dd0577, dotnet/runtime 6aa72daa, dotnet/install-scripts 5147e323,
  dotnet/runtimelab 527a5657; https://dot.net/v1/dotnet-install.sh; the dotnet-experimental feed index;
  learn.microsoft.com pages on dotnet environment variables (git 6b94bd8a), NuGet global packages folders and CLI
  environment variables
  (git 2bc45989), MSBuild properties (git eb76160a), dotnet build (git 4bce4d81), and threading configuration (git
  028e3437); https://releases.ubuntu.com/22.04.5/ubuntu-22.04.5-wsl-amd64.manifest.
- Python and C: https://pypi.org/pypi/numpy/json and https://pypi.org/pypi/numpy/2.2.6/json; the numpy wheel's tail
  (range read); https://github.com/numpy/numpy/releases/tag/v2.2.6; numpy v2.2.6, OpenBLAS v0.3.29, CPython v3.10.12,
  and GCC releases/gcc-12.3.0 (gcc-mirror/gcc) sources; https://packages.ubuntu.com/jammy-updates/g++-12 and the gcc-12
  filelist; http://archive.ubuntu.com/ubuntu/dists/jammy-updates/{main,universe}/binary-amd64/Packages.gz;
  docs.python.org 3.10 (venv, sys.path, command line); https://man7.org/linux/man-pages/man2/execve.2.html;
  https://docs.astral.sh/uv/reference/cli/; https://peps.python.org/pep-0619/; https://devguide.python.org/versions/.
- Flang: llvm-project at 26eb4285 (flang/, clang/, llvm/, mlir/ files cited above) and the GitHub contents, trees, and
  commits APIs; llvmorg-17.0.6, 21.1.8, 22.1.8, and 23.1.3 and 0cd28174a1f6 for the later passes; the GitHub compare
  `repos/llvm/llvm-project/compare/26eb4285b56edd8c897642078d91f16ff0fd3472...llvmorg-17.0.6` and that tag's commit
  6009708b4367 (read again 2026-10-17);
  https://apt.llvm.org/jammy/dists/ and the llvm-toolchain-jammy-NN Packages and Release files;
  https://discourse.llvm.org/t/comparing-build-times-for-flang/57199;
  https://lists.macports.org/pipermail/macports-users/2025-March/053432.html;
  https://lists.llvm.org/pipermail/flang-dev/2019-November/000066.html;
  https://llvm.googlesource.com/llvm-project/+/083c683969b2436afdc45becadc955841f5f4d31%5E%21.
- LLVM import: llvm-project at 26eb4285 (mlir/ and llvm/ files cited above), llvmorg-18.1.3 and llvmorg-23.1.3, the
  compare and releases APIs; https://llvm.org/docs/DeveloperPolicy.html; Polygeist at 77c04bb2
  (include/polygeist/Passes/Passes.td, lib/polygeist/Passes/CanonicalizeFor.cpp); dotnet/runtimelab 527a5657.
- CUDA path and xDSL: Polygeist at 77c04bb2 (tools/cgeist, lib/polygeist/Passes, include/polygeist, README.md,
  .github/workflows/build.yml, tools/cgeist/Test); llvm-project at 26eb4285 (clang/ CUDA files, mlir/ SCF and DLTI
  files) and the commit lists after it; https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.6.3.json;
  openxla/xla at 416df8ba99880855e536c483df6e9dcadce82931; https://pypi.org/pypi/xdsl/json; xDSL at a2544953 and
  d1f33aac, and c624080a58; nixpkgs at 39ad350a0602fa0a58a544344e3e9187526ea45c.
- Repository at 9cbd9bd: lassi/executors/sandbox.py, lassi/executors/native.py, lassi/toolchains/ (gcc.py, _base.py,
  pins.py, runner.py), tools/local/install-batch1.sh, tools/local/host-profile.sh, toolchains/gcc.pin,
  toolchains/cuda.pin, assets/harness/c/lassi_io.h, docs/BIBLE.md, plans/p12-frontends.md, plans/p5-ir.md,
  plans/local-plan.md, plans/spikes/local-host.md, plans/spikes/p0-nvcc.md, plans/spikes/p5-mlir-pins.md; and the
  stash `stash@{0}^3` (toolchains/polygeist.pin and polygeist.sh).
- Review additions, read 2026-10-18 with `curl -sS -f` into a session scratch directory: raw files at
  https://raw.githubusercontent.com/llvm/llvm-project/<ref>/<path> for 26eb4285
  (llvm/cmake/modules/HandleLLVMOptions.cmake, mlir/lib/Dialect/DLTI/DLTI.cpp,
  flang/include/flang/Optimizer/Transforms/Passes.td) and llvmorg-23.1.3
  (mlir/lib/Dialect/DLTI/DLTI.cpp, mlir/lib/Target/LLVMIR/ModuleImport.cpp, flang/lib/Lower/Bridge.cpp,
  flang/lib/Optimizer/Support/DataLayout.cpp, flang/lib/Frontend/FrontendActions.cpp);
  https://apt.llvm.org/jammy/dists/llvm-toolchain-jammy-23/main/binary-amd64/Packages (sha256 as before);
  http://archive.ubuntu.com/ubuntu/dists/jammy-updates/main/binary-amd64/Packages.gz; and `git show` of the stash's
  toolchains/polygeist.pin (:55, :63) and polygeist.sh (:1-26, :367).
