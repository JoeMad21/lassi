# Spike P1.1: the upstream LASSI pin and the HeCBench pin

- Task: P1.1 in `plans/p1-faithful.md` (Accept: upstream LASSI pinned at
  74b4681; the 20 upstream `*_main` files checked against HeCBench 7d2d3c5;
  the HeCBench commits holding all 20; the source of each model-facing file
  and the HeCBench pin for support files). By the Decision Log entry of
  2026-09-24 (task P1.1), the pin is a manifest and a fetch tool, not the
  plan's submodule (see Upstream pin below).
- Date: 2026-09-24 (EDT). The rx ids embed the build host's clock
  (UTC-07:00), so they read 2026-09-23.
- Where: the workstation (Windows, Git Bash, git 2.55.0.windows.3 as
  `git --version` prints it), branch `p1-faithful` at 6aa8155 (the HeCBench
  commands and the submodule attempt) and at 0ea93e1 (the fetch tool runs),
  with this task's changes uncommitted each time, and `rx run` on alpha01
  (git 2.34.1 as `rx doctor` reports it).
- Evidence label: exploratory. The values below are git object ids, counts
  read from trees, and host readings from the cited rx runs (free space,
  scratch use, a pytest time); all are exploratory. Git derives object ids
  from content, so any clone reproduces them. None of them is a measurement.
  The rx runs went from a dirty tree (a snapshot commit) and are
  exploratory; the rerun from the task's clean commit goes into the P1.1
  STATUS note.

## Question

1. Does `tools/fetch_upstream.py` check out SPEAR-UIC/LASSI at 74b4681 into
   `third_party/LASSI`, on the workstation and on the build host?
2. For each of the 20 upstream `*_main` files: its blob id, and whether
   HeCBench 7d2d3c5 (the P0 manifest pin) holds that blob at
   `src/<app>-<omp|cuda>/main.*`.
3. Which HeCBench commits hold all 20?
4. Where does each model-facing file come from, and which HeCBench commit
   supplies the support files?

## Classification

Factual: every answer is read from git trees.

## Method

- Upstream: first a submodule, then (after the decision below) the manifest
  `assets/upstream/lassi.yaml` and `tools/fetch_upstream.py`, which fetches
  exactly the pinned commit into `third_party/LASSI` (gitignored, never
  tracked). No upstream code ran. The only upstream file text read was the `#include "..."`
  lines of the 20 `*_main` files (`git grep`, below). Nothing from
  `prompt_dictionary.py` or the notebook is quoted here (OQ-018);
  `tests/bench/test_upstream_pin.py` checks that no 40-character run of
  either appears in this file.
- HeCBench: a clone without blobs and without a checkout
  (`--filter=blob:none --no-checkout`) in the workstation's
  temporary directory, never under the repository. Blob ids come from trees,
  so no source file was needed; git fetched only the few blobs that the
  `git show` and `git diff --stat` lines below print from.
- `$HB` below is the path of that clone.

## Commands and outputs

### Upstream pin

First attempt: a submodule. On the workstation:

```
$ git submodule add https://github.com/SPEAR-UIC/LASSI third_party/LASSI
$ git -C third_party/LASSI checkout --quiet --detach 74b46812523f2ff79b53b6880a4521690d7478b0
$ git submodule status third_party/LASSI
 74b46812523f2ff79b53b6880a4521690d7478b0 third_party/LASSI (heads/main)
$ git -C third_party/LASSI log -1 --format='%H %ad' --date=iso 74b4681
74b46812523f2ff79b53b6880a4521690d7478b0 2024-09-23 18:17:51 -0500
```

Upstream's `main` is at the pin (`heads/main`). On the build host, after
`rx doctor` (ok, 427.5 GB free on the scratch disk) and
`du -sh /mnt/nvme10/joseph_ufl` (94G, rx 20260923-225936-exec-f402), the
submodule update reached GitHub and checked out the pin (a dirty snapshot, so
exploratory):

```
$ uv run tools/rx.py run -- 'git submodule update --init third_party/LASSI && git -C third_party/LASSI rev-parse HEAD'
[rx] dirty tree sent as snapshot 70afc357d433 (exploratory; not reportable)
Submodule 'third_party/LASSI' (https://github.com/SPEAR-UIC/LASSI) registered for path 'third_party/LASSI'
Cloning into '/mnt/nvme10/joseph_ufl/lassi-wt/desktop-8r113ei-p1-faithful/third_party/LASSI'...
Submodule path 'third_party/LASSI': checked out '74b46812523f2ff79b53b6880a4521690d7478b0'
74b46812523f2ff79b53b6880a4521690d7478b0

[rx] id=20260923-225948-desktop-8r113ei-p1-faithful-0f11 rc=0 state=done
```

The submodule was dropped before any commit. A staged gitlink fails the
text-policy check of staged paths: the checker reads every staged path as a
blob of this repository (`git show :<path>`), and a gitlink names a commit of
another repository. That checker may not be changed to pass a commit
(AGENTS.md), so the pin became a manifest and a fetch tool on the model of
`tools/fetch_bench.py`. The workstation keeps the checkout the submodule
clone left: `third_party/LASSI/.git` is a file pointing into
`.git/modules/third_party/LASSI`, at the pin and clean; the fetch tool
accepts it as it is.

Second attempt: `assets/upstream/lassi.yaml` (url, commit, path) and
`tools/fetch_upstream.py`.

On the workstation (at 0ea93e1, this task's changes uncommitted), the
default run finds the checkout the submodule clone left and changes nothing.
A fetch into an empty directory under the workstation's temporary directory
reaches GitHub, and a rerun there changes nothing. `$D` stands for that
directory; the tool prints its full path.

```
$ uv run python tools/fetch_upstream.py
fetch_upstream: third_party/LASSI already holds https://github.com/SPEAR-UIC/LASSI at 74b46812523f2ff79b53b6880a4521690d7478b0
$ uv run python tools/fetch_upstream.py --dest "$D"
fetch_upstream: fetched https://github.com/SPEAR-UIC/LASSI at 74b46812523f2ff79b53b6880a4521690d7478b0 into $D
$ git -C "$D" rev-parse HEAD
74b46812523f2ff79b53b6880a4521690d7478b0
$ uv run python tools/fetch_upstream.py --dest "$D"
fetch_upstream: $D already holds https://github.com/SPEAR-UIC/LASSI at 74b46812523f2ff79b53b6880a4521690d7478b0
```

On the build host, after `rx doctor` (ok, 427.6 GB free on the scratch
disk) and `du -sh /mnt/nvme10/joseph_ufl` (94G, rx
20260923-234424-exec-1e24). Both runs went from dirty snapshots, so they are
exploratory. The slot still holds the checkout of the submodule run above
(its `.git` is a file pointing into the slot's `modules` directory), and the
tool accepts it unchanged:

```
$ uv run tools/rx.py run -- 'uv run python tools/fetch_upstream.py && git -C third_party/LASSI rev-parse HEAD'
[rx] dirty tree sent as snapshot cfebf61db229 (exploratory; not reportable)
fetch_upstream: third_party/LASSI already holds https://github.com/SPEAR-UIC/LASSI at 74b46812523f2ff79b53b6880a4521690d7478b0
74b46812523f2ff79b53b6880a4521690d7478b0

[rx] id=20260923-234443-desktop-8r113ei-p1-faithful-b135 rc=0 state=done
```

The two remote fetch-tool tests then fetched the pin from GitHub into an
empty pytest temporary directory under the scratch root, and ran the default
fetch twice in the slot:

```
$ uv run tools/rx.py run -- 'head -c 200 third_party/LASSI/.git; echo; uv run pytest -q -m remote -k fetch_tool -p no:cacheprovider tests/bench/test_upstream_pin.py; echo "pytest_rc=$?"'
[rx] dirty tree sent as snapshot f1d02c38817e (exploratory; not reportable)
gitdir: ../../../../lassi.git/worktrees/desktop-8r113ei-p1-faithful/modules/third_party/LASSI

..                                                                       [100%]
2 passed, 30 deselected in 1.59s
pytest_rc=0

[rx] id=20260923-234455-desktop-8r113ei-p1-faithful-a1fa rc=0 state=done
```

### The 20 upstream files

```
$ git -C third_party/LASSI ls-tree -r 74b4681 -- translated_code/input_codes
100644 blob d0f36289ec57ae0b8d301373d65282c1f00d2663	translated_code/input_codes/HeCBench/atomicCost/atomicCost-cuda_main.cu
100644 blob 7851d6da8d9f72074a5a78600b8abb09c9931f75	translated_code/input_codes/HeCBench/atomicCost/atomicCost-omp_main.cpp
100644 blob ba21f7cf1105dc3bfc4a4bccb606b6b27076c3b9	translated_code/input_codes/HeCBench/bsearch/bsearch-cuda_main.cu
100644 blob bf485da7969bc5fd439a4ac12efb26501bc4e0a9	translated_code/input_codes/HeCBench/bsearch/bsearch-omp_main.cpp
100644 blob 714e36eb096f72bdb341bd581dbb3cf375cf8a93	translated_code/input_codes/HeCBench/colorwheel/colorwheel-cuda_main.cu
100644 blob cf865d1c607f428af9697688756fbbfa48d88091	translated_code/input_codes/HeCBench/colorwheel/colorwheel-omp_main.cpp
100644 blob 3ee82f7ff5daf0cca3af269008edafefc1664be8	translated_code/input_codes/HeCBench/dense-embedding/dense-embedding-cuda_main.cu
100644 blob 0dd8a571a1fd612e9fc29c97368ee1dc099a361d	translated_code/input_codes/HeCBench/dense-embedding/dense-embedding-omp_main.cpp
100644 blob 3d54ed880fc8ce3ee6483bbb7414969fa6692234	translated_code/input_codes/HeCBench/entropy/entropy-cuda_main.cu
100644 blob 273fb27df98db883e7180a3e4737b20981864974	translated_code/input_codes/HeCBench/entropy/entropy-omp_main.cpp
100644 blob 778edd2abfb2386cb4d42696d672427ef1cb9987	translated_code/input_codes/HeCBench/jacobi/jacobi-cuda_main.cu
100644 blob e920b53cb803e2f9f55e55520d7c269c50c993b8	translated_code/input_codes/HeCBench/jacobi/jacobi-omp_main.cpp
100644 blob 2ee1a77eb1ce7542de152cd977b68f4bd7cf3ac5	translated_code/input_codes/HeCBench/layout/layout-cuda_main.cu
100644 blob b0d64f2900734071965b8c9c84ab8d1f6dcfcd85	translated_code/input_codes/HeCBench/layout/layout-omp_main.cpp
100644 blob 8edcba8acc015d7e73b87c117413d2706602cac4	translated_code/input_codes/HeCBench/matrix-rotate/matrix-rotate-cuda_main.cu
100644 blob a67ddc5afebf1fe1bda90dbfec75299fc7b070f2	translated_code/input_codes/HeCBench/matrix-rotate/matrix-rotate-omp_main.cpp
100644 blob 18230cfe044ba3c35d2b2ad8bbd90ed8ae9c1bed	translated_code/input_codes/HeCBench/pathfinder/pathfinder-cuda_main.cu
100644 blob b298e20731fdd3a6d6bf365d48fe3611d841071e	translated_code/input_codes/HeCBench/pathfinder/pathfinder-omp_main.cpp
100644 blob afc9d5088f3ad26d2fe8163759fcaf39b7eb676c	translated_code/input_codes/HeCBench/randomAccess/randomAccess-cuda_main.cu
100644 blob c038377f19db37086b6daf38726300b73fd11282	translated_code/input_codes/HeCBench/randomAccess/randomAccess-omp_main.cpp
```

### HeCBench at 7d2d3c5

```
$ git clone --quiet --filter=blob:none --no-checkout https://github.com/zjin-lcf/HeCBench "$HB"
$ git ls-remote https://github.com/zjin-lcf/HeCBench HEAD refs/heads/master
7d2d3c567be522a2104065165de0a4a233a6ea1a	HEAD
7d2d3c567be522a2104065165de0a4a233a6ea1a	refs/heads/master
$ git -C "$HB" rev-list --all --count
10845
$ git -C "$HB" rev-list --count master
5940
$ git -C third_party/LASSI ls-tree -r 74b4681 -- translated_code/input_codes |
  while read mode type blob path; do
    base=${path##*/}; dir=${base%_main.*}; ext=${base##*.}; line="$base"
    for c in 7d2d3c5 692cba3; do
      got=$(git -C "$HB" rev-parse --verify --quiet "$c:src/$dir/main.$ext")
      if [ "$got" = "$blob" ]; then line="$line $c=yes"; else line="$line $c=no(${got:0:7})"; fi
    done; echo "$line"
  done
atomicCost-cuda_main.cu 7d2d3c5=no(d6faa56) 692cba3=yes
atomicCost-omp_main.cpp 7d2d3c5=no(feb7aa9) 692cba3=yes
bsearch-cuda_main.cu 7d2d3c5=no(37c3558) 692cba3=yes
bsearch-omp_main.cpp 7d2d3c5=yes 692cba3=yes
colorwheel-cuda_main.cu 7d2d3c5=no(566f40d) 692cba3=yes
colorwheel-omp_main.cpp 7d2d3c5=no(809de4d) 692cba3=yes
dense-embedding-cuda_main.cu 7d2d3c5=no(0833887) 692cba3=yes
dense-embedding-omp_main.cpp 7d2d3c5=no(339237c) 692cba3=yes
entropy-cuda_main.cu 7d2d3c5=no(1ed49f2) 692cba3=yes
entropy-omp_main.cpp 7d2d3c5=yes 692cba3=yes
jacobi-cuda_main.cu 7d2d3c5=no(e2bf447) 692cba3=yes
jacobi-omp_main.cpp 7d2d3c5=no(5362aa6) 692cba3=yes
layout-cuda_main.cu 7d2d3c5=yes 692cba3=yes
layout-omp_main.cpp 7d2d3c5=yes 692cba3=yes
matrix-rotate-cuda_main.cu 7d2d3c5=yes 692cba3=yes
matrix-rotate-omp_main.cpp 7d2d3c5=yes 692cba3=yes
pathfinder-cuda_main.cu 7d2d3c5=no(68a168d) 692cba3=yes
pathfinder-omp_main.cpp 7d2d3c5=no(89e3346) 692cba3=yes
randomAccess-cuda_main.cu 7d2d3c5=no(e55268d) 692cba3=yes
randomAccess-omp_main.cpp 7d2d3c5=no(99738c5) 692cba3=yes
```

In the `no` cases, the short id in parentheses is the blob HeCBench holds
at that path today.

### Commits holding all 20

The scan asks git, for every commit on every ref of the clone, which blob
sits at each of the 20 HeCBench paths (one `cat-file --batch-check` call over
216,900 `<commit>:<path>` names), and keeps the commits where all 20 equal
upstream's blobs. `mains.txt` holds `<blob> <path>` from the `ls-tree` above.

```
$ git -C third_party/LASSI ls-tree -r 74b4681 -- translated_code/input_codes | awk '{print $3, $4}' > mains.txt
$ uv run --no-project python scan.py "$HB" mains.txt
commits scanned: 10845
commits holding all 20: 274
   692cba32c5744f6ef024cca59f65e9488edba8bf
   3abcff49456177402030b639a6a6f2a76db7f1a1
   85b14743a45d65a1f9c0d7b07b00a716a2fdc001
atomicCost-cuda_main.cu d0f36289ec57ae0b8d301373d65282c1f00d2663 held in 660 commits
atomicCost-omp_main.cpp 7851d6da8d9f72074a5a78600b8abb09c9931f75 held in 664 commits
bsearch-cuda_main.cu ba21f7cf1105dc3bfc4a4bccb606b6b27076c3b9 held in 1606 commits
bsearch-omp_main.cpp bf485da7969bc5fd439a4ac12efb26501bc4e0a9 held in 3172 commits
colorwheel-cuda_main.cu 714e36eb096f72bdb341bd581dbb3cf375cf8a93 held in 2408 commits
colorwheel-omp_main.cpp cf865d1c607f428af9697688756fbbfa48d88091 held in 2408 commits
dense-embedding-cuda_main.cu 3ee82f7ff5daf0cca3af269008edafefc1664be8 held in 804 commits
dense-embedding-omp_main.cpp 0dd8a571a1fd612e9fc29c97368ee1dc099a361d held in 804 commits
entropy-cuda_main.cu 3d54ed880fc8ce3ee6483bbb7414969fa6692234 held in 1616 commits
entropy-omp_main.cpp 273fb27df98db883e7180a3e4737b20981864974 held in 3172 commits
jacobi-cuda_main.cu 778edd2abfb2386cb4d42696d672427ef1cb9987 held in 1200 commits
jacobi-omp_main.cpp e920b53cb803e2f9f55e55520d7c269c50c993b8 held in 612 commits
layout-cuda_main.cu 2ee1a77eb1ce7542de152cd977b68f4bd7cf3ac5 held in 2834 commits
layout-omp_main.cpp b0d64f2900734071965b8c9c84ab8d1f6dcfcd85 held in 3172 commits
matrix-rotate-cuda_main.cu 8edcba8acc015d7e73b87c117413d2706602cac4 held in 3172 commits
matrix-rotate-omp_main.cpp a67ddc5afebf1fe1bda90dbfec75299fc7b070f2 held in 3172 commits
pathfinder-cuda_main.cu 18230cfe044ba3c35d2b2ad8bbd90ed8ae9c1bed held in 2829 commits
pathfinder-omp_main.cpp b298e20731fdd3a6d6bf365d48fe3611d841071e held in 1361 commits
randomAccess-cuda_main.cu afc9d5088f3ad26d2fe8163759fcaf39b7eb676c held in 890 commits
randomAccess-omp_main.cpp c038377f19db37086b6daf38726300b73fd11282 held in 890 commits
```

`scan.py`:

```
"""Scan HeCBench history for the upstream LASSI *_main blobs (P1.1 spike helper)."""
import subprocess, sys, collections
HB = sys.argv[1]
rows = [l.split() for l in open(sys.argv[2]) if l.strip()]
files = []
for blob, path in rows:
    base = path.rsplit("/", 1)[1]
    app = base.rsplit("_main.", 1)[0]
    ext = base.rsplit(".", 1)[1]
    files.append((base, blob, f"src/{app}/main.{ext}"))
commits = subprocess.run(["git", "-C", HB, "rev-list", "--all", "--topo-order"], capture_output=True, text=True, check=True).stdout.split()
inp = "".join(f"{c}:{p}\n" for c in commits for _, _, p in files)
out = subprocess.run(["git", "-C", HB, "cat-file", "--batch-check=%(objectname)"], input=inp, capture_output=True, text=True, check=True).stdout.splitlines()
assert len(out) == len(commits) * len(files), (len(out), len(commits))
held = collections.defaultdict(list)
allc = []
i = 0
for c in commits:
    ok = 0
    for base, blob, p in files:
        got = out[i].split()[0]
        i += 1
        if got == blob:
            held[base].append(c)
            ok += 1
    if ok == len(files):
        allc.append(c)
print("commits scanned:", len(commits))
print("commits holding all 20:", len(allc))
with open("all20.txt", "w", newline="\n") as fh:
    fh.write("".join(c + "\n" for c in allc))
for c in allc[:3]:
    print("  ", c)
for base, blob, p in files:
    print(base, blob, "held in", len(held[base]), "commits")
```

Where those commits lie (`fp.txt` is `git rev-list --first-parent master`,
`master.txt` is `git rev-list master`, `all20.txt` is the scan's list):

```
$ grep -c -F -x -f master.txt all20.txt
137
$ grep -c -F -x -f fp.txt all20.txt
132
$ grep -n -F -x -f all20.txt fp.txt | sed -n '1p;$p' | cut -d: -f1
1256
1387
$ git -C "$HB" log --no-walk=unsorted --format='%h ad=%ad cd=%cd %s' --date=short 692cba3 2e85a17 a015d79 | cut -c1-94
692cba32 ad=2024-08-12 cd=2026-04-02 [expdist] set the scale size properly
2e85a170 ad=2024-02-19 cd=2026-04-02 [layout] set treeSize as a compile-time value in the CUDA
a015d79a ad=2024-08-12 cd=2026-04-02 [jacobi-omp] reduce the cost of atomics with the OMP redu
$ git -C "$HB" diff --stat 692cba3 a015d79 -- 'src/*-omp/main.cpp' 'src/*-cuda/main.cu' | tail -2
 src/jacobi-omp/main.cpp | 62 +++++++++++--------------------------------------
 1 file changed, 13 insertions(+), 49 deletions(-)
$ git -C "$HB" rev-list origin/HeCBench-OMP > omp.txt; git -C "$HB" rev-list origin/HeCBench-SYCL > sycl.txt
$ grep -v -F -x -f master.txt all20.txt | grep -c -F -x -f omp.txt; grep -v -F -x -f master.txt all20.txt | grep -c -F -x -f sycl.txt
137
137
```

Position 1 of `fp.txt` is 7d2d3c5 (master's tip). The 132 first-parent
commits are one unbroken run, positions 1256 to 1387, from 692cba3 (newest)
back to 2e85a17. The first-parent commit after 692cba3, a015d79, changes
`src/jacobi-omp/main.cpp`, and no later first-parent commit holds all 20.
Master's commit dates read 2026-04-02 on commits authored in 2024, so
master's history was rewritten then; the 137 commits off master are on the
branches HeCBench-OMP and HeCBench-SYCL, with their own commit and tree ids.

### Support files

```
$ git -C third_party/LASSI grep -n -E '#include +"' 74b4681 -- translated_code/input_codes
74b4681:translated_code/input_codes/HeCBench/entropy/entropy-cuda_main.cu:6:#include "reference.h"
74b4681:translated_code/input_codes/HeCBench/entropy/entropy-omp_main.cpp:6:#include "reference.h"
$ for c in 692cba3 7d2d3c5; do echo "== $c"; for d in atomicCost bsearch colorwheel dense-embedding entropy jacobi layout matrix-rotate pathfinder randomAccess; do for l in omp cuda; do git -C "$HB" ls-tree --name-only $c src/$d-$l/ | grep -v -E '/(main\.(cpp|cu)|Makefile.*|CMakeLists.txt|LICENSE)$'; done; done; done
== 692cba3
src/entropy-cuda/reference.h
== 7d2d3c5
src/entropy-cuda/reference.h
$ for c in 692cba3 7d2d3c5; do echo "$c $(git -C "$HB" rev-parse $c:src/entropy-cuda/reference.h)"; git -C "$HB" show $c:src/entropy-omp/Makefile | grep -n -E 'reference|-I'; done
692cba3 112810d1560ceacac9e9d57a94285bfc54a2b1bf
27:CFLAGS := $(EXTRA_CFLAGS) -std=c++14 -Wall -I../entropy-cuda
55:%.o: %.cpp ../entropy-cuda/reference.h
7d2d3c5 112810d1560ceacac9e9d57a94285bfc54a2b1bf
27:CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../entropy-cuda
55:%.o: %.cpp ../entropy-cuda/reference.h
$ git -C "$HB" log -1 --format='%h ad=%ad' --date=short -- src/entropy-cuda/reference.h
8fc336b5 ad=2023-09-21
```

## Results

Blob ids are from the pinned upstream tree. "Held" means HeCBench holds the
same blob at `src/<app>-<omp|cuda>/main.cpp` or `main.cu` at that commit.

| Upstream file | Blob id | Held at 7d2d3c5 | Held at 692cba3 | Model-facing source |
| --- | --- | --- | --- | --- |
| `translated_code/input_codes/HeCBench/atomicCost/atomicCost-cuda_main.cu` | `d0f36289ec57ae0b8d301373d65282c1f00d2663` | no | yes | HeCBench 692cba3 |
| `translated_code/input_codes/HeCBench/atomicCost/atomicCost-omp_main.cpp` | `7851d6da8d9f72074a5a78600b8abb09c9931f75` | no | yes | HeCBench 692cba3 |
| `translated_code/input_codes/HeCBench/bsearch/bsearch-cuda_main.cu` | `ba21f7cf1105dc3bfc4a4bccb606b6b27076c3b9` | no | yes | HeCBench 692cba3 |
| `translated_code/input_codes/HeCBench/bsearch/bsearch-omp_main.cpp` | `bf485da7969bc5fd439a4ac12efb26501bc4e0a9` | yes | yes | HeCBench 692cba3 |
| `translated_code/input_codes/HeCBench/colorwheel/colorwheel-cuda_main.cu` | `714e36eb096f72bdb341bd581dbb3cf375cf8a93` | no | yes | HeCBench 692cba3 |
| `translated_code/input_codes/HeCBench/colorwheel/colorwheel-omp_main.cpp` | `cf865d1c607f428af9697688756fbbfa48d88091` | no | yes | HeCBench 692cba3 |
| `translated_code/input_codes/HeCBench/dense-embedding/dense-embedding-cuda_main.cu` | `3ee82f7ff5daf0cca3af269008edafefc1664be8` | no | yes | HeCBench 692cba3 |
| `translated_code/input_codes/HeCBench/dense-embedding/dense-embedding-omp_main.cpp` | `0dd8a571a1fd612e9fc29c97368ee1dc099a361d` | no | yes | HeCBench 692cba3 |
| `translated_code/input_codes/HeCBench/entropy/entropy-cuda_main.cu` | `3d54ed880fc8ce3ee6483bbb7414969fa6692234` | no | yes | HeCBench 692cba3 |
| `translated_code/input_codes/HeCBench/entropy/entropy-omp_main.cpp` | `273fb27df98db883e7180a3e4737b20981864974` | yes | yes | HeCBench 692cba3 |
| `translated_code/input_codes/HeCBench/jacobi/jacobi-cuda_main.cu` | `778edd2abfb2386cb4d42696d672427ef1cb9987` | no | yes | HeCBench 692cba3 |
| `translated_code/input_codes/HeCBench/jacobi/jacobi-omp_main.cpp` | `e920b53cb803e2f9f55e55520d7c269c50c993b8` | no | yes | HeCBench 692cba3 |
| `translated_code/input_codes/HeCBench/layout/layout-cuda_main.cu` | `2ee1a77eb1ce7542de152cd977b68f4bd7cf3ac5` | yes | yes | HeCBench 692cba3 |
| `translated_code/input_codes/HeCBench/layout/layout-omp_main.cpp` | `b0d64f2900734071965b8c9c84ab8d1f6dcfcd85` | yes | yes | HeCBench 692cba3 |
| `translated_code/input_codes/HeCBench/matrix-rotate/matrix-rotate-cuda_main.cu` | `8edcba8acc015d7e73b87c117413d2706602cac4` | yes | yes | HeCBench 692cba3 |
| `translated_code/input_codes/HeCBench/matrix-rotate/matrix-rotate-omp_main.cpp` | `a67ddc5afebf1fe1bda90dbfec75299fc7b070f2` | yes | yes | HeCBench 692cba3 |
| `translated_code/input_codes/HeCBench/pathfinder/pathfinder-cuda_main.cu` | `18230cfe044ba3c35d2b2ad8bbd90ed8ae9c1bed` | no | yes | HeCBench 692cba3 |
| `translated_code/input_codes/HeCBench/pathfinder/pathfinder-omp_main.cpp` | `b298e20731fdd3a6d6bf365d48fe3611d841071e` | no | yes | HeCBench 692cba3 |
| `translated_code/input_codes/HeCBench/randomAccess/randomAccess-cuda_main.cu` | `afc9d5088f3ad26d2fe8163759fcaf39b7eb676c` | no | yes | HeCBench 692cba3 |
| `translated_code/input_codes/HeCBench/randomAccess/randomAccess-omp_main.cpp` | `c038377f19db37086b6daf38726300b73fd11282` | no | yes | HeCBench 692cba3 |

- HeCBench 7d2d3c5, the P0 pin, holds 6 of the 20 files byte for byte
  (bsearch-omp, entropy-omp, layout in both languages, matrix-rotate in both
  languages). HeCBench changed the other 14 after 692cba3.
- 274 HeCBench commits hold all 20 files at their HeCBench paths; they are
  listed below. The newest of them on master's first-parent line is
  692cba3 (authored 2024-08-12); upstream LASSI committed its copies on
  2024-09-23.
- The only support file any upstream `*_main` file needs is entropy's
  `reference.h` (bible, Source Papers, LASSI quirk table, entropy row). It is
  the same blob at 692cba3 and at 7d2d3c5.
- The remote test `test_spike_hecbench_claims_hold_in_a_hecbench_clone`
  rechecks this table, every listed commit, and the support-file paths
  against a blobless HeCBench clone on alpha01. It passed from a dirty
  snapshot (rx 20260923-234734-desktop-8r113ei-p1-faithful-d670,
  exploratory).

## HeCBench commits holding all 20

274 commits hold all 20 upstream blobs at their HeCBench paths. On master's
first-parent line, 132, newest first (692cba3 back to 2e85a17):

```
692cba32c5744f6ef024cca59f65e9488edba8bf
3abcff49456177402030b639a6a6f2a76db7f1a1
85b14743a45d65a1f9c0d7b07b00a716a2fdc001
1c7b33841990bdb969c8505b5b4da4303dfaabc8
c37f33be002991d05b21cb6e24237b4b4ba480c6
4eb0d5f6ac82cda8ce6f7c3be507192a740c0316
b75ddc5e3e5d625e5ec876bf26697a1d119ba786
dbcf6438638645858d4c501a79a12c5688f29dcc
043355820ed656e9889ba17669a6251f5df3c1f2
7c18b7fb9eb909148655c6a32a2586b7a522842e
43014946bdaef3ec45a5bca6dcd849f7881744b0
ced5b90a6bfdcbfae8c24d18168bbc25ce35e122
c25cf5c7c7aee6b21e7ced177e29613e2840acbc
3b1698682b13646a0b73aebca188639ea383c52d
d267007283c861e77b5d88e5a13f1aa5ae883c9e
a261143e3562f26cf61605f9ab6b0d70251b1a42
3f12fe3c2e7c2e3e41637edf36e4ef12313cf427
6200ed0e257257a3646dee8166886fb97515d9bc
ee68f32128580254352002e8112ea31a73dce2c6
d67953553865cdb5958f48f419dacf3de14e4017
c831a132030add37319472312b37ef30074ecbf7
895073c194669138afdfde28a70b93ef2b4d229b
7985e844fc58fdb411d46bea5283f14ddc54faf4
039d8cc5a0b5f35fae9335de17e4e78b810de4ba
053ebae3cbf3c5e371f48fd7203b95a6c51fa1d4
cc56802d3f8e33f8c8f188a756791e186cd1ee68
ff82becde817efd59d60c14059e0a4b0a4b1e368
d672e82cc1c96ea599d6e0a819798c92fb13c1f6
424c45b12d7ed2339521a4c4543da8202cb3ecec
36980fdc3d34e9ba30b8d72da5d866529a11aa4e
1ec416e37e43721017d5e6f5f08f7753c9349fc3
dc0283dc809a2b0bec6216ff80d4d29e2a97a6e0
b09e09d9e8d1e1276b074780389b2f304b27c4c6
04f78fc912adb0c97f9acf983fa686984befa554
56c0ae38439101bb289e57f2aa21a00bbdcdea1e
6b2da6e1d3785cffe365e5dc99521af8d76d9494
e23cc6ea1e5c519bed95a342673b72665a657884
7f065c1fa494df9858646886386a1b696dd67cd5
00e8ec1af1df0b922ffd6798882a6e93b108c81c
1fa7dcdab3de16924b200441ee5a52b6e8085256
317c4c9d7fa3fb2f8eda8bf4b70b925a85c520e4
199f29758c6c1bec706eb176dc52cada2bb4da36
e53186216ac234110bf8dfa86a75294599ba53e1
05bd75e08224fa61038b670007501236ecc46c97
9f5df26792fe09d620e14be9b7b14812fdfd188b
98cf08ff7599ffc5adb42a10fa4469166709e047
3df7fbc6a1a904989527ceca4efbf27e29d98f91
60e407757ee5d36fc45bd436b8755a857995dd20
072420d21030f4b4b34c72a040c73865e46f8e26
342495aa0909e7b0152f27166ac4193b8c01d816
dd7681043bb3683c287363067fe4bee514bf8eda
8adb1752b148cc0d6d642c8a66722a45d6ef6573
cec28e10dd530d1a883e9c997d311cf58953eaf8
bea2028147f244fc35f756a3e7005b6c7595febb
e4a6244621dc90ddf4ca0877dc2abda4a97b5b13
41711886557ba9d188854545b26ab281b3a5f7c5
abbd3cb15f5f9fac7dae651b2fe13e2a87aa1d97
c99f3518fbbdef9e0e107f60617b17b51831f0a2
25990885c6cd13314d971ea0320264269ae9971e
193da8c830f4643b480e9c7812059be62ed3f08f
00a8c3c0cc92967c535b64122635e28ec18167c2
8c621caeac793f173ace1341235c3d6397286da9
fa2e2a18e0ac922bc89ef0098a473ffecdb99d9e
8cf367289c48148146dee3d352fa94bf0cea5676
12b76576e4a1b02f9cda3658fbd8a2f885160e88
24c23e185baf96eb8db2050c1e7114ef49e6e49d
672b2dba1b3063f19ca3c6270d26cb13fdfd548e
d69218d26de767cd0595cb17003a56c5b50a923d
7f9a070e772b3c57adeeea4414e253c7abf97bd9
4434dcf921a35f04698b999029409d894d817b19
49a8bda49b5781043b0c1bd59c4c135854c5582a
7b522fb7fd7273dc80933319cdb9664ce59bf2d7
e24b9b43e77ae5c6762db311987e96653305e9ba
d044b2fb944d61e6ffbd7835fe234e18932a5861
92f413276e0cf72640dd784948e74851ba2b8e7a
1ee8c2d95eb1011199344d46e1d13e39ff223675
44bc3a1676abf73201e113175386cc0c226294cc
b0bf5115c6192735072fe942089e380ed78e6856
b42737e55541025e3644c5e0da2fe674699c58ff
e10f7a0f5ecfd5190f7583095eb0321c587e8100
d000edcac7fba6f48f2623c36514724f7d260c5b
86e143331120ff2619533e52fc3701f46d084e3e
d875f04e04232f947ee0c7e6329e64c8f0c48513
cab9b1662e0c9bf6d6ad2bbe06979b99fc2c4d50
c9d5f16df0b479f2016926c41bcd9f34328b51c0
c5d378c88d85a5ab26690dc0209fbb328dbbe415
664868c8682285119cfd778df7e643bd82487fcd
1fdb687bdc48c55cce9617f580dbf17b994607c0
eb51bb7b21e6170a97f775fe1a8ac429d290d19d
a694b590e0eb29ac2a019f57e27e0069a8ed3c53
c0ee7267879c84f5b705bd3fb53649021d9e1d1d
5166dff80833702c2e3ba0900188dc3582d7ef68
f15ab39e29f1d9e42031bccc5875c76bec07826a
d1c525688fffa00f024f7107d974a4993b8db19d
e8cecb5c454bbbd9962103e706903d6cf65cad35
9b0c742495985442777aa3dcacdefdfe8aa26868
1208c04cf9a0825aac285ae76bba27808ba84f32
f9f95af2953307fae5daba0378a0a88e8c0d7d9e
d443f6107ade4acd949f39f494b05bfda6fcb09a
585513e47f39064fdaa782e43ffad07d200e9735
25a08f14aeaed7dd9373ec9562a62ee804cc32dc
e5e477086f814e0bf78665905b0c7da9294a535d
115103e03c7b754ea70aa6ee4dd233a64bed4cdd
08a191e1459837254d5ffd86b1017e44b4e7b5eb
b596e6fce1e7dabd3a56ceb452f3c1447b9511a4
24cffce00477f963bf31a9c16e3adda6842b17fb
abc47e05b796f8b511122de39bc9a33c45b8dc33
bb4d40f9e3f32bcdf3756f372d700582070d0c4f
240a2c421e59f211d5c44c5c52f04cb688ddd975
a4a450fb69dbfc6c032a35624397925203b3d336
36776dc84d61c2bba5b48acca60e874b62cb9afe
518b94c7636ec07739df0a81eebd90843b65af1a
b6f024be3d153e3f0739ec878874e1292a79ef5d
ef7b188c054712f7a7a82cbc9912b365ac30712a
045b62511a8ac8b62e64159b9c1890c466748c55
0ba8828705558a07298d7f2c64ce1bf47963402a
ee1be2b718cdb2f2f4281bcce8de05c536d96158
df8dcb8c2abe0bee3d903105895f0e2796bc3d1d
c2e9388bca72afefccc3cf2deba160a85bd51ba9
6fb5dfe7ea9782f0b19117c4fe555ee5018df161
83e37b0761a4d19dffcabe27991623b3540c4b1f
ceef258cd6e3d6bf0151f2171ab2a3de42f950d3
f38429cdc52087e5b944a93f74f55c275dc52ffe
7b01a2cebc1d110f379854fe10a00dc52682c526
91a9e54716125b90910ae457c6a3f6fae36e042b
6b4adf9ecf6469088cd173f350f26c74fc0f42b4
62cc6634f94161e9907019278553c1b9ba008efe
29474c25c2efc0543586b69539f37361148e9366
d894268285b3cf91c938498342959ce54ff188c9
79cc39afb24ffed242d6eeea3b3922bab9db1958
ab93dee7a5038c49f366765b4c2d7ebcc7519aa7
2e85a17062e60f265f1f6dc794ca06b767a3b733
```

Reachable from master but off its first-parent line, 5:

```
63817f6030a976102c2c634e171a48a055dad211
2fa4e965b0fa4ecda62a4b0d82497ee82bf5af70
536959abb28579b80dfc186d003ff1c86bb8ee2c
ad5223d99c65507ef8c2811eff001823f65e051c
c58eb482e5f7f83e8378c974e6dca6ac76e5576a
```

Not reachable from master, 137, all on the branches HeCBench-OMP and
HeCBench-SYCL:

```
e79d1895fcd2a5094ff2f7ce03e95957d36aa7dd
105931744023f57b2364b0297dd3ed70f0929965
741a0cc38d723ff7c36958bec2cf32009e60ec02
6d09947fe5df36cdae5f4a0beea164a2956c1846
b7d1ecd25a9837eb03e4f982876ab38f838c5743
635ce6b45e24dbe3ff6b575d2d66be5e053addb6
321ababefa2af97862832c5b03e6b9e87d16788d
235eecaa7f74d7274fef9503cc98e023cd6e9f54
2cc8809c5be633f32b50cd62897c2a8ab514ed19
ae85390c22f181e7ae1a1e4c9374625f0dc7de1c
d1ab7e2c8c64e6559e9a0fbe762b70cf28829c03
d047b1779aa4c0e61ad8ef7c7e91803bc049c176
74c8387d38e63f84c1c3eb81d4cd0766f4b439c7
56ec807787f13a9888f5252bd6bad9c13706a130
f0535879583244ca87a942d0efa4c64e2ab9ce96
aad74973f5ceb1e059829e46725f8dbd573cd546
c255431610c195d4f7203bad069eb39b8766c05b
8811e2366585e4dea67de3ffca9cd5bca5744b8c
486205aa17eb4fe8ce423829c0a5cd242fcbf9c1
57be46fd33437e0b0af783f8b141531f6fb06d49
2a6adcd75f0b09515bca424c85641b7faa85b342
f959cdfa5c5461de32638a5693eca2dcfec0ba16
96e77f262d08fe25bb165e36dc5f5d4f07c27178
018fb4a0c4ec0db65876f30fccacea833d145d46
0468555cb767fdc497e205e3b45b5ba20deaf29a
e2c592e62597c03152df36d7cc9fed5ced9fe5c8
37933b871add6c09962d85aa6fb6a14078f40b4d
4c0ab5cb9716b29ac3b9f79d880adae0864f093c
8b7575379ffe38b4fcec071e0a9a0affbae07648
1623ff83ebe744303918c39c5429ffec786d3ebe
f47befbc46cb4f464f6acb754747c9afe811a0e3
f5f07f3ed044c638316937d9163754a540c8fa12
7c54a2f367fd312a775f03bd4c685e34cfbed0d3
91436668a07e96e2ceb5da5722636e84515f63a0
e0f8b3f28231831b447a4dd5bf5c44909298231c
3a0d90c45d3495e9e7c75dd1bd386fead48d34b2
09b8555ab130d344ea88af0310d815ea1229a44d
24a396d1d232459479bc9af99f35e2cf95029910
a2926235458eecbaf026885fd6a618c318b8e1ee
573270f40c3fa8a88f33b5f73d7bcf5e486ac52c
a959792d29717ea48259a99b3deb6debb19f6619
ba8f7b4511fce6835729e7e48593252d8371e427
e68dc4665e4786f3b83fdce35db95dbb2f993002
f1f280adea0c29e0b8c6b8e20315e619138e9391
93090c87251728eec03e8bd79788f2321e5c7e26
882bd90126534d20daf465907f13eef3d9fb05bc
3612daa0cffed58bdda1893a08ee64405baf600e
9b12ee6fb904214a41f684eb880cb22d9d28cf27
5e7005d5a22d6ba7698c704b4b52a8902009063c
63753bfd48cab35e77859d9261e1465e94cd6d28
50bbdee8e01a8d52830694758633b2753cbd9d6f
88c5bcb94f68003ba304b6d373f9a4641dee5e2e
7376855a9fa57fa5ee4158446781de63d646ff14
17545ee567d515599d082f36f6be2d4190e5f9f9
f00b8dbacff731bbcb71e69e634f869a1397a332
d8f060f4836c5b2894fdbf47cbb818aceb902a45
0bead9d88a89e2cdbbb6731091dda58b02e67580
b2b014134fd9fc9de36fbb2b48ed8c3041d33a74
980d2fe61d35cb37c5914acd630cc0882dac07b1
76f00c1a0bc2c8d93293f9424a1e518097aa1cf9
bac294e5727a1d8eb74acd633438dd343267031a
ebe7d624798a7c574af24c4cf856bec3d7500b21
ca1565782187ee0a18b4e786519bce64658aec05
7071a6d624b753ece005b00ced9c44c94fcf9235
a22ffe95e20cbc265619ddfcde919f1ce7ee8404
8643a86ca6aaf7a0c7eff7d5c6b1d5ea822af669
03c9b1d267447bf6022979294ce407dbcf3732ab
0b71dbe1c73367923e63601271d3be088d4f1297
11b1995a72de5a207d8ed8df424f3b348263ed5b
334b23df476c42dd6563353bebd2313ceb9ed674
879bad1a53531465bb796a0e0dd4f8ddba61bb4c
152817aee5e3bfe035c59d249caa8cc6882c9204
608bb1d0180502634de0a42869aab254a35e51b0
afc30dcba1dc48fa274627075e5f48d3367332d8
32c14a55ea45c981b495cfc33203f19ba5a4d84d
495273cbb00e5131febec692d10c375523de63d2
a07a108c9965702f6c9485c7fa1da4c2d38bf9bd
fcff2dcc102258de25659eb6dda7e028dd4d5141
99af37419495b150ed034337cbf516ff11816200
a53540539afe810c9a9a1fef934263c59e22db03
54b6a35bb7594fefbe7a8e49a2a32cd77f0e71ba
f0db10ee2c0f08918eed102c37eafa054244a48b
a04169edef34d8b5b69bcb14db70e7cb7ac84d90
9293abb39df29df492efa2dab99251bb793122a0
27faa34408bcf7368dd5e220415c32af1708460e
eed8ecc8f87a128ffcb76fa6eede943d30992192
94d1eeda67fdf797f31bf39521f603a877abbb19
2c9d15e3d512a4451780666f050a8fd817f6dbdb
3b35b419e8370d94b0855b43b82bee17783d5605
3d31c647431c02e542f1f128ee49c0ac7944633e
a56ff7d452c0d737a00376f8e701401b6ad7691e
01399ee4f5b1b3bab1033d612e7739d6483e4674
b8a57105267316e9308ebc15c5c8208862a6f3ad
93b9a10d54d7cc2de71fb1f1e72c01bc35202def
809a15f69164f69eba541faf0544f479c925dc22
c498cc9b1a77d74c022a67772d4de7149a9cf090
2a92345393176f7610f071bf9ee6956d7969cdc2
6928efcc6a44e32047b8c0133b1f7c9213bef2b2
913e16c19d99f67f7e6450e8c1f77b5909c1b5f2
713666888344a9749beaf10780a047fa69afa2a3
51863a45a7476756238948742d20235e9787dcfe
ff3202fbd160f65b1051a54a6f1ee2e08e884cbf
f049a8b39194bdfe289977cf4abbfc996727f3cd
3bb79acb9434fe68fe367428eb513428264a80a8
03f0d5fd0b86824181377e07f1c1d1e30271428b
f90d772099307afa04963bd7e60b5d3e861cc32a
bfb86960532b9df8142967307e81f69ed058569b
dde50fadfff25ba0ae96ef3c02c2239458fdc976
086b4901a3caef6b8d004bd2e0bff426a3b527f1
b5b455d9ec58b5534c0a9923dbc917110b93994a
835c0c4c1d6dfcea06abb76c60fec14295f6f9f3
92c403be17c089863399fac0d5632fd09915418f
68ec56bb6609644721c7c35acb2b703b1f2b8715
3bd56818d593b15f31fa8fd908ec15a3a66f6775
06432afbc4c1d7fd2802c6afbef5fa780ced0d5e
a5f46685640eb84a685fe4a5261139cb93c03ce9
ecef12115796a10cbc82c113ac0c437c1e9e47d2
2644db9b392fcac154e07a999cdf52eb9272aeb0
8147dd85019b331a1d30d86e3e33e3a2152566d8
b400bb8df0aba96ec44441feac2d3d038a13c44c
cc3b92d09638a3d393d6f877e8230b55e1834ca5
2746d52eda1e577fe5aea77d8a8632ca19647d2a
8ef9c0fee5894b911276d464f69c74363647ca32
86bd0c5cd309b838e6c47a95ba6edf75f3ddda7a
927f02a64fff26d7ea6ddad76e38043f19834be9
d2badf31211124c4776cee02b031b286b697e484
3f0ef07cc7c325cb4c63f17e789c29c563bd1774
e5dceb7a0872256886e1f58723d4d0f1bb789fbd
4c86eeb1747d04e317758a4608b741dbb60b5f68
cb787f185f16c202d5e2c66f7327ef4236c576a1
38ff682d44a492144dd89d5a77c31f44309bae75
e67ea888fea1f34b6e1e10bb3dc998cf55d27cd3
ccf333e17d5316b84e4d8834435d647a8399f5ae
2bfd031f3addc25cc6572525834ae0cc6f4dec3c
a138cc7a037be8b151930402f14ea7f798092cda
2c1b8c4304f0a475d1662f568d99eaaceb3cf0a7
54ed1477719142b80c6ecf1f4c5ff0c01f70ecc0
```

## Support files

HeCBench pin for support files: `692cba32c5744f6ef024cca59f65e9488edba8bf`

- `src/entropy-cuda/reference.h` (blob 112810d, unchanged in HeCBench since
  8fc336b5 of 2023-09-21, so the same at 692cba3 and 7d2d3c5). Both entropy
  sources include it, and upstream LASSI's tree lacks it. HeCBench keeps it
  only beside the CUDA source; its entropy-omp Makefile reaches it with
  `-I../entropy-cuda`. P1.2 places it in the build directory of both entropy
  languages as a harness file.
- No other upstream `*_main` file includes a local header.

## Decision

- Pin upstream LASSI by `assets/upstream/lassi.yaml` (url
  https://github.com/SPEAR-UIC/LASSI, commit
  74b46812523f2ff79b53b6880a4521690d7478b0, path `third_party/LASSI`) and
  `tools/fetch_upstream.py`, not by a submodule (Upstream pin, above).
  Nothing under `third_party/` is tracked; `.gitignore` ignores
  `third_party/LASSI/`. This is a Decision Log entry. The P1.G gate commands
  that start with `git submodule update --init third_party/LASSI` run
  `uv run python tools/fetch_upstream.py` instead.
- Re-pin HeCBench from 7d2d3c5 to 692cba32c5744f6ef024cca59f65e9488edba8bf
  (plans/PHASE-NOTES.md, P1: "re-pin if they differ"; they differ for 14 of
  20 files). Faithful mode needs upstream's bytes (Agent Rule 4), and 692cba3
  is the newest commit on master's first-parent line holding all 20, so every
  model-facing file and the support file come from one tree. It is reachable
  from master, so the shallow fetch by id that `tools/fetch_bench.py` uses
  works for it (checked on the workstation: `git fetch --depth 1 origin
  692cba32c5744f6ef024cca59f65e9488edba8bf` into an empty repository, then a
  detached checkout, gives that HEAD).
- Model-facing source: HeCBench 692cba3 at `src/<app>-<omp|cuda>/main.*` for
  every app and language. Each is byte-identical to upstream LASSI's file at
  74b4681 (same blob id); P1.2's manifest records the sha256 of upstream's
  file for each, so a fetch that differs fails.
- The pin change is a Decision Log entry. P1.2 applies it to
  `assets/bench/lassi-hecbench-10.yaml`. P0's layout evidence at 7d2d3c5 still
  describes the same layout sources: both layout files are the same blobs at
  7d2d3c5 and 692cba3.
