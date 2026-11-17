"""Read-only, for the Tracy profiling kit: the Tracy install record, the cards' IOMMU types, hugepages, quarantine
markers, and an inventory of exact-match trials whose target is a TT-Metalium program (tt, ttkh, ttkernel).
Opens no device node and writes nothing. Run from the gate's exec root.

Task P17.65: this is the text run through `rx exec` as rx 20261116-161608-exec-0f2e (output:
inventory-rx-20261116-161608-exec-0f2e.txt), with this paragraph added and one long line wrapped for the linter."""
import glob
import hashlib
import json
import os
import time

T0 = time.time()
ENV = {k: os.environ.get(k, "-") for k in ("LASSI_TOOLCHAINS", "LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_SHARED")}
print("##### env", json.dumps(ENV))


def head(path, n=3):
    try:
        with open(path) as handle:
            return [next(handle).rstrip() for _ in range(n)]
    except (OSError, StopIteration) as exc:
        return [f"unreadable: {exc}"]


tc = ENV["LASSI_TOOLCHAINS"]
print("##### toolchains", sorted(os.listdir(tc)) if os.path.isdir(tc) else "missing")
for name in ("tt-metal@5280a9cf-tracy", "tt-metal@5280a9cf"):
    rec = os.path.join(tc, name, "lassi-install.txt")
    print(f"##### install {name}:", " | ".join(x[:200] for x in head(rec, 4)) if os.path.exists(rec) else "no record")
cache = os.path.join(tc, "tt-metal@5280a9cf-tracy", "build_Release", "CMakeCache.txt")
if os.path.exists(cache):
    with open(cache) as handle:
        print("##### tracy cache:", [line.strip() for line in handle
                                     if line.startswith(("ENABLE_TRACY", "TRACY_NO_BROAD", "TRACY_ONLY_LOCAL"))])
tools = glob.glob(os.path.join(tc, "tt-metal@5280a9cf-tracy", "build_Release", "tools", "profiler", "bin", "*"))
print("##### tracy tools:", [os.path.basename(t) for t in tools] or "none built")

print("##### cards")
for node in sorted(glob.glob("/sys/class/tenstorrent/tenstorrent!*")):
    addr = os.path.basename(os.path.realpath(f"{node}/device"))
    grp = f"/sys/bus/pci/devices/{addr}/iommu_group"
    gtype = head(f"{grp}/type", 1)[0] if os.path.exists(grp) else "no group"
    print(node.rsplit("!", 1)[1], addr, os.path.basename(os.path.realpath(grp)), gtype)
hp = "/sys/kernel/mm/hugepages/hugepages-1048576kB"
print("##### hugepages free", head(f"{hp}/free_hugepages", 1)[0], "of", head(f"{hp}/nr_hugepages", 1)[0])
locks = os.path.join(ENV["LASSI_SCRATCH"], "locks")
print("##### locks", sorted(os.listdir(locks)) if os.path.isdir(locks) else "no locks dir")

runs = os.path.join(ENV["LASSI_RUNS_ROOT"], "runs") if ENV["LASSI_RUNS_ROOT"] != "-" else "lassi-runs/runs"
print("##### runs root", runs, os.path.realpath(runs))
paths = []
for direction in ("cpp-tt", "cpp-ttkh", "cpp-ttkernel"):
    paths += glob.glob(f"{runs}/*/*/*/*/{direction}/*/run*/trial.json")
print("##### candidate trial.json files", len(paths), f"{time.time() - T0:.0f}s")

seen = {}
rows = []
for path in sorted(paths):
    try:
        with open(path) as handle:
            trial = json.load(handle)
    except (OSError, ValueError):
        continue
    attempts = trial.get("attempts") or []
    if not attempts:
        continue
    last = attempts[-1]
    align = last.get("alignment")
    mean = align.get("mean") if isinstance(align, dict) else align
    if last.get("stage_reached") != "S5" or mean != 1.0:
        continue
    files = last.get("files") or {}
    digest = hashlib.sha1(json.dumps(files, sort_keys=True).encode()).hexdigest()[:10]
    parts = path.split("/")
    sweep, model, bench, direction, item = parts[-8], parts[-6], parts[-5], parts[-4], parts[-3]
    host = (last.get("guards") or {}).get("host_compute")
    kernels = sum(1 for name in files if name.startswith("kernels/") or "/kernels/" in name)
    key = (item, direction, digest)
    if key in seen:
        seen[key] += 1
        continue
    seen[key] = 1
    rows.append([item, bench, direction, model, sweep, str(len(attempts)), str(host), str(kernels), digest,
                 str(sum(len(v) for v in files.values() if isinstance(v, str))), os.path.relpath(path, runs)])

print("##### exact matches, unique by item, direction, and files:", len(rows), f"{time.time() - T0:.0f}s")
print("item\tbench\tdirection\tmodel\tsweep\tattempts\thost_compute\tkernels\tfiles_sha1\tbytes\tpath")
for row in sorted(rows, key=lambda r: (r[1], r[0], r[6], r[3])):
    print("\t".join(row))
print("##### done", f"{time.time() - T0:.0f}s")
