# Spike: the host skeleton's first compile at the pin (task P17.33)

- Task: P17.33 (plans/p17-portable.md). Bible: Project Recipes (Notes, Skeleton), Toolchain Pins (tt-metal.pin,
  ttmetal-host). Approved in the working session of 2026-10-25 (OQ-092's answer, recorded on branch p9-ee: "One
  skeleton build").
- Exploratory: the run was from a dirty snapshot of this worktree, so it is never [MEASURED] (AGENTS.md Results).
  The compile from a clean commit stays open in P17.33.

## Command

From C:/dev/lassi-p17 at 2026-10-25 22:15 EDT, with assets/skeletons/tt-host-skeleton-v0/ and
plans/runs/night/skeleton_build.sh uncommitted:

```
PYTHONIOENCODING=utf-8 uv run tools/rx.py run -- 'bash plans/runs/night/skeleton_build.sh'
```

## Output (rx 20261025-191541-desktop-8r113ei-p16-adversarial-d4c2; the command line cut at 220 characters)

```
[rx] dirty tree sent as snapshot d0e1c8a282d2 (exploratory; not reportable)
core limit (soft hard): 1 1
commit: d0e1c8a282d2eea85d6e4766f4322df9c2c9b23d (0 changed paths)
pin: tt-metal 5280a9cfb00998fd49667a29523d03aee905c129; tree: /mnt/nvme10/joseph_ufl/toolchains/tt-metal@5280a9cf
host.cpp sha256: b29049b7f7c043258446af352f43212dbb177ce19dae425b801d90ad92209b63
compiler: /usr/bin/clang++-20 (Ubuntu clang version 20.1.8 (++20250708082409+6fb913d3e2ec-1~exp1~20250708202428.132))
build directory: /mnt/nvme10/joseph_ufl/skeleton-build.aegjXo
command: /usr/bin/clang++-20 -DENCHANTUM_ENABLE_MSVC_SPEEDUP=1 -DFMT_HEADER_ONLY=1 -DSPDLOG_FMT_EXTERNAL -DSPDLOG_FWRITE_UNLOCKED -DTRACY_IMPORTS -DTRACY_TIMER_FALLBACK -DTT_ENABLE_LIGHT_METAL_TRACE=1 -I/mnt/nvme10/josep
compile exit status: 0
artifact: main, 64912 bytes
error lines: 0; warning lines: 0
---- first 40 lines of errors ----
---- end ----

[rx] id=20261025-191541-desktop-8r113ei-p16-adversarial-d4c2 rc=0 state=done
```

## Reading

host.cpp as it stood then (sha256 b29049b7...) compiled and linked against the pinned tt-metal with the
ttmetal-host command line, with no error and no warning. The committed host.cpp adds one comment line above main
(sha256 5756f7da..., MANIFEST.yaml), so the clean-commit compile is still to be run. Nothing was run on ttsim: the
skeleton's placeholder kernel paths hold no kernel.
