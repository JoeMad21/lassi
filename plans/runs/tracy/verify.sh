#!/usr/bin/env bash
# Verification of the Tracy build of tt-metal 5280a9cf (task P17.42), for the owner's window: on alpha01, from the
# repository root of a clean checkout (an rx run slot), with the gate's environment (LASSI_TOOLCHAINS, LASSI_RUNS_ROOT,
# LASSI_SCRATCH, TMPDIR), after the install job has finished:
#
#   uv run tools/rx.py job start --big --name p1742-tt-metal-tracy -- \
#     'LASSI_ALLOWED_ROOT=/mnt/nvme10/joseph_ufl LASSI_STOP_GIB=145 bash toolchains/tt-metal.sh'
#   uv run tools/rx.py run -- 'bash plans/runs/tracy/verify.sh'
#
# The install builds the pin's commit with ENABLE_TRACY=ON and Tracy's TRACY_NO_BROADCAST and TRACY_ONLY_LOCALHOST
# options at $LASSI_TOOLCHAINS/tt-metal@5280a9cf-tracy (toolchains/tt-metal.pin); the P4.2 install at
# tt-metal@5280a9cf stays as it is. It runs no program and opens no device, but its command names tt-metal, which the
# gate's tt_guard refuses while the cards are present and its tt_silicon class is disabled (tools/server/gate.py,
# check_command), so it runs in an owner window with that class enabled, or as the owner otherwise decides. PROJECTED,
# not measured: about 4 minutes of 8 jobs and about 3 GiB, from P4.2's 3 min 44 s and 3.0 GiB (results/
# p4-tt-install). Install it before toolchains/tt-xla.sh, whose venv links its sfpi. This script's own command names
# no tt-metal word; it runs every program on ttsim.
#
# This script opens no device and runs no program outside the sandbox. It reads the install's files, then runs the
# five tt-pairs-v0 references on ttsim, a simulator (never silicon; no wall time is performance, Agent Rule 2), through
# the remote test the P4.13 gate used. Each step prints "step <n> <name>: PASS|FAIL|INFO <detail>":
#   1 the install record: the pinned commit on its first line, the Tracy flags on its flags line, and the tracked
#     CPM source list (lassi.toolchains.ttmetal_build, as every run checks it);
#   2 the configure: ENABLE_TRACY, TRACY_NO_BROADCAST, and TRACY_ONLY_LOCALHOST on in CMakeCache.txt;
#   3 the Tracy client: its object's DEFINES in build.ninja, and its library holds none of the broadcast addresses
#     and environment switches that TRACY_NO_BROADCAST and TRACY_ONLY_LOCALHOST compile out (tracy 0aaefbb6,
#     public/client/TracyProfiler.cpp:1743-1764, public/common/TracySocket.cpp:457-461 and :478);
#   4 the device profiler variable is accepted: libtt_metal.so lacks the abort text a build without Tracy raises for
#     TT_METAL_DEVICE_PROFILER (tt_metal/llrt/rtoptions.cpp:675) and holds two strings that exist only in the
#     profiler's Tracy-only code (tt_metal_profiler.cpp:853, profiler.cpp:2326), as plans/spikes/tt-perf.md read the
#     tt-xla wheel's library (Finding 10);
#   5 INFO: the gate example's build line in the new build.ninja beside the pin's HOST_* words, which keep the P4.2
#     line (toolchains/tt-metal.pin says why); the expected extra words are -DTRACY_ENABLE, -DTRACY_NO_BROADCAST,
#     -DTRACY_ONLY_LOCALHOST, -fno-omit-frame-pointer, and -rdynamic;
#   6 the five tt-pairs-v0 references build against the new tree and pass on ttsim as before:
#     LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -m remote tests/bench/test_tt_pairs_remote.py, so a program that loads
#     the Tracy client inside the sandbox's empty network namespace still runs (its listener then cannot bind).
# A run on the cards with device_profiler kernel (the device time itself) is a later owner-started silicon run, not
# this script. Exit status: 0 when every step passed, 1 when one failed, 2 on a missing variable or core limit, 3 when
# the install is missing. Running it on alpha01 needs the owner's yes in the working session (Agent Rule 17).
set -u
fail=0
report() { echo "step $1: $2 ${3:-}"; [ "$2" = FAIL ] && fail=1; return 0; }
for name in LASSI_TOOLCHAINS LASSI_RUNS_ROOT LASSI_SCRATCH TMPDIR; do
  if [ -z "${!name:-}" ]; then echo "verify: $name is not set (the gate sets it on alpha01)" >&2; exit 2; fi
done
# The 1-byte core limit every build and run sets (AGENTS.md, Local Execution; toolchains/tt-metal.sh), read back.
if ! prlimit --pid $$ --core=1:1 \
  || ! read -r core_soft core_hard < <(prlimit --pid $$ --core --raw --noheadings --output SOFT,HARD) \
  || [ "$core_soft $core_hard" != "1 1" ]; then
  echo "verify: cannot hold RLIMIT_CORE at exactly 1 byte, soft and hard (prlimit, read back as 1 and 1)" >&2
  exit 2
fi
metal_pin() { (source toolchains/tt-metal.pin && eval "echo \"\$$1\""); }
prefix="$(metal_pin PREFIX_NAME)"
tree="$LASSI_TOOLCHAINS/$prefix"
build="$tree/$(metal_pin BUILD_DIR)"
dirty=yes
[ -z "$(git status --porcelain 2>/dev/null)" ] && dirty=no
echo "verify: commit $(git rev-parse HEAD 2>/dev/null) dirty=$dirty"
echo "verify: host $(hostname) date $(date -u +%Y-%m-%dT%H:%M:%SZ) tree $tree"
echo "verify: core limit read back as ($core_soft, $core_hard)"
if [ ! -f "$tree/lassi-install.txt" ]; then
  echo "verify: $tree holds no lassi-install.txt; run the install job first (the header says how)" >&2
  exit 3
fi

# 1. The install record and the CPM sources, through the checks every run makes.
if out="$(uv run python -I -c '
import sys
from pathlib import Path
from lassi.toolchains.pins import read_pin
from lassi.toolchains.ttmetal_build import check_cpm_sources, check_install_record
tree = Path(sys.argv[1])
pin = read_pin("tt-metal")
check_install_record(tree, pin)
check_cpm_sources(tree)
flags = [line.split() for line in (tree / "lassi-install.txt").read_text().splitlines() if line.startswith("flags ")]
wanted = ("-DENABLE_TRACY=ON", "-DTRACY_NO_BROADCAST=ON", "-DTRACY_ONLY_LOCALHOST=ON")
missing = [word for word in wanted if not any(word in words for words in flags)]
sys.exit(f"the flags line lacks {missing}" if missing else 0)
' "$tree" 2>&1)"; then
  report "1 install-record" PASS "first line names tt-metal $(metal_pin COMMIT); Tracy flags recorded; CPM list tracked"
else
  report "1 install-record" FAIL "$(echo "$out" | tail -n 3 | tr '\n' ' ')"
fi

# 2. The configured cache.
cache_ok=1
for key in ENABLE_TRACY TRACY_NO_BROADCAST TRACY_ONLY_LOCALHOST; do
  value="$(grep -E "^${key}:[A-Z]+=" "$build/CMakeCache.txt" 2>/dev/null | cut -d= -f2-)"
  echo "verify: CMakeCache $key=$value"
  [ "$value" = ON ] || cache_ok=0
done
if [ "$cache_ok" = 1 ]; then report "2 cmake-cache" PASS; else report "2 cmake-cache" FAIL "an option is not ON"; fi

# 3. The Tracy client: its object's definitions and its library's strings.
defines="$(awk '
  /^build / { inside = ($0 ~ /\/TracyClient\.dir\/.*TracyClient\.cpp\.o:/) ; next }
  inside && /^  DEFINES = / { sub(/^  DEFINES = /, ""); print; exit }
' "$build/build.ninja" 2>/dev/null)"
echo "verify: TracyClient DEFINES = $defines"
library="$build/lib/libtracy.so.0.10.0"
tracy_ok=1
for word in $(metal_pin TRACY_DEFINES); do
  case " $defines " in *" $word "*) ;; *) tracy_ok=0 ;; esac
done
if [ -f "$library" ]; then
  for text in 255.255.255.255 127.255.255.255 TRACY_ONLY_LOCALHOST TRACY_ONLY_IPV4; do
    count="$(grep -a -c -F "$text" "$library" || true)"
    echo "verify: $library lines holding '$text': $count (0 expected: compiled out)"
    [ "$count" = 0 ] || tracy_ok=0
  done
else
  echo "verify: $library is missing"
  tracy_ok=0
fi
if [ "$tracy_ok" = 1 ]; then
  report "3 tracy-client" PASS "no broadcast and no listener on other interfaces are compiled in"
else
  report "3 tracy-client" FAIL "a definition is missing or a compiled-out string is present"
fi

# 4. The device profiler variable is accepted by this build.
metal_lib="$build/tt_metal/libtt_metal.so"
if [ -f "$metal_lib" ]; then
  abort="$(grep -a -c -F "requires a Tracy-enabled build of tt-metal" "$metal_lib" || true)"
  never="$(grep -a -c -F "Device profiling never finished" "$metal_lib" || true)"
  noc="$(grep -a -c -F "Profiler NoC events are enabled" "$metal_lib" || true)"
  echo "verify: $metal_lib: abort text $abort (0 expected), Tracy-only strings $never and $noc (1 or more expected)"
  if [ "$abort" = 0 ] && [ "$never" != 0 ] && [ "$noc" != 0 ]; then
    report "4 profiler-variable" PASS "TT_METAL_DEVICE_PROFILER is accepted (the profiler is compiled in)"
  else
    report "4 profiler-variable" FAIL "the library still holds the abort, or lacks the profiler's code"
  fi
else
  report "4 profiler-variable" FAIL "$metal_lib is missing"
fi

# 5. The gate example's build line, for the record (the pin's HOST_* words keep the P4.2 line).
example="metal_example_$(metal_pin EXAMPLES | cut -d' ' -f1)"
awk -v target="$example" '
  /^build / {
    inside = (index($0, "/" target ".dir/") > 0 || index($0, "build programming_examples/" target ":") > 0)
    next
  }
  inside && /^  (DEFINES|FLAGS|LINK_FLAGS) = / { print "verify: " target " " $0 }
' "$build/build.ninja" 2>/dev/null | sort -u
echo "verify: pin HOST_DEFINES = $(metal_pin HOST_DEFINES)"
echo "verify: pin HOST_FLAGS = $(metal_pin HOST_FLAGS)"
echo "verify: pin HOST_LINK_FLAGS = $(metal_pin HOST_LINK_FLAGS)"
report "5 host-line" INFO "compare the lines above; any difference beyond the five expected words is a finding"

# 6. The five tt-pairs-v0 references on ttsim against the new tree.
if LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -m remote tests/bench/test_tt_pairs_remote.py; then
  report "6 tt-pairs-v0-on-ttsim" PASS "the five references build against $prefix and pass on ttsim as before"
else
  report "6 tt-pairs-v0-on-ttsim" FAIL "see the pytest output above"
fi

echo "verify: done, $([ "$fail" = 0 ] && echo "every step passed" || echo "a step failed")"
exit "$fail"
