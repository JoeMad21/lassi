#!/usr/bin/env bash
# Local dry run of the P5.3 batch (plans/spikes/p5-tier-b.md, Dry run). Workstation only: never on alpha01.
# Needs Linux (WSL is fine) with user namespaces and a systemd user manager, git, network to github.com, and a
# venv holding the slot's packages (pyyaml, pyarrow) and numpy 2.2.6.
#
# Usage: bash dryrun.sh <case> <new dry dir> <venv> <codec checkout at 81817eb>
#   case main:       the full batch with injected faults (faults below), wall limits scaled by 0.05;
#   case order:      P53_BUDGET_S=440 and a hang in the baseline, which runs last, after the multi_tile stages and the
#                    "asis" runs, so the hang starves nothing;
#   case budget:     P53_BUDGET_S=390 and a hang in the first multi_tile stage, so the budget skips later builds;
#   case stop:       a hang in the first multi_tile stage, and the run's STOP file appears 25 s after run.sh starts,
#                    during that hang, so every later build and run, then compare and the scratch du, are skipped;
#   case stop-fetch: the run's STOP file appears while the fetch runs, so run.sh's check before prepare skips the rest;
#   case stop-batch: the run's STOP file appears as the batch step starts, so tier_b.py's check before the probe
#                    skips the probe and every build and run;
#   case size:       a 1100 MiB file appears in the run directory as the batch step starts, so the run directory cap
#                    skips every later step (the dry run removes its own file before copying the run out);
#   case dirty:      an uncommitted change in the copy, which preflight must refuse;
#   case copy:       no faults, and run.sh --alpha01-copy on a changed clone of the codec in the fake scratch root;
#   case copy-fail:  as copy, with a file the account cannot read in the clone (a bind mount of /etc/shadow, never
#                    read), so the copy comparison step fails and the run goes on.
#
# It runs the real run.sh and tier_b.py, the real lassi build_toolchain, compile sandbox, ttsim executor, and run
# sandbox, against stand-ins: a copy of the repository whose toolchains/tt-metal.pin names a fake clang++-20 and
# whose toolchains/ttsim.pin carries the sha256 of fake library and descriptor files; a fake toolchains root
# (install record, CMake package list); a fake uv that runs the venv's python; and dry_standin.py as the compiler
# and the programs. The scratch root is a tmpfs mounted at /mnt/nvme10/joseph_ufl inside a user and mount namespace
# (with /mnt/c bound back), so run.sh's root checks pass unchanged. It tests the batch's logic, not tt-metal or ttsim.
set -euo pipefail
case_name="$1"
dry="$(realpath -m "$2")"
venv="$(realpath "$3")"
codec="$(realpath "$4")"
here="$(cd "$(dirname "$0")" && pwd)"
source_repo="$(cd "$here/../../.." && pwd)"
[ ! -e "$dry" ] || { echo "dry: $dry exists; give a new directory"; exit 2; }
mkdir -p "$dry/repo" "$dry/fakebin" "$dry/template/toolchains/tt-metal@5280a9cf" \
  "$dry/template/toolchains/ttsim@v1.3.4" "$dry/template/lassi-runs" "$dry/template/tmp" "$dry/out"

# The repository copy: what run.sh, tier_b.py, and lassi need, at the working tree's state.
(cd "$source_repo" && tar -cf - --exclude=__pycache__ lassi toolchains tools plans/spikes/p5-tier-b pyproject.toml \
  uv.lock) | tar -xf - -C "$dry/repo"
printf 'p53 dry-run stand-in, not libttsim\n' >"$dry/template/toolchains/ttsim@v1.3.4/libttsim_wh.so"
printf 'p53 dry-run stand-in, not a SoC descriptor\n' >"$dry/template/toolchains/ttsim@v1.3.4/soc_descriptor.yaml"
lib_sha="$(sha256sum "$dry/template/toolchains/ttsim@v1.3.4/libttsim_wh.so" | cut -d' ' -f1)"
soc_sha="$(sha256sum "$dry/template/toolchains/ttsim@v1.3.4/soc_descriptor.yaml" | cut -d' ' -f1)"
sed -i "s/^SHA256=[0-9a-f]*/SHA256=$lib_sha/; s/^SOC_DESCRIPTOR_SHA256=[0-9a-f]*/SOC_DESCRIPTOR_SHA256=$soc_sha/" \
  "$dry/repo/toolchains/ttsim.pin"
sed -i "s|^EXECUTABLE=.*|EXECUTABLE=$dry/fakebin/clang++-20|" "$dry/repo/toolchains/tt-metal.pin"
commit="$(sed -n 's/^COMMIT=//p' "$dry/repo/toolchains/tt-metal.pin")"
echo "tt-metal $commit https://github.com/tenstorrent/tt-metal.git" \
  >"$dry/template/toolchains/tt-metal@5280a9cf/lassi-install.txt"
grep -v '^#' "$dry/repo/toolchains/tt-metal-cpm-sources.txt" \
  >"$dry/template/toolchains/tt-metal@5280a9cf/lassi-cpm-sources.txt"

cat >"$dry/fakebin/clang++-20" <<EOF
#!/bin/sh
exec /usr/bin/python3 -B "$dry/repo/plans/spikes/p5-tier-b/dry_standin.py" clang "$dry/faults.json" "$codec" "\$@"
EOF
cat >"$dry/fakebin/uv" <<EOF
#!/bin/sh
# Fake uv for the dry run: drops run's options and runs the venv's python with the rest.
if [ "\$1" = --version ]; then echo "uv (p53 dry-run stand-in)"; exit 0; fi
[ "\$1" = run ] || { echo "fake uv: only run is supported" >&2; exit 2; }
shift
while [ \$# -gt 0 ]; do
  case "\$1" in
    --frozen | --offline | --no-project) shift ;;
    --with) shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
[ "\$1" = python ] || { echo "fake uv: expected python, got \$1" >&2; exit 2; }
shift
exec "$venv/bin/python" "\$@"
EOF
# util-linux 2.39 (Ubuntu 24.04, WSL) returns 16 from `mount --move` in a user namespace when it cannot write the
# userspace mount table, which the sandbox has made read-only by then; alpha01's 2.37 does not (P4.9 and P4.11
# measured the sandbox there). Inside the dry run's namespace mount(8) is wrapped to accept 16 alone.
cp /usr/bin/mount "$dry/fakebin/mount.real"
cat >"$dry/fakebin/mount.wrap" <<EOF
#!/bin/sh
"$dry/fakebin/mount.real" "\$@"
status=\$?
[ "\$status" -eq 16 ] && exit 0
exit "\$status"
EOF
chmod 755 "$dry/fakebin/clang++-20" "$dry/fakebin/uv" "$dry/fakebin/mount.wrap" "$dry/fakebin/mount.real"
case "$case_name" in
  main) echo '{"dev0:mt-polarquant": "ub", "asis:mt-residual": "build-error", "dev0:mt-residual": "build-error",
    "dev0:mt-rnorm": "hang", "dev0:mt-qjl-projection": "jit-error"}' >"$dry/faults.json" ;;
  stop | budget) echo '{"dev0:mt-rotation": "hang"}' >"$dry/faults.json" ;;
  order | stop-fetch | stop-batch | size) echo '{"dev0:baseline": "hang"}' >"$dry/faults.json" ;;
  *) echo '{}' >"$dry/faults.json" ;;
esac
(cd "$dry/repo" && git init -q && git add -A && git -c user.name=p53-dry -c user.email=p53-dry@invalid \
  commit -q -m "p53 dry-run copy")
if [ "$case_name" = dirty ]; then echo "# dry-run change" >>"$dry/repo/pyproject.toml"; fi
args=()
if [ "$case_name" = copy ] || [ "$case_name" = copy-fail ]; then
  # A changed clone: one file edited, one removed, one added, and a large file recorded by size only.
  git clone -q "$codec" "$dry/template/TurboQuant-Tenstorrent"
  echo "// p53 dry-run edit" >>"$dry/template/TurboQuant-Tenstorrent/host/turboquant_launcher.cpp"
  rm "$dry/template/TurboQuant-Tenstorrent/build.log"
  echo "extra" >"$dry/template/TurboQuant-Tenstorrent/notes.txt"
  head -c $((65 * 1024 * 1024)) /dev/zero >"$dry/template/TurboQuant-Tenstorrent/dump_big.bin"
  args=(--alpha01-copy /mnt/nvme10/joseph_ufl/TurboQuant-Tenstorrent)
fi

budget=""; scale=0.05
case "$case_name" in order) budget=440 ;; budget) budget=390 ;; esac
mnt="/tmp/p53-dry-$$"
mkdir -p "$mnt/c" "$mnt/runuser"
id="dry-$case_name-$(date +%H%M%S)"
set +e
unshare -rm --propagation private env DRY="$dry" MNT="$mnt" ID="$id" CASE="$case_name" BUDGET="$budget" \
  SCALE="$scale" ARGS="${args[*]:-}" PATH="$dry/fakebin:/usr/sbin:/usr/bin:/sbin:/bin" bash -c '
  set -e
  mount --rbind /mnt/c "$MNT/c"
  # A private /run, so mount(8) in the sandbox can write its userspace mount table (the host /run/mount is root
  # owned), with /run/user bound back for systemd-run --user.
  mount --rbind /run/user "$MNT/runuser"
  mount -t tmpfs p53-dry-run /run
  mkdir -p /run/user
  mount --rbind "$MNT/runuser" /run/user
  mount -t tmpfs p53-dry-mnt /mnt
  mkdir -p /mnt/c /mnt/nvme10/joseph_ufl
  mount --rbind "$MNT/c" /mnt/c
  mount -t tmpfs -o size=4g p53-dry-scratch /mnt/nvme10/joseph_ufl
  mount --bind "$DRY/fakebin/mount.wrap" /usr/bin/mount
  cp -a "$DRY/template/." /mnt/nvme10/joseph_ufl/
  export HOME=/mnt/nvme10/joseph_ufl LASSI_SCRATCH=/mnt/nvme10/joseph_ufl
  export LASSI_TOOLCHAINS=/mnt/nvme10/joseph_ufl/toolchains LASSI_RUNS_ROOT=/mnt/nvme10/joseph_ufl/lassi-runs
  export LASSI_RX_RUN_ID="$ID" TMPDIR=/mnt/nvme10/joseph_ufl/tmp P53_WALL_SCALE="$SCALE"
  if [ -n "$BUDGET" ]; then export P53_BUDGET_S="$BUDGET"; fi
  run=/mnt/nvme10/joseph_ufl/lassi-runs/p53-tier-b/$ID
  # when_exists FILE: wait up to 300 s for FILE; status 0 when it appeared.
  when_exists() { timeout 300 sh -c "until [ -e \"\$1\" ]; do sleep 0.05; done" sh "$1"; }
  case "$CASE" in
    copy-fail) : >/mnt/nvme10/joseph_ufl/TurboQuant-Tenstorrent/p53-unreadable
      mount --bind /etc/shadow /mnt/nvme10/joseph_ufl/TurboQuant-Tenstorrent/p53-unreadable ;;
    stop) (sleep 25; touch "$run/STOP") & ;;
    stop-fetch) (when_exists "$run/src/.git/HEAD" && touch "$run/STOP") & ;;
    stop-batch) (when_exists "$run/local/batch.stderr.txt" && touch "$run/STOP") & ;;
    size) (when_exists "$run/local/batch.stderr.txt" && head -c $((1100 * 1024 * 1024)) /dev/zero \
      >"$run/raw/p53-dry-fill.bin") & ;;
  esac
  set +e
  # ARGS is empty or "--alpha01-copy <path>" (no spaces), split on purpose.
  (cd "$DRY/repo" && bash plans/spikes/p5-tier-b/run.sh $ARGS) >"$DRY/out/stdout.txt" 2>"$DRY/out/stderr.txt"
  echo "run.sh exit status $?" >>"$DRY/out/stdout.txt"
  wait
  rm -f "$run/raw/p53-dry-fill.bin"
  if [ -d "$run" ]; then cp -a "$run" "$DRY/out/run"; fi
'
status=$?
set -e
rmdir "$mnt/c" "$mnt/runuser" "$mnt"
echo "dry: case $case_name, namespace status $status; run.sh stdout $(wc -l <"$dry/out/stdout.txt") lines"
tail -n 3 "$dry/out/stdout.txt"
