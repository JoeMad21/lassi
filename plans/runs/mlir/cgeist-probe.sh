#!/usr/bin/env bash
# Exploratory read for the owner (2026-10-26): which tt-pairs-v0 C++ sources Polygeist's cgeist lowers to MLIR, and
# to which level (scf, affine, linalg). Run from the repository root of an rx run slot on alpha01:
#   bash plans/runs/mlir/cgeist-probe.sh
# It reads the five items' C++ sources and lassi_io.h, runs the pinned cgeist and polygeist-opt, writes only under
# $LASSI_RUNS_ROOT/cgeist-probe/<stamp>/, and prints each step's return code and the dialect op counts. CPU only.
set -u
prlimit --pid $$ --core=1:1 || exit 2
out="${LASSI_RUNS_ROOT:?}/cgeist-probe/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out"
pg="${LASSI_TOOLCHAINS:?}/polygeist@77c04bb2/build/bin"
inc=assets/harness/c
echo "sources at $(git log --oneline -1); cgeist $("$pg/cgeist" --version 2>&1 | grep -m1 version)"
for f in assets/bench/tt-pairs-v0/*/cpp/*.cpp; do
  n=$(basename "$f" .cpp)
  timeout 300 "$pg/cgeist" "$f" -S --function='*' -I "$inc" -o "$out/$n.scf.mlir" > "$out/$n.scf.err" 2>&1; a=$?
  timeout 300 "$pg/cgeist" "$f" -S --function='*' --raise-scf-to-affine -I "$inc" -o "$out/$n.affine.mlir" \
    > "$out/$n.affine.err" 2>&1; b=$?
  c=skip
  if [ -s "$out/$n.affine.mlir" ]; then
    timeout 300 "$pg/polygeist-opt" "$out/$n.affine.mlir" --raise-affine-to-linalg -o "$out/$n.linalg.mlir" \
      > "$out/$n.linalg.err" 2>&1; c=$?
  fi
  echo "== $n scf rc=$a affine rc=$b linalg rc=$c"
  for k in scf affine linalg; do
    if [ -s "$out/$n.$k.mlir" ]; then
      echo "   $k ops: $(grep -oE '(affine|scf|memref|arith|func|llvm|linalg|math|cf|polygeist)\.[a-z_]+' \
        "$out/$n.$k.mlir" | cut -d. -f1 | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
    fi
    if [ -s "$out/$n.$k.err" ]; then
      echo "   $k err: $(head -c 400 "$out/$n.$k.err" | tr '\n' ' ')"
    fi
  done
done
# The compute functions alone, where an item has one apart from main.
for pair in loopback:copy_tiles matmul_multi_core:multiply; do
  item=${pair%%:*}; fn=${pair##*:}
  f="assets/bench/tt-pairs-v0/$item/cpp/$item.cpp"
  "$pg/clang++" -c "$f" -I "$inc" -o "$out/$item.o" > "$out/$item.o.err" 2>&1
  fn=$(nm "$out/$item.o" | awk -v f="$fn" '$2 == "T" && index($3, f) {print $3; exit}')
  echo "== $item mangled name: ${fn:-none}"
  [ -n "$fn" ] || continue
  for mode in scf affine; do
    flag=""; [ "$mode" = affine ] && flag="--raise-scf-to-affine"
    timeout 300 "$pg/cgeist" "$f" -S --function="$fn" $flag -I "$inc" -o "$out/$item.$fn.$mode.mlir" \
      > "$out/$item.$fn.$mode.err" 2>&1; rc=$?
    echo "== $item $fn $mode rc=$rc"
    if [ -s "$out/$item.$fn.$mode.mlir" ]; then
      echo "   ops: $(grep -oE '(affine|scf|memref|arith|func|llvm|linalg|math|cf|polygeist)\.[a-z_]+' \
        "$out/$item.$fn.$mode.mlir" | cut -d. -f1 | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
    fi
    [ -s "$out/$item.$fn.$mode.err" ] && echo "   err: $(head -c 300 "$out/$item.$fn.$mode.err" | tr '\n' ' ')"
  done
  if [ -s "$out/$item.$fn.affine.mlir" ]; then
    timeout 300 "$pg/polygeist-opt" "$out/$item.$fn.affine.mlir" --raise-affine-to-linalg \
      -o "$out/$item.$fn.linalg.mlir" > "$out/$item.$fn.linalg.err" 2>&1
    echo "== $item $fn linalg rc=$?"
    [ -s "$out/$item.$fn.linalg.mlir" ] && echo "   ops: $(grep -oE '(affine|scf|linalg|memref|arith)\.[a-z_]+' \
      "$out/$item.$fn.linalg.mlir" | cut -d. -f1 | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
    [ -s "$out/$item.$fn.linalg.err" ] && echo "   err: $(head -c 300 "$out/$item.$fn.linalg.err" | tr '\n' ' ')"
  fi
done
echo "out=$out"
