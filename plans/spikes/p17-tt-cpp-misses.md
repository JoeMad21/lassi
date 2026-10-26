# Spike: why tt -> cpp candidates that ran missed, and why others never built (task P17.27)

- Task: P17.27 (plans/p17-portable.md). Bible: Harness Contract (lassi_io.h), Toolchain Pins (gcc.pin); Agent Rule
  5 (tt-pairs-v0 is a development set, OQ-083 (b), so its outcomes may steer prompt work; the fix states facts about
  the harness and the toolchains, never an item's answer).
- Date: 2026-10-25, in the owner's window. The reads were approved in the working session (OQ-092's answer,
  recorded on branches p9-ee and p15-judges). Every command below read records under alpha01's runs root with
  `rx exec` (read only, from the scratch root /mnt/nvme10/joseph_ufl); nothing was built or run. rx ids use
  alpha01's clock (UTC-7): 18:22 there is 21:22 EDT.
- Records: the tt -> cpp trials under lassi-runs/runs/ of the RNGD sweeps of 2026-10-22 and 2026-10-23 (passes
  20261022-2100 to 20261023-2334; results/rngd-matrix-20261025/trials.tsv lists them: 55 tt -> cpp trials, 10 at
  alignment 0.0, 8 at 1.0, 37 with none); one sample each.

## Question

P17.27 was opened on a hypothesis: tt -> cpp candidates that ran clean and missed the tolerance truncate bf16
instead of rounding to nearest. Is that what the records show, and what do the build errors of the candidates that
did not build name?

## Commands and outputs

1. One alignment-0.0 trial read whole, rx 20261025-182206-exec-4381:

```
d=lassi-runs/runs/sweep-20261023-2018-qwen3-coder-30b--fz-maxabs/fz-maxabs/furiosa-ai--Qwen3-Coder-30B-A3B-Instruct-FP8/tt-pairs-v0/tt-cpp/eltwise_binary/run01; ls $d; grep -n -i -A3 "oracle\|alignment\|max_abs\|mismatch\|outcome" $d/trial.md | head -60
```

   Output (abridged to the two alignment tables): the baseline's `| c | 0.99999884901841 | 0.001953125 | 32768 |
   true | - |` and the candidate's `| c | - | - | - | false | shape mismatch: (65536,) in the reference, () in the
   candidate |`.

2. The candidate's write, rx 20261025-182218-exec-3ee9 (the same $d):

```
find $d/attempt00 -maxdepth 3 | head; for f in $(find $d/attempt00 -name "*.cpp" | head -2); do echo "== $f"; grep -n -B4 -A2 "lassi_io_write" $f; done
```

   Output: attempt00/build/eltwise_binary.cpp lines 77 to 80, `lassi_io_free(&a); lassi_io_free(&b);` and then
   `const int status = lassi_io_write(argv[3], "c", LASSI_IO_F32, a.rank, a.dims, c.data());`.
   assets/harness/c/lassi_io.h's lassi_io_free frees dims and storage and then runs `memset(array, 0, sizeof
   *array)`, so a.rank is 0 and a.dims NULL at that write, which lassi_io_write accepts as a scalar: a rank-0 array.

3. Every tt -> cpp trial at alignment 0.0, by the note of its last failing output row, rx 20261025-182244-exec-3dff:

```
for f in $(find lassi-runs/runs -path "*tt-cpp*" -name trial.md 2>/dev/null); do a=$(grep -m1 "^| Alignment |" $f | cut -d"|" -f3 | tr -d " "); [ "$a" = "0.0" ] || continue; it=$(echo $f | sed "s#.*/tt-cpp/##; s#/run01.*##"); n=$(grep -E "^\| [a-z_0-9]+ \| .* \| false \|" $f | tail -1 | sed "s/.*| false | //" | cut -c1-90); echo "$it | $n"; done | sort | uniq -c | sort -rn | head -20
```

```
      5 eltwise_binary | shape mismatch: (65536,) in the reference, () in the candidate |
      1 matmul_single_core | unmatched NaN at element 1013 in C order: 162.0 in the reference, nan in the candidate (15
      1 matmul_single_core | pcc 0.026649749066143806 is below the pcc threshold 0.9805358808476967 |
      1 eltwise_sfpu | unmatched NaN at element 0 in C order: 1.234375 in the reference, nan in the candidate (31
      1 eltwise_sfpu | max_abs 1.7109375 is above the max_abs threshold 0.015625 |
      1 eltwise_binary | dtype mismatch: f32 in the reference, bf16 in the candidate |
```

4. The order of the free and the write in the last attempt of each of the five rank-0 trials, rx
   20261025-184804-exec-1b43:

```
for t in $(grep -l "shape mismatch: (65536,) in the reference, () in the candidate" $(find lassi-runs/runs -path "*tt-cpp/eltwise_binary/*" -name trial.md) 2>/dev/null); do d=$(dirname $t); last=$(ls -d $d/attempt* | sort | tail -1); f=$(find $last -name "eltwise_binary.cpp" | head -1); fr=$(grep -n "lassi_io_free(&a)" $f | head -1 | cut -d: -f1); wr=$(grep -n "lassi_io_write" $f | head -1 | cut -d: -f1); echo "$(echo $d | sed "s#lassi-runs/runs/##; s#/fz.*##; s#/mmfz.*##") $(basename $last) free_a_line=$fr write_line=$wr"; done
```

```
sweep-20261022-2202-qwen3-coder-30b-maxabs/.../tt-cpp/eltwise_binary/run01 attempt00 free_a_line=33 write_line=74
sweep-20261023-2018-qwen3-coder-30b--fz-maxabs attempt00 free_a_line=35 write_line=80
sweep-20261022-2149-qwen3-coder-30b-maxabs/.../tt-cpp/eltwise_binary/run01 attempt00 free_a_line=35 write_line=79
sweep-20261023-2334-qwen3-coder-30b--fz-maxabs attempt02 free_a_line=39 write_line=87
sweep-20261023-2334-qwen3-coder-30b--mmfz-maxabs attempt15 free_a_line=53 write_line=96
```

   (The first and third paths are shortened here with "..."; the output printed them whole.)

5. The compile errors in the tt -> cpp attempts of three items with many correction-cap ends at S1 (trials.tsv:
   matmul_multi_core 9, eltwise_sfpu 7, matmul_single_core 6, eltwise_binary 6), from the first 60 compile.stderr
   files that `find` lists for each item (every attempt, whatever the trial's end), each error line's location
   cut and its digits masked as N, rx 20261025-182259-exec-8688:

```
for it in eltwise_sfpu matmul_single_core matmul_multi_core; do echo "== $it"; find lassi-runs/runs -path "*tt-cpp/$it/*" -name compile.stderr 2>/dev/null | head -60 | xargs grep -h "error:" 2>/dev/null | sed -E "s/^[^:]*:[0-9]+:[0-9]+: //; s/[0-9]+/N/g" | cut -c1-110 | sort | uniq -c | sort -rn | head -6; done
```

```
== eltwise_sfpu
     80 error: 'tt' is not a class, namespace, or enumeration
     30 error: base operand of '->' is not a pointer
     26 error: 'tt' has not been declared
     23 fatal error: tt-metalium/constants.hpp: No such file or directory
     18 error: 'program' was not declared in this scope
     12 error: request for member 'get' in 'device', which is of non-class type 'int'
== matmul_single_core
     38 fatal error: tt-metalium/bfloatN.hpp: No such file or directory
      2 fatal error: tt-metalium/distributed.hpp: No such file or directory
      1 fatal error: tt-metalium/constants.hpp: No such file or directory
== matmul_multi_core
     35 fatal error: tt-metalium/bfloatN.hpp: No such file or directory
     28 fatal error: api/dataflow/dataflow_api.h: No such file or directory
     25 fatal error: api/compute/tile_move_copy.h: No such file or directory
```

## Findings [MEASURED, alpha01's records, read 2026-10-25]

- No alignment-0.0 trial is shown missing by one bf16 unit, so the truncation hypothesis is not what these records
  show. Five of the ten wrote a rank-0 output where the reference writes (65536,), and in all five the last attempt
  calls lassi_io_free(&a) before its first lassi_io_write (command 4); the one read whole (commands 1 and 2) passes
  a.rank and a.dims to that write. One of the five is the last attempt after 15 corrections with the mismatch
  report. One wrote bf16 where the
  source writes f32; the other four are wrong values (NaN or far off).
- In the compile errors read for the three items' candidates (command 5), the commonest errors are missing
  tt-metalium and kernel API headers and undeclared tt:: names, which gcc-native's include path does not hold
  (lassi/toolchains/gcc.py runs g++ -O3 -fopenmp with no -I, -L, or -l).
- Both are facts about the harness and the toolchains that tt-host-v0 never states: generate.txt says what
  lassi_io_read and lassi_io_write return but not that lassi_io_free zeroes the array, and it says the cpp program
  is built by g++ 12 but not that no tt-metal header is on its path.

## Consequence

tt-host-v1 (task P17.27) is tt-host-v0 with two generic statements, in generate.txt, correct.txt, and
correct_mismatch.txt: the cpp program has no tt-metal header or library on its path, uses the C++ standard
library, OpenMP, and lassi_io.h, and does any bf16 conversion, tiling, and tile math itself; and lassi_io_free
zeroes the array, so each output is written before an input whose dims it uses is freed. tt-host-v0 is unchanged,
so earlier results stay comparable. Whether the statements help is measured only by a run that names tt-host-v1;
nothing here measures that.
