# Spike: what breaks in the C++ -> TT MLIR path (2026-10-26)

- Asked by the owner on 2026-10-26: "we need to identify the issues in MLIR lowering from C++ to TT." Covers the two
  halves of the path: the model writing ttkernel MLIR that tt-mlir lowers to TT-Metalium kernels (task P17.36), and
  Polygeist's cgeist lowering the C++ source to MLIR (task P5.5's tools, not yet in any prompt).
- Evidence: the records of the window run's ttkernel recipe, pulled read-only after the owner's yes in the working
  session (rx pull --path lassi-runs/runs/sweep-20261026-0556-gpt-oss-120b--ttkernel-maxabs; job rx
  20261026-055605-window-119f from clean commit 3d4e0e9; 3 trials, 27 attempts, gpt-oss-120b, ttsim v1.3.4); and three
  exploratory cgeist probes from dirty snapshots of 3d4e0e9 (rx 20261026-150631-suite-1d03, 20261026-150716-suite-f1c2,
  20261026-150758-suite-7eab; plans/runs/mlir/cgeist-probe.sh; outputs under lassi-runs/cgeist-probe/ on alpha01).
  Development results on tt-pairs-v0 (OQ-083 (b)); the probes are exploratory, never [MEASURED].

## Half 1: the model writes ttkernel MLIR [MEASURED, ttsim v1.3.4 on alpha01, rx 20261026-055605-window-119f]

Of 27 attempts (3 loopback trials, each a generation and 8 corrections), 23 ended at the verify step (ttmlir-opt's
parser and verifier), 4 built, and none passed.

| Cause | Attempts | Example diagnostic |
| --- | ---: | --- |
| A type the dialect does not have | 4 | unknown type `tensor_accessor_args` in dialect `ttkernel` |
| An op the dialect does not have | 2 | custom op 'ttkernel.InterleavedAddrGenFast' is unknown |
| The op's assembly format not followed | 12 | expected ')'; expected operation name in quotes; expected non-function type; expected SSA operand; expected attribute value; expected '(' |
| Other MLIR errors | 5 | undefined symbol alias id '0' (3); 'i32' vs 'index' on one value; arith.index_cast without 'to' |
| Built, then stopped on ttsim | 4 | UnimplementedFunctionality: coord_to_tile: coord (0,0) |

Findings:

1. The model does not know the ttkernel dialect's syntax at the pin. It borrows names from TT-Metalium's C++ API
   (TensorAccessorArgs, InterleavedAddrGenFast) that have no ttkernel op or type, and it misses ops' custom assembly
   formats. The same error often repeats over three corrections (run01 attempts 2 to 4; run02 attempts 3, 4, 6), so one
   parser message is not enough feedback to fix it.
2. Every built attempt addressed DRAM at the NOC coordinate (0, 0) with `ttkernel.get_noc_addr(%c0, %c0, %addr)`
   (run02 attempt 1, read), not through a DRAM bank as the passing smoke kernel did (get_noc_addr_from_bank_id). ttsim
   cannot map (0, 0) to a tile and reports UnimplementedFunctionality. The loop then labels the attempt a simulator
   gap, "not a model failure" (OQ-098), but the cause is the model's addressing, which would also be wrong on a card
   (inferred). The gap rule mislabels this case.
3. Only loopback can be written at all: a ttkernel kernel names a circular buffer only through an argument, and the
   other four items' host programs pass none (Level Rules; OQ-103).
4. The dialect pack is cut at 40,960 bytes, which leaves about 100 ops out (task P17.36's report); whether the ops the
   model needed were in it is not yet checked.

Fixes to weigh (none built):

- On a verify error, feed back the exact definition and assembly format of the op or type named, or of the nearest
  ttkernel names when it is unknown, as the C++ path's nearest-names lookup does.
- Make sure the pack keeps every data-movement op whole (get_arg_val, the bank-based get_noc_addr forms,
  noc_async_read and write, the barriers), and say in the prompt that DRAM is addressed by bank, never by a NOC
  coordinate.
- Read ttsim's coord_to_tile gap on a coordinate that is no tile as an addressing error fed back to the model, not
  as a gap (a change to OQ-098's rule, so an owner choice).
- OQ-103 (a), host programs that pass circular buffer indices, to reach the other four items.

## Half 2: cgeist lowers the C++ source to MLIR [exploratory, dirty snapshots of 3d4e0e9]

1. cgeist (Polygeist 77c04bb2, LLVM 18.0.0git) aborts on all five items' whole sources, with or without
   --raise-scf-to-affine, on the same assertion: `ValueCategory::store ... Assertion 'toStore.isReference' failed`
   (tools/cgeist/Lib/ValueCategory.cc:154), at a store into a `memref<256xi8>` made by polygeist.pointer2memref. The
   same 256-byte value in all five points at code they share, most likely the I/O harness lassi_io.h (inferred).
2. Three items keep their computation inside main beside the I/O (eltwise_binary, eltwise_sfpu, matmul_single_core);
   loopback (copy_tiles) and matmul_multi_core (multiply) have it in a function of its own. Lowering one function
   needs cgeist's --function with the C++ mangled name; the probe's lookup of those names found none (its compile step
   likely lacked a clang++ driver in the Polygeist build; not checked), so no function was lowered alone.
3. The pinned polygeist-opt has no --raise-affine-to-linalg pass ("Unknown command line argument"), so this Polygeist
   gives scf or affine at best, never linalg. Linalg needs another raiser or another pin (OQ-065).

Fixes to weigh (none built): a compute-only C translation unit per item, with the computation in an extern "C"
function and no I/O, which cgeist can lower by name; then decide the target level the prompt shows (affine is
available now; linalg needs OQ-065).

## Built after this spike (task P17.41, 2026-10-26)

Each fix above is built as an opt-in or a tracked file, with tests on fakes only (plans/p17-portable.md, P17.41):
the definitions of a verify error's names (dialect_lookup), the pack keeping every data-movement op whole, the bank
rule and the mistakes above in tt-mlir-v0's prompts, loop.coord_gap_as_error for the coord_to_tile gap, the four
ttkernel host programs of OQ-103 (a), and the compute-only C units with the source_mlir step on cgeist-affine at the
affine level. Nothing above has run since on alpha01: plans/runs/mlir/lowering-probe.sh and ttkernel-hosts.sh are
the window's checks.

## Addendum 2026-10-29: the reference card behind two of the errors (task P17.44)

From the records of rx 20261026-055605-window-119f (trial loopback/run02, its generation prompt, read from the
pulled run tree on the workstation): the prompt opens with the card the model wrote in summarize_context ("Earlier you
wrote a reference card of the ttkernel MLIR dialect of tt-mlir 0.9.0.dev20260221. Your card follows"). Two of its rows,
verbatim (the second cut after its result column):

```
| **ttkernel.get_compile_time_arg_val** | `I32Attr` | `TTKernel_ArgResultType` | `ttkernel.get_compile_time_arg_val(#0) : (I32Attr) -> (TTKernel_ArgResultType)` |
| **ttkernel.TensorAccessorArgs** | `Optional<I32>`, `Optional<I32>`, `Optional<TTKernel_TensorAccessorArgs>`, `OptionalAttr<StrAttr>`, `OptionalAttr<StrAttr>` | `TTKernel_TensorAccessorArgs` | ... |
```

The trial's third attempt wrote `ttkernel.get_compile_time_arg_val(#0) : (i32) -> i32`, which ttmlir-opt refused
with "undefined symbol alias id '0'", and its ninth was refused with "unknown type `tensor_accessor_args` in dialect
`ttkernel`". The card spells a TableGen
constraint (`TTKernel_TensorAccessorArgs`) where the dialect's textual type is `!ttkernel.TensorAccessorArgs`, and an
attribute placeholder (`#0`) where the form takes an integer literal.
