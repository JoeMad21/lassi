# Spike: lowering TTIR and D2M to TTKernel and TT-Metal at the pin (task P17.46, 2026-10-29)

## Question

Can the model write tensor-level TTIR, or D2M, and let the pinned tt-mlir (0.9.0.dev20260221, commit
5f396fd6ef85780bf1b6b85bef9c7d2f84a85f3c) lower it to TTKernel and TT-Metal, write a flatbuffer, and run it on ttsim?
Today the model writes ttkernel MLIR or TT-Metal C++ kernels (plans/spikes/mlir-issues.md). P17.47 (references and
retrieval for TTIR and D2M), P17.48 (the ttir and d2m languages, their toolchains, and a runtime executor), and P17.49
(prompt sets and recipes) depend on the answer.

Classification: factual for the pipelines, the lowering of each item, the ops, and what the install holds (answered
below by reading the pinned sources and by running the pinned tools); a choice for building a runtime, which changes
toolchains/tt-mlir.pin (a pin change: the owner's approval and a Decision Log entry; owner-queue draft at the end).

## Evidence and provenance

- Sources: tt-mlir at 5f396fd6 read through `gh api` (contents and the git tree; a tarball of the commit, sha256
  67afd5515290f7c98dc44170bba1a12d489a98d14044c5b7af2b3843850f03dc, unpacked in a scratch directory outside the
  repository); tt-metal at 5280a9cfb00998fd49667a29523d03aee905c129 through `gh api`; the Tenstorrent package index
  (https://pypi.eng.aws.tenstorrent.com/, read 2026-10-29). Every file:line below is at those commits.
- Runs: seven `uv run tools/rx.py exec` calls on alpha01 under the owner's standing permission of 2026-10-26 (short
  CPU-only runs): CPU only, each under one minute, no device opened, no model, no `--big`, no slot (exec has none),
  outputs only under `$LASSI_RUNS_ROOT/ttir-d2m-spike/` (about 2.5 MB in three directories; nothing deleted). The
  hand-written modules and the probe scripts were sent inside the exec command, not from a commit, so every lowering
  result here is exploratory, never [MEASURED]; each module's sha256 prefix is listed so a rerun from a clean commit
  can be matched. rx ids use alpha01's clock (UTC-7); the probes print UTC.

| rx id | What |
| --- | --- |
| 20261029-172018-exec-ff8a | Read-only inventory of the tt-mlir install: install record, bin, lib, CMakeCache options, du, `--help` lists |
| 20261029-172249-exec-e512 | probe1: nine TTIR modules through the frontend, to TTKernel, the full pipeline, and the flatbuffer |
| 20261029-172321-exec-a6f3 | Printed three D2M outputs of probe1 (the frontend's own D2M form) and the flatbuffer's header |
| 20261029-172648-exec-0ae2 | probe2: hand-written D2M modules, single-core matmul options, gather, the D2M re-entry, the pinned explicit test |
| 20261029-173004-exec-4263 | probe3: loopback in D2M on one core and DRAM to DRAM, 160x320 in TTIR, matmul without multicast |
| 20261029-173021-exec-073f | Read two lines of probe outputs (the explicit-form translate error; loopback's program ops) |
| 20261029-173239-exec-34f2 | Read-only: runtime products in the tt-mlir build, the pinned tt-metal tree's libraries and CMakeCache, MPI, df |

Disk on the shared mount read 75G free (98% used) before the first run and after the last (df, rx ff8a and 34f2).

### The commands each case ran

From the probes' run.sh (sha256 prefixes: probe1 4f694ce370a32a21, probe2 6b26029dabcfcec7, probe3 a463d71162d81a08),
with `M=$LASSI_TOOLCHAINS/tt-mlir@0.9.0.dev20260221`, `OPT=$M/build/tt-mlir/bin/ttmlir-opt`,
`TR=$M/build/tt-mlir/bin/ttmlir-translate` (ttmlir-opt sha256 58c8997c56d7538a..., ttmlir-translate c038995900f7f36b...),
each under `timeout 240` and `prlimit --pid $$ --core=1:1` (read back as 1 1):

```
$OPT --ttir-to-ttmetal-fe-pipeline="<opts>" m/<module>.mlir -o out/<case>.d2m.mlir                      # D2M level
$OPT --ttir-to-ttmetal-fe-pipeline="<opts>" --ttir-to-ttmetal-me-pipeline="<opts>" --convert-d2m-to-ttkernel \
     m/<module>.mlir -o out/<case>.ttkernel.mlir                                                         # TTKernel level
$OPT --ttir-to-ttmetal-pipeline="<opts>" m/<module>.mlir -o out/<case>.ttmetal.mlir                     # TTMetal + EmitC
$TR --ttmetal-to-flatbuffer out/<case>.ttmetal.mlir -o out/<case>.ttm                                   # flatbuffer
```

The TTKernel command is the form of test/ttmlir/Conversion/D2MToTTKernel/dataformat_template_args.mlir:1; the full
pipeline and translate are test/ttmlir/Silicon/TTMetal/n150/simple_add.mlir:1-3 without its system-desc-path.

## 1. Pipelines and options at the pin

Registered in lib/Dialect/TTMetal/Pipelines/TTMetalPipelines.cpp:286-303, all on one options struct
(include/ttmlir/Dialect/TTMetal/Pipelines/TTMetalPipelines.h:15-189); `ttmlir-opt --help` on alpha01 lists them (rx ff8a):

| Pipeline | What it runs |
| --- | --- |
| `--ttir-to-ttmetal-pipeline` | Mark forward functions, wrap a device module, hoist tagged ops to a CPU module, then fe, me, be on the device module, then the TTIR-to-LLVM CPU pipeline (:256-280) |
| `--ttir-to-ttmetal-fe-pipeline` | Register device (system desc), TTIR-to-TTIR decomposition, rank normalization, `ttir-to-d2m`, grid selection, `d2m-lower-to-layout` (:74-116) |
| `--ttir-to-ttmetal-me-pipeline` | Elementwise fusion, bufferization, `d2m-allocate`, interchange, tile loops, linalg-to-affine, DST access, DMA lowering, regions to functions (:118-226) |
| `--ttir-to-ttmetal-be-pipeline` | `convert-d2m-to-ttkernel`, DST sections, `convert-d2m-to-ttmetal`, hoist inits, `convert-ttkernel-to-emitc`, form expressions (:228-254) |
| `--ttir-bufferization-pipeline` | One-shot bufferization (:42-63) |

Options (TTMetalPipelines.h, line of each): mesh-shape (:17), override-device-shape (:20), system-desc-path (:27),
mock-system-desc-arch wormhole_b0 or blackhole, default wormhole_b0 (:36-45), max-dst-physical-size-tiles (:47),
matmul-interchange (:52), use-tile-matmul, default false (:61), collapse-tensors-2d, default true (:68),
default-input-memspace and default-output-memspace l1 or dram, default l1 (:74-87), disable-tolayout-folding (:91),
insert-profiler-traces (:98), set-math-fidelity LoFi to HiFi4, default HiFi4 (:104-111), num-stream-buffers, default 2
(:114), allow-l1-output-spilling (:122), stream-insert-policy always or infer, default infer (:131), available-l1-addr-range
(:138), test-assume-l1-capacity (:143), test-buffer-size-policy min or max (:147), ttnn-mode (:154),
global-data-format-target (:160), enable-op-scheduler (:167, hard-enabled anyway at TTMetalPipelines.cpp:173),
enable-multicast-inference, default true (:174), debug-d2m-coalescing-inference (:183). Translation:
`ttmlir-translate --ttmetal-to-flatbuffer` (lib/Target/TTMetal/TTMetalToFlatbufferRegistration.cpp:22-35).

The pinned lit tests use these RUN lines (each file's first lines):

- Full pipeline and flatbuffer: Silicon/TTMetal/n150/simple_add.mlir:1-3 (`system-desc-path=%system_desc_path%
  override-device-shape=1,1`), simple_eltwise.mlir:1-3, simple_3d.mlir:1 (`collapse-tensors-2d=false`),
  n300/simple_add.mlir:1 (`mesh-shape=1,2`); Conversion/D2MToTTKernel/use_tile_matmul_{true,false}.mlir:1.
- Stages: Conversion/TTIRToD2M/*.mlir:1 (`--ttcore-register-device --ttir-to-d2m [--d2m-grid-selection]
  [--d2m-materialize-view-returns]`, some with `ttnn-mode=true`); five of the nine Conversion/D2MToTTKernel tests
  (`--ttcore-register-device --convert-d2m-to-ttkernel`); Conversion/D2MToTTMetal/generic_lowering.mlir:1; Dialect/D2M/generic/
  explicit_datamovement_form.mlir:2 (`--ttcore-register-device --ttir-to-ttmetal-me-pipeline --ttir-to-ttmetal-be-pipeline`).
- Marked `// UNSUPPORTED: true`: Silicon/TTMetal/n150/perf/*, simple_reduce*.mlir, to_layout.mlir, tiled_reblock.mlir,
  typecast.mlir, and Conversion/D2MToTTMetal/decomposable_lowering_{gather,reduce_or}.mlir. Two of them hold forms the
  pinned ops refuse: decomposable_lowering_gather.mlir passes ttir.gather three operands, and probe1 got
  `'ttir.gather' op expected 2 operands, but found 3`; typecast.mlir passes ttir.typecast two operands where
  TTIROps.td:906 (TTIR_ElementwiseUnaryOp, :279-300) takes one (read, not run).

Facts from the runs that the plan needs:

- The entry for both languages is the full pipeline. The fe output fed back through `--ttir-to-ttmetal-me-pipeline
  --ttir-to-ttmetal-be-pipeline` lowers to TTMetal but the flatbuffer step refuses it: `error: function is missing
  required 'tt.function_type' attribute during flatbuffer translation` (probe2, reentry_*), the attribute
  (include/ttmlir/FunctionTypes.h:65) that TTCoreMarkFunctionsAsForwardPass sets
  (lib/Dialect/TTCore/Transforms/MarkFunctionsAsForward.cpp:17, :43) at the head of the full pipeline
  (TTMetalPipelines.cpp:261).
- The pinned explicit-datamovement D2M test (Dialect/D2M/generic/explicit_datamovement_form.mlir) lowers by its RUN line
  and by the full pipeline, but neither output translates: `operation being parsed with an unregistered dialect`
  at a `d2m.stream_layout` left in the host function (probe2 and probe3; rx 073f line 11).
- The flatbuffer's file identifier is `TTM0` (od of eltwise_binary.ttm, rx a6f3).
- With no system-desc-path the device is a mock Wormhole: grid 8x8, l1_size 1499136, l1_unreserved_base 1024,
  dram_unreserved_base 1024, supported data types f32, f16, bf16, bfp_*, u32, u16, u8, si32 (the D2M output's
  #system_desc, rx a6f3; createDefaultWormholeSystemDesc, lib/Dialect/TTCore/IR/TTCoreOpsTypes.cpp:152-163;
  TTCoreRegisterDevice.cpp:46-53). The compiled programs place L1 buffers from address 1024 (probe3,
  d2m_loopback_g1: `ttmetal.create_buffer ... address = 1024`). tt-metal at its pin reserves the low 32 KiB of L1 and
  allocates from above MEM_MAP_END plus the kernel config area (tt_metal/hw/inc/internal/tt-1xx/wormhole/dev_mem_map.h:33,
  :71, :162; tt_metal/llrt/hal/tt-1xx/wormhole/wh_hal_tensix.cpp:47, :62-63), the runtime writes buffers at the
  compiler's addresses (runtime/lib/ttmetal/executor.cpp:426-431), and its address check is compiled out unless
  TT_RUNTIME_DEBUG (runtime/include/tt/runtime/debug.h:24-39; executor_utils.h:44-47). So a flatbuffer built against
  the mock descriptor must never run on ttsim or a card; one built for running needs `system-desc-path` of a
  descriptor the runtime wrote from the device (runtime/lib/common/system_desc.cpp:139-163, :265-285;
  SystemDesc derives from Flatbuffer, whose `store(path)` writes it, runtime/include/tt/runtime/types.h:273-279, :292).

## 2. The tt-pairs-v0 items, by hand, in TTIR and D2M

Shapes from assets/bench/tt-pairs-v0.yaml. A rank-1 vector of 65536 or 51200 elements is passed as a 2-D tensor with
the same row-major bytes (256x256, 160x320); the rank-1 form also lowers (below). "Flatbuffer" means lowered and
translated; nothing ran.

| Item | TTIR (module, options) | D2M (hand-written) |
| --- | --- | --- |
| eltwise_binary | Flatbuffer. `ttir.add` 256x256 bf16, defaults: grid 8x8, 4 programs (tilize a, tilize b, add, untilize), 8 kernels, add_tiles; 52,288 bytes. With the manifest's f32 output (`ttir.typecast` after the add): 5 programs, typecast_tile; 65,144 bytes. Rank 1 (65536): grid 1x8, 7 programs, 117,616 bytes | Flatbuffer. to_layout in, one generic of d2m.tile_add on grid 8x8, to_layout out: 4 programs, 8 kernels, the same kernel calls as TTIR; 52,944 bytes |
| eltwise_sfpu | Flatbuffer. `ttir.exp` 256x256 bf16: 3 programs, 6 kernels, exp_tile_init and exp_tile; 37,928 bytes | Flatbuffer. One generic of d2m.tile_exp: 3 programs; 38,456 bytes |
| loopback | No TTIR form: TTIR has no copy op, and `return %arg0` lowers to 0 programs (a 3,016-byte flatbuffer with no device work), in L1 or DRAM | 5x5 grid as written: abort, `calculateReblockMapForGrid ... Assertion ... "New grid shape must evenly divide tensor shape" failed` (include/ttmlir/AffineMapUtils.h:268). The same with `override-device-shape=5,5`: flatbuffer, 3 programs, 35,784 bytes. Written on a 1x1 grid: flatbuffer, 37,232 bytes, but grid selection re-gridded it to 5x8 and padded 320 columns to 512 (host_strides 512x1, rx 073f). DRAM interleaved, DRAM to DRAM: abort, `DeviceAttr::getMemoryMap ... "All dims in grid shape for DRAM interleaved memref must be 1"` (lib/Dialect/TTCore/IR/TTCoreOpsTypes.cpp:1671) |
| matmul_multi_core | Flatbuffer. `ttir.matmul` 640x640x640 bf16, defaults: 640 aligned to 768 (dim_alignments 256x256), grid 8x8, 3x3 tiles per core, block factors [1, 1, 8]; 4 programs, 9 kernels, NOC multicast (8 lines) and semaphores (25); 108,056 bytes. DRAM memspaces: 137,008 bytes. `enable-multicast-inference=false`: no multicast or semaphore, 76,856 bytes | Flatbuffer. The fe output's generic without its mcast loads: 4 programs, no multicast; 75,176 bytes (78,112 with multicast inference off) |
| matmul_single_core | First error at `override-device-shape=1,1`: `'func.func' op required L1 memory usage 5734400 exceeds memory capacity 1498112 (usable space is [1024, 1499136))`. DRAM memspaces: 3276800. Adding stream-insert-policy=always, allow-l1-output-spilling=true, num-stream-buffers=1, test-buffer-size-policy=min: 3276800 to 4915200; none fits. Controls: 256x256x256 on 1x1 is a flatbuffer (66,608 bytes); 640^3 on 2x2 is a flatbuffer (126,872 bytes) | Not written: the same allocator places the operands, so a 640^3 single-core D2M module meets the same limit (inferred) |

Every lowering above took 0.1 s or less of wall time per command on alpha01 (exploratory).

The hand-written modules (sha256 prefix; probe directory):

- probe1, TTIR: eltwise_binary 57a8ee3569367377, eltwise_binary_f32out 567e6889cd5e6099, eltwise_binary_1d
  36d92c63969afde6, eltwise_sfpu a8e6e256524862e9, loopback 776bd5e2b7ef5b0d, matmul a138d01aa35aaa77, gather
  62c2c043d758e95c, scatter 0f7293dc822a75a3, i32_add 9702bf8a5d6833c9.
- probe2: d2m_eltwise_binary 0d7f361730f0f311, d2m_eltwise_sfpu 2f02fa3b051f9acd, d2m_loopback 1335ea7631a03629,
  d2m_matmul 46a248b8a24d8db0, gather2 223a1693c455c60f, mm256 8a777ef55537ceec, pinned_explicit_abs 15c86ba85d0fc5d2
  (the pinned test, unchanged).
- probe3: d2m_loopback_g1 e47e540779dc5360, d2m_loopback_dram c58cf3dad91db508, ttir_exp_160x320 e4c3d5c60f9dd03a.

TTIR, eltwise_binary (form of Silicon/TTMetal/n150/simple_add.mlir:5-12; eltwise_sfpu swaps in
`"ttir.exp"(%arg0) : (tensor<256x256xbf16>) -> tensor<256x256xbf16>` as simple_eltwise.mlir:96; matmul is
`"ttir.matmul"(%lhs, %rhs) : (tensor<640x640xbf16>, tensor<640x640xbf16>) -> tensor<640x640xbf16>` as
use_tile_matmul_true.mlir:18):

```mlir
func.func @eltwise_binary(%arg0: tensor<256x256xbf16>, %arg1: tensor<256x256xbf16>) -> tensor<256x256xbf16> {
  %1 = "ttir.add"(%arg0, %arg1) : (tensor<256x256xbf16>, tensor<256x256xbf16>) -> tensor<256x256xbf16>
  return %1 : tensor<256x256xbf16>
}
```

D2M, eltwise_binary (d2m.to_layout as test/ttmlir/Dialect/D2M/lower_to_layout.mlir:25-26; the generic, remote_load,
remote_store, and linalg.generic as the pinned frontend prints them, rx a6f3). eltwise_sfpu has one input and
`"d2m.tile_exp"(%x)`; loopback has no linalg.generic and stores the loaded shard; matmul uses the maps
(d0, d2), (d2, d1), (d0, d1), iterator types parallel, parallel, reduction, block_factors [1, 1, 8], loads
`%a1[%i, %k]` and `%b1[%k, %j]`, and `"d2m.tile_matmul"(%x, %y, %o)`:

```mlir
#layout = #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded, index_map = map(0)>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
!tile = !ttcore.tile<32x32, bf16>
!tiles = tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #layout>
!shard = tensor<1x1x!ttcore.tile<32x32, bf16>>
!cb = !d2m.cb<tensor<1x1x!ttcore.tile<32x32, bf16>>>
func.func @eltwise_binary(%a: tensor<256x256xbf16>, %b: tensor<256x256xbf16>) -> tensor<256x256xbf16> {
  %a0 = d2m.empty() : !tiles
  %a1 = d2m.to_layout %a, %a0 : tensor<256x256xbf16> into !tiles -> !tiles
  %b0 = d2m.empty() : !tiles
  %b1 = d2m.to_layout %b, %b0 : tensor<256x256xbf16> into !tiles -> !tiles
  %c0 = d2m.empty() : !tiles
  %c1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%a1, %b1 : !tiles, !tiles)
      outs(%c0 : !tiles) {
  ^unified0(%cb0: !cb, %cb1: !cb, %cb2: !cb):
    %i = d2m.block_index(0) : index
    %j = d2m.block_index(1) : index
    %ea = tensor.empty() : !shard
    %la = d2m.remote_load %ea %a1[%i, %j] : !shard, !tiles -> !shard
    %eb = tensor.empty() : !shard
    %lb = d2m.remote_load %eb %b1[%i, %j] : !shard, !tiles -> !shard
    %ec = tensor.empty() : !shard
    %s = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%la, %lb : !shard, !shard) outs(%ec : !shard) {
    ^bb0(%x: !tile, %y: !tile, %o: !tile):
      %t = "d2m.tile_add"(%x, %y) : (!tile, !tile) -> !tile
      linalg.yield %t : !tile
    } -> !shard
    %st = d2m.remote_store %c0[%i, %j] %s : !tiles, !shard -> !tiles
    d2m.yield %st : (!tiles)
  } : !tiles
  %h0 = d2m.empty() : tensor<256x256xbf16>
  %h1 = d2m.to_layout %c1, %h0 : !tiles into tensor<256x256xbf16> -> tensor<256x256xbf16>
  return %h1 : tensor<256x256xbf16>
}
```

Observations for the model's task at each level:

- TTIR is one op per item here (plus a typecast for eltwise_binary's f32 output): the compiler chooses the grid, the
  layouts, the tilize and untilize programs, the circular buffers, and the kernels. Loopback is not expressible.
- D2M is about 30 lines per item and the first level that expresses data movement (remote_load, remote_store), so
  loopback exists there. The grid the module writes is not binding: d2m-grid-selection re-grids it (1x1 became 5x8),
  and a written grid that does not divide the tensor aborts the compiler unless override-device-shape matches it.
- The compiler aborts on two hand-written inputs (assertions, exit status 134, this build has assertions on:
  `--version` prints "Optimized build with assertions", rx ff8a). A toolchain must read an abort as a compile failure
  with the assertion text as the diagnostic.

## 3. Which ops the D2M path lowers

- TTIR to D2M (populateTTIRToD2MPatterns, lib/Conversion/TTIRToD2M/TTIRToD2M.cpp:2232-2319): elementwise abs, add,
  bitwise and/not/or/xor, ceil, clamp (scalar and tensor), cos, div, erf, erfc, exp, floor, gelu, hardsigmoid, log,
  logical and/not/or/xor, multiply, maximum, minimum, neg, pow, reciprocal, relu, rsqrt, sigmoid, sign, silu, sin,
  sqrt, subtract, tan, tanh, where, equal, not_equal, greater_than, greater_equal, less_than, less_equal, typecast;
  reductions max and sum; rearrange, reshape, slice_static, concatenate_heads, permute (transpose canonicalizes to
  permute); to_layout, empty, full, mesh_shard, matmul, arange. The pass marks the whole TTIR dialect illegal (:2366)
  and applies a partial conversion (:2381), so any other TTIR op left after decomposition is an error.
- D2M to TTKernel (populateD2MToTTKernelPatterns, lib/Conversion/D2MToTTKernel/D2MToTTKernel.cpp:1974-2073): FPU
  tile_bcast, tile_matmul, tile_matmul_block, tile_reduce_sum, tile_reduce_max, tile_add/sub/mul; SFPU unary ops as
  above, binary add, sub, mul, bitwise and/or/xor, div, maximum, minimum, pow, ternary where; tilize and untilize;
  fill, masks, arange, transpose; DMA read, write, wait; circular buffer wait, reserve, push, pop; semaphores; memref
  load and store; core index. The D2M dialect's tile and region ops are in include/ttmlir/Dialect/D2M/IR/
  D2MGenericRegionOps.td (79 definitions: tile_* compute ops, dma_read/dma_write/dma_wait, remote_load/remote_store,
  semaphore_*, wait/reserve/push/pop, block_index, core_index, and others).
- Exercised by the probes: add, exp, matmul, typecast (bf16 to f32), and add on si32 tiles (lowered to add_tiles;
  whether that is exact for integers was not checked). Copy has no op; in D2M it is a load and a store.
- Gather and scatter: TTIR defines gather (TTIROps.td:3629), scatter (:4789), and embedding (:2666). The TTIR-to-TTIR
  decomposition, whose default config is ttnn (include/ttmlir/Conversion/Passes.td:45-57), marks gather illegal in
  the ttnn and ttmetal configs (lib/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecompositionPass.cpp:67-72) and
  rewrites it to embedding, which TTIRToD2M has no pattern for: probe2's gather2 stopped at `failed to legalize
  operation 'ttir.embedding' that was explicitly marked illegal`, as the pinned UNSUPPORTED test says ("Full lowering
  requires: Embedding: issue 3024", decomposable_lowering_gather.mlir:3-5). Scatter (the form of
  Dialect/TTNN/simple_scatter.mlir:34) stopped at `failed to legalize operation 'ttir.scatter' that was explicitly
  marked illegal`. D2M has no gather or scatter op, and remote_load and remote_store index shards of a grid, not
  elements (D2MGenericRegionOps.td; the fe outputs, rx a6f3). The supported data types stop at si32 and u32 (the
  system descriptor above), so bale-v0-tt's i64 data has no device type either.

## 4. A runtime in the pinned install

No. On alpha01 the install's CMakeCache reads TTMLIR_ENABLE_RUNTIME OFF, TTMLIR_ENABLE_TTRT OFF,
TTMLIR_ENABLE_BINDINGS_PYTHON OFF, TT_RUNTIME_ENABLE_TTMETAL OFF, TT_RUNTIME_ENABLE_TTNN OFF (rx ff8a), as
toolchains/tt-mlir.pin's CMAKE_FLAGS set them (tt-mlir.pin:65-75); the build tree's bin holds only ttmlir-opt and
ttmlir-translate, and no libTTMLIRRuntime, libTTRuntime*, or libTTBinary exists under build/tt-mlir/runtime or lib
(rx 34f2); src/tt-mlir/third_party holds only CMakeLists.txt, so no tt-metal was fetched. No pinned prebuilt runtime
exists either: the release wheel ttmlir-0.9.0.dev20260221-cp311 (sha256 37b8fd1e...b962e, from the Tenstorrent index)
holds the compiler's Python modules and no runtime or ttrt member (its .so members: _mlir, _mlirDialectsLinalg,
_mlirDialectsQuant, _mlirLinalgPasses, _ttmlir, libTTMLIRPythonCAPI), and the index has no ttrt package (404,
2026-10-29). tt-xla's wheel (toolchains/tt-xla.pin) carries a runtime built at tt-mlir c7b8bd77, another commit, and
is not installed on alpha01 (rx ff8a's toolchains list).

What enabling it takes, read at the pin:

- CMake: TTMLIR_ENABLE_RUNTIME=ON (CMakeLists.txt:33), which turns on TT_RUNTIME_ENABLE_TTNN, _TTMETAL, and
  _DISTRIBUTED by default (:50-58); TTNN and DISTRIBUTED can be set OFF. The TTMetal runtime library links only
  tt_metal and device (runtime/lib/ttmetal/CMakeLists.txt:12-30), but libTTMLIRRuntime links METAL_LIBS, which names
  TT-NN's _ttnncpp.so whatever the options (runtime/CMakeLists.txt:29; runtime/lib/CMakeLists.txt:16-53).
- tt-metal: with the runtime on, tt-mlir fetches and builds its own tt-metal as an ExternalProject at TT_METAL_VERSION
  5280a9cf, the same commit as LASSI's pin, inside tt-mlir's source tree (third_party/CMakeLists.txt:3, :87-131), with
  WITH_PYTHON_BINDINGS=ON, ENABLE_DISTRIBUTED=ON on x86, ENABLE_TRACY from TT_RUNTIME_ENABLE_PERF_TRACE (default OFF),
  and its default target, so TT-NN is built. There is no option to use an installed tt-metal tree. LASSI's
  tt-metal@5280a9cf-tracy has libtt_metal.so, libdevice.so, libtt_stl.so, and libtracy but no TT-NN library
  (WITH_PYTHON_BINDINGS OFF, ENABLE_DISTRIBUTED OFF, ENABLE_TRACY ON, TT_UNITY_BUILDS ON; rx 34f2), so it cannot stand
  in without a patch to tt-mlir's CMake. alpha01 has /opt/openmpi-v5.0.7-ulfm (rx 34f2), which ENABLE_DISTRIBUTED uses.
- ttrt and Python: ttrt builds only with TTMLIR_ENABLE_BINDINGS_PYTHON and MLIR_ENABLE_BINDINGS_PYTHON
  (tools/CMakeLists.txt:6, :23-25); the pinned LLVM is built with MLIR_ENABLE_BINDINGS_PYTHON=OFF (tt-mlir.pin:64), so
  ttrt means rebuilding LLVM too. ttrt's run has no option to load given inputs (only --init, --identity, --non-zero,
  --seed, ...; tools/ttrt/common/run.py:20-340), saves tensors as torch .pt files (tools/ttrt/common/util.py:741-784),
  and refuses a flatbuffer whose system descriptor differs from the device's unless --ignore-version
  (util.py:868-907). A small C++ driver on libTTMLIRRuntime fits LASSI better (section 5).
- Size and time, PROJECTED (not measured): tt-metal at the pin has 1899 .cpp files under ttnn/cpp, 811 of them device
  kernels compiled at run time, so about 1088 host files, beside about 244 under tt_metal (the git tree at 5280a9cf,
  2026-10-29). LASSI's tt_metal-only build ran 676 Ninja steps in a 3 min 44 s job with a 131 s configure at 32 jobs
  [MEASURED, results/p4-tt-install]. The TurboQuant checkout, which builds TT-NN with the Python bindings, measured
  5.2 GiB (plans/spikes/p4-tt-pins.md). tt-mlir CI built tt-mlir with the runtime and tt-metal in 9 min 46 s on 64
  CPUs with a warm ccache (plans/spikes/p5-mlir-pins.md). So a runtime build is PROJECTED at about 20 to 60 minutes on
  alpha01 at 32 jobs (no ccache) and about 6 to 9 GiB more space (a second tt-metal with TT-NN, about 5 to 7 GiB, plus
  tt-mlir's runtime and a rebuilt build tree), reusing the installed LLVM build. It is longer than 10 minutes, so it
  needs an owner window or the owner's yes, and it is a pin change.

## 5. A TT-Metal flatbuffer on ttsim

Not run (no runtime). By reading, nothing found rules it out, and four things must hold:

- Same tt-metal: the runtime's tt-metal is 5280a9cf (third_party/CMakeLists.txt:3), the commit LASSI runs on ttsim
  v1.3.4, where tt-metal's TT_METAL_SIMULATOR makes the simulator the target (plans/spikes/p4-ttsim-runtime.md, from
  rtoptions.cpp:387-405). The runtime opens a MeshDevice (runtime/lib/ttmetal/runtime.cpp:249-280), as the tt-pairs
  host programs do (every assets/bench/tt-pairs-v0/*/tt/*.cpp uses MeshDevice), and those passed on ttsim (P4.9).
- Slow dispatch: ttsim runs with TT_METAL_SLOW_DISPATCH_MODE=1 (bible, Execution Backends, the ttsim row). The
  runtime records an event at every return (runtime/lib/ttmetal/executor.cpp:264-266) and synchronizes input events
  (:138-143); at tt-metal 5280a9cf the slow-dispatch queue returns a dummy event and finishes by waiting for idle cores
  (tt_metal/distributed/sd_mesh_command_queue.cpp:128-162), EventSynchronize returns at once without fast dispatch
  (tt_metal/distributed/distributed.cpp:27-37), and only traces throw (sd_mesh_command_queue.cpp:170-175), which these
  flatbuffers do not use (inferred: no trace op in the TTMetal outputs read). Kernels go to the JIT as strings
  (CreateKernelFromString, runtime/lib/ttmetal/executor_utils.h:356-379).
- System descriptor: the flatbuffer must be compiled with system-desc-path of a descriptor the runtime wrote on ttsim
  (section 1, last bullet); the mock one is unsafe. Whether ttsim's grid and L1 base match a real n150's is unmeasured.
- Kernel features: the default matmul lowering uses NOC multicast and semaphores (probe3 counts); whether ttsim v1.3.4
  runs NOC multicast is unmeasured (that the reference matmul_multi_core, which passed on ttsim, uses no multicast is
  inferred from its kernel names, reader_mm_output_tiles_partitioned and writer_unary_interleaved_start_id, not read). `enable-multicast-inference=false` removes multicast from both the
  TTIR and the D2M matmul (probe3). tt-mlir's own golden tests at the pin know a "sim" environment
  (TT_METAL_SIMULATOR set; test/python/golden/conftest.py:38, :153-157) and skip a TTNN case there while keeping the
  ttmetal one (test/python/golden/ttir_ops/eltwise/test_ttir_unary.py:344), which suggests upstream ran the TTMetal
  path on a simulator; no CI job at the pin does (.github workflows, grep), so this is not evidence of a pass.

Outputs as files: the C++ API takes host tensors and gives them back (createBorrowedHostTensor, runtime.h:55; submit,
:268-270; toHost and memcpy, implemented for TTMetal at runtime/lib/ttmetal/runtime.cpp:514, :580; program input and
output descriptions, types.h:330-335). dumpTensor is not implemented for TTMetal (runtime/lib/runtime.cpp:1186-1196).
So an executor needs a small driver: read each lassi_io input, check its dtype and element count against the program's
input description, submit, copy each output to host, and write it as a lassi_io file named by the item's outputs. The
programs take and return host tensors of the module's logical shape (probe3: `enqueue_write_buffer` from a host memref
with `#ttcore.host_layout`, padding included), so the flat lassi_io arrays map onto the 2-D shapes byte for byte.

## Finding

The pinned ttmlir-opt lowers hand-written TTIR and D2M for eltwise_binary, eltwise_sfpu, and matmul_multi_core through
D2M and TTKernel to TT-Metal, and ttmlir-translate writes their flatbuffers; loopback exists only in D2M, and
matmul_single_core at 640^3 does not fit one core's L1, while gather and scatter do not lower at all (exploratory, CPU
only). The pinned install has no runtime, so no flatbuffer can run yet: running one needs the runtime built at the same
pin (a pin change), a ttsim system descriptor, and a small driver, and whether ttsim then runs these programs is
unmeasured.

Confidence: high for sections 1 to 3 (the pinned tools ran on every module; read-only facts from the install's
CMakeCache); medium for section 4's requirements (read at the pin, not built) and low for its time and space
(PROJECTED); low to medium for section 5 (reading only).

## Consequences for the plan

- P17.48 toolchains: one command per language, the full pipeline then the translate:
  `ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=<ttsim.ttsys> enable-multicast-inference=false" <m>.mlir
  -o <m>.ttmetal.mlir` and `ttmlir-translate --ttmetal-to-flatbuffer <m>.ttmetal.mlir -o <m>.ttm`. Not the fe/me/be
  split (the translate refuses its output) and not the explicit-datamovement form (stream_layout survives). Read
  diagnostics as MLIR `<file>:<line>:<col>: error:` lines plus assertion aborts (exit status 134). A compile-only
  toolchain can run today; until a ttsim descriptor exists it compiles against the mock descriptor and its flatbuffers
  are compile evidence only, refused by any executor.
- Per item: TTIR fits eltwise_binary, eltwise_sfpu, and matmul (one op each, so the model's work there is small);
  D2M fits loopback (the only level with data movement) and all four others. matmul_single_core at 640^3 has no
  single-core lowering in either language; as "c = a b" the compiler takes it on a grid of its choosing.
- P17.48 executor: a runtime build (owner item below), a C++ driver on libTTMLIRRuntime run in the sandbox with the
  ttsim row's environment, one runtime-written ttsim descriptor recorded with the pins, and a refusal of any flatbuffer
  whose embedded descriptor is not that one (the runtime does not check addresses).
- P17.47: leave `// UNSUPPORTED: true` tests out of the TTIR and D2M corpora or mark them (two carry forms the pinned
  ops refuse), and substitute or drop `%system_desc_path%` in the Silicon tests' RUN lines.
- bale-v0-tt has no TTIR or D2M route at the pin: no gather or scatter lowering, no element-indexed data movement in
  D2M, and no 64-bit integer type on the device.
- The bible's Target Rules say the model writes each target's kernel-level representation directly (IR Strategy,
  Target Rules); ttir and d2m arms above ttkernel need that sentence changed by the owner's direction of 2026-10-29
  with a Decision Log entry when P17.48 adds the levels.

## Proposed bible edit (IR Strategy, after the "Precedents for the ttkernel output path" paragraph)

Deferred to task P17.48, which adds the TTIR and D2M levels; the owner's question below was filed as OQ-111 and answered (b) on 2026-10-29 (plans/OWNER-QUEUE.md).

> TTIR and D2M above ttkernel (task P17.46, plans/spikes/ttir-d2m.md; exploratory runs of the pinned ttmlir-opt and
> ttmlir-translate on alpha01's CPU, rx 20261029-172249-exec-e512, 20261029-172648-exec-0ae2, and
> 20261029-173004-exec-4263, with hand-written modules sent in the command, so not [MEASURED] until rerun from a clean
> commit): `ttmlir-opt --ttir-to-ttmetal-pipeline` lowers a TTIR or a D2M module through D2M and TTKernel to TTMetal
> with EmitC kernels (lib/Dialect/TTMetal/Pipelines/TTMetalPipelines.cpp:256-303 at 5f396fd6), and
> `ttmlir-translate --ttmetal-to-flatbuffer` writes the flatbuffer; the fe, me, and be sub-pipelines run alone do not
> give a module the translate accepts. Hand-written modules of eltwise_binary (ttir.add), eltwise_sfpu (ttir.exp), and
> matmul_multi_core (ttir.matmul) reach a flatbuffer in TTIR and in D2M; loopback has no TTIR form and reaches one only
> in D2M; matmul_single_core at 640x640x640 fits no single core's L1 under any allocator option tried; ttir.gather
> (through ttir.embedding) and ttir.scatter do not lower, and D2M has neither. The pinned install has no runtime
> (toolchains/tt-mlir.pin: TTMLIR_ENABLE_RUNTIME OFF), so no flatbuffer runs; a flatbuffer compiled without
> system-desc-path uses the mock Wormhole descriptor, whose L1 base is 1024, below tt-metal's reserved L1, and never
> runs on ttsim or a card. [OPEN] the runtime build (OQ-<n>).

Decision Log entry (newest first; the count sentence goes up by one, and 2026-10-29's count by one):

> | 2026-10-29 | TTIR and D2M lower to TT-Metal flatbuffers at the joint pin, and the pin has no runtime to run them
> (task P17.46; IR Strategy): the full pipeline `--ttir-to-ttmetal-pipeline` and `--ttmetal-to-flatbuffer` are the
> route for both levels; flatbuffers compiled against the mock system descriptor never run; the runtime build is
> [OPEN] (OQ-<n>) | Hand-written TTIR and D2M modules of four tt-pairs-v0 items reached flatbuffers on the pinned
> tools, loopback only in D2M, matmul_single_core at 640^3 in neither, and gather and scatter in neither (exploratory,
> rx 20261029-172249-exec-e512, -172648-exec-0ae2, -173004-exec-4263); the install's CMakeCache has the runtime, ttrt,
> and the bindings off (rx 20261029-172018-exec-ff8a); the mock descriptor's L1 base is 1024 (TTCoreOpsTypes.cpp:152-163)
> and the runtime writes buffers at the compiler's addresses with its check compiled out (executor.cpp:426-431,
> debug.h:24-39). plans/spikes/ttir-d2m.md records the commands, outputs, and sources |

## Owner-queue item (draft; filed as OQ-111 with its own option text, answered (b) on 2026-10-29)

```
## OQ-<n> A tt-mlir Runtime At The Pin, To Run TTIR And D2M Flatbuffers On ttsim
State: OPEN
Kind: decision
Blocks: P17.48 (its executor; the compile-only toolchains can be built without it), P17.49's runs
Evidence: plans/spikes/ttir-d2m.md (sections 4 and 5)
Question: The pinned tt-mlir lowers hand-written TTIR and D2M for four tt-pairs-v0 items to TT-Metal flatbuffers, but
the install has no runtime (TTMLIR_ENABLE_RUNTIME OFF), so none can run. Running them on ttsim needs tt-mlir's
runtime built at the same commit, which builds its own copy of tt-metal 5280a9cf with TT-NN, a pin change. Build it?
Options:
(a) Build the TTMetal runtime only (TTMLIR_ENABLE_RUNTIME ON; TTNN runtime, distributed runtime, Python bindings, ttrt
    off) at tt-mlir 5f396fd6 in a new prefix beside the current install, reusing its LLVM build; tt-mlir builds its own
    tt-metal 5280a9cf with TT-NN (no option takes an installed tree). Add a small C++ driver on libTTMLIRRuntime for
    lassi_io inputs and outputs, and write one ttsim system descriptor with it. PROJECTED: 20 to 60 minutes on alpha01
    at 32 jobs and 6 to 9 GiB (not measured); one owner window; a Decision Log entry. Whether ttsim then runs the
    programs is unmeasured until the first run.
(b) As (a) with Python bindings and ttrt. Also rebuilds LLVM with MLIR's Python bindings (larger and longer; not
    projected), and ttrt cannot load given inputs, so LASSI would still need its own driver.
(c) No runtime build: patch tt-mlir's CMake to build the TTMetal runtime against the installed tt-metal@5280a9cf-tracy
    (no TT-NN there, so the link of TT-NN's library must be removed too), or write a LASSI flatbuffer interpreter on the
    installed tree. Saves the second tt-metal; a local patch of tt-mlir, more code to own, and the same pin question.
(d) Defer: build P17.48's ttir and d2m toolchains compile-only (flatbuffers as compile evidence, never run) and keep the
    model on ttkernel and TT-Metal C++ for runs.
Recommendation: (a). It runs tt-mlir's own runtime at LASSI's joint pin with the fewest moving parts, keeps the current
install unchanged, and the driver is needed under every option; schedule the build in the next owner window, with the
first ttsim run of each Tier A item's TTIR flatbuffer as its check.
Answer:
```

## Unmeasured statements in this spike

- Whether any flatbuffer here runs on ttsim, or gives correct results; whether ttsim v1.3.4 runs NOC multicast.
- The runtime build's time and space (PROJECTED above) and whether it builds on alpha01 as configured.
- That slow dispatch suffices for the TTMetal runtime (read at the pin, not run).
- Whether ttsim's descriptor (grid, L1 base, DRAM) matches a card's; whether add_tiles on si32 tiles is exact.
- That a 640^3 single-core D2M module meets the same L1 limit (inferred from the shared allocator).
- That upstream ran the TTMetal path on a simulator (inferred from the golden tests' "sim" handling).
- typecast.mlir's two-operand form being refused (read against TTIROps.td, not run).

## Sources

- tt-mlir 5f396fd6: https://github.com/tenstorrent/tt-mlir/tree/5f396fd6ef85780bf1b6b85bef9c7d2f84a85f3c, read
  2026-10-29: lib/Dialect/TTMetal/Pipelines/TTMetalPipelines.cpp, include/ttmlir/Dialect/TTMetal/Pipelines/
  TTMetalPipelines.h, lib/Conversion/TTIRToD2M/TTIRToD2M.cpp, lib/Conversion/D2MToTTKernel/D2MToTTKernel.cpp,
  lib/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecompositionPass.cpp, include/ttmlir/Conversion/Passes.td,
  include/ttmlir/Dialect/TTIR/IR/TTIROps.td, include/ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.td,
  lib/Dialect/TTCore/IR/TTCoreOpsTypes.cpp, lib/Dialect/TTCore/Transforms/TTCoreRegisterDevice.cpp,
  lib/Target/TTMetal/TTMetalToFlatbuffer.cpp and its registration, CMakeLists.txt, third_party/CMakeLists.txt,
  tools/CMakeLists.txt, runtime/CMakeLists.txt, runtime/lib/CMakeLists.txt, runtime/lib/ttmetal/{CMakeLists.txt,
  runtime.cpp, executor.cpp, executor_utils.h}, runtime/lib/common/{CMakeLists.txt, system_desc.cpp},
  runtime/lib/runtime.cpp, runtime/include/tt/runtime/{runtime.h, types.h, debug.h}, tools/ttrt/common/{run.py,
  util.py}, test/python/golden/conftest.py, test/python/golden/ttir_ops/eltwise/test_ttir_unary.py, and the lit tests
  under test/ttmlir/{Silicon/TTMetal, Conversion/TTIRToD2M, Conversion/D2MToTTKernel, Conversion/D2MToTTMetal,
  Dialect/D2M, Dialect/TTNN/simple_scatter.mlir}.
- tt-metal 5280a9cf: https://github.com/tenstorrent/tt-metal/tree/5280a9cfb00998fd49667a29523d03aee905c129, read
  2026-10-29: tt_metal/distributed/sd_mesh_command_queue.cpp, tt_metal/distributed/distributed.cpp,
  tt_metal/hw/inc/internal/tt-1xx/wormhole/dev_mem_map.h, tt_metal/llrt/hal/tt-1xx/wormhole/wh_hal_tensix.cpp,
  CMakeLists.txt, and the git tree (source counts).
- Tenstorrent package index, read 2026-10-29: https://pypi.eng.aws.tenstorrent.com/ttmlir/ (the 0.9.0.dev20260221
  cp311 wheel, sha256 37b8fd1e7ee1d12b1ba531df9e63038db3cbea97acab2f21547083b4a5bb962e, member list read),
  https://pypi.eng.aws.tenstorrent.com/ttrt/ (404).
- This repository: toolchains/tt-mlir.pin, toolchains/tt-metal.pin, toolchains/ttsim.pin, toolchains/tt-xla.pin,
  assets/bench/tt-pairs-v0.yaml, results/p4-tt-install/summary.md, plans/spikes/p4-tt-pins.md,
  plans/spikes/p5-mlir-pins.md, plans/spikes/p4-ttsim-runtime.md, plans/spikes/mlir-issues.md, docs/BIBLE.md
  (IR Strategy; Execution Backends, the ttsim row; ttsim Facts).
