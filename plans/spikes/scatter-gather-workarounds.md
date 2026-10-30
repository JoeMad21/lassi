# Spike: routes to gather and scatter at the joint pin (2026-10-29)

## Question

The pinned TTIR -> D2M -> TTKernel/TT-Metal path does not lower ttir.gather (decomposed to ttir.embedding, which has
no TTIRToD2M pattern) or ttir.scatter, D2M's remote_load and remote_store move whole shards, and si32 is the widest
device integer (plans/spikes/ttir-d2m.md, section 3). The owner asked for a plan to express scatter and gather. Which
routes work at tt-mlir 5f396fd6ef85780bf1b6b85bef9c7d2f84a85f3c (0.9.0.dev20260221) and tt-metal
5280a9cfb00998fd49667a29523d03aee905c129, what blocks the others, and which bale-v0-tt items each covers? P17.48
(the ttir and d2m languages and their executor) and any bale-v0-tt arm above ttkh depend on the answer.

Classification: factual for what the pinned sources accept (answered below by reading them); a choice for which route
LASSI builds (owner-queue draft at the end).

## Evidence and provenance

- Sections 1 to 6: reading only. No rx command, nothing on alpha01, nothing run. Nothing here is [MEASURED]; every
  statement about run-time behavior is read from source and marked unmeasured where it matters. The section "Compile
  probe at the pin (2026-10-29)" below ran the pinned compiler on alpha01's CPU (task P17.57); it is exploratory.
- tt-mlir 5f396fd6: the commit's tarball (sha256 67afd5515290f7c98dc44170bba1a12d489a98d14044c5b7af2b3843850f03dc,
  the same file plans/spikes/ttir-d2m.md used), unpacked in a scratch directory outside the repository and read with
  grep. tt-metal 5280a9cf: `gh api repos/tenstorrent/tt-metal/contents/<path>?ref=5280a9cfb00998fd49667a29523d03aee905c129`
  (raw), and the commit's git tree from `gh api .../git/trees/5280a9cf...?recursive=1` (17701 entries, not truncated).
  Read 2026-10-29. Every file:line below is at those commits; tt-metal paths are under ttnn/cpp/ttnn/operations/
  unless written in full.

## 1. D2M: can a generic region compute a DMA address from data?

Two levels of data movement exist in D2M, and they differ exactly on this point.

- `d2m.remote_load` and `d2m.remote_store`, the ops the frontend writes and a model would write at the tensor level,
  take grid indices only and always move an entire shard: "Loads an _entire shard_ ... RemoteLoadOp indices correspond
  to the _grid dimensions only_" (include/ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.td:1055-1080; the verifier checks
  the index count equals the grid rank, lib/Dialect/D2M/IR/D2MGenericRegionOps.cpp:179-260). The indices are SSA
  `index` values, so a shard index could come from data, but the granularity is a shard, not an element. A gather of
  table elements cannot be written at this level.
- `d2m.dma_read` and `d2m.dma_write` ("Fully indexed and lowered D2M DMA Op", D2MGenericRegionOps.td:906-1018) take
  `Variadic<Index>` source and destination indices and a static element count `numElems` (I64Attr). The verifiers ask
  only for 3 remote indices and 1 local index and matching element types; nothing requires constant indices
  (D2MGenericRegionOps.cpp:69-117 for write, :119-143 for read). The lowering forwards the SSA offset into the address:
  buildNocAddress casts the third index to i32 and adds it to the buffer base, then emits `ttkernel.get_noc_addr`
  (L1: [gridY, gridX, byte offset]) or `get_noc_addr_from_bank_id` (DRAM: [_, bank, byte offset])
  (lib/Conversion/D2MToTTKernel/D2MToTTKernel.cpp:1410-1444); buildL1Address does the same for the local side
  (:1446-1453); D2MDMAReadRewriter emits `ttkernel.noc_async_read` with size numElems times the element size
  (:1455-1500). The units are bytes after the memory map (LowerLoadStoreOpsToDMA.cpp:268-300 builds them from
  remote_load through `applyMap`).
- A datamovement thread may read an integer from L1: convert-d2m-to-ttkernel keeps `memref.load` and `memref.store` on
  integer element types legal ("Allow loads and stores to integer element types. i.e. riscv accesses to L1.",
  lib/Conversion/D2MToTTKernel/D2MToTTKernelPass.cpp:85-93), and convert-ttkernel-to-emitc lowers them through
  upstream MemRefToEmitC (lib/Conversion/TTKernelToEmitC/TTKernelToEmitC.cpp:959, :973-974).

So the IR and the lowering allow a gather written as: in a datamovement thread, `dma_read` the index shard into L1,
`dma_wait`, then per element `memref.load` an index, compute `offset = idx * elem_bytes` with arith, `dma_read` one
element of the table at `[gy, gx, offset]` into the output buffer at `i * elem_bytes`, and `dma_wait`. Where it
stops: nothing in the pipeline produces dma_read from data (the frontend only lowers remote_load through memory maps,
LowerLoadStoreOpsToDMA.cpp), and dma_read is defined on post-bufferization memrefs whose bases are allocated buffers,
so a model would have to write the post-allocation, explicit-datamovement form by hand. The nearest pinned precedent,
test/ttmlir/Dialect/D2M/generic/explicit_datamovement_form.mlir, lowers but its output does not translate
(`d2m.stream_layout` survives; plans/spikes/ttir-d2m.md, section 1). Whether a hand-written dma_read module survives
the me and be pipelines to a flatbuffer is unmeasured. Verdict: allowed by the IR and the lowering, blocked by the
lack of an entry point; this level is as hard to write as TTKernel and gains nothing over route 2.

## 2. A hand-written TTKernel kernel inside a compiler-built program

Two upstream precedents at the pin, both compiled by the pinned tools and run by upstream's golden tests (on silicon
with a fabric; not run here):

- TTNN level, `ttnn.generic`: "Generic operation capable of running a program with custom kernels. Each kernel is
  described with a symbol reference to its function ... plus compile and runtime arguments" (include/ttmlir/Dialect/
  TTNN/IR/TTNNOps.td:2997-3014). test/python/golden/fabric_api_snippets/test_generic_op_unicast.mlir:10-56 is a TTNN
  function (ttnn.to_layout, to_device, ttnn.generic, to_memory_config, from_device) whose generic op runs
  `@datamovement_kernel0`, a hand-written ttkernel function (:59, `ttkernel.thread = #ttkernel.thread<noc>`). The test
  compiles it with `ttcore-register-device` and `ttnn_to_flatbuffer_bin` only (test/python/golden/test_fabric_apis.py:
  1066-1080). The translation turns each kernel symbol into C++ with translateTopLevelKernelToCpp
  (lib/Target/TTNN/TTNNToFlatbuffer.cpp:2931-2945; lib/Target/TTKernel/TTKernelToCpp.cpp:407), and the TT-NN runtime
  has the op (runtime/lib/ttnn/operations/generic/generic_op.cpp). Kernel arguments include a tensor's buffer address
  (`#ttnn.kernel_arg_address_of_tensor<i>`, TTNNOpsAttrs.td:958-966), CB indices (:948), semaphores (:968), and named
  args (:978). TTNNOps.td:3016 also defines `ttnn.d2m_subgraph`, D2M-compiled subgraphs as ttnn.generic ops inside a
  TTNN program (TTNNPipelines.h:278, enable-d2m-fusing-pass).
- TTMetal level: test/python/golden/fabric_api_snippets/test_fabric_unicast.mlir:7-16 is a hand-written TTMetal host
  function (`ttmetal.create_buffer` with an explicit L1 address 104128, enqueue_write_buffer, enqueue_program naming
  `@datamovement_kernel0`, enqueue_read_buffer, finish) and :20-58 a hand-written ttkernel kernel; the test runs it
  through `ttir-to-ttmetal-pipeline` and `ttmetal_to_flatbuffer_bin` (test_fabric_apis.py:144-160), the same command
  P17.48 plans for TTIR and D2M. The test_generic_op_unicast.mlir host function also writes
  `tt.function_type = "forward_device"` and the kernel `tt.function_type = "kernel"` by hand (:10, :59), the
  attribute whose absence made the translate refuse the fe/me/be re-entry (plans/spikes/ttir-d2m.md, section 1);
  that adding it by hand fixes the re-entry is inferred, unmeasured.

In ttkernel a data-dependent read is ordinary: `ttkernel.noc_async_read(noc_addr, l1_addr, size)` takes SSA values
(include/ttmlir/Dialect/TTKernel/IR/TTKernelOps.td:2377-2389), with `get_noc_addr_from_bank_id` (:2363),
`get_interleaved_addr_gen_fast` and its get_noc_addr (:2836, :2850), TensorAccessor and its get_noc_addr (:3227,
:3241), `reinterpret_cast<volatile tt_l1_ptr uint32_t*>` (:2808) and `store_to_l1` (:2823). What is not settled:
TTKernel has no op named for a scalar L1 load (grep of TTKernelOps.td); a kernel reads an index either through
`memref.load` on an integer memref (legal into EmitC, TTKernelToEmitC.cpp:973-974) or EmitC ops on the L1 pointer
(the pointer type converts to an EmitC pointer, :129). Which of these a hand-written ttkernel function can use
end to end is unmeasured. The bale-v0-ttkernel histo kernel already does the same work in TT-Metal C++
(assets/bench/bale-v0-ttkernel/histo/kernels/histo_kernel.cpp:41-42, int64 pointers into L1).

Supported splice at the pin: yes for TTNN (`ttnn.generic` beside TT-NN ops in one TTNN function, upstream-tested
form) and for TTMetal (host ops plus kernels, upstream-tested form). Not shown: a TTIR module that contains a
hand-written kernel. In TTNN the dense parts would be TT-NN ops the model writes in the TTNN dialect, or TTIR
lowered by `--ttir-to-ttnn-backend-pipeline` and then spliced, a two-step flow whose acceptance of ttnn.generic in the
input is unmeasured. In TTMetal, compiler buffers come from d2m-allocate and hand-written ones carry fixed addresses,
so mixing TTIR ops and a hand-written program in one function risks overlapping L1 (inferred; unmeasured).

## 3. TTIR -> TTNN at the pin

The decomposition rewrites ttir.gather to ttir.embedding in both the ttnn and ttmetal configs
(lib/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecompositionPass.cpp:67-72). TTIRToTTNN then has patterns for
embedding (lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp:369-384 -> ttnn.embedding), scatter (:2505-2518 ->
ttnn.scatter with dim and scatter_reduce_type), sort (:548-565 -> ttnn.sort with dim, descending, stable), cumsum
(:498-511 -> ttnn.moreh_cumsum), argmax (:339-366), all registered at :3061-3105. TTNN has no gather and no topk op
at the pin (TTNNOps.td: embedding :997, moreh_cumsum :1120, sort :1234, scatter :2547; no gather or topk def), and
TTIR has no topk or one_hot op (TTIROps.td). The ttir.scatter op is torch-style along one dim, with reduce type
SUM, PROD, MIN, MAX, or INVALID (TTIROps.td:4789-4834).

What TT-NN at 5280a9cf accepts, and what tt-mlir's workarounds force:

| Op | tt-metal 5280a9cf | tt-mlir workaround at 5f396fd6 | Exactness for bale |
| --- | --- | --- | --- |
| ttnn.embedding | weights must be BFLOAT16 and ROW_MAJOR; indices UINT32 or BFLOAT16; for tiled output the table width must be a multiple of 32 (embedding/device/embedding_device_operation.cpp:33-36, :54-58) | indices to row-major ui32, weights to row-major bf16, output to tiled bf16 (lib/Dialect/TTNN/IR/TTNNWorkaroundsPass.cpp:149-166) | bf16 holds integers exactly only to 256; a value table must be split into 8-bit digits (one bf16 column each) and recombined |
| ttnn.scatter | reductions "add" and "multiply" only (data_movement/scatter/scatter.cpp:24, :63-69); amin and amax fail there although tt-mlir's runtime maps MIN and MAX to them (runtime/lib/ttnn/operations/data_movement/scatter.cpp:26-52); index INT32, UINT8, UINT16, or UINT32, input and source the same dtype, interleaved (scatter/device/scatter_device_operation.cpp:29-55); no fp32 tiled, and no tiled int32 longer than 256 on the scatter axis (scatter.cpp:104-140); rank-equal operands, index no longer than source (:45-90) | row-major whenever input or source is a tiled f32, bf16, or 32-bit integer (TTNNWorkaroundsPass.cpp:212-250), which sidesteps the 256 limit | the generic path does the reduction on the RISC-V in the element type (`return input + source_value;`, scatter/device/kernels/dataflow/reader_scatter.cpp:28-49, applied per index at :80-91), so si32 add is exact by reading; bf16 with a reduction takes a separate program (scatter_device_operation.cpp:15-20) |
| ttnn.sort | 4-D, tiled, BFLOAT16 or UINT16 input, last dim a multiple of 64, height a multiple of 32 (data_movement/sort/device/sort_device_operation.cpp:65-83); bitonic, padded to a power of two (sort.cpp:56-85); takes `stable` (sort.cpp:158-203) | integer input converted to ui16, f32 to bf16; indices ui16 or ui32 by padded size (TTNNWorkaroundsPass.cpp:836-870) | keys below 65536 are exact as ui16 (bale's row_key, col_key, weights, columns); that `stable=true` gives a stable order is unmeasured |
| ttnn.moreh_cumsum | calls ttnn::cumsum (moreh/moreh_cumsum/moreh_cumsum.cpp:10-17); tiled, interleaved, on device; no dtype check in validate (reduction/accumulation/device/accumulation_device_operation.cpp:36-60) | none found | whether int32 cumsum is exact is unmeasured |

Data types: the TTNN pipeline runs ttir-element-type-normalization (lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp:
252-256), which narrows i64 to 32 bits (constants by `truncSSat(32)`, lib/Dialect/TTIR/Transforms/
ElementTypeNormalization.cpp:112). So a bale module must use si32 or ui32 at its boundary and LASSI's driver must
convert the i64 lassi_io files (all bale-v0-tt values except ig's table and outputs fit 32 bits; ig's table spans
plus or minus 2^40).

Control flow: the TTNN flatbuffer has no loop op; the only non-library op is FuncCallOp (include/ttmlir/Target/TTNN/
operations/mlir_native.fbs:6-10). A data-dependent number of rounds (sssp, toposort) needs a fixed unroll bound or a
host loop that reruns a one-round program.

On ttsim: the runtime being built tonight includes TT-NN, but upstream's golden tests skip at least one TTNN case on
the simulator while keeping the ttmetal one (test/python/golden/ttir_ops/eltwise/test_ttir_unary.py:344; plans/
spikes/ttir-d2m.md, section 5). Whether these TT-NN ops run on ttsim v1.3.4 is unmeasured and is the first thing to
measure.

## 4. Dense reformulations

| Formulation | Ops at the pin | Size for bale-v0-tt | Exactness |
| --- | --- | --- | --- |
| Gather as one_hot(idx) @ table | one-hot by `ttir.eq` of a broadcast index against `ttir.arange`, then `ttir.matmul`; both lower in TTIRToD2M and TTIRToTTNN (TTIRToD2M.cpp:2232-2319; TTIRToTTNN.cpp:1338) | ig: 16384 x 16384 one-hot (2.7e8 elements, 512 MiB bf16), above the 64 cores' L1 (about 92 MiB, 1499136 bytes each less reserved); DRAM memspaces or TT-NN's DRAM interleaving needed; 16384 x 16384 x 32 MACs with 8-bit digit columns | one nonzero product per output, so exact if each table entry is representable: bf16 needs 8-bit digits (41-bit values: 6 digits); f32 matmul input precision on the FPU is unmeasured |
| Scatter-add / histogram as one_hot(idx)^T @ values or a column sum | ttir.eq, ttir.sum (D2M and TTNN) or matmul | histo: 16384 x 4096 one-hot (6.7e7 elements, 128 MiB bf16) | counts up to 16384 need f32 accumulation (exact to 2^24); bf16 accumulation is not exact above 256; FPU reduce accumulation mode unmeasured |
| Sort / argsort | ttir.sort to ttnn.sort (TTNN only; no TTIRToD2M pattern) | bale keys below 65536, lengths 2048 to 32768 | ui16 keys exact; stability unmeasured |
| Prefix sums | ttir.cumsum to ttnn.moreh_cumsum (TTNN only); in D2M a prefix sum along a dim of length n is a matmul by an n x n triangular ones matrix | transpose 2000 columns, permute 2048 | int32 cumsum exactness unmeasured; the matmul form inherits matmul precision |
| Gather by a permutation | inverse q by scatter of arange at p, then y = scatter(x at q): no gather needed | any permutation (sort results) | exact (copy scatter, si32) |
| Non-permutation gather by sort-join | stable sort of keys 2j (table) and 2*idx+1 (requests), cumsum of value deltas per 16-bit limb forward-fills each request, scatter back by the sort permutation | ig: 32768 keys, below 65536 so ui16 | limbs keep partial sums in range; relies on int32 cumsum (unmeasured) and stable sort (unmeasured) |

## 5. The eight bale-v0-tt apps by route

Route names: A = TTIR -> TTNN with ttir.scatter, ttir.sort, ttir.cumsum, ttir.embedding, and dense ops; B = TTNN
with ttnn.generic and a model-written ttkernel kernel; C = TTMetal host ops with a model-written ttkernel kernel; D =
TTIR -> TTMetal (D2M) with dense one-hot ops only; E = D2M dma_read (section 1); F = host-side gather and scatter
(CPU-hoisted ops, section 6); ttkh = today's C++ TT-Metalium program.

| App | A (TTIR -> TTNN) | B or C (kernel splice) | D (D2M dense) |
| --- | --- | --- | --- |
| histo | `ttir.scatter(zeros si32[4096], index, ones si32[16384]) {dim = 0, scatter_reduce_type = SUM}`; one op | yes, the ttkh kernel's loop in ttkernel | one-hot 16384 x 4096 and a column sum in f32 |
| ig | ttir.gather -> embedding with the table as 8-bit digit columns (16384 x 32 bf16), recombined to i64 on the host; or the sort-join | yes, exact, i64 bytes moved as bytes | one-hot matmul with digit columns; host recombination |
| triangle | L by a flat copy-scatter of ones at max(u,v)*2048+min(u,v), u != v (repeats collapse), reshape 2048 x 2048, typecast, `ttir.matmul(L, L)`, multiply by L, sum in f32 (about 85, PROJECTED in the manifest) | yes | L as onehot(u)^T @ onehot(v) (8192 x 2048 each) clamped to 1, then the same; 2048^3 MACs |
| transpose_matrix | dense B (1000 x 2000) by flat copy-scatter; counts by `ttir.sum` over rows; offsets by `ttir.cumsum` with a leading 0; each nonzero's slot = offset[j] + (cumsum of B down column j) - 1, value = row id (broadcast arange); masked copy-scatter into a padded column[8000] | yes | needs prefix sums (triangular matmul) and the final placement scatter (one-hot 8000 x 8000): heavy |
| permute_matrix | row_dest by stable `ttir.sort` of row_key (ui16) and the inverse scatter; col_dest likewise; dense B by flat scatter; slot = new_offset[row_dest[s]] + per-row prefix; value = col_dest[c] broadcast along rows; masked scatter into padded column[8192] | yes | no sort in D2M: argsort by rank counting (2048 x 2048 compares and a row sum) then one-hot placement: heavy |
| randperm | 8 rounds unrolled: aims are static slices of darts; per round a 2048 x 4096 mask, winner per slot by a min reduction of dart ids (min as -max(-x)), placement by a row reduction; then compaction by cumsum and copy-scatter (amin scatter is refused at the pin, so the winner rule is dense) | yes | the same dense rounds; compaction by one-hot placement |
| sssp | per round dense min-plus on a 2048 x 2048 weight matrix; building W with the lightest repeat needs amin scatter (refused) or a sort by weight and an ordered copy-scatter (write order unmeasured); rounds until no change need a host loop (no loop op) | yes, rounds inside the kernel | as A, with no scatter: W from one-hot products is heavy; host loop needed |
| toposort | per level dense: remaining-row counts by matvec, level rows where the count is 1, positions by cumsum; key ranks by stable sort; up to 2048 levels need a host loop | yes | as A without sort and cumsum; host loop needed |

What the model writes per route: A, one TTIR function per item of 1 (histo) to about 30 ops (permute_matrix,
randperm), no kernels; B, a TTNN function (layouts, to_device, ttnn.generic with its CBs and args) plus one ttkernel
data-movement function, about the ttkh program's content in MLIR; C, a TTMetal host function with buffer addresses
plus kernels; D, TTIR as A but only with ops TTIRToD2M lowers.

## 6. Host-side decomposition

Ops tagged `ttir.should_hoist` (TTIROpsAttrs.td:81-83) move to a CPU module in both pipelines
(lib/Dialect/TTMetal/Pipelines/TTMetalPipelines.cpp:266-267; lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp:243-244) and
lower through TTIR-to-Linalg, which has gather and cumsum patterns (lib/Conversion/TTIRToLinalg/TTIRToLinalg.cpp:1684,
:2078) and no scatter or sort pattern by grep. So ig's gather can run on the host CPU inside the flatbuffer. For
bale the irregular access is the benchmark, so a host-side gather measures the host, not the device; it fits as a
labelled baseline, not as a device result.

## Finding

At the pin, the TTIR -> TTNN path lowers ttir.scatter (add or copy; min and max are refused by TT-NN), ttir.sort (ui16
or bf16 keys), ttir.cumsum, and ttir.gather (through a bf16-only embedding), and TTNN's `ttnn.generic` runs
model-written ttkernel kernels beside TT-NN ops in a form upstream compiles and tests, so every bale-v0-tt app has a
route at this pin. The D2M path can express a data-dependent DMA only in its post-allocation dma_read form, which no
pinned entry point carries to a flatbuffer, so D2M alone covers bale only through dense one-hot formulations.

Confidence: high for what the pinned sources define and refuse (read at the commits); low for anything at run time:
none of it ran, TT-NN on ttsim is unmeasured, and stable sort, int32 cumsum, ordered copy-scatter, and f32 matmul
precision are unmeasured.

## Consequences for the plan

- Route A (TTIR -> TTNN) needs: the TTNN pipeline toolchain (`--ttir-to-ttnn-backend-pipeline`, then
  `ttmlir-translate --ttnn-to-flatbuffer`), a TT-NN executor on the runtime being built (its TTNN half), i64 to si32
  input conversion with a range check, ig's table as 8-bit digits with host recombination (or the sort-join),
  padded outputs trimmed by a length the program also returns (transpose_matrix, permute_matrix), and a host loop
  driver for sssp and toposort. First measurement: whether ttnn.scatter, ttnn.sort, ttnn.moreh_cumsum, and
  ttnn.embedding run on ttsim v1.3.4.
- Route B (ttnn.generic) needs the same TTNN executor plus a language whose files are a TTNN function and ttkernel
  functions; it reaches all eight apps exactly, and its first check is one kernel that reads an index from L1.
- Route C reuses the TTMetal executor P17.48 already plans; route D and E need nothing new but cover less (D) or
  nothing yet (E).
- A newer pin is not needed for routes A to D. It would be needed for amin/amax scatter in TT-NN, a non-bf16
  embedding, a TTIR-level kernel splice, or a D2M frontend that emits data-dependent DMA; none was checked at a newer
  commit here.

## Proposed bible edit (IR Strategy, after the TTIR and D2M paragraph of 2026-10-29)

> Gather and scatter at the joint pin (plans/spikes/scatter-gather-workarounds.md, read at tt-mlir 5f396fd6 and
> tt-metal 5280a9cf, nothing run): the TTMetal (D2M) path lowers neither; its remote_load and remote_store move whole
> shards (D2MGenericRegionOps.td:1055-1080), and only the post-allocation d2m.dma_read takes a data-dependent offset
> (D2MToTTKernel.cpp:1410-1500), which no pinned entry point carries to a flatbuffer. The TTNN path lowers ttir.scatter
> to ttnn.scatter (add and copy; TT-NN refuses amin and amax, scatter.cpp:24), ttir.sort to ttnn.sort (ui16 or bf16
> keys), ttir.cumsum to ttnn.moreh_cumsum, and ttir.gather through ttir.embedding to a bf16-only ttnn.embedding
> (embedding_device_operation.cpp:36). TTNN's ttnn.generic runs hand-written ttkernel kernels beside TT-NN ops
> (TTNNOps.td:2997-3014; upstream test_generic_op_unicast.mlir), and the TTMetal pipeline accepts hand-written TTMetal
> host ops with ttkernel kernels (test_fabric_unicast.mlir). [OPEN] which route LASSI builds (OQ-<n>).

Decision Log entry (newest first; the count sentence and 2026-10-29's count go up by one):

> | 2026-10-29 | Gather and scatter have routes at the joint pin through TTNN, not through D2M (IR Strategy): ttir.scatter,
> ttir.sort, ttir.cumsum, and a bf16 embedding lower to TT-NN, and ttnn.generic splices model-written ttkernel kernels;
> the route LASSI builds is [OPEN] (OQ-<n>) | Read at tt-mlir 5f396fd6 and tt-metal 5280a9cf: TTIRToTTNN.cpp:369-384,
> :498-565, :2505-2518; TTNNWorkaroundsPass.cpp:149-166, :212-250, :836-870; scatter.cpp:24; embedding_device_operation.cpp:36;
> D2MGenericRegionOps.td:1055-1080; D2MToTTKernel.cpp:1410-1500; TTNNOps.td:2997-3014. Nothing ran; ttsim support of
> the TT-NN ops is unmeasured. plans/spikes/scatter-gather-workarounds.md records the sources |

## Owner-queue item (draft)

```
## OQ-<n> Which Route Expresses Gather And Scatter For bale-v0-tt Above ttkh
State: OPEN
Kind: decision
Blocks: bale-v0-tt arms in TTIR, TTNN, or TTKernel MLIR (P17.48 follow-ups); none of today's ttkh work
Evidence: plans/spikes/scatter-gather-workarounds.md, plans/spikes/ttir-d2m.md
Question: The pinned D2M path cannot lower gather or scatter, but the pinned TTNN path lowers ttir.scatter (add and
copy), ttir.sort, ttir.cumsum, and a bf16 embedding, and ttnn.generic runs model-written ttkernel kernels beside
TT-NN ops. None of it has run on ttsim. Which route should LASSI build for bale-v0-tt above ttkh?
Options:
(a) TTIR -> TTNN (route A): the model writes one TTIR function per item with scatter, sort, cumsum, and dense ops.
    Covers histo, ig (digit split), triangle, transpose_matrix, permute_matrix, and randperm in one program; sssp and
    toposort need a host loop driver. LASSI builds the TTNN toolchain and executor, i64 conversion, padded outputs,
    and the loop driver. Least model code; exactness rests on unmeasured int32 cumsum, stable sort, and TT-NN on ttsim.
(b) TTNN with ttnn.generic (route B): the model writes a TTNN function and a ttkernel kernel per item. Covers all
    eight exactly, in an upstream-tested form; LASSI builds the TTNN executor and a ttnn-plus-ttkernel language. The
    model's work is close to ttkh's, in MLIR.
(c) Both: (a) as the compiler-first arm and (b) as the escape hatch for items (a) cannot express or gets wrong,
    with the TTNN executor shared.
(d) Stay on ttkh for bale and keep TTIR and D2M for tt-pairs only.
Recommendation: (c), starting with one ttsim measurement before any build beyond the executor: ttir.scatter (histo)
and one ttnn.generic kernel that reads an index from L1, on the runtime built tonight. If TT-NN does not run on ttsim,
fall back to (d) and route C (TTMetal host ops with ttkernel kernels, which the TTMetal executor already covers).
Answer:
```

## Unmeasured statements in this spike

- That any TT-NN op named here (scatter, sort, moreh_cumsum, embedding, generic) runs on ttsim v1.3.4, or on a card.
- That a hand-written d2m.dma_read module, or the fe/me/be re-entry with `tt.function_type` added by hand, reaches a
  flatbuffer.
- That a hand-written ttkernel function can read an L1 scalar end to end (memref.load or EmitC ops).
- That ttnn.sort with stable=true is stable; that ttnn.moreh_cumsum is exact on int32; that ttnn.scatter writes
  repeated indices in index order; the FPU's precision for f32 matmul and reductions.
- That `--ttir-to-ttnn-backend-pipeline` accepts a module that already holds ttnn.generic; that mixing TTIR ops and
  hand-written TTMetal buffers in one function avoids L1 overlap.
- Every size and op count in sections 4 and 5 is arithmetic from the manifest's shapes (PROJECTED), not a run.

## Compile probe at the pin (2026-10-29)

Task P17.57 (route A, the owner's answer to OQ-112: "Go with your plans in order, starting with A."). Exploratory:
the hand-written modules and the steps were sent inside one `rx exec` command from a dirty worktree, not from a
commit, so nothing here is [MEASURED]; a rerun from a clean commit can match each module by its sha256 prefix. CPU
only: compiler tools alone, no device opened, no runtime, no model, no `--big`; each exec well under 10 minutes. Runs on
alpha01 under the owner's standing permission for short CPU-only runs, while the window2 sweep ran there; rx ids use
alpha01's clock (UTC-7), and the probe printed UTC 2026-10-30T03:47:57Z to 03:47:59Z.

| rx id | What |
| --- | --- |
| 20261029-204757-exec-d2a1 | probe1: eight TTIR modules through verify, the TTNN backend pipeline, and the TTNN flatbuffer translation |
| 20261029-204857-exec-9cd1 | Read-only: the eight flatbuffers' first 16 bytes and the lowered bodies of sort, cumsum, the gathers, scatter_args, and identity |
| 20261029-205004-exec-e50a | Read-only: the first 16 bytes of four TT-Metal flatbuffers of plans/spikes/ttir-d2m.md's probe1 |

Tools: the install of toolchains/tt-mlir.pin, `M=$LASSI_TOOLCHAINS/tt-mlir@0.9.0.dev20260221`, ttmlir-opt sha256
58c8997c56d7538a... and ttmlir-translate c038995900f7f36b... (the prefixes plans/spikes/ttir-d2m.md recorded);
`--version` prints LLVM 22.0.0git, "Optimized build with assertions". Outputs under
`$LASSI_RUNS_ROOT/ttnn-probe/probe1/` (1.8M; the modules in m/, everything else in out/; nothing deleted). Each case
ran, with `prlimit --pid $$ --core=1:1` read back as 1 1:

```
OPT=$M/build/tt-mlir/bin/ttmlir-opt; TR=$M/build/tt-mlir/bin/ttmlir-translate
timeout 240 $OPT --mlir-disable-threading m/<case>.mlir -o out/<case>.verified.mlir
timeout 240 $OPT --mlir-disable-threading --ttir-to-ttnn-backend-pipeline m/<case>.mlir -o out/<case>.ttnn.mlir
timeout 240 $TR --ttnn-to-flatbuffer out/<case>.ttnn.mlir -o out/<case>.ttnn
od -A d -c -N 16 out/<case>.ttnn
```

No system-desc-path and no mock-system-desc-arch: the pipeline's defaults register the mock wormhole_b0 descriptor
(TTNNPipelines.h:182-202; TTNNPipelines.cpp:34-42), as ttir-ttmetal-compile compiles TT-Metal, so every flatbuffer
here is compile evidence only. `--mlir-disable-threading` kept the probe on one thread beside the sweep; the toolchain
does not pass it. The option names are read at the pin: `system-desc-path` and `mock-system-desc-arch`
(include/ttmlir/Dialect/TTNN/Utils/PassOverrides.h:32-33), the pipeline `ttir-to-ttnn-backend-pipeline`
(TTNNPipelines.cpp:599-603), and the translation `ttnn-to-flatbuffer` (lib/Target/TTNN/TTNNToFlatbufferRegistration.cpp:23-24).

The modules (SYNTHETIC, written for the probe; no bench item's file):

| Case | sha256 prefix | Module |
| --- | --- | --- |
| scatter_args | 72090220f932cba9 | `ttir.scatter` SUM along dim 0: input si32[4096], index si32[16384], source si32[16384], all arguments |
| scatter_histo | df32555b0c257400 | The same scatter of `ttir.zeros` si32[4096] and `ttir.ones` si32[16384] made in the module; the index the one argument |
| scatter_i32 | 087c24a378ac6067 | scatter_args in signless i32, the integer form of the pinned Dialect/TTNN/simple_scatter.mlir |
| sort | d5590d78581456b8 | `ttir.sort` of si32[2048], dim -1, ascending, stable = true; values and indices si32[2048] |
| cumsum | b0d83e618056b068 | `ttir.cumsum` of si32[2048] along dim 0 |
| gather_si32 | a84566b86d4fca0a | `ttir.gather` of rows of bf16[16384, 32] by si32[16384, 1] to bf16[16384, 32], the form of gather_3 in the pinned Dialect/TTNN/embedding/gather_to_embedding.mlir (offset_dims [1], collapsed_slice_dims [0], start_index_map [0], index_vector_dim 1, slice_sizes [1, 32]) |
| gather_ui32 | 7d1c9ab7d9db9fd4 | gather_si32 with ui32 indices |
| identity | 02e36630c463c6e5 | Returns its si32[2048] argument |

Results (exit status 0 means the step passed; times are wall time from `date +%s%N` around each command):

| Case | verify | lower | ttnn ops in the lowered module (count) | translate | Flatbuffer bytes, identifier at bytes 8 to 12 |
| --- | --- | --- | --- | --- | --- |
| scatter_args | 0 | 0, 35 ms | scatter 64, slice_static 128, to_layout 130, deallocate 324 | 0, 39 ms | 435,864, TTNN |
| scatter_histo | 0 | 0, 37 ms | scatter 64, slice_static 128, to_layout 129, zeros 1, ones 1, get_device 2, deallocate 323 (and ttcore.load_cached 2) | 0, 39 ms | 460,416, TTNN |
| scatter_i32 | 0 | 0, 40 ms | as scatter_args | 0, 33 ms | 433,632, TTNN |
| sort | 0 | 0, 32 ms | sort 1, typecast 3, deallocate 4 | 0, 23 ms | 8,360, TTNN |
| cumsum | 0 | 0, 27 ms | moreh_cumsum 1, reshape 2, deallocate 3 | 0, 23 ms | 7,104, TTNN |
| gather_si32 | 0 | 0, 28 ms | embedding 1, typecast 1, to_layout 2, reshape 1, deallocate 6 | 0, 17 ms | 11,024, TTNN |
| gather_ui32 | 0 | 0, 26 ms | embedding 1, to_layout 2, reshape 1, deallocate 5 | 0, 22 ms | 9,568, TTNN |
| identity | 0 | 0, 29 ms | none | 0, 18 ms | 3,024, TTNN |

Per op (rx d2a1 and 9cd1):

- ttir.scatter (SUM, si32, histo's shape): passes. The lowering splits the 16384 indices into 64 `ttnn.scatter` ops of
  256 indices each, each after a `ttnn.slice_static` of the index and of the source and a `ttnn.to_layout` to
  row-major, chained through the 4096-element input; SUM is kept on each and every operand stays si32 (the split
  upstream's simple_scatter.mlir checks as 284 indices to 256 + 28; the row-major move is the workaround of section 3).
  Signless i32 prints as si32 after element type normalization. With zeros and ones made in the module, const-eval
  hoists `ttnn.zeros` and `ttnn.ones` (with the 64 slices of the ones) into two const_eval functions read by
  `ttcore.load_cached`, and the forward function keeps the 64 scatters.
- ttir.sort (si32 keys, 2048): passes, onto ui16. A `ttnn.typecast` takes the keys from si32 to ui16 before
  `ttnn.sort` (descending = false, dim = -1 : si8, stable = true), the values come back to si32 by a second typecast,
  and the indices come out as ui16 and are typecast to si32 (the workaround of section 3, TTNNWorkaroundsPass.cpp:
  836-870). So the program is exact only for keys in [0, 65535], as bale's are; what the typecast does with others is
  unmeasured.
- ttir.cumsum (si32, 2048): passes, si32 kept. The [2048] tensor is reshaped to [2048, 1, 1, 1] and
  `ttnn.moreh_cumsum` runs along dim 0 in tile layout, `memref<2048x1x!ttcore.tile<32x32, si32>>`: one 32 x 32 tile
  per element, 2048 tiles (8 MiB of si32 tiles in DRAM, PROJECTED by arithmetic). No other shape was probed.
- ttir.gather (bf16 table 16384 x 32, 16384 indices): passes, through `ttnn.embedding` as section 3 read: indices
  [16384, 1] and the table both taken to row-major by `ttnn.to_layout`, the embedding giving [16384, 1, 32] bf16, and a
  `ttnn.reshape` to [16384, 32]. si32 indices get one `ttnn.typecast` to ui32 first; ui32 indices none. The table
  stays bf16, so ig needs section 5's 8-bit digit columns.
- identity: lowers to a forward function with no ttnn op at all and still translates (3,024 bytes), so a check for
  device work must look for a ttnn op that computes, not for a flatbuffer.

Every case:

- One warning from the pipeline at the module's first line, `m/<case>.mlir:2:1: warning: Empty argument type map
  provided. Skipping argument type population. This may affect subsequent compile steps` (the argument-type pass that
  createTTNNPipelineTTIRPasses adds, TTNNPipelines.cpp:44-45, with no argument-types option given).
- The lowered forward function keeps the module's element types and shapes at its boundary (si32, ui32, bf16), each
  with a `#ttnn_layout` encoding (DRAM, interleaved, tiled), and carries `tt.function_type = "forward_device"`.
- The TTNN flatbuffer is size-prefixed: bytes 0 to 4 hold its length less 4 (7100 for cumsum's 7104 bytes), 4 to 8
  the root offset (28), and 8 to 12 the identifier TTNN (all eight, rx 9cd1), as the schema names it
  (include/ttmlir/Target/TTNN/binary.fbs:18) and TTNNToFlatbuffer.cpp:4205 writes it
  (FinishSizePrefixedTTNNBinaryBuffer).

A correction this probe found: the TT-Metal flatbuffers of plans/spikes/ttir-d2m.md are size-prefixed too, TTM0 at
bytes 8 to 12 of eltwise_binary.ttm, eltwise_binary_1d.ttm, eltwise_binary_f32out.ttm, and eltwise_sfpu.ttm (rx e50a;
TTMetalToFlatbuffer.cpp:1068, FinishSizePrefixedTTMetalBinaryBuffer). lassi/toolchains/ttir_compile.py, at this
probe's base (bd5e51f), read the identifier at bytes 4 to 8. sweep13's first launch had already shown it on the
branch: it scored every TTIR attempt that reached the translation as no-artifact, and commit 7d3f3e8 (task P17.49)
then accepted the size-prefixed form beside the plain one. P17.57 narrows the check to bytes 8 to 12 for both
routes (TT-Metal and TT-NN), the form the pinned translations write.

Consequences:

- The toolchain ttir-ttnn-compile (lassi/toolchains/ttir_compile.py, TtirTtnnCompile) runs these three commands
  without `--mlir-disable-threading`, refuses d2m and ttnn ops in the model's module, requires a ttnn op that
  computes (outside get_device, deallocate, to_layout, to_device, from_device, to_memory_config, and the creation
  ops), and requires TTNN at bytes 8 to 12.
- P17.58's first ttsim measurement should cover these four lowered forms as they are: the 64 chained scatters, the
  ui16 sort between two typecasts, the tile-per-element cumsum, and the row-major embedding; it also needs a ttsim
  system descriptor in place of the mock one.
- A ttnn suite's item contracts: si32 (or ui32) at the module's boundary, as the lowered functions keep it; sort keys
  below 65536; gather tables as bf16 columns of 8-bit digits.

Still unmeasured: everything at run time (whether any of these programs runs on ttsim or a card, whether the 64 chained
scatters give histo's counts, the sort's stability, int32 cumsum's exactness), and every shape not probed.

## Sources

- tt-mlir 5f396fd6, https://github.com/tenstorrent/tt-mlir/tree/5f396fd6ef85780bf1b6b85bef9c7d2f84a85f3c, read
  2026-10-29: include/ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.td; lib/Dialect/D2M/IR/D2MGenericRegionOps.cpp;
  lib/Dialect/D2M/Transforms/LowerLoadStoreOpsToDMA.cpp; lib/Conversion/D2MToTTKernel/{D2MToTTKernel.cpp,
  D2MToTTKernelPass.cpp}; lib/Conversion/TTKernelToEmitC/TTKernelToEmitC.cpp; include/ttmlir/Dialect/TTKernel/IR/
  TTKernelOps.td; include/ttmlir/Dialect/TTNN/IR/{TTNNOps.td, TTNNOpsAttrs.td}; lib/Target/TTNN/TTNNToFlatbuffer.cpp;
  lib/Target/TTKernel/TTKernelToCpp.cpp; include/ttmlir/Target/TTNN/operations/mlir_native.fbs;
  lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp; lib/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecompositionPass.cpp;
  lib/Conversion/TTIRToLinalg/TTIRToLinalg.cpp; lib/Dialect/TTNN/IR/TTNNWorkaroundsPass.cpp;
  include/ttmlir/Dialect/TTIR/IR/{TTIROps.td, TTIROpsAttrs.td}; include/ttmlir/Dialect/TTIR/Transforms/Passes.td;
  lib/Dialect/TTIR/Transforms/ElementTypeNormalization.cpp; lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp;
  include/ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h; lib/Dialect/TTMetal/Pipelines/TTMetalPipelines.cpp;
  runtime/lib/ttnn/operations/{data_movement/scatter.cpp, generic/generic_op.cpp};
  test/python/golden/{test_fabric_apis.py, fabric_api_snippets/test_generic_op_unicast.mlir,
  fabric_api_snippets/test_fabric_unicast.mlir}.
- tt-metal 5280a9cf, https://github.com/tenstorrent/tt-metal/tree/5280a9cfb00998fd49667a29523d03aee905c129, read
  2026-10-29 through gh api: ttnn/cpp/ttnn/operations/data_movement/scatter/{scatter.cpp, scatter_enums.hpp,
  device/scatter_device_operation.cpp, device/scatter_program_factory.cpp, device/kernels/dataflow/reader_scatter.cpp,
  device/kernels/scatter_common.hpp}; embedding/device/embedding_device_operation.cpp;
  data_movement/sort/{sort.cpp, device/sort_device_operation.cpp}; data_movement/gather/device/gather_device_operation.cpp;
  moreh/moreh_cumsum/moreh_cumsum.cpp; reduction/accumulation/device/accumulation_device_operation.cpp; the git tree.
- This repository: assets/bench/bale-v0-tt.yaml, assets/bench/bale-v0-ttkernel/histo/kernels/histo_kernel.cpp,
  plans/spikes/ttir-d2m.md.
