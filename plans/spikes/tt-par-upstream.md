# Spike: tt-metal's multi-core programming examples at the pin, read as tt-par-v0 development items (task P17.67)

- Task: P17.67, its first acceptance item (plans/p17-portable.md, "The multi-core C++ -> TT test" and "P17.67");
  OQ-129 (the owner's answers of 2026-11-17: development and held-out items, both variants, upstream examples and
  programs written here).
- Bible: Benchmark Suites (a new suite and its splits), Harness Contract (lassi_io.h; the 2**20-element input cap,
  lassi.harness.contract MAX_INPUT_ELEMENTS), Oracles (binary_io), Execution Backends (ttsim Facts, bible 761; the
  tt_silicon row), Toolchain Pins (Install, bible 575); Agent Rules 1, 2, 3, and 5.
- Status: sources only. No rx command of any kind ran, nothing touched alpha01, and no device or simulator ran. The
  upstream files were read from public GitHub through `gh api` on the workstation (Windows, Git Bash) on 2026-11-17
  and kept in a session scratch directory; nothing downloaded was built or executed. Local readings: the tracked
  manifests (assets/bench/tt-pairs-v0.yaml, assets/train/tt-train-v0/dataset.yaml, assets/rag/tt-metal-docs/corpus.yaml
  and MANIFEST.yaml) and the device log of an earlier owner-started card run already pulled into
  results/tracy-20261116/ (rx 20261116-212311-tracy-0bc3), read as a file.
- Date: 2026-11-17.
- Pin: tt-metal 5280a9cfb00998fd49667a29523d03aee905c129 (toolchains/tt-metal.pin COMMIT). At the pin
  tt_metal/programming_examples is git tree ee668c433a49eb608a61c08a51d7a7bccc4280fa.
- Citations: `ex <path>:<line>` is tt_metal/programming_examples/<path> at the pin; `metal <path>:<line>` is any other
  tt-metal file at the pin; `bible <line>` is docs/BIBLE.md in this worktree.
- Labels: [read] read in the cited source; [inferred] reasoned from what was read; [arith] arithmetic over an example's
  own selection code (a port in a scratch script, not a run); [unknown] nothing read here settles it. The only
  [MEASURED] value is the card-0 core count, quoted from the pulled run with its provenance.
- Conventions: upstream code is paraphrased, as the sibling spikes do, except each example's own result check, quoted
  (one or two lines) with its file and line, since an item's tolerance is that check. Paths in the kernel tables are
  under tt_metal/programming_examples/ at the pin.

## Question

For each multi-core programming example at the pin (vecadd_multi_core, vecadd_sharding, pad_multi_core,
matmul/matmul_multicore_reuse, matmul/matmul_multicore_reuse_mcast, contributed/multicast; NoC_tile_transfer and
shard_data_rm briefly): what it computes and at what sizes, how it picks its cores, how it checks its result, which
kernel files it launches with which arguments, and whether it can be a tt-par-v0 item as tt-pairs-v0.yaml defines one
(a host-only C++ program reproduces its outputs; each input at most 2**20 elements; its kernels run unmodified under a
host program written here that reads and writes through lassi_io.h; its output spans more tiles than the compute grid has cores,
ttsim's grid included, as given or with raised sizes); its standing in tt-train-v0 and tt-metal-docs; and
its relation to the tt-pairs-v0 items. P17.67's manifest, its programs, and its ttsim runs depend on the answer, and
P17.68's held-out items must avoid every computation the chosen development items perform.

## Classification

Factual for every reading below. Which examples become items was the owner's to confirm, since an assigned split
never changes: the owner chose all five, at the sizes recommended below, in the working session of 2026-11-17
(OQ-129). The bible edit (a Benchmark Suites row) waits for the manifest.

## Commands and outputs

1. The examples tree at the pin:

```
gh api "repos/tenstorrent/tt-metal/contents/tt_metal?ref=5280a9cfb00998fd49667a29523d03aee905c129" \
  --jq '.[] | select(.name=="programming_examples") | {name, sha, type}'
-> {"name":"programming_examples","sha":"ee668c433a49eb608a61c08a51d7a7bccc4280fa","type":"dir"}
gh api "repos/tenstorrent/tt-metal/git/trees/ee668c433a49eb608a61c08a51d7a7bccc4280fa?recursive=1" > tree.json
-> truncated: False, 245 entries; 54 blobs under vecadd_multi_core/, vecadd_sharding/, pad_multi_core/, matmul/,
   contributed/multicast/, NoC_tile_transfer/, and shard_data_rm/
```

   matmul/matmul_multicore_reuse/ and matmul/matmul_multicore_reuse_mcast/ hold no kernels directory: both launch the
   kernels of matmul/matmul_common/kernels/ [read].

2. Each of the 54 blobs, as raw bytes, then a check and the digests (a stdlib Python script outside the download
   directory):

```
gh api -H "Accept: application/vnd.github.raw" \
  "repos/tenstorrent/tt-metal/contents/tt_metal/programming_examples/<path>?ref=5280a9cfb00998fd49667a29523d03aee905c129"
# per file: sha1(b"blob %d\0" % len(data) + data) and len(data) against the tree entry; sha256(data)
-> ok 54 bad 0
```

   Every sha256 below is of those exact bytes, each file's git blob id and size having matched the pinned tree.

3. Cross-check of those digests against the two tracked manifests that were built from the pinned tree on alpha01
   (tt-pairs-v0.yaml's kernels, read in rx 20261004-222751-exec-f0dd; tt-metal-docs MANIFEST.yaml):

```
compared 44 equal 44 different 0 not in either manifest 10
```

   The ten not compared are the CMakeLists.txt files.

4. Other files read at the pin the same way: ex CMakeLists.txt, ex contributed/CMakeLists.txt; metal
   tt_metal/common/core_coord.cpp, tt_metal/common/work_split.cpp, tt_metal/api/tt-metalium/work_split.hpp,
   tt_metal/api/tt-metalium/bfloat16.hpp, tt_metal/impl/data_format/bfloat16.cpp, tt_metal/api/tt-metalium/
   kernel_types.hpp, tt_metal/hw/inc/api/compute/reg_api.h, tt_metal/hw/inc/api/dataflow/dataflow_api.h,
   tt_metal/jit_build/build.cpp, tt_stl/tt_stl/assert.hpp,
   tt_metal/core_descriptors/wormhole_b0_80_arch.yaml, tt_metal/llrt/core_descriptor.cpp and .hpp, and
   tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml (sha256 24fd3dfae80435a7d9113e255d6d6af9cdf83f6ae3b1c385e1d48a24795ccf49,
   as plans/spikes/p4-tt-pins.md recorded).

5. The matmul hosts' core selection: bmm_op_utils::get_large_matmul_params and get_core_range (ex
   matmul/matmul_common/bmm_op.hpp:117-221, :223-232) ported line for line to Python and evaluated [arith]:

```
M     N     K    grid  per-core (Mt,Nt)  subblock  blocks (r x c)  cores  out tiles  CB KiB/core  A elems   B elems
640   640   640  8x8   (20,4)            (4,2)     1 x 5           5      400        352          409600    409600
640   640   640  8x7   (20,4)            (4,2)     1 x 5           5      400        352          409600    409600
640   640   640  8x9   (20,4)            (4,2)     1 x 5           5      400        352          409600    409600
3584  3072  768  8x7   (16,12)           (4,2)     7 x 8           56     10752      608          2752512   2359296
3584  3072  256  8x7   (16,12)           (4,2)     7 x 8           56     10752      608          917504    786432
1568  1024  256  8x8   (7,4)             (7,1)     7 x 8           56     1568       144          401408    262144
1568  1024  256  8x7   (7,4)             (7,1)     7 x 8           56     1568       144          401408    262144
1568  1024  256  8x9   (7,4)             (7,1)     7 x 8           56     1568       144          401408    262144
2048  2048  256  8x7   (16,8)            (4,2)     4 x 8           32     4096       448          524288    524288
```

   (3584 x 3072 x 768 and x 256 give the same blocks on 8x8 and 8x9.) A search of every M and N from 64 to 2048 in
   steps of 32 and K in {64, 128, 256, 320, 512, 640}, both inputs at most 2**20 elements and the output over 64
   tiles, found no size whose rule uses more than 56 cores on both 8x8 and 8x7; the smallest output that reaches 56
   with at least two block rows and two block columns is M 1568, N 1024 [arith].

6. The card-0 grid, from the pulled run (local files):

```
results/tracy-20261116/report/profiler/029-reference-matmul_multi_core/.logs/cpp_device_perf_report.csv
  -> DEVICE ARCH wormhole_b0, CORE COUNT 56, AVAILABLE WORKER CORE COUNT 56
results/tracy-20261116/report/profiler/029-reference-matmul_multi_core/.logs/profile_log_device.csv
  -> 56 distinct (core_x, core_y); core_x in {1,2,3,4,6,7,8,9} (8 values), core_y in {1,3,4,5,7,8,9} (7 values)
results/tracy-20261116/provenance.json
  -> id 20261116-212311-tracy-0bc3, commit 5d75218b6d948b13860f4adb9adce5033423198e, dirty False, host alpha01
```

7. The retrieval corpus, per example directory (a scratch script over assets/rag/tt-metal-docs/MANIFEST.yaml; trimmed):

```
vecadd_multi_core      excluded: host and all three kernels ('listed' or 'tier-a-overlap')
vecadd_sharding        indexed: host (11 chunks), kernels/add_sharding.cpp (3); runs excised
pad_multi_core         indexed: host (9), both kernels (4, 1); runs excised; tech_reports/prog_examples/pad_multi_core/pad_multi_core.md indexed (17)
matmul_multicore_reuse excluded: host (tier-a-overlap, 67 of 263 code runs)
matmul_..._mcast       excluded: host (tier-a-overlap, 59 of 379 code runs)
matmul/matmul_common   indexed: bmm_op.hpp, bmm_large_block_zm.cpp, reader_bmm_tile_layout.cpp, the four mcast readers,
                       writer_bmm_tile_layout.cpp; excluded: bmm.cpp, reader_bmm_8bank_output_tiles_partitioned.cpp,
                       writer_unary_interleaved_start_id.cpp
contributed/multicast  indexed: host (11) and all four kernels; tech_reports/prog_examples/multicast/multicast.md indexed (28)
NoC_tile_transfer      indexed: host and all four kernels; its tech report indexed (13)
shard_data_rm          indexed: kernels/reader_sharded_rm.cpp; excluded: host (tier-a-overlap)
docs                   excluded: matmul_multi_core_optimized.rst, data_reuse.rst, data_mcast.rst and their tech_reports copies
```

## The compute grids the counts use

- The plan asks for more output tiles than the grid has cores, ttsim's grid included. At the pin,
  compute_with_storage_grid_size comes from
  metal tt_metal/core_descriptors/wormhole_b0_80_arch.yaml by product name, which metal
  tt_metal/llrt/core_descriptor.hpp:34-41 picks from the number of harvested rows: 0 galaxy, 1 nebula_x1, 2 nebula_x2.
  With row dispatch and one command queue the ranges end at [7,8] (galaxy, 8x9 = 72; :9-13), [7,7] (nebula_x1, 8x8 =
  64; :55-59), and [7,6] (nebula_x2, 8x7 = 56; :149-153) [read].
- alpha01's card 0: 56 cores, 8 columns by 7 rows [MEASURED 2026-11-16: rx 20261116-212311-tracy-0bc3, commit 5d75218
  (clean), tt-metal 5280a9cf-tracy, Wormhole n300 card 0 serial 01000146118320F8, firmware bundle 19.5.0.0, pins and
  driver as results/tracy-20261116/provenance.json records; the tracked matmul_multi_core reference's device log].
  It is the nebula_x2 grid [inferred]: split_work_to_cores uses every grid core when the tiles outnumber them
  (metal tt_metal/common/work_split.cpp:345-347), and that reference splits 400 tiles. The device log's header line
  reads "Max Compute Cores: 64", while the same run's cpp_device_perf_report.csv gives CORE COUNT 56 and AVAILABLE
  WORKER CORE COUNT 56; the 56 is the grid that run used [MEASURED, as above].
- ttsim v1.3.4: its soc descriptor (the pinned tree's wormhole_b0_80_arch.yaml) lists 80 functional workers and no
  harvested ones (metal tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml:110-125), so the core descriptor would pick
  galaxy and 8x9 = 72 [PROJECTED: inferred from the descriptors; no run here shows ttsim's grid; P17.70's spike
  counts cores under ttsim].
- So "more tiles than cores" is checked against 72, the largest of the three grids, and then holds against 64 and 56.
  Every core count below except the one measured 56 above is PROJECTED: arithmetic from an example's code, not a run.

## vecadd_multi_core

1. Computes c = a + b elementwise on bfloat16: 640 tiles (ex vecadd_multi_core/vecadd_multi_core.cpp:108), so a, b,
   and c are 655,360 bf16 values each, written flat to three DRAM interleaved buffers of one tile per page (:123-125),
   never tilized (an elementwise add does not depend on the layout). Inputs: std::mt19937 with seed 0x1234567 (or
   --seed), uniform_real_distribution<float>(0, 10), a and b drawn alternately element by element (:89, :127-134).
2. Cores: compute_with_storage_grid_size (:136-138), three 4-tile Float16_b circular buffers c_0, c_1, c_2 on the
   whole grid (:141-146), and split_work_to_cores(grid, 640, row_major true) (:152-162); each working core gets its
   tile count and first tile id (:225-242). It uses min(640, grid cores) cores: PROJECTED 64 on 8x8, 56 on card 0
   (24 cores with 12 tiles, 32 with 11), and 72 on 8x9 [read; counts arithmetic, no run]. The header comment's "4 cores" (:5-7) is stale [read].
3. Own check (:283-284), then a message; it returns 0 either way (:295-303):
   `float expected = static_cast<float>(a_data[i]) + static_cast<float>(b_data[i]);`
   `if (std::abs(static_cast<float>(c_data[i]) - expected) > 0.3f) {  // Allow some tolerance due to BFP16 precision`
   So: absolute 0.3 against the f32 sum of the bf16 inputs.
4. Kernels (CreateKernel path = upstream path; each path string is split over two source lines):

| CreateKernel (ex vecadd_multi_core.cpp) | Upstream path (under tt_metal/programming_examples/) | Bytes | sha256 |
| --- | --- | --- | --- |
| reader, :195-204, RISCV_0, NOC RISCV_0_default | vecadd_multi_core/kernels/interleaved_tile_read_multi_core.cpp | 3434 | c044be02a23eaf2447e77ff6b37d0023355b6a2a4bed5a2686f24dd64780970f |
| writer, :205-213, RISCV_1, NOC RISCV_1_default | vecadd_multi_core/kernels/tile_write_multi_core.cpp | 1647 | fd273d6022c361740530a2bfbc34664a39676ccd0c04306f484b266fb5a3fc4b |
| compute, :214-223, math_approx_mode false | vecadd_multi_core/kernels/add_multi_core.cpp | 2977 | 0d0c8fb1301dc5e0cc06349d43c9bf0ef1496166b79a6b13e9db99c23599260d |

   - reader: compile args [c_0, c_1, TensorAccessorArgs(a)..., TensorAccessorArgs(b)...] plus named compile args
     {"c_0": 0, "c_1": 1} (:181-185); it reads index 0, the name "c_1", a's accessor at offset 2 and b's after it
     (interleaved_tile_read_multi_core.cpp:19-20, :32-35). Runtime args: a address, b address, tile count, first tile
     (:11-14; host :234).
   - writer: compile args [c_2, TensorAccessorArgs(c)...] (:186-187; kernel :13, :18). Runtime args: c address, tile
     count, first tile (:8-10; host :235).
   - compute: compile args [c_0, c_1, c_2] plus named {"c_0", "c_1", "c_2"} (:188-193); it reads index 0, the name
     "c_1", and index 2 (add_multi_core.cpp:18-21). Runtime args: tile count, first tile (:12-13; host :236).
   - Quirks: the compute loop calls the deprecated release_dst() after tile_regs_release() each tile
     (add_multi_core.cpp:59, :66), which repeats the math and pack "dest section done" steps (metal
     tt_metal/hw/inc/api/compute/reg_api.h:68-96); the JIT builds kernels with -Werror but -Wno-deprecated-declarations
     (metal tt_metal/jit_build/build.cpp:172-176), so it compiles [read]; whether the doubled step is harmless on ttsim
     is [unknown] until the reference runs. tile_write_multi_core.cpp includes only <cstdint> [read].
   - unpack_to_dest: not taken by the host's settings: no fp32_dest_acc_en, no unpack_to_dest_mode, Float16_b buffers
     (ComputeConfig defaults, metal tt_metal/api/tt-metalium/kernel_types.hpp:76-83) [read].
5. Item: yes, at its own sizes. A host-only C++ program computes bf16(float(a) + float(b)); each input holds 655,360
   elements; the kernels need only buffer addresses, tile ranges, and circular-buffer ids, which a host written here
   passes; the output is 640 tiles, more than 64.
6. tt-train-v0: excluded ("duplicate of Tier A eltwise_binary"), not included. tt-metal-docs: the host and all three
   kernels are excluded from the index (the reader as eltwise_binary's read_tiles.cpp renamed, by corpus.yaml's list;
   the others by Tier A overlap) [read].
7. tt-pairs-v0: the same computation as eltwise_binary (an elementwise bf16 add), split over the grid; eltwise_binary
   adds 64 tiles on one core, draws b from [-1, 0), widens its output to f32, and checks 1e-2.

## vecadd_sharding

1. Computes c = a + b elementwise on bfloat16 over 4 x 4 tiles (ex vecadd_sharding/vecadd_sharding.cpp:151-152), 16,384
   values per array, held in L1 buffers sharded over the cores (:186-189), with no reader or writer kernel. Inputs:
   mt19937 seed 0x1234567, uniform [0, 10), a and b alternately (:112, :192-200).
2. Cores: a fixed table, height 4 x 1, width 1 x 4, block 2 x 2, default height (:114-120); the range is
   CoreRange((0,0), (num_cores_y - 1, num_cores_x - 1)) (:169), so the default height layout runs on (0,0)-(3,0),
   four cores along x though the comment says y; its ShardSpecBuffer passes a shard shape of {128, 32} for height
   (:171-183) [read]. Each circular buffer is set to its sharded buffer's address (:57-70, :207-209). 4 cores, 4 tiles
   each [read].
3. Own check (:263-264), then a message; it returns 0 either way (:271-279):
   `float expected = static_cast<float>(a_data[i]) + static_cast<float>(b_data[i]);`
   `if (std::abs(static_cast<float>(c_data[i]) - expected) > 0.2f) {  // Allow some error due to BFP16 precision`
   So: absolute 0.2.
4. Kernel:

| CreateKernel (ex vecadd_sharding.cpp) | Upstream path | Bytes | sha256 |
| --- | --- | --- | --- |
| compute, :212-219, math_approx_mode false | vecadd_sharding/kernels/add_sharding.cpp | 2319 | 9b89bd03d6e8dedec93c8a778f9aa204f56fa3f8d639f350f32fd41ec316b2fd |

   Compile args [c_0, c_1, c_2] (:219; kernel :19-22). Runtime args: tiles per core, one value set on the whole range
   (:227; kernel :24). The kernel never waits on its input buffers and never reserves its output (:12-16, :56-58): it
   relies on the shards being in L1 before it starts and on each circular buffer being exactly one shard [read].
   unpack_to_dest: not taken (defaults, Float16_b) [read].
5. Item: yes, with sizes raised. At 16 tiles its output is below 64 tiles. A host written here can shard over the
   grid instead: the kernel takes only buffer ids and a tile count. Proposed: 448 tiles (458,752 elements per input),
   height-sharded over the 8 x 7 range (0,0)-(7,6), 56 cores of 8 tiles, which fits 8x7, 8x8, and 8x9 grids alike and
   needs 3 x 8 x 2 KiB = 48 KiB of L1 per core [arith]; 448 is the least common multiple of 56 and 64, so the same
   count also splits evenly over 64 cores if the owner prefers the whole grid where it is 8x8.
6. tt-train-v0: excluded (duplicate of eltwise_binary, by judgment alone). tt-metal-docs: host and kernel indexed, with
   Tier A runs excised; corpus.yaml records why both were kept [read].
7. tt-pairs-v0: the same computation as eltwise_binary; the structure differs (sharded L1 buffers, a compute kernel
   only).

## pad_multi_core

1. Pads a row-major bf16 matrix of 64 x 32 (ex pad_multi_core/pad_multi_core.cpp:29-30) to 64 x 64 (:50-51) with the
   value 2.0 (:45) on the right of each row: columns 0-31 copy the row, columns 32-63 hold 2.0 [inferred from the
   reader's loop, pad_reader_dims_rm_interleaved.cpp:72-110, under the host's arguments]. Data move as "sticks" of 4
   bytes, one uint32 packing two bf16 values (:31-35), first value in the low half (metal
   tt_metal/impl/data_format/bfloat16.cpp:92-97), so a flat bf16 row is already in that layout [read]. DRAM buffers
   have 4-byte pages (:63-77). Input: deterministic, element k of the source is bf16(k + 1), k = 0 to 2047 (:36-42),
   rounded to nearest even (bfloat16.cpp:77), so values above 256 are not exact [read]. A one-element pad buffer is
   created and written (:44-47, :70-72, :196) but no kernel reads it: the pad value goes in as a compile arg [read].
2. Cores: fixed CoreRange (0,0)-(0,3), 4 cores (:57-60); 16 rows each, src_M / num_cores with no remainder handling
   (:151-175) [read].
3. Own check: none. It prints the source and padded matrices as raw bf16 bit patterns (:177-209) and ends main without
   a return statement (:211-212) [read]. An item needs a tolerance chosen here; the output is a copy plus a constant,
   so exact equality (max_abs 0) is the natural choice, but it is not upstream's.
4. Kernels ("borrowed from TTNN production code", :130):

| CreateKernel (ex pad_multi_core.cpp) | Upstream path | Bytes | sha256 |
| --- | --- | --- | --- |
| reader, :131-138, RISCV_0, NOC RISCV_0_default | pad_multi_core/kernels/pad_reader_dims_rm_interleaved.cpp | 5216 | 31ad8a0d23a9dca3eebc6da3d4b8c740c23a02e27e4ed711262318773e17482b |
| writer, :139-146, RISCV_1, NOC RISCV_1_default | pad_multi_core/kernels/pad_writer_dims_rm_interleaved.cpp | 1417 | ee711f0765c06c770261681e5cd6b6cacb9cd99023945ad4c366113a5af4903b |

   - reader compile args, in order (host :101-124; kernel :26-56): 0 N (source rows), 1 H (1), 2 C (packed sticks per
     source row), 3 stick bytes (4), 4 N_padded, 5 H_padded (1), 6 C_padded (packed sticks per padded row), 7
     stick_size_padded (64), 8 stick_size_padded_front (0; also read as the front-padding flag), 9
     stick_size_padded_end (0), 10 num_zero_pad_sticks_read (1), 11 last_zero_stick_size (64), 12 not_pad_by_zero (1),
     13 packed pad value, 14 row_major_min_bytes (64), 15 front pad sticks (0), 16 end pad sticks (0), 17 padded
     sticks per stick (1), 18 stick_size_padded_aligned (64), 19 unaligned (0), then TensorAccessorArgs(source) from
     20. Of these, 4 (N_padded), 9, 10, 11, and 14 to 17 are read but never used [read].
   - reader runtime args: source address, sticks per core, sticks per barrier (one padded row), first source stick,
     front pads n, c, h (kernel :18-24); the host passes 11 values, the last four zeros the kernel never reads
     (:157-169) [read]. The reader's row and column counters start at 0 on every core (:72-73), so each core's share
     must start at a row boundary [inferred].
   - writer compile args: [c_0, stick bytes (4), stick_size_padded_aligned (64), TensorAccessorArgs(destination)...]
     (:126-128; kernel :14-17). Runtime args: destination address, sticks per core, sticks per barrier, first
     destination stick (:171; kernel :9-12).
   - One CircularBufferConfig of one padded row of 64-byte slots, shared by c_0, c_1, and c_2 as UInt32 with pages
     64, 64, and 4 bytes (:79-98); the pad pattern written into c_1 lies in that shared space, and the reader seeds the
     first word of every slot with the pad value anyway (reader :65, :82) [read]. No compute kernel, so no
     unpack_to_dest question.
5. Item: yes, with sizes raised and a tolerance chosen here. The C++ program is a copy plus a constant; inputs are far
   under the cap; the kernels take every size as an argument. Proposed: source [512, 128] bf16 padded to [512, 256]
   (65,536 input elements; 131,072 output elements, 128 tiles' worth), the 512 rows split by whole rows over the grid
   with split_work_to_cores (8 rows a core on 64 cores; 9 or 10 on 56; 7 or 8 on 72), pad value 2.0 as upstream, and
   inputs drawn from [-1, 1) so no input equals the pad value [arith]. Each stick is one 4-byte NoC read or write, so
   this size moves 65,536 output sticks against upstream's 2,048 [arith]. Risk [unknown]: 4-byte DRAM pages and
   4-byte NoC transfers under ttsim v1.3.4, which has flagged unaligned accesses before (plans/spikes/
   p4-ttsim-runtime.md, NonContractualBehavior); the reference's ttsim run settles it. The output is row-major, so
   "tiles" here means 1024-element units.
6. tt-train-v0: excluded (rule 3, a `} | } | }` run). tt-metal-docs: host, both kernels, and its tech report indexed,
   with runs excised; the writer is in the similarity report at 0.701 [read].
7. tt-pairs-v0: no item pads; the nearest is loopback (a DRAM copy through L1). A pad is a new computation for the
   development set, and none of P17.68's five held-out computations pads.

## matmul/matmul_multicore_reuse

1. Computes C = A B on bfloat16 at M = N = K = 640, batch 1 (ex matmul/matmul_multicore_reuse/matmul_multicore_reuse.cpp:
   360-363). Inputs: create_random_vector_of_bfloat16_native, A uniform [0, 1) - 0.4 with seed 123, B uniform [0, 1) -
   0.3 with seed 12522 (:375-376; metal tt_metal/impl/data_format/bfloat16.cpp:99-109, mt19937). A CPU golden in f32
   accumulation rounded to bf16 (:26-57, :380); inputs tilized before upload and the result untilized (:383-389).
2. Cores: compute_with_storage_grid_size (:83-85); in0_block_w 2 (:98); per-core block and subblock from
   get_large_matmul_params (:105-109; bmm_op.hpp:117-221), which takes the largest per-core block its rule allows;
   num_cores_to_corerangeset over the blocks, row-wise (:171-176); block b on core (b mod num_cores_x, b div
   num_cores_x) (:269-275). At 640: per-core 20 x 4 tiles, 1 x 5 blocks, 5 cores on 8x7, 8x8, and 8x9 [arith,
   command 5]. Its divisibility checks are TT_ASSERT (:119-121, :174), which compile to nothing outside DEBUG
   (metal tt_stl/tt_stl/assert.hpp:155-168) [read].
3. Own check (:393-395); an exception propagates, else "Test Passed" (:404-408):
   `float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);`
   `TT_FATAL(pearson > 0.99, "PCC not high enough. Result PCC: {}, Expected PCC: 0.99", pearson);`
   So: PCC above 0.99 against the golden (check_bfloat16_vector_pcc, bmm_op.hpp:74-108); binary_io passes at or above
   its threshold, as tt-pairs-v0 notes for 0.97.
4. Kernels: all under matmul/matmul_common/kernels/. CreateKernel names them as OVERRIDE_KERNEL_PREFIX
   "matmul/matmul_common/kernels/..." (:243, :252, :262); the examples' CMake sets the prefix to
   "tt_metal/programming_examples/" (ex CMakeLists.txt:8), so in-tree the path is the upstream path [read].

| CreateKernel (ex matmul_multicore_reuse.cpp) | Upstream path | Bytes | sha256 |
| --- | --- | --- | --- |
| reader, :241-248, RISCV_1, NOC RISCV_1_default | matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout.cpp | 4349 | 8bc4f61e7780d664ed7825757fdfdc440f05fe50c8060cdfb809803b681267ef |
| writer, :250-257, RISCV_0, NOC RISCV_0_default | matmul/matmul_common/kernels/dataflow/writer_bmm_tile_layout.cpp | 3357 | e17aec12dd837bba1d91e66ec455611df745ff07c897f6172acd821329c1b600 |
| compute, :260-264, math_fidelity HiFi4 | matmul/matmul_common/kernels/compute/bmm_large_block_zm.cpp | 5566 | 8c59140dbaf49e584af6fd1ea443d00be684f3a708bfb0fd8a7ff735539aec30 |

   The host also includes matmul/matmul_common/bmm_op.hpp (7084 bytes, sha256
   0b924f53a37fd90be5ec858165fa2c8d66199123e3b704847dce6db38e9d00f4) for the block rule and the PCC [read].
   - reader compile args: TensorAccessorArgs(A)..., TensorAccessorArgs(B)... (:230-232; kernel :51-54). Runtime args,
     21 (:277-304; kernel :10-40): A address, A first tile (Kt x per_core_M x block row), A stride w (1), A stride h
     (Kt), A next-block stride (in0_block_w), in0_block_w, in0 block h (per_core_M), in0 block tiles; B address, B
     first tile (per_core_N x block column), B stride w (1), B stride h (Nt), B next-block stride (in0_block_w x Nt),
     in1 block w (per_core_N), in1 block h (in0_block_w), in1 block tiles; num_blocks (Kt / in0_block_w); MtKt, KtNt,
     batch, bcast_B. It writes circular buffers 0 and 1 by number (:42-43).
   - writer compile args: TensorAccessorArgs(C)... (:234-235; kernel :32). Runtime args, 13 (:306-323; kernel :9-25):
     C address, C first tile, stride w (1), stride h (Nt), next subblock stride w (out_subblock_w), next subblock
     stride h (out_subblock_h x Nt), out_subblock_w, out_subblock_h, subblock tiles, subblocks across, subblocks down,
     MtNt, batch. It reads circular buffer 16 (:27).
   - compute compile args, 12 (:146-162; kernel :11-22): in0_block_w, in0_num_subblocks, in0_block_num_tiles,
     in0_subblock_num_tiles, in1_num_subblocks, in1_block_num_tiles, in1_per_core_w, num_blocks, out_subblock_h,
     out_subblock_w, out_subblock_num_tiles, batch. No runtime args. It uses c_0, c_1, c_16, and c_24 by number; with
     more than one K block it spills partial sums to c_24 in Float16_b and reloads them with copy_tile (:42-50,
     :76-88), so partial sums round to bf16 between blocks [read].
   - Circular buffers: c_0 of 2 x per_core_M x in0_block_w tiles, c_1 of 2 x per_core_N x in0_block_w, and c_16 with
     c_24 in one shared config of per_core_M x per_core_N tiles, all Float16_b (:208-225) [read]. unpack_to_dest: not
     taken by the host's settings (math_fidelity only, Float16_b) [read].
5. Item: yes. At 640 its output is 400 tiles, more than 64, with inputs of 409,600 elements; but its own rule puts the
   work on 5 cores, so as a reference it is barely multi-core. Raised as proposed for the mcast item, M 1568, N 1024,
   K 256 (Mt 49, Nt 32, Kt 8, num_blocks 4), the same rule gives 7 x 8 blocks of 7 x 4 tiles, 56 cores on 8x7, 8x8, and
   8x9, 144 KiB of circular buffers a core, inputs of 401,408 and 262,144 elements, and 1,568 output tiles [arith].
   The host written here cannot track bmm_op.hpp's text; it either computes the blocks with code of its own (and the
   manifest records that they equal upstream's rule at the item's size) or names bmm_op.hpp as a support file by path
   and sha256, as the kernels are named.
6. tt-train-v0: excluded (duplicate of Tier A matmul_multi_core). tt-metal-docs: the host is excluded (Tier A overlap),
   its kernels and bmm_op.hpp are indexed, and the data_reuse walkthrough and its copy are excluded [read]. LASSI's
   pinned install builds metal_example_matmul_multicore_reuse (toolchains/tt-metal.pin EXAMPLES) and never runs it;
   upstream's ttsim CI job lists it but skipped it at the pin (bible 761) [read].
7. tt-pairs-v0: the same computation as matmul_multi_core (bf16 matmul, PCC against a CPU golden; 0.97 there, 0.99
   here), with K blocking, input reuse in L1, and subblocked output instead of one output tile at a time.

## matmul/matmul_multicore_reuse_mcast

1. Computes C = A B on bfloat16 at M 3584, N 3072, K 768 (ex matmul/matmul_multicore_reuse_mcast/
   matmul_multicore_reuse_mcast.cpp:483-486), inputs as the reuse example's (same generator, seeds 123 and 12522,
   offsets -0.4 and -0.3; :498-499), the same golden (:22-53), tilized and untilized the same way (:506-512).
2. Cores: the same block rule (:79-81, :104-109); get_core_range (bmm_op.hpp:223-232) gives a rectangle of block rows
   by block columns from (0,0) (:166-180). Core (0,0) reads both inputs and multicasts A blocks along its row and B
   blocks down its column; the rest of column 0 send A and receive B; the rest of row 0 receive A and send B; the others
   receive both (:182-203, :344-455), with four semaphores on every core, initial value INVALID (:334-337). At its
   defaults: 7 x 8 blocks of 16 x 12 tiles, 56 cores, 608 KiB of circular buffers a core [arith, command 5]. With one
   block row or column it cannot run: the ranges for the other roles become CoreRange((0,1),(0,0)) and the like, which
   the CoreRange constructor refuses with TT_FATAL (metal tt_metal/common/core_coord.cpp:38-43); at 640^3 the blocks
   are 1 x 5, so the example aborts at that size [inferred from the code].
3. Own check (:516-518):
   `float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);`
   `TT_FATAL(pearson > 0.98, "PCC not high enough. Result PCC: {}, Expected PCC: 0.98", pearson);`
   So: PCC above 0.98.
4. Kernels: all under matmul/matmul_common/kernels/. Every CreateKernel path names
   "tt_metal/programming_examples/matmul_common/kernels/..." (:272, :281-282, :291-292, :301-302, :311, :320, :330),
   a directory that does not exist at the pin (the tree has matmul/matmul_common/), and its include is
   <matmul_common/bmm_op.hpp> (:13) where the reuse example has <bmm_op.hpp>. Neither ex matmul/CMakeLists.txt (:4-8)
   nor any other CMake file at the pin builds it; the directory holds only the .cpp [read]. So the example as written
   would not find its kernels at the pin, and nothing on record shows it running at the pin [inferred]. A host
   written here names the kernels at their real paths, so this does not block an item.

| CreateKernel (ex matmul_multicore_reuse_mcast.cpp) | Upstream path (matmul/matmul_common/kernels/...) | Bytes | sha256 |
| --- | --- | --- | --- |
| :270-277, core (0,0), RISCV_1, NOC RISCV_0_default | dataflow/reader_bmm_tile_layout_in0_sender_in1_sender.cpp | 12159 | 31dd377003eb2bc302bd508a9603591e28838f8d6d0a0e4fc28581c0ba50cebd |
| :279-287, column 0 below row 0, RISCV_1, NOC RISCV_0_default | dataflow/reader_bmm_tile_layout_in0_sender_in1_receiver.cpp | 8409 | c6d1e619e3c792806183c50b2c50394889e1574bf8c260e52806325ee310cd07 |
| :289-297, row 0 right of column 0, RISCV_1, NOC RISCV_1_default | dataflow/reader_bmm_tile_layout_in0_receiver_in1_sender.cpp | 8621 | 793023368a44508b537c7c119ccc00928fdb2e8ff8ec30249c31b4a01d126988 |
| :299-307, the rest, RISCV_1, NOC RISCV_1_default | dataflow/reader_bmm_tile_layout_in0_receiver_in1_receiver.cpp | 4906 | c893a77b3566f5af187c55f9afffe7c5c9ecce442ad51c7a5470053859df4ca0 |
| :309-316, all but column 0, RISCV_0, NOC RISCV_0_default; and :318-325, column 0, RISCV_0, NOC RISCV_1_default | dataflow/writer_bmm_tile_layout.cpp | 3357 | e17aec12dd837bba1d91e66ec455611df745ff07c897f6172acd821329c1b600 |
| :328-332, all cores, math_fidelity HiFi4 | compute/bmm_large_block_zm.cpp | 5566 | 8c59140dbaf49e584af6fd1ea443d00be684f3a708bfb0fd8a7ff735539aec30 |

   - Every reader gets compile args TensorAccessorArgs(A)..., TensorAccessorArgs(B)... (:258-260); the sender-sender
     and receiver-sender readers declare both accessors, the sender-receiver reader only A's, and the
     receiver-receiver reader none [read]. Runtime args, 39, the same vector for each role (:362-409; each reader
     :11-63): the reuse reader's first 17 (0-16); then for A's multicast the destination rectangle's NOC start x, y
     (the row's rightmost core), end x, y (the core right of column 0), the destination count (columns - 1), the
     sender's NOC x, y (column 0 of the row), and the sender and receiver semaphore ids; then the same nine for B's
     multicast down the column (start at the column's bottom core, end at the core below row 0, count rows - 1, sender
     at row 0); then MtKt, KtNt, batch, bcast_B. NOC coordinates come from worker_core_from_logical_core (:355-360).
   - writer: compile args TensorAccessorArgs(C)... (:262-263); runtime args the reuse writer's 13 (:411-427).
   - compute: the reuse example's 12 compile args (:145-161). Circular buffers as there (:235-252).
   - unpack_to_dest: not taken by the host's settings, as the reuse example [read].
5. Item: yes, with sizes changed: its defaults hold 2,752,512 and 2,359,296 input elements, over the cap. Two choices
   that keep 7 x 8 = 56 cores [arith]: (a) M 1568, N 1024, K 256: per-core 7 x 4 tiles, subblock 7 x 1, 144 KiB a core,
   inputs 401,408 and 262,144 elements, 1,568 output tiles, 12,544 tile products (Mt Nt Kt); (b) upstream's M 3584 and
   N 3072 with K cut to 256: upstream's own per-core blocks (16 x 12, subblock 4 x 2), inputs 917,504 and 786,432,
   10,752 output tiles (11,010,048 values), 86,016 tile products. For comparison tt-pairs-v0 matmul_multi_core does
   8,000 tile products. Both keep at least two block rows and columns on 8x7, 8x8, and 8x9. Risk [unknown]: multicast
   writes and semaphores under ttsim v1.3.4 have not run in this project; the reference's ttsim run settles it.
6. tt-train-v0: excluded (duplicate of matmul_multi_core). tt-metal-docs: the host is excluded (Tier A overlap); the
   four readers, the writer, the compute kernel, and bmm_op.hpp are indexed; the data_mcast walkthrough and its copy
   are excluded [read].
7. tt-pairs-v0: the same computation as matmul_multi_core, with A and B blocks multicast along rows and columns.

## contributed/multicast

1. Copies one 32 x 32 bf16 tile, an identity matrix (create_identity_matrix, metal bfloat16.cpp:150-156; ex
   contributed/multicast/multicast.cpp:242-243), from DRAM to core (0,0), which multicasts it to cores (1,0)-(3,0);
   each receiver writes its copy to its own tile of a three-tile DRAM output (:194-199, :271-276). Input 1,024 values;
   output 3 tiles.
2. Cores: fixed, sender (0,0) and receivers (1,0)-(3,0) (:173-187), two semaphores on all four (:190-191); 4 cores.
3. Own check (verify_tiles, :132-133 in its loop :128-138; called :308), a PASS or FAIL line per tile, return 0
   either way (:312):
   `float golden = static_cast<float>(golden_tile[(i * TILE_WIDTH) + j]);`
   `if (received != golden) {`
   So: exact equality of every received tile with the input tile.
4. Kernels (CreateKernel path = upstream path):

| CreateKernel (ex multicast.cpp) | Upstream path (contributed/multicast/kernels/...) | Bytes | sha256 |
| --- | --- | --- | --- |
| coordinator, :211-215, core (0,0), RISCV_0, NOC RISCV_0_default | dataflow/coordinator_kernel.cpp | 3364 | 841a057cd0cea271b7b4d8cae86c77b649ca5bde8f0033a66c9bec8e43ee313b |
| inbound, :218-222, receivers, RISCV_0, NOC RISCV_0_default | dataflow/inbound_kernel.cpp | 3286 | d6bbfa96dcc5445f1e10202ae57377d7e8a08295b0a7bafdc63b0675bdad8694 |
| outbound, :223-227, receivers, RISCV_1, NOC RISCV_1_default | dataflow/outbound_kernel.cpp | 1181 | 56faba1292e8d9b1fc10ad7b9477f588d24cab6b8ab81be421b6d3fa0c1045cd |
| compute, :231-239, receivers, HiFi4, fp32_dest_acc_en false | compute/void_compute_kernel.cpp | 531 | 6bfa632aef9fbf6418b0720d6df00c65aa9ac08f4ba255dd0346d39eede8069b |

   coordinator: no compile args; runtime args, 10 (:248-261; kernel :12-21): receiver rectangle NOC start x, y and end
   x, y, sender and receiver semaphore ids, DRAM bank id (0), source address, tile bytes (2048), destination count; it
   reads the tile by bank id (:24, :31). inbound: no compile args; runtime args sender NOC x, y and the two semaphore
   ids (:264-268; kernel :21-24). outbound: compile args TensorAccessorArgs(output)... (:203-208; kernel :16); runtime
   args output address and tile index (:273-275; kernel :10-11). compute: none; it only prints. Circular buffers c_0
   and c_16, one tile each (:198-199).
5. Item: no. Each receiver writes exactly one tile, so with the kernels unmodified the output has as many tiles as
   receivers. At the pin the sender may sit inside the multicast rectangle, which then skips it (metal
   tt_metal/hw/inc/api/dataflow/dataflow_api.h:900-907 for the data, :1546-1553 for the semaphore), so a host written
   here could place receivers on every core but the sender: PROJECTED 63 tiles on 8x8, 55 on card 0's 8x7, 71 on
   8x9. The
   output is always one tile fewer than the grid has cores, so no size makes it span more tiles than cores, and on an
   n300 chip never more than 64 [inferred]. Its computation, one tile replicated, is also loopback's copy in another
   shape, and as a development item it would sit close to P17.68's held-out col_scale, which broadcasts a row.
6. tt-train-v0: excluded (rule 3). tt-metal-docs: host, four kernels, and tech_reports/prog_examples/multicast/
   multicast.md indexed, with runs excised [read].
7. tt-pairs-v0: nearest loopback (a copy); no item multicasts.

## NoC_tile_transfer and shard_data_rm

- NoC_tile_transfer moves one 2048-byte tile of uint16 values DRAM -> core (0,0) -> core (0,1) over the NoC -> DRAM,
  with one semaphore handshake, four kernels on two fixed cores (ex NoC_tile_transfer/noc_tile_transfer.cpp:30-35,
  :77-110), and only prints the first element beside its expected value (:121), no check. One output tile on two
  cores, kernels written for exactly that: it adds nothing as a multi-core item and cannot be raised past one tile
  with its kernels unmodified [read; inferred]. tt-train-v0 excludes it as a loopback duplicate; tt-metal-docs indexes
  it.
- shard_data_rm cannot be an item: it is in tt-train-v0's include list (an included example is training data), and it
  writes no output buffer; its kernel prints each core's shard through DPRINT (ex shard_data_rm/kernels/
  reader_sharded_rm.cpp:24-39) [read].

## Summary table

Cores are at the example's own sizes and are all PROJECTED (arithmetic from each example's code; none of these
examples has run here; the one measured count is tt-pairs-v0 matmul_multi_core's 56 on card 0): 8x8 is 64 cores,
card 0 is 8x7 (56), ttsim is 8x9 (72, inferred). Inputs are the largest input's element count.

| Example | Computes | Default size (largest input) | Cores, PROJECTED (8x8 / card 0 / ttsim) | Own check | Out tiles > 72 | Inputs <= 2**20 | tt-train-v0 | tt-metal-docs | tt-pairs-v0 relation | Item |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vecadd_multi_core | a + b, bf16 | 640 tiles (655,360) | 64 / 56 / 72 | abs 0.3 | 640: yes | yes | excluded | host and kernels excluded | eltwise_binary's computation | yes, as is |
| vecadd_sharding | a + b, bf16, sharded L1 | 16 tiles (16,384) | 4, fixed | abs 0.2 | 16: no; 448 proposed | yes | excluded | indexed | eltwise_binary's computation | yes, raised |
| pad_multi_core | right pad, row-major bf16 | 64x32 -> 64x64 (2,048) | 4, fixed | none (prints) | 4: no; 128 proposed | yes | excluded | indexed | none (nearest loopback) | yes, raised; tolerance chosen here |
| matmul_multicore_reuse | C = A B, bf16 | 640^3 (409,600) | 5 / 5 / 5 | PCC > 0.99 | 400: yes | yes | excluded | host excluded, kernels indexed | matmul_multi_core's computation | yes; raise to use 56 cores |
| matmul_multicore_reuse_mcast | C = A B, bf16, multicast | 3584x3072x768 (2,752,512) | 56 / 56 / 56 | PCC > 0.98 | 10,752: yes | no; K 256 or smaller M, N fit | excluded | host excluded, kernels indexed | matmul_multi_core's computation | yes, resized |
| contributed/multicast | one tile replicated | 1 tile in, 3 out (1,024) | 4, fixed | exact | 3: no; at most grid cores - 1 | yes | excluded | indexed | nearest loopback | no |
| NoC_tile_transfer | one tile, core to core | 1 tile (1,024) | 2, fixed | none (prints) | 1: no | yes | excluded | indexed | loopback's computation | no |
| shard_data_rm | shard and print | 16 x 1 | 4, fixed | none (DPRINT) | no output | yes | included | kernel indexed, host excluded | none | no (included in training) |

## Recommendation: the development items of tt-par-v0

Five items, each with a host-only C++ program and a TT host program written here that runs the upstream kernels
unmodified, named by path and sha256 (the table rows above), with no tt-metal text tracked:

1. vecadd_multi_core at its own size: a and b [655360] bf16, uniform [0, 10) as upstream draws them; c bf16; max_abs
   0.3 (its check); split_work_to_cores over the grid (64 on 8x8, 56 on card 0). It is the plainest multi-core host
   (grid, split, per-core start tile), the pattern P17.71 and P17.72 teach.
2. vecadd_sharding raised to 448 tiles (a, b [458752] bf16, uniform [0, 10)), height-sharded over the 8 x 7 range,
   56 cores of 8 tiles; max_abs 0.2 (its check). It covers the sharded-L1, compute-only host, which no other item has.
3. pad_multi_core raised to source [512, 128] -> [512, 256], pad 2.0, inputs uniform [-1, 1), rows split over the
   grid; max_abs 0, chosen here since upstream has no check. It is the only data-movement-only item, with a 20-argument
   reader, a hard host for a model to write from the kernel source alone.
4. matmul_multicore_reuse at M 1568, N 1024, K 256: A uniform [-0.4, 0.6), B uniform [-0.3, 0.7) as upstream; PCC
   0.99 (its check); 56 cores under its own block rule. At upstream's 640^3 it would be valid but use 5 cores.
5. matmul_multicore_reuse_mcast at the same M 1568, N 1024, K 256 (choice (a) above), same inputs; PCC 0.98 (its
   check); 56 cores in upstream's 7 x 8 sender and receiver layout. Same output and grid as item 4, so the pair
   isolates the dataflow (per-core reads against row and column multicast). Choice (b), upstream's M and N with K 256,
   keeps upstream's exact blocks at 6.9 times the output.

Not items: contributed/multicast (one output tile per receiver, so always fewer tiles than cores), NoC_tile_transfer
(one tile, two fixed cores), shard_data_rm (in tt-train-v0, and no output). P17.68's held-out items then avoid
elementwise add, matmul, and pad, as planned; leaving multicast out keeps a row broadcast open for them.

Each item's reference core count depends on the grid for items 1 and 3 (the whole grid) and not for 2, 4, and 5 (56
everywhere from 8x7 up). The manifest should record the count with the grid it was counted on (card 0, or ttsim once
P17.70 counts it), not one bare number.

Analytic work at these sizes, for the manifest [arith; properties of the items, never measurements]: item 1, 655,360
adds and 3,932,160 bytes (three bf16 arrays); item 2, 458,752 adds and 2,752,512 bytes; item 3, no arithmetic, 393,216
bytes (65,536 bf16 read and 131,072 written); items 4 and 5, 822,083,584 FLOPs (2MNK; 411,041,792 MACs) and 4,538,368
bytes ((401,408 + 262,144 + 1,605,632) bf16).

Before the manifest commit: none of these kernels has run in this project, and upstream's own ttsim CI ran none of
these five at the pin (its list, plans/spikes/p4-tt-pins.md:125, names only matmul_multicore_reuse of them, which it
skipped; bible 761). P17.67 requires each TT reference to pass against its C++ program on ttsim; an item whose
reference fails there (the doubled release_dst of item 1, the 4-byte transfers of item 3, and the multicast and
semaphores of item 5 are the likeliest places) goes to the owner queue with the run as evidence, since dropping it
changes the owner's choice (OQ-129).

## Consequences for the plan

- P17.67's manifest: five items as above, sizes changed for 2, 3, 4, and 5; pad's tolerance and the mcast size choice
  were the owner's (answered below). The kernels' support paths can be tails holding a kernels directory, as in
  tt-pairs-v0: vecadd_multi_core/kernels/..., vecadd_sharding/kernels/add_sharding.cpp, pad_multi_core/kernels/...,
  and matmul/matmul_common/kernels/... (the tail the reuse example itself names). tools/fetch_bench.py copies from the
  pinned tree built in place, which holds every file here, though the examples' install rules list none of
  vecadd_multi_core, vecadd_sharding, or pad_multi_core (ex CMakeLists.txt:41-90) [read].
- The tt-metal-docs corpus holds parts of these examples:
  vecadd_sharding, pad_multi_core, contributed/multicast, and the matmul_common kernels with bmm_op.hpp; the corpus
  excludes vecadd_multi_core's host and kernels, the reuse and mcast hosts, and their walkthroughs as near copies of
  Tier A code (command 7). It does not change their split: development items may steer prompt work either way.
- tt-train-v0 needs no change now: every recommended example is already excluded. Its spec names only tt-pairs-v0
  (suite:); a later rebuild should also refuse tt-par-v0's development items, which the registry refuses to training
  once P17.31's split exists.
- P17.70's ttsim core count, PROJECTED: items 2, 4, and 5 should read 56 under ttsim if its grid is 8x9 as inferred;
  items 1 and 3 would read 72 [inferred]. A different reading tests the inference.
- P17.68: the owner chose row_sum, col_max, transpose_2d, row_softmax, and col_scale (OQ-129); none is an elementwise
  add, a matmul, or a pad.

## The owner's answer

Asked in the working session of 2026-11-17, with these options: (a) all five, sized as recommended above; (b) as
(a), but mcast at upstream's M 3584 and N 3072 with K 256; (c) three, one per computation (vecadd_multi_core,
pad_multi_core, matmul_multicore_reuse); (d) as (a) without pad_multi_core. The owner chose (a), "All five,
resized", recorded in plans/OWNER-QUEUE.md OQ-129, so no separate item is filed.

Once the manifest exists, the bible gains a Benchmark Suites row beside tt-pairs-v0 (a draft, to be completed with
P17.68's held-out items and the splits as P17.31 names them):

```
| tt-par-v0 | Multi-core C++ -> TT items: development items from tt-metal's multi-core programming examples at the pin (vecadd_multi_core, vecadd_sharding, pad_multi_core, matmul_multicore_reuse, matmul_multicore_reuse_mcast), the upstream kernels unmodified with TT host programs and C++ counterparts written here, named by path and sha256 only; held-out items written here (task P17.68) (tasks P17.67, P17.68) | Hand-written | Development items (OQ-129), held-out items eval |
```

with a Decision Log entry dated on the commit, naming OQ-129 and this spike, and the count sentence raised.

## Sources

- tenstorrent/tt-metal at 5280a9cfb00998fd49667a29523d03aee905c129, read 2026-11-17 through the GitHub contents and
  git trees APIs: https://github.com/tenstorrent/tt-metal/tree/5280a9cfb00998fd49667a29523d03aee905c129/tt_metal/programming_examples
  (vecadd_multi_core, vecadd_sharding, pad_multi_core, matmul (matmul_common, matmul_multi_core, matmul_single_core,
  matmul_multicore_reuse, matmul_multicore_reuse_mcast), contributed (CMakeLists.txt, multicast), NoC_tile_transfer,
  shard_data_rm, CMakeLists.txt); tt_metal/common/core_coord.cpp, work_split.cpp; tt_metal/api/tt-metalium/
  work_split.hpp, bfloat16.hpp, kernel_types.hpp; tt_metal/impl/data_format/bfloat16.cpp; tt_metal/hw/inc/api/compute/
  reg_api.h; tt_metal/hw/inc/api/dataflow/dataflow_api.h; tt_metal/jit_build/build.cpp; tt_stl/tt_stl/assert.hpp;
  tt_metal/core_descriptors/wormhole_b0_80_arch.yaml; tt_metal/llrt/core_descriptor.cpp and .hpp;
  tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml.
- This repository: assets/bench/tt-pairs-v0.yaml; assets/train/tt-train-v0/dataset.yaml; assets/rag/tt-metal-docs/
  corpus.yaml and MANIFEST.yaml; lassi/harness/contract.py (MAX_INPUT_ELEMENTS); lassi/harness/lassi_io.py (input
  dists); assets/harness/c/lassi_io.h; tools/fetch_bench.py; toolchains/tt-metal.pin; docs/BIBLE.md 575 and 761;
  plans/spikes/p4-tt-pins.md, p4-ttsim-runtime.md; plans/OWNER-QUEUE.md OQ-129.
- results/tracy-20261116/ (the pull of rx 20261116-212311-tracy-0bc3, commit 5d75218, on alpha01; provenance.json,
  report/provenance.json, and report/profiler/029-reference-matmul_multi_core/.logs/), read locally, untracked in this
  worktree when read.
