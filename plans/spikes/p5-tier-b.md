# Spike P5.3: Tier B sources at the joint pin

- Task: P5.3 (plans/p5-ir.md, "### P5.3"). Bible: Benchmark Suites (the Tier B row, TT tiers, Kernels in a prompt, Manifest keys, Item tolerance), Toolchain Pins (joint pin; the TurboQuant checkout), ttsim Facts, Execution Backends (ttsim row), Harness Contract; Agent Rules 2, 5, 9, 12, 15, and 17.
- Status: DRAFT. The local half is done: the GitHub reading, the tt-metal text and API comparisons, the batch, and its dry run. The host half waits for the main session: one read-only search and one batch, which OQ-050 approved as it describes them, and the parts of the batch its wording does not name, which wait for the owner's yes (Owner queue). Every host result below is marked PLACEHOLDER.
- Date: 2026-10-10. GitHub and tt-metal files were read that evening (EDT) with `gh api` and shallow, blobless, sparse git checkouts in the session scratchpad.
- Base: branch p5-ir at 382b700. Files: this report and plans/spikes/p5-tier-b/ (run.sh, tier_b.py, dryrun.sh, dry_standin.py).
- Remote activity: none. No rx command of any kind ran (Agent Rule 17).
- Device: none opened anywhere. Every value below is from reading source or from local checks on the workstation. The dry run used stand-ins, not tt-metal or ttsim.
- Output conventions: no tt-metal or codec source text appears here (OQ-018 practice). Code is cited by file and line and paraphrased; only identifiers, file names, hashes, and numbers are quoted. The codec README's figures are the codec's own claims about its silicon runs, never LASSI measurements.

## Question

Which sources does Tier B of tt-pairs-v0 come from? For each stage the bible's Tier B row names (WHT, PolarQuant, QJL, dequant), which baseline and multi_tile kernels, host programs, and reference functions implement it, and what does the repository hold beyond those stages? Does each kernel build and run at the joint pin (tt-metal 5280a9cf, ttsim v1.3.4), and does its output match the codec's NumPy reference? Does any kernel hold tt-metal text, and what is the license state?

Why it matters: OQ-048 asks the owner for Tier B's source pin, items, holding, and the rule for kernels that fail at the pin. P5.13 builds Tier B from the answer, and P5.14 and P5.G round-trip it. The bible records only that the kernels were written against tt-metal 91f5a51 with local silicon-mode edits and are unverified at the pin (Toolchain Pins; plans/spikes/p4-tt-pins.md, The TurboQuant checkout).

## Classification

Factual: the sources, the stage map, the text overlap, the API distance, and whether each program builds and runs on ttsim can all be answered by reading and running. The rest is OQ-048's choice: which items, how the kernels are held, and what happens to a kernel that fails. This spike supplies evidence and a draft Response for OQ-048, and files no new decision.

## Method

1. Codec: `gh api` reads of the repository, its branches, its commits on main, the compare of each branch with main, the tree at the head, and its contributors. A full clone (0.6 MB) went into the session scratchpad, where it was read as untrusted data. Nothing from it ran apart from importing its reference module for the local checks.
2. tt-metal at 91f5a51 (the TurboQuant checkout's commit) and at 5280a9cf (the pin): shallow, blobless, sparse checkouts at depth 1 of tt_metal/api, tt_metal/hw/inc, tt_metal/hostdevcommon, tt_metal/kernels, tt_metal/programming_examples, tt_metal/impl/kernels, tt_metal/jit_build, and tt_stl/tt_stl. Separate checkouts held ttnn's operation kernels (ttnn/cpp/ttnn/operations/**/kernels and ttnn/cpp/ttnn/kernel, 877 files at the pin and 909 at 91f5a51). Single files came from the contents API: tt_metal/impl/data_format/tilize_utils.cpp at both commits, and tt_metal/distributed/mesh_device.cpp at the pin. tt-llk's compare between the two submodule commits came from the compare API. tt-metal was never cloned whole.
3. Text overlap: a scratch script normalized each source (comments, include lines, and blank or brace-only lines dropped; whitespace collapsed). For each codec file it counted the normalized lines that occur in any tt-metal file, found the longest run of consecutive lines shared with one tt-metal file, and matched comment lines of 30 characters or more.
4. API distance: a scratch script listed the declarations of every tt-metal name the codec uses, at both commits, with file and line. Each header that differs between the commits was diffed.
5. Local checks with the codec's reference under NumPy 2.2.6 (`uv run --no-project --with numpy==2.2.6`, workstation): Facts checked locally, below.
6. The batch (plans/spikes/p5-tier-b/run.sh and tier_b.py) and its dry run in WSL Ubuntu 24.04 with stand-ins (Dry run, below). The read-only search's command text was checked with the gate's own check_command and run on a stand-in tree (Commands, step 2).

## The codec's sources on GitHub

- Repository JoeMad21/TurboQuant-Tenstorrent: public, not a fork, created 2026-05-13, default branch main.
- main is at 81817eb09076d17b29491c6646b54c6927cac0bd (2026-05-26 17:50:46Z, "Update README.md"), the commit OQ-048 names, so it has not moved since planning. main has 10 commits, from 2026-05-13 to 2026-05-26, all authored by the GitHub account JoeMad21, which is also this repository's commit identity. The repository's pushed_at of 2026-07-27 comes from the other branch.
- Branch john-changes is at b78894f1ece9e3e5c0372919fb77f0bf1a436269: 2 commits ahead of main and 0 behind, by the account broncosfan915 (2026-06-22 "First Multi Device Change (In progress)" and 2026-07-27 "Multi-device implementation"). It changes only host/turboquant_host.cpp (+374/-206) and host/turboquant_launcher.cpp (+40/-14). There the baseline host opens its mesh through MeshDevice::create with a MeshDeviceConfig, taking a 1x1 shape or every device per a new --num-devices flag, instead of a fixed chip id. No kernel changes, so every kernel's blob is the same at 81817eb and b78894f.
- The tree at 81817eb has 68 files, 520,034 bytes, as counted by tier_b.py prepare in the dry run's real fetch:
  - baseline: kernels/ (rotation_kernel.cpp, polarquant_kernel.cpp, qjl_kernel.cpp, dequant_attn_kernel.cpp, dataflow/{reader_kernel, writer_kernel, stage_dump_writer}.cpp, codebooks.h, turboquant_layout.h), host/{turboquant_host, turboquant_launcher}.cpp, and include/ (copies of the two headers);
  - multi_tile/: 21 kernels (compute, reader, and writer for rotation, polarquant, inverse_rotation, residual, qjl_project, rsquare, and rnorm_matmul), six host programs, include/tile_layout.h, run_pipeline.sh, and seven tests;
  - reference/turboquant.py (17,718 bytes) and a committed reference/__pycache__/turboquant.cpython-310.pyc, which is timestamp-checked (flags 0) and which the batch never imports;
  - tests/ (four baseline tests), codebook_gen.py, CMakeLists.txt, HANDOFF.md, README.md, multi_wormhole/README.md (a placeholder), and build.log;
  - five dumps of N = 32 vectors: dump_input.bin (8,192 bytes), dump_quant_indices.bin (2,048), dump_quant_centroids.bin (8,192), dump_qjl.bin (576), and dump_output.bin (3,072). They were committed in ca11cf7 (2026-05-25) and are the codec's own outputs; the repository does not record which build or device produced them.
- License: no license file (no LICENSE, COPYING, or NOTICE file in the tree). The README's last section refers the reader to a LICENSE.txt that the tree lacks. The README also refers to docs/perf/, which is absent, and describes a `--cores N` flag that no multi_tile host program at 81817eb parses: each runs on core {0, 0}. So the README describes a later state of the code than the commit holds (an inference; see the alpha01 copy below).
- Build: CMakeLists.txt points TT_METAL_HOME at the TurboQuant checkout path, links $TT_METAL_HOME/build/lib, and defines TQ_DIM=128, TQ_BITS=4, TQ_QJL_DIM=128, HOST_BUILD, and FMT_HEADER_ONLY for every host program, plus TQ_QJL_SEED=0xCAFEBABEu for the projection host (CMakeLists.txt:75-81, :216). These equal the headers' own defaults (include/turboquant_layout.h:30-44; turboquant_multi_tile_proj.cpp:28-30), and HOST_BUILD is used nowhere. So ttmetal-host's pinned build line, which adds no defines, compiles the same values.

### The alpha01 copy

PLACEHOLDER: the result of the read-only search (Commands, step 2). If it finds a copy, the file-by-file comparison is batch option `--alpha01-copy`, which needs the owner's approval (Owner queue).

## Stages, kernels, host programs, and reference functions

Processors: "BRISC" is DataMovementProcessor::RISCV_0, "NCRISC" is RISCV_1, and "TRISC" means a compute kernel (ComputeConfig with math_fidelity HiFi4 and compile_args only). The baseline runs all of its math in scalar software float on data-movement cores.

| Stage (bible Tier B row) | Baseline kernels | Baseline host | multi_tile kernels | multi_tile host | reference/turboquant.py |
| --- | --- | --- | --- | --- | --- |
| WHT | kernels/rotation_kernel.cpp (BRISC; reads its vectors from DRAM itself; sign diagonal from the codebooks.h table, WHT, scaling); kernels/dataflow/stage_dump_writer.cpp (NCRISC; writes a CB to DRAM) | host/turboquant_host.cpp stage 0 (:166-177, writer :214-228) | rotation_compute.cpp (TRISC tile matmul with a 128 x 128 matrix the host builds), rotation_reader.cpp (BRISC), rotation_writer.cpp (NCRISC) | multi_tile/host/turboquant_multi_tile.cpp | wht :45-73, make_sign_diagonal :85-91, apply_rotation :94-96 |
| PolarQuant | kernels/polarquant_kernel.cpp (BRISC; scalar codebook search; packed 4-bit indices and centroids), with stage_dump_writer, run twice | turboquant_host.cpp stages 1a and 1b (:178-188, :293-307) | polarquant_compute.cpp (TRISC SFPU sum of 15 threshold steps; centroids only, no indices), polarquant_reader.cpp, polarquant_writer.cpp | turboquant_multi_tile_pq.cpp | Codebook :108-112, build_codebook :172-180, polarquant_encode :187-203, polarquant_decode :206-213 (codebooks.h is generated by codebook_gen.py) |
| QJL | kernels/qjl_kernel.cpp (BRISC on 32 cores, one vector each: inverse rotation, residual, r_norm, 128 LFSR sign rows; writes DRAM itself) | turboquant_host.cpp stage 2 (:309-420) | four programs: residual_* (stage 2b1), qjl_project_* (2b2; the host packs the sign bits), rsquare_* then rnorm_matmul_* (2c, two programs in one host) | turboquant_multi_tile_resid.cpp, _proj.cpp, _rnorm.cpp | QJLProjector :220-280 draws S from NumPy's RandomState, which the kernels do not use; the kernels' LFSR rows are reproduced by the codec's tests/test_qjl.py:178-196; TurboQuantCodec.encode :346-381 |
| dequant | kernels/dequant_attn_kernel.cpp (dataflow API: centroid lookup, inverse WHT, dot product with a query from runtime arguments, QJL correction; waits on CB 2 and CB 16, writes CB 17) | none: no host program launches it | none | none | decode :387-390, attention_dot :396-421, QJLProjector.inner_product_correction :253-280 |

Beyond the four stages:

- Inverse rotation: multi_tile stage 2a, inverse_rotation_{compute, reader, writer}.cpp with turboquant_multi_tile_ir.cpp; reference iwht :76-78 and apply_inverse_rotation :99-101. In the baseline it sits inside qjl_kernel.cpp.
- Residual and r_norm: multi_tile stages 2b1 and 2c. In the baseline both sit inside qjl_kernel.cpp.
- Record packing into 18-byte records: host-side only (turboquant_host.cpp:421-443; multi_tile/tests/test_qjl_end_to_end.py), no kernel.
- host/turboquant_launcher.cpp: a command-line wrapper that starts turboquant_host; it has no tt-metal code.
- Kernels no host program launches: kernels/dataflow/reader_kernel.cpp and writer_kernel.cpp, which use the deprecated CB enum names, and dequant_attn_kernel.cpp (found by the kernel-path string literals of every host program; tier_b.py kernel_paths).
- The multi_tile pipeline reads the baseline's dumps: stage 2a reads dump_quant_centroids.bin (multi_tile/run_pipeline.sh), and stage 2b1 reads dump_input.bin.

Kernel sha256 at 81817eb (sha256 of the file bytes, the form the manifest's upstream entries use):

| Kernel | sha256 |
| --- | --- |
| kernels/rotation_kernel.cpp | 0b3f99f028225ad544094b8ac11c3b0f5eb90a467e5ceda4433415c2155f6198 |
| kernels/polarquant_kernel.cpp | 22afa38ccc2067b75fcf3724d3526012efd1b4c1baa115e440cb2147c888d6e8 |
| kernels/qjl_kernel.cpp | 34322f600ac2694a53e2d257006a32c373b42b85d8ba165343ef1849818b3cda |
| kernels/dequant_attn_kernel.cpp | 9966d56aa5bdec519ae7df25c56029fc369da57c9d44dee362a6d5ef7e456d32 |
| kernels/dataflow/stage_dump_writer.cpp | 78ec557e18aff5ac9107096b10ed11a8c5c970f07fe0cdec4611ce730ba3517c |
| kernels/dataflow/reader_kernel.cpp | f2ff2a5a82a1be31433b924f67c2f59c751a86f75404f5f02c3f352294df3836 |
| kernels/dataflow/writer_kernel.cpp | be8e62975fed178b752f1143fc7e746659e732ce5c89c9587eb0eba16b23c360 |
| kernels/codebooks.h | 426803e9f0e5a9171f28d64cbee5b682aa0365dddf905bac9aed2c8dfd855122 |
| kernels/turboquant_layout.h | ad0501f6db62c8dff6223cdd5b1dc1dba235fcb898639c13aca58572d15d2147 |
| multi_tile/kernels/rotation_compute.cpp | f44fe8ad0d43df35bc30998d170855d4194dda6df1d4fb873c270b0cd4991743 |
| multi_tile/kernels/rotation_reader.cpp | 921981f6b68fbd914d5144b0249e4daa5382543ea8d9038156d6a04109207634 |
| multi_tile/kernels/rotation_writer.cpp | 71030a68d60617b195ffce397370f0b22c992ec6e8047b9e94fe58332c5be0c4 |
| multi_tile/kernels/polarquant_compute.cpp | 3811d2f767e6d06ff6e9c48438e5fc75186206e0e6ec9ae2a02fade526a9fffb |
| multi_tile/kernels/polarquant_reader.cpp | dac41a024dc08d986e1182116719faa4b8af80d3b3d479b828933f8cb4b6e118 |
| multi_tile/kernels/polarquant_writer.cpp | 400ef51c037ce107465bc9548d5e9101f899ada1e164698b8abe32e8793d998d |
| multi_tile/kernels/inverse_rotation_compute.cpp | 0a69f98c67cf62d2d9dca88c2c5d71f8294d9f006e60e7fdab577cb2631d1179 |
| multi_tile/kernels/inverse_rotation_reader.cpp | d186eeac5054aece0c295731f58ab5b189c173050adc51801660649bd15f10aa |
| multi_tile/kernels/inverse_rotation_writer.cpp | abb3a11f0252c1739e8b27afb3f0e2b83e39ccce40cf887d8dcda50c39560463 |
| multi_tile/kernels/residual_compute.cpp | 15566ec09dae3eea77a8a74e7929fd851958bedf5e30a03634b8137bd56305e8 |
| multi_tile/kernels/residual_reader.cpp | 7aee44b0854e131840dde28705035ed82e153beca4ed392b7974a06e337f78b5 |
| multi_tile/kernels/residual_writer.cpp | 9f19b1c181c9e64b58473ff6dbac27a70192536665d5921330f492f1e1ef9274 |
| multi_tile/kernels/qjl_project_compute.cpp | d22e08470fb7e65444a715bb2d4ada3b4bd792677d4cbf26eb5d2b4278788707 |
| multi_tile/kernels/qjl_project_reader.cpp | 35066f17c0a944183421766170bbdb9fe3248d4c11b35b13c52e90a69d4fddb7 |
| multi_tile/kernels/qjl_project_writer.cpp | 819cec4819227b7e291791672454dc99d60c236ee84edba99528a1ea793a2407 |
| multi_tile/kernels/rsquare_compute.cpp | 0943c2a43cc81c62372218b5a4a53f6a2dbd606e2de151759cf2ad41343bc1cd |
| multi_tile/kernels/rsquare_reader.cpp | f48c24be48ca6a07a0a6ad18dc712b715ebf3bd03eed317a118c27c192cedd45 |
| multi_tile/kernels/rsquare_writer.cpp | 837ce2573ba4e89297abed87f4f3d794418bad3598fa598c18227f99c9c284c8 |
| multi_tile/kernels/rnorm_matmul_compute.cpp | c85283b02b99a64bb8839d39de70754357fee9b2d97b3f1877db928737006071 |
| multi_tile/kernels/rnorm_matmul_reader.cpp | 79aa8ed9ea233d735f846c3a998986971fdc6bd5bcd31dd0c75a46287862f1d9 |
| multi_tile/kernels/rnorm_matmul_writer.cpp | 592180deacea069b175a2231bf91922cbf01f9d9df22b0b36d85876264cad82f |

### Facts checked locally

These are exploratory checks on the workstation with NumPy 2.2.6, not host results:

- The kernels' sign table, SIGN_DIAGONAL_d128 in include/codebooks.h, equals the reference's make_sign_diagonal(128, seed=0) element for element, as a comment in the codec's tests/test_rotation.py:43 claims. So the reference's own apply_rotation and apply_inverse_rotation with seed 0 are the oracles for both rotation kernels.
- The committed dump_input.bin is byte for byte the baseline host's own generator output at N = 32 (turboquant_host.cpp:57-75, seed 42, then bf16 by truncation, :43-46, :257-260).
- tier_b.py compare, given the committed dumps as baseline outputs, reports: centroids max_abs 0.00090341 (bound 0.005), QJL bits match 1.0 (bound 0.98), and r_norm max relative error 0.00486621 (bound 0.01). The rotation and index checks need dump_rotated.bin, which is not committed. So the codec's committed outputs, which the README attributes to silicon, pass the codec's own thresholds under the batch's reference port. That checks the compare code; it measures nothing.
- The dry run's stand-in baseline (double-precision math, bf16 by truncation) reproduced dump_quant_indices.bin, dump_quant_centroids.bin, dump_qjl.bin, and dump_output.bin byte for byte. That checks the port again, not the kernels.

## Proposed item map (OQ-048, items option (c))

Provisional: the map waits for the batch's results. A kernel that fails at the pin leaves its item out under OQ-048's recommendation (a). Each item is one TT program with the codec's kernels, plus a TT host program and a C++ counterpart written here (holding (a)).

| Proposed item | Stage | Codec kernels | Oracle (reference function) | The codec's own bound (file:line) | binary_io fit |
| --- | --- | --- | --- | --- | --- |
| tq-wht-baseline | WHT | rotation_kernel.cpp, dataflow/stage_dump_writer.cpp (with codebooks.h) | apply_rotation | max_abs 0.005 and mean_abs 0.001 (tests/test_rotation.py:184, :188) | max_abs yes; the mean bound has no binary_io metric |
| tq-wht-multitile | WHT | rotation_{reader, compute, writer}.cpp | apply_rotation | max_abs 0.01, against the baseline's output (multi_tile/tests/test_rotation_tile.py:47) | max_abs |
| tq-polarquant-baseline | PolarQuant | polarquant_kernel.cpp, stage_dump_writer.cpp (with codebooks.h) | polarquant_encode, polarquant_decode | index match 0.99 (tests/test_polarquant.py:229); centroids max_abs 0.005 (:277) | centroids max_abs; a match rate has no binary_io metric |
| tq-polarquant-multitile | PolarQuant | polarquant_{reader, compute, writer}.cpp | decode of encode, with the threshold sense of the kernel (searchsorted side right, multi_tile/tests/test_polarquant_tile.py:45) | max_abs 0.01 (:89-90) | max_abs |
| tq-qjl-baseline | QJL | qjl_kernel.cpp (with codebooks.h) | inverse rotation, residual, LFSR projection, r_norm | bits match 0.98 (tests/test_qjl.py:251); r_norm max relative error 0.01 (:289) | neither is a binary_io metric |
| tq-inverse-rotation-multitile | beyond the four (part of dequant in the reference's decode) | inverse_rotation_{reader, compute, writer}.cpp | apply_inverse_rotation | max_abs 0.01 (multi_tile/tests/test_inverse_rotation_tile.py:84) | max_abs |
| tq-residual-multitile | QJL (multi_tile split) | residual_{reader, compute, writer}.cpp | x - x_hat | max_abs 0.001 (multi_tile/tests/test_residual_tile.py:27) | max_abs |
| tq-qjl-projection-multitile | QJL (multi_tile split) | qjl_project_{reader, compute, writer}.cpp | r times the LFSR S transposed | bit match 0.998 where the projection is at least 0.01 in magnitude, and 0.95 overall (multi_tile/tests/test_qjl_project_tile.py:71) | no binary_io metric for the bits; the projection B itself has no stated bound |
| tq-rnorm-multitile | QJL (multi_tile split) | rsquare_{reader, compute, writer}.cpp and rnorm_matmul_{reader, compute, writer}.cpp | the norm of r | max relative error 0.02 (multi_tile/tests/test_rnorm_tile.py:48) | no relative metric in binary_io |
| held: tq-dequant-baseline | dequant | dequant_attn_kernel.cpp | attention_dot | none in the codec's tests | needs data-movement kernels the codec does not have, so its item would carry kernels written here |

Notes for P5.13:

- Item tolerance accepts only pcc, max_abs, and ulp (bible, Item tolerance). The codec bounds several stages by match rates or relative errors, so mapping each to a declared tolerance is P5.13's design work, or the owner's choice where it changes what the check means.
- dequant has a baseline kernel only, and no host program or feeder kernels in the codec, so OQ-048's items option (a) would need data-movement kernels written here.
- The multi_tile rewrite has no dequant stage. It has six encoder stages instead of the bible's three encoder stages.

## Does any kernel hold tt-metal text?

Method (Method, step 3): 36 codec sources (the 28 kernel .cpp files, the 2 baseline host sources, and the 6 multi_tile host programs) against 1424 tt-metal files at the pin (tt_metal API, hw/inc, hostdevcommon, kernels, programming examples, impl/kernels, jit_build, tt_stl, and ttnn's operation kernels) and 1476 at 91f5a51.

Result, the same at both commits:

- No codec comment line of 30 characters or more occurs in any tt-metal file (0 of 273 such lines across the 36 files).
- The longest run of consecutive normalized code lines shared with one tt-metal file is 3. Those runs are API idioms:
  - the designated fields of a DataMovementConfig (shared with programming_examples/contributed/vecadd/vecadd.cpp);
  - three get_arg_val reads of Mt, Kt, and Nt (shared with programming_examples/matmul/matmul_common/kernels/dataflow/reader_bmm_8bank_output_tiles_partitioned.cpp);
  - two loop headers and tile_regs_acquire (shared with programming_examples/matmul/matmul_single_core/kernels/compute/mm.cpp).
  Every other shared line is a single API call such as cb_pop_front with its arguments: 1 to 13 lines per kernel, of 14 to 115 normalized lines.
- Reading: the kernels use tt-metal's API in its documented forms and copy no example's text beyond those idioms. That bears on holding (b), tracking the kernels here (OQ-018). Confidence: medium-high. The comparison is line-based, so a copied block with renamed identifiers would not show, and LLK sources and tests/ were not in the corpus.

## API distance: tt-metal 91f5a51 against the pin 5280a9cf

Every tt-metal name the codec's host programs and kernels use, with its declaration at both commits. Paths are relative to tt_metal/api/tt-metalium (host) or tt_metal/hw/inc (kernel) unless given in full. "Same" means the same signature (and, for structs, the same fields in the same order).

Host programs:

| Name | 91f5a51 | 5280a9cf | Status |
| --- | --- | --- | --- |
| CreateProgram | host_api.hpp:148 | host_api.hpp:148 | same |
| CreateKernel (file, core spec, config) | host_api.hpp:164 | host_api.hpp:164 | the pin's config variant also takes EthernetConfig (a superset); the codec passes DataMovementConfig or ComputeConfig, which both accept |
| CreateCircularBuffer | host_api.hpp:211 | host_api.hpp:211 | same |
| SetRuntimeArgs (core spec, initializer list) | host_api.hpp:505 | host_api.hpp:499 | same |
| CircularBufferConfig (total size, map of CB index to DataFormat); set_page_size | circular_buffer_config.hpp:34, :59 | circular_buffer_config.hpp:35, :60 | same |
| DataMovementConfig (processor, noc, defines); ComputeConfig (math_fidelity, compile_args) | kernel_types.hpp:39, :76 | kernel_types.hpp:39, :76 | same (lines 30-70 identical) |
| DataMovementProcessor RISCV_0, RISCV_1; NOC RISCV_0_default, RISCV_1_default | data_types.hpp:12-25 | data_types.hpp:12-25 | same |
| MathFidelity HiFi4 | base_types.hpp:10, :14 | base_types.hpp:10, :14 | same |
| CoreCoord; CoreRange (start, end) | core_coord.hpp:30, :42 | core_coord.hpp:31, :43 | same |
| BufferType DRAM | buffer_types.hpp:30-31 | buffer_types.hpp:30-31 | same |
| MeshBuffer::create, address; ReplicatedBufferConfig (size); DeviceLocalBufferConfig (page_size, buffer_type) | mesh_buffer.hpp:77, :99, :41, :24 | the same lines | file identical |
| MeshWorkload; add_program (range, program) | mesh_workload.hpp:20, :35 | mesh_workload.hpp:20, :35 | same |
| EnqueueMeshWorkload | distributed.hpp:36 | distributed.hpp:37 | same |
| EnqueueWriteMeshBuffer; EnqueueReadMeshBuffer (resizes the destination vector) | distributed.hpp:70, :79 | distributed.hpp:71, :80 | same |
| MeshDevice::create_unit_mesh (device id, ...) | mesh_device.hpp:298 | mesh_device.hpp:290 | same |
| mesh_command_queue; compute_with_storage_grid_size; close | mesh_device.hpp:285, :125, :171 | mesh_device.hpp:277, :125, :171 | same |
| MeshCoordinate (c0, c1); MeshCoordinateRange (start, end) and (shape); shape | mesh_coord.hpp:67, :122, :131, :144 | mesh_coord.hpp:68, :123, :132, :145 | same |
| hal::get_dram_alignment | hal.hpp:39 | hal.hpp:39 | same |
| bfloat16 (constructor from a value; one uint16_t member) | bfloat16.hpp:17-28 | bfloat16.hpp:17-28 | file identical |
| tilize_nfaces, untilize_nfaces (templates) | tilize_utils.hpp:89, :92 | tilize_utils.hpp:89, :92 | same; both instantiated for bfloat16 in tt_metal/impl/data_format/tilize_utils.cpp:572, :577, a file identical at both commits |
| tt::CB c_in0 to c_in3, c_out0 (deprecated enum) | tt_metal/hostdevcommon/api/hostdevcommon/kernel_structs.h:80-102 | the same lines | same |
| tt::DataFormat Float16_b, RawUInt8 | tt_backend_api_types.hpp:27, :39 | tt_backend_api_types.hpp:27, :39 | same |
| Headers host_api.hpp, distributed.hpp, hal.hpp, device.hpp, bfloat16.hpp, tilize_utils.hpp | present | present | same |

Kernels:

| Name | 91f5a51 | 5280a9cf | Status |
| --- | --- | --- | --- |
| Include paths api/dataflow/dataflow_api.h, api/compute/{common, tile_move_copy, matmul, eltwise_binary, eltwise_binary_sfpu, compute_kernel_api}.h, api/compute/eltwise_unary/{binop_with_scalar, comp, eltwise_unary, sqrt}.h | present | present | same paths |
| get_arg_val | api/dataflow/dataflow_api.h:149 | :151 | same |
| get_compile_time_arg_val (a macro) | api/compile_time_args.h:62 | :62 | same |
| cb_reserve_back, cb_push_back, cb_wait_front, cb_pop_front, get_write_ptr, get_read_ptr (dataflow) | dataflow_api.h:389, :200, :468, :242, :306, :327 | :391, :202, :470, :244, :308, :329 | same |
| cb_wait_front, cb_pop_front, cb_reserve_back, cb_push_back (compute) | api/compute/cb_api.h:44, :76, :93, :131 | :41, :73, :90, :128 | same |
| InterleavedAddrGen (bank_base_address, page_size); InterleavedAddrGenFast (bank_base_address, page_size, data_format) | internal/dataflow/dataflow_api_addrgen.h:276, :340 | :268, :332 | same fields |
| get_noc_addr (page id, address generator) | dataflow_api_addrgen.h:294 | :286 | same |
| noc_async_read; noc_async_write | dataflow_api.h:546, :822 | :548, :824 | same |
| noc_async_read_barrier; noc_async_write_barrier | dataflow_api.h:1731, :1761 | :1733, :1763 | same |
| noc_async_read_tile; noc_async_write_tile (InterleavedAddrGenFast overloads) | dataflow_api.h:1158, :1328 | :1160, :1330 | same |
| mm_init; matmul_tiles (five arguments) | api/compute/matmul.h:88, :137 | :88, :137 | same |
| tile_regs_acquire, tile_regs_commit, tile_regs_wait, tile_regs_release | api/compute/reg_api.h:42, :83, :51, :94 | the same lines | file unchanged |
| pack_tile | api/compute/pack.h:61 | :61 | file unchanged |
| copy_tile | api/compute/tile_move_copy.h:95 | :95 | same |
| init_sfpu | api/compute/eltwise_unary/eltwise_unary.h:46 | :46 | same blob |
| unary_ge_tile_init; unary_ge_tile | eltwise_unary/comp.h:174, :167 | :174, :167 | same blob |
| binop_with_scalar_tile_init; add_unary_tile; mul_unary_tile | eltwise_unary/binop_with_scalar.h:89, :28, :36 | the same lines | same blob |
| add_binary_tile_init; add_binary_tile (three DST indices) | api/compute/eltwise_binary_sfpu.h:66, :35 | :66, :35 | same blob |
| binary_op_init_common; sub_tiles_init; sub_tiles; mul_tiles_init; mul_tiles | api/compute/eltwise_binary.h:31, :119, :199, :88, :142 | :31, :114, :194, :83, :137 | same signatures; see the fidelity change below |
| sqrt_tile_init; sqrt_tile | eltwise_unary/sqrt.h:18, :35 | :18, :35 | same blob |
| tt::CBIndex c_0 to c_3, c_16, c_17 | kernel_structs.h:12-30 | :12-30 | same |
| DataFormat Float16_b (kernel side) | internal/tt-1xx/wormhole/wormhole_b0_defines/tensix_types.h:220 | :220 | same |
| CB c_in0, c_out1 (only in the unlaunched reader_kernel.cpp and writer_kernel.cpp) | kernel_structs.h:80-103 | the same lines | same |

Differences between the commits that could matter (all read, none run):

- eltwise_binary.h: at 91f5a51, add and subtract ops (their init and their math call) run at MathFidelity::LoFi whatever the kernel's fidelity (eltwise_binary.h:65-71, :177, :201 at 91f5a51). At the pin they run at the kernel's MATH_FIDELITY (:65-66, :172, :196). So the codec's residual_compute.cpp (sub_tiles, HiFi4) takes more fidelity phases at the pin than it did where it was written. The Tier A eltwise_binary example takes the same path at the pin (add_tiles with HiFi4) and passed (ttsim Facts).
- tt-llk: the pin's submodule commit f9909668 is 46 tt-llk commits behind 91f5a51's d090e3bf. Among them are "Matmul without utilizing MOP", "Eltwise binary cleanup and haloize mode in sub", and "Add LLK support for fp32_dest_acc control". Only their titles were read.
- JIT: the simple kernel-syntax transform, which lets a compute kernel define kernel_main at file scope as every codec compute kernel does, is identical at both commits (tt_metal/jit_build/genfiles.cpp:64-134 at 91f5a51, :60-130 at the pin). At both, the kernel's own directory is on its include path (tt_metal/impl/kernels/kernel.cpp:248-255 and :247-254), so codebooks.h beside a kernel is found. At both, a relative kernel path is looked up in the working directory first (:60 and :59). 91f5a51 adds a fourth TRISC source for Quasar only.
- Other differences in the headers the codec includes (dataflow_api.h's generated-descriptor include names, a NOC atomic argument, compute_kernel_api.h's power and sfpu_reduce, mesh_device.hpp's get_system_mesh_id, circular_buffer_config.hpp's address_offset) touch nothing the codec calls.
- Device opening: every host program opens chip 6 with MeshDevice::create_unit_mesh(6) (turboquant_host.cpp:263; turboquant_multi_tile.cpp:212, _pq.cpp:93, _ir.cpp:143, _resid.cpp:71, _proj.cpp:123, _rnorm.cpp:93). The multi_wormhole README says chip 6 is the PCIe-attached chip of TurboQuant's cluster. At the pin, create_unit_mesh calls create_unit_meshes, which looks each id up in the control plane before creating anything (tt_metal/distributed/mesh_device.cpp:448-483, :379-403). ttsim presents one chip (ttsim Facts).

Predictions (PREDICTION, from reading; the batch tests each):

- PREDICTION: all seven host programs build at the pin with ttmetal-host, warnings at most. Every name they use is declared at the pin with the same signature, their headers are present, and the tilize templates are instantiated for bfloat16. Confidence: high.
- PREDICTION: as written, none runs on ttsim. Each aborts at device open, since chip 6 does not exist on a one-chip simulator (exit status 134, a tt-metal fatal or an uncaught exception). The multi_tile programs read their input files first, and the baseline writes dump_input.bin first. Confidence: high.
- PREDICTION: with the device id changed to 0 ("dev0"), the multi_tile kernels JIT-compile and run, and their outputs meet the codec's bounds; the sub_tiles fidelity change is not expected to change values. Confidence: medium. The tt-llk difference is unread, and the JIT compiler (sfpi 7.25.0 at the pin against 7.29.0 in TurboQuant's checkout) differs.
- PREDICTION: the baseline kernels compile at the pin and run slowly. They do soft-float math on data-movement cores, and the QJL stage does 16384 multiply-adds per core on 32 cores, so the run may take minutes on ttsim (P4.9 saw about 430 kHz of simulated clock with one busy core). A ttsim finding is possible in qjl_kernel.cpp, which writes an 18-byte record to DRAM in one NOC write. Confidence: low on both.
- PREDICTION: no kernel takes ttsim issue #18's path. No host sets fp32_dest_acc_en or unpack_to_dest_mode, every compute CB is Float16_b (the baseline's RawUInt8 CBs serve data-movement kernels only), and the baseline has no compute kernel. Confidence: high. The batch reads the JIT's generated descriptors to confirm.
- dequant_attn_kernel.cpp, reader_kernel.cpp, and writer_kernel.cpp are not launched by any codec host program, so the batch cannot check them. Their API names exist at the pin.

## The batch

Files (plans/spikes/p5-tier-b/; each states its contract in its header):

- run.sh, the orchestrator, runs as one `rx job` from a clean commit (Commands, step 4):
  1. Preflight. It refuses an rx snapshot commit or a dirty slot (as P4.9's batch did), a root outside /mnt/nvme10/joseph_ufl, a set gate STOP, a missing TMPDIR, an existing run directory, a missing tool or pin value, a core limit that is not exactly one byte, and a scratch root that 1 GiB more would push past the 115 GiB stop line.
  2. Fetch. The codec at 81817eb goes by its public URL into $LASSI_RUNS_ROOT/p53-tier-b/$LASSI_RX_RUN_ID/src: git init, then a depth-1 fetch of that commit with no global or system git configuration (GIT_CONFIG_GLOBAL=/dev/null, GIT_CONFIG_NOSYSTEM=1), no hooks, no fsmonitor, no credential helper, and no file protocol, then a check of HEAD.
  3. tier_b.py prepare.
  4. With `--alpha01-copy` only: tier_b.py copy, its own step, bounded at 610 s; a failure or a timeout is reported and the run goes on.
  5. A uv warmup, then tier_b.py batch under the slot's offline uv environment.
  6. NumPy probes, then tier_b.py compare under a Python with NumPy: the host's python3, then the user site, then a uv environment from the cache only. If none has NumPy, compare runs locally after the pull.
  7. summarize, then read-only host checks (counts of this account's new files under /tmp, /var/tmp, and /dev/shm; changes in the pinned trees; the default JIT cache roots; the scratch du before and after; the run directory's du).

  It reads the tt-metal and ttsim versions and tree names it records and checks from toolchains/tt-metal.pin and toolchains/ttsim.pin. It exits with the batch's status, so the gate's record shows a batch that did not finish: 0; 2 when lassi's install checks refused; 3 when the probe failed; 4 when the budget could not start the batch; 5 when a stop file or the run directory cap ended it early, or when run.sh saw a stop file after a finished batch. A step the budget skips is recorded in its step record and leaves the status alone. An earlier failure ends it with its own status (2 in preflight; 1 for the fetch, prepare, or the uv warmup).
- tier_b.py, the helper (standard library, plus NumPy for compare and lassi for batch):
  - batch first runs a probe script through the ttsim executor. It refuses to run any program unless the sandbox's /dev holds only the sandbox's own 13 entries and /sys/bus, /sys/class, and every /sys/devices entry but system are empty. A directory the probe cannot list counts as not empty (-1). The probe also records whether /proc/bus/pci is present and how many entries it lists, and does not gate on it (Safety).
  - It then builds each of the seven host programs with the Toolchain ttmetal-host (build_toolchain, as tools/ttsim_smoke.py does). Each program's host source and headers sit at the build root and each kernel at the path the program names.
  - Each program is built twice: as written ("asis"), and "dev0", a changed copy in the run directory only, where its one create_unit_mesh(6) becomes create_unit_mesh(0) (the edit is refused unless exactly one occurrence exists). The kernels are never changed. The evidence says so: summary.txt and compare.txt carry the line "dev0 = a changed copy: create_unit_mesh(6) -> create_unit_mesh(0) in the run directory only, not the codec's program", compare.json carries it under "variants", and every dev0 build and run record carries the patch.
  - Each built program runs through lassi.executors.ttsim on N = 32 vectors with Limits of 16 CPUs and 8192 MiB, under the ttsim row's settings: TT_METAL_SIMULATOR, slow dispatch, its own JIT cache in its workdir, the pinned tree read-only, no HOME.
  - Order: the seven "asis" builds; the six multi_tile stages, each a dev0 build and run, in pipeline order; the seven "asis" runs, the device-open check; the baseline's dev0 build and run last. So a hang in the baseline (a 900 s wall limit plus a 900 s Watcher rerun) cannot starve the multi_tile stages or the asis runs. Each multi_tile stage reads the previous stage's output, and stages 0, 2a, and 2b1 read the codec's committed dump_input.bin and dump_quant_centroids.bin, the baseline outputs that the codec's multi_tile/run_pipeline.sh takes as its inputs (:5-6).
  - Before the probe and before every build and run, it checks the stop files, the run directory's size, and the budget. A stop file, or a run directory holding more than 1 GiB of blocks (OQ-050's disk figure), skips that step and every later one; a budget that cannot hold a step skips that step.
  - A missing input comes from the codec's committed dumps (dump_input.bin, dump_quant_centroids.bin) or is synthesized with the reference math, and each run records where each input came from.
  - Records per program and variant: build result and diagnostics (counts, warning codes, first errors), exit status, hang, sim_ub, sim_gap, findings, kernel JIT errors, the Watcher note flag, simulator wall time (labeled), each output's sha256, and the unpack_to_dest facts read from the JIT's generated descriptors. Those facts are the DST_ACCUM_MODE values and the unpack source and destination formats (tt_metal/jit_build/genfiles.cpp:291-294, :487-490 at the pin), flagged when Float32, Int32, or UInt32 appear or dest accumulation is on.
  - compare checks each dev0 output against reference/turboquant.py, imported from a copy of that one file after its sha256 is checked against the file at 81817eb (a4a79ca85104ecb26fc822d24cf66ed6d3f1ad9011bb62fe354c34839a3cf0b0). It uses the kernels' sign table (checked equal to make_sign_diagonal(128, 0)), the kernels' own codebook from include/codebooks.h, and the LFSR sign rows. Each check uses the codec's own threshold, cited by file and line. The baseline's outputs are also compared byte for byte with the committed dumps.
  - copy (run.sh's option `--alpha01-copy DIR`) compares a found copy file by file, read only. It reads git state with --no-optional-locks, core.fsmonitor off, no hooks, and no global or system configuration, and never a remote URL; hashes files up to 64 MiB (larger files by size only); and stops walking at 5000 files. run.sh refuses a copy under LASSI's own trees, the toolchains root, the shared tt-metal checkout, or .cache.
- dryrun.sh and dry_standin.py are workstation-only (Dry run).

Safety on the shared host:

- Every program run goes through the ttsim executor's sandbox. Its private /dev holds no device node, /sys/bus, /sys/class, and /sys/devices/* (system excepted) are covered (lassi/executors/sandbox.py, steps 6 and 8), the scratch root and HOME are hidden, and TT_METAL_SIMULATOR names ttsim.
- The Wormhole cards are back on alpha01's PCI bus (bible, Environment State). So P4.9's preflight, which refused a host with a vendor 1e52 device, cannot be used. The probe confirms the sandbox's view instead, and the batch runs nothing tt-metal outside the sandbox. The probe checks exactly this: the sandbox's /dev holds only its own 13 entries, and /sys/bus, /sys/class, and every /sys/devices entry but system list no entry. /proc/bus/pci stays readable inside the sandbox (lassi/executors/sandbox.py:289-290), a known sandbox limit: it lists the host's PCI devices and opens none. The probe records its presence and entry count in device-probe.json and does not gate on it.
- No command text names a device path or the Tenstorrent SMI tool.
- Each run's JIT cache and logs stay in its workdir, so no shared cache is touched (PHASE-NOTES P4, the collision rule).
- The TurboQuant checkout is hidden inside both sandboxes, as part of the scratch root, and nothing names it but the copy option's refusal.
- A core limit of one byte, verified, covers the whole batch. The report directory is plain ASCII with no source text, and program streams and compiler stderr go to local/.
- compare imports the codec's reference/turboquant.py on the host, outside the sandbox. It is the owner's own code (every commit on main is by the account JoeMad21, this repository's identity), not model-generated code (Agent Rule 6), at a commit whose HEAD the fetch and prepare check, from a copy whose sha256 compare checks first. Importing it runs only its imports, of __future__, hashlib, struct, typing, NumPy, and numpy.typing (reference/turboquant.py:25-32). The rest of its top level defines functions and classes whose bodies hold annotations, docstrings, and methods, and its self-tests run only under its __main__ guard (:546).
- The copy comparison, if approved, runs `git status` in the copy, which can run a clean filter that the copy's own configuration names (git-lfs, for one). The copy is a checkout of this account's own projects.
- Nothing deletes a file it did not create: lassi removes only its own temporary files (the compile's private TMPDIR, the --version check's directory, a build directory's stale output, and the ttsim executor's @ttsim/watcher copy). Every step is bounded (PROJECTED costs).
- Stop files (the run's STOP and the gate STOP): run.sh checks both before the fetch, prepare, the copy comparison, the uv warmup, the batch, the NumPy probes, compare, and the scratch du. The first it sees skips that step and every later one, with "stop file"; summarize and the other host checks, read only and bounded, still run. Inside the batch, tier_b.py checks both before the probe and before every build and run.
- If an outer timeout fires (the batch step's or rx's), the sandboxed program then running is not killed with it, since the sandbox starts it in its own session (lassi/toolchains/_base.py:268). It outlives run.sh by at most its wall limit plus 7 s (907 s for the baseline), ended by its own innermost timeout or the scope's RuntimeMaxSec backstop (lassi/executors/sandbox.py:1301).

Output under $LASSI_RUNS_ROOT/p53-tier-b/<rx id>/:

- report/ (batch.txt, source.tsv, license.txt, programs.tsv, device.txt, device-probe.json, steps/*.json, compare.json and compare.txt, summary.txt, checks.txt, and alpha01-copy.json if the option was used);
- outputs/ (the input and output dumps, binary numbers, a few KiB each);
- local/ (capped streams and stderr, which may quote source; pull only into .rx/pulls/);
- raw/ (build directories and JIT caches; never pulled);
- src/ (the codec checkout).

## Dry run

Environment: WSL Ubuntu 24.04 on the workstation (kernel 6.18, util-linux 2.39.3, systemd user manager), with a venv holding pyyaml, pyarrow 26.0.0, tiktoken, and numpy 2.2.6. `bash plans/spikes/p5-tier-b/dryrun.sh <case> <new dir> <venv> <codec clone>` runs the real run.sh and tier_b.py, the real lassi build_toolchain, compile sandbox, ttsim executor, and run sandbox, against stand-ins:

- a copy of the repository whose toolchains/tt-metal.pin names a fake clang++-20 (dry_standin.py) and whose toolchains/ttsim.pin carries the sha256 of fake library and descriptor files;
- a fake toolchains root (install record, CMake package list);
- a fake uv that runs the venv's python;
- dry_standin.py as the programs: it emulates each codec program's file behavior with the reference math and writes a JIT descriptor header in the pin's format.

The scratch root is a tmpfs mounted at /mnt/nvme10/joseph_ufl inside a user and mount namespace, so run.sh's checks run unchanged. The fetch was real: GitHub, from the workstation.

Cases, last run 2026-10-10 from 21:09 to 21:16 EDT on the files as they stand in this draft (all ten in one pass); run.sh's stdout ran 2 to 63 lines (under 120). Every case's walls were scaled by 0.05.

- main, with injected faults (a build error in mt-residual, both variants; a ttsim UB finding in mt-polarquant; a kernel JIT error in mt-qjl-projection; a hang in mt-rnorm). Every path ran, in the new order: the seven asis builds, the six multi_tile stages, the seven asis runs, then the baseline's dev0 build and run. The probe passed (13 /dev entries, 15 /sys directories empty) and recorded /proc/bus/pci as present with 4 entries (WSL's), without gating on it. The build error, the finding (sim_ub True), the JIT diagnostic (exit 134), and the hang (exit 124, hang True, with the executor's Watcher rerun) were each recorded. mt-rotation and mt-inverse-rotation read the codec's committed dumps; mt-qjl-projection and mt-rnorm ran on a synthesized dump_r_tile.bin, since mt-residual did not build. Seven reference checks passed on the stand-ins' outputs, and the dev0 baseline's five dumps matched the committed ones byte for byte. The six built "asis" programs exited 134, and the baseline's dump_input.bin from its "asis" run matched the committed file. summary.txt and compare.txt carried the dev0 line, compare.json its "variants" key, and every dev0 build and run record the patch. Exit 0.
- order: P53_BUDGET_S=440 and a hang in the baseline. The six multi_tile stages passed their runs and the seven asis runs finished before the baseline's dev0 run hung (exit 124, Watcher rerun), so the hang starved nothing. The summary marks the baseline's five checks A (failed for an absent file). Exit 0.
- budget: P53_BUDGET_S=390 and a hang in the first multi_tile stage. After the hang the budget skipped the five later multi_tile dev0 builds and the baseline's ("the batch budget cannot hold a 300 s step"), and the seven asis runs, which need less, still ran. Exit 0: a budget skip is recorded per step and does not change the status.
- stop: a hang in the first multi_tile stage, with the run's STOP file appearing 25 s after run.sh started, during that hang. Every later build and run inside the batch was skipped with "stop file" (batch status 5); run.sh then skipped the NumPy probes, compare, and the scratch du with "stop file", and still ran summarize and the bounded checks. Exit 5.
- stop-fetch: the run's STOP file appeared while the fetch ran. run.sh's check before prepare skipped prepare and every later step; summarize and the bounded checks ran. Exit 5.
- stop-batch: the run's STOP file appeared as the batch step started. tier_b.py's check before the probe skipped the probe and every build and run (device-probe.json records the skip), and run.sh skipped the NumPy probes and the rest. Exit 5.
- size: a 1100 MiB file appeared in the run directory as the batch step started. The cap check before the probe reported "the run directory holds 1101 MiB, over its 1024 MiB cap" and skipped the probe and every build and run. Exit 5.
- copy: no faults, with `--alpha01-copy` on a changed clone (one file edited, one removed, one added, one 65 MiB file). The comparison ran as its own step and reported 66 same, 1 different, 1 only on GitHub, 2 only in the copy, 3 status lines, and walk capped False. All 11 reference checks passed. Exit 0.
- copy-fail: as copy, with a file the account cannot read in the clone (a bind mount of /etc/shadow; the read failed, so nothing was read). The comparison step ended with status 1 (PermissionError, in local/copy.stderr.txt), run.sh said so, and the batch went on: the probe, all builds and runs, and the 11 reference checks, which passed. Exit 0. So a file the account cannot read ends the comparison, not the run.
- dirty: an uncommitted change in the copy; preflight refused it (exit 2).

Found and fixed by the dry run or its review before this draft:

- the order in which inputs are synthesized: a synthesized file now comes after the files it is computed from;
- a duplicated class word in finding lines, and a summary line over 200 characters;
- compare ran on "asis" outputs, which only adds absent-file failures, so it now covers dev0 runs and the committed-dump comparison;
- a run's budget need of twice the wall limit would have skipped the baseline's run after the builds, so it is now one wall limit plus 60 s, with the batch step's outer timeout allowing one Watcher rerun past the budget;
- the alpha01-copy walk left out __pycache__, which made the committed .pyc look missing;
- lines over 120 characters (ruff and the shell files);
- a clean refusal, with no traceback, when lassi's install checks refuse.

Found by the pre-launch audit of 2026-10-10 and fixed in this draft:

- the read-only search named lassi-gate, which the gate's deny pattern refuses, and the tt-metal checkout twice, which the gate's tt guard refuses now that the cards are back (Commands, step 2);
- the stated worst case, about 6100 s, left out the NumPy probes and the kill-after graces (the old code's worst case was 6570 s); the bound is now restated from the code, 6680 s, with the job's timeout at 7200 s (PROJECTED costs);
- run.sh read no stop file between its own steps, and the batch ran the probe without a stop check;
- "no PCI view" was wrong: /proc/bus/pci stays readable in the sandbox, so the probe now records it, and the report says what the probe checks;
- dev0 was not labelled as a changed copy in summary.txt, compare.txt, compare.json, or the run records;
- the baseline ran first, so its hang could starve the multi_tile stages; it now runs last;
- the run directory had no size check inside the batch; a step now starts only while it holds at most 1 GiB;
- "nothing is deleted" was too broad: lassi removes temporary files it created;
- the fetch could read a global git configuration, and tier_b.py's git calls left fsmonitor on;
- the copy comparison ran inside prepare, so its failure ended the run; it is now its own bounded step;
- a probe directory that could not be listed counted as empty;
- run.sh exited 0 whatever the batch's status;
- the run directory's du was unbounded, and batch.txt hard-coded the tt-metal and ttsim versions instead of reading the pins;
- the Owner queue listed the fetch as beyond OQ-050, which names it, and left out the change of verb, the doctor runs, the job polls, and the pulls.

Found by this draft's dry run and fixed: a stop file that run.sh saw after the batch's last step had finished left the exit status 0; run.sh now exits 5 after any stop.

A dry-run-only adaptation: util-linux 2.39 returns status 16 from `mount --move` inside a user namespace when it cannot update its userspace mount table, which the sandbox has made read-only by then. So dryrun.sh wraps mount(8) to accept status 16 inside its namespace. alpha01 has Ubuntu 22.04's util-linux 2.37, where P4.9 and P4.11 measured the sandbox working. The production files need no change.

The dry run could not cover:

- the real clang++-20 build of the codec's host programs against the pinned tree;
- the kernel JIT (sfpi), ttsim, real findings, real Watcher logs, and real wall times;
- the host's uv offline environment, and whether the host's python3 has NumPy;
- du over the real scratch root;
- the gate's environment (its check_command was run locally on the command texts; Commands, steps 2 and 4);
- util-linux 2.37's behavior in the sandbox;
- the outer timeouts: no step's `timeout` fired, so a program outliving the batch step (Safety) was not seen.

ruff passes on plans/spikes/p5-tier-b, and tools/check_text_policy.py --files passes on the four batch files and this report.

## PROJECTED costs

PROJECTED, not measured:

- Time: typically 10 to 35 minutes. That is about 14 builds of 10 to 60 s each (P4.13's five remote item tests took 68 s in all, exploratory; plans/PHASE-NOTES.md, P4), the baseline's run at 1 to 15 minutes (soft float on data-movement cores; the 32-core QJL stage), six multi_tile runs of 5 to 60 s, seven "asis" runs of a few seconds, the fetch, the uv warmup, two du runs, and compare.
- Bound, from the code. Each `timeout -k K T` ends its step by T + K seconds:
  - the scratch du before the budget starts: 600 + 10 = 610 s;
  - from the budget's start to the batch step's end: 3930 s. The fetch (300 + 10), prepare (600 + 10), the copy comparison (600 + 10, only with `--alpha01-copy`), and the uv warmup (600 + 30) spend the 3000 s budget, at most 2160 s in all. The batch step starts only while 300 s of the budget are left, which therefore always holds, and its bound is what is left plus 900 s for one Watcher rerun, plus a 30 s kill grace: 3000 + 900 + 30 = 3930 s;
  - the NumPy probes: (60 + 10) + (60 + 10) + (120 + 10) = 270 s;
  - compare, 600 + 10 = 610 s, and summarize, 120 + 10 = 130 s;
  - the host checks: two finds of 120 + 10, two JIT cache finds of 30 + 5, the scratch du of 600 + 10, and the run directory's du of 120 + 10, so 130 + 130 + 35 + 35 + 610 + 130 = 1070 s;
  - an allowance of 60 s for the unbounded small commands: git rev-parse, status, init, and checkout of small trees, mkdir, date, the pin reads, a sha256sum of the 162 KB simulator library, and the stop checks.

  Total: 610 + 3930 + 270 + 610 + 130 + 1070 + 60 = 6680 s (1 h 51 min). The job's timeout is 7200 s, 520 s above the bound. 7200 s is the gate's maximum for rx run, and rx job's own cap is higher, so the value serves either verb. Inside the batch step the budget keeps the work under the step's bound: a run starts only when the budget holds its wall limit plus 60 s, and the run with its Watcher rerun takes at most twice its wall limit plus 34 s, plus the executor's file copies (each sandboxed run ends by its wall limit plus 17 s, lassi/executors/sandbox.py:1496). So it ends about its wall limit minus 26 s past the budget's end, 874 s for the baseline, inside the 900 s the step allows.
- Disk: 30 to 350 MB in the run directory. That covers 14 build directories with host binaries (1 to 20 MB each, an estimate), seven JIT caches of 3 to 7 MB (P4.9 measured 2996 to 6260 KiB per example), the codec checkout (about 1 MB with .git), outputs under 0.2 MB, and capped logs. If the slot is new, rx adds its worktree and the slot's .venv, about 0.3 GB. A NumPy environment from uv's cache adds about 30 MB under the scratch cache if the host's python3 lacks NumPy. Total under 1 GB (OQ-050). The batch enforces the run directory's share: no build, run, or probe starts once the run directory holds more than 1 GiB.
- CPU and memory: one process at a time. clang++ compiles single-threaded. Each ttsim run sets TT_METAL_THREADCOUNT=16 under the sandbox's CPU-time cap for 16 CPUs, mostly 1 to 2 busy cores. Memory is limited to 8192 MiB per run; P4.9's examples peaked at 114 MiB resident.
- Network: one fetch of about 0.6 MB from github.com.

## Commands for the main session

Each step runs only with the owner's yes in the working session (Agent Rule 17). OQ-050 approved the search and the batch as it describes them; Owner queue lists what goes beyond that wording, the doctor runs, job polls, and pulls included. Use PYTHONIOENCODING=utf-8 on every rx call.

1. Commit plans/spikes/p5-tier-b.md and plans/spikes/p5-tier-b/ (P5.3's commit, after review), then check out that commit in the clean worktree C:/dev/lassi-clean (plans/LESSONS.md, alpha01).
2. The read-only search, one rx exec. It reads directory entries only. Its text names no device path, no Tenstorrent SMI tool, no gate directory, and no tt-metal path, so neither the gate's deny patterns nor its tt guard refuse it. It prunes LASSI's run and worktree trees, the toolchains root, caches, v0's trees, the ROCm and CUDA installs, .git, and build trees (CMakeFiles, .cpmcache, build_Release). Outside those prunes it searches /mnt/nvme10/joseph_ufl to depth 5, including the gate's directory and the shared tt-metal checkout (directory entries only), and /home/joseph_ufl when it exists and is not the scratch root:

```
PYTHONIOENCODING=utf-8 uv run tools/rx.py exec -- 'for r in /mnt/nvme10/joseph_ufl /home/joseph_ufl; do [ -d "$r" ] || continue; [ "$r" != /mnt/nvme10/joseph_ufl ] && [ "$(realpath "$r")" = /mnt/nvme10/joseph_ufl ] && continue; timeout -k 10 600 find "$r" -maxdepth 5 \( -name lassi-runs -o -name lassi-wt -o -name toolchains -o -name .cache -o -name amd -o -name cuda-12.6.3 -o -name corpus-alpha01 -o -name corpus-parquet -o -name mlir-corpus-pipeline -o -name .git -o -name CMakeFiles -o -name .cpmcache -o -name build_Release \) -prune -o \( -name turboquant_layout.h -o -name codebook_gen.py -o \( -type d -iname "*turboquant*" \) \) -print 2>/dev/null; done | head -n 60; echo "search done"'
```

   Checked on the workstation on 2026-10-10, with no host activity. The gate's own check_command (tools/server/gate.py:297-312, with tools/server/config.default.json's deny patterns and device patterns, and the tt guard made active by a stand-in device node) passes this text. It refuses the previous draft's text by the deny pattern /lassi-gate/ (config.default.json:22), and, with that term removed, by the tt guard (gate.py:309-312), which refuses any text naming tt-metal while the cards are present and tt_silicon is disabled. The command ran with bash -c, as the gate runs commands, on a stand-in tree mounted at /mnt/nvme10/joseph_ufl in a WSL user and mount namespace, with /home/joseph_ufl a symbolic link to it. It printed a codec clone's directory, its include/ and kernels/turboquant_layout.h, and its codebook_gen.py; a TurboQuant-named directory at depth 2 inside a stand-in tt-metal/ and the header in it; another at depth 5; one inside a stand-in lassi-gate/; then "search done", in 0.6 s. It printed nothing under the pruned names (lassi-runs, lassi-wt, toolchains, .cache, .git, build_Release, CMakeFiles) or at depth 6, and nothing twice through the home link.

   Expected output: "search done" alone when no copy exists. Each copy adds about 3 to 5 lines: its directory when TurboQuant-named, include/ and kernels/turboquant_layout.h, and codebook_gen.py. A TurboQuant-named directory in the tt-metal checkout or the gate's directory adds a line of its own. At most 61 lines with the head cap, plus rx's own lines. It finishes in seconds and is bounded at 610 s per root searched (one root when the home is the scratch root), under the gate's default timeout of 1800 s for rx exec.
3. Space and health: `PYTHONIOENCODING=utf-8 uv run tools/rx.py doctor`.
4. The batch, as an rx job, from C:/dev/lassi-clean at that commit:

```
PYTHONIOENCODING=utf-8 uv run tools/rx.py job start --name p53-tier-b --slot <slot> --timeout 7200 -- 'bash plans/spikes/p5-tier-b/run.sh'
```

   The slot: C:/dev/lassi-clean holds a detached commit, so without `--slot` rx names a new slot, <machine>-detached-<the commit's first 8 hex digits>, with its own worktree and .venv (about 0.3 GB), unless `--slot` names an existing one. No record in this repository names a p5-ir slot on alpha01 (P5's host steps so far were rx exec, which uses no slot), so the main session chooses: an existing slot from rx doctor's list, or a new one. The gate's check_command passes this text and the `--alpha01-copy` form below (checked as for step 2).

   Poll with `rx job wait <id> --timeout 600 --interval 300` and read the output with `rx job tail <id> -n 120`; the job id is the rx id, and the run directory is lassi-runs/p53-tier-b/<rx id>. The job's status is run.sh's (The batch). If the search found a copy and the owner approves its comparison, end the command with `-- 'bash plans/spikes/p5-tier-b/run.sh --alpha01-copy <path>'`. To stop the batch, use `rx exec -- 'touch /mnt/nvme10/joseph_ufl/lassi-runs/p53-tier-b/<rx id>/STOP'`, a write that needs its own yes, never rx job kill.
5. `PYTHONIOENCODING=utf-8 uv run tools/rx.py doctor` again.
6. Pulls, each into an empty .rx/pulls/<basename> first (plans/LESSONS.md):

```
PYTHONIOENCODING=utf-8 uv run tools/rx.py pull <rx id> --into results/p5-tier-b
PYTHONIOENCODING=utf-8 uv run tools/rx.py pull --path lassi-runs/p53-tier-b/<rx id>/report --into results/p5-tier-b
PYTHONIOENCODING=utf-8 uv run tools/rx.py pull --path lassi-runs/p53-tier-b/<rx id>/outputs
PYTHONIOENCODING=utf-8 uv run tools/rx.py pull --path lassi-runs/p53-tier-b/<rx id>/local
```

   Pull local/ only into the untracked .rx/pulls/local. If compare did not run on the host (report/batch.txt says "compare python: none"), run it locally on a directory that holds src/ (a local clone at 81817eb, checked out with core.autocrlf false, since compare checks reference/turboquant.py's sha256), outputs/ (the pull), and report/ (the pull): `uv run --no-project --with numpy==2.2.6 python -I -B plans/spikes/p5-tier-b/tier_b.py compare --run <dir>`, then `summarize`.

## Host results

PLACEHOLDER: the read-only search (step 2), with its rx id and output.

PLACEHOLDER: the batch (step 4), with its rx id, clean or dirty from the rx record, and the host's tool versions from report/batch.txt. Per program and variant:

| Program | Build asis | Build dev0 | Run asis (exit; kind) | Run dev0 (exit, hang, sim_ub, sim_gap, JIT errors, outputs) | Reference checks (dev0) | unpack_to_dest |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| mt-rotation | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| mt-polarquant | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| mt-inverse-rotation | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| mt-residual | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| mt-qjl-projection | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| mt-rnorm | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |

PLACEHOLDER: each failure's cause (an API change, a simulator finding class, or ttsim Facts), the device probe, the comparison with the committed dumps, the host checks, the scratch du before and after, and the simulator wall times (sizing only).

## Finding (draft)

- Sources: main at 81817eb has not moved since planning. A newer branch, john-changes (b78894f), changes only the baseline host and the launcher, so the kernels are identical on both. No license file exists, and the README points to a missing LICENSE.txt. Every commit on main is by the account JoeMad21.
- The bible's four stages map onto the codec as tabled above. WHT, PolarQuant, and QJL each have a baseline kernel and a multi_tile kernel set. dequant has a baseline kernel only, and no codec host program launches it. The multi_tile rewrite adds inverse rotation, residual, and r_norm as separate stages.
- No kernel holds tt-metal text beyond API idioms of at most three lines.
- API distance at the API level is nil: every tt-metal name the codec uses is declared at the pin with the same signature. The known semantic differences are the add and subtract fidelity in eltwise_binary.h, the tt-llk commits between the two submodule pins, and the JIT compiler's sfpi version.
- Every codec host program opens chip 6, which ttsim lacks. So as written none can run on ttsim (PREDICTION). The batch checks the kernels through a one-literal device-id change.
- PLACEHOLDER: build and run results, which kernels pass at the pin, and the comparisons.
- Confidence: high for the GitHub facts and the API table; medium-high for the text check; the run predictions as marked.

## Consequences for the plan

- P5.13 (after OQ-048):
  - Tier B's TT host programs must be written here in any holding, since the codec's hosts open chip 6. That matches holding (a)'s plan, so no extra work follows.
  - The kernel pin can stay 81817eb even if the owner prefers john-changes, since the kernel blobs are the same.
  - Several of the codec's bounds are not binary_io metrics, so P5.13 must map them or ask.
  - dequant needs data-movement kernels written here or stays out.
- P5.14: the raiser and spec coverage list in the P5.14 entry stands. The batch adds only the facts about which kernels JIT-compile and run.
- No task changes state until the host results arrive. A kernel that fails at the pin becomes evidence for OQ-048's failing-kernel option, not a task.

## Proposed bible edit

None at this task. P5.3's files are this report and the OQ-048 Response (plans/p5-ir.md, P5.3). The Tier B facts enter the bible with P5.13, after the owner answers OQ-048: Benchmark Suites, the Tier B row and TT tiers, with a Decision Log entry. If the main session wants the stage map recorded earlier, the candidate is one sentence in TT tiers, Tier B. It would say that the codec's multi_tile rewrite splits the encoder into six programs and has no dequant stage, that its dequant kernel has no host program, and that every codec host opens chip 6, which ttsim lacks; plans/spikes/p5-tier-b.md would be the reference.

## Owner queue

Before the batch runs (for the main session to put to the owner with the batch's summary; Agent Rule 17):

1. What OQ-050 approved for P5.3, summarized to the owner before launch under its option (a): one read-only search for an alpha01 copy of TurboQuant-Tenstorrent, and one batch from a clean commit that fetches the repository at 81817eb into the scratch root, builds "the codec's host programs" against the pinned tt-metal in the sandbox, runs them on ttsim on small inputs, and checks them with the codec's NumPy reference. Its closing note says any other server step asks again.
2. What goes beyond that wording, each needing the owner's yes:
   - The "dev0" variant: each host program is also built and run as a changed copy, its one device-id literal changed from 6 to 0 in the run directory only, since as written every host opens chip 6 and ttsim has one chip.
   - A NumPy environment: when the host's python3 lacks NumPy, uv may create a small one from its cache, with no download, under the scratch cache.
   - The job verb: OQ-050 names one `rx run`; AGENTS.md treats work that may pass about 20 minutes as an `rx job`. The batch is PROJECTED at 10 to 35 minutes (OQ-050 said 30 to 60) and bounded at 6680 s, under the job's timeout of 7200 s (PROJECTED costs). The job may also create a new slot (Commands, step 4).
   - The copy comparison, if the search finds a copy: whether the batch may compare it file by file, read only (`--alpha01-copy <path>`).
   - The steps around the batch that OQ-050 names for P5.2 only or not at all: rx doctor before and after, rx job wait and tail polls, the pulls, and a STOP touch if the batch must stop.
3. Devices. The Wormhole cards are on alpha01's PCI bus again, so the batch cannot refuse a host that has them, as P4.9's did. It runs every program only in the ttsim executor's sandbox, after a probe through that sandbox confirms that the sandbox's /dev holds only its own 13 entries and that /sys/bus, /sys/class, and every /sys/devices entry but system are empty. /proc/bus/pci stays readable in the sandbox (lassi/executors/sandbox.py:289-290), a known sandbox limit that lists PCI devices and opens none; the probe records it without gating on it. The gate's tt_silicon class stays disabled, and no command names a device.

Draft Response for OQ-048 (not yet in plans/OWNER-QUEUE.md; the main session adds it after the host results replace the PLACEHOLDERs):

```
Response (task P5.3, <date>): GitHub, read 2026-10-10: main is still at 81817eb09076d17b29491c6646b54c6927cac0bd (10 commits, 2026-05-13 to 2026-05-26, all by the account JoeMad21). A branch, john-changes (b78894f, 2 commits by another account, 2026-06-22 and 2026-07-27), changes only the baseline host and the launcher (a multi-device mesh open), so every kernel is the same blob at both commits. No license file; the README points to a LICENSE.txt and a docs/perf/ the tree lacks and describes a --cores flag no host at 81817eb parses. Stage map (plans/spikes/p5-tier-b.md): WHT, PolarQuant, and QJL each have a baseline kernel (scalar software float on data-movement cores; QJL fused with inverse rotation, residual, and r_norm) and a multi_tile kernel set (TRISC tile matmul or SFPU); dequant has only kernels/dequant_attn_kernel.cpp, which no codec host program launches and which needs feeder kernels the codec lacks; the multi_tile rewrite adds inverse rotation, residual, and r_norm as their own programs and has no dequant. reference/turboquant.py holds the oracles for WHT, PolarQuant, inverse rotation, and dequant; its QJLProjector draws a different S than the kernels' LFSR, which the codec's tests/test_qjl.py reproduces. No kernel holds tt-metal text beyond API idioms of at most three lines (0 comment lines shared). Every tt-metal name the codec uses is declared at 5280a9cf with the same signature as at 91f5a51; the differences found are add and subtract fidelity in eltwise_binary.h, 46 tt-llk commits, and the JIT compiler's sfpi version. Every codec host opens chip 6, which ttsim lacks, so the batch ran each host as written and with the id changed to 0. Results at the pin on ttsim v1.3.4 (rx <id>, commit <sha>, clean): PLACEHOLDER (builds, runs, reference checks, unpack_to_dest, failures and causes). Proposed items under option (c): tq-wht-baseline, tq-wht-multitile, tq-polarquant-baseline, tq-polarquant-multitile, tq-qjl-baseline, tq-inverse-rotation-multitile, tq-residual-multitile, tq-qjl-projection-multitile, tq-rnorm-multitile, with tq-dequant-baseline held until you say whether an item may carry data-movement kernels written here; minus any kernel that fails at the pin, under failing-kernel option (a). Several of the codec's own bounds are match rates or relative errors, which binary_io's pcc, max_abs, and ulp do not express; P5.13 maps them or asks. Holding: (a) needs no license; (b) needs a license or your written permission, and the text check found nothing that bars it. The recommendation stands as written (source (a) at 81817eb, items (c), holding (a), failing kernel (a)), with the map above for items (c).
```

## Sources

All read on 2026-10-10.

- Codec: https://github.com/JoeMad21/TurboQuant-Tenstorrent (API: repos/JoeMad21/TurboQuant-Tenstorrent, /branches, /commits?sha=main, /compare/main...john-changes, /git/trees/81817eb09076d17b29491c6646b54c6927cac0bd?recursive=1, /contributors), and a clone at 81817eb with the john-changes branch. Files read: README.md, HANDOFF.md, CMakeLists.txt, .gitignore, host/*.cpp, multi_tile/host/*.cpp, include/*.h, multi_tile/include/tile_layout.h, kernels/**/*.cpp, multi_tile/kernels/*.cpp, multi_tile/run_pipeline.sh, multi_tile/README.md, multi_wormhole/README.md, reference/turboquant.py, tests/test_rotation.py, tests/test_polarquant.py, tests/test_qjl.py, multi_tile/tests/*.py, and the committed .pyc header.
- tt-metal at https://github.com/tenstorrent/tt-metal/tree/91f5a51fb64cbf5f7ab58f1dd6b6b674c2672ebd and https://github.com/tenstorrent/tt-metal/tree/5280a9cfb00998fd49667a29523d03aee905c129: the sparse directories of Method, step 2; tt_metal/impl/data_format/tilize_utils.cpp at both; tt_metal/distributed/mesh_device.cpp at the pin.
- tt-llk: https://github.com/tenstorrent/tt-llk/compare/f990966829c2831bf2a58d8ce31779acbb476139...d090e3bf (46 commits; titles only).
- LASSI: docs/BIBLE.md (Benchmark Suites, Toolchain Pins, ttsim Facts, Execution Backends, Harness Contract, Environment State); plans/p5-ir.md; plans/OWNER-QUEUE.md (OQ-048, OQ-050); plans/spikes/p4-tt-pins.md; plans/spikes/p4-ttsim-runtime.md and its batch; results/p4-ttsim-runtime/runs.tsv; lassi/toolchains/ttmetal_build.py; lassi/executors/ttsim.py; lassi/executors/sandbox.py; tools/ttsim_smoke.py; toolchains/tt-metal.pin; plans/PHASE-NOTES.md (P4, P5); plans/LESSONS.md.
