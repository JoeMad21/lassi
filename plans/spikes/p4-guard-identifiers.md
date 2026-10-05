# Spike: the CPU -> TT guard's identifier sets at the tt-metal pin (task P4.12)

Date: 2026-10-04. Host: alpha01, through read-only `rx exec`. Tree: `$LASSI_TOOLCHAINS/tt-metal@5280a9cf` (toolchains/tt-metal.pin).

Purpose: lassi.toolchains.ttmetal_guard names three identifier sets (KERNEL_CREATORS, COMPUTE_CONFIGS, DEVICE_READS) and cites the declaring header lines. The planning reads behind those lines (rx 20261001-054855-exec-6d60 and rx 20261001-054907-exec-a7d4) kept no output in the repository, so the declarations were read again here and the output is recorded below. [MEASURED] for the facts quoted; nothing here is a timing.

## Commands

rx 20261004-162204-exec-29e2:

```
T=$LASSI_TOOLCHAINS/tt-metal@5280a9cf; A=$T/tt_metal/api/tt-metalium; echo "## commit"; git -C $T rev-parse HEAD; cd $A || exit 1;
echo "## creators"; grep -nE "(CreateKernel|CreateKernelFromString)[[:space:]]*\(" host_api.hpp experimental/host_api.hpp | cut -c1-150 | head -12;
grep -nE "^[[:space:]]*(struct|class)[[:space:]]+(KernelDescriptor|ProgramDescriptor|ComputeConfigDescriptor)([[:space:]]|$|\{)" program_descriptors.hpp | head -6;
grep -nE "Program[[:space:]]*\([[:space:]]*const[[:space:]]+ProgramDescriptor" program.hpp | cut -c1-150 | head -4;
echo "## compute"; grep -nE "^[[:space:]]*(struct|class)[[:space:]]+ComputeConfig([[:space:]]|$|\{)" kernel_types.hpp | head -4;
echo "## reads"; grep -nE "(EnqueueReadMeshBuffer|ReadShard)[[:space:]]*\(" distributed.hpp | cut -c1-150 | head -8;
grep -nE "(ReadFromBuffer|ReadFromDeviceL1|ReadFromDeviceDRAMChannel|ReadRegFromDevice)[[:space:]]*\(" tt_metal.hpp | cut -c1-150 | head -16;
echo "## lightmetal"; grep -nE "LightMetalReplay" experimental/lightmetal/lightmetal_replay.hpp | cut -c1-150 | head -6;
echo "## other dirs naming the creators"; grep -rlE "(CreateKernel|CreateKernelFromString)[[:space:]]*\(" $T/tt_metal/api | head -8
```

rx 20261004-162216-exec-362c: `sed -n "96,140p" $LASSI_TOOLCHAINS/tt-metal@5280a9cf/tt_metal/api/tt-metalium/tt_metal.hpp | cut -c1-140`

rx 20261004-162228-exec-16ba: `grep -nE "(CreateKernel|CreateKernelFromString)[[:space:]]*\(" $LASSI_TOOLCHAINS/tt-metal@5280a9cf/tt_metal/api/tt-metalium/kernel_types.hpp | cut -c1-150 | head -6`

All three ended rc=0.

## Output

rx 20261004-162204-exec-29e2 (the lines after "## creators" without a file name are from program_descriptors.hpp, then program.hpp; under "## compute" from kernel_types.hpp; under "## reads" from distributed.hpp, then tt_metal.hpp):

```
## commit
5280a9cfb00998fd49667a29523d03aee905c129
## creators
host_api.hpp:164:KernelHandle CreateKernel(
host_api.hpp:184:KernelHandle CreateKernelFromString(
experimental/host_api.hpp:42:    //     CreateKernel(program, "kernel.cpp", core, QuasarDataMovementConfig{.compile_args = compile_args,
experimental/host_api.hpp:59:KernelHandle CreateKernel(
85:struct ComputeConfigDescriptor {
101:struct KernelDescriptor {
136:struct ProgramDescriptor {
29:    explicit Program(const ProgramDescriptor& descriptor);
## compute
76:struct ComputeConfig {
## reads
51:void ReadShard(
80:void EnqueueReadMeshBuffer(
100:void ReadFromBuffer(Buffer& buffer, uint8_t* host_buffer);
112:void ReadFromBuffer(Buffer& buffer, std::vector<DType>& host_buffer) {
116:    ReadFromBuffer(buffer, reinterpret_cast<uint8_t*>(host_buffer.data()));
119:void ReadFromBuffer(const std::shared_ptr<Buffer>& buffer, std::vector<DType>& host_buffer) {
120:    ReadFromBuffer(*buffer, host_buffer);
275:bool ReadFromDeviceDRAMChannel(IDevice* device, int dram_channel, uint32_t address, std::span<uint8_t> host_buffer);
292:bool ReadFromDeviceDRAMChannel(
351:bool ReadFromDeviceL1(
374:bool ReadFromDeviceL1(
382:bool ReadRegFromDevice(IDevice* device, const CoreCoord& logical_core, uint32_t address, uint32_t& regval);
## lightmetal
17:class LightMetalReplayImpl;
20:class LightMetalReplay {
23:    explicit LightMetalReplay(LightMetalBinary&& binary, IDevice* device = nullptr);
24:    LightMetalReplay(LightMetalReplay&&) noexcept;
25:    ~LightMetalReplay();
27:    LightMetalReplay(const LightMetalReplay&) = delete;
## other dirs naming the creators
/mnt/nvme10/joseph_ufl/toolchains/tt-metal@5280a9cf/tt_metal/api/tt-metalium/kernel_types.hpp
/mnt/nvme10/joseph_ufl/toolchains/tt-metal@5280a9cf/tt_metal/api/tt-metalium/host_api.hpp
/mnt/nvme10/joseph_ufl/toolchains/tt-metal@5280a9cf/tt_metal/api/tt-metalium/experimental/host_api.hpp
```

rx 20261004-162216-exec-362c, tt_metal.hpp lines 123-139: line 123, then lines 135-139 (lines 96-99 end WriteToBuffer, lines 100-121 hold the ReadFromBuffer declarations above, line 122 is blank, and lines 124-134 hold a documentation comment; all left out here):

```
void ReadShard(Buffer& buffer, uint8_t* host_buffer, const uint32_t& core_id);
template <typename DType>
void ReadShard(Buffer& buffer, std::vector<DType>& host_buffer, const uint32_t& core_id) {
    host_buffer.resize(buffer.page_size() * buffer.shard_spec().num_pages());
    ReadShard(buffer, reinterpret_cast<uint8_t*>(host_buffer.data()), core_id);
}
```

rx 20261004-162228-exec-16ba, kernel_types.hpp:

```
53:    //     CreateKernel(program, "kernel.cpp", core, DataMovementConfig{.compile_args = compile_args,
93:    //     CreateKernel(program, "kernel.cpp", core, ComputeConfig{.compile_args = compile_args, .named_compile_args =
114:    //     CreateKernel(program, "kernel.cpp", core, EthernetConfig{.compile_args = compile_args, .named_compile_args =
```

## Result

- The tree is at the pinned commit 5280a9cf.
- KERNEL_CREATORS: CreateKernel at host_api.hpp:164 and experimental/host_api.hpp:59 (the Quasar overload); CreateKernelFromString at host_api.hpp:184; KernelDescriptor at program_descriptors.hpp:101; ProgramDescriptor at program_descriptors.hpp:136, with `explicit Program(const ProgramDescriptor& descriptor)` at program.hpp:29. Under tt_metal/api, only host_api.hpp and experimental/host_api.hpp declare CreateKernel or CreateKernelFromString; kernel_types.hpp has `CreateKernel(`, the call or declaration form the grep looked for, only in comments (lines 53, 93, 114), as experimental/host_api.hpp:42 does.
- COMPUTE_CONFIGS: ComputeConfig at kernel_types.hpp:76; ComputeConfigDescriptor at program_descriptors.hpp:85.
- DEVICE_READS: EnqueueReadMeshBuffer at distributed.hpp:80; ReadShard at distributed.hpp:51 and, as a second pair of overloads, at tt_metal.hpp:123 (the declaration) and :136 (the template), lines 123-139 with its comment; ReadFromBuffer at tt_metal.hpp:100, 112, and 119 (lines 100-121); ReadFromDeviceDRAMChannel at tt_metal.hpp:275 and :292; ReadFromDeviceL1 at tt_metal.hpp:351 and :374; ReadRegFromDevice at tt_metal.hpp:382.
- Correction: the planning text cited ReadFromBuffer at "tt_metal.hpp:100-120 and :123-138". Lines 123-139 hold ReadShard's tt_metal.hpp overloads, not ReadFromBuffer; ReadShard is already in DEVICE_READS, so the set is unchanged and only the citation moves.
- LightMetalReplay is a class at experimental/lightmetal/lightmetal_replay.hpp:20 (constructor at :23). Whether it can create kernels inside a model program was not read; the guard names it only as a possible kernel creator outside KERNEL_CREATORS.
