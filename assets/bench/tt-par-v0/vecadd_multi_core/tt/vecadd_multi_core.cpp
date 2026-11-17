// Written for LASSI-DF tt-par-v0 (task P17.67); runs the unmodified upstream vecadd_multi_core kernels from tt-metal 5280a9cf.
//
// Adds two bf16 vectors element by element on the device, c = a + b, with the tiles shared out over the compute
// grid: split_work_to_cores gives each working core a run of consecutive tiles, its reader moves one tile of a and
// one of b at a time from DRAM into two circular buffers, its compute kernel adds each pair on the matrix unit, and
// its writer stores each sum tile in a third DRAM buffer. The host reads that buffer back, widens each bf16 sum to
// f32, and writes it.
//
// Usage: vecadd_multi_core <a> <b> <c>
//   a, b  lassi_io input files, each a bf16 array; both hold the same number of elements, a nonzero multiple of
//         1024 (one 32 x 32 tile)
//   c     the lassi_io output file: an f32 array named "c" with a's shape
// The arrays go to DRAM unchanged and are never tilized: an elementwise add does not depend on where an element sits
// inside a tile, so each run of 1024 elements fills one tile page as it is.
//
// Device settings, as the upstream example sets them except where noted: three circular buffers of four tiles each
// (c_0 and c_1 in, c_2 out), all Float16_b, on every working core (upstream places them on the whole grid, the same
// cores at 640 tiles); the reader on RISCV_0 with compile args [c_0, c_1, a's
// TensorAccessorArgs, b's TensorAccessorArgs] and named compile args c_0 and c_1; the writer on RISCV_1 with
// compile args [c_2, c's TensorAccessorArgs]; the compute kernel with math fidelity HiFi4, math_approx_mode false,
// compile args [c_0, c_1, c_2], and named compile args c_0, c_1, and c_2. No other ComputeConfig field is set, so
// fp32_dest_acc_en stays off and unpack_to_dest_mode stays empty. Each working core's runtime args are, for the
// reader, a's and b's addresses, its tile count, and its first tile; for the writer, c's address, its tile count,
// and its first tile; for the compute kernel, its tile count and its first tile.
// A tt-metal error is an exception that ends the program through std::terminate.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "lassi_io.h"

using namespace tt;
using namespace tt::tt_metal;

namespace {

// A tile is 32 x 32 elements; every element on the device is a 2-byte bf16, so a tile page takes 2048 bytes.
constexpr uint64_t kTileElements = 32u * 32u;
constexpr uint32_t kTileBytes = static_cast<uint32_t>(kTileElements * sizeof(uint16_t));
// Four tiles per circular buffer, as upstream sizes them, so a producer can run ahead of its consumer.
constexpr uint32_t kTilesPerCb = 4u;

// The upstream kernels, at the paths the build places them, relative to the working directory.
constexpr const char* kReaderKernel = "vecadd_multi_core/kernels/interleaved_tile_read_multi_core.cpp";
constexpr const char* kWriterKernel = "vecadd_multi_core/kernels/tile_write_multi_core.cpp";
constexpr const char* kComputeKernel = "vecadd_multi_core/kernels/add_multi_core.cpp";

using BufferPtr = std::shared_ptr<distributed::MeshBuffer>;

struct KernelIds {
    KernelHandle reader;
    KernelHandle writer;
    KernelHandle compute;
};

// The f32 value of a bf16 bit pattern: the pattern is the upper half of that f32.
float widen_bf16(uint16_t bits) {
    const uint32_t word = static_cast<uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &word, sizeof value);
    return value;
}

// Return how many tiles a and b hold, or 0 after saying on stderr why they cannot be added here.
uint32_t tile_count(const lassi_io_array& a, const lassi_io_array& b) {
    if (a.dtype != LASSI_IO_BF16 || b.dtype != LASSI_IO_BF16 || a.count != b.count) {
        std::fprintf(stderr, "vecadd_multi_core: a and b must be bf16 arrays with one element count\n");
        return 0;
    }
    if (a.count == 0u || a.count % kTileElements != 0u || a.count / kTileElements > UINT32_MAX / kTileBytes) {
        std::fprintf(stderr,
                     "vecadd_multi_core: the element count must be a nonzero multiple of 1024 that fits one DRAM "
                     "buffer\n");
        return 0;
    }
    return static_cast<uint32_t>(a.count / kTileElements);
}

// Create a DRAM buffer of `tiles` tiles, one tile per page.
BufferPtr dram_buffer(distributed::MeshDevice& device, uint32_t tiles) {
    distributed::DeviceLocalBufferConfig local{.page_size = kTileBytes, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig whole{.size = static_cast<uint64_t>(kTileBytes) * tiles};
    return distributed::MeshBuffer::create(whole, local, &device);
}

// Create the circular buffers and the three kernels on `cores`, with the compile-time args each kernel reads.
KernelIds create_kernels(
    Program& program, const CoreRangeSet& cores, const BufferPtr& a_buf, const BufferPtr& b_buf,
    const BufferPtr& c_buf) {
    for (const CBIndex index : {CBIndex::c_0, CBIndex::c_1, CBIndex::c_2}) {
        CreateCircularBuffer(
            program, cores,
            CircularBufferConfig(kTilesPerCb * kTileBytes, {{index, DataFormat::Float16_b}})
                .set_page_size(index, kTileBytes));
    }
    const uint32_t in0 = static_cast<uint32_t>(CBIndex::c_0);
    const uint32_t in1 = static_cast<uint32_t>(CBIndex::c_1);
    const uint32_t out0 = static_cast<uint32_t>(CBIndex::c_2);
    // The reader takes its first slot by index 0 and its second by the name "c_1"; a's accessor args start at
    // index 2 and b's follow them.
    std::vector<uint32_t> reader_args = {in0, in1};
    TensorAccessorArgs(*a_buf).append_to(reader_args);
    TensorAccessorArgs(*b_buf).append_to(reader_args);
    const std::unordered_map<std::string, uint32_t> reader_names = {{"c_0", in0}, {"c_1", in1}};
    // The writer takes its slot by index 0; c's accessor args start at index 1.
    std::vector<uint32_t> writer_args = {out0};
    TensorAccessorArgs(*c_buf).append_to(writer_args);
    // The compute kernel takes its slots by index 0, by the name "c_1", and by index 2.
    const std::vector<uint32_t> compute_args = {in0, in1, out0};
    const std::unordered_map<std::string, uint32_t> compute_names = {{"c_0", in0}, {"c_1", in1}, {"c_2", out0}};
    KernelIds ids{};
    ids.reader = CreateKernel(
        program, kReaderKernel, cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_args,
            .named_compile_args = reader_names});
    ids.writer = CreateKernel(
        program, kWriterKernel, cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = writer_args});
    ids.compute = CreateKernel(
        program, kComputeKernel, cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .math_approx_mode = false,
            .compile_args = compute_args,
            .named_compile_args = compute_names});
    return ids;
}

// Build the program: split the n_tiles tiles over the device's compute grid in row-major core order, as upstream
// does, create the circular buffers and kernels on the working cores, and give each core its runtime args. Cores
// are walked group 1 then group 2, range by range and core by core, and each takes the next `per_core` tiles from
// `first_tile` on, so every tile goes to exactly one core.
Program make_program(
    distributed::MeshDevice& device, const BufferPtr& a_buf, const BufferPtr& b_buf, const BufferPtr& c_buf,
    uint32_t n_tiles) {
    const auto split = split_work_to_cores(device.compute_with_storage_grid_size(), n_tiles, true);
    Program program = CreateProgram();
    const KernelIds ids = create_kernels(program, std::get<1>(split), a_buf, b_buf, c_buf);
    const uint32_t a_addr = static_cast<uint32_t>(a_buf->address());
    const uint32_t b_addr = static_cast<uint32_t>(b_buf->address());
    const uint32_t c_addr = static_cast<uint32_t>(c_buf->address());
    const std::pair<const CoreRangeSet*, uint32_t> groups[2] = {
        {&std::get<2>(split), std::get<4>(split)}, {&std::get<3>(split), std::get<5>(split)}};
    uint32_t first_tile = 0;
    for (const auto& group : groups) {
        const uint32_t per_core = group.second;
        for (const CoreRange& range : group.first->ranges()) {
            for (const CoreCoord& core : range) {
                SetRuntimeArgs(program, ids.reader, core, {a_addr, b_addr, per_core, first_tile});
                SetRuntimeArgs(program, ids.writer, core, {c_addr, per_core, first_tile});
                SetRuntimeArgs(program, ids.compute, core, {per_core, first_tile});
                first_tile += per_core;
            }
        }
    }
    return program;
}

}  // namespace

// Over 60 lines: the device read and output conversion stay in main's own body, as PHASE-NOTES' P4.13 guard hint asks.
int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: vecadd_multi_core <a> <b> <c>\n");
        return 2;
    }
    lassi_io_array a = {};
    lassi_io_array b = {};
    if (lassi_io_read(argv[1], &a) != 0 || lassi_io_read(argv[2], &b) != 0) {
        lassi_io_free(&a);
        lassi_io_free(&b);
        return 1;
    }
    const uint32_t n_tiles = tile_count(a, b);
    if (n_tiles == 0u) {
        lassi_io_free(&a);
        lassi_io_free(&b);
        return 1;
    }

    // Host copies of the inputs in the device's element format, bf16, which the files already hold.
    std::vector<uint16_t> a_host(a.count);
    std::vector<uint16_t> b_host(b.count);
    std::memcpy(a_host.data(), a.data, a.nbytes);
    std::memcpy(b_host.data(), b.data, b.nbytes);

    std::shared_ptr<distributed::MeshDevice> device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = device->mesh_command_queue();
    // Three DRAM buffers of n_tiles tile pages: the two inputs and the result, which is its own buffer.
    BufferPtr a_buf = dram_buffer(*device, n_tiles);
    BufferPtr b_buf = dram_buffer(*device, n_tiles);
    BufferPtr c_buf = dram_buffer(*device, n_tiles);
    distributed::EnqueueWriteMeshBuffer(cq, a_buf, a_host, false);
    distributed::EnqueueWriteMeshBuffer(cq, b_buf, b_host, false);

    distributed::MeshWorkload workload;
    Program program = make_program(*device, a_buf, b_buf, c_buf, n_tiles);
    workload.add_program(distributed::MeshCoordinateRange(device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    // Read the sums back from c's own buffer; the read sizes the vector to the buffer.
    std::vector<uint16_t> c_device;
    distributed::EnqueueReadMeshBuffer(cq, c_device, c_buf, true);
    const bool closed = device->close();

    int status = 1;
    if (!closed) {
        std::fprintf(stderr, "vecadd_multi_core: the device did not close cleanly\n");
    } else if (c_device.size() != a.count) {
        std::fprintf(stderr, "vecadd_multi_core: the device returned a result of the wrong size\n");
    } else {
        // Widen each bf16 sum to f32 for the output file; the widening is exact.
        std::vector<float> c(a.count);
        for (uint64_t i = 0; i < a.count; ++i) {
            c[i] = widen_bf16(c_device[i]);
        }
        status = lassi_io_write(argv[3], "c", LASSI_IO_F32, a.rank, a.dims, c.data());
    }
    lassi_io_free(&a);
    lassi_io_free(&b);
    return status == 0 ? 0 : 1;
}
