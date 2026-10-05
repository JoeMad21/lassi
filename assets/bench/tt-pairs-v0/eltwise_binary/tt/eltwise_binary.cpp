// Written for LASSI-DF tt-pairs-v0 (task P4.13); runs the unmodified upstream eltwise_binary kernels from tt-metal 5280a9cf.
//
// Adds two bf16 arrays element by element on one Tensix core: c = a + b.
// Usage: eltwise_binary <a> <b> <c>
//   a, b  lassi_io input files, each a bf16 array; both hold the same number of
//         elements, a nonzero multiple of 1024 (one 32 x 32 tile)
//   c     the lassi_io output file: an f32 array named "c" with a's shape
// The arrays go to DRAM unchanged, each run of 1024 elements filling one tile
// page. A reader kernel moves one page of a and one of b at a time into two
// circular buffers, a compute kernel adds them on the matrix unit, and a
// writer kernel stores each sum page in a third DRAM buffer. The host reads
// that buffer back, widens each bf16 sum to f32, and writes it.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "lassi_io.h"

using namespace tt::tt_metal;

namespace {

// A tile is 32 x 32 elements; every element on the device is a 2-byte bf16.
constexpr uint64_t kTileElements = 32u * 32u;
constexpr uint32_t kTileBytes = static_cast<uint32_t>(kTileElements * sizeof(uint16_t));
// Two tiles per circular buffer let a producer fill one slot while the consumer reads the other.
constexpr uint32_t kTilesPerCb = 2u;

// The f32 value of a bf16 bit pattern: the pattern is the upper half of that f32.
float widen_bf16(uint16_t bits) {
    const uint32_t word = static_cast<uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &word, sizeof value);
    return value;
}

// Give core a circular buffer at slot index holding kTilesPerCb Float16_b tiles, one tile per page.
void make_tile_cb(Program& program, const CoreCoord& core, tt::CBIndex index) {
    CircularBufferConfig config(kTilesPerCb * kTileBytes, {{index, tt::DataFormat::Float16_b}});
    config.set_page_size(index, kTileBytes);
    CreateCircularBuffer(program, core, config);
}

}  // namespace

// Over 60 lines: the device read and output conversion stay in main's own body, as PHASE-NOTES' P4.13 guard hint asks.
int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: eltwise_binary <a> <b> <c>\n");
        return 2;
    }
    lassi_io_array a = {};
    lassi_io_array b = {};
    if (lassi_io_read(argv[1], &a) != 0 || lassi_io_read(argv[2], &b) != 0) {
        lassi_io_free(&a);
        lassi_io_free(&b);
        return 1;
    }
    if (a.dtype != LASSI_IO_BF16 || b.dtype != LASSI_IO_BF16 || a.count != b.count || a.count == 0u ||
        a.count % kTileElements != 0u || a.count / kTileElements > UINT32_MAX) {
        std::fprintf(stderr,
                     "eltwise_binary: a and b must be bf16 arrays with one element count, a nonzero multiple of "
                     "1024\n");
        lassi_io_free(&a);
        lassi_io_free(&b);
        return 1;
    }
    const uint32_t n_tiles = static_cast<uint32_t>(a.count / kTileElements);

    // Host copies of the inputs in the device's element format, bf16, which the files already hold.
    std::vector<uint16_t> a_host(a.count);
    std::vector<uint16_t> b_host(b.count);
    std::memcpy(a_host.data(), a.data, a.nbytes);
    std::memcpy(b_host.data(), b.data, b.nbytes);

    std::shared_ptr<distributed::MeshDevice> device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& queue = device->mesh_command_queue();

    // Three DRAM buffers of n_tiles tile pages: two inputs and the result, which is its own buffer.
    distributed::DeviceLocalBufferConfig page_layout{.page_size = kTileBytes, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_size{.size = static_cast<uint64_t>(n_tiles) * kTileBytes};
    auto a_dram = distributed::MeshBuffer::create(buffer_size, page_layout, device.get());
    auto b_dram = distributed::MeshBuffer::create(buffer_size, page_layout, device.get());
    auto c_dram = distributed::MeshBuffer::create(buffer_size, page_layout, device.get());
    distributed::EnqueueWriteMeshBuffer(queue, a_dram, a_host, false);
    distributed::EnqueueWriteMeshBuffer(queue, b_dram, b_host, false);

    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};
    // Slots 0 and 1 carry a and b to the compute kernel; slot 16 carries the sums to the writer.
    make_tile_cb(program, core, tt::CBIndex::c_0);
    make_tile_cb(program, core, tt::CBIndex::c_1);
    make_tile_cb(program, core, tt::CBIndex::c_16);

    // Compile-time arguments: the accessor layout of a then b for the reader, of c for the writer.
    std::vector<uint32_t> reader_layout;
    TensorAccessorArgs(*a_dram).append_to(reader_layout);
    TensorAccessorArgs(*b_dram).append_to(reader_layout);
    std::vector<uint32_t> writer_layout;
    TensorAccessorArgs(*c_dram).append_to(writer_layout);

    KernelHandle reader = CreateKernel(
        program, "eltwise_binary/kernels/dataflow/read_tiles.cpp", core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_layout});
    KernelHandle writer = CreateKernel(
        program, "eltwise_binary/kernels/dataflow/write_tile.cpp", core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = writer_layout});
    KernelHandle adder = CreateKernel(
        program, "eltwise_binary/kernels/compute/tiles_add.cpp", core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4});

    // Runtime arguments: DRAM addresses and the tile count, in the order each kernel reads them.
    const uint32_t a_addr = static_cast<uint32_t>(a_dram->address());
    const uint32_t b_addr = static_cast<uint32_t>(b_dram->address());
    const uint32_t c_addr = static_cast<uint32_t>(c_dram->address());
    SetRuntimeArgs(program, reader, core, {a_addr, b_addr, n_tiles});
    SetRuntimeArgs(program, writer, core, {c_addr, n_tiles});
    SetRuntimeArgs(program, adder, core, {n_tiles});

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(queue, workload, false);
    distributed::Finish(queue);

    // Read the sums back from c's buffer and widen them to f32 for the output file.
    std::vector<uint16_t> c_device;
    distributed::EnqueueReadMeshBuffer(queue, c_device, c_dram, true);
    std::vector<float> c(a.count);
    for (uint64_t i = 0; i < a.count; ++i) {
        c[i] = widen_bf16(c_device[i]);
    }

    if (!device->close()) {
        std::fprintf(stderr, "eltwise_binary: the device did not close cleanly\n");
        lassi_io_free(&a);
        lassi_io_free(&b);
        return 1;
    }
    const int status = lassi_io_write(argv[3], "c", LASSI_IO_F32, a.rank, a.dims, c.data());
    lassi_io_free(&a);
    lassi_io_free(&b);
    return status == 0 ? 0 : 1;
}
