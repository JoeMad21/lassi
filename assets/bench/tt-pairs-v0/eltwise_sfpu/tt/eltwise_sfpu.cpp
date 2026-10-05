// Written for LASSI-DF tt-pairs-v0 (task P4.13); runs the unmodified upstream eltwise_sfpu kernels from tt-metal 5280a9cf.
//
// Usage: eltwise_sfpu <src0 file> <result file>
//
// src0 is a bf16 lassi_io array whose element count is a whole number of 32 x 32 tiles. The program copies
// it to a DRAM buffer and runs three kernels on Tensix core (0, 0): a reader that moves one tile at a time
// into circular buffer c_0, a compute kernel that takes exp of each tile on the SFPU and packs it into
// c_16, and a writer that moves each tile of c_16 to a second DRAM buffer. It reads that buffer back and
// writes it unchanged as the bf16 array "result", with src0's shape. It computes no golden value and
// checks nothing. Both circular buffers are Float16_b, and the compute config sets neither
// fp32_dest_acc_en nor unpack_to_dest_mode, so the kernels never take the Float32 unpack-to-dest path.
#include <cstdint>
#include <cstdio>
#include <memory>
#include <utility>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "lassi_io.h"

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr uint32_t kTileElements = constants::TILE_WIDTH * constants::TILE_HEIGHT;
// A bf16 element takes two bytes, so one tile, the page of both DRAM buffers, takes 2048.
constexpr uint32_t kTileBytes = static_cast<uint32_t>(sizeof(uint16_t)) * kTileElements;
// Two tiles per circular buffer, so one side can fill a tile while the other drains the previous one.
constexpr uint32_t kTilesPerCircularBuffer = 2;
constexpr uint32_t kInputCb = CBIndex::c_0;
constexpr uint32_t kOutputCb = CBIndex::c_16;

struct Kernels {
    KernelHandle reader;
    KernelHandle writer;
    KernelHandle compute;
};

// Return how many tiles src0 holds, or 0 after saying on stderr why it cannot run.
uint32_t tile_count(const lassi_io_array& src0) {
    if (src0.dtype != LASSI_IO_BF16) {
        std::fprintf(stderr, "eltwise_sfpu: src0 is not a bf16 array\n");
        return 0;
    }
    if (src0.count == 0 || src0.count % kTileElements != 0) {
        std::fprintf(stderr, "eltwise_sfpu: src0 does not hold a whole, nonzero number of 32 x 32 tiles\n");
        return 0;
    }
    const uint64_t tiles = src0.count / kTileElements;
    if (tiles > UINT32_MAX / kTileBytes) {
        std::fprintf(stderr, "eltwise_sfpu: src0 is too large for one DRAM buffer\n");
        return 0;
    }
    return static_cast<uint32_t>(tiles);
}

// Add a Float16_b circular buffer of kTilesPerCircularBuffer tile pages at `index` on `core`.
void add_circular_buffer(Program& program, const CoreCoord& core, uint32_t index) {
    CircularBufferConfig config =
        CircularBufferConfig(kTilesPerCircularBuffer * kTileBytes, {{index, DataFormat::Float16_b}})
            .set_page_size(index, kTileBytes);
    CreateCircularBuffer(program, core, config);
}

// Create the three kernels on `core`; each data-movement kernel gets its DRAM buffer's accessor as compile args.
Kernels add_kernels(Program& program, const CoreCoord& core, distributed::MeshBuffer& src,
                    distributed::MeshBuffer& dst) {
    std::vector<uint32_t> reader_compile_args;
    TensorAccessorArgs(src).append_to(reader_compile_args);
    std::vector<uint32_t> writer_compile_args;
    TensorAccessorArgs(dst).append_to(writer_compile_args);
    Kernels kernels{};
    kernels.reader = CreateKernel(
        program, "kernels/dataflow/read_tile.cpp", core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_args});
    kernels.writer = CreateKernel(
        program, "kernels/dataflow/write_tile.cpp", core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_args});
    kernels.compute = CreateKernel(
        program, "kernels/compute/eltwise_sfpu.cpp", core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false});
    return kernels;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: eltwise_sfpu <src0 file> <result file>\n");
        return 2;
    }
    lassi_io_array src0 = {};
    if (lassi_io_read(argv[1], &src0) != 0) {
        return 1;
    }
    const uint32_t n_tiles = tile_count(src0);
    if (n_tiles == 0) {
        lassi_io_free(&src0);
        return 1;
    }
    // The device takes the bf16 bit patterns as the file holds them.
    const uint16_t* src0_bits = static_cast<const uint16_t*>(src0.data);
    std::vector<uint16_t> staged(src0_bits, src0_bits + src0.count);

    std::shared_ptr<distributed::MeshDevice> device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = device->mesh_command_queue();
    distributed::DeviceLocalBufferConfig dram{.page_size = kTileBytes, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig whole{.size = static_cast<uint64_t>(kTileBytes) * n_tiles};
    std::shared_ptr<distributed::MeshBuffer> src_buffer = distributed::MeshBuffer::create(whole, dram, device.get());
    std::shared_ptr<distributed::MeshBuffer> dst_buffer = distributed::MeshBuffer::create(whole, dram, device.get());

    Program program = CreateProgram();
    const CoreCoord core = {0, 0};
    add_circular_buffer(program, core, kInputCb);
    add_circular_buffer(program, core, kOutputCb);
    const Kernels kernels = add_kernels(program, core, *src_buffer, *dst_buffer);

    distributed::EnqueueWriteMeshBuffer(cq, src_buffer, staged, false);
    SetRuntimeArgs(program, kernels.compute, core, {n_tiles});
    SetRuntimeArgs(program, kernels.reader, core, {static_cast<uint32_t>(src_buffer->address()), n_tiles});
    SetRuntimeArgs(program, kernels.writer, core, {static_cast<uint32_t>(dst_buffer->address()), n_tiles});

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    // The device's result goes to the output file bit for bit; the host computes nothing on it.
    std::vector<uint16_t> result;
    distributed::EnqueueReadMeshBuffer(cq, result, dst_buffer, true);
    const bool closed = device->close();
    if (!closed || result.size() != src0.count) {
        std::fprintf(stderr, "eltwise_sfpu: the device did not close cleanly or returned the wrong element count\n");
        lassi_io_free(&src0);
        return 1;
    }
    const int status = lassi_io_write(argv[2], "result", LASSI_IO_BF16, src0.rank, src0.dims, result.data());
    lassi_io_free(&src0);
    return status == 0 ? 0 : 1;
}
