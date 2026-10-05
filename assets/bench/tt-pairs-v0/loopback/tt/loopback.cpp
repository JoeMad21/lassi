// Written for LASSI-DF tt-pairs-v0 (task P4.13); runs the unmodified upstream loopback kernels from tt-metal 5280a9cf.
//
// Loopback: one data-movement kernel on core (0, 0) moves the input from one DRAM buffer to another, one
// 32x32 bf16 tile (2048 bytes) at a time through a one-tile L1 buffer. No compute kernel is created.
//
// Usage: loopback <input> <output>
//   input   lassi_io file, bf16, rank 1, a positive multiple of 1024 elements (whole tiles)
//   output  lassi_io file written here, bf16, rank 1, as many elements as the input
// The elements are kept in file order; nothing is converted, tilized, or checked against the input.
#include <cstdint>
#include <cstdio>
#include <memory>
#include <utility>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "lassi_io.h"

using namespace tt::tt_metal;

namespace {

// A tile holds 32 x 32 bf16 values; the kernel moves one tile of 2048 bytes per step.
constexpr uint32_t kTileValues = 32u * 32u;
constexpr uint32_t kTileBytes = kTileValues * static_cast<uint32_t>(sizeof(uint16_t));

}  // namespace

// Over 60 lines: the device read stays in main's own body, as PHASE-NOTES' P4.13 guard hint asks.
int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: loopback <input> <output>\n");
        return 2;
    }
    lassi_io_array src = {};
    if (lassi_io_read(argv[1], &src) != 0) {
        return 1;
    }
    if (src.dtype != LASSI_IO_BF16 || src.rank != 1u || src.count == 0u || src.count % kTileValues != 0u) {
        std::fprintf(stderr, "loopback: the input must be a rank-1 bf16 array of whole 32x32 tiles\n");
        lassi_io_free(&src);
        return 1;
    }
    const uint64_t count = src.count;
    const uint32_t tiles = static_cast<uint32_t>(count / kTileValues);

    // Stage the input's bf16 bit patterns for the device as they are.
    const uint16_t* src_values = static_cast<const uint16_t*>(src.data);
    std::vector<uint16_t> staged(src_values, src_values + count);

    std::shared_ptr<distributed::MeshDevice> device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = device->mesh_command_queue();

    // Two DRAM buffers of the whole array and one L1 buffer of a single tile, each paged by tile.
    distributed::DeviceLocalBufferConfig dram_pages{.page_size = kTileBytes, .buffer_type = BufferType::DRAM};
    distributed::DeviceLocalBufferConfig l1_pages{.page_size = kTileBytes, .buffer_type = BufferType::L1};
    distributed::ReplicatedBufferConfig whole_array{.size = static_cast<uint64_t>(kTileBytes) * tiles};
    distributed::ReplicatedBufferConfig one_tile{.size = kTileBytes};
    auto scratch_l1 = distributed::MeshBuffer::create(one_tile, l1_pages, device.get());
    auto in_dram = distributed::MeshBuffer::create(whole_array, dram_pages, device.get());
    auto out_dram = distributed::MeshBuffer::create(whole_array, dram_pages, device.get());

    // The kernel's compile-time arguments: the accessor arguments of the source buffer, then the destination's.
    std::vector<uint32_t> accessor_args;
    TensorAccessorArgs(*in_dram->get_backing_buffer()).append_to(accessor_args);
    TensorAccessorArgs(*out_dram->get_backing_buffer()).append_to(accessor_args);

    Program program = CreateProgram();
    const CoreCoord core = {0, 0};
    KernelHandle copier = CreateKernel(
        program,
        "loopback/kernels/loopback_dram_copy.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = accessor_args});

    distributed::EnqueueWriteMeshBuffer(cq, in_dram, staged, false);

    // Runtime arguments, in the kernel's order: L1 scratch address, source address, destination address, tiles.
    const std::vector<uint32_t> run_args = {
        static_cast<uint32_t>(scratch_l1->address()),
        static_cast<uint32_t>(in_dram->address()),
        static_cast<uint32_t>(out_dram->address()),
        tiles};
    SetRuntimeArgs(program, copier, core, run_args);

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    // Read the destination buffer back; the read sizes the vector to the buffer.
    std::vector<uint16_t> moved;
    distributed::EnqueueReadMeshBuffer(cq, moved, out_dram, true);
    if (!device->close()) {
        std::fprintf(stderr, "loopback: the device did not close cleanly\n");
        lassi_io_free(&src);
        return 1;
    }
    if (moved.size() != count) {
        std::fprintf(stderr, "loopback: the device returned a result of the wrong size\n");
        lassi_io_free(&src);
        return 1;
    }

    const uint64_t dims[1] = {count};
    const int status = lassi_io_write(argv[2], "output", LASSI_IO_BF16, 1u, dims, moved.data());
    lassi_io_free(&src);
    return status;
}
