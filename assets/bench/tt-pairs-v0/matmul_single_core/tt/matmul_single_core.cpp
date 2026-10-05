// Written for LASSI-DF tt-pairs-v0 (task P4.13); runs the unmodified upstream matmul_single_core kernels from tt-metal 5280a9cf.
//
// Single-core matrix multiply: C = A x B on core (0, 0). A reader kernel brings one 32x32 bf16 tile of A and
// one of B at a time from DRAM into two-tile circular buffers, the compute kernel multiplies and accumulates
// tiles along K with the matrix engine, and a writer kernel stores each finished tile of C back to DRAM.
// The host tilizes A and B before staging them and untilizes C after reading it back.
//
// Usage: matmul_single_core <a> <b> <c>
//   a  lassi_io file, bf16, shape [M, K]
//   b  lassi_io file, bf16, shape [K, N]
//   c  lassi_io file written here, bf16, shape [M, N]
// M, K, and N are positive multiples of 32 and at most 8192, and K equals N: the writer kernel reads its third
// runtime argument as the tile count of N, and the host passes the tile count of K there, as the upstream host
// does, so the tile counts agree only when K equals N. The result is written as the device returned it.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>

#include "lassi_io.h"

using namespace tt::tt_metal;

namespace {

// A tile is 32 x 32 bf16 values (2048 bytes); every DRAM buffer is paged by tile.
constexpr uint32_t kTileRows = tt::constants::TILE_HEIGHT;
constexpr uint32_t kTileCols = tt::constants::TILE_WIDTH;
constexpr uint32_t kTileBytes = static_cast<uint32_t>(sizeof(bfloat16)) * kTileRows * kTileCols;
// Each circular buffer holds two tiles, so a data-movement kernel can fill one while the other is in use.
constexpr uint32_t kTilesPerCircularBuffer = 2u;
// The largest dimension accepted; it keeps every byte count and runtime argument within 32 bits.
constexpr uint64_t kMaxDim = 8192u;

// The kernels, at the paths the build places them relative to the working directory.
constexpr const char* kReaderKernel = "matmul/matmul_single_core/kernels/dataflow/reader_single_core_mm.cpp";
constexpr const char* kWriterKernel = "matmul/matmul_single_core/kernels/dataflow/writer_single_core_mm.cpp";
constexpr const char* kComputeKernel = "matmul/matmul_single_core/kernels/compute/mm.cpp";

// Return why the inputs cannot be multiplied here, or nullptr when they can; only the arrays' headers are read.
const char* shape_problem(const lassi_io_array& a, const lassi_io_array& b) {
    if (a.dtype != LASSI_IO_BF16 || b.dtype != LASSI_IO_BF16 || a.rank != 2u || b.rank != 2u) {
        return "a and b must be rank-2 bf16 arrays";
    }
    if (a.dims[1] != b.dims[0]) {
        return "the columns of a must equal the rows of b";
    }
    const uint64_t dims[3] = {a.dims[0], a.dims[1], b.dims[1]};
    for (const uint64_t dim : dims) {
        if (dim == 0u || dim % kTileRows != 0u || dim > kMaxDim) {
            return "every dimension must be a positive multiple of 32 and at most 8192";
        }
    }
    if (b.dims[0] != b.dims[1]) {
        return "b must be square (K equal to N), as the kernels' runtime arguments require";
    }
    return nullptr;
}

// Create a DRAM buffer of `bytes` bytes, replicated over the mesh and paged by tile.
std::shared_ptr<distributed::MeshBuffer> dram_buffer(distributed::MeshDevice* mesh, uint64_t bytes) {
    const distributed::DeviceLocalBufferConfig pages{.page_size = kTileBytes, .buffer_type = BufferType::DRAM};
    const distributed::ReplicatedBufferConfig whole{.size = bytes};
    return distributed::MeshBuffer::create(whole, pages, mesh);
}

// Create the bf16 (Float16_b) circular buffer `index` on `core`, two tiles deep and paged by tile.
void add_circular_buffer(Program& program, const CoreCoord& core, uint32_t index) {
    const CircularBufferConfig config =
        CircularBufferConfig(kTilesPerCircularBuffer * kTileBytes, {{index, tt::DataFormat::Float16_b}})
            .set_page_size(index, kTileBytes);
    CreateCircularBuffer(program, core, config);
}

// Create the reader, writer, and compute kernels on `core` with the upstream example's arguments.
// The tile counts are mt (rows of A and C), kt (the shared dimension), and nt (columns of B and C).
void add_kernels(
    Program& program,
    const CoreCoord& core,
    const std::shared_ptr<distributed::MeshBuffer>& a_dram,
    const std::shared_ptr<distributed::MeshBuffer>& b_dram,
    const std::shared_ptr<distributed::MeshBuffer>& c_dram,
    uint32_t mt,
    uint32_t kt,
    uint32_t nt) {
    // Compile-time arguments: the accessor arguments of A's buffer, then B's, for the reader; C's for the writer.
    std::vector<uint32_t> reader_accessors;
    TensorAccessorArgs(*a_dram).append_to(reader_accessors);
    TensorAccessorArgs(*b_dram).append_to(reader_accessors);
    std::vector<uint32_t> writer_accessors;
    TensorAccessorArgs(*c_dram).append_to(writer_accessors);

    const KernelHandle reader = CreateKernel(
        program,
        kReaderKernel,
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_accessors});
    const KernelHandle writer = CreateKernel(
        program,
        kWriterKernel,
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_accessors});
    // The compute kernel takes the tile counts at compile time and no runtime argument.
    CreateKernel(
        program,
        kComputeKernel,
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = {mt, kt, nt}});

    // Runtime arguments in the upstream host's order: the reader gets both input addresses and the tile counts,
    // the writer the output address and the same three counts.
    const uint32_t a_addr = static_cast<uint32_t>(a_dram->address());
    const uint32_t b_addr = static_cast<uint32_t>(b_dram->address());
    const uint32_t c_addr = static_cast<uint32_t>(c_dram->address());
    SetRuntimeArgs(program, reader, core, {a_addr, b_addr, mt, kt, nt});
    SetRuntimeArgs(program, writer, core, {c_addr, mt, kt, nt});
}

}  // namespace

// Over 60 lines: the device read and output conversion stay in main's own body, as PHASE-NOTES' P4.13 guard hint asks.
int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: matmul_single_core <a> <b> <c>\n");
        return 2;
    }
    lassi_io_array a = {};
    lassi_io_array b = {};
    if (lassi_io_read(argv[1], &a) != 0 || lassi_io_read(argv[2], &b) != 0) {
        lassi_io_free(&a);
        return 1;
    }
    const char* problem = shape_problem(a, b);
    if (problem != nullptr) {
        std::fprintf(stderr, "matmul_single_core: %s\n", problem);
        lassi_io_free(&a);
        lassi_io_free(&b);
        return 1;
    }
    const uint32_t m = static_cast<uint32_t>(a.dims[0]);
    const uint32_t k = static_cast<uint32_t>(a.dims[1]);
    const uint32_t n = static_cast<uint32_t>(b.dims[1]);

    std::shared_ptr<distributed::MeshDevice> mesh = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = mesh->mesh_command_queue();
    auto a_dram = dram_buffer(mesh.get(), uint64_t{sizeof(bfloat16)} * a.count);
    auto b_dram = dram_buffer(mesh.get(), uint64_t{sizeof(bfloat16)} * b.count);
    auto c_dram = dram_buffer(mesh.get(), uint64_t{sizeof(bfloat16)} * m * n);

    Program program = CreateProgram();
    const CoreCoord core = {0, 0};
    add_circular_buffer(program, core, tt::CBIndex::c_0);
    add_circular_buffer(program, core, tt::CBIndex::c_1);
    add_circular_buffer(program, core, tt::CBIndex::c_16);
    add_kernels(program, core, a_dram, b_dram, c_dram, m / kTileRows, k / kTileCols, n / kTileCols);

    // Stage the inputs: copy each file's bf16 bit patterns as they are, then reorder them into 32x32 tiles.
    std::vector<bfloat16> a_rows(a.count);
    std::memcpy(a_rows.data(), a.data, a.nbytes);
    std::vector<bfloat16> b_rows(b.count);
    std::memcpy(b_rows.data(), b.data, b.nbytes);
    std::vector<bfloat16> a_tiles = tilize_nfaces(a_rows, m, k);
    std::vector<bfloat16> b_tiles = tilize_nfaces(b_rows, k, n);
    distributed::EnqueueWriteMeshBuffer(cq, a_dram, a_tiles, false);
    distributed::EnqueueWriteMeshBuffer(cq, b_dram, b_tiles, false);

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);

    // Read C back in tile order (the read sizes the vector to the buffer), then restore row-major order.
    std::vector<bfloat16> c_tiles;
    distributed::EnqueueReadMeshBuffer(cq, c_tiles, c_dram, true);
    if (!mesh->close()) {
        std::fprintf(stderr, "matmul_single_core: the device did not close cleanly\n");
        lassi_io_free(&a);
        lassi_io_free(&b);
        return 1;
    }
    const std::vector<bfloat16> c_rows = untilize_nfaces(c_tiles, m, n);

    const uint64_t c_dims[2] = {a.dims[0], b.dims[1]};
    const int status = lassi_io_write(argv[3], "c", LASSI_IO_BF16, 2u, c_dims, c_rows.data());
    lassi_io_free(&a);
    lassi_io_free(&b);
    return status;
}
