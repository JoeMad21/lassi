// Written for LASSI-DF tt-par-v0 (task P17.68); a held-out item: row softmax of a bf16 matrix over the compute grid.
//
// Row softmax on the device: y[i, j] = exp(x[i, j] - m_i) / s_i, with m_i the largest value of row i and s_i the
// sum over k of exp(x[i, k] - m_i). The work unit is a tile row: 32 matrix rows, all cols / 32 of whose tiles a core
// keeps in its circular buffers at once. split_work_to_cores shares the tile rows out over the device's compute
// grid, and each working core gets its first tile row and its tile-row count as runtime args. On each core the
// reader kernel fills the reduce scaler tile once and then fetches one tile row at a time; the compute kernel takes
// each row's maximum (row reduce, MAX), subtracts it (column broadcast), takes exp, sums each row (row reduce, SUM),
// takes the reciprocal, and multiplies by it (column broadcast); the writer kernel stores the result tiles. Every
// output value comes from these kernels: the host only puts x into tile order and the result back into rows.
//
// Usage: row_softmax <x> <y>
//   x  lassi_io file, bf16, rank 2, [rows, cols]
//   y  lassi_io file written here, bf16, rank 2, [rows, cols], row-major
// rows must be a positive multiple of 32, at most 65536; cols a positive multiple of 32, at most 2048, so that one
// tile row (Wt = cols / 32 tiles, at most 64) and its intermediates fit one core's L1.
//
// Device settings, on every working core, each circular buffer Float16_b with 2048-byte tile pages:
//   c_0   2 Wt tiles  input, two tile rows deep, so the reader fetches the next tile row during compute
//   c_1   1 tile      the reduce scaler (1.0 in the first row of each face, zero elsewhere)
//   c_2   1 tile      each row's maximum, in column 0
//   c_3   Wt tiles    exp(x - max) for the whole tile row, read by the sum and again by the multiply
//   c_4   1 tile      each row's 1 / sum, in column 0
//   c_16  Wt tiles    output tiles for the writer
// That is 4 Wt + 3 tiles: at cols 256 (Wt 8), 35 tiles or 71,680 bytes (70 KiB) per core; 518 KiB at cols 2048.
// The reader runs on RISCV_1 and the writer on RISCV_0, each given Wt and then its DRAM buffer's TensorAccessorArgs
// as compile args. The compute kernel gets Wt as its one compile arg, math fidelity HiFi4, and math_approx_mode
// false; no other ComputeConfig field is set, so fp32_dest_acc_en stays off (dest holds 16-bit values) and
// unpack_to_dest_mode stays empty. Runtime args: reader and writer get their buffer's address, the first tile row,
// and the tile-row count; the compute kernel gets the tile-row count.
// A tt-metal error is an exception that ends the program through std::terminate.
#include <cstdint>   // the fixed-width integer types
#include <cstdio>    // std::fprintf for refusals
#include <cstring>   // std::memcpy of the input's bf16 patterns
#include <memory>    // std::shared_ptr handles of the mesh and its buffers
#include <tuple>     // std::get on split_work_to_cores' result
#include <utility>   // std::pair and std::move
#include <vector>    // host copies of x and y

#include <tt-metalium/work_split.hpp>            // split_work_to_cores
#include <tt-metalium/host_api.hpp>              // programs, kernels, circular buffers, runtime args
#include <tt-metalium/distributed.hpp>           // the unit mesh, its buffers, and its command queue
#include <tt-metalium/device.hpp>                // the device's compute grid
#include <tt-metalium/tensor_accessor_args.hpp>  // the DRAM layout args each data-movement kernel reads
#include <tt-metalium/tilize_utils.hpp>          // tilize_nfaces and untilize_nfaces
#include <tt-metalium/bfloat16.hpp>              // the element type those two work on

#include "lassi_io.h"

// tt::CBIndex and tt::DataFormat, and the tt::tt_metal host API, are used unqualified below.
using namespace tt::tt_metal;
using namespace tt;

namespace {

static_assert(sizeof(bfloat16) == sizeof(uint16_t), "a device bf16 has the size of a lassi_io bf16");

// Tile geometry: a 32 x 32 tile of bf16 values is one 2048-byte page in DRAM and in every circular buffer.
constexpr uint32_t kSide = 32;
constexpr uint32_t kPageBytes = kSide * kSide * static_cast<uint32_t>(sizeof(uint16_t));
// Shape limits; kMaxCols keeps 4 Wt + 3 tiles of circular buffers per core at or under 518 KiB.
constexpr uint64_t kMaxRows = 65536;
constexpr uint64_t kMaxCols = 2048;

// Where the build places the kernels, relative to the working directory.
constexpr const char* kReaderSource = "kernels/dataflow/row_softmax_reader.cpp";
constexpr const char* kComputeSource = "kernels/compute/row_softmax.cpp";
constexpr const char* kWriterSource = "kernels/dataflow/row_softmax_writer.cpp";

using DramBuffer = std::shared_ptr<distributed::MeshBuffer>;

// The matrix in tiles: tile_rows work units of tiles_per_row tiles each.
struct TileShape {
    uint32_t tile_rows;
    uint32_t tiles_per_row;
};

struct SoftmaxKernels {
    KernelHandle reader;
    KernelHandle compute;
    KernelHandle writer;
};

// Read the lassi_io file at `path` into *x and check it is a bf16 matrix within the limits above; return 0, or
// print the reason and return 1 with *x released.
int open_input(const char* path, lassi_io_array* x) {
    if (lassi_io_read(path, x) != 0) {
        return 1;
    }
    const char* reason = nullptr;
    if (x->dtype != LASSI_IO_BF16) {
        reason = "x must be bf16";
    } else if (x->rank != 2u) {
        reason = "x must be a matrix (rank 2)";
    } else if (x->dims[0] == 0u || x->dims[0] % kSide != 0u || x->dims[0] > kMaxRows) {
        reason = "the row count must be a positive multiple of 32, at most 65536";
    } else if (x->dims[1] == 0u || x->dims[1] % kSide != 0u || x->dims[1] > kMaxCols) {
        reason = "the column count must be a positive multiple of 32, at most 2048";
    }
    if (reason != nullptr) {
        std::fprintf(stderr, "row_softmax: %s: %s\n", path, reason);
        lassi_io_free(x);
        return 1;
    }
    return 0;
}

// A DRAM buffer of `pages` tile pages on the unit mesh.
DramBuffer tile_buffer(distributed::MeshDevice& mesh, uint32_t pages) {
    const distributed::DeviceLocalBufferConfig per_device{.page_size = kPageBytes, .buffer_type = BufferType::DRAM};
    const distributed::ReplicatedBufferConfig replicated{.size = static_cast<uint64_t>(kPageBytes) * pages};
    return distributed::MeshBuffer::create(replicated, per_device, &mesh);
}

// The six circular buffers of the table above, each its own Float16_b configuration, on `cores`.
void add_row_buffers(Program& program, const CoreRangeSet& cores, uint32_t tiles_per_row) {
    const std::pair<CBIndex, uint32_t> depths[] = {
        {CBIndex::c_0, 2u * tiles_per_row}, {CBIndex::c_1, 1u}, {CBIndex::c_2, 1u},
        {CBIndex::c_3, tiles_per_row},      {CBIndex::c_4, 1u}, {CBIndex::c_16, tiles_per_row},
    };
    for (const auto& [index, tiles] : depths) {
        const CircularBufferConfig config =
            CircularBufferConfig(tiles * kPageBytes, {{index, DataFormat::Float16_b}}).set_page_size(index, kPageBytes);
        CreateCircularBuffer(program, cores, config);
    }
}

// The reader, compute, and writer kernels on `cores`, with their compile args.
SoftmaxKernels add_row_kernels(
    Program& program, const CoreRangeSet& cores, const DramBuffer& x_dram, const DramBuffer& y_dram,
    uint32_t tiles_per_row) {
    std::vector<uint32_t> reader_compile = {tiles_per_row};
    TensorAccessorArgs(*x_dram).append_to(reader_compile);
    std::vector<uint32_t> writer_compile = {tiles_per_row};
    TensorAccessorArgs(*y_dram).append_to(writer_compile);
    SoftmaxKernels made{};
    made.reader = CreateKernel(
        program, kReaderSource, cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_compile});
    made.writer = CreateKernel(
        program, kWriterSource, cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_compile});
    made.compute = CreateKernel(
        program, kComputeSource, cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false, .compile_args = {tiles_per_row}});
    return made;
}

// Share the tile rows out over the compute grid, build the program on the working cores, and give each core its
// runtime args. split_work_to_cores puts every working core in one of two groups, each core of a group taking the
// same number of tile rows; the cores are walked group by group, range by range, and each takes the next run of
// tile rows, so every tile row goes to exactly one core.
Program build_program(
    distributed::MeshDevice& mesh, const DramBuffer& x_dram, const DramBuffer& y_dram, const TileShape& shape) {
    const auto split = split_work_to_cores(mesh.compute_with_storage_grid_size(), shape.tile_rows);
    const CoreRangeSet& working = std::get<1>(split);
    Program program = CreateProgram();
    add_row_buffers(program, working, shape.tiles_per_row);
    const SoftmaxKernels kernels = add_row_kernels(program, working, x_dram, y_dram, shape.tiles_per_row);
    const uint32_t x_address = static_cast<uint32_t>(x_dram->address());
    const uint32_t y_address = static_cast<uint32_t>(y_dram->address());
    const std::pair<const CoreRangeSet*, uint32_t> groups[] = {
        {&std::get<2>(split), std::get<4>(split)}, {&std::get<3>(split), std::get<5>(split)}};
    uint32_t next_tile_row = 0;
    for (const auto& [group, share] : groups) {
        for (const CoreRange& range : group->ranges()) {
            for (const CoreCoord& core : range) {
                SetRuntimeArgs(program, kernels.reader, core, {x_address, next_tile_row, share});
                SetRuntimeArgs(program, kernels.compute, core, {share});
                SetRuntimeArgs(program, kernels.writer, core, {y_address, next_tile_row, share});
                next_tile_row += share;
            }
        }
    }
    return program;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: row_softmax <x> <y>\n");
        return 2;
    }
    lassi_io_array x = {};
    if (open_input(argv[1], &x) != 0) {
        return 1;
    }
    const uint32_t rows = static_cast<uint32_t>(x.dims[0]);
    const uint32_t cols = static_cast<uint32_t>(x.dims[1]);
    const TileShape shape{rows / kSide, cols / kSide};
    const uint32_t pages = shape.tile_rows * shape.tiles_per_row;

    // x in the device's tile order: tile (r, c) of the matrix is page r * Wt + c, its four faces in order.
    std::vector<bfloat16> x_rows(static_cast<size_t>(rows) * cols);
    std::memcpy(x_rows.data(), x.data, x_rows.size() * sizeof(bfloat16));
    const std::vector<bfloat16> x_tiles = tilize_nfaces(x_rows, rows, cols);

    std::shared_ptr<distributed::MeshDevice> mesh = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& queue = mesh->mesh_command_queue();
    // Not const: the enqueue calls take the buffer handles by non-const reference.
    DramBuffer x_dram = tile_buffer(*mesh, pages);
    DramBuffer y_dram = tile_buffer(*mesh, pages);
    distributed::EnqueueWriteMeshBuffer(queue, x_dram, x_tiles, false);

    Program softmax_program = build_program(*mesh, x_dram, y_dram, shape);
    distributed::MeshWorkload job;
    job.add_program(distributed::MeshCoordinateRange(mesh->shape()), std::move(softmax_program));
    distributed::EnqueueMeshWorkload(queue, job, false);

    // The blocking read waits for the workload; the result comes back in tile order and is put into rows.
    std::vector<bfloat16> y_tiles(static_cast<size_t>(rows) * cols);
    distributed::EnqueueReadMeshBuffer(queue, y_tiles, y_dram, true);
    const std::vector<bfloat16> y_rows = untilize_nfaces(y_tiles, rows, cols);
    const bool closed = mesh->close();

    int status = 1;
    if (!closed) {
        std::fprintf(stderr, "row_softmax: the device did not close cleanly\n");
    } else {
        const uint64_t dims[2] = {x.dims[0], x.dims[1]};
        status = lassi_io_write(argv[2], "y", LASSI_IO_BF16, 2u, dims, y_rows.data());
    }
    lassi_io_free(&x);
    return status == 0 ? 0 : 1;
}
