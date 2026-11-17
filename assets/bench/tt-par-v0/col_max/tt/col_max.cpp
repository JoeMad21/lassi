// Written for LASSI-DF tt-par-v0 (task P17.68); a held-out item: the maximum of each column of a bf16 matrix, spread over the compute grid.
//
// What it computes, on the device: m[j] = the maximum over i of a[i, j], for a bf16 matrix a of rows x cols.
//
// How the work is spread: the unit of work is a tile column, the 32 columns of a covered by one column of 32 x 32
// tiles (rows / 32 tiles). split_work_to_cores deals the cols / 32 tile columns out over the device's compute grid,
// each working core taking a run of consecutive tile columns, given to its kernels as runtime args (its first tile
// column and how many it owns). On each working core:
//   - the reader (kernels/dataflow/read_tile_columns.cpp, RISCV_1) fills circular buffer c_1 once with a scaler
//     tile (bf16 1.0 in the first row of each 16 x 16 face, zero elsewhere), then streams the tiles of its tile
//     columns from DRAM into c_0, one tile at a time, top to bottom down each tile column;
//   - the compute kernel (kernels/compute/max_tile_columns.cpp) reduces every tile of a tile column along its
//     columns into one Dst tile (reduce_tile, MAX over REDUCE_COL), whose first row then holds the 32 column maxima,
//     and packs it into c_16;
//   - the writer (kernels/dataflow/write_column_maxima.cpp, RISCV_0) stores that tile in DRAM at its tile column's
//     page.
// The host reads the result pages back, lays them out as a 32 x cols row-major matrix, takes its first row, and
// writes it as m. Each maximum is one of a's own bf16 values; no value of m is computed on the host.
//
// Usage: col_max <a> <m>
//   a  lassi_io file, bf16, rank 2, [rows, cols]: rows and cols positive multiples of 32, at most 65536 each
//   m  lassi_io file written here, bf16, rank 1, [cols]
//
// Device settings: three circular buffers on every working core, all Float16_b with one tile per page: c_0 (two
// pages of a), c_1 (one page, the scaler), and c_16 (two result pages). The reader and the writer each get their
// DRAM buffer's TensorAccessorArgs as compile args. The compute kernel's ComputeConfig sets only math fidelity
// HiFi4; every other field keeps its default, so there are no compile args, fp32_dest_acc_en stays off (a 16-bit
// Dst, which holds a bf16 value exactly), and unpack_to_dest_mode stays empty.
// A tt-metal error is an exception that ends the program through std::terminate.
#include "lassi_io.h"

// Standard library: fixed-width integers and messages on stderr,
#include <cstdint>
#include <cstdio>
// byte copies and shared ownership of the device and its buffers,
#include <cstring>
#include <memory>
// the work split's tuple and moving the program into its workload,
#include <tuple>
#include <utility>
// and host-side arrays.
#include <vector>

// tt-metal host API: the bf16 type and the tile edge,
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
// the device, its mesh buffers, workloads, and command queue,
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
// programs, circular buffers, kernels, and runtime args,
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
// and tile order on the host and the work split over the grid.
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt;
using namespace tt::tt_metal;
// TILE_HEIGHT and TILE_WIDTH come from here.
using namespace tt::constants;

namespace {

static_assert(sizeof(bfloat16) == sizeof(uint16_t), "col_max moves bf16 bit patterns through bfloat16 unchanged");

// One tile of bf16: 32 x 32 values in 2048 bytes, the page size of both DRAM buffers and of every circular buffer.
constexpr uint32_t kTileValueCount = TILE_HEIGHT * TILE_WIDTH;
constexpr uint32_t kTileByteCount = kTileValueCount * static_cast<uint32_t>(sizeof(bfloat16));
// The largest row or column count accepted. Tile counts (at most 2048) and their products (at most 2^22) then fit
// the uint32 runtime args; buffer sizes in bytes are computed in 64 bits.
constexpr uint64_t kSideLimit = 65536;
// Pages per circular buffer: two where a tile streams through, so the producer can work one tile ahead, and one
// for the scaler, which is written once.
constexpr uint32_t kStreamDepth = 2;
constexpr uint32_t kScaleDepth = 1;

// The kernels, at the paths this program's build places them, relative to the working directory.
constexpr const char* kReadTileColumns = "kernels/dataflow/read_tile_columns.cpp";
constexpr const char* kMaxTileColumns = "kernels/compute/max_tile_columns.cpp";
constexpr const char* kWriteColumnMaxima = "kernels/dataflow/write_column_maxima.cpp";

using TileBuffer = std::shared_ptr<distributed::MeshBuffer>;

// The tile counts of a: tiles down one tile column, and tile columns (the units of work).
struct TileShape {
    uint32_t tiles_down;
    uint32_t tile_columns;
};

struct ColMaxKernels {
    KernelHandle reader;
    KernelHandle reducer;
    KernelHandle writer;
};

// Read the lassi_io file at path into *a and check that it holds a bf16 matrix whose two sides are positive
// multiples of 32, at most kSideLimit; return true, or say why on stderr, release *a, and return false.
bool load_usable_a(const char* path, lassi_io_array* a) {
    if (lassi_io_read(path, a) != 0) {
        return false;
    }
    const char* refusal = nullptr;
    if (a->dtype != LASSI_IO_BF16) {
        refusal = "a must be bf16";
    } else if (a->rank != 2u) {
        refusal = "a must have rank 2";
    } else {
        for (uint32_t side = 0; side < 2u; ++side) {
            const uint64_t extent = a->dims[side];
            if (extent == 0u || extent % TILE_WIDTH != 0u || extent > kSideLimit) {
                refusal = "each side of a must be a positive multiple of 32, at most 65536";
            }
        }
    }
    if (refusal == nullptr) {
        return true;
    }
    std::fprintf(stderr, "col_max: %s: %s\n", path, refusal);
    lassi_io_free(a);
    return false;
}

// A DRAM buffer of `pages` pages, one tile per page.
TileBuffer tile_buffer(distributed::MeshDevice& chip, uint32_t pages) {
    distributed::DeviceLocalBufferConfig page_shape{.page_size = kTileByteCount, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig total{.size = static_cast<uint64_t>(kTileByteCount) * pages};
    return distributed::MeshBuffer::create(total, page_shape, &chip);
}

// Give every core of `cores` the three circular buffers: c_0 for a's tiles, c_1 for the scaler, c_16 for results.
void add_col_max_buffers(Program& program, const CoreRangeSet& cores) {
    const std::pair<CBIndex, uint32_t> layout[3] = {
        {CBIndex::c_0, kStreamDepth}, {CBIndex::c_1, kScaleDepth}, {CBIndex::c_16, kStreamDepth}};
    for (const auto& [index, depth] : layout) {
        CircularBufferConfig cb_config(depth * kTileByteCount, {{index, DataFormat::Float16_b}});
        cb_config.set_page_size(index, kTileByteCount);
        CreateCircularBuffer(program, cores, cb_config);
    }
}

// Create the reader, compute, and writer kernels on `cores`, the data-movement ones with their buffer's accessor.
ColMaxKernels add_col_max_kernels(
    Program& program, const CoreRangeSet& cores, const TileBuffer& a_tiles, const TileBuffer& maxima_tiles) {
    std::vector<uint32_t> a_accessor_args;
    TensorAccessorArgs(*a_tiles).append_to(a_accessor_args);
    std::vector<uint32_t> maxima_accessor_args;
    TensorAccessorArgs(*maxima_tiles).append_to(maxima_accessor_args);
    // The reader on the second data-movement processor and its NoC, the writer on the first and its NoC.
    DataMovementConfig column_reader{};
    column_reader.processor = DataMovementProcessor::RISCV_1;
    column_reader.noc = NOC::RISCV_1_default;
    column_reader.compile_args = a_accessor_args;
    DataMovementConfig maxima_writer{};
    maxima_writer.processor = DataMovementProcessor::RISCV_0;
    maxima_writer.noc = NOC::RISCV_0_default;
    maxima_writer.compile_args = maxima_accessor_args;
    // Only the math fidelity is set; every other field keeps its default (no compile args, fp32_dest_acc_en off).
    ComputeConfig column_reducer{};
    column_reducer.math_fidelity = MathFidelity::HiFi4;
    ColMaxKernels kernels{};
    kernels.reader = CreateKernel(program, kReadTileColumns, cores, column_reader);
    kernels.reducer = CreateKernel(program, kMaxTileColumns, cores, column_reducer);
    kernels.writer = CreateKernel(program, kWriteColumnMaxima, cores, maxima_writer);
    return kernels;
}

// Build the program: split the tile columns over the device's compute grid, put the buffers and kernels on the
// working cores, and give each working core its runtime args. Group one (the cores with one tile column more, if
// the split is uneven) is walked first, then group two, range by range and core by core; each core takes the next
// run of tile columns, so every tile column goes to exactly one core, and any grid of at least one core works.
Program make_col_max_program(
    distributed::MeshDevice& chip, const TileBuffer& a_tiles, const TileBuffer& maxima_tiles, const TileShape& shape) {
    const auto split = split_work_to_cores(chip.compute_with_storage_grid_size(), shape.tile_columns);
    Program program = CreateProgram();
    add_col_max_buffers(program, std::get<1>(split));
    const ColMaxKernels kernels = add_col_max_kernels(program, std::get<1>(split), a_tiles, maxima_tiles);
    const uint32_t a_address = static_cast<uint32_t>(a_tiles->address());
    const uint32_t maxima_address = static_cast<uint32_t>(maxima_tiles->address());
    const std::pair<const CoreRangeSet*, uint32_t> groups[2] = {
        {&std::get<2>(split), std::get<4>(split)}, {&std::get<3>(split), std::get<5>(split)}};
    uint32_t next_tile_column = 0;
    for (const auto& [group_cores, columns_per_core] : groups) {
        for (const CoreRange& block : group_cores->ranges()) {
            for (const CoreCoord& core : block) {
                SetRuntimeArgs(
                    program, kernels.reader, core,
                    {a_address, next_tile_column, columns_per_core, shape.tiles_down, shape.tile_columns});
                SetRuntimeArgs(program, kernels.reducer, core, {columns_per_core, shape.tiles_down});
                SetRuntimeArgs(program, kernels.writer, core, {maxima_address, next_tile_column, columns_per_core});
                next_tile_column += columns_per_core;
            }
        }
    }
    return program;
}

}  // namespace

// Read a, find its column maxima on the device, and write the device's maxima as m. The read of the result pages
// and the writing of m stay in main's own body, as PHASE-NOTES' P4.13 hint on the host-compute guard asks.
int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: col_max <a> <m>\n");
        return 2;
    }
    lassi_io_array a = {};
    if (!load_usable_a(argv[1], &a)) {
        return 1;
    }
    const uint32_t rows = static_cast<uint32_t>(a.dims[0]);
    const uint32_t cols = static_cast<uint32_t>(a.dims[1]);
    const TileShape shape{rows / TILE_HEIGHT, cols / TILE_WIDTH};

    // a in the device's tile order: tiles in row-major order, each tile stored as its four 16 x 16 faces.
    std::vector<bfloat16> a_by_rows(static_cast<size_t>(rows) * cols);
    std::memcpy(a_by_rows.data(), a.data, a_by_rows.size() * sizeof(bfloat16));
    const std::vector<bfloat16> a_by_tiles = tilize_nfaces(a_by_rows, rows, cols);

    std::shared_ptr<distributed::MeshDevice> chip = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& queue = chip->mesh_command_queue();
    // One DRAM page per tile of a, and one result page per tile column.
    TileBuffer a_tiles = tile_buffer(*chip, shape.tiles_down * shape.tile_columns);
    TileBuffer maxima_tiles = tile_buffer(*chip, shape.tile_columns);
    distributed::EnqueueWriteMeshBuffer(queue, a_tiles, a_by_tiles, false);

    distributed::MeshWorkload col_max_job;
    Program program = make_col_max_program(*chip, a_tiles, maxima_tiles, shape);
    col_max_job.add_program(distributed::MeshCoordinateRange(chip->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(queue, col_max_job, false);
    distributed::Finish(queue);

    // The result pages, read back from their own buffer; the read sizes the vector to the buffer.
    std::vector<bfloat16> maxima_pages;
    distributed::EnqueueReadMeshBuffer(queue, maxima_pages, maxima_tiles, true);
    const bool closed = chip->close();

    int status = 1;
    if (!closed) {
        std::fprintf(stderr, "col_max: the device did not close cleanly\n");
    } else if (maxima_pages.size() != static_cast<size_t>(shape.tile_columns) * kTileValueCount) {
        std::fprintf(stderr, "col_max: the device returned a result of the wrong size\n");
    } else {
        // Page p is the tile of tile column p, so the pages in order form a 32 x cols matrix of tiles; its first
        // row holds the column maxima, moved here as bit patterns.
        const std::vector<bfloat16> maxima_rows = untilize_nfaces(maxima_pages, TILE_HEIGHT, cols);
        std::vector<uint16_t> m(cols);
        std::memcpy(m.data(), maxima_rows.data(), m.size() * sizeof(uint16_t));
        const uint64_t m_dims[1] = {a.dims[1]};
        status = lassi_io_write(argv[2], "m", LASSI_IO_BF16, 1u, m_dims, m.data());
    }
    lassi_io_free(&a);
    return status == 0 ? 0 : 1;
}
