// Written for LASSI-DF tt-par-v0 (task P17.68); a held-out item: the sum of each row of a bf16 matrix, spread over the compute grid.
//
// What it computes, on the device: s[i] = the sum over j of a[i, j], for a bf16 matrix a of rows x cols.
//
// How the work is spread: the unit of work is a tile row, the 32 rows of a covered by one row of 32 x 32 tiles
// (cols / 32 tiles). split_work_to_cores deals the rows / 32 tile rows out over the device's compute grid, each
// working core taking a run of consecutive tile rows, given to its kernels as runtime args (its first tile row and
// how many it owns). On each working core:
//   - the reader (kernels/dataflow/read_tile_rows.cpp, RISCV_1) fills circular buffer c_1 once with a scaler tile
//     (bf16 1.0 in the first row of each 16 x 16 face, zero elsewhere), then streams the tiles of its tile rows from
//     DRAM into c_0, one tile at a time, left to right along each tile row;
//   - the compute kernel (kernels/compute/sum_tile_rows.cpp) reduces every tile of a tile row along its rows into
//     one Dst tile (reduce_tile, SUM over REDUCE_ROW), whose first column then holds the 32 row sums, and packs it
//     into c_16;
//   - the writer (kernels/dataflow/write_row_sums.cpp, RISCV_0) stores that tile in DRAM at its tile row's page.
// The host reads the result pages back, lays them out as a rows x 32 row-major matrix, takes its first column,
// widens each bf16 sum to f32 (an exact step), and writes s. No value of s is computed on the host.
//
// Usage: row_sum <a> <s>
//   a  lassi_io file, bf16, rank 2, [rows, cols]: rows and cols positive multiples of 32, at most 65536 each
//   s  lassi_io file written here, f32, rank 1, [rows]
//
// Device settings: three circular buffers on every working core, all Float16_b with one tile per page: c_0 (two
// pages of a), c_1 (one page, the scaler), and c_16 (two result pages). The reader and the writer each get their
// DRAM buffer's TensorAccessorArgs as compile args. The compute kernel's ComputeConfig sets only math fidelity
// HiFi4; every other field keeps its default, so there are no compile args, fp32_dest_acc_en stays off (the sums
// accumulate in a 16-bit Dst), and unpack_to_dest_mode stays empty. HiFi4 runs every fidelity phase, so each bf16
// element times the scaler 1.0 reaches Dst with all of its mantissa.
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

static_assert(sizeof(bfloat16) == sizeof(uint16_t), "row_sum copies a's bf16 bit patterns into bfloat16 as they are");

// One tile of bf16: 32 x 32 values in 2048 bytes, the page size of both DRAM buffers and of every circular buffer.
constexpr uint32_t kValuesPerTile = TILE_HEIGHT * TILE_WIDTH;
constexpr uint32_t kBytesPerTile = kValuesPerTile * static_cast<uint32_t>(sizeof(bfloat16));
// The largest row or column count accepted. Tile counts (at most 2048) and their products (at most 2^22) then fit
// the uint32 runtime args; buffer sizes in bytes are computed in 64 bits.
constexpr uint64_t kLongestSide = 65536;
// Pages per circular buffer: two where a tile streams through, so the producer can work one tile ahead, and one
// for the scaler, which is written once.
constexpr uint32_t kStreamingPages = 2;
constexpr uint32_t kScalerPages = 1;

// The kernels, at the paths this program's build places them, relative to the working directory.
constexpr const char* kReadTileRows = "kernels/dataflow/read_tile_rows.cpp";
constexpr const char* kSumTileRows = "kernels/compute/sum_tile_rows.cpp";
constexpr const char* kWriteRowSums = "kernels/dataflow/write_row_sums.cpp";

using DramTiles = std::shared_ptr<distributed::MeshBuffer>;

// The tile counts of a: tile rows (the units of work) and tiles in one tile row.
struct TileCounts {
    uint32_t tile_rows;
    uint32_t tiles_across;
};

struct RowSumKernels {
    KernelHandle reader;
    KernelHandle reducer;
    KernelHandle writer;
};

// Read the lassi_io file at path into *a and check that it holds a bf16 matrix whose two sides are positive
// multiples of 32, at most kLongestSide; return true, or say why on stderr, release *a, and return false.
bool read_usable_a(const char* path, lassi_io_array* a) {
    if (lassi_io_read(path, a) != 0) {
        return false;
    }
    const char* why_not = nullptr;
    if (a->dtype != LASSI_IO_BF16) {
        why_not = "a must be bf16";
    } else if (a->rank != 2u) {
        why_not = "a must have rank 2";
    } else {
        for (uint32_t side = 0; side < 2u; ++side) {
            const uint64_t length = a->dims[side];
            if (length == 0u || length % TILE_WIDTH != 0u || length > kLongestSide) {
                why_not = "each side of a must be a positive multiple of 32, at most 65536";
            }
        }
    }
    if (why_not == nullptr) {
        return true;
    }
    std::fprintf(stderr, "row_sum: %s: %s\n", path, why_not);
    lassi_io_free(a);
    return false;
}

// The f32 value of a bf16 bit pattern, which is the upper half of that f32's bits, so the step is exact.
float bf16_bits_as_f32(uint16_t bits) {
    const uint32_t f32_bits = static_cast<uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &f32_bits, sizeof value);
    return value;
}

// A DRAM buffer of `pages` pages, one tile per page.
DramTiles dram_tiles(distributed::MeshDevice& chip, uint32_t pages) {
    distributed::DeviceLocalBufferConfig per_page{.page_size = kBytesPerTile, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig whole{.size = static_cast<uint64_t>(kBytesPerTile) * pages};
    return distributed::MeshBuffer::create(whole, per_page, &chip);
}

// Give every core of `cores` the three circular buffers: c_0 for a's tiles, c_1 for the scaler, c_16 for results.
void add_row_sum_buffers(Program& program, const CoreRangeSet& cores) {
    const std::pair<CBIndex, uint32_t> slots[3] = {
        {CBIndex::c_0, kStreamingPages}, {CBIndex::c_1, kScalerPages}, {CBIndex::c_16, kStreamingPages}};
    for (const auto& [slot, pages] : slots) {
        CircularBufferConfig slot_config(pages * kBytesPerTile, {{slot, DataFormat::Float16_b}});
        slot_config.set_page_size(slot, kBytesPerTile);
        CreateCircularBuffer(program, cores, slot_config);
    }
}

// Create the reader, compute, and writer kernels on `cores`, the data-movement ones with their buffer's accessor.
RowSumKernels add_row_sum_kernels(
    Program& program, const CoreRangeSet& cores, const DramTiles& a_dram, const DramTiles& sums_dram) {
    std::vector<uint32_t> reader_compile_args;
    TensorAccessorArgs(*a_dram).append_to(reader_compile_args);
    std::vector<uint32_t> writer_compile_args;
    TensorAccessorArgs(*sums_dram).append_to(writer_compile_args);
    // The reader on the second data-movement processor and its NoC, the writer on the first and its NoC.
    DataMovementConfig tile_row_reader{};
    tile_row_reader.processor = DataMovementProcessor::RISCV_1;
    tile_row_reader.noc = NOC::RISCV_1_default;
    tile_row_reader.compile_args = reader_compile_args;
    DataMovementConfig row_sum_writer{};
    row_sum_writer.processor = DataMovementProcessor::RISCV_0;
    row_sum_writer.noc = NOC::RISCV_0_default;
    row_sum_writer.compile_args = writer_compile_args;
    // Only the math fidelity is set; every other field keeps its default (no compile args, fp32_dest_acc_en off).
    ComputeConfig row_reducer{};
    row_reducer.math_fidelity = MathFidelity::HiFi4;
    RowSumKernels kernels{};
    kernels.reader = CreateKernel(program, kReadTileRows, cores, tile_row_reader);
    kernels.reducer = CreateKernel(program, kSumTileRows, cores, row_reducer);
    kernels.writer = CreateKernel(program, kWriteRowSums, cores, row_sum_writer);
    return kernels;
}

// Build the program: split the tile rows over the device's compute grid, put the buffers and kernels on the
// working cores, and give each working core its runtime args. Group one (the cores with one tile row more, if the
// split is uneven) is walked first, then group two, range by range and core by core; each core takes the next run
// of tile rows, so every tile row goes to exactly one core, and any grid of at least one core works.
Program make_row_sum_program(
    distributed::MeshDevice& chip, const DramTiles& a_dram, const DramTiles& sums_dram, const TileCounts& counts) {
    const auto split = split_work_to_cores(chip.compute_with_storage_grid_size(), counts.tile_rows);
    Program program = CreateProgram();
    add_row_sum_buffers(program, std::get<1>(split));
    const RowSumKernels kernels = add_row_sum_kernels(program, std::get<1>(split), a_dram, sums_dram);
    const uint32_t a_address = static_cast<uint32_t>(a_dram->address());
    const uint32_t sums_address = static_cast<uint32_t>(sums_dram->address());
    const std::pair<const CoreRangeSet*, uint32_t> groups[2] = {
        {&std::get<2>(split), std::get<4>(split)}, {&std::get<3>(split), std::get<5>(split)}};
    uint32_t next_tile_row = 0;
    for (const auto& [group_cores, rows_per_core] : groups) {
        for (const CoreRange& block : group_cores->ranges()) {
            for (const CoreCoord& core : block) {
                SetRuntimeArgs(
                    program, kernels.reader, core, {a_address, next_tile_row, rows_per_core, counts.tiles_across});
                SetRuntimeArgs(program, kernels.reducer, core, {rows_per_core, counts.tiles_across});
                SetRuntimeArgs(program, kernels.writer, core, {sums_address, next_tile_row, rows_per_core});
                next_tile_row += rows_per_core;
            }
        }
    }
    return program;
}

}  // namespace

// Read a, sum its rows on the device, and write the device's sums, widened to f32, as s. The read of the result
// pages and the writing of s stay in main's own body, as PHASE-NOTES' P4.13 hint on the host-compute guard asks.
int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: row_sum <a> <s>\n");
        return 2;
    }
    lassi_io_array a = {};
    if (!read_usable_a(argv[1], &a)) {
        return 1;
    }
    const uint32_t rows = static_cast<uint32_t>(a.dims[0]);
    const uint32_t cols = static_cast<uint32_t>(a.dims[1]);
    const TileCounts counts{rows / TILE_HEIGHT, cols / TILE_WIDTH};

    // a in the device's tile order: tiles in row-major order, each tile stored as its four 16 x 16 faces.
    std::vector<bfloat16> a_row_major(static_cast<size_t>(rows) * cols);
    std::memcpy(a_row_major.data(), a.data, a_row_major.size() * sizeof(bfloat16));
    const std::vector<bfloat16> a_in_tiles = tilize_nfaces(a_row_major, rows, cols);

    std::shared_ptr<distributed::MeshDevice> chip = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& queue = chip->mesh_command_queue();
    // One DRAM page per tile of a, and one result page per tile row.
    DramTiles a_dram = dram_tiles(*chip, counts.tile_rows * counts.tiles_across);
    DramTiles sums_dram = dram_tiles(*chip, counts.tile_rows);
    distributed::EnqueueWriteMeshBuffer(queue, a_dram, a_in_tiles, false);

    distributed::MeshWorkload row_sum_job;
    Program program = make_row_sum_program(*chip, a_dram, sums_dram, counts);
    row_sum_job.add_program(distributed::MeshCoordinateRange(chip->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(queue, row_sum_job, false);
    distributed::Finish(queue);

    // The result pages, read back from their own buffer; the read sizes the vector to the buffer.
    std::vector<bfloat16> sum_pages;
    distributed::EnqueueReadMeshBuffer(queue, sum_pages, sums_dram, true);
    const bool closed = chip->close();

    int status = 1;
    if (!closed) {
        std::fprintf(stderr, "row_sum: the device did not close cleanly\n");
    } else if (sum_pages.size() != static_cast<size_t>(counts.tile_rows) * kValuesPerTile) {
        std::fprintf(stderr, "row_sum: the device returned a result of the wrong size\n");
    } else {
        // Page p is the tile of tile row p, so the pages in order form a rows x 32 matrix of tiles; its first
        // column holds the row sums.
        const std::vector<bfloat16> sum_columns = untilize_nfaces(sum_pages, rows, TILE_WIDTH);
        std::vector<float> s(rows);
        for (uint32_t i = 0; i < rows; ++i) {
            uint16_t sum_bits = 0;
            std::memcpy(&sum_bits, &sum_columns[static_cast<size_t>(i) * TILE_WIDTH], sizeof sum_bits);
            s[i] = bf16_bits_as_f32(sum_bits);
        }
        const uint64_t s_dims[1] = {a.dims[0]};
        status = lassi_io_write(argv[2], "s", LASSI_IO_F32, 1u, s_dims, s.data());
    }
    lassi_io_free(&a);
    return status == 0 ? 0 : 1;
}
