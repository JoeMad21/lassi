// Written for LASSI-DF tt-par-v0 (task P17.68); a held-out item: B = A^T for a bf16 matrix, tile by tile on the device over its compute grid.
//
// The device transposes a row-major bf16 matrix A of M rows and N columns into B, of N rows and M columns. The work
// unit is one tile of B: B has N / 32 rows of M / 32 tiles, numbered row by row, and split_work_to_cores gives each
// working core a run of consecutive numbers. For B's tile (i, j) the core's reader fetches A's tile (j, i), its
// compute kernel turns that tile over its main diagonal (transpose_wh_tile), and its writer stores the result as B's
// tile (i, j). The host only reorders values: A into tiles before the run, B's tiles back into rows after it.
//
// Usage: transpose_2d <a> <b>
//   a  lassi_io file, bf16, rank 2, [M, N]
//   b  lassi_io file written here, bf16, rank 2, [N, M], row-major: b[j][i] = a[i][j], as the device left it
// M and N must be positive multiples of 32 (whole tiles), at most 65536.
//
// Device settings: on every working core two circular buffers of two Float16_b tiles each, c_0 for A's tiles and
// c_16 for B's; the reader on RISCV_1 with A's TensorAccessorArgs as its compile args, the writer on RISCV_0 with
// B's; the compute kernel with math fidelity HiFi4 (a transpose multiplies nothing, so the fidelity cannot change a
// value) and no compile args. No other ComputeConfig field is set, so the destination register keeps its 16-bit
// format and nothing unpacks straight into it. Runtime args per working core: the reader gets A's address, the
// core's first tile of B, its tile count, and A's tile rows and tile columns; the compute kernel its tile count; the
// writer B's address, the core's first tile of B, and its tile count. A core outside the split gets no kernel.
// A tt-metal error is an exception that ends the program through std::terminate.
#include <cstdint>  // fixed-width counts, sizes, and addresses
#include <cstdio>   // the usage line and refusals
#include <cstring>  // copying A's bit patterns out of the lassi_io array
#include <memory>   // shared handles of the mesh device and its buffers
#include <tuple>    // the fields of split_work_to_cores' result
#include <utility>  // handing the program to the workload
#include <vector>   // host copies of tiles and rows

#include <tt-metalium/bfloat16.hpp>              // bfloat16, the element type of every staged copy
#include <tt-metalium/constants.hpp>             // TILE_HEIGHT and TILE_WIDTH
#include <tt-metalium/device.hpp>                // the compute grid of a device
#include <tt-metalium/distributed.hpp>           // mesh device, mesh buffers, command queue, workload
#include <tt-metalium/host_api.hpp>              // programs, circular buffers, kernels, runtime args
#include <tt-metalium/tensor_accessor_args.hpp>  // DRAM addressing args for the data-movement kernels
#include <tt-metalium/tilize_utils.hpp>          // tilize_nfaces and untilize_nfaces
#include <tt-metalium/work_split.hpp>            // split_work_to_cores

#include "lassi_io.h"  // the harness's array files

using namespace tt::tt_metal;
namespace mesh_api = tt::tt_metal::distributed;

namespace {

static_assert(sizeof(bfloat16) == 2, "a bf16 value takes two bytes in a lassi_io file and on the device");

// A tile's edge, and the bytes of one 32 x 32 tile of bf16 values, the page of every DRAM buffer and circular
// buffer here.
constexpr uint32_t kEdge = tt::constants::TILE_WIDTH;
constexpr uint32_t kTileBytes = static_cast<uint32_t>(sizeof(bfloat16)) * tt::constants::TILE_HEIGHT * kEdge;
// Tiles per circular buffer: two, so a producer can fill one while its consumer works on the other.
constexpr uint32_t kSlots = 2;
// The largest dim accepted. Each side then has at most 2048 tiles and the matrix at most 2^22, which the uint32
// runtime args hold; buffer sizes in bytes are formed in 64 bits.
constexpr uint64_t kMaxDim = 65536;

// This item's kernels, at the paths the build places them, relative to the working directory.
constexpr const char* kReaderPath = "kernels/dataflow/read_mirror_tiles.cpp";
constexpr const char* kComputePath = "kernels/compute/transpose_tiles.cpp";
constexpr const char* kWriterPath = "kernels/dataflow/write_tiles.cpp";

using PagesPtr = std::shared_ptr<mesh_api::MeshBuffer>;

// A's size in tiles; B's is the same with the two swapped.
struct TileGrid {
    uint32_t rows;  // A's tile rows, which are B's tile columns
    uint32_t cols;  // A's tile columns, which are B's tile rows
};

struct TransposeKernels {
    KernelHandle reader;
    KernelHandle compute;
    KernelHandle writer;
};

// True when dim is a whole number of tiles, from one tile to kMaxDim.
bool whole_tiles(uint64_t dim) {
    return dim != 0u && dim % kEdge == 0u && dim <= kMaxDim;
}

// Load the lassi_io file at path into *x and keep it only when it is a bf16 matrix whose two dims are whole tiles:
// return 0 then, and otherwise 1, after a message (lassi_io_read prints its own) and with *x released.
int load_tiled_matrix(const char* path, lassi_io_array* x) {
    const int unreadable = lassi_io_read(path, x);
    if (unreadable != 0) {
        return unreadable;
    }
    const bool bf16_matrix = x->dtype == LASSI_IO_BF16 && x->rank == 2u;
    if (bf16_matrix && whole_tiles(x->dims[0]) && whole_tiles(x->dims[1])) {
        return 0;
    }
    std::fprintf(stderr, "transpose_2d: %s: not a bf16 matrix whose two dims are multiples of 32 from 32 to 65536\n",
                 path);
    lassi_io_free(x);
    return 1;
}

// The rows x cols matrix x holds, as bit patterns in the device's tile order: tiles row by row, each tile as its
// four 16 x 16 faces.
std::vector<bfloat16> in_tile_order(const lassi_io_array& x, uint32_t rows, uint32_t cols) {
    std::vector<bfloat16> by_rows(static_cast<size_t>(rows) * cols);
    std::memcpy(by_rows.data(), x.data, by_rows.size() * sizeof(bfloat16));
    return tilize_nfaces(by_rows, rows, cols);
}

// A DRAM buffer of `tiles` pages, one tile to a page.
PagesPtr tile_pages(mesh_api::MeshDevice& mesh, uint32_t tiles) {
    const mesh_api::DeviceLocalBufferConfig paging{.page_size = kTileBytes, .buffer_type = BufferType::DRAM};
    const mesh_api::ReplicatedBufferConfig extent{.size = static_cast<uint64_t>(kTileBytes) * tiles};
    return mesh_api::MeshBuffer::create(extent, paging, &mesh);
}

// Give every core of `cores` its two circular buffers and the item's three kernels.
TransposeKernels add_kernels(Program& program, const CoreRangeSet& cores, const PagesPtr& a_pages,
                             const PagesPtr& b_pages) {
    for (const tt::CBIndex slot : {tt::CBIndex::c_0, tt::CBIndex::c_16}) {
        CreateCircularBuffer(
            program, cores,
            CircularBufferConfig(kSlots * kTileBytes, {{slot, tt::DataFormat::Float16_b}})
                .set_page_size(slot, kTileBytes));
    }
    std::vector<uint32_t> fetch_layout;
    TensorAccessorArgs(*a_pages).append_to(fetch_layout);
    std::vector<uint32_t> store_layout;
    TensorAccessorArgs(*b_pages).append_to(store_layout);
    TransposeKernels made{};
    made.reader = CreateKernel(
        program, kReaderPath, cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = fetch_layout});
    made.compute = CreateKernel(
        program, kComputePath, cores, ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = {}});
    made.writer = CreateKernel(
        program, kWriterPath, cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = store_layout});
    return made;
}

// The program: split B's tiles over the device's compute grid, put the buffers and kernels on the working cores,
// and give each working core its share as runtime args. The cores are taken group by group (first the group whose
// cores take one tile more), range by range and core by core; each takes the next `share` tiles from `next_tile`
// on, so every tile of B belongs to exactly one core, and one core takes them all on a one-core grid.
Program build_program(
    mesh_api::MeshDevice& mesh, const PagesPtr& a_pages, const PagesPtr& b_pages, const TileGrid& grid) {
    const uint32_t b_tiles = grid.rows * grid.cols;
    const auto split = split_work_to_cores(mesh.compute_with_storage_grid_size(), b_tiles);
    Program program = CreateProgram();
    const TransposeKernels kernels = add_kernels(program, std::get<1>(split), a_pages, b_pages);
    const uint32_t a_addr = static_cast<uint32_t>(a_pages->address());
    const uint32_t b_addr = static_cast<uint32_t>(b_pages->address());
    const CoreRangeSet* group_cores[2] = {&std::get<2>(split), &std::get<3>(split)};
    const uint32_t group_share[2] = {std::get<4>(split), std::get<5>(split)};
    uint32_t next_tile = 0;
    for (int group = 0; group < 2; ++group) {
        const uint32_t share = group_share[group];
        for (const CoreRange& range : group_cores[group]->ranges()) {
            for (const CoreCoord& core : range) {
                SetRuntimeArgs(program, kernels.reader, core, {a_addr, next_tile, share, grid.rows, grid.cols});
                SetRuntimeArgs(program, kernels.compute, core, {share});
                SetRuntimeArgs(program, kernels.writer, core, {b_addr, next_tile, share});
                next_tile += share;
            }
        }
    }
    return program;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: transpose_2d <a> <b>\n");
        return 2;
    }
    lassi_io_array a = {};
    if (load_tiled_matrix(argv[1], &a) != 0) {
        return 1;
    }
    const uint32_t m = static_cast<uint32_t>(a.dims[0]);
    const uint32_t n = static_cast<uint32_t>(a.dims[1]);
    const TileGrid grid{m / kEdge, n / kEdge};

    // A in tile order, for its DRAM buffer.
    std::vector<bfloat16> a_staged = in_tile_order(a, m, n);

    std::shared_ptr<mesh_api::MeshDevice> mesh = mesh_api::MeshDevice::create_unit_mesh(0);
    mesh_api::MeshCommandQueue& queue = mesh->mesh_command_queue();
    // A and B have the same number of tiles; B gets its own buffer.
    PagesPtr a_pages = tile_pages(*mesh, grid.rows * grid.cols);
    PagesPtr b_pages = tile_pages(*mesh, grid.rows * grid.cols);
    mesh_api::EnqueueWriteMeshBuffer(queue, a_pages, a_staged, false);

    Program program = build_program(*mesh, a_pages, b_pages, grid);
    mesh_api::MeshWorkload transpose_run;
    transpose_run.add_program(mesh_api::MeshCoordinateRange(mesh->shape()), std::move(program));
    mesh_api::EnqueueMeshWorkload(queue, transpose_run, false);
    mesh_api::Finish(queue);

    // B's tiles, read back from B's buffer, then laid out as N rows of M values.
    std::vector<bfloat16> b_fetched(static_cast<size_t>(m) * n);
    mesh_api::EnqueueReadMeshBuffer(queue, b_fetched, b_pages, true);
    std::vector<bfloat16> b_rows = untilize_nfaces(b_fetched, n, m);
    const bool closed = mesh->close();

    int status = 1;
    if (!closed) {
        std::fprintf(stderr, "transpose_2d: the device did not close cleanly\n");
    } else {
        const uint64_t b_dims[2] = {a.dims[1], a.dims[0]};
        status = lassi_io_write(argv[2], "b", LASSI_IO_BF16, 2u, b_dims, b_rows.data());
    }
    lassi_io_free(&a);
    return status;
}
