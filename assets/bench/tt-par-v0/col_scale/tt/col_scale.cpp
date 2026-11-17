// Written for LASSI-DF tt-par-v0 (task P17.68); a held-out item: C[i, j] = A[i, j] * s[j] for a bf16 matrix A and row vector s, on the device over its compute grid.
//
// The device scales each column j of a row-major bf16 matrix A (M rows, N columns) by s[j]. The work unit is one
// tile of C: C has M / 32 rows of N / 32 tiles, numbered row by row, and split_work_to_cores gives each working core
// a run of consecutive numbers. For C's tile t the core's reader fetches A's tile t and the scale tile of t's tile
// column, its compute kernel multiplies the two with the scale tile's first row broadcast down all 32 rows
// (mul_tiles_bcast_rows), and its writer stores the product as C's tile t.
//
// The scale tiles. The host lays s out, and computes nothing from it: a 32-row matrix of N columns whose first row
// is s and whose other rows are zero, put in tile order, so scale tile k holds s[32k] to s[32k + 31] in its first
// row. A row broadcast reads only that first row, which every row of A's tile is multiplied by.
//
// Usage: col_scale <a> <s> <c>
//   a  lassi_io file, bf16, rank 2, [M, N]
//   s  lassi_io file, bf16, rank 1, [N]
//   c  lassi_io file written here, bf16, rank 2, [M, N], row-major: each product as the device rounded it
// M and N must be positive multiples of 32 (whole tiles), at most 65536.
//
// Device settings: on every working core three circular buffers of two Float16_b tiles each, c_0 for A's tiles,
// c_1 for scale tiles, and c_16 for C's; the reader on RISCV_1 with A's and then s's TensorAccessorArgs as its
// compile args, the writer on RISCV_0 with C's; the compute kernel with math fidelity HiFi4, so each bf16 product
// uses all four fidelity phases, and no compile args. No other ComputeConfig field is set, so the destination
// register keeps its 16-bit format and nothing unpacks straight into it. Runtime args per working core: the reader
// gets A's and the scale tiles' addresses, the core's first tile of C, its tile count, and the tile columns of A;
// the compute kernel its tile count; the writer C's address, the core's first tile of C, and its tile count. A core
// outside the split gets no kernel.
// A tt-metal error is an exception that ends the program through std::terminate.
#include <cstdint>  // fixed-width counts, sizes, and addresses
#include <cstdio>   // the usage line and refusals
#include <cstring>  // copying bit patterns out of the lassi_io arrays
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

// A tile's edge and height, and the bytes of one 32 x 32 tile of bf16 values, the page of every DRAM buffer and
// circular buffer here.
constexpr uint32_t kEdge = tt::constants::TILE_WIDTH;
constexpr uint32_t kTileRows = tt::constants::TILE_HEIGHT;
constexpr uint32_t kTileBytes = static_cast<uint32_t>(sizeof(bfloat16)) * kTileRows * kEdge;
// Tiles per circular buffer: two, so a producer can fill one while its consumer works on the other.
constexpr uint32_t kSlots = 2;
// The largest dim accepted. Each side then has at most 2048 tiles and the matrix at most 2^22, which the uint32
// runtime args hold; buffer sizes in bytes are formed in 64 bits.
constexpr uint64_t kMaxDim = 65536;

// This item's kernels, at the paths the build places them, relative to the working directory.
constexpr const char* kReaderPath = "kernels/dataflow/read_tile_and_scale.cpp";
constexpr const char* kComputePath = "kernels/compute/scale_columns.cpp";
constexpr const char* kWriterPath = "kernels/dataflow/write_tiles.cpp";

using PagesPtr = std::shared_ptr<mesh_api::MeshBuffer>;

// A's size in tiles, which is C's.
struct TileGrid {
    uint32_t rows;  // tile rows of A and C
    uint32_t cols;  // tile columns of A and C, and the number of scale tiles
};

// The three DRAM buffers: A's tiles, the scale tiles, and C's tiles, each its own buffer.
struct ScalePages {
    PagesPtr a;
    PagesPtr scale;
    PagesPtr c;
};

struct ScaleKernels {
    KernelHandle reader;
    KernelHandle compute;
    KernelHandle writer;
};

// True when dim is a whole number of tiles, from one tile to kMaxDim.
bool whole_tiles(uint64_t dim) {
    return dim != 0u && dim % kEdge == 0u && dim <= kMaxDim;
}

// Load the lassi_io file at path into *x and keep it only when it is bf16 of rank `rank` (2 for A, 1 for s) with
// every dim a whole number of tiles: return 0 then, and otherwise 1, after a message (lassi_io_read prints its own)
// and with *x released.
int load_tiled_bf16(const char* path, uint32_t rank, lassi_io_array* x) {
    const int unreadable = lassi_io_read(path, x);
    if (unreadable != 0) {
        return unreadable;
    }
    bool fits = x->dtype == LASSI_IO_BF16 && x->rank == rank;
    for (uint32_t d = 0; fits && d < rank; ++d) {
        fits = whole_tiles(x->dims[d]);
    }
    if (fits) {
        return 0;
    }
    std::fprintf(stderr, "col_scale: %s: not a rank-%u bf16 array whose dims are multiples of 32 from 32 to 65536\n",
                 path, static_cast<unsigned>(rank));
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

// The scale tiles for s, of cols values: a kTileRows x cols matrix whose first row holds s's bit patterns and whose
// other rows are zero (a value-initialized bfloat16 is all zero bits), in tile order. Only a layout: no value of s
// is changed or combined with another.
std::vector<bfloat16> scale_tiles(const lassi_io_array& s, uint32_t cols) {
    std::vector<bfloat16> first_row_only(static_cast<size_t>(kTileRows) * cols);
    std::memcpy(first_row_only.data(), s.data, static_cast<size_t>(cols) * sizeof(bfloat16));
    return tilize_nfaces(first_row_only, kTileRows, cols);
}

// A DRAM buffer of `tiles` pages, one tile to a page.
PagesPtr tile_pages(mesh_api::MeshDevice& mesh, uint32_t tiles) {
    const mesh_api::DeviceLocalBufferConfig paging{.page_size = kTileBytes, .buffer_type = BufferType::DRAM};
    const mesh_api::ReplicatedBufferConfig extent{.size = static_cast<uint64_t>(kTileBytes) * tiles};
    return mesh_api::MeshBuffer::create(extent, paging, &mesh);
}

// Give every core of `cores` its three circular buffers and the item's three kernels.
ScaleKernels add_kernels(Program& program, const CoreRangeSet& cores, const ScalePages& pages) {
    for (const tt::CBIndex slot : {tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16}) {
        CreateCircularBuffer(
            program, cores,
            CircularBufferConfig(kSlots * kTileBytes, {{slot, tt::DataFormat::Float16_b}})
                .set_page_size(slot, kTileBytes));
    }
    // The reader finds A's paging args from index 0 and the scale tiles' right after them.
    std::vector<uint32_t> fetch_layout;
    TensorAccessorArgs(*pages.a).append_to(fetch_layout);
    TensorAccessorArgs(*pages.scale).append_to(fetch_layout);
    std::vector<uint32_t> store_layout;
    TensorAccessorArgs(*pages.c).append_to(store_layout);
    ScaleKernels made{};
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

// The program: split C's tiles over the device's compute grid, put the buffers and kernels on the working cores,
// and give each working core its share as runtime args. The cores are taken group by group (first the group whose
// cores take one tile more), range by range and core by core; each takes the next `share` tiles from `next_tile`
// on, so every tile of C belongs to exactly one core, and one core takes them all on a one-core grid.
Program build_program(mesh_api::MeshDevice& mesh, const ScalePages& pages, const TileGrid& grid) {
    const uint32_t c_tiles = grid.rows * grid.cols;
    const auto split = split_work_to_cores(mesh.compute_with_storage_grid_size(), c_tiles);
    Program program = CreateProgram();
    const ScaleKernels kernels = add_kernels(program, std::get<1>(split), pages);
    const uint32_t a_addr = static_cast<uint32_t>(pages.a->address());
    const uint32_t scale_addr = static_cast<uint32_t>(pages.scale->address());
    const uint32_t c_addr = static_cast<uint32_t>(pages.c->address());
    const CoreRangeSet* group_cores[2] = {&std::get<2>(split), &std::get<3>(split)};
    const uint32_t group_share[2] = {std::get<4>(split), std::get<5>(split)};
    uint32_t next_tile = 0;
    for (int group = 0; group < 2; ++group) {
        const uint32_t share = group_share[group];
        for (const CoreRange& range : group_cores[group]->ranges()) {
            for (const CoreCoord& core : range) {
                SetRuntimeArgs(program, kernels.reader, core, {a_addr, scale_addr, next_tile, share, grid.cols});
                SetRuntimeArgs(program, kernels.compute, core, {share});
                SetRuntimeArgs(program, kernels.writer, core, {c_addr, next_tile, share});
                next_tile += share;
            }
        }
    }
    return program;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: col_scale <a> <s> <c>\n");
        return 2;
    }
    lassi_io_array a = {};
    lassi_io_array s = {};
    if (load_tiled_bf16(argv[1], 2u, &a) != 0 || load_tiled_bf16(argv[2], 1u, &s) != 0) {
        lassi_io_free(&a);
        return 1;
    }
    if (s.dims[0] != a.dims[1]) {
        std::fprintf(stderr, "col_scale: a has %llu columns but s has %llu values\n",
                     static_cast<unsigned long long>(a.dims[1]), static_cast<unsigned long long>(s.dims[0]));
        lassi_io_free(&a);
        lassi_io_free(&s);
        return 1;
    }
    const uint32_t m = static_cast<uint32_t>(a.dims[0]);
    const uint32_t n = static_cast<uint32_t>(a.dims[1]);
    const TileGrid grid{m / kEdge, n / kEdge};

    // A and the scale tiles in tile order, for their DRAM buffers.
    std::vector<bfloat16> a_staged = in_tile_order(a, m, n);
    std::vector<bfloat16> scale_staged = scale_tiles(s, n);

    std::shared_ptr<mesh_api::MeshDevice> mesh = mesh_api::MeshDevice::create_unit_mesh(0);
    mesh_api::MeshCommandQueue& queue = mesh->mesh_command_queue();
    // Not const: the mesh buffer read and writes take a buffer handle by plain reference.
    ScalePages pages{tile_pages(*mesh, grid.rows * grid.cols), tile_pages(*mesh, grid.cols),
                     tile_pages(*mesh, grid.rows * grid.cols)};
    mesh_api::EnqueueWriteMeshBuffer(queue, pages.a, a_staged, false);
    mesh_api::EnqueueWriteMeshBuffer(queue, pages.scale, scale_staged, false);

    Program program = build_program(*mesh, pages, grid);
    mesh_api::MeshWorkload scale_run;
    scale_run.add_program(mesh_api::MeshCoordinateRange(mesh->shape()), std::move(program));
    mesh_api::EnqueueMeshWorkload(queue, scale_run, false);
    mesh_api::Finish(queue);

    // C's tiles, read back from C's own buffer, then laid out as M rows of N values.
    std::vector<bfloat16> c_fetched(static_cast<size_t>(m) * n);
    mesh_api::EnqueueReadMeshBuffer(queue, c_fetched, pages.c, true);
    std::vector<bfloat16> c_rows = untilize_nfaces(c_fetched, m, n);
    const bool closed = mesh->close();

    int status = 1;
    if (!closed) {
        std::fprintf(stderr, "col_scale: the device did not close cleanly\n");
    } else {
        const uint64_t c_dims[2] = {a.dims[0], a.dims[1]};
        status = lassi_io_write(argv[3], "c", LASSI_IO_BF16, 2u, c_dims, c_rows.data());
    }
    lassi_io_free(&a);
    lassi_io_free(&s);
    return status;
}
