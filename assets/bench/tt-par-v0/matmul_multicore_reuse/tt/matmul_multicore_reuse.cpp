// Written for LASSI-DF tt-par-v0 (task P17.67); runs the unmodified upstream matmul_multicore_reuse kernels from tt-metal 5280a9cf.
//
// Matrix multiply on the device, c = a x b, with c cut into equal blocks of whole tiles and one block per core. Each
// core's reader copies its block's strip of a (the block's tile rows) and strip of b (the block's tile columns) from
// DRAM into L1, two tiles of K at a time; its compute kernel multiplies the two strips subblock by subblock, parking
// the partial sums in a spill buffer between K steps; and its writer stores the finished block, subblock by subblock.
// Each core reads its own strips; no data passes between cores.
//
// Usage: matmul_multicore_reuse <a> <b> <c>
//   a  lassi_io file, bf16, rank 2, [M, K]
//   b  lassi_io file, bf16, rank 2, [K, N]
//   c  lassi_io file written here, bf16, rank 2, [M, N]: the device result in row-major order
// M and N must be positive multiples of 32 and K a positive multiple of 64 (the kernels step through K two tiles at
// a time), each at most 65536, and the block rule below must place M and N on the device's grid. The inputs are
// already bf16, so the host converts no value; it only reorders a and b into tiles and the result back into rows.
//
// The block rule is the upstream host's (get_large_matmul_params in matmul/matmul_common/bmm_op.hpp at the pin),
// restated here in code of its own. A block is block_rows x block_cols tiles of c, with K steps of two tiles, and
// must fit an L1 budget of 400 tiles: two steps of a's strip, two of b's, and the block itself. Along each dim, the
// prime factors of the tile count that exceed the grid's extent along it must all fall in one block. The rest is
// searched as plan_blocks says, and the subblock (the tiles the compute kernel holds at once) is the first shape of
// kSubblockShapes that divides the block. At the item's size, M 1568, K 256, N 1024 (49 x 8 x 32 tiles), on a grid
// 8 cores wide and 7, 8, or 9 tall, it gives blocks of 7 x 4 tiles and subblocks of 7 x 1: 7 x 8 = 56 blocks, so 56
// cores, (0, 0) to (7, 6), and 4 K steps (PROJECTED: arithmetic over the rule, no run; P17.70 counts the cores).
// The program prints the plan it uses.
//
// Device settings, as the upstream example sets them: on every working core, circular buffers c_0 (two steps of a's
// strip) and c_1 (two steps of b's strip), and c_16 (the output) sharing one config of one block with c_24 (the
// spill), all Float16_b and paged by tile; the reader on RISCV_1 (NOC 1) and the writer on RISCV_0 (NOC 0), each
// given its buffers' TensorAccessorArgs as compile args; the compute kernel with math fidelity HiFi4 and its twelve
// block sizes as compile args. No other ComputeConfig field is set, so fp32_dest_acc_en stays off and
// unpack_to_dest_mode stays empty. A tt-metal error is an exception that ends the program through std::terminate.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/work_split.hpp>

#include "lassi_io.h"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

namespace {

static_assert(sizeof(bfloat16) == 2, "a bf16 value is two bytes, as in a lassi_io file");

// One 32 x 32 tile of bf16 values: 2048 bytes, the page size of every DRAM buffer and circular buffer here.
constexpr uint32_t kTileBytes = static_cast<uint32_t>(sizeof(bfloat16)) * TILE_HEIGHT * TILE_WIDTH;
// The K tiles each reader step brings and each compute step consumes.
constexpr uint32_t kStepTiles = 2;
// The rule's L1 budget per core, in tiles.
constexpr int64_t kBudgetTiles = 400;
// One product: the kernels' batch count, with b not broadcast over it.
constexpr uint32_t kBatch = 1;
// The largest accepted dim: it keeps every tile count, and every product of two, inside 32 bits.
constexpr uint64_t kMaxDim = 65536;

// The subblock shapes (rows, cols) the rule tries, in this order, largest tile count first.
constexpr uint32_t kSubblockShapes[][2] = {
    {4, 2}, {2, 4}, {8, 1}, {1, 8},  // eight tiles
    {7, 1}, {1, 7},                  // seven
    {3, 2}, {2, 3}, {6, 1}, {1, 6},  // six
    {5, 1}, {1, 5},                  // five
    {2, 2}, {4, 1}, {1, 4},          // four
    {3, 1}, {1, 3},                  // three
    {2, 1}, {1, 2},                  // two
    {1, 1},                          // one
};

// The upstream kernels, at the paths the build places them, relative to the working directory.
constexpr const char* kReaderKernel = "matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout.cpp";
constexpr const char* kWriterKernel = "matmul/matmul_common/kernels/dataflow/writer_bmm_tile_layout.cpp";
constexpr const char* kComputeKernel = "matmul/matmul_common/kernels/compute/bmm_large_block_zm.cpp";

using BufferPtr = std::shared_ptr<distributed::MeshBuffer>;

// Tile counts along M, K, and N.
struct TileCounts {
    uint32_t mt;
    uint32_t kt;
    uint32_t nt;
};

// One core's block of c and the subblock its compute kernel holds at once, in tiles.
struct BlockPlan {
    uint32_t block_rows;
    uint32_t block_cols;
    uint32_t sub_rows;
    uint32_t sub_cols;
};

struct KernelIds {
    KernelHandle reader;
    KernelHandle writer;
    KernelHandle compute;
};

// The DRAM addresses of a, b, and c, as the kernels take them.
struct Addresses {
    uint32_t a;
    uint32_t b;
    uint32_t c;
};

// The product of the prime factors of `tiles`, counted with multiplicity, that exceed `cores`: one block along this
// dim must hold all of them, since the blocks along it cannot outnumber the cores.
uint32_t forced_block(uint32_t tiles, uint32_t cores) {
    uint32_t forced = 1;
    uint32_t rest = tiles;
    for (uint32_t p = 2; p * p <= rest;) {
        if (rest % p != 0) {
            ++p;
            continue;
        }
        rest /= p;
        if (p > cores) {
            forced *= p;
        }
    }
    if (rest > 1 && rest > cores) {
        forced *= rest;
    }
    return forced;
}

// Every divisor of r above one, ascending; only one when r is one.
std::vector<uint32_t> divisor_choices(uint32_t r) {
    std::vector<uint32_t> out;
    for (uint32_t d = 2; d <= r; ++d) {
        if (r % d == 0) {
            out.push_back(d);
        }
    }
    if (out.empty()) {
        out.push_back(1);
    }
    return out;
}

// The largest block size along one dim that keeps 2 w (rows + cols) + rows x cols within the budget when the other
// dim's block size is `other` (w = kStepTiles), with the rule's truncating signed division; 0 when nothing fits.
uint32_t budget_limit(uint32_t other) {
    const int64_t w = kStepTiles;
    const int64_t room = kBudgetTiles - 2 * w * static_cast<int64_t>(other);
    const int64_t limit = room / (2 * w + static_cast<int64_t>(other));
    return limit > 0 ? static_cast<uint32_t>(limit) : 0u;
}

// Whether blocks of rows x cols tiles of an mt x nt tile output fit on the grid, one per core.
bool on_grid(const TileCounts& t, uint32_t rows, uint32_t cols, const CoreCoord& grid) {
    return t.mt / rows <= grid.y && t.nt / cols <= grid.x;
}

// Set *plan to blocks of rows x cols tiles with the first subblock shape that divides them; false when none does.
bool pick_subblock(uint32_t rows, uint32_t cols, BlockPlan* plan) {
    for (const auto& shape : kSubblockShapes) {
        if (rows % shape[0] == 0 && cols % shape[1] == 0) {
            *plan = BlockPlan{rows, cols, shape[0], shape[1]};
            return true;
        }
    }
    return false;
}

// With the other dim's block forced to `forced` tiles, grow this dim's block from `least` by the divisors of
// tiles / least, smallest first, while it stays within the budget, stopping at the first that does not.
uint32_t grow_beside(uint32_t forced, uint32_t tiles, uint32_t least) {
    uint32_t size = least;
    for (const uint32_t d : divisor_choices(tiles / least)) {
        if (d * least > budget_limit(forced)) {
            break;
        }
        size = d * least;
    }
    return size;
}

// Neither dim forced: try each column size from divisor_choices(nt) in turn, give the rows the largest of
// divisor_choices(mt) within the budget beside it (keeping the last choice when none fits), and take the first
// pair whose blocks fit on the grid.
bool plan_free(const TileCounts& t, const CoreCoord& grid, BlockPlan* plan) {
    uint32_t rows = 1;
    for (const uint32_t cols : divisor_choices(t.nt)) {
        const uint32_t limit = budget_limit(cols);
        for (const uint32_t d : divisor_choices(t.mt)) {
            if (d <= limit) {
                rows = d;
            }
        }
        if (on_grid(t, rows, cols, grid) && pick_subblock(rows, cols, plan)) {
            return true;
        }
    }
    return false;
}

// The block rule (the header comment): set *plan for an mt x nt tile output on `grid` (x columns, y rows of cores)
// and return true, or return false when the rule finds no blocks.
bool plan_blocks(const TileCounts& t, const CoreCoord& grid, BlockPlan* plan) {
    const uint32_t row_least = forced_block(t.mt, static_cast<uint32_t>(grid.y));
    const uint32_t col_least = forced_block(t.nt, static_cast<uint32_t>(grid.x));
    if (col_least > budget_limit(row_least)) {
        return false;
    }
    if (row_least > 1) {
        const uint32_t cols = grow_beside(row_least, t.nt, col_least);
        return on_grid(t, row_least, cols, grid) && pick_subblock(row_least, cols, plan);
    }
    if (col_least > 1) {
        const uint32_t rows = grow_beside(col_least, t.mt, row_least);
        return on_grid(t, rows, col_least, grid) && pick_subblock(rows, col_least, plan);
    }
    return plan_free(t, grid, plan);
}

// Read the lassi_io file at path into *x and check that it holds a bf16 matrix whose dims are positive multiples
// of 32, at most kMaxDim; return 0, or print why and return 1 with *x released.
int read_matrix(const char* path, lassi_io_array* x) {
    if (lassi_io_read(path, x) != 0) {
        return 1;
    }
    const char* problem = nullptr;
    if (x->dtype != LASSI_IO_BF16) {
        problem = "the array must be bf16";
    } else if (x->rank != 2u) {
        problem = "the array must have rank 2";
    } else {
        for (uint32_t i = 0; i < 2u; ++i) {
            if (x->dims[i] == 0u || x->dims[i] % TILE_WIDTH != 0u || x->dims[i] > kMaxDim) {
                problem = "each dim must be a positive multiple of 32, at most 65536";
            }
        }
    }
    if (problem != nullptr) {
        std::fprintf(stderr, "matmul_multicore_reuse: %s: %s\n", path, problem);
        lassi_io_free(x);
        return 1;
    }
    return 0;
}

// Read a and b and check that a's columns match b's rows, a whole number of K steps; return 0, or print why and
// return 1 with both released.
int read_inputs(const char* a_path, const char* b_path, lassi_io_array* a, lassi_io_array* b) {
    if (read_matrix(a_path, a) != 0) {
        return 1;
    }
    if (read_matrix(b_path, b) != 0) {
        lassi_io_free(a);
        return 1;
    }
    const char* problem = nullptr;
    if (a->dims[1] != b->dims[0]) {
        problem = "the columns of a must equal the rows of b";
    } else if (a->dims[1] % (kStepTiles * TILE_WIDTH) != 0u) {
        problem = "K must be a multiple of 64, the kernels' K step";
    }
    if (problem != nullptr) {
        std::fprintf(stderr, "matmul_multicore_reuse: %s\n", problem);
        lassi_io_free(a);
        lassi_io_free(b);
        return 1;
    }
    return 0;
}

// Copy the rows x cols bf16 matrix held by x and return it in the device's tile order.
std::vector<bfloat16> tiled(const lassi_io_array& x, uint32_t rows, uint32_t cols) {
    std::vector<bfloat16> flat(static_cast<size_t>(rows) * cols);
    std::memcpy(flat.data(), x.data, flat.size() * sizeof(bfloat16));
    return tilize_nfaces(flat, rows, cols);
}

// Create a DRAM buffer of `tiles` tiles, one tile per page.
BufferPtr dram_buffer(distributed::MeshDevice& device, uint32_t tiles) {
    distributed::DeviceLocalBufferConfig local{.page_size = kTileBytes, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig whole{.size = static_cast<uint64_t>(kTileBytes) * tiles};
    return distributed::MeshBuffer::create(whole, local, &device);
}

// Plan the blocks on the device's grid and print the plan; return false, after saying why, when the rule finds none.
bool plan_on_device(distributed::MeshDevice& device, const TileCounts& t, BlockPlan* plan) {
    const CoreCoord grid = device.compute_with_storage_grid_size();
    if (!plan_blocks(t, grid, plan)) {
        std::fprintf(stderr, "matmul_multicore_reuse: the block rule places no %u x %u tile output on a %u x %u grid\n",
                     t.mt, t.nt, static_cast<unsigned>(grid.x), static_cast<unsigned>(grid.y));
        return false;
    }
    std::printf(
        "matmul_multicore_reuse: %u x %u blocks of %u x %u tiles, subblocks %u x %u, %u K steps, on a %u x %u grid\n",
        t.mt / plan->block_rows, t.nt / plan->block_cols, plan->block_rows, plan->block_cols, plan->sub_rows,
        plan->sub_cols, t.kt / kStepTiles, static_cast<unsigned>(grid.x), static_cast<unsigned>(grid.y));
    return true;
}

// The compute kernel's twelve compile-time args, in the kernel's order.
std::vector<uint32_t> compute_compile_args(const TileCounts& t, const BlockPlan& p) {
    const uint32_t down = p.block_rows / p.sub_rows;
    const uint32_t across = p.block_cols / p.sub_cols;
    return {
        kStepTiles,                        // K tiles per step
        down,                              // subblocks down a block
        p.sub_rows * kStepTiles * down,    // tiles of a's strip per step
        p.sub_rows * kStepTiles,           // tiles of a's strip per subblock row
        across,                            // subblocks across a block
        p.sub_cols * kStepTiles * across,  // tiles of b's strip per step
        p.sub_cols * across,               // b's strip width in tiles
        t.kt / kStepTiles,                 // K steps
        p.sub_rows,
        p.sub_cols,
        p.sub_rows * p.sub_cols,           // tiles per subblock
        kBatch,
    };
}

// The reader's 21 runtime args for the block at block row `row`, block column `col`, in the kernel's order.
std::vector<uint32_t> reader_runtime_args(
    const Addresses& addr, const TileCounts& t, const BlockPlan& p, uint32_t row, uint32_t col) {
    return {
        addr.a,
        t.kt * p.block_rows * row,   // a: the strip's first tile
        1u,                          // a: the next tile along K
        t.kt,                        // a: the next tile row
        kStepTiles,                  // a: the next K step
        kStepTiles,                  // a: strip width per step
        p.block_rows,                // a: strip height
        kStepTiles * p.block_rows,   // a: tiles per step
        addr.b,
        p.block_cols * col,          // b: the strip's first tile
        1u,                          // b: the next tile along N
        t.nt,                        // b: the next tile row
        kStepTiles * t.nt,           // b: the next K step
        p.block_cols,                // b: strip width
        kStepTiles,                  // b: strip height per step
        p.block_cols * kStepTiles,   // b: tiles per step
        t.kt / kStepTiles,           // K steps
        t.mt * t.kt,                 // a's tiles per batch
        t.kt * t.nt,                 // b's tiles per batch
        kBatch,
        0u,                          // b is not broadcast over the batch
    };
}

// The writer's 13 runtime args for the block at block row `row`, block column `col`, in the kernel's order.
std::vector<uint32_t> writer_runtime_args(
    uint32_t c_addr, const TileCounts& t, const BlockPlan& p, uint32_t row, uint32_t col) {
    return {
        c_addr,
        col * p.block_cols + row * p.block_rows * t.nt,  // the block's first tile of c
        1u,                                              // the next tile along a row
        t.nt,                                            // the next tile row
        p.sub_cols,                                      // the next subblock across
        p.sub_rows * t.nt,                               // the next subblock down
        p.sub_cols,
        p.sub_rows,
        p.sub_cols * p.sub_rows,                         // tiles per subblock
        p.block_cols / p.sub_cols,                       // subblocks across
        p.block_rows / p.sub_rows,                       // subblocks down
        t.mt * t.nt,                                     // c's tiles per batch
        kBatch,
    };
}

// Create the four circular buffers on `cores`: c_0 and c_1 two K steps deep, and c_16 with c_24 in one shared
// config of one block, all Float16_b and paged by tile.
void create_circular_buffers(Program& program, const CoreRangeSet& cores, const BlockPlan& p) {
    const uint32_t a_tiles = 2u * p.block_rows * kStepTiles;
    const uint32_t b_tiles = 2u * p.block_cols * kStepTiles;
    const uint32_t out_tiles = p.block_rows * p.block_cols;
    CreateCircularBuffer(
        program, cores,
        CircularBufferConfig(a_tiles * kTileBytes, {{CBIndex::c_0, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_0, kTileBytes));
    CreateCircularBuffer(
        program, cores,
        CircularBufferConfig(b_tiles * kTileBytes, {{CBIndex::c_1, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_1, kTileBytes));
    const std::map<uint8_t, DataFormat> out_formats = {
        {CBIndex::c_16, DataFormat::Float16_b}, {CBIndex::c_24, DataFormat::Float16_b}};
    CreateCircularBuffer(
        program, cores,
        CircularBufferConfig(out_tiles * kTileBytes, out_formats)
            .set_page_size(CBIndex::c_16, kTileBytes)
            .set_page_size(CBIndex::c_24, kTileBytes));
}

// Create the three kernels on `cores`.
KernelIds create_kernels(
    Program& program, const CoreRangeSet& cores, const BufferPtr& a_buf, const BufferPtr& b_buf,
    const BufferPtr& c_buf, const std::vector<uint32_t>& compute_args) {
    std::vector<uint32_t> reader_accessors;
    TensorAccessorArgs(*a_buf).append_to(reader_accessors);
    TensorAccessorArgs(*b_buf).append_to(reader_accessors);
    std::vector<uint32_t> writer_accessors;
    TensorAccessorArgs(*c_buf).append_to(writer_accessors);
    KernelIds ids{};
    ids.reader = CreateKernel(
        program, kReaderKernel, cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_accessors});
    ids.writer = CreateKernel(
        program, kWriterKernel, cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_accessors});
    ids.compute = CreateKernel(
        program, kComputeKernel, cores,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_args});
    return ids;
}

// Build the program: one core per block, taken row-wise from (0, 0) over the grid as the upstream host takes them
// (block number b on core (b mod grid width, b div grid width), blocks numbered across each block row in turn).
Program make_program(
    distributed::MeshDevice& device, const BufferPtr& a_buf, const BufferPtr& b_buf, const BufferPtr& c_buf,
    const TileCounts& t, const BlockPlan& plan) {
    const CoreCoord grid = device.compute_with_storage_grid_size();
    const uint32_t across = t.nt / plan.block_cols;
    const uint32_t blocks = (t.mt / plan.block_rows) * across;
    const CoreRangeSet cores = num_cores_to_corerangeset(blocks, grid, true);
    Program program = CreateProgram();
    create_circular_buffers(program, cores, plan);
    const KernelIds ids = create_kernels(program, cores, a_buf, b_buf, c_buf, compute_compile_args(t, plan));
    const Addresses addr{
        static_cast<uint32_t>(a_buf->address()), static_cast<uint32_t>(b_buf->address()),
        static_cast<uint32_t>(c_buf->address())};
    for (uint32_t block = 0; block < blocks; ++block) {
        const CoreCoord core{block % grid.x, block / grid.x};
        const std::vector<uint32_t> reader = reader_runtime_args(addr, t, plan, block / across, block % across);
        const std::vector<uint32_t> writer = writer_runtime_args(addr.c, t, plan, block / across, block % across);
        SetRuntimeArgs(program, ids.reader, core, reader);
        SetRuntimeArgs(program, ids.writer, core, writer);
    }
    return program;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: matmul_multicore_reuse <a> <b> <c>\n");
        return 2;
    }
    lassi_io_array a = {};
    lassi_io_array b = {};
    if (read_inputs(argv[1], argv[2], &a, &b) != 0) {
        return 1;
    }
    const uint32_t m = static_cast<uint32_t>(a.dims[0]);
    const uint32_t k = static_cast<uint32_t>(a.dims[1]);
    const uint32_t n = static_cast<uint32_t>(b.dims[1]);
    const TileCounts t{m / TILE_HEIGHT, k / TILE_WIDTH, n / TILE_WIDTH};

    // Stage the inputs in tile order.
    std::vector<bfloat16> a_tiles = tiled(a, m, k);
    std::vector<bfloat16> b_tiles = tiled(b, k, n);

    std::shared_ptr<distributed::MeshDevice> device = distributed::MeshDevice::create_unit_mesh(0);
    BlockPlan plan{};
    if (!plan_on_device(*device, t, &plan)) {
        device->close();
        lassi_io_free(&a);
        lassi_io_free(&b);
        return 1;
    }
    distributed::MeshCommandQueue& cq = device->mesh_command_queue();
    BufferPtr a_buf = dram_buffer(*device, t.mt * t.kt);
    BufferPtr b_buf = dram_buffer(*device, t.kt * t.nt);
    BufferPtr c_buf = dram_buffer(*device, t.mt * t.nt);
    distributed::EnqueueWriteMeshBuffer(cq, a_buf, a_tiles, false);
    distributed::EnqueueWriteMeshBuffer(cq, b_buf, b_tiles, false);

    distributed::MeshWorkload workload;
    Program program = make_program(*device, a_buf, b_buf, c_buf, t, plan);
    workload.add_program(distributed::MeshCoordinateRange(device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);

    // Read the result tiles back from c's own buffer and put them in row-major order.
    std::vector<bfloat16> c_tiles(static_cast<size_t>(m) * n);
    distributed::EnqueueReadMeshBuffer(cq, c_tiles, c_buf, true);
    std::vector<bfloat16> c = untilize_nfaces(c_tiles, m, n);
    const bool closed = device->close();

    int status = 1;
    if (!closed) {
        std::fprintf(stderr, "matmul_multicore_reuse: the device did not close cleanly\n");
    } else {
        const uint64_t dims[2] = {a.dims[0], b.dims[1]};
        status = lassi_io_write(argv[3], "c", LASSI_IO_BF16, 2u, dims, c.data());
    }
    lassi_io_free(&a);
    lassi_io_free(&b);
    return status;
}
