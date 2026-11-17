// Written for LASSI-DF tt-par-v0 (task P17.67); runs the unmodified upstream vecadd_sharding kernels from tt-metal 5280a9cf.
//
// Adds two bf16 vectors element by element on the device, c = a + b, with a, b, and c each height-sharded over the
// L1 memory of a fixed block of 56 Tensix cores, 8 wide and 7 tall (cores (0, 0) to (7, 6)): core k of the block,
// in row-major order, holds tiles k * T to (k + 1) * T - 1 of each vector, T tiles per core. Each core's three
// circular buffers sit at its own shards' addresses, so the compute kernel, the only kernel, adds its tiles of a and
// b in place and packs each sum into its shard of c; no reader or writer kernel runs. The host writes a and b to
// their sharded buffers, reads c back in element order, widens each bf16 sum to f32, and writes it.
//
// Usage: vecadd_sharding <a> <b> <c>
//   a, b  lassi_io input files, each a bf16 array; both hold the same number of elements, a nonzero multiple of
//         57344 (56 cores x 1024, so every core holds the same whole number of 32 x 32 tiles), at most 64 tiles
//         per core
//   c     the lassi_io output file: an f32 array named "c" with a's shape
// The device's compute grid must be at least 8 x 7. The arrays are never tilized: an elementwise add does not depend
// on where an element sits inside a tile, so each run of 1024 elements fills one tile page as it is.
//
// Device settings: three L1 buffers, each HEIGHT_SHARDED over the block in row-major core order, with a page of
// 32 x 32 elements, the vector laid out as one column of tile pages, and a shard of T tiles in that column; three
// circular buffers c_0 and c_1 (in) and c_2 (out), each Float16_b, one tile per page, T tiles long, and set to the
// address of a's, b's, and c's buffer; the compute kernel with math fidelity HiFi4, math_approx_mode false, and
// compile args [c_0, c_1, c_2], and T as its one runtime arg on every core of the block. No other ComputeConfig
// field is set, so fp32_dest_acc_en stays off and unpack_to_dest_mode stays empty.
// A tt-metal error is an exception that ends the program through std::terminate.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>

#include "lassi_io.h"

using namespace tt;
using namespace tt::tt_metal;

namespace {

// A tile is 32 x 32 elements; every element on the device is a 2-byte bf16, so a tile page takes 2048 bytes.
constexpr uint32_t kTileEdge = 32u;
constexpr uint64_t kTileElements = static_cast<uint64_t>(kTileEdge) * kTileEdge;
constexpr uint32_t kTileBytes = static_cast<uint32_t>(kTileElements * sizeof(uint16_t));
// The block the buffers are sharded over: 8 cores wide (x 0 to 7) by 7 tall (y 0 to 6). Card 0's 8 x 7 compute
// grid (its device log, rx 20261116-212311-tracy-0bc3) holds it whole, as 8 x 8 and 8 x 9 grids do.
constexpr uint32_t kBlockWidth = 8u;
constexpr uint32_t kBlockHeight = 7u;
constexpr uint32_t kBlockCores = kBlockWidth * kBlockHeight;
// The most tiles one core may hold: three shards of 64 tiles take 384 KiB of a core's L1.
constexpr uint64_t kMaxTilesPerCore = 64u;

// The upstream kernel, at the path the build places it, relative to the working directory.
constexpr const char* kComputeKernel = "vecadd_sharding/kernels/add_sharding.cpp";

using BufferPtr = std::shared_ptr<distributed::MeshBuffer>;

// The f32 value of a bf16 bit pattern: the pattern is the upper half of that f32.
float widen_bf16(uint16_t bits) {
    const uint32_t word = static_cast<uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &word, sizeof value);
    return value;
}

// Return how many tiles each core of the block holds, or 0 after saying on stderr why a and b cannot be added here.
uint32_t tiles_per_core(const lassi_io_array& a, const lassi_io_array& b) {
    if (a.dtype != LASSI_IO_BF16 || b.dtype != LASSI_IO_BF16 || a.count != b.count) {
        std::fprintf(stderr, "vecadd_sharding: a and b must be bf16 arrays with one element count\n");
        return 0;
    }
    const uint64_t block_elements = kTileElements * kBlockCores;
    if (a.count == 0u || a.count % block_elements != 0u || a.count / block_elements > kMaxTilesPerCore) {
        std::fprintf(stderr,
                     "vecadd_sharding: the element count must be a nonzero multiple of 57344 (56 cores x 1024), at "
                     "most 64 tiles per core\n");
        return 0;
    }
    return static_cast<uint32_t>(a.count / block_elements);
}

// Return whether the device's compute grid holds the block, saying on stderr why when it does not.
bool grid_holds_block(distributed::MeshDevice& device) {
    const CoreCoord grid = device.compute_with_storage_grid_size();
    if (grid.x >= kBlockWidth && grid.y >= kBlockHeight) {
        return true;
    }
    std::fprintf(stderr, "vecadd_sharding: the compute grid is %llu x %llu, smaller than the 8 x 7 block\n",
                 static_cast<unsigned long long>(grid.x), static_cast<unsigned long long>(grid.y));
    return false;
}

// Create an L1 buffer of `tiles` tile pages, height-sharded over `cores`: the vector is a column of `tiles` pages,
// and each core, in row-major order, holds the next `per_core` of them.
BufferPtr sharded_l1_buffer(
    distributed::MeshDevice& device, const CoreRangeSet& cores, uint32_t tiles, uint32_t per_core) {
    const std::array<uint32_t, 2> shard_shape = {per_core * kTileEdge, kTileEdge};
    const std::array<uint32_t, 2> page_shape = {kTileEdge, kTileEdge};
    const std::array<uint32_t, 2> shape_in_pages = {tiles, 1u};
    const ShardSpecBuffer spec(cores, shard_shape, ShardOrientation::ROW_MAJOR, page_shape, shape_in_pages);
    distributed::DeviceLocalBufferConfig local{
        .page_size = kTileBytes,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(spec, TensorMemoryLayout::HEIGHT_SHARDED)};
    distributed::ReplicatedBufferConfig whole{.size = static_cast<uint64_t>(kTileBytes) * tiles};
    return distributed::MeshBuffer::create(whole, local, &device);
}

// Create circular buffer `index` on `cores`: Float16_b, one tile per page, `per_core` tiles long, at the address of
// `shards` on each core, so the kernel reads or writes that core's shard in place.
void shard_circular_buffer(
    Program& program, const CoreRangeSet& cores, CBIndex index, uint32_t per_core,
    const distributed::MeshBuffer& shards) {
    CreateCircularBuffer(
        program, cores,
        CircularBufferConfig(per_core * kTileBytes, {{index, DataFormat::Float16_b}})
            .set_page_size(index, kTileBytes)
            .set_globally_allocated_address(*shards.get_backing_buffer()));
}

// Build the program: the three circular buffers on the block, and the compute kernel there with its compile-time
// circular buffer ids and the tiles per core as its one runtime arg, the same on every core.
Program make_program(
    const CoreRangeSet& cores, const BufferPtr& a_buf, const BufferPtr& b_buf, const BufferPtr& c_buf,
    uint32_t per_core) {
    Program program = CreateProgram();
    shard_circular_buffer(program, cores, CBIndex::c_0, per_core, *a_buf);
    shard_circular_buffer(program, cores, CBIndex::c_1, per_core, *b_buf);
    shard_circular_buffer(program, cores, CBIndex::c_2, per_core, *c_buf);
    const std::vector<uint32_t> compile_args = {
        static_cast<uint32_t>(CBIndex::c_0), static_cast<uint32_t>(CBIndex::c_1), static_cast<uint32_t>(CBIndex::c_2)};
    const KernelHandle adder = CreateKernel(
        program, kComputeKernel, cores,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false, .compile_args = compile_args});
    SetRuntimeArgs(program, adder, cores, {per_core});
    return program;
}

}  // namespace

// Over 60 lines: the device read and output conversion stay in main's own body, as PHASE-NOTES' P4.13 guard hint asks.
int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: vecadd_sharding <a> <b> <c>\n");
        return 2;
    }
    lassi_io_array a = {};
    lassi_io_array b = {};
    if (lassi_io_read(argv[1], &a) != 0 || lassi_io_read(argv[2], &b) != 0) {
        lassi_io_free(&a);
        lassi_io_free(&b);
        return 1;
    }
    const uint32_t per_core = tiles_per_core(a, b);
    if (per_core == 0u) {
        lassi_io_free(&a);
        lassi_io_free(&b);
        return 1;
    }
    const uint32_t n_tiles = per_core * kBlockCores;

    // Host copies of the inputs in the device's element format, bf16, which the files already hold.
    std::vector<uint16_t> a_host(a.count);
    std::vector<uint16_t> b_host(b.count);
    std::memcpy(a_host.data(), a.data, a.nbytes);
    std::memcpy(b_host.data(), b.data, b.nbytes);

    std::shared_ptr<distributed::MeshDevice> device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = device->mesh_command_queue();
    if (!grid_holds_block(*device)) {
        static_cast<void>(device->close());
        lassi_io_free(&a);
        lassi_io_free(&b);
        return 1;
    }
    const CoreRangeSet cores(CoreRange({0, 0}, {kBlockWidth - 1, kBlockHeight - 1}));
    // Three sharded L1 buffers of n_tiles tile pages: the two inputs and the result, which is its own buffer.
    BufferPtr a_buf = sharded_l1_buffer(*device, cores, n_tiles, per_core);
    BufferPtr b_buf = sharded_l1_buffer(*device, cores, n_tiles, per_core);
    BufferPtr c_buf = sharded_l1_buffer(*device, cores, n_tiles, per_core);
    distributed::EnqueueWriteMeshBuffer(cq, a_buf, a_host, false);
    distributed::EnqueueWriteMeshBuffer(cq, b_buf, b_host, false);

    distributed::MeshWorkload workload;
    Program program = make_program(cores, a_buf, b_buf, c_buf, per_core);
    workload.add_program(distributed::MeshCoordinateRange(device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    // Read the sums back from c's own buffer in element order; the read sizes the vector to the buffer.
    std::vector<uint16_t> c_device;
    distributed::EnqueueReadMeshBuffer(cq, c_device, c_buf, true);
    const bool closed = device->close();

    int status = 1;
    if (!closed) {
        std::fprintf(stderr, "vecadd_sharding: the device did not close cleanly\n");
    } else if (c_device.size() != a.count) {
        std::fprintf(stderr, "vecadd_sharding: the device returned a result of the wrong size\n");
    } else {
        // Widen each bf16 sum to f32 for the output file; the widening is exact.
        std::vector<float> c(a.count);
        for (uint64_t i = 0; i < a.count; ++i) {
            c[i] = widen_bf16(c_device[i]);
        }
        status = lassi_io_write(argv[3], "c", LASSI_IO_F32, a.rank, a.dims, c.data());
    }
    lassi_io_free(&a);
    lassi_io_free(&b);
    return status == 0 ? 0 : 1;
}
