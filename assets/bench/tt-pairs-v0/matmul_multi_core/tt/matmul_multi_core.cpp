// Written for LASSI-DF tt-pairs-v0 (task P4.13); runs the unmodified upstream matmul_multi_core kernels from tt-metal 5280a9cf.
//
// Matrix multiply on the device, c = a x b, with the output tiles shared out over the compute grid: each working
// core gets a run of consecutive output tiles (split_work_to_cores), its reader streams the matching tiles of a and
// b, its compute kernel sums the K-dimension tile products for each output tile, and its writer stores the results.
//
// Usage: matmul_multi_core <a> <b> <c>
//   a  lassi_io file, bf16, rank 2, [M, K]
//   b  lassi_io file, bf16, rank 2, [K, N]
//   c  lassi_io file written here, bf16, rank 2, [M, N]: the device result in row-major order
// M, K, and N must be positive multiples of 32 (whole tiles), at most 65536. The inputs are already bf16, so the
// host converts no value; it only reorders a and b into tiles for the device and the result back into rows.
//
// Device settings, as the upstream example sets them: three circular buffers of two tiles each (c_0 and c_1 in,
// c_16 out), all Float16_b, on every working core; the reader on RISCV_1 and the writer on RISCV_0, each given its
// buffers' TensorAccessorArgs as compile args; the compute kernel with math fidelity HiFi4 and empty compile args.
// No other ComputeConfig field is set, so fp32_dest_acc_en stays off and unpack_to_dest_mode stays empty.
// A tt-metal error is an exception that ends the program through std::terminate.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <tuple>
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
// Each circular buffer holds two tiles.
constexpr uint32_t kBufferTiles = 2;
// The largest accepted dim. It keeps each dim, each tile count (at most 2048 along M, K, or N), and each product of
// two tile counts (at most 2^22) inside 32 bits, as the uint32 TileCounts fields and the uint32 runtime arguments
// that carry them need. Buffer sizes in bytes are computed in 64 bits: at this dim one buffer takes 8 GiB.
constexpr uint64_t kMaxDim = 65536;

// The upstream kernels, at the paths the build places them, relative to the working directory.
constexpr const char* kReaderKernel = "kernels/dataflow/reader_mm_output_tiles_partitioned.cpp";
constexpr const char* kWriterKernel = "kernels/dataflow/writer_unary_interleaved_start_id.cpp";
constexpr const char* kComputeKernel = "kernels/compute/mm.cpp";

using BufferPtr = std::shared_ptr<distributed::MeshBuffer>;

// Tile counts along M, K, and N.
struct TileCounts {
    uint32_t mt;
    uint32_t kt;
    uint32_t nt;
};

struct KernelIds {
    KernelHandle reader;
    KernelHandle writer;
    KernelHandle compute;
};

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
        std::fprintf(stderr, "matmul_multi_core: %s: %s\n", path, problem);
        lassi_io_free(x);
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

// Create the circular buffers and the three kernels on `cores`.
KernelIds create_kernels(
    Program& program, const CoreRangeSet& cores, const BufferPtr& a_buf, const BufferPtr& b_buf,
    const BufferPtr& c_buf) {
    for (const CBIndex index : {CBIndex::c_0, CBIndex::c_1, CBIndex::c_16}) {
        CreateCircularBuffer(
            program, cores,
            CircularBufferConfig(kBufferTiles * kTileBytes, {{index, DataFormat::Float16_b}})
                .set_page_size(index, kTileBytes));
    }
    std::vector<uint32_t> reader_args;
    TensorAccessorArgs(*a_buf).append_to(reader_args);
    TensorAccessorArgs(*b_buf).append_to(reader_args);
    std::vector<uint32_t> writer_args;
    TensorAccessorArgs(*c_buf).append_to(writer_args);
    KernelIds ids{};
    ids.reader = CreateKernel(
        program, kReaderKernel, cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_args});
    ids.writer = CreateKernel(
        program, kWriterKernel, cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_args});
    ids.compute = CreateKernel(
        program, kComputeKernel, cores, ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = {}});
    return ids;
}

// Build the program: split the Mt x Nt output tiles over the device's compute grid, create the kernels on the
// working cores, and give each core its runtime args. Cores are walked group 1 then group 2, range by range and
// core by core, and each takes the next `per_core` output tiles from `first_tile` on.
Program make_program(
    distributed::MeshDevice& device, const BufferPtr& a_buf, const BufferPtr& b_buf, const BufferPtr& c_buf,
    const TileCounts& t) {
    const auto split = split_work_to_cores(device.compute_with_storage_grid_size(), t.mt * t.nt);
    Program program = CreateProgram();
    const KernelIds ids = create_kernels(program, std::get<1>(split), a_buf, b_buf, c_buf);
    const uint32_t a_addr = static_cast<uint32_t>(a_buf->address());
    const uint32_t b_addr = static_cast<uint32_t>(b_buf->address());
    const uint32_t c_addr = static_cast<uint32_t>(c_buf->address());
    const std::pair<const CoreRangeSet*, uint32_t> groups[2] = {
        {&std::get<2>(split), std::get<4>(split)}, {&std::get<3>(split), std::get<5>(split)}};
    uint32_t first_tile = 0;
    for (const auto& group : groups) {
        const uint32_t per_core = group.second;
        for (const CoreRange& range : group.first->ranges()) {
            for (const CoreCoord& core : range) {
                SetRuntimeArgs(program, ids.reader, core, {a_addr, b_addr, t.mt, t.kt, t.nt, first_tile, per_core});
                SetRuntimeArgs(program, ids.writer, core, {c_addr, per_core, first_tile});
                SetRuntimeArgs(program, ids.compute, core, {per_core, t.kt});
                first_tile += per_core;
            }
        }
    }
    return program;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: matmul_multi_core <a> <b> <c>\n");
        return 2;
    }
    lassi_io_array a = {};
    lassi_io_array b = {};
    if (read_matrix(argv[1], &a) != 0) {
        return 1;
    }
    if (read_matrix(argv[2], &b) != 0) {
        lassi_io_free(&a);
        return 1;
    }
    if (a.dims[1] != b.dims[0]) {
        std::fprintf(stderr, "matmul_multi_core: a has %llu columns but b has %llu rows\n",
                     static_cast<unsigned long long>(a.dims[1]), static_cast<unsigned long long>(b.dims[0]));
        lassi_io_free(&a);
        lassi_io_free(&b);
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
    distributed::MeshCommandQueue& cq = device->mesh_command_queue();
    BufferPtr a_buf = dram_buffer(*device, t.mt * t.kt);
    BufferPtr b_buf = dram_buffer(*device, t.kt * t.nt);
    BufferPtr c_buf = dram_buffer(*device, t.mt * t.nt);
    distributed::EnqueueWriteMeshBuffer(cq, a_buf, a_tiles, false);
    distributed::EnqueueWriteMeshBuffer(cq, b_buf, b_tiles, false);

    distributed::MeshWorkload workload;
    Program program = make_program(*device, a_buf, b_buf, c_buf, t);
    workload.add_program(distributed::MeshCoordinateRange(device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);

    // Read the result tiles back from c's own buffer and put them in row-major order.
    std::vector<bfloat16> c_tiles(static_cast<size_t>(m) * n);
    distributed::EnqueueReadMeshBuffer(cq, c_tiles, c_buf, true);
    std::vector<bfloat16> c = untilize_nfaces(c_tiles, m, n);
    const bool closed = device->close();

    int status = 1;
    if (!closed) {
        std::fprintf(stderr, "matmul_multi_core: the device did not close cleanly\n");
    } else {
        const uint64_t dims[2] = {a.dims[0], b.dims[1]};
        status = lassi_io_write(argv[3], "c", LASSI_IO_BF16, 2u, dims, c.data());
    }
    lassi_io_free(&a);
    lassi_io_free(&b);
    return status;
}
