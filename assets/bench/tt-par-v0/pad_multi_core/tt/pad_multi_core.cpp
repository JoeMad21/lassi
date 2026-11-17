// Written for LASSI-DF tt-par-v0 (task P17.67); runs the unmodified upstream pad_multi_core kernels from tt-metal 5280a9cf.
//
// Right padding on the device: every row of a row-major bf16 matrix is copied and then filled out to 256 columns
// with the value 2.0. The rows are shared out over the compute grid in whole rows (split_work_to_cores); on each
// working core a reader kernel lays out one padded row at a time in L1 and a writer kernel stores it to DRAM.
//
// Usage: pad_multi_core <src> <dst>
//   src  lassi_io file, bf16, rank 2, [rows, cols]: rows from 1 to 65536, cols even, from 2 to 256
//   dst  lassi_io file written here, bf16, rank 2, [rows, 256], row-major: the first cols values of each row are
//        src's row, the rest are 2.0
// The data stay row-major: nothing is tilized, converted, or checked against the input.
//
// How the kernels see the data. Both DRAM buffers are paged by stick: one 4-byte word holding two adjacent
// bf16 values of a row, the first in the low half, which is how a row-major bf16 array already lies in memory on a
// little-endian host. A source row is cols / 2 sticks and a padded row 128. For each padded row the reader gives
// every stick a 64-byte slot in L1, writes the packed pad value into the slot's first word, and, when the stick
// lies inside the source row, reads the source stick over that word; the writer stores the first 4 bytes of each
// slot to the destination stick of the same index. Only one word per slot is ever set, so a stick must be exactly
// 4 bytes, and the source row width must be even.
//
// Device settings: one circular buffer configuration the size of one padded row (128 slots of 64 bytes) holds c_0
// (the slots, 64-byte pages), c_1 (the pad pattern, 64-byte pages), and c_2 (an alignment scratch, 4-byte pages) in
// one shared L1 region, as the upstream example lays them out, each Float16_b (upstream names UInt32; no compute
// kernel exists to read either format). The reader runs on RISCV_0 and the writer on RISCV_1, each given its
// compile args and then its DRAM buffer's TensorAccessorArgs. No compute kernel is created.
// A tt-metal error is an exception that ends the program through std::terminate.
#include <cstdint>
#include <cstdio>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "lassi_io.h"

using namespace tt;
using namespace tt::tt_metal;

namespace {

// The width every row is padded to, and the pad value 2.0 as a bf16 bit pattern.
constexpr uint64_t kPaddedCols = 256;
constexpr uint16_t kPadBits = 0x4000;
// The most rows accepted. It keeps every stick count and stick index inside the uint32 kernel arguments, and the
// destination buffer at most 32 MiB of data (each 4-byte page takes a 32-byte DRAM slot, so 256 MiB of DRAM).
constexpr uint64_t kMaxRows = 65536;

// A stick: one 4-byte word of two bf16 values, the page of both DRAM buffers and the size of every NoC transfer.
constexpr uint32_t kStickBytes = 4;
constexpr uint32_t kValuesPerStick = kStickBytes / static_cast<uint32_t>(sizeof(uint16_t));
// Sticks per padded row: the reader's batch, and the number of slots in the circular buffer.
constexpr uint32_t kPaddedSticks = static_cast<uint32_t>(kPaddedCols) / kValuesPerStick;
// The L1 slot each stick gets in c_0, which is also the length of the pad pattern the reader writes into c_1.
constexpr uint32_t kSlotBytes = 64;
// The pad value twice over, as the reader expects it: one word, the same bf16 pattern in both halves.
constexpr uint32_t kPackedPad = (static_cast<uint32_t>(kPadBits) << 16) | kPadBits;

// The upstream kernels, at the paths the build places them, relative to the working directory.
constexpr const char* kReaderKernel = "pad_multi_core/kernels/pad_reader_dims_rm_interleaved.cpp";
constexpr const char* kWriterKernel = "pad_multi_core/kernels/pad_writer_dims_rm_interleaved.cpp";

using BufferPtr = std::shared_ptr<distributed::MeshBuffer>;

// The source matrix's shape as the kernels count it.
struct PadShape {
    uint32_t rows;        // rows of the source and of the result
    uint32_t src_sticks;  // sticks per source row: cols / 2
};

struct KernelIds {
    KernelHandle reader;
    KernelHandle writer;
};

// Read the lassi_io file at path into *x and check that it holds a bf16 matrix the kernels can pad: 1 to kMaxRows
// rows and an even number of columns from 2 to kPaddedCols; return 0, or print why and return 1 with *x released.
int read_source(const char* path, lassi_io_array* x) {
    if (lassi_io_read(path, x) != 0) {
        return 1;
    }
    const char* problem = nullptr;
    if (x->dtype != LASSI_IO_BF16) {
        problem = "the array must be bf16";
    } else if (x->rank != 2u) {
        problem = "the array must have rank 2";
    } else if (x->dims[0] == 0u || x->dims[0] > kMaxRows) {
        problem = "the row count must be from 1 to 65536";
    } else if (x->dims[1] == 0u || x->dims[1] % kValuesPerStick != 0u || x->dims[1] > kPaddedCols) {
        problem = "the column count must be even, from 2 to 256";
    }
    if (problem != nullptr) {
        std::fprintf(stderr, "pad_multi_core: %s: %s\n", path, problem);
        lassi_io_free(x);
        return 1;
    }
    return 0;
}

// Create a DRAM buffer of `sticks` pages of one stick each.
BufferPtr stick_buffer(distributed::MeshDevice& device, uint32_t sticks) {
    distributed::DeviceLocalBufferConfig local{.page_size = kStickBytes, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig whole{.size = static_cast<uint64_t>(kStickBytes) * sticks};
    return distributed::MeshBuffer::create(whole, local, &device);
}

// The reader's compile args in the kernel's order, indices 0 to 19, then the source buffer's TensorAccessorArgs
// from index 20. The kernel counts a padded tensor as N x C x H sticks; the matrix is N = rows, C = sticks per row,
// H = 1, with no rows or front sticks added. Entries marked "not used" are read by the kernel but never acted on
// here; they carry the values the upstream example gives them.
std::vector<uint32_t> reader_compile_args(const PadShape& s, const BufferPtr& src_buf) {
    std::vector<uint32_t> args = {
        s.rows,         // 0: source rows; a stick is read only while the core's row counter is below this
        1u,             // 1: stick rows per source row
        s.src_sticks,   // 2: sticks per source row; a stick at or past this index in its row is padding
        kStickBytes,    // 3: bytes per stick, the source page size and the size of each read
        s.rows,         // 4: rows of the result (not used)
        1u,             // 5: stick rows per padded row
        kPaddedSticks,  // 6: sticks per padded row, where the kernel's column counter wraps
        kSlotBytes,     // 7: bytes of pad pattern written into c_1 before the first row
        0u,             // 8: bytes of padding in front of each stick: none, so the front-padding path is off
        0u,             // 9: (not used)
        1u,             // 10: (not used)
        kSlotBytes,     // 11: (not used)
        1u,             // 12: pad with the value in 13 rather than with zeros
        kPackedPad,     // 13: the packed pad value
        kSlotBytes,     // 14: (not used)
        0u,             // 15: (not used)
        0u,             // 16: (not used)
        1u,             // 17: (not used)
        kSlotBytes,     // 18: the stride from one stick's slot to the next in c_0
        0u,             // 19: read each stick straight into its slot (not through the c_2 scratch)
    };
    TensorAccessorArgs(*src_buf).append_to(args);
    return args;
}

// Create the shared circular buffer configuration and the two data-movement kernels on `cores`.
KernelIds create_kernels(
    Program& program, const CoreRangeSet& cores, const BufferPtr& src_buf, const BufferPtr& dst_buf,
    const PadShape& s) {
    CircularBufferConfig row_buffers =
        CircularBufferConfig(
            kPaddedSticks * kSlotBytes,
            {{CBIndex::c_0, DataFormat::Float16_b},
             {CBIndex::c_1, DataFormat::Float16_b},
             {CBIndex::c_2, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_0, kSlotBytes)
            .set_page_size(CBIndex::c_1, kSlotBytes)
            .set_page_size(CBIndex::c_2, kStickBytes);
    CreateCircularBuffer(program, cores, row_buffers);
    // The writer's compile args: the circular buffer it drains, the bytes it stores per stick, and the slot stride,
    // then the destination buffer's TensorAccessorArgs from index 3.
    std::vector<uint32_t> writer_args = {static_cast<uint32_t>(CBIndex::c_0), kStickBytes, kSlotBytes};
    TensorAccessorArgs(*dst_buf).append_to(writer_args);
    KernelIds ids{};
    ids.reader = CreateKernel(
        program, kReaderKernel, cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_args(s, src_buf)});
    ids.writer = CreateKernel(
        program, kWriterKernel, cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = writer_args});
    return ids;
}

// Build the program: split the rows over the device's compute grid, create the kernels on the working cores, and
// give each core its runtime args. Cores are walked group 1 then group 2, range by range and core by core, and each
// takes the next `per_core` rows from `first_row` on, so every core's share starts at a row boundary, as the
// reader's counters, which start at 0 on every core, require.
Program make_program(
    distributed::MeshDevice& device, const BufferPtr& src_buf, const BufferPtr& dst_buf, const PadShape& s) {
    const auto split = split_work_to_cores(device.compute_with_storage_grid_size(), s.rows);
    Program program = CreateProgram();
    const KernelIds ids = create_kernels(program, std::get<1>(split), src_buf, dst_buf, s);
    const uint32_t src_addr = static_cast<uint32_t>(src_buf->address());
    const uint32_t dst_addr = static_cast<uint32_t>(dst_buf->address());
    const std::pair<const CoreRangeSet*, uint32_t> groups[2] = {
        {&std::get<2>(split), std::get<4>(split)}, {&std::get<3>(split), std::get<5>(split)}};
    uint32_t first_row = 0;
    for (const auto& group : groups) {
        const uint32_t per_core = group.second;
        const uint32_t sticks = per_core * kPaddedSticks;
        for (const CoreRange& range : group.first->ranges()) {
            for (const CoreCoord& core : range) {
                // Reader: source address, padded sticks to produce, sticks per batch (one padded row), first source
                // stick, and no front padding along N, C, or H.
                SetRuntimeArgs(
                    program, ids.reader, core, {src_addr, sticks, kPaddedSticks, first_row * s.src_sticks, 0u, 0u, 0u});
                // Writer: destination address, sticks to store, sticks per batch, first destination stick.
                SetRuntimeArgs(program, ids.writer, core, {dst_addr, sticks, kPaddedSticks, first_row * kPaddedSticks});
                first_row += per_core;
            }
        }
    }
    return program;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: pad_multi_core <src> <dst>\n");
        return 2;
    }
    lassi_io_array src = {};
    if (read_source(argv[1], &src) != 0) {
        return 1;
    }
    const PadShape s{static_cast<uint32_t>(src.dims[0]), static_cast<uint32_t>(src.dims[1] / kValuesPerStick)};
    const uint64_t padded_count = src.dims[0] * kPaddedCols;

    // Stage the input's bf16 bit patterns for the device as they are: the row-major order is already stick order.
    const uint16_t* src_values = static_cast<const uint16_t*>(src.data);
    std::vector<uint16_t> staged(src_values, src_values + src.count);

    std::shared_ptr<distributed::MeshDevice> device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = device->mesh_command_queue();
    BufferPtr src_buf = stick_buffer(*device, s.rows * s.src_sticks);
    BufferPtr dst_buf = stick_buffer(*device, s.rows * kPaddedSticks);
    distributed::EnqueueWriteMeshBuffer(cq, src_buf, staged, false);

    distributed::MeshWorkload workload;
    Program program = make_program(*device, src_buf, dst_buf, s);
    workload.add_program(distributed::MeshCoordinateRange(device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    // Read the destination buffer back; the read sizes the vector to the buffer, and its stick order is row-major.
    std::vector<uint16_t> padded;
    distributed::EnqueueReadMeshBuffer(cq, padded, dst_buf, true);
    const bool closed = device->close();

    int status = 1;
    if (!closed) {
        std::fprintf(stderr, "pad_multi_core: the device did not close cleanly\n");
    } else if (padded.size() != padded_count) {
        std::fprintf(stderr, "pad_multi_core: the device returned a result of the wrong size\n");
    } else {
        const uint64_t dims[2] = {src.dims[0], kPaddedCols};
        status = lassi_io_write(argv[2], "dst", LASSI_IO_BF16, 2u, dims, padded.data());
    }
    lassi_io_free(&src);
    return status;
}
