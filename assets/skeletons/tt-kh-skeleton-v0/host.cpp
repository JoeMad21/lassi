// tt-host-skeleton-v0: an item-agnostic tt-metal host program skeleton for tt-metal commit 5280a9cf, in C++20.
// Written from the pinned tt-metalium headers at that commit (tt_metal/api/tt-metalium, the declarations of the
// tt-metalium-5280a9cf-full API pack) and the host programs of tt-metal's add_2_integers_in_riscv and shard_data_rm
// programming examples at that commit. It solves no task: every call has the form the pinned headers declare, and
// each block from a BLANK line to its END BLANK line holds placeholder code (one input, one output, and a reader, a
// compute, and a writer kernel on one core) so the file compiles as written. Fill or extend only those blocks.

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>

#include "lassi_io.h"

using namespace tt::tt_metal;

// One tile is 32 x 32 bf16 values: 1024 elements, a 2048-byte page of a buffer or a circular buffer.
constexpr uint32_t TILE_ELEMENTS = 32 * 32;
constexpr uint32_t TILE_BYTES = TILE_ELEMENTS * 2;

// The cores a circular buffer or a kernel is placed on: one core, a CoreRange rectangle, or a CoreRangeSet.
using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;

// BLANK io: the number of input files and of output files, in the order the source program takes them.
constexpr int NUM_INPUTS = 1;
constexpr int NUM_OUTPUTS = 1;
// END BLANK io

// BLANK layout: true when the kernels read and write 32x32 tiles in tile order (a matmul's do), false when they
// treat a buffer as a flat run of values.
constexpr bool TILED = false;
// END BLANK layout

// Return the number of whole tiles that hold `count` values.
static uint32_t tiles_for(uint64_t count) {
    return static_cast<uint32_t>((count + TILE_ELEMENTS - 1) / TILE_ELEMENTS);
}

// Return an input's values as bf16 bits, zero-padded to `tiles` whole tiles; an f32 value keeps its upper 16 bits.
static std::vector<uint16_t> to_bf16(const lassi_io_array& array, uint32_t tiles) {
    std::vector<uint16_t> values(static_cast<size_t>(tiles) * TILE_ELEMENTS, 0);
    for (uint64_t i = 0; i < array.count && i < values.size(); ++i) {
        uint32_t bits = 0;
        if (array.dtype == LASSI_IO_BF16) {
            bits = static_cast<uint32_t>(static_cast<const uint16_t*>(array.data)[i]) << 16;
        } else if (array.dtype == LASSI_IO_F32) {
            memcpy(&bits, static_cast<const float*>(array.data) + i, sizeof(bits));
        }
        values[i] = static_cast<uint16_t>(bits >> 16);
    }
    return values;
}

// Create a MeshBuffer of `bytes` bytes (whole tiles) on `device`, one tile per page: DRAM, or L1 for a kernel's
// scratch. Keep the result in a named std::shared_ptr variable: the write and read calls take it by reference.
static std::shared_ptr<distributed::MeshBuffer> make_buffer(
    distributed::MeshDevice* device, uint32_t bytes, BufferType type = BufferType::DRAM) {
    distributed::DeviceLocalBufferConfig local{.page_size = TILE_BYTES, .buffer_type = type};
    distributed::ReplicatedBufferConfig replicated{.size = bytes};
    return distributed::MeshBuffer::create(replicated, local, device);
}

// Create circular buffer `index` of `tiles` bf16 tiles on `cores` of `program`, one tile per page.
static CBHandle make_cb(Program& program, const CoreSpec& cores, tt::CBIndex index, uint32_t tiles) {
    CircularBufferConfig config = CircularBufferConfig(tiles * TILE_BYTES, {{index, tt::DataFormat::Float16_b}})
                                      .set_page_size(index, TILE_BYTES);
    return CreateCircularBuffer(program, cores, config);
}

// Write `values` (bf16 bits read back from the device) to `path` as array `name` of `dtype`, LASSI_IO_BF16 as they
// are or LASSI_IO_F32 as the f32 of each value; `values` holds at least the product of `dims` elements.
static int write_output(const char* path, const char* name, uint32_t dtype, uint32_t rank, const uint64_t* dims,
                        const std::vector<uint16_t>& values) {
    if (dtype == LASSI_IO_BF16) {
        return lassi_io_write(path, name, dtype, rank, dims, values.data());
    }
    if (dtype != LASSI_IO_F32) {
        fprintf(stderr, "%s: write_output writes LASSI_IO_BF16 and LASSI_IO_F32 arrays only\n", path);
        return 1;
    }
    std::vector<float> wide(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        uint32_t bits = static_cast<uint32_t>(values[i]) << 16;
        memcpy(&wide[i], &bits, sizeof(bits));
    }
    return lassi_io_write(path, name, dtype, rank, dims, wide.data());
}

// One function: blanks in one place in prompt order; device reads and output conversion in the body the guard reads.
int main(int argc, char** argv) {
    // Take the input files, then the output files, from the arguments.
    if (argc != 1 + NUM_INPUTS + NUM_OUTPUTS) {
        fprintf(stderr, "usage: %s <%d input files> <%d output files>\n", argv[0], NUM_INPUTS, NUM_OUTPUTS);
        return 1;
    }
    const char* const* input_paths = argv + 1;
    const char* const* output_paths = argv + 1 + NUM_INPUTS;

    // BLANK inputs: one zeroed lassi_io_array variable per input file, read through the harness header in the order
    // the source program takes them; each stays allocated until every output is written.
    lassi_io_array in0 = {};
    if (lassi_io_read(input_paths[0], &in0) != 0) {
        return 1;
    }
    // END BLANK inputs

    // BLANK sizes: the counts the buffers and kernels need, from the inputs' count, rank, and dims (here every buffer
    // holds as many values as input 0, in whole tiles).
    const uint32_t n_tiles = tiles_for(in0.count);
    // END BLANK sizes

    // Open device 0 as a 1x1 mesh and take its command queue.
    std::shared_ptr<distributed::MeshDevice> device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& queue = device->mesh_command_queue();

    // BLANK buffers: one DRAM buffer per input and per output, in whole tiles, each through make_buffer; a kernel's
    // L1 scratch is make_buffer(device.get(), bytes, BufferType::L1).
    std::shared_ptr<distributed::MeshBuffer> in0_buffer = make_buffer(device.get(), n_tiles * TILE_BYTES);
    std::shared_ptr<distributed::MeshBuffer> out0_buffer = make_buffer(device.get(), n_tiles * TILE_BYTES);
    // END BLANK buffers

    // BLANK writes: each input as bf16 bits in whole tiles, tilized when TILED as a rows x cols matrix (both multiples
    // of 32), then written to its buffer; each vector stays alive until Finish.
    std::vector<uint16_t> in0_values = to_bf16(in0, n_tiles);
    if (TILED) {
        in0_values = tilize_nfaces(in0_values, n_tiles * 32, 32);
    }
    distributed::EnqueueWriteMeshBuffer(queue, in0_buffer, in0_values, false);
    // END BLANK writes

    // Create the program.
    Program program = CreateProgram();

    // BLANK cores: the cores the program runs on: one core here; CoreRange({x0, y0}, {x1, y1}) for a rectangle.
    CoreCoord core = {0, 0};
    // END BLANK cores

    // BLANK circular_buffers: one make_cb per circular buffer index a kernel reads or writes, with its depth in tiles
    // (c_0 and c_1 for inputs, c_16 for the output, others for intermediates).
    make_cb(program, core, tt::CBIndex::c_0, 2);
    make_cb(program, core, tt::CBIndex::c_16, 2);
    // END BLANK circular_buffers

    // BLANK kernels: each kernel the harness provides, by its path exactly as the prompt lists it, with the compile
    // args in the order the kernel reads them: readers and writers through DataMovementConfig (a reader on RISCV_1,
    // a writer on RISCV_0) with a buffer's TensorAccessorArgs appended, compute kernels through ComputeConfig.
    std::vector<uint32_t> reader_args = {};
    TensorAccessorArgs(*in0_buffer).append_to(reader_args);
    KernelHandle reader = CreateKernel(
        program,
        "kernels/dataflow/reader.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_args});
    std::vector<uint32_t> compute_args = {n_tiles};
    KernelHandle compute = CreateKernel(
        program,
        "kernels/compute/compute.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_args});
    std::vector<uint32_t> writer_args = {};
    TensorAccessorArgs(*out0_buffer).append_to(writer_args);
    KernelHandle writer = CreateKernel(
        program,
        "kernels/dataflow/writer.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_args});
    // END BLANK kernels

    // BLANK runtime_args: each kernel's runtime args on each of its cores, in the order the kernel reads them
    // (buffer addresses and counts); on several cores, one SetRuntimeArgs per CoreCoord with that core's share.
    SetRuntimeArgs(program, reader, core, {static_cast<uint32_t>(in0_buffer->address()), n_tiles});
    SetRuntimeArgs(program, compute, core, {n_tiles});
    SetRuntimeArgs(program, writer, core, {static_cast<uint32_t>(out0_buffer->address()), n_tiles});
    // END BLANK runtime_args

    // Run the program on the mesh and wait for it to finish.
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(queue, workload, false);
    distributed::Finish(queue);

    // BLANK reads: read each output buffer back (blocking), untilized when TILED as the rows x cols it was written.
    std::vector<uint16_t> out0_values;
    distributed::EnqueueReadMeshBuffer(queue, out0_values, out0_buffer, true);
    if (TILED) {
        out0_values = untilize_nfaces(out0_values, n_tiles * 32, 32);
    }
    // END BLANK reads

    // Close the device.
    device->close();

    // BLANK outputs: write each output with the array name, dtype, rank, and dims the source program writes (here
    // input 0's shape); an output of another shape passes its own uint64_t dims array.
    int status = 0;
    status |= write_output(output_paths[0], "out0", LASSI_IO_BF16, in0.rank, in0.dims, out0_values);
    // END BLANK outputs

    // BLANK free: lassi_io_free for each input, only here, after every output is written (it zeroes the array's
    // rank and dims).
    lassi_io_free(&in0);
    // END BLANK free
    return status;
}
