// SYNTHETIC: written for tests/toolchains/test_ttmetal_guard.py; not tt-metal source
// A data-movement-only program: one data-movement kernel copies the staged input to the output buffer on the
// device, and the host reads the result back, converts it to float32, and writes it. No compute kernel config
// is named, so the guard tags the program and does not fail it.
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>

#include "lassi_io.h"

using namespace tt;
using namespace tt::tt_metal;

namespace {

// A bf16 value is the upper 16 bits of the float32 of the same value.
uint16_t to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof bits);
    return static_cast<uint16_t>(bits >> 16);
}

float from_bf16(uint16_t half) {
    uint32_t bits = static_cast<uint32_t>(half) << 16;
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof value);
    return value;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        return 2;
    }
    lassi_io_array a = {};
    if (lassi_io_read(argv[1], &a) != 0) {
        return 1;
    }
    const uint32_t n = static_cast<uint32_t>(a.count);
    const float* af = static_cast<const float*>(a.data);

    // Stage the input for the device as bf16.
    std::vector<uint16_t> a_bf(n);
    for (uint32_t i = 0; i < n; ++i) {
        a_bf[i] = to_bf16(af[i]);
    }

    std::shared_ptr<distributed::MeshDevice> device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = device->mesh_command_queue();
    const uint32_t bytes = n * static_cast<uint32_t>(sizeof(uint16_t));
    distributed::DeviceLocalBufferConfig dram{.page_size = bytes, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig whole{.size = bytes};
    auto a_buf = distributed::MeshBuffer::create(whole, dram, device.get());
    auto c_buf = distributed::MeshBuffer::create(whole, dram, device.get());
    distributed::EnqueueWriteMeshBuffer(cq, a_buf, a_bf, false);

    Program program = CreateProgram();
    const CoreCoord core = {0, 0};
    KernelHandle mover = CreateKernel(
        program, "kernels/copy.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    const uint32_t a_addr = static_cast<uint32_t>(a_buf->address());
    const uint32_t c_addr = static_cast<uint32_t>(c_buf->address());
    SetRuntimeArgs(program, mover, core, {a_addr, c_addr, n});

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);

    // Read the copy back and convert it to float32 on the host.
    std::vector<uint16_t> res;
    distributed::EnqueueReadMeshBuffer(cq, res, c_buf, true);
    std::vector<float> c(n);
    for (uint32_t i = 0; i < n; ++i) {
        c[i] = from_bf16(res[i]);
    }
    device->close();

    uint64_t dims[1] = {n};
    if (lassi_io_write(argv[2], "c", LASSI_IO_F32, 1u, dims, c.data()) != 0) {
        return 1;
    }
    lassi_io_free(&a);
    return 0;
}
