// Written for LASSI-DF tt-pairs-v0 (task P4.13); the CPU counterpart of the upstream eltwise_sfpu example from tt-metal 5280a9cf.
//
// It runs no kernel: it computes on the host CPU what the TT version's kernels compute.
//
// Usage: eltwise_sfpu <src0 file> <result file>
//
// For each element x of the bf16 array src0 it writes bf16(exp(x)): x widened to float, std::exp taken in
// float, and the value rounded to bf16 to nearest, ties to even. The output is the bf16 array "result",
// with src0's shape.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "lassi_io.h"

namespace {

// A bf16 value is the upper 16 bits of the float of the same value.
float bf16_to_float(uint16_t half) {
    const uint32_t bits = static_cast<uint32_t>(half) << 16;
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof value);
    return value;
}

// Round a float to bf16 to nearest, ties to even; a NaN stays a quiet NaN.
uint16_t float_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof bits);
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        return static_cast<uint16_t>((bits >> 16) | 0x0040u);
    }
    const uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits + rounding) >> 16);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: eltwise_sfpu <src0 file> <result file>\n");
        return 2;
    }
    lassi_io_array src0 = {};
    if (lassi_io_read(argv[1], &src0) != 0) {
        return 1;
    }
    if (src0.dtype != LASSI_IO_BF16) {
        std::fprintf(stderr, "eltwise_sfpu: src0 is not a bf16 array\n");
        lassi_io_free(&src0);
        return 1;
    }
    const uint16_t* x = static_cast<const uint16_t*>(src0.data);
    std::vector<uint16_t> result(src0.count);
    for (uint64_t i = 0; i < src0.count; ++i) {
        result[i] = float_to_bf16(std::exp(bf16_to_float(x[i])));
    }
    const int status = lassi_io_write(argv[2], "result", LASSI_IO_BF16, src0.rank, src0.dims, result.data());
    lassi_io_free(&src0);
    return status == 0 ? 0 : 1;
}
