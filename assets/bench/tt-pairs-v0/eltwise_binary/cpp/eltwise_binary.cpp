// Written for LASSI-DF tt-pairs-v0 (task P4.13); the CPU counterpart of the upstream eltwise_binary example from tt-metal 5280a9cf.
//
// Adds two bf16 arrays element by element on the host CPU: c = a + b.
// Usage: eltwise_binary <a> <b> <c>
//   a, b  lassi_io input files, each a bf16 array; both hold the same number of elements
//   c     the lassi_io output file: an f32 array named "c" with a's shape
// Each bf16 element is widened to f32 exactly, and each sum is computed and kept in f32.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "lassi_io.h"

namespace {

// The f32 value of a bf16 bit pattern: the pattern is the upper half of that f32.
float widen_bf16(uint16_t bits) {
    const uint32_t word = static_cast<uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &word, sizeof value);
    return value;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: eltwise_binary <a> <b> <c>\n");
        return 2;
    }
    lassi_io_array a = {};
    lassi_io_array b = {};
    if (lassi_io_read(argv[1], &a) != 0 || lassi_io_read(argv[2], &b) != 0) {
        lassi_io_free(&a);
        lassi_io_free(&b);
        return 1;
    }
    if (a.dtype != LASSI_IO_BF16 || b.dtype != LASSI_IO_BF16 || a.count != b.count) {
        std::fprintf(stderr, "eltwise_binary: a and b must be bf16 arrays with one element count\n");
        lassi_io_free(&a);
        lassi_io_free(&b);
        return 1;
    }

    const uint16_t* a_bits = static_cast<const uint16_t*>(a.data);
    const uint16_t* b_bits = static_cast<const uint16_t*>(b.data);
    std::vector<float> c(a.count);
    for (uint64_t i = 0; i < a.count; ++i) {
        c[i] = widen_bf16(a_bits[i]) + widen_bf16(b_bits[i]);
    }

    const int status = lassi_io_write(argv[3], "c", LASSI_IO_F32, a.rank, a.dims, c.data());
    lassi_io_free(&a);
    lassi_io_free(&b);
    return status == 0 ? 0 : 1;
}
