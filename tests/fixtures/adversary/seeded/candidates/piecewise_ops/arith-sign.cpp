// SYNTHETIC seeded candidate arith-sign for adv-seeded-v0 piecewise_ops (task P16.8; tests/fixtures/adversary/seeded/index.json): written by hand, not model output.
//
// A piecewise map, element by element: y = 0 - 3x for x < 0 and y = 0.5x otherwise, all in f32.
// Usage: piecewise_ops <x> <y>
//   x  the lassi_io input file: an f32 array named "x"
//   y  the lassi_io output file: an f32 array named "y" with x's shape
// Deterministic and single-threaded: no thread and no OpenMP construct.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "lassi_io.h"

namespace {

float map(float x) {
    if (x < 0.0f) {
        return 0.0f + 3.0f * x;
    }
    return 0.5f * x;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: piecewise_ops <x> <y>\n");
        return 2;
    }
    lassi_io_array x = {};
    if (lassi_io_read(argv[1], &x) != 0) {
        lassi_io_free(&x);
        return 1;
    }
    if (x.dtype != LASSI_IO_F32) {
        std::fprintf(stderr, "piecewise_ops: x must be an f32 array\n");
        lassi_io_free(&x);
        return 1;
    }
    const float* in = static_cast<const float*>(x.data);
    std::vector<float> y(x.count);
    for (uint64_t i = 0; i < x.count; ++i) {
        y[i] = map(in[i]);
    }
    const int status = lassi_io_write(argv[2], "y", LASSI_IO_F32, x.rank, x.dims, y.data());
    lassi_io_free(&x);
    return status == 0 ? 0 : 1;
}
