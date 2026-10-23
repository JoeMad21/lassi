// SYNTHETIC faulty candidate large-zero for the adversary stage's remote tests (task P16.10; tests/core/
// test_adversary_stage_remote.py): written by hand from scale/cpp/scale.cpp, not model output, outside
// adv-seeded-v0 and every suite.
//
// The reference with one fault: y[k] = 2 * x[k] only where |x[k]| < 100, else 0. The held-out inputs lie in
// [0, 10), so it agrees with the reference there; the contract allows x up to 1000 in magnitude.
// Usage: scale <x> <y>
//   x  the lassi_io input file: a rank-1 f32 array named "x"
//   y  the lassi_io output file: an f32 array named "y" of the same shape
// Deterministic and single-threaded: no thread and no OpenMP construct.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "lassi_io.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: scale <x> <y>\n");
        return 2;
    }
    lassi_io_array x = {};
    if (lassi_io_read(argv[1], &x) != 0) {
        lassi_io_free(&x);
        return 1;
    }
    if (x.dtype != LASSI_IO_F32 || x.rank != 1 || x.count == 0) {
        std::fprintf(stderr, "scale: x must be a non-empty rank-1 f32 array\n");
        lassi_io_free(&x);
        return 1;
    }
    const float* in = static_cast<const float*>(x.data);
    std::vector<float> y(x.count);
    for (uint64_t k = 0; k < x.count; ++k) {
        y[k] = std::fabs(in[k]) < 100.0f ? 2.0f * in[k] : 0.0f;
    }
    const uint64_t dims[1] = {x.count};
    const int status = lassi_io_write(argv[2], "y", LASSI_IO_F32, 1, dims, y.data());
    lassi_io_free(&x);
    return status == 0 ? 0 : 1;
}
