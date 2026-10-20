// SYNTHETIC seeded candidate special-fmax for adv-seeded-v0 pair_max (task P16.8; tests/fixtures/adversary/seeded/index.json): written by hand, not model output.
//
// The larger of each adjacent pair, with NaN propagated: y[k] is NaN when x[2k] or x[2k+1] is NaN, else the
// larger of the two (x[2k + 1] when they are equal).
// Usage: pair_max <x> <y>
//   x  the lassi_io input file: a rank-1 f32 array named "x" with an even element count
//   y  the lassi_io output file: an f32 array named "y" with half as many elements
// Deterministic and single-threaded: no thread and no OpenMP construct.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "lassi_io.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: pair_max <x> <y>\n");
        return 2;
    }
    lassi_io_array x = {};
    if (lassi_io_read(argv[1], &x) != 0) {
        lassi_io_free(&x);
        return 1;
    }
    if (x.dtype != LASSI_IO_F32 || x.rank != 1 || x.count % 2 != 0) {
        std::fprintf(stderr, "pair_max: x must be a rank-1 f32 array with an even element count\n");
        lassi_io_free(&x);
        return 1;
    }
    const float* in = static_cast<const float*>(x.data);
    const uint64_t pairs = x.count / 2;
    std::vector<float> y(pairs);
    for (uint64_t k = 0; k < pairs; ++k) {
        const float a = in[2 * k];
        const float b = in[2 * k + 1];
        y[k] = std::fmax(a, b);
    }
    const uint64_t dims[1] = {pairs};
    const int status = lassi_io_write(argv[2], "y", LASSI_IO_F32, 1, dims, y.data());
    lassi_io_free(&x);
    return status == 0 ? 0 : 1;
}
