// SYNTHETIC seeded candidate precision-float for adv-seeded-v0 prefix_sum (task P16.8; tests/fixtures/adversary/seeded/index.json): written by hand, not model output.
//
// Running sums: y[i] = x[0] + ... + x[i], accumulated in double, each stored as the f32 nearest to it.
// Usage: prefix_sum <x> <y>
//   x  the lassi_io input file: an f32 array named "x"
//   y  the lassi_io output file: an f32 array named "y" with x's shape (C order)
// Deterministic and single-threaded: no thread and no OpenMP construct.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "lassi_io.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: prefix_sum <x> <y>\n");
        return 2;
    }
    lassi_io_array x = {};
    if (lassi_io_read(argv[1], &x) != 0) {
        lassi_io_free(&x);
        return 1;
    }
    if (x.dtype != LASSI_IO_F32) {
        std::fprintf(stderr, "prefix_sum: x must be an f32 array\n");
        lassi_io_free(&x);
        return 1;
    }
    const float* in = static_cast<const float*>(x.data);
    std::vector<float> y(x.count);
    float total = 0.0f;
    for (uint64_t i = 0; i < x.count; ++i) {
        total += in[i];
        y[i] = total;
    }
    const int status = lassi_io_write(argv[2], "y", LASSI_IO_F32, x.rank, x.dims, y.data());
    lassi_io_free(&x);
    return status == 0 ? 0 : 1;
}
