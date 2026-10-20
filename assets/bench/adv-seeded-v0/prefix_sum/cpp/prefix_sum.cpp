// Written for LASSI adv-seeded-v0 (task P16.8): a synthetic reference program, not upstream code.
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
    double total = 0.0;
    for (uint64_t i = 0; i < x.count; ++i) {
        total += static_cast<double>(in[i]);
        y[i] = static_cast<float>(total);
    }
    const int status = lassi_io_write(argv[2], "y", LASSI_IO_F32, x.rank, x.dims, y.data());
    lassi_io_free(&x);
    return status == 0 ? 0 : 1;
}
