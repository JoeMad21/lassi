// Written for LASSI adv-seeded-v0 (task P16.8): a synthetic reference program, not upstream code.
//
// An affine scale widened to 64 bits, element by element: y = 3000x + 7, computed in int64.
// Usage: scale_i64 <x> <y>
//   x  the lassi_io input file: an i32 array named "x"
//   y  the lassi_io output file: an i64 array named "y" with x's shape
// Deterministic and single-threaded: no thread and no OpenMP construct.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "lassi_io.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: scale_i64 <x> <y>\n");
        return 2;
    }
    lassi_io_array x = {};
    if (lassi_io_read(argv[1], &x) != 0) {
        lassi_io_free(&x);
        return 1;
    }
    if (x.dtype != LASSI_IO_I32) {
        std::fprintf(stderr, "scale_i64: x must be an i32 array\n");
        lassi_io_free(&x);
        return 1;
    }
    const int32_t* in = static_cast<const int32_t*>(x.data);
    std::vector<int64_t> y(x.count);
    for (uint64_t i = 0; i < x.count; ++i) {
        y[i] = static_cast<int64_t>(in[i]) * 3000 + 7;
    }
    const int status = lassi_io_write(argv[2], "y", LASSI_IO_I64, x.rank, x.dims, y.data());
    lassi_io_free(&x);
    return status == 0 ? 0 : 1;
}
