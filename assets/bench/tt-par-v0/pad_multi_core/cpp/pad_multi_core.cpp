// Written for LASSI-DF tt-par-v0 (task P17.67); the CPU counterpart of the upstream pad_multi_core example from tt-metal 5280a9cf.
//
// Right padding: every row of a row-major bf16 matrix is copied and then filled out to 256 columns with the value
// 2.0 (bf16 bit pattern 0x4000). Values move as bit patterns and none is converted.
//
// Usage: pad_multi_core <src> <dst>
//   src  lassi_io file, bf16, rank 2, [rows, cols]: rows from 1 to 65536, cols even, from 2 to 256
//   dst  lassi_io file written here, bf16, rank 2, [rows, 256], row-major: the first cols values of each row are
//        src's row, the rest are 2.0
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "lassi_io.h"

namespace {

// The width every row is padded to, and the pad value 2.0 as a bf16 bit pattern.
constexpr uint64_t kPaddedCols = 256;
constexpr uint16_t kPadBits = 0x4000;
// The most rows accepted.
constexpr uint64_t kMaxRows = 65536;

// Read the lassi_io file at path into *x and check that it holds a bf16 matrix of 1 to kMaxRows rows and an even
// number of columns from 2 to kPaddedCols; return 0, or print why and return 1 with *x released.
int read_source(const char* path, lassi_io_array* x) {
    if (lassi_io_read(path, x) != 0) {
        return 1;
    }
    const char* problem = nullptr;
    if (x->dtype != LASSI_IO_BF16) {
        problem = "the array must be bf16";
    } else if (x->rank != 2u) {
        problem = "the array must have rank 2";
    } else if (x->dims[0] == 0u || x->dims[0] > kMaxRows) {
        problem = "the row count must be from 1 to 65536";
    } else if (x->dims[1] == 0u || x->dims[1] % 2u != 0u || x->dims[1] > kPaddedCols) {
        problem = "the column count must be even, from 2 to 256";
    }
    if (problem != nullptr) {
        std::fprintf(stderr, "pad_multi_core: %s: %s\n", path, problem);
        lassi_io_free(x);
        return 1;
    }
    return 0;
}

// Write the rows x kPaddedCols padding of the row-major rows x cols matrix src into dst: each row's cols values
// copied as they are, then the pad value up to kPaddedCols.
void pad_rows(const uint16_t* src, uint16_t* dst, int64_t rows, int64_t cols) {
    const int64_t width = static_cast<int64_t>(kPaddedCols);
    for (int64_t r = 0; r < rows; ++r) {
        const uint16_t* in = src + r * cols;
        uint16_t* out = dst + r * width;
        std::memcpy(out, in, static_cast<size_t>(cols) * sizeof(uint16_t));
        std::fill(out + cols, out + width, kPadBits);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: pad_multi_core <src> <dst>\n");
        return 2;
    }
    lassi_io_array src = {};
    if (read_source(argv[1], &src) != 0) {
        return 1;
    }
    const uint64_t rows = src.dims[0];
    const uint64_t cols = src.dims[1];

    std::vector<uint16_t> dst(rows * kPaddedCols);
    pad_rows(static_cast<const uint16_t*>(src.data), dst.data(), static_cast<int64_t>(rows),
             static_cast<int64_t>(cols));

    const uint64_t dims[2] = {rows, kPaddedCols};
    const int status = lassi_io_write(argv[2], "dst", LASSI_IO_BF16, 2u, dims, dst.data());
    lassi_io_free(&src);
    return status;
}
