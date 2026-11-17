// Written for LASSI-DF tt-par-v0 (task P17.68); a held-out item: row softmax of a bf16 matrix, on the host.
//
// Row softmax: y[i, j] = exp(x[i, j] - m_i) / s_i, where m_i is the largest value of row i and s_i is the sum over k
// of exp(x[i, k] - m_i). Each bf16 input is widened to float32; the maximum, the shifted exponentials (std::exp on
// float), their sum (in ascending column order), and the quotient are all float32, and each quotient is rounded once
// to bf16 (to nearest, ties to even).
//
// Usage: row_softmax <x> <y>
//   x  lassi_io file, bf16, rank 2, [rows, cols]
//   y  lassi_io file written here, bf16, rank 2, [rows, cols], row-major
// rows must be a positive multiple of 32, at most 65536, and cols a positive multiple of 32, at most 2048.
#include <cmath>    // std::exp on float
#include <cstdint>  // the fixed-width integer types
#include <cstdio>   // std::fprintf for refusals
#include <cstring>  // std::memcpy between float and its bit pattern
#include <vector>   // the output and the scratch row

#include "lassi_io.h"

namespace {

// What both dims must be a multiple of, and the largest row and column counts accepted.
constexpr uint64_t kEdge = 32;
constexpr uint64_t kRowLimit = 65536;
constexpr uint64_t kColLimit = 2048;

// A bf16 pattern is the top half of the float32 of the same value, so shifting it up 16 bits gives that float32.
float bf16_to_float(uint16_t bits) {
    uint32_t widened = bits;
    widened <<= 16;
    float out;
    std::memcpy(&out, &widened, sizeof(out));
    return out;
}

// The bf16 pattern nearest to `value`, ties to even. A NaN becomes the quiet NaN of the same sign.
uint16_t float_to_bf16(float value) {
    uint32_t pattern;
    std::memcpy(&pattern, &value, sizeof(pattern));
    const bool not_a_number = (pattern & 0x7f800000u) == 0x7f800000u && (pattern & 0x007fffffu) != 0u;
    if (not_a_number) {
        return static_cast<uint16_t>(0x7fc0u | ((pattern >> 16) & 0x8000u));
    }
    // Add just under half of the 16 dropped bits' range, plus the lowest kept bit, then drop them: ties go to even.
    pattern += 0x7fffu + ((pattern >> 16) & 1u);
    return static_cast<uint16_t>(pattern >> 16);
}

// Read the lassi_io file at `path` into *x and check it is a bf16 matrix within the limits above. Return 0, or
// print the reason and return 1 with *x released.
int load_matrix(const char* path, lassi_io_array* x) {
    if (lassi_io_read(path, x) != 0) {
        return 1;
    }
    const char* reason = nullptr;
    if (x->dtype != LASSI_IO_BF16) {
        reason = "x must be bf16";
    } else if (x->rank != 2u) {
        reason = "x must be a matrix (rank 2)";
    } else if (x->dims[0] == 0u || x->dims[0] % kEdge != 0u || x->dims[0] > kRowLimit) {
        reason = "the row count must be a positive multiple of 32, at most 65536";
    } else if (x->dims[1] == 0u || x->dims[1] % kEdge != 0u || x->dims[1] > kColLimit) {
        reason = "the column count must be a positive multiple of 32, at most 2048";
    }
    if (reason != nullptr) {
        std::fprintf(stderr, "row_softmax: %s: %s\n", path, reason);
        lassi_io_free(x);
        return 1;
    }
    return 0;
}

// One row: `in` holds `cols` bf16 patterns, `out` receives their softmax as bf16 patterns, and `scratch` (cols
// floats) holds the shifted exponentials between the two passes.
void softmax_one_row(const uint16_t* in, uint16_t* out, float* scratch, int64_t cols) {
    float peak = bf16_to_float(in[0]);
    for (int64_t j = 1; j < cols; ++j) {
        const float v = bf16_to_float(in[j]);
        if (v > peak) {
            peak = v;
        }
    }
    float total = 0.0f;
    for (int64_t j = 0; j < cols; ++j) {
        scratch[j] = std::exp(bf16_to_float(in[j]) - peak);
        total += scratch[j];
    }
    for (int64_t j = 0; j < cols; ++j) {
        out[j] = float_to_bf16(scratch[j] / total);
    }
}

// All rows of the row-major rows x cols matrix `x` into `y`, one after another, reusing one scratch row.
void softmax_all_rows(const uint16_t* x, uint16_t* y, int64_t rows, int64_t cols) {
    std::vector<float> scratch(static_cast<size_t>(cols));
    for (int64_t i = 0; i < rows; ++i) {
        softmax_one_row(x + i * cols, y + i * cols, scratch.data(), cols);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: row_softmax <x> <y>\n");
        return 2;
    }
    lassi_io_array x = {};
    if (load_matrix(argv[1], &x) != 0) {
        return 1;
    }
    const int64_t rows = static_cast<int64_t>(x.dims[0]);
    const int64_t cols = static_cast<int64_t>(x.dims[1]);
    std::vector<uint16_t> y(static_cast<size_t>(rows * cols));
    softmax_all_rows(static_cast<const uint16_t*>(x.data), y.data(), rows, cols);
    const uint64_t shape[2] = {x.dims[0], x.dims[1]};
    const int status = lassi_io_write(argv[2], "y", LASSI_IO_BF16, 2u, shape, y.data());
    lassi_io_free(&x);
    return status == 0 ? 0 : 1;
}
