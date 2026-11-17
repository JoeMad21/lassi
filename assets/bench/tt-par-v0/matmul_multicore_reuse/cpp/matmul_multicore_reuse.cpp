// Written for LASSI-DF tt-par-v0 (task P17.67); the CPU counterpart of the upstream matmul_multicore_reuse example from tt-metal 5280a9cf.
//
// Matrix multiply, c = a x b. Each output element is the sum over k of a[i][k] * b[k][j], the bf16 inputs widened to
// float32 and the products added in float32 in ascending k, and the sum rounded once to bf16 (to nearest, ties to
// even).
//
// Usage: matmul_multicore_reuse <a> <b> <c>
//   a  lassi_io file, bf16, rank 2, [M, K]
//   b  lassi_io file, bf16, rank 2, [K, N]
//   c  lassi_io file written here, bf16, rank 2, [M, N], row-major
// M and N must be positive multiples of 32 and K a positive multiple of 64, each at most 65536.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "lassi_io.h"

namespace {

// What every dim must be a multiple of, what K must also be a multiple of, and the largest dim.
constexpr uint64_t kDimStep = 32;
constexpr uint64_t kInnerStep = 64;
constexpr uint64_t kMaxDim = 65536;

// A bf16 value is the upper 16 bits of the float32 of the same value.
float widen(uint16_t half) {
    const uint32_t bits = static_cast<uint32_t>(half) << 16;
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof value);
    return value;
}

// Round a float32 to bf16, to nearest with ties to even; a NaN stays a NaN.
uint16_t narrow(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof bits);
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        return static_cast<uint16_t>((bits >> 16) | 0x0040u);
    }
    const uint32_t bias = 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits + bias) >> 16);
}

// Read the lassi_io file at path into *x and check that it holds a bf16 matrix whose dims are positive multiples
// of 32, at most kMaxDim; return 0, or print why and return 1 with *x released.
int read_matrix(const char* path, lassi_io_array* x) {
    if (lassi_io_read(path, x) != 0) {
        return 1;
    }
    const char* problem = nullptr;
    if (x->dtype != LASSI_IO_BF16) {
        problem = "the array must be bf16";
    } else if (x->rank != 2u) {
        problem = "the array must have rank 2";
    } else {
        for (uint32_t i = 0; i < 2u; ++i) {
            if (x->dims[i] == 0u || x->dims[i] % kDimStep != 0u || x->dims[i] > kMaxDim) {
                problem = "each dim must be a positive multiple of 32, at most 65536";
            }
        }
    }
    if (problem != nullptr) {
        std::fprintf(stderr, "matmul_multicore_reuse: %s: %s\n", path, problem);
        lassi_io_free(x);
        return 1;
    }
    return 0;
}

// Write the m x n product of the row-major bf16 matrices a (m x k) and b (k x n) into c, row-major bf16. One row of
// float32 sums is built in ascending k for each row of c in turn.
void multiply(const uint16_t* a, const uint16_t* b, uint16_t* c, int64_t m, int64_t k, int64_t n) {
    std::vector<float> sums(static_cast<size_t>(n));
    for (int64_t i = 0; i < m; ++i) {
        std::fill(sums.begin(), sums.end(), 0.0f);
        for (int64_t p = 0; p < k; ++p) {
            const float left = widen(a[i * k + p]);
            const uint16_t* row = b + p * n;
            for (int64_t j = 0; j < n; ++j) {
                sums[static_cast<size_t>(j)] += left * widen(row[j]);
            }
        }
        for (int64_t j = 0; j < n; ++j) {
            c[i * n + j] = narrow(sums[static_cast<size_t>(j)]);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: matmul_multicore_reuse <a> <b> <c>\n");
        return 2;
    }
    lassi_io_array a = {};
    lassi_io_array b = {};
    if (read_matrix(argv[1], &a) != 0) {
        return 1;
    }
    if (read_matrix(argv[2], &b) != 0) {
        lassi_io_free(&a);
        return 1;
    }
    int status = 1;
    if (a.dims[1] != b.dims[0]) {
        std::fprintf(stderr, "matmul_multicore_reuse: a has %llu columns but b has %llu rows\n",
                     static_cast<unsigned long long>(a.dims[1]), static_cast<unsigned long long>(b.dims[0]));
    } else if (a.dims[1] % kInnerStep != 0u) {
        std::fprintf(stderr, "matmul_multicore_reuse: K (%llu) must be a multiple of 64\n",
                     static_cast<unsigned long long>(a.dims[1]));
    } else {
        const int64_t m = static_cast<int64_t>(a.dims[0]);
        const int64_t k = static_cast<int64_t>(a.dims[1]);
        const int64_t n = static_cast<int64_t>(b.dims[1]);
        std::vector<uint16_t> c(static_cast<size_t>(m * n));
        multiply(static_cast<const uint16_t*>(a.data), static_cast<const uint16_t*>(b.data), c.data(), m, k, n);
        const uint64_t dims[2] = {a.dims[0], b.dims[1]};
        status = lassi_io_write(argv[3], "c", LASSI_IO_BF16, 2u, dims, c.data());
    }
    lassi_io_free(&a);
    lassi_io_free(&b);
    return status;
}
