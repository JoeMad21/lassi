// Written for LASSI-DF tt-pairs-v0 (task P4.13); the CPU counterpart of the upstream matmul_single_core example from tt-metal 5280a9cf.
//
// Matrix multiply on the host: C = A x B. Each element of C is a float32 sum over k = 0, 1, ..., K - 1 of
// A[i][k] * B[k][j], each bf16 operand widened to float32 exactly, and the sum is rounded to the nearest bf16,
// ties to even.
//
// Usage: matmul_single_core <a> <b> <c>
//   a  lassi_io file, bf16, shape [M, K]
//   b  lassi_io file, bf16, shape [K, N]
//   c  lassi_io file written here, bf16, shape [M, N]
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "lassi_io.h"

namespace {

// Return the float32 of a bf16 bit pattern: the pattern is the float32's upper 16 bits.
float widen_bf16(uint16_t bits) {
    const uint32_t wide = static_cast<uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &wide, sizeof value);
    return value;
}

// Return the bf16 nearest to a float32, ties to even; a NaN stays a NaN (quiet, with its sign).
uint16_t round_to_bf16(float value) {
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof bits);
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        return static_cast<uint16_t>((bits >> 16) | 0x0040u);
    }
    const uint32_t lsb = (bits >> 16) & 1u;
    return static_cast<uint16_t>((bits + 0x7fffu + lsb) >> 16);
}

// Return why the inputs cannot be multiplied, or nullptr when they can; only the arrays' headers are read.
const char* shape_problem(const lassi_io_array& a, const lassi_io_array& b) {
    if (a.dtype != LASSI_IO_BF16 || b.dtype != LASSI_IO_BF16 || a.rank != 2u || b.rank != 2u) {
        return "a and b must be rank-2 bf16 arrays";
    }
    if (a.dims[1] != b.dims[0]) {
        return "the columns of a must equal the rows of b";
    }
    return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: matmul_single_core <a> <b> <c>\n");
        return 2;
    }
    lassi_io_array a = {};
    lassi_io_array b = {};
    if (lassi_io_read(argv[1], &a) != 0 || lassi_io_read(argv[2], &b) != 0) {
        lassi_io_free(&a);
        return 1;
    }
    const char* problem = shape_problem(a, b);
    if (problem != nullptr) {
        std::fprintf(stderr, "matmul_single_core: %s\n", problem);
        lassi_io_free(&a);
        lassi_io_free(&b);
        return 1;
    }
    const std::size_t m = static_cast<std::size_t>(a.dims[0]);
    const std::size_t k = static_cast<std::size_t>(a.dims[1]);
    const std::size_t n = static_cast<std::size_t>(b.dims[1]);

    // Widen A as it is, and B transposed, so that each element of C reads both operands contiguously.
    const uint16_t* a_bits = static_cast<const uint16_t*>(a.data);
    const uint16_t* b_bits = static_cast<const uint16_t*>(b.data);
    std::vector<float> a_rows(m * k);
    std::vector<float> b_cols(k * n);
    for (std::size_t i = 0; i < m * k; ++i) {
        a_rows[i] = widen_bf16(a_bits[i]);
    }
    for (std::size_t r = 0; r < k; ++r) {
        for (std::size_t j = 0; j < n; ++j) {
            b_cols[j * k + r] = widen_bf16(b_bits[r * n + j]);
        }
    }

    std::vector<uint16_t> c(m * n);
    for (std::size_t i = 0; i < m; ++i) {
        const float* row = a_rows.data() + i * k;
        for (std::size_t j = 0; j < n; ++j) {
            const float* col = b_cols.data() + j * k;
            float sum = 0.0f;
            for (std::size_t r = 0; r < k; ++r) {
                sum += row[r] * col[r];
            }
            c[i * n + j] = round_to_bf16(sum);
        }
    }

    const uint64_t c_dims[2] = {a.dims[0], b.dims[1]};
    const int status = lassi_io_write(argv[3], "c", LASSI_IO_BF16, 2u, c_dims, c.data());
    lassi_io_free(&a);
    lassi_io_free(&b);
    return status;
}
