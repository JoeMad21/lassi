// Written for LASSI-DF tt-par-v0 (task P17.68); a held-out item: the maximum of each column of a bf16 matrix, on the host CPU.
//
// What it computes: m[j] = the maximum over i of a[i, j], for a bf16 matrix a of rows x cols. Elements are compared
// by value (each bf16 widened to f32 exactly) and the largest one's bit pattern is written, so every m[j] is one of
// a's own values and no rounding happens anywhere.
//
// Usage: col_max <a> <m>
//   a  lassi_io file, bf16, rank 2, [rows, cols]: rows and cols positive multiples of 32, at most 65536 each
//   m  lassi_io file written here, bf16, rank 1, [cols]
#include "lassi_io.h"

// Fixed-width integers and messages on stderr,
#include <cstdint>
#include <cstdio>
// then bit copies and the output array.
#include <cstring>
#include <vector>

namespace {

// The limits on a: each side a positive multiple of kSideStep, at most kSideLimit.
constexpr uint64_t kSideStep = 32;
constexpr uint64_t kSideLimit = 65536;

// The value of a bf16 bit pattern: the pattern is the upper half of an f32 of the same value.
float bf16_as_f32(uint16_t bits) {
    const uint32_t f32_bits = static_cast<uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &f32_bits, sizeof value);
    return value;
}

// Read the lassi_io file at path into *a and check it against those limits; return true, or say why on
// stderr, release *a, and return false.
bool load_usable_a(const char* path, lassi_io_array* a) {
    if (lassi_io_read(path, a) != 0) {
        return false;
    }
    const char* refusal = nullptr;
    if (a->dtype != LASSI_IO_BF16) {
        refusal = "a must be bf16";
    } else if (a->rank != 2u) {
        refusal = "a must have rank 2";
    } else {
        for (uint32_t side = 0; side < 2u; ++side) {
            const uint64_t extent = a->dims[side];
            if (extent == 0u || extent % kSideStep != 0u || extent > kSideLimit) {
                refusal = "each side of a must be a positive multiple of 32, at most 65536";
            }
        }
    }
    if (refusal == nullptr) {
        return true;
    }
    std::fprintf(stderr, "col_max: %s: %s\n", path, refusal);
    lassi_io_free(a);
    return false;
}

// Write into maxima[j] the bit pattern of the largest element of column j of the row-major rows x cols bf16
// matrix held by a_bits; of equal values the topmost is kept.
void max_each_column(const uint16_t* a_bits, uint16_t* maxima, int64_t rows, int64_t cols) {
    for (int64_t j = 0; j < cols; ++j) {
        uint16_t best_bits = a_bits[j];
        float best = bf16_as_f32(best_bits);
        for (int64_t i = 1; i < rows; ++i) {
            const uint16_t bits = a_bits[i * cols + j];
            const float value = bf16_as_f32(bits);
            if (value > best) {
                best = value;
                best_bits = bits;
            }
        }
        maxima[j] = best_bits;
    }
}

}  // namespace

// Read a, find the largest element of each of its columns, and write those elements as m.
int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: col_max <a> <m>\n");
        return 2;
    }
    lassi_io_array a = {};
    if (!load_usable_a(argv[1], &a)) {
        return 1;
    }
    const uint64_t rows = a.dims[0];
    const uint64_t cols = a.dims[1];

    std::vector<uint16_t> m(cols);
    max_each_column(static_cast<const uint16_t*>(a.data), m.data(), static_cast<int64_t>(rows),
                    static_cast<int64_t>(cols));

    const uint64_t m_dims[1] = {cols};
    const int status = lassi_io_write(argv[2], "m", LASSI_IO_BF16, 1u, m_dims, m.data());
    lassi_io_free(&a);
    return status == 0 ? 0 : 1;
}
