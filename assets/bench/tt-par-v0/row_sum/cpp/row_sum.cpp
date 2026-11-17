// Written for LASSI-DF tt-par-v0 (task P17.68); a held-out item: the sum of each row of a bf16 matrix, on the host CPU.
//
// What it computes: s[i] = the sum over j of a[i, j], for a bf16 matrix a of rows x cols. Each bf16 element is
// widened to a double exactly and the row's elements are added in double in ascending j, then the sum is rounded
// once to f32 (to nearest). For the item's inputs every partial sum is exact in a double, so s is the exact row
// sum rounded to f32.
//
// Usage: row_sum <a> <s>
//   a  lassi_io file, bf16, rank 2, [rows, cols]: rows and cols positive multiples of 32, at most 65536 each
//   s  lassi_io file written here, f32, rank 1, [rows]
#include "lassi_io.h"

// Fixed-width integers and messages on stderr,
#include <cstdint>
#include <cstdio>
// then bit copies and the output array.
#include <cstring>
#include <vector>

namespace {

// The limits on a: each side a positive multiple of kSideStep, at most kLongestSide.
constexpr uint64_t kSideStep = 32;
constexpr uint64_t kLongestSide = 65536;

// The value of a bf16 bit pattern: the pattern is the upper half of an f32 of the same value.
double bf16_value(uint16_t bits) {
    const uint32_t f32_bits = static_cast<uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &f32_bits, sizeof value);
    return static_cast<double>(value);
}

// Read the lassi_io file at path into *a and check it against those limits; return true, or say why on
// stderr, release *a, and return false.
bool read_usable_a(const char* path, lassi_io_array* a) {
    if (lassi_io_read(path, a) != 0) {
        return false;
    }
    const char* why_not = nullptr;
    if (a->dtype != LASSI_IO_BF16) {
        why_not = "a must be bf16";
    } else if (a->rank != 2u) {
        why_not = "a must have rank 2";
    } else {
        for (uint32_t side = 0; side < 2u; ++side) {
            const uint64_t length = a->dims[side];
            if (length == 0u || length % kSideStep != 0u || length > kLongestSide) {
                why_not = "each side of a must be a positive multiple of 32, at most 65536";
            }
        }
    }
    if (why_not == nullptr) {
        return true;
    }
    std::fprintf(stderr, "row_sum: %s: %s\n", path, why_not);
    lassi_io_free(a);
    return false;
}

// Write into sums[i] the f32 sum of row i of the row-major rows x cols bf16 matrix held by a_bits.
void sum_each_row(const uint16_t* a_bits, float* sums, int64_t rows, int64_t cols) {
    for (int64_t i = 0; i < rows; ++i) {
        const uint16_t* row = a_bits + i * cols;
        double total = 0.0;
        for (int64_t j = 0; j < cols; ++j) {
            total += bf16_value(row[j]);
        }
        sums[i] = static_cast<float>(total);
    }
}

}  // namespace

// Read a, sum each of its rows, and write the sums as s.
int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: row_sum <a> <s>\n");
        return 2;
    }
    lassi_io_array a = {};
    if (!read_usable_a(argv[1], &a)) {
        return 1;
    }
    const uint64_t rows = a.dims[0];
    const uint64_t cols = a.dims[1];

    std::vector<float> s(rows);
    sum_each_row(static_cast<const uint16_t*>(a.data), s.data(), static_cast<int64_t>(rows),
                 static_cast<int64_t>(cols));

    const uint64_t s_dims[1] = {rows};
    const int status = lassi_io_write(argv[2], "s", LASSI_IO_F32, 1u, s_dims, s.data());
    lassi_io_free(&a);
    return status == 0 ? 0 : 1;
}
