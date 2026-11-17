// Written for LASSI-DF tt-par-v0 (task P17.68); a held-out item: C[i, j] = A[i, j] * s[j] for a bf16 matrix A and row vector s, computed on the host.
//
// Each column j of a row-major bf16 matrix A (M rows, N columns) is scaled by s[j]. Each bf16 value is widened to
// f32 and the product formed in f32, where it is exact (two 8-bit significands give at most 16 significant bits); the
// product is then rounded to bf16, to nearest with ties to even.
//
// Usage: col_scale <a> <s> <c>
//   a  lassi_io file, bf16, rank 2, [M, N]
//   s  lassi_io file, bf16, rank 1, [N]
//   c  lassi_io file written here, bf16, rank 2, [M, N], row-major
// M and N must be positive multiples of 32, at most 65536.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "lassi_io.h"

namespace {

// What every dim must be a multiple of, and the largest dim accepted.
constexpr uint64_t kDimStep = 32;
constexpr uint64_t kMaxDim = 65536;

// The f32 whose upper half is the bf16 pattern `pattern`, which is the same value: bf16 is f32 cut to 16 bits.
float f32_of_bf16(uint16_t pattern) {
    uint32_t word = pattern;
    word <<= 16;
    float same_value;
    std::memcpy(&same_value, &word, sizeof same_value);
    return same_value;
}

// The bf16 pattern nearest to x, a tie going to the pattern whose last bit is 0; a NaN keeps its sign and stays NaN.
uint16_t bf16_nearest_even(float x) {
    uint32_t word;
    std::memcpy(&word, &x, sizeof word);
    const bool not_a_number = (word & 0x7fffffffu) > 0x7f800000u;
    if (not_a_number) {
        return static_cast<uint16_t>(0x7fc0u | ((word >> 16) & 0x8000u));
    }
    const uint32_t kept_lsb = (word >> 16) & 1u;
    word += 0x7fffu + kept_lsb;
    return static_cast<uint16_t>(word >> 16);
}

// True when dim is a positive multiple of kDimStep, at most kMaxDim.
bool usable_dim(uint64_t dim) {
    return dim != 0u && dim % kDimStep == 0u && dim <= kMaxDim;
}

// Load the lassi_io file at path into *x and keep it only when it is bf16 of rank `rank` (2 for A, 1 for s) with
// every dim usable: return 0 then, and otherwise 1, after a message (lassi_io_read prints its own) and with *x
// released.
int load_bf16(const char* path, uint32_t rank, lassi_io_array* x) {
    const int unreadable = lassi_io_read(path, x);
    if (unreadable != 0) {
        return unreadable;
    }
    bool fits = x->dtype == LASSI_IO_BF16 && x->rank == rank;
    for (uint32_t d = 0; fits && d < rank; ++d) {
        fits = usable_dim(x->dims[d]);
    }
    if (fits) {
        return 0;
    }
    std::fprintf(stderr, "col_scale: %s: not a rank-%u bf16 array whose dims are multiples of 32 from 32 to 65536\n",
                 path, static_cast<unsigned>(rank));
    lassi_io_free(x);
    return 1;
}

// Write into c (rows x cols, row-major bf16) each element of a (rows x cols) times the s value of its column, the
// product formed exactly in f32 and rounded once to bf16. s is widened once, then c is filled row by row.
void scale_columns(const uint16_t* a, const uint16_t* s, uint16_t* c, int64_t rows, int64_t cols) {
    std::vector<float> s_wide(static_cast<size_t>(cols));
    for (int64_t j = 0; j < cols; ++j) {
        s_wide[static_cast<size_t>(j)] = f32_of_bf16(s[j]);
    }
    const float* column_scale = s_wide.data();
    for (int64_t i = 0; i < rows; ++i) {
        const uint16_t* a_row = a + i * cols;
        uint16_t* c_row = c + i * cols;
        for (int64_t j = 0; j < cols; ++j) {
            c_row[j] = bf16_nearest_even(f32_of_bf16(a_row[j]) * column_scale[j]);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: col_scale <a> <s> <c>\n");
        return 2;
    }
    lassi_io_array a = {};
    lassi_io_array s = {};
    if (load_bf16(argv[1], 2u, &a) != 0 || load_bf16(argv[2], 1u, &s) != 0) {
        lassi_io_free(&a);
        return 1;
    }
    int status = 1;
    if (s.dims[0] != a.dims[1]) {
        std::fprintf(stderr, "col_scale: a has %llu columns but s has %llu values\n",
                     static_cast<unsigned long long>(a.dims[1]), static_cast<unsigned long long>(s.dims[0]));
    } else {
        const int64_t rows = static_cast<int64_t>(a.dims[0]);
        const int64_t cols = static_cast<int64_t>(a.dims[1]);
        std::vector<uint16_t> c(static_cast<size_t>(rows * cols));
        scale_columns(static_cast<const uint16_t*>(a.data), static_cast<const uint16_t*>(s.data), c.data(), rows,
                      cols);
        const uint64_t c_dims[2] = {a.dims[0], a.dims[1]};
        status = lassi_io_write(argv[3], "c", LASSI_IO_BF16, 2u, c_dims, c.data());
    }
    lassi_io_free(&a);
    lassi_io_free(&s);
    return status;
}
