// Written for LASSI-DF tt-par-v0 (task P17.68); a held-out item: B = A^T for a bf16 matrix, computed on the host.
//
// B[j][i] = A[i][j] for a row-major bf16 matrix A of M rows and N columns, B having N rows and M columns. Values move
// as bit patterns and none is converted. The copy goes one 32 x 32 block at a time, so the rows a block reads and the
// rows it writes stay in cache.
//
// Usage: transpose_2d <a> <b>
//   a  lassi_io file, bf16, rank 2, [M, N]
//   b  lassi_io file written here, bf16, rank 2, [N, M], row-major
// M and N must be positive multiples of 32, at most 65536.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "lassi_io.h"

namespace {

// The block edge: every dim must be a multiple of it.
constexpr int64_t kBlock = 32;
// The largest dim accepted.
constexpr uint64_t kMaxDim = 65536;

// True when dim is a whole number of blocks, from one block to kMaxDim.
bool whole_blocks(uint64_t dim) {
    return dim != 0u && dim % static_cast<uint64_t>(kBlock) == 0u && dim <= kMaxDim;
}

// Load the lassi_io file at path into *x and keep it only when it is a bf16 matrix whose two dims are whole blocks:
// return 0 then, and otherwise 1, after a message (lassi_io_read prints its own) and with *x released.
int load_block_matrix(const char* path, lassi_io_array* x) {
    const int unreadable = lassi_io_read(path, x);
    if (unreadable != 0) {
        return unreadable;
    }
    const bool bf16_matrix = x->dtype == LASSI_IO_BF16 && x->rank == 2u;
    if (bf16_matrix && whole_blocks(x->dims[0]) && whole_blocks(x->dims[1])) {
        return 0;
    }
    std::fprintf(stderr, "transpose_2d: %s: not a bf16 matrix whose two dims are multiples of 32 from 32 to 65536\n",
                 path);
    lassi_io_free(x);
    return 1;
}

// Write into b (cols x rows, row-major) the transpose of a (rows x cols, row-major), one band of kBlock rows of b
// after another; a band is filled block by block, each block from the matching block of a.
void transpose(const uint16_t* a, uint16_t* b, int64_t rows, int64_t cols) {
    const int64_t bands = cols / kBlock;
    for (int64_t band = 0; band < bands; ++band) {
        const int64_t b_row0 = band * kBlock;
        for (int64_t b_col0 = 0; b_col0 < rows; b_col0 += kBlock) {
            for (int64_t r = b_row0; r < b_row0 + kBlock; ++r) {
                uint16_t* out = b + r * rows;
                for (int64_t c = b_col0; c < b_col0 + kBlock; ++c) {
                    out[c] = a[c * cols + r];
                }
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: transpose_2d <a> <b>\n");
        return 2;
    }
    lassi_io_array a = {};
    if (load_block_matrix(argv[1], &a) != 0) {
        return 1;
    }
    const int64_t rows = static_cast<int64_t>(a.dims[0]);
    const int64_t cols = static_cast<int64_t>(a.dims[1]);

    std::vector<uint16_t> b(static_cast<size_t>(rows * cols));
    transpose(static_cast<const uint16_t*>(a.data), b.data(), rows, cols);

    const uint64_t b_dims[2] = {a.dims[1], a.dims[0]};
    const int status = lassi_io_write(argv[2], "b", LASSI_IO_BF16, 2u, b_dims, b.data());
    lassi_io_free(&a);
    return status;
}
