// bale-v0 transpose_matrix, the OpenMP C++ program (task P17.43). Written here for LASSI; not upstream code.
//
// The transpose of a sparse matrix. The matrix A, of num_rows rows and num_cols columns, is built from drawn entries:
// entry e is the nonzero (entry_row[e], entry_col[e]), and an entry drawn more than once counts once. A is stored in
// compressed sparse rows, its rows in ascending order and each row's columns sorted ascending. Its transpose, of
// num_cols rows and num_rows columns, is built by a counting sort of A's entries by column, as the serial program
// builds it, with A's rows split into bands of consecutive rows (64 bands, or one per row when A has fewer rows) that
// OpenMP threads on the host CPU share: each band counts the entries of each column c in its own rows; the running
// sums of the column totals give the row offsets of the transpose, and inside row c of the transpose each band's slots
// follow those of the bands before it; then each band visits its rows in ascending order and, for each entry (r, c),
// puts r in its next free slot of row c. No two bands share a slot, and the bands are fixed by num_rows alone, not by
// the number of threads, so each row of the transpose comes out with its columns sorted ascending and the outputs equal
// the serial program's for any number of threads. A is built serially. It follows the algorithm of transpose_matrix in
// bale's src/other_serial/C/spmat_utils.c, the routine that bale's src/other_serial/C/transpose_matrix.c runs
// (github.com/jdevinney/bale at 1b8f673; those files are Copyright (c) 2020, Institute for Defense Analyses, under a
// BSD-3-Clause-style license), with the bands written here; no bale text is copied here.
// Usage: transpose_matrix <num_rows> <num_cols> <entry_row> <entry_col> <offset> <column>
//   num_rows   the lassi_io input file: an i64 scalar, the number of rows of A, at least 1
//   num_cols   the lassi_io input file: an i64 scalar, the number of columns of A, at least 1
//   entry_row  the lassi_io input file: a one-dimensional i64 array of rows in [0, num_rows), the row of each entry
//   entry_col  the lassi_io input file: a one-dimensional i64 array of columns in [0, num_cols), as long as entry_row
//   offset     the lassi_io output file: an i64 array named "offset" of shape [num_cols + 1], the transpose's row
//              offsets: row c of the transpose holds column[offset[c]] to column[offset[c + 1] - 1]
//   column     the lassi_io output file: an i64 array named "column" of shape [nnz], nnz the number of distinct
//              entries: for each c in turn, the rows r of A that hold an entry (r, c), sorted ascending
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

#include "lassi_io.h"

namespace {

// The number of bands of consecutive rows of A that the transpose is split into (fewer when A has fewer rows). It is
// fixed, not the number of threads, so every band, and every slot its entries take, is the same for any thread count.
constexpr int64_t kBands = 64;

// A sparse matrix in compressed sparse rows, with no values: the columns of row r are column[offset[r]] to
// column[offset[r + 1] - 1], sorted ascending.
struct SparseRows {
    std::vector<int64_t> offset;
    std::vector<int64_t> column;
};

// Read the lassi_io file at `path` into `array`; return false, saying why, unless it holds i64 data of `rank` dims.
bool read_i64(const char* path, uint32_t rank, lassi_io_array* array) {
    if (lassi_io_read(path, array) != 0) {
        return false;
    }
    if (array->dtype != LASSI_IO_I64 || array->rank != rank) {
        std::fprintf(stderr, "transpose_matrix: %s must hold an i64 array of rank %u\n", path,
                     static_cast<unsigned>(rank));
        return false;
    }
    return true;
}

// Build A, of num_rows rows, from the m entries (row[e], col[e]): each distinct entry once, rows in ascending order,
// each row's columns sorted ascending.
SparseRows build_matrix(int64_t num_rows, const int64_t* row, const int64_t* col, int64_t m) {
    std::vector<std::pair<int64_t, int64_t>> entries;
    entries.reserve(static_cast<std::size_t>(m));
    for (int64_t e = 0; e < m; ++e) {
        entries.emplace_back(row[e], col[e]);
    }
    std::sort(entries.begin(), entries.end());
    entries.erase(std::unique(entries.begin(), entries.end()), entries.end());
    SparseRows a;
    a.offset.assign(static_cast<std::size_t>(num_rows) + 1, 0);
    a.column.reserve(entries.size());
    for (const std::pair<int64_t, int64_t>& entry : entries) {
        a.offset[static_cast<std::size_t>(entry.first) + 1] += 1;
        a.column.push_back(entry.second);
    }
    for (std::size_t r = 0; r < static_cast<std::size_t>(num_rows); ++r) {
        a.offset[r + 1] += a.offset[r];
    }
    return a;
}

// Return the first row of band b of `bands` bands over num_rows rows; band b holds rows band_start(b) to
// band_start(b + 1) - 1.
int64_t band_start(int64_t b, int64_t bands, int64_t num_rows) {
    return b * num_rows / bands;
}

// Set slot[b * num_cols + c] to the number of entries of column c in band b's rows, the bands counted in parallel.
void count_columns(const SparseRows& a, int64_t num_rows, int64_t num_cols, int64_t bands, int64_t* slot) {
    const int64_t* a_offset = a.offset.data();
    const int64_t* a_column = a.column.data();
#pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < bands; ++b) {
        int64_t* count = slot + b * num_cols;
        const int64_t first = a_offset[band_start(b, bands, num_rows)];
        const int64_t end = a_offset[band_start(b + 1, bands, num_rows)];
        for (int64_t k = first; k < end; ++k) {
            count[a_column[k]] += 1;
        }
    }
}

// Set t_offset, of num_cols + 1 values, to the transpose's row offsets, and turn each count slot[b * num_cols + c]
// into the first slot of row c of the transpose that band b fills; the columns are done in parallel, all but the
// running sums of the column totals, which are serial.
void place_bands(int64_t num_cols, int64_t bands, int64_t* slot, int64_t* t_offset) {
    t_offset[0] = 0;
#pragma omp parallel for schedule(static)
    for (int64_t c = 0; c < num_cols; ++c) {
        int64_t total = 0;
        for (int64_t b = 0; b < bands; ++b) {
            total += slot[b * num_cols + c];
        }
        t_offset[c + 1] = total;
    }
    for (int64_t c = 0; c < num_cols; ++c) {
        t_offset[c + 1] += t_offset[c];
    }
#pragma omp parallel for schedule(static)
    for (int64_t c = 0; c < num_cols; ++c) {
        int64_t next = t_offset[c];
        for (int64_t b = 0; b < bands; ++b) {
            const int64_t count = slot[b * num_cols + c];
            slot[b * num_cols + c] = next;
            next += count;
        }
    }
}

// Return the transpose of A, which has num_rows rows and num_cols columns, by a counting sort of its entries by
// column over bands of A's rows: count each band's entries per column, place each band's slots, then let each band
// put its rows, in ascending order, each at its next free slot of its column's row of the transpose.
SparseRows transpose(const SparseRows& a, int64_t num_rows, int64_t num_cols) {
    const int64_t bands = std::min(kBands, num_rows);
    // slot[b * num_cols + c]: first the count of band b's entries in column c, then band b's next free slot of row c
    // of the transpose.
    std::vector<int64_t> slots(static_cast<std::size_t>(bands * num_cols), 0);
    int64_t* slot = slots.data();
    count_columns(a, num_rows, num_cols, bands, slot);
    SparseRows t;
    t.offset.assign(static_cast<std::size_t>(num_cols) + 1, 0);
    t.column.assign(a.column.size(), 0);
    place_bands(num_cols, bands, slot, t.offset.data());
    const int64_t* a_offset = a.offset.data();
    const int64_t* a_column = a.column.data();
    int64_t* t_column = t.column.data();
#pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < bands; ++b) {
        int64_t* next = slot + b * num_cols;
        for (int64_t r = band_start(b, bands, num_rows); r < band_start(b + 1, bands, num_rows); ++r) {
            for (int64_t k = a_offset[r]; k < a_offset[r + 1]; ++k) {
                const int64_t c = a_column[k];
                t_column[next[c]] = r;
                next[c] += 1;
            }
        }
    }
    return t;
}

// Check the inputs, build A, transpose it, and write the transpose to `offset_out` and `column_out`; return the exit
// status.
int transpose_matrix(const lassi_io_array& num_rows, const lassi_io_array& num_cols, const lassi_io_array& entry_row,
                     const lassi_io_array& entry_col, const char* offset_out, const char* column_out) {
    const int64_t rows = *static_cast<const int64_t*>(num_rows.data);
    const int64_t cols = *static_cast<const int64_t*>(num_cols.data);
    if (rows < 1 || cols < 1) {
        std::fprintf(stderr, "transpose_matrix: num_rows and num_cols must be at least 1\n");
        return 1;
    }
    if (entry_row.count != entry_col.count) {
        std::fprintf(stderr, "transpose_matrix: entry_row and entry_col must have the same length\n");
        return 1;
    }
    const int64_t* row = static_cast<const int64_t*>(entry_row.data);
    const int64_t* col = static_cast<const int64_t*>(entry_col.data);
    const int64_t m = static_cast<int64_t>(entry_row.count);
    for (int64_t e = 0; e < m; ++e) {
        if (row[e] < 0 || row[e] >= rows || col[e] < 0 || col[e] >= cols) {
            std::fprintf(stderr, "transpose_matrix: entry %lld lies outside the num_rows by num_cols matrix\n",
                         static_cast<long long>(e));
            return 1;
        }
    }
    const SparseRows a = build_matrix(rows, row, col, m);
    const SparseRows t = transpose(a, rows, cols);
    const uint64_t offset_dims[1] = {static_cast<uint64_t>(t.offset.size())};
    const uint64_t column_dims[1] = {static_cast<uint64_t>(t.column.size())};
    if (lassi_io_write(offset_out, "offset", LASSI_IO_I64, 1u, offset_dims, t.offset.data()) != 0) {
        return 1;
    }
    return lassi_io_write(column_out, "column", LASSI_IO_I64, 1u, column_dims, t.column.data()) == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 7) {
        std::fprintf(stderr,
                     "usage: transpose_matrix <num_rows> <num_cols> <entry_row> <entry_col> <offset> <column>\n");
        return 2;
    }
    lassi_io_array num_rows = {};
    lassi_io_array num_cols = {};
    lassi_io_array entry_row = {};
    lassi_io_array entry_col = {};
    int status = 1;
    if (read_i64(argv[1], 0u, &num_rows) && read_i64(argv[2], 0u, &num_cols) && read_i64(argv[3], 1u, &entry_row) &&
        read_i64(argv[4], 1u, &entry_col)) {
        status = transpose_matrix(num_rows, num_cols, entry_row, entry_col, argv[5], argv[6]);
    }
    lassi_io_free(&num_rows);
    lassi_io_free(&num_cols);
    lassi_io_free(&entry_row);
    lassi_io_free(&entry_col);
    return status;
}
