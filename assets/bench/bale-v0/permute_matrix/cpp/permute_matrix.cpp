// bale-v0 permute_matrix, the serial C++ program (task P17.43). Written here for LASSI; not upstream code.
//
// Moves the rows and the columns of a sparse n x n matrix A by two permutations. A is built from pairs of indices: it
// has a nonzero at (nz_row[e], nz_col[e]) for every pair e, a repeated pair counting once, and is stored in compressed
// sparse rows, each row's columns sorted ascending. The permutations are not drawn; each comes from drawn keys: row r
// of A goes to row row_dest[r], the position of r when the rows are ordered by (row_key[r], r) ascending, so rows with
// equal keys keep their index order, and column c goes to column col_dest[c], found the same way from col_key. Row i
// of the result is the row s of A with row_dest[s] == i, each of its columns c replaced by col_dest[c], in the order
// the columns have in row s (the result's rows are not sorted again). It follows the algorithm of the generic serial
// routine of bale's src/other_serial/C/permute_matrix.c, whose kernel is in src/other_serial/C/spmat_utils.c
// (github.com/jdevinney/bale at 1b8f673; those files are
// Copyright (c) 2020, Institute for Defense Analyses, under a BSD-3-Clause-style license); no bale text is copied
// here. Single-threaded.
// Usage: permute_matrix <num_rows> <nz_row> <nz_col> <row_key> <col_key> <offset> <column>
//   num_rows  the lassi_io input file: an i64 scalar, the side n of the square matrix A, at least 1
//   nz_row    the lassi_io input file: a one-dimensional i64 array of rows in [0, n), one per pair
//   nz_col    the lassi_io input file: a one-dimensional i64 array of columns in [0, n), as long as nz_row
//   row_key   the lassi_io input file: a one-dimensional i64 array of n keys, one per row of A
//   col_key   the lassi_io input file: a one-dimensional i64 array of n keys, one per column of A
//   offset    the lassi_io output file: an i64 array named "offset" of shape [n + 1], the result's row offsets:
//             row i holds column[offset[i]] to column[offset[i + 1] - 1]
//   column    the lassi_io output file: an i64 array named "column" of shape [nnz], the result's columns, row by row,
//             where nnz is the number of distinct pairs
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <utility>
#include <vector>

#include "lassi_io.h"

namespace {

// A matrix in compressed sparse rows: the columns of row r are column[offset[r]] to column[offset[r + 1] - 1].
struct SparseRows {
    std::vector<int64_t> offset;
    std::vector<int64_t> column;
};

// The five input arrays, in argument order.
struct Inputs {
    lassi_io_array num_rows;
    lassi_io_array nz_row;
    lassi_io_array nz_col;
    lassi_io_array row_key;
    lassi_io_array col_key;
};

// Read the lassi_io file at `path` into `array`; return false, saying why, unless it holds i64 data of `rank` dims.
bool read_i64(const char* path, uint32_t rank, lassi_io_array* array) {
    if (lassi_io_read(path, array) != 0) {
        return false;
    }
    if (array->dtype != LASSI_IO_I64 || array->rank != rank) {
        std::fprintf(stderr, "permute_matrix: %s must hold an i64 array of rank %u\n", path,
                     static_cast<unsigned>(rank));
        return false;
    }
    return true;
}

// Write `values` to the lassi_io file at `path` as a one-dimensional i64 array named `name`; return false on failure.
bool write_i64(const char* path, const char* name, const std::vector<int64_t>& values) {
    const uint64_t dims[1] = {static_cast<uint64_t>(values.size())};
    return lassi_io_write(path, name, LASSI_IO_I64, 1u, dims, values.data()) == 0;
}

// Return true when the inputs describe an n x n matrix and its keys; otherwise say why and return false.
bool check_inputs(const Inputs& in) {
    const int64_t n = *static_cast<const int64_t*>(in.num_rows.data);
    if (n < 1) {
        std::fprintf(stderr, "permute_matrix: num_rows must be at least 1\n");
        return false;
    }
    if (in.nz_row.count != in.nz_col.count) {
        std::fprintf(stderr, "permute_matrix: nz_row and nz_col must have the same length\n");
        return false;
    }
    if (in.row_key.count != static_cast<uint64_t>(n) || in.col_key.count != static_cast<uint64_t>(n)) {
        std::fprintf(stderr, "permute_matrix: row_key and col_key must each hold num_rows keys\n");
        return false;
    }
    const int64_t* row = static_cast<const int64_t*>(in.nz_row.data);
    const int64_t* col = static_cast<const int64_t*>(in.nz_col.data);
    const int64_t m = static_cast<int64_t>(in.nz_row.count);
    for (int64_t e = 0; e < m; ++e) {
        if (row[e] < 0 || row[e] >= n || col[e] < 0 || col[e] >= n) {
            std::fprintf(stderr, "permute_matrix: pair %lld lies outside [0, num_rows)\n", static_cast<long long>(e));
            return false;
        }
    }
    return true;
}

// Build the n x n matrix A from the m pairs (row[e], col[e]): a repeated pair counts once, and each row's columns are
// sorted ascending.
SparseRows build_matrix(int64_t n, const int64_t* row, const int64_t* col, int64_t m) {
    std::vector<std::pair<int64_t, int64_t>> entries;
    entries.reserve(static_cast<std::size_t>(m));
    for (int64_t e = 0; e < m; ++e) {
        entries.emplace_back(row[e], col[e]);
    }
    std::sort(entries.begin(), entries.end());
    entries.erase(std::unique(entries.begin(), entries.end()), entries.end());
    SparseRows a;
    a.offset.assign(static_cast<std::size_t>(n) + 1, 0);
    a.column.reserve(entries.size());
    for (const std::pair<int64_t, int64_t>& entry : entries) {
        a.offset[static_cast<std::size_t>(entry.first) + 1] += 1;
        a.column.push_back(entry.second);
    }
    for (std::size_t r = 0; r < static_cast<std::size_t>(n); ++r) {
        a.offset[r + 1] += a.offset[r];
    }
    return a;
}

// Return where each of the n indices goes: dest[i] is the position of i when the indices are ordered by (key[i], i)
// ascending, so indices with equal keys keep their order.
std::vector<int64_t> destinations(const int64_t* key, int64_t n) {
    std::vector<int64_t> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), int64_t{0});
    std::sort(order.begin(), order.end(),
              [key](int64_t a, int64_t b) { return key[a] < key[b] || (key[a] == key[b] && a < b); });
    std::vector<int64_t> dest(static_cast<std::size_t>(n));
    for (int64_t p = 0; p < n; ++p) {
        dest[static_cast<std::size_t>(order[static_cast<std::size_t>(p)])] = p;
    }
    return dest;
}

// Return A with row r moved to row row_dest[r] and column c to column col_dest[c]: row i of the result is the row s of
// A with row_dest[s] == i, its columns mapped through col_dest in the order they have in A.
SparseRows permute(const SparseRows& a, const std::vector<int64_t>& row_dest, const std::vector<int64_t>& col_dest,
                   int64_t n) {
    std::vector<int64_t> row_source(static_cast<std::size_t>(n));
    const int64_t* to_row = row_dest.data();
    int64_t* source = row_source.data();
    for (int64_t r = 0; r < n; ++r) {
        source[to_row[r]] = r;
    }
    const int64_t* a_offset = a.offset.data();
    const int64_t* a_column = a.column.data();
    const int64_t* to_column = col_dest.data();
    SparseRows result;
    result.offset.assign(static_cast<std::size_t>(n) + 1, 0);
    result.column.assign(a.column.size(), 0);
    int64_t* offset = result.offset.data();
    int64_t* column = result.column.data();
    int64_t pos = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t s = source[i];
        for (int64_t k = a_offset[s]; k < a_offset[s + 1]; ++k) {
            column[pos] = to_column[a_column[k]];
            ++pos;
        }
        offset[i + 1] = pos;
    }
    return result;
}

// Check the inputs, permute the matrix, and write the result to `out_offset` and `out_column`; return the exit status.
int permute_and_write(const Inputs& in, const char* out_offset, const char* out_column) {
    if (!check_inputs(in)) {
        return 1;
    }
    const int64_t n = *static_cast<const int64_t*>(in.num_rows.data);
    const SparseRows a = build_matrix(n, static_cast<const int64_t*>(in.nz_row.data),
                                      static_cast<const int64_t*>(in.nz_col.data),
                                      static_cast<int64_t>(in.nz_row.count));
    const std::vector<int64_t> row_dest = destinations(static_cast<const int64_t*>(in.row_key.data), n);
    const std::vector<int64_t> col_dest = destinations(static_cast<const int64_t*>(in.col_key.data), n);
    const SparseRows result = permute(a, row_dest, col_dest, n);
    if (!write_i64(out_offset, "offset", result.offset) || !write_i64(out_column, "column", result.column)) {
        return 1;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 8) {
        std::fprintf(stderr,
                     "usage: permute_matrix <num_rows> <nz_row> <nz_col> <row_key> <col_key> <offset> <column>\n");
        return 2;
    }
    Inputs in = {};
    int status = 1;
    if (read_i64(argv[1], 0u, &in.num_rows) && read_i64(argv[2], 1u, &in.nz_row) && read_i64(argv[3], 1u, &in.nz_col) &&
        read_i64(argv[4], 1u, &in.row_key) && read_i64(argv[5], 1u, &in.col_key)) {
        status = permute_and_write(in, argv[6], argv[7]);
    }
    lassi_io_free(&in.num_rows);
    lassi_io_free(&in.nz_row);
    lassi_io_free(&in.nz_col);
    lassi_io_free(&in.row_key);
    lassi_io_free(&in.col_key);
    return status;
}
