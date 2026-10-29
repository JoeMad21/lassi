// bale-v0 toposort, the serial C++ program (task P17.43). Written here for LASSI; not upstream code.
//
// Finds a row and a column permutation that make a morally upper triangular sparse matrix upper triangular. The
// matrix A, of n rows and n columns, is built from drawn arrays. U is the upper triangular matrix with every
// diagonal entry (i, i) and, for each drawn pair (u, v) with u != v, the entry (min(u, v), max(u, v)); a pair (u, u)
// is dropped, and an entry drawn more than once counts once. Row i of U becomes row rank(row_key, i) of A, and column
// j of U becomes column rank(col_key, j), where rank(key, i) is the position of i when 0 to n - 1 are sorted by key,
// ties by index (a stable argsort). A is stored as one word per row and as its columns: each row's word is
// 2**32 times the number of its remaining nonzeros plus the sum of their columns, so a row with one nonzero left
// holds that column in its low bits, which are read through a 16-bit mask as bale reads them (right for every n
// below 65536); each column lists the rows of A that hold it.
// The permutations are defined so that they are unique (bale accepts any pair that works), by peeling A in levels.
// Level 0 is every row of A with exactly one nonzero; each later level is every row with exactly one nonzero left
// once the columns of all earlier levels are crossed out of every row. The rows of a level take the next positions
// from the end in increasing row order: the k-th row taken, counting from 0 over all levels, is a row r with one
// remaining column c, and row_perm[r] = col_perm[c] = n - 1 - k. Only once every row of a level has its position are
// the level's columns crossed out (each row that holds column c loses 2**32 + c from its word), and the next level
// is found by a sweep over all rows. It follows the algorithm of toposort_matrix_loop in bale's
// src/other_serial/C/toposort.c, with these levels in place of its sweeps, and builds A as that file's
// generate_toposort_input does, from an upper triangular matrix and two permutations (github.com/jdevinney/bale at
// 1b8f673; that file is Copyright (c) 2020, Institute for Defense Analyses, under a BSD-3-Clause-style license);
// no bale text is copied here. Single-threaded.
// Usage: toposort <num_rows> <extra_u> <extra_v> <row_key> <col_key> <row_perm> <col_perm>
//   num_rows  the lassi_io input file: an i64 scalar, the number of rows and columns n, at least 1
//   extra_u   the lassi_io input file: a one-dimensional i64 array of indexes in [0, n), one end of each drawn pair
//   extra_v   the lassi_io input file: a one-dimensional i64 array of indexes in [0, n), the other ends, as long
//   row_key   the lassi_io input file: a one-dimensional i64 array of n keys, whose stable argsort orders U's rows
//   col_key   the lassi_io input file: a one-dimensional i64 array of n keys, whose stable argsort orders U's columns
//   row_perm  the lassi_io output file: an i64 array named "row_perm" of shape [n], the row that row r of A moves to
//   col_perm  the lassi_io output file: an i64 array named "col_perm" of shape [n], the column that column c of A
//             moves to; moved so, every nonzero (r, c) of A has row_perm[r] <= col_perm[c]
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

#include "lassi_io.h"

namespace {

// What one remaining nonzero adds to its row's word, besides its column: 1 in the count, the high 32 bits.
constexpr int64_t kOne = int64_t{1} << 32;
// bale's mask for the remaining column of a row with one nonzero left.
constexpr int64_t kColumnMask = 0xFFFF;

// A nonzero of a matrix, as (row, column).
using Entry = std::pair<int64_t, int64_t>;

// The columns of A: the rows of A that hold a nonzero in column c are row[offset[c]] to row[offset[c + 1] - 1].
struct Columns {
    std::vector<int64_t> offset;
    std::vector<int64_t> row;
};

// Read the lassi_io file at `path` into `array`; return false, saying why, unless it holds i64 data of `rank` dims.
bool read_i64(const char* path, uint32_t rank, lassi_io_array* array) {
    if (lassi_io_read(path, array) != 0) {
        return false;
    }
    if (array->dtype != LASSI_IO_I64 || array->rank != rank) {
        std::fprintf(stderr, "toposort: %s must hold an i64 array of rank %u\n", path, static_cast<unsigned>(rank));
        return false;
    }
    return true;
}

// Return rank, where rank[i] is the position of i when 0 to n - 1 are sorted by key[i], ties by i.
std::vector<int64_t> stable_rank(const int64_t* key, int64_t n) {
    std::vector<Entry> order(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        order[static_cast<std::size_t>(i)] = Entry(key[i], i);
    }
    std::sort(order.begin(), order.end());
    std::vector<int64_t> rank(static_cast<std::size_t>(n));
    for (int64_t position = 0; position < n; ++position) {
        rank[static_cast<std::size_t>(order[static_cast<std::size_t>(position)].second)] = position;
    }
    return rank;
}

// Return the nonzeros of A: those of U (its diagonal and (min(u[e], v[e]), max(u[e], v[e])) for each of the m pairs
// with u[e] != v[e], each once), with row i of U moved to row row_rank[i] and column j to column col_rank[j].
std::vector<Entry> moral_entries(int64_t n, const int64_t* u, const int64_t* v, int64_t m,
                                 const std::vector<int64_t>& row_rank, const std::vector<int64_t>& col_rank) {
    std::vector<Entry> upper;
    upper.reserve(static_cast<std::size_t>(n + m));
    for (int64_t i = 0; i < n; ++i) {
        upper.emplace_back(i, i);
    }
    for (int64_t e = 0; e < m; ++e) {
        if (u[e] != v[e]) {
            upper.emplace_back(std::min(u[e], v[e]), std::max(u[e], v[e]));
        }
    }
    std::sort(upper.begin(), upper.end());
    upper.erase(std::unique(upper.begin(), upper.end()), upper.end());
    std::vector<Entry> entries;
    entries.reserve(upper.size());
    for (const Entry& entry : upper) {
        entries.emplace_back(row_rank[static_cast<std::size_t>(entry.first)],
                             col_rank[static_cast<std::size_t>(entry.second)]);
    }
    return entries;
}

// Return the columns of A, an n by n matrix with these nonzeros, by a counting sort of the nonzeros by column.
Columns columns_of(const std::vector<Entry>& entries, int64_t n) {
    Columns columns;
    columns.offset.assign(static_cast<std::size_t>(n) + 1, 0);
    columns.row.assign(entries.size(), 0);
    for (const Entry& entry : entries) {
        columns.offset[static_cast<std::size_t>(entry.second) + 1] += 1;
    }
    for (std::size_t c = 0; c < static_cast<std::size_t>(n); ++c) {
        columns.offset[c + 1] += columns.offset[c];
    }
    std::vector<int64_t> next(columns.offset.begin(), columns.offset.end() - 1);
    for (const Entry& entry : entries) {
        columns.row[static_cast<std::size_t>(next[static_cast<std::size_t>(entry.second)]++)] = entry.first;
    }
    return columns;
}

// Peel A level by level and write row_perm and col_perm, n each; return how many rows were taken, which is n
// whenever A is morally upper triangular with a full diagonal, as every A built here is.
int64_t toposort(const std::vector<Entry>& entries, const Columns& columns, int64_t n, int64_t* row_perm,
                 int64_t* col_perm) {
    std::vector<int64_t> words(static_cast<std::size_t>(n), 0);
    for (const Entry& entry : entries) {
        words[static_cast<std::size_t>(entry.first)] += kOne + entry.second;
    }
    int64_t* word = words.data();
    const int64_t* offset = columns.offset.data();
    const int64_t* holder = columns.row.data();
    std::vector<int64_t> level_rows(static_cast<std::size_t>(n));
    std::vector<int64_t> level_cols(static_cast<std::size_t>(n));
    int64_t* level_row = level_rows.data();
    int64_t* level_col = level_cols.data();
    int64_t taken = 0;
    while (taken < n) {
        int64_t width = 0;
        for (int64_t r = 0; r < n; ++r) {
            if ((word[r] >> 32) == 1) {
                level_row[width++] = r;
            }
        }
        if (width == 0) {
            break;
        }
        const int64_t last = n - 1 - taken;
        for (int64_t k = 0; k < width; ++k) {
            const int64_t r = level_row[k];
            const int64_t c = word[r] & kColumnMask;
            row_perm[r] = last - k;
            col_perm[c] = last - k;
            level_col[k] = c;
        }
        for (int64_t k = 0; k < width; ++k) {
            const int64_t c = level_col[k];
            const int64_t drop = kOne + c;
            for (int64_t p = offset[c]; p < offset[c + 1]; ++p) {
                word[holder[p]] -= drop;
            }
        }
        taken += width;
    }
    return taken;
}

// Return true when every value of the array lies in [0, n); otherwise say which does not and return false.
bool all_below(const lassi_io_array& array, int64_t n, const char* what) {
    const int64_t* value = static_cast<const int64_t*>(array.data);
    for (uint64_t i = 0; i < array.count; ++i) {
        if (value[i] < 0 || value[i] >= n) {
            std::fprintf(stderr, "toposort: %s[%llu] is outside [0, num_rows)\n", what,
                         static_cast<unsigned long long>(i));
            return false;
        }
    }
    return true;
}

// Check the inputs, build A, peel it, and write the permutations to `row_out` and `col_out`; return the exit status.
int run(const lassi_io_array* in, const char* row_out, const char* col_out) {
    const int64_t n = *static_cast<const int64_t*>(in[0].data);
    if (n < 1) {
        std::fprintf(stderr, "toposort: num_rows must be at least 1\n");
        return 1;
    }
    if (in[1].count != in[2].count) {
        std::fprintf(stderr, "toposort: extra_u and extra_v must have the same length\n");
        return 1;
    }
    if (in[3].count != static_cast<uint64_t>(n) || in[4].count != static_cast<uint64_t>(n)) {
        std::fprintf(stderr, "toposort: row_key and col_key must each hold num_rows keys\n");
        return 1;
    }
    if (!all_below(in[1], n, "extra_u") || !all_below(in[2], n, "extra_v")) {
        return 1;
    }
    const std::vector<int64_t> row_rank = stable_rank(static_cast<const int64_t*>(in[3].data), n);
    const std::vector<int64_t> col_rank = stable_rank(static_cast<const int64_t*>(in[4].data), n);
    const std::vector<Entry> entries =
        moral_entries(n, static_cast<const int64_t*>(in[1].data), static_cast<const int64_t*>(in[2].data),
                      static_cast<int64_t>(in[1].count), row_rank, col_rank);
    const Columns columns = columns_of(entries, n);
    std::vector<int64_t> row_perm(static_cast<std::size_t>(n), 0);
    std::vector<int64_t> col_perm(static_cast<std::size_t>(n), 0);
    const int64_t taken = toposort(entries, columns, n, row_perm.data(), col_perm.data());
    if (taken != n) {
        std::fprintf(stderr, "toposort: took %lld rows, not num_rows = %lld\n", static_cast<long long>(taken),
                     static_cast<long long>(n));
        return 1;
    }
    const uint64_t dims[1] = {static_cast<uint64_t>(n)};
    if (lassi_io_write(row_out, "row_perm", LASSI_IO_I64, 1u, dims, row_perm.data()) != 0) {
        return 1;
    }
    return lassi_io_write(col_out, "col_perm", LASSI_IO_I64, 1u, dims, col_perm.data()) == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 8) {
        std::fprintf(stderr,
                     "usage: toposort <num_rows> <extra_u> <extra_v> <row_key> <col_key> <row_perm> <col_perm>\n");
        return 2;
    }
    lassi_io_array in[5] = {};
    const uint32_t ranks[5] = {0u, 1u, 1u, 1u, 1u};
    bool ok = true;
    for (int i = 0; i < 5 && ok; ++i) {
        ok = read_i64(argv[1 + i], ranks[i], &in[i]);
    }
    const int status = ok ? run(in, argv[6], argv[7]) : 1;
    for (lassi_io_array& array : in) {
        lassi_io_free(&array);
    }
    return status;
}
