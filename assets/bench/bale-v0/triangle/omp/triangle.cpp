// bale-v0 triangle, the OpenMP C++ program (task P17.40). Written here for LASSI; not upstream code.
//
// Counts the triangles of an undirected graph. The graph is built from pairs of vertices: a pair (u, u) is dropped,
// every other pair is the edge {u, v}, and a repeated edge counts once. The edges are stored as the strictly
// lower-triangular adjacency matrix L in compressed sparse rows (row max(u, v), column min(u, v)), each row's columns
// sorted. A triangle a > b > c is then counted once, at the entry (a, b) of L, whose rows a and b share the column c:
// the count is the sum, over every entry (i, j) of L, of the number of columns rows i and j share, each found by one
// merge over the two sorted rows. The rows of L are spread over OpenMP threads on the host CPU, and the per-row
// counts are summed by a reduction, so the total equals that of a serial loop. The graph is built serially. It follows
// the algorithm of the generic serial kernel in bale's src/other_serial/C/triangle.c (github.com/jdevinney/bale at
// 1b8f673; that file is Copyright (c) 2020, Institute for Defense Analyses, under a BSD-3-Clause-style license);
// no bale text is copied here.
// Usage: triangle <num_vertices> <edge_u> <edge_v> <triangles>
//   num_vertices  the lassi_io input file: an i64 scalar, the number of vertices n, at least 1
//   edge_u        the lassi_io input file: a one-dimensional i64 array of vertices in [0, n), one end of each pair
//   edge_v        the lassi_io input file: a one-dimensional i64 array of vertices in [0, n), the other ends
//   triangles     the lassi_io output file: an i64 scalar named "triangles", the number of triangles
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

#include "lassi_io.h"

namespace {

// The strictly lower-triangular adjacency matrix L in compressed sparse rows: the columns of row r are
// column[offset[r]] to column[offset[r + 1] - 1], sorted ascending, each below r.
struct LowerTriangle {
    std::vector<int64_t> offset;
    std::vector<int64_t> column;
};

// Read the lassi_io file at `path` into `array`; return false, saying why, unless it holds i64 data of `rank` dims.
bool read_i64(const char* path, uint32_t rank, lassi_io_array* array) {
    if (lassi_io_read(path, array) != 0) {
        return false;
    }
    if (array->dtype != LASSI_IO_I64 || array->rank != rank) {
        std::fprintf(stderr, "triangle: %s must hold an i64 array of rank %u\n", path, static_cast<unsigned>(rank));
        return false;
    }
    return true;
}

// Build L for n vertices from the m pairs (u[e], v[e]).
LowerTriangle lower_triangle(int64_t n, const int64_t* u, const int64_t* v, int64_t m) {
    std::vector<std::pair<int64_t, int64_t>> entries;
    entries.reserve(static_cast<std::size_t>(m));
    for (int64_t e = 0; e < m; ++e) {
        if (u[e] != v[e]) {
            entries.emplace_back(std::max(u[e], v[e]), std::min(u[e], v[e]));
        }
    }
    std::sort(entries.begin(), entries.end());
    entries.erase(std::unique(entries.begin(), entries.end()), entries.end());
    LowerTriangle lower;
    lower.offset.assign(static_cast<std::size_t>(n) + 1, 0);
    lower.column.reserve(entries.size());
    for (const std::pair<int64_t, int64_t>& entry : entries) {
        lower.offset[static_cast<std::size_t>(entry.first) + 1] += 1;
        lower.column.push_back(entry.second);
    }
    for (std::size_t r = 0; r < static_cast<std::size_t>(n); ++r) {
        lower.offset[r + 1] += lower.offset[r];
    }
    return lower;
}

// Return how many columns rows a and b of L share, by one merge over the two sorted rows.
int64_t shared_columns(const int64_t* offset, const int64_t* column, int64_t a, int64_t b) {
    int64_t p = offset[a];
    int64_t q = offset[b];
    const int64_t p_end = offset[a + 1];
    const int64_t q_end = offset[b + 1];
    int64_t shared = 0;
    while (p < p_end && q < q_end) {
        if (column[p] < column[q]) {
            ++p;
        } else if (column[q] < column[p]) {
            ++q;
        } else {
            ++shared;
            ++p;
            ++q;
        }
    }
    return shared;
}

// Return the number of triangles of the graph that L holds for n vertices, the rows counted in parallel.
int64_t count_triangles(const LowerTriangle& lower, int64_t n) {
    const int64_t* offset = lower.offset.data();
    const int64_t* column = lower.column.data();
    int64_t total = 0;
#pragma omp parallel for reduction(+ : total) schedule(dynamic, 64)
    for (int64_t row = 0; row < n; ++row) {
        for (int64_t k = offset[row]; k < offset[row + 1]; ++k) {
            total += shared_columns(offset, column, row, column[k]);
        }
    }
    return total;
}

// Check the inputs, count the triangles, and write the count to `out`; return the exit status.
int triangles(const lassi_io_array& num_vertices, const lassi_io_array& edge_u, const lassi_io_array& edge_v,
              const char* out) {
    const int64_t n = *static_cast<const int64_t*>(num_vertices.data);
    if (n < 1) {
        std::fprintf(stderr, "triangle: num_vertices must be at least 1\n");
        return 1;
    }
    if (edge_u.count != edge_v.count) {
        std::fprintf(stderr, "triangle: edge_u and edge_v must have the same length\n");
        return 1;
    }
    const int64_t* u = static_cast<const int64_t*>(edge_u.data);
    const int64_t* v = static_cast<const int64_t*>(edge_v.data);
    const int64_t m = static_cast<int64_t>(edge_u.count);
    for (int64_t e = 0; e < m; ++e) {
        if (u[e] < 0 || u[e] >= n || v[e] < 0 || v[e] >= n) {
            std::fprintf(stderr, "triangle: pair %lld names a vertex outside [0, num_vertices)\n",
                         static_cast<long long>(e));
            return 1;
        }
    }
    const LowerTriangle lower = lower_triangle(n, u, v, m);
    const int64_t total = count_triangles(lower, n);
    return lassi_io_write(out, "triangles", LASSI_IO_I64, 0u, nullptr, &total) == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: triangle <num_vertices> <edge_u> <edge_v> <triangles>\n");
        return 2;
    }
    lassi_io_array num_vertices = {};
    lassi_io_array edge_u = {};
    lassi_io_array edge_v = {};
    int status = 1;
    if (read_i64(argv[1], 0u, &num_vertices) && read_i64(argv[2], 1u, &edge_u) && read_i64(argv[3], 1u, &edge_v)) {
        status = triangles(num_vertices, edge_u, edge_v, argv[4]);
    }
    lassi_io_free(&num_vertices);
    lassi_io_free(&edge_u);
    lassi_io_free(&edge_v);
    return status;
}
