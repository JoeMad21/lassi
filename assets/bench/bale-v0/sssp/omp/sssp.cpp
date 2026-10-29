// bale-v0 sssp, the OpenMP C++ program (task P17.43). Written here for LASSI; not upstream code.
//
// Single-source shortest paths from vertex 0 in a directed graph with integer edge weights. The graph is built from
// drawn triples (tail, head, weight): a triple with tail == head is dropped, every other one is the edge tail -> head,
// and a pair (tail, head) drawn more than once is one edge with the lightest of its weights. The distances are found
// by Bellman-Ford in rounds: vertex 0 starts at 0 and every other vertex unreached; each round starts from the last
// round's distances and relaxes every edge whose tail's distance changed in the last round (vertex 0 alone in the
// first round), adding the edge's weight to the tail's distance of the last round; the rounds stop at the first one
// that changes no distance, at the latest round num_vertices. A vertex no path reaches gets the distance -1. Here the
// edges are stored in compressed sparse rows by head, the transpose of bale's rows, so that each round is a pull: the
// vertices are spread over OpenMP threads on the host CPU, the thread that holds a vertex sets its new distance to the
// least of its last distance and the relaxations of its incoming edges, and whether any distance changed is found by
// a reduction. A minimum does not depend on the order it is taken in, so every round gives the distances of the
// serial program's round, whatever the number of threads. The graph is built serially. It follows the algorithm of
// bale's Bellman-Ford that recycles its three arrays of tentative distances, in src/other_serial/C/sssp_bellmanford.c,
// as bale's app src/other_serial/C/sssp.c runs it (directed, weighted, no loops, source 0) (github.com/jdevinney/bale
// at 1b8f673; those files are Copyright (c) 2020, Institute for Defense Analyses, under a BSD-3-Clause-style
// license); no bale text is copied here.
// Usage: sssp <num_vertices> <edge_tail> <edge_head> <edge_weight> <distance>
//   num_vertices  the lassi_io input file: an i64 scalar, the number of vertices n, in [1, 2**31)
//   edge_tail     the lassi_io input file: a one-dimensional i64 array of vertices in [0, n), the tail of each edge
//   edge_head     the lassi_io input file: a one-dimensional i64 array of vertices in [0, n), the heads, as long
//   edge_weight   the lassi_io input file: a one-dimensional i64 array of weights in [0, 2**32), as long
//   distance      the lassi_io output file: an i64 array named "distance" of shape [n], the length of a shortest path
//                 from vertex 0 to each vertex, or -1 when no path reaches it
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <tuple>
#include <vector>

#include "lassi_io.h"

namespace {

// The distance of a vertex no path has reached; nothing is ever added to it.
constexpr int64_t kUnreached = INT64_MAX;
// The bounds that keep every sum exact: a path has fewer than 2**31 edges, each lighter than 2**32.
constexpr int64_t kVertexLimit = int64_t{1} << 31;
constexpr int64_t kWeightLimit = int64_t{1} << 32;

// The graph in compressed sparse rows by head: the edges into vertex h come from tail[offset[h]] to
// tail[offset[h + 1] - 1], ascending and distinct, with their weights at the same positions of weight.
struct Graph {
    std::vector<int64_t> offset;
    std::vector<int64_t> tail;
    std::vector<int64_t> weight;
};

// Read the lassi_io file at `path` into `array`; return false, saying why, unless it holds i64 data of `rank` dims.
bool read_i64(const char* path, uint32_t rank, lassi_io_array* array) {
    if (lassi_io_read(path, array) != 0) {
        return false;
    }
    if (array->dtype != LASSI_IO_I64 || array->rank != rank) {
        std::fprintf(stderr, "sssp: %s must hold an i64 array of rank %u\n", path, static_cast<unsigned>(rank));
        return false;
    }
    return true;
}

// Build the graph of n vertices from the m triples (tail[e], head[e], weight[e]), its rows by head.
Graph build_graph(int64_t n, const int64_t* tail, const int64_t* head, const int64_t* weight, int64_t m) {
    std::vector<std::tuple<int64_t, int64_t, int64_t>> edges;
    edges.reserve(static_cast<std::size_t>(m));
    for (int64_t e = 0; e < m; ++e) {
        if (tail[e] != head[e]) {
            edges.emplace_back(head[e], tail[e], weight[e]);
        }
    }
    // Sorted, the triples of one pair (tail, head) lie together, the lightest first; the others are dropped.
    std::sort(edges.begin(), edges.end());
    Graph graph;
    graph.offset.assign(static_cast<std::size_t>(n) + 1, 0);
    graph.tail.reserve(edges.size());
    graph.weight.reserve(edges.size());
    for (std::size_t k = 0; k < edges.size(); ++k) {
        const int64_t h = std::get<0>(edges[k]);
        const int64_t t = std::get<1>(edges[k]);
        if (k > 0 && std::get<0>(edges[k - 1]) == h && std::get<1>(edges[k - 1]) == t) {
            continue;
        }
        graph.offset[static_cast<std::size_t>(h) + 1] += 1;
        graph.tail.push_back(t);
        graph.weight.push_back(std::get<2>(edges[k]));
    }
    for (std::size_t r = 0; r < static_cast<std::size_t>(n); ++r) {
        graph.offset[r + 1] += graph.offset[r];
    }
    return graph;
}

// Return the distance from vertex 0 to each of the n vertices of `graph`, kUnreached where no path reaches; the
// vertices of each round are spread over the threads, each writing only its own new distance.
std::vector<int64_t> shortest_paths(const Graph& graph, int64_t n) {
    const int64_t* offset = graph.offset.data();
    const int64_t* tail = graph.tail.data();
    const int64_t* weight = graph.weight.data();
    // before: the distances of the round before the last; last: the last round's; next: this round's.
    std::vector<int64_t> before(static_cast<std::size_t>(n), kUnreached);
    std::vector<int64_t> last(before);
    std::vector<int64_t> next(before);
    last[0] = 0;
    int changed = 1;
    while (changed != 0) {
        changed = 0;
        const int64_t* before_d = before.data();
        const int64_t* last_d = last.data();
        int64_t* next_d = next.data();
#pragma omp parallel for reduction(| : changed) schedule(static)
        for (int64_t h = 0; h < n; ++h) {
            int64_t least = last_d[h];
            for (int64_t k = offset[h]; k < offset[h + 1]; ++k) {
                const int64_t t = tail[k];
                // Only a distance the last round changed is relaxed; it fell, so it is below kUnreached.
                if (before_d[t] != last_d[t] && last_d[t] + weight[k] < least) {
                    least = last_d[t] + weight[k];
                }
            }
            next_d[h] = least;
            if (least != last_d[h]) {
                changed = 1;
            }
        }
        before.swap(last);
        last.swap(next);
    }
    return last;
}

// Return true when every edge names vertices in [0, n) and a weight in [0, 2**32); else say which does not.
bool edges_ok(int64_t n, const int64_t* tail, const int64_t* head, const int64_t* weight, int64_t m) {
    for (int64_t e = 0; e < m; ++e) {
        if (tail[e] < 0 || tail[e] >= n || head[e] < 0 || head[e] >= n) {
            std::fprintf(stderr, "sssp: edge %lld names a vertex outside [0, num_vertices)\n",
                         static_cast<long long>(e));
            return false;
        }
        if (weight[e] < 0 || weight[e] >= kWeightLimit) {
            std::fprintf(stderr, "sssp: edge %lld has a weight outside [0, 2**32)\n", static_cast<long long>(e));
            return false;
        }
    }
    return true;
}

// Check the inputs, find the distances, and write them to `out`; return the exit status.
int sssp(const lassi_io_array& num_vertices, const lassi_io_array& edge_tail, const lassi_io_array& edge_head,
         const lassi_io_array& edge_weight, const char* out) {
    const int64_t n = *static_cast<const int64_t*>(num_vertices.data);
    if (n < 1 || n >= kVertexLimit) {
        std::fprintf(stderr, "sssp: num_vertices must be in [1, 2**31)\n");
        return 1;
    }
    if (edge_head.count != edge_tail.count || edge_weight.count != edge_tail.count) {
        std::fprintf(stderr, "sssp: edge_tail, edge_head, and edge_weight must have the same length\n");
        return 1;
    }
    const int64_t* tail = static_cast<const int64_t*>(edge_tail.data);
    const int64_t* head = static_cast<const int64_t*>(edge_head.data);
    const int64_t* weight = static_cast<const int64_t*>(edge_weight.data);
    const int64_t m = static_cast<int64_t>(edge_tail.count);
    if (!edges_ok(n, tail, head, weight, m)) {
        return 1;
    }
    std::vector<int64_t> distance = shortest_paths(build_graph(n, tail, head, weight, m), n);
    for (int64_t& d : distance) {
        if (d == kUnreached) {
            d = -1;
        }
    }
    const uint64_t dims[1] = {static_cast<uint64_t>(n)};
    return lassi_io_write(out, "distance", LASSI_IO_I64, 1u, dims, distance.data()) == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: sssp <num_vertices> <edge_tail> <edge_head> <edge_weight> <distance>\n");
        return 2;
    }
    lassi_io_array num_vertices = {};
    lassi_io_array edge_tail = {};
    lassi_io_array edge_head = {};
    lassi_io_array edge_weight = {};
    int status = 1;
    if (read_i64(argv[1], 0u, &num_vertices) && read_i64(argv[2], 1u, &edge_tail) &&
        read_i64(argv[3], 1u, &edge_head) && read_i64(argv[4], 1u, &edge_weight)) {
        status = sssp(num_vertices, edge_tail, edge_head, edge_weight, argv[5]);
    }
    lassi_io_free(&num_vertices);
    lassi_io_free(&edge_tail);
    lassi_io_free(&edge_head);
    lassi_io_free(&edge_weight);
    return status;
}
