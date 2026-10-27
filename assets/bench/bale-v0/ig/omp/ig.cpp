// bale-v0 ig, the OpenMP C++ program (task P17.40). Written here for LASSI; not upstream code.
//
// An index gather: tgt[i] = table[index[i]] for every request i. The requests are spread over OpenMP threads on the
// host CPU; each writes only its own tgt[i], so the result equals that of a serial loop. It follows the algorithm of
// the generic serial gather in bale's src/other_serial/C/ig.c (github.com/jdevinney/bale at 1b8f673; that file is
// Copyright (c) 2020, Institute for Defense Analyses, under a BSD-3-Clause-style license); no bale text is copied
// here.
// Usage: ig <table> <index> <tgt>
//   table  the lassi_io input file: a one-dimensional i64 array, the values to read from
//   index  the lassi_io input file: a one-dimensional i64 array of requests, each a position in [0, length of table)
//   tgt    the lassi_io output file: an i64 array named "tgt" of index's shape
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "lassi_io.h"

namespace {

// Read the lassi_io file at `path` into `array`; return false, saying why, unless it holds i64 data of `rank` dims.
bool read_i64(const char* path, uint32_t rank, lassi_io_array* array) {
    if (lassi_io_read(path, array) != 0) {
        return false;
    }
    if (array->dtype != LASSI_IO_I64 || array->rank != rank) {
        std::fprintf(stderr, "ig: %s must hold an i64 array of rank %u\n", path, static_cast<unsigned>(rank));
        return false;
    }
    return true;
}

// Gather table[index[i]] for every request in parallel and write the results to `out`; return the exit status.
int gather(const lassi_io_array& table, const lassi_io_array& index, const char* out) {
    const int64_t* values = static_cast<const int64_t*>(table.data);
    const int64_t size = static_cast<int64_t>(table.count);
    const int64_t* requests = static_cast<const int64_t*>(index.data);
    const int64_t n = static_cast<int64_t>(index.count);
    for (int64_t i = 0; i < n; ++i) {
        if (requests[i] < 0 || requests[i] >= size) {
            std::fprintf(stderr, "ig: index[%lld] is outside [0, length of table)\n", static_cast<long long>(i));
            return 1;
        }
    }
    std::vector<int64_t> tgt(static_cast<std::size_t>(n), 0);
    int64_t* result = tgt.data();
#pragma omp parallel for
    for (int64_t i = 0; i < n; ++i) {
        result[i] = values[requests[i]];
    }
    return lassi_io_write(out, "tgt", LASSI_IO_I64, index.rank, index.dims, result) == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: ig <table> <index> <tgt>\n");
        return 2;
    }
    lassi_io_array table = {};
    lassi_io_array index = {};
    int status = 1;
    if (read_i64(argv[1], 1u, &table) && read_i64(argv[2], 1u, &index)) {
        status = gather(table, index, argv[3]);
    }
    lassi_io_free(&table);
    lassi_io_free(&index);
    return status;
}
