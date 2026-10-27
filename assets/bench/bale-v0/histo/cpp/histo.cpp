// bale-v0 histo, the serial C++ program (task P17.40). Written here for LASSI; not upstream code.
//
// A histogram of updates: counts[t] is the number of updates i with index[i] == t, built by adding 1 to
// counts[index[i]] for every update, in order. It follows the algorithm of the generic serial histogram in bale's
// src/other_serial/C/histo.c (github.com/jdevinney/bale at 1b8f673; that file is
// Copyright (c) 2020, Institute for Defense Analyses, under a BSD-3-Clause-style license); no bale text is copied
// here. Single-threaded.
// Usage: histo <table_size> <index> <counts>
//   table_size  the lassi_io input file: an i64 scalar, the number of bins T, at least 1
//   index       the lassi_io input file: a one-dimensional i64 array of updates, each a bin in [0, T)
//   counts      the lassi_io output file: an i64 array named "counts" of shape [T]
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
        std::fprintf(stderr, "histo: %s must hold an i64 array of rank %u\n", path, static_cast<unsigned>(rank));
        return false;
    }
    return true;
}

// Count the updates of each bin and write the counts to `out`; return the exit status.
int histogram(const lassi_io_array& table_size, const lassi_io_array& index, const char* out) {
    const int64_t bins = *static_cast<const int64_t*>(table_size.data);
    if (bins < 1) {
        std::fprintf(stderr, "histo: table_size must be at least 1\n");
        return 1;
    }
    const int64_t* updates = static_cast<const int64_t*>(index.data);
    const int64_t n = static_cast<int64_t>(index.count);
    for (int64_t i = 0; i < n; ++i) {
        if (updates[i] < 0 || updates[i] >= bins) {
            std::fprintf(stderr, "histo: index[%lld] is outside [0, table_size)\n", static_cast<long long>(i));
            return 1;
        }
    }
    std::vector<int64_t> counts(static_cast<std::size_t>(bins), 0);
    int64_t* bin = counts.data();
    for (int64_t i = 0; i < n; ++i) {
        bin[updates[i]] += 1;
    }
    const uint64_t dims[1] = {static_cast<uint64_t>(bins)};
    return lassi_io_write(out, "counts", LASSI_IO_I64, 1u, dims, bin) == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: histo <table_size> <index> <counts>\n");
        return 2;
    }
    lassi_io_array table_size = {};
    lassi_io_array index = {};
    int status = 1;
    if (read_i64(argv[1], 0u, &table_size) && read_i64(argv[2], 1u, &index)) {
        status = histogram(table_size, index, argv[3]);
    }
    lassi_io_free(&table_size);
    lassi_io_free(&index);
    return status;
}
