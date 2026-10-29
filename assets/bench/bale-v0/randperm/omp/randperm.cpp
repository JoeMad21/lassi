// bale-v0 randperm, the OpenMP C++ program (task P17.43). Written here for LASSI; not upstream code.
//
// A random permutation of 0 to n - 1 by throwing n darts at a board of 2n slots, its random numbers drawn ahead as an
// input: darts holds R rounds of n target slots, R = length(darts) / n, each slot in [0, 2n). The permutation is this
// function of num_darts and darts alone:
//   1. Every slot starts empty and every dart 0 to n - 1 unplaced.
//   2. Rounds k = 0 to R - 1 are synchronous: every dart i unplaced at the start of round k aims at slot
//      darts[k * n + i]. A slot occupied at the start of the round takes no dart; an empty slot takes the
//      lowest-numbered dart aimed at it in that round. Every other dart stays unplaced and is thrown again next round.
//   3. The permutation lists the darts of the occupied slots in increasing slot order, then the darts still unplaced
//      after round R - 1 in increasing dart order, so it holds every dart exactly once.
// Here each round runs on OpenMP threads on the host CPU in two phases over the darts. In the claim phase every
// unplaced dart aimed at an empty slot claims it, and a min reduction over the claims leaves each slot's
// lowest-numbered claimant; in the place phase each dart that holds its slot's claim takes the slot. The board is only
// read in the claim phase and only written in the place phase, each slot by its one winner, so the permutation is the
// same for any number of threads and any schedule, and equals the serial program's. The listing of step 3 is serial.
// It follows the dart model of bale's src/other_serial/C/randperm.c, n darts on a board of 2n slots, each redrawn
// until it finds a free slot, then the occupied slots read in slot order (github.com/jdevinney/bale at 1b8f673;
// that file is Copyright (c) 2020, Institute for Defense Analyses, under a BSD-3-Clause-style license);
// no bale text is copied here. Deviations from bale: bale places the darts one at a time, redrawing each from its
// own generator until it finds a free slot; here the draws are an input, the throws are synchronous rounds in which
// the lowest-numbered dart wins a contested slot, and the rounds are bounded, the darts left after them listed last,
// so the permutation is a function of the inputs whatever order the threads run in.
// Usage: randperm <num_darts> <darts> <perm>
//   num_darts  the lassi_io input file: an i64 scalar, the number of darts n, at least 1
//   darts      the lassi_io input file: a one-dimensional i64 array of R * n target slots, R at least 1, each a slot
//              in [0, 2n); round k's slot for dart i is darts[k * n + i]
//   perm       the lassi_io output file: an i64 array named "perm" of shape [n], a permutation of 0 to n - 1
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "lassi_io.h"

namespace {

// The value of an empty slot; every dart is at least 0.
constexpr int64_t kEmpty = -1;

// The claim of a slot no dart has claimed: above every dart, and the start value of a min reduction over int64_t.
constexpr int64_t kNoClaim = std::numeric_limits<int64_t>::max();

// Read the lassi_io file at `path` into `array`; return false, saying why, unless it holds i64 data of `rank` dims.
bool read_i64(const char* path, uint32_t rank, lassi_io_array* array) {
    if (lassi_io_read(path, array) != 0) {
        return false;
    }
    if (array->dtype != LASSI_IO_I64 || array->rank != rank) {
        std::fprintf(stderr, "randperm: %s must hold an i64 array of rank %u\n", path, static_cast<unsigned>(rank));
        return false;
    }
    return true;
}

// Throw the n darts for `rounds` rounds at `board` (2n slots, all empty), round k's slots at darts + k * n; set
// placed[i] to 1 for each dart i that takes a slot. `claim` is scratch space of 2n values.
void throw_darts(int64_t n, int64_t rounds, const int64_t* darts, int64_t* board, uint8_t* placed, int64_t* claim) {
    const int64_t slots = 2 * n;
    for (int64_t k = 0; k < rounds; ++k) {
        const int64_t* target = darts + k * n;
#pragma omp parallel for
        for (int64_t s = 0; s < slots; ++s) {
            claim[s] = kNoClaim;
        }
        // Claim phase: each slot's claim becomes the lowest unplaced dart aimed at it while it is empty.
#pragma omp parallel for reduction(min : claim[0 : slots])
        for (int64_t i = 0; i < n; ++i) {
            if (placed[i] == 0 && board[target[i]] == kEmpty && i < claim[target[i]]) {
                claim[target[i]] = i;
            }
        }
        // Place phase: the dart that holds a slot's claim takes the slot.
#pragma omp parallel for
        for (int64_t i = 0; i < n; ++i) {
            if (placed[i] == 0 && claim[target[i]] == i) {
                board[target[i]] = i;
                placed[i] = 1;
            }
        }
    }
}

// Write into perm (n values) the darts of the occupied slots of `board` in slot order, then the unplaced darts in
// dart order.
void list_darts(int64_t n, const int64_t* board, const uint8_t* placed, int64_t* perm) {
    int64_t next = 0;
    for (int64_t s = 0; s < 2 * n; ++s) {
        if (board[s] != kEmpty) {
            perm[next++] = board[s];
        }
    }
    for (int64_t i = 0; i < n; ++i) {
        if (placed[i] == 0) {
            perm[next++] = i;
        }
    }
}

// Check the inputs, build the permutation in parallel rounds, and write it to `out`; return the exit status.
int build_permutation(const lassi_io_array& num_darts, const lassi_io_array& darts, const char* out) {
    const int64_t n = *static_cast<const int64_t*>(num_darts.data);
    if (n < 1) {
        std::fprintf(stderr, "randperm: num_darts must be at least 1\n");
        return 1;
    }
    const int64_t count = static_cast<int64_t>(darts.count);
    if (count == 0 || count % n != 0) {
        std::fprintf(stderr, "randperm: the length of darts must be a positive multiple of num_darts\n");
        return 1;
    }
    const int64_t* target = static_cast<const int64_t*>(darts.data);
    for (int64_t j = 0; j < count; ++j) {
        if (target[j] < 0 || target[j] >= 2 * n) {
            std::fprintf(stderr, "randperm: darts[%lld] is outside [0, 2 * num_darts)\n", static_cast<long long>(j));
            return 1;
        }
    }
    std::vector<int64_t> board(static_cast<std::size_t>(2 * n), kEmpty);
    std::vector<uint8_t> placed(static_cast<std::size_t>(n), 0);
    std::vector<int64_t> claim(static_cast<std::size_t>(2 * n), kNoClaim);
    std::vector<int64_t> perm(static_cast<std::size_t>(n), 0);
    throw_darts(n, count / n, target, board.data(), placed.data(), claim.data());
    list_darts(n, board.data(), placed.data(), perm.data());
    const uint64_t dims[1] = {static_cast<uint64_t>(n)};
    return lassi_io_write(out, "perm", LASSI_IO_I64, 1u, dims, perm.data()) == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: randperm <num_darts> <darts> <perm>\n");
        return 2;
    }
    lassi_io_array num_darts = {};
    lassi_io_array darts = {};
    int status = 1;
    if (read_i64(argv[1], 0u, &num_darts) && read_i64(argv[2], 1u, &darts)) {
        status = build_permutation(num_darts, darts, argv[3]);
    }
    lassi_io_free(&num_darts);
    lassi_io_free(&darts);
    return status;
}
