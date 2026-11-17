// Written for LASSI-DF tt-par-v0 (task P17.68); a held-out item: col_max's compute kernel, one reduce per owned tile column.
//
// Runs on every working core. Runtime args, in order:
//   0  how many tile columns this core owns
//   1  how many tiles make one tile column
// For each owned tile column it takes the column's tiles from c_0, top to bottom, and reduces each one along its
// columns into the same Dst tile (reduce_tile, MAX over REDUCE_COL, with the scaler tile of ones from c_1), so Dst
// keeps the largest element of each of the 32 columns seen so far. The finished tile, whose first row holds the 32
// column maxima (the reduce's pack mask zeroes the rest), goes to c_16 for the writer. The scaler tile stays at the
// front of c_1 throughout.
#include <cstdint>

// reduce.h takes its default reduce kind and direction from these two names, so they come before it; the calls
// below also name both explicitly.
#define REDUCE_OP PoolType::MAX
#define REDUCE_DIM ReduceDim::REDUCE_COL

// Through its own includes, reduce.h also brings the circular buffer, Dst register, and pack calls used below.
#include "api/compute/reduce.h"

void kernel_main() {
    const uint32_t owned_tile_columns = get_arg_val<uint32_t>(0);
    const uint32_t tiles_down = get_arg_val<uint32_t>(1);

    constexpr uint32_t column_tiles_cb = tt::CBIndex::c_0;
    constexpr uint32_t unit_scale_cb = tt::CBIndex::c_1;
    constexpr uint32_t column_max_cb = tt::CBIndex::c_16;
    // Each reduce reads the front tile of c_0 and of c_1 (tile index 0 of each) and keeps its maxima in Dst tile 0.
    constexpr uint32_t max_dst = 0;

    // Configure unpack (a's tiles to SrcA, the scaler to SrcB), math, and pack for these three buffers, then
    // set the reduce up for them.
    compute_kernel_hw_startup(column_tiles_cb, unit_scale_cb, column_max_cb);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_COL>(column_tiles_cb, unit_scale_cb, column_max_cb);

    // The reader pushes the scaler once, before any tile of a.
    cb_wait_front(unit_scale_cb, 1);

    for (uint32_t done = 0; done < owned_tile_columns; ++done) {
        tile_regs_acquire();
        for (uint32_t down = 0; down < tiles_down; ++down) {
            cb_wait_front(column_tiles_cb, 1);
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_COL>(column_tiles_cb, unit_scale_cb, 0, 0, max_dst);
            cb_pop_front(column_tiles_cb, 1);
        }
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(column_max_cb, 1);
        pack_tile(max_dst, column_max_cb);
        cb_push_back(column_max_cb, 1);
        tile_regs_release();
    }
}
