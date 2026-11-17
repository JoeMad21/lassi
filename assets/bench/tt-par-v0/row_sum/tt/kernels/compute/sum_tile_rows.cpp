// Written for LASSI-DF tt-par-v0 (task P17.68); a held-out item: row_sum's compute kernel, one reduce per owned tile row.
//
// Runs on every working core. Runtime args, in order:
//   0  how many tile rows this core owns
//   1  how many tiles make one tile row
// For each owned tile row it takes the row's tiles from c_0 in order and reduces each one along its rows into the
// same Dst tile (reduce_tile, SUM over REDUCE_ROW, with the scaler tile from c_1), so Dst accumulates the sum of
// every element of each of the 32 rows. The finished tile, whose first column holds the 32 row sums (the reduce's
// pack mask zeroes the rest), goes to c_16 for the writer. The scaler tile stays at the front of c_1 throughout.
// The sums accumulate in a 16-bit Dst: the host program leaves fp32_dest_acc_en off.
#include <cstdint>

// reduce.h takes its default reduce kind and direction from these two names, so they come before it; the calls
// below also name both explicitly.
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

// Through its own includes, reduce.h also brings the circular buffer, Dst register, and pack calls used below.
#include "api/compute/reduce.h"

void kernel_main() {
    const uint32_t owned_tile_rows = get_arg_val<uint32_t>(0);
    const uint32_t tiles_across = get_arg_val<uint32_t>(1);

    constexpr uint32_t row_tiles_cb = tt::CBIndex::c_0;
    constexpr uint32_t scaler_cb = tt::CBIndex::c_1;
    constexpr uint32_t row_sums_cb = tt::CBIndex::c_16;
    // Each reduce reads the front tile of c_0 and of c_1 (tile index 0 of each) and accumulates in Dst tile 0.
    constexpr uint32_t sum_dst = 0;

    // Configure unpack (a's tiles to SrcA, the scaler to SrcB), math, and pack for these three buffers, then
    // set the reduce up for them.
    compute_kernel_hw_startup(row_tiles_cb, scaler_cb, row_sums_cb);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(row_tiles_cb, scaler_cb, row_sums_cb);

    // The reader pushes the scaler once, before any tile of a.
    cb_wait_front(scaler_cb, 1);

    for (uint32_t done = 0; done < owned_tile_rows; ++done) {
        tile_regs_acquire();
        for (uint32_t across = 0; across < tiles_across; ++across) {
            cb_wait_front(row_tiles_cb, 1);
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(row_tiles_cb, scaler_cb, 0, 0, sum_dst);
            cb_pop_front(row_tiles_cb, 1);
        }
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(row_sums_cb, 1);
        pack_tile(sum_dst, row_sums_cb);
        cb_push_back(row_sums_cb, 1);
        tile_regs_release();
    }
}
