// Written for LASSI-DF tt-par-v0 (task P17.68); a held-out item: row softmax, the compute kernel.
//
// For each tile row of this core's share (32 matrix rows, Wt tiles wide), four passes over tiles kept in L1:
//   1. row maxima: a row reduce with MAX over the Wt input tiles in c_0, accumulated in one dest tile, leaves each
//      row's largest value in column 0 of that tile, which goes to c_2;
//   2. shifted exponentials: for each input tile, the matrix unit subtracts c_2's column 0 from every column (a
//      column broadcast), the vector unit takes exp of the difference in place, and the tile goes to c_3; c_3 then
//      holds the whole tile row of exp(x - max);
//   3. reciprocal sums: a row reduce with SUM over the Wt tiles of c_3, accumulated in one dest tile, leaves each
//      row's sum in column 0; the vector unit replaces it by its reciprocal, and the tile goes to c_4;
//   4. outputs: each tile of c_3 times c_4's column 0 (a column broadcast) goes to c_16 for the writer.
// The row reduces read the scaler tile in c_1 (the reader kernel's 1.0 pattern), so they give the plain maximum and
// the plain sum. Both reductions use only column 0 of their result tiles, and both broadcasts read only column 0.
// The host sets math fidelity HiFi4, so the reduce products and the final multiply keep the bf16 operands' full
// mantissas, and leaves dest at 16 bits; exp and the reciprocal are their non-approximate forms.
//
// Compile args: 0 is Wt, the tiles in one tile row. Runtime args: 0 is how many tile rows this core handles.
#include <cstdint>  // uint32_t

// reduce.h's templates name these as their defaults; every reduce call below names its own type and dimension.
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "api/compute/eltwise_binary.h"        // binary_op_init_common
#include "api/compute/bcast.h"                 // the column-broadcast subtract and multiply
#include "api/compute/eltwise_unary/exp.h"     // exp_tile
#include "api/compute/eltwise_unary/recip.h"   // recip_tile
#include "api/compute/reduce.h"                // the row reduces

namespace {

constexpr uint32_t kCbX = tt::CBIndex::c_0;        // input tiles, a whole tile row at a time
constexpr uint32_t kCbScaler = tt::CBIndex::c_1;   // the reduce scaler tile, filled once by the reader
constexpr uint32_t kCbRowMax = tt::CBIndex::c_2;   // per-row maxima in column 0
constexpr uint32_t kCbExp = tt::CBIndex::c_3;      // exp(x - max) for the whole tile row
constexpr uint32_t kCbInvSum = tt::CBIndex::c_4;   // per-row 1 / sum in column 0
constexpr uint32_t kCbY = tt::CBIndex::c_16;       // output tiles for the writer

// Every pass works in dest tile 0.
constexpr uint32_t kDstTile = 0;

// Pass 1: the maxima of the tile row's 32 rows, into one tile pushed to kCbRowMax.
void find_row_maxima(uint32_t wt) {
    cb_wait_front(kCbX, wt);
    tile_regs_acquire();
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(kCbX, kCbScaler, kCbRowMax);
    for (uint32_t t = 0; t < wt; ++t) {
        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(kCbX, kCbScaler, t, 0, kDstTile);
    }
    reduce_uninit();
    cb_reserve_back(kCbRowMax, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(kDstTile, kCbRowMax);
    tile_regs_release();
    cb_push_back(kCbRowMax, 1);
}

// Pass 2: exp(x - row max) for each tile of the row, pushed to kCbExp one tile at a time; frees the input tiles.
void exponentiate_shifted(uint32_t wt) {
    cb_wait_front(kCbRowMax, 1);
    exp_tile_init<false>();
    sub_bcast_cols_init_short(kCbX, kCbRowMax);
    for (uint32_t t = 0; t < wt; ++t) {
        cb_reserve_back(kCbExp, 1);
        tile_regs_acquire();
        sub_tiles_bcast_cols(kCbX, kCbRowMax, t, 0, kDstTile);
        exp_tile<false>(kDstTile);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(kDstTile, kCbExp);
        tile_regs_release();
        cb_push_back(kCbExp, 1);
    }
    cb_pop_front(kCbRowMax, 1);
    cb_pop_front(kCbX, wt);
}

// Pass 3: 1 / (the sum of each row of kCbExp), into one tile pushed to kCbInvSum. kCbExp stays for pass 4.
void invert_row_sums(uint32_t wt) {
    cb_wait_front(kCbExp, wt);
    cb_reserve_back(kCbInvSum, 1);
    tile_regs_acquire();
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(kCbExp, kCbScaler, kCbInvSum);
    for (uint32_t t = 0; t < wt; ++t) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(kCbExp, kCbScaler, t, 0, kDstTile);
    }
    reduce_uninit();
    recip_tile_init();
    recip_tile(kDstTile);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(kDstTile, kCbInvSum);
    tile_regs_release();
    cb_push_back(kCbInvSum, 1);
}

// Pass 4: each kCbExp tile times its rows' 1 / sum, pushed to kCbY one tile at a time; frees kCbExp and kCbInvSum.
void normalize_rows(uint32_t wt) {
    cb_wait_front(kCbInvSum, 1);
    mul_bcast_cols_init_short(kCbExp, kCbInvSum);
    for (uint32_t t = 0; t < wt; ++t) {
        cb_reserve_back(kCbY, 1);
        tile_regs_acquire();
        mul_tiles_bcast_cols(kCbExp, kCbInvSum, t, 0, kDstTile);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(kDstTile, kCbY);
        tile_regs_release();
        cb_push_back(kCbY, 1);
    }
    cb_pop_front(kCbInvSum, 1);
    cb_pop_front(kCbExp, wt);
}

}  // namespace

void kernel_main() {
    const uint32_t tile_row_count = get_arg_val<uint32_t>(0);
    constexpr uint32_t tiles_per_row = get_compile_time_arg_val(0);

    // One-time setup of the unpacker, matrix unit, and packer; every circular buffer here is Float16_b, so no
    // later pass needs a data-format change, only its own operation init.
    binary_op_init_common(kCbX, kCbScaler, kCbRowMax);
    cb_wait_front(kCbScaler, 1);

    for (uint32_t r = 0; r < tile_row_count; ++r) {
        find_row_maxima(tiles_per_row);
        exponentiate_shifted(tiles_per_row);
        invert_row_sums(tiles_per_row);
        normalize_rows(tiles_per_row);
    }
    cb_pop_front(kCbScaler, 1);
}
