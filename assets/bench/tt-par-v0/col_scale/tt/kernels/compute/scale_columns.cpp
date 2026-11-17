// Written for LASSI-DF tt-par-v0 (task P17.68); a held-out item: col_scale's compute kernel, which multiplies each tile of A by the first row of its scale tile, broadcast down the rows.
//
// Runtime args: 0 how many tiles of C this core makes. A's tile arrives in c_0 and its scale tile in c_1, one of each
// per tile of C; the product leaves through c_16 in the order the pairs came. mul_tiles_bcast_rows reads row 0 of
// the c_1 tile only and multiplies every row of the c_0 tile by it, element by element, so element (r, k) of the
// product is A's (r, k) times row 0's k-th value; row 0 of scale tile j holds s[32j] to s[32j + 31].
// The multiply runs at the math fidelity the host program sets (HiFi4: all four fidelity phases, each adding its
// partial product into the 16-bit destination register).
#include <cstdint>

#include "api/compute/bcast.h"

// One pass over this core's tiles of C: wait for a pair, multiply, hand the product on.
void kernel_main() {
    const uint32_t products = get_arg_val<uint32_t>(0);

    constexpr uint32_t kMatrixCb = tt::CBIndex::c_0;
    constexpr uint32_t kScaleCb = tt::CBIndex::c_1;
    constexpr uint32_t kProductCb = tt::CBIndex::c_16;
    // The one destination register slot this kernel uses.
    constexpr uint32_t kProductSlot = 0;

    // Configure unpack, math, and pack once for a multiply of c_0 by the row broadcast of c_1, packed into c_16.
    init_bcast<ELWMUL, BroadcastType::ROW>(kMatrixCb, kScaleCb, kProductCb);

    for (uint32_t made = 0; made < products; ++made) {
        cb_wait_front(kMatrixCb, 1);
        cb_wait_front(kScaleCb, 1);
        tile_regs_acquire();
        mul_tiles_bcast_rows(kMatrixCb, kScaleCb, 0, 0, kProductSlot);
        tile_regs_commit();
        cb_pop_front(kMatrixCb, 1);
        cb_pop_front(kScaleCb, 1);

        cb_reserve_back(kProductCb, 1);
        tile_regs_wait();
        pack_tile(kProductSlot, kProductCb);
        tile_regs_release();
        cb_push_back(kProductCb, 1);
    }
}
