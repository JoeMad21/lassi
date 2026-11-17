// Written for LASSI-DF tt-par-v0 (task P17.68); a held-out item: transpose_2d's compute kernel, which turns each tile over its main diagonal.
//
// Runtime args: 0 how many tiles this core turns. Each tile arrives in c_0 from the reader and leaves through c_16
// for the writer, in the order it came. For a Float16_b tile, transpose_wh_tile has the unpacker transpose the four
// 16 x 16 faces as blocks and each face inside itself, moves the result into the destination register unchanged,
// and the packer writes it out; no value is computed, so every bf16 bit pattern survives.
#include <cstdint>

#include "api/compute/transpose_wh.h"

// One pass over this core's tiles: wait, turn, hand on.
void kernel_main() {
    const uint32_t turns = get_arg_val<uint32_t>(0);

    constexpr uint32_t kMirrorCb = tt::CBIndex::c_0;
    constexpr uint32_t kTurnedCb = tt::CBIndex::c_16;
    // The one destination register slot this kernel uses.
    constexpr uint32_t kTurnSlot = 0;

    // Configure unpack, math, and pack once for a transpose read from c_0 and packed into c_16.
    transpose_wh_init(kMirrorCb, kTurnedCb);

    for (uint32_t turned = 0; turned < turns; ++turned) {
        cb_wait_front(kMirrorCb, 1);
        tile_regs_acquire();
        transpose_wh_tile(kMirrorCb, 0, kTurnSlot);
        tile_regs_commit();
        cb_pop_front(kMirrorCb, 1);

        cb_reserve_back(kTurnedCb, 1);
        tile_regs_wait();
        pack_tile(kTurnSlot, kTurnedCb);
        tile_regs_release();
        cb_push_back(kTurnedCb, 1);
    }
}
