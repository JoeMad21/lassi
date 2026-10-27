// tt-kh-skeleton-v0: an item-agnostic compute kernel skeleton for tt-metal commit 5280a9cf.
// Written from the pinned kernel headers (tt_metal/hw/inc/api at that commit, the declarations of the
// tt-metalium-5280a9cf-kh API pack) and the compute kernel pages of tt-metal's documentation at that commit. It
// solves no task: its placeholder code copies each tile of circular buffer c_0 through dst register 0 into c_16,
// for the host skeleton's compute args. Fill or extend only the blocks from a BLANK line to its END BLANK line.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

void kernel_main() {
    // BLANK compute_args: the compile args, read in the order the host's ComputeConfig compile_args gives them (here
    // the tile count); runtime args come from get_arg_val<uint32_t>(i).
    constexpr uint32_t n_tiles = get_compile_time_arg_val(0);
    // END BLANK compute_args

    // BLANK compute_init: the circular buffers this kernel reads and writes, then each operation's init call before
    // the loop (an SFPU function's own init, from its header under api/compute/, comes after these).
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    init_sfpu(cb_in0, cb_out0);
    copy_tile_init(cb_in0);
    // END BLANK compute_init

    for (uint32_t tile = 0; tile < n_tiles; ++tile) {
        cb_wait_front(cb_in0, 1);
        tile_regs_acquire();
        // BLANK compute_tile: compute one output tile into dst register 0 (here a copy of the input tile; an SFPU
        // function on dst 0 follows the copy).
        copy_tile(cb_in0, 0, 0);
        // END BLANK compute_tile
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out0, 1);
        pack_tile(0, cb_out0);
        cb_push_back(cb_out0, 1);
        cb_pop_front(cb_in0, 1);
        tile_regs_release();
    }
}
