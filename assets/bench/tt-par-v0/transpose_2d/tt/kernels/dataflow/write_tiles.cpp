// Written for LASSI-DF tt-par-v0 (task P17.68); a held-out item: transpose_2d's writer, which stores each turned tile at its own place in B.
//
// Runtime args: 0 B's DRAM address; 1 this core's first tile of B; 2 how many tiles of B it stores. Compile args:
// B's TensorAccessorArgs from index 0. The compute kernel hands over the tiles in the order the reader asked for
// them, which is the order of B's tile numbers, so the n-th tile out of c_16 is tile first + n of B.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// One pass over this core's tiles of B, one turned tile stored per pass.
void kernel_main() {
    const uint32_t b_base = get_arg_val<uint32_t>(0);
    const uint32_t first_of_b = get_arg_val<uint32_t>(1);
    const uint32_t count_of_b = get_arg_val<uint32_t>(2);

    // Turned tiles wait here for this kernel.
    constexpr uint32_t kTurnedCb = tt::CBIndex::c_16;
    // B's pages are tiles of the same size and format as c_16's pages.
    const uint32_t turned_page = get_tile_size(kTurnedCb);
    constexpr auto b_paging = TensorAccessorArgs<0>();
    const auto b_tiles = TensorAccessor(b_paging, b_base, turned_page);

    const uint32_t past_last = first_of_b + count_of_b;
    for (uint32_t b_tile = first_of_b; b_tile < past_last; ++b_tile) {
        cb_wait_front(kTurnedCb, 1);
        const uint32_t turned_slot = get_read_ptr(kTurnedCb);
        noc_async_write_page(b_tile, b_tiles, turned_slot);
        noc_async_write_barrier();
        cb_pop_front(kTurnedCb, 1);
    }
}
