// Written for LASSI-DF tt-par-v0 (task P17.68); a held-out item: transpose_2d's reader, which fetches for each tile of B the tile of A that mirrors it.
//
// Runtime args: 0 A's DRAM address; 1 this core's first tile of B; 2 how many tiles of B it makes; 3 A's tile rows
// (B's tile columns); 4 A's tile columns (B's tile rows). Compile args: A's TensorAccessorArgs from index 0.
//
// B's tiles are numbered row by row, so tile number t of B sits at tile row t / (A's tile rows) and tile column
// t % (A's tile rows). Its source is the tile of A with those two coordinates swapped: A's tile row t % (A's tile
// rows) and tile column t / (A's tile rows), whose number in A, also counted row by row, is
// (t % A's tile rows) * (A's tile columns) + t / (A's tile rows). Each source tile goes into c_0 as it is; the
// compute kernel does the turning inside the tile.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// One pass over this core's tiles of B, one mirrored tile of A fetched per pass.
void kernel_main() {
    const uint32_t a_base = get_arg_val<uint32_t>(0);
    const uint32_t first_of_b = get_arg_val<uint32_t>(1);
    const uint32_t count_of_b = get_arg_val<uint32_t>(2);
    const uint32_t a_tile_rows = get_arg_val<uint32_t>(3);
    const uint32_t a_tile_cols = get_arg_val<uint32_t>(4);

    // Tiles of A wait here for the compute kernel.
    constexpr uint32_t kMirrorCb = tt::CBIndex::c_0;
    // A's pages are tiles of the same size and format as c_0's pages.
    const uint32_t mirror_page = get_tile_size(kMirrorCb);
    constexpr auto a_paging = TensorAccessorArgs<0>();
    const auto a_tiles = TensorAccessor(a_paging, a_base, mirror_page);

    const uint32_t past_last = first_of_b + count_of_b;
    for (uint32_t b_tile = first_of_b; b_tile < past_last; ++b_tile) {
        const uint32_t b_tile_row = b_tile / a_tile_rows;  // one of A's tile columns
        const uint32_t b_tile_col = b_tile % a_tile_rows;  // one of A's tile rows
        const uint32_t a_tile = b_tile_col * a_tile_cols + b_tile_row;
        cb_reserve_back(kMirrorCb, 1);
        const uint32_t mirror_slot = get_write_ptr(kMirrorCb);
        noc_async_read_page(a_tile, a_tiles, mirror_slot);
        noc_async_read_barrier();
        cb_push_back(kMirrorCb, 1);
    }
}
