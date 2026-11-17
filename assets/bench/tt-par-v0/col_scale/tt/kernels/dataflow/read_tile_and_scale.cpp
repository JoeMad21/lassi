// Written for LASSI-DF tt-par-v0 (task P17.68); a held-out item: col_scale's reader, which fetches each tile of A with the scale tile of its tile column.
//
// Runtime args: 0 A's DRAM address; 1 the scale tiles' DRAM address; 2 this core's first tile of C; 3 how many tiles
// of C it makes; 4 the tile columns of A. Compile args: A's TensorAccessorArgs from index 0, then the scale tiles'.
//
// C's tiles, like A's, are numbered row by row, so tile t of C is made from tile t of A and lies in tile column
// t % (tile columns), whose 32 values of s fill the first row of scale tile number t % (tile columns). The pair goes
// into c_0 (A's tile) and c_1 (the scale tile) together, one pair per tile of C.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// One pass over this core's tiles of C, one tile of A and one scale tile fetched per pass.
void kernel_main() {
    const uint32_t a_base = get_arg_val<uint32_t>(0);
    const uint32_t scale_base = get_arg_val<uint32_t>(1);
    const uint32_t first_of_c = get_arg_val<uint32_t>(2);
    const uint32_t count_of_c = get_arg_val<uint32_t>(3);
    const uint32_t tile_cols = get_arg_val<uint32_t>(4);

    // A's tiles and the scale tiles wait here for the compute kernel.
    constexpr uint32_t kMatrixCb = tt::CBIndex::c_0;
    constexpr uint32_t kScaleCb = tt::CBIndex::c_1;
    // Both DRAM buffers have pages of one tile, the size and format of their circular buffers' pages.
    const uint32_t matrix_page = get_tile_size(kMatrixCb);
    const uint32_t scale_page = get_tile_size(kScaleCb);
    constexpr auto a_paging = TensorAccessorArgs<0>();
    constexpr auto scale_paging = TensorAccessorArgs<a_paging.next_compile_time_args_offset()>();
    const auto a_tiles = TensorAccessor(a_paging, a_base, matrix_page);
    const auto scale_tiles = TensorAccessor(scale_paging, scale_base, scale_page);

    const uint32_t past_last = first_of_c + count_of_c;
    for (uint32_t c_tile = first_of_c; c_tile < past_last; ++c_tile) {
        const uint32_t column_of_tiles = c_tile % tile_cols;
        cb_reserve_back(kMatrixCb, 1);
        cb_reserve_back(kScaleCb, 1);
        const uint32_t matrix_slot = get_write_ptr(kMatrixCb);
        const uint32_t scale_slot = get_write_ptr(kScaleCb);
        noc_async_read_page(c_tile, a_tiles, matrix_slot);
        noc_async_read_page(column_of_tiles, scale_tiles, scale_slot);
        noc_async_read_barrier();
        cb_push_back(kMatrixCb, 1);
        cb_push_back(kScaleCb, 1);
    }
}
