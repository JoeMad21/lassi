// Written for LASSI-DF tt-par-v0 (task P17.68); a held-out item: col_max's writer, one result tile per owned tile column.
//
// Runs on every working core. Runtime args, in order:
//   0  the DRAM address of the result buffer (one bf16 tile per page, one page per tile column of a)
//   1  the first tile column this core owns
//   2  how many consecutive tile columns it owns
// Compile args: the result buffer's TensorAccessorArgs, from index 0.
// Each tile the compute kernel packs into c_16 holds one tile column's 32 maxima in its first row; it is stored at
// the page whose index is that tile column, so the pages end up in tile-column order whichever core wrote them.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// Stores this core's result tiles in DRAM, one page per owned tile column.
void kernel_main() {
    const uint32_t maxima_dram_addr = get_arg_val<uint32_t>(0);
    const uint32_t first_tile_column = get_arg_val<uint32_t>(1);
    const uint32_t owned_tile_columns = get_arg_val<uint32_t>(2);

    constexpr uint32_t column_max_cb = tt::CBIndex::c_16;

    // The result pages are tiles of c_16's format, so c_16's page size is theirs.
    constexpr auto maxima_layout = TensorAccessorArgs<0>();
    const auto maxima_pages = TensorAccessor(maxima_layout, maxima_dram_addr, get_tile_size(column_max_cb));

    for (uint32_t tile_column = first_tile_column; tile_column < first_tile_column + owned_tile_columns;
         ++tile_column) {
        cb_wait_front(column_max_cb, 1);
        noc_async_write_tile(tile_column, maxima_pages, get_read_ptr(column_max_cb));
        noc_async_write_barrier();
        cb_pop_front(column_max_cb, 1);
    }
}
