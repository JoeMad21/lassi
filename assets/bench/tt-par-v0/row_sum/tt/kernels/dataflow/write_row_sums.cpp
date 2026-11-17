// Written for LASSI-DF tt-par-v0 (task P17.68); a held-out item: row_sum's writer, one result tile per owned tile row.
//
// Runs on every working core. Runtime args, in order:
//   0  the DRAM address of the result buffer (one bf16 tile per page, one page per tile row of a)
//   1  the first tile row this core owns
//   2  how many consecutive tile rows it owns
// Compile args: the result buffer's TensorAccessorArgs, from index 0.
// Each tile the compute kernel packs into c_16 holds one tile row's 32 sums in its first column; it is stored at
// the page whose index is that tile row, so the pages end up in tile-row order whichever core wrote them.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// Stores this core's result tiles in DRAM, one page per owned tile row.
void kernel_main() {
    const uint32_t sums_dram_addr = get_arg_val<uint32_t>(0);
    const uint32_t first_tile_row = get_arg_val<uint32_t>(1);
    const uint32_t owned_tile_rows = get_arg_val<uint32_t>(2);

    constexpr uint32_t row_sums_cb = tt::CBIndex::c_16;

    // The result pages are tiles of c_16's format, so c_16's page size is theirs.
    constexpr auto sums_layout = TensorAccessorArgs<0>();
    const auto sum_pages = TensorAccessor(sums_layout, sums_dram_addr, get_tile_size(row_sums_cb));

    for (uint32_t tile_row = first_tile_row; tile_row < first_tile_row + owned_tile_rows; ++tile_row) {
        cb_wait_front(row_sums_cb, 1);
        noc_async_write_tile(tile_row, sum_pages, get_read_ptr(row_sums_cb));
        noc_async_write_barrier();
        cb_pop_front(row_sums_cb, 1);
    }
}
