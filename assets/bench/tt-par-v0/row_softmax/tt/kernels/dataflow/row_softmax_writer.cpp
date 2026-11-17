// Written for LASSI-DF tt-par-v0 (task P17.68); a held-out item: row softmax, the writer kernel.
//
// Runs on the other data-movement core of each working Tensix core. It waits until the compute kernel has put a
// whole tile row of results (Wt tiles) in c_16, stores those tiles to y in DRAM, and frees them, once per tile row
// of this core's share.
//
// Compile args: 0 is Wt, the tiles in one tile row; from 1 on, y's TensorAccessorArgs.
// Runtime args: 0 is y's DRAM address, 1 the first tile row this core handles, 2 how many tile rows it handles.
// y's tiles lie in DRAM in row-major tile order, as x's do, so tile row r goes to tiles r * Wt to r * Wt + Wt - 1.
#include <cstdint>  // uint32_t

#include "api/dataflow/dataflow_api.h"  // circular buffers, runtime and compile args, TensorAccessor, NoC writes

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t first_tile_row = get_arg_val<uint32_t>(1);
    const uint32_t tile_row_count = get_arg_val<uint32_t>(2);

    constexpr uint32_t tiles_per_row = get_compile_time_arg_val(0);
    constexpr auto y_layout = TensorAccessorArgs<1>();

    constexpr uint32_t cb_y = tt::CBIndex::c_16;

    const uint32_t tile_bytes = get_tile_size(cb_y);
    const auto y_pages = TensorAccessor(y_layout, y_addr, tile_bytes);

    for (uint32_t r = first_tile_row; r < first_tile_row + tile_row_count; ++r) {
        cb_wait_front(cb_y, tiles_per_row);
        uint32_t l1_addr = get_read_ptr(cb_y);
        for (uint32_t t = 0; t < tiles_per_row; ++t) {
            noc_async_write_page(r * tiles_per_row + t, y_pages, l1_addr);
            l1_addr += tile_bytes;
        }
        // Every store must land before the tiles are freed for the compute kernel to overwrite.
        noc_async_write_barrier();
        cb_pop_front(cb_y, tiles_per_row);
    }
}
