// Written for LASSI-DF tt-par-v0 (task P17.68); a held-out item: col_scale's writer, which stores each product tile at its own place in C.
//
// Runtime args: 0 C's DRAM address; 1 this core's first tile of C; 2 how many tiles of C it stores. Compile args:
// C's TensorAccessorArgs from index 0. The products leave the compute kernel in the order the reader fetched their
// tiles, the order of C's tile numbers, so the n-th tile out of c_16 is tile first + n of C.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// One pass over this core's tiles of C, one product tile stored per pass.
void kernel_main() {
    const uint32_t c_base = get_arg_val<uint32_t>(0);
    const uint32_t first_of_c = get_arg_val<uint32_t>(1);
    const uint32_t count_of_c = get_arg_val<uint32_t>(2);

    // Product tiles wait here for this kernel.
    constexpr uint32_t kProductCb = tt::CBIndex::c_16;
    // C's pages are tiles of the same size and format as c_16's pages.
    const uint32_t product_page = get_tile_size(kProductCb);
    constexpr auto c_paging = TensorAccessorArgs<0>();
    const auto c_tiles = TensorAccessor(c_paging, c_base, product_page);

    const uint32_t past_last = first_of_c + count_of_c;
    for (uint32_t c_tile = first_of_c; c_tile < past_last; ++c_tile) {
        cb_wait_front(kProductCb, 1);
        const uint32_t product_slot = get_read_ptr(kProductCb);
        noc_async_write_page(c_tile, c_tiles, product_slot);
        noc_async_write_barrier();
        cb_pop_front(kProductCb, 1);
    }
}
