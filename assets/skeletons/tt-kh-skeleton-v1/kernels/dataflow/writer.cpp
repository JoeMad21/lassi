// tt-kh-skeleton-v1: an item-agnostic writer kernel skeleton for tt-metal commit 5280a9cf, a data movement kernel.
// Written from tt-kh-skeleton-v0's writer and the pinned kernel headers (tt_metal/hw/inc/api at that commit, the
// declarations of the tt-metalium-5280a9cf-kh API pack). It solves no task: its placeholder code writes this core's
// tiles of circular buffer c_16 to one output buffer, from its first tile, for the host skeleton's writer args on
// each core. Fill or extend only the blocks from a BLANK line to its END BLANK line.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // BLANK writer_args: the runtime args, read in the order the host's SetRuntimeArgs passes them (here the output
    // buffer's address, then this core's first tile and its tile count).
    uint32_t out0_addr = get_arg_val<uint32_t>(0);
    uint32_t first_tile = get_arg_val<uint32_t>(1);
    uint32_t n_tiles = get_arg_val<uint32_t>(2);
    // END BLANK writer_args

    // BLANK writer_accessors: one accessor per buffer written, from the compile args the host appended with
    // TensorAccessorArgs(*buffer).append_to(...), as in the reader.
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    const uint32_t page_bytes = get_tile_size(cb_out0);
    constexpr auto out0_args = TensorAccessorArgs<0>();
    const auto out0 = TensorAccessor(out0_args, out0_addr, page_bytes);
    // END BLANK writer_accessors

    // BLANK writer_loop: per tile of this core's share, from its first tile, wait for a page the compute kernel
    // pushed, write it to the buffer, wait for the write, and pop the page.
    for (uint32_t tile = first_tile; tile < first_tile + n_tiles; ++tile) {
        cb_wait_front(cb_out0, 1);
        uint32_t l1_addr = get_read_ptr(cb_out0);
        noc_async_write_page(tile, out0, l1_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_out0, 1);
    }
    // END BLANK writer_loop
}
