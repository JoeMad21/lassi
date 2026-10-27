// tt-kh-skeleton-v0: an item-agnostic reader kernel skeleton for tt-metal commit 5280a9cf, a data movement kernel.
// Written from the pinned kernel headers (tt_metal/hw/inc/api at that commit, the declarations of the
// tt-metalium-5280a9cf-kh API pack) and the kernel pages of tt-metal's documentation at that commit. It solves no
// task: its placeholder code reads every tile of one input buffer into circular buffer c_0, for the host skeleton's
// reader args. Fill or extend only the blocks from a BLANK line to its END BLANK line.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // BLANK reader_args: the runtime args, read in the order the host's SetRuntimeArgs passes them (here the input
    // buffer's address and the tile count).
    uint32_t in0_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);
    // END BLANK reader_args

    // BLANK reader_accessors: one accessor per buffer read, from the compile args the host appended with
    // TensorAccessorArgs(*buffer).append_to(...): TensorAccessorArgs<0>() for the first buffer, then
    // TensorAccessorArgs<in0_args.next_compile_time_args_offset()>() for the next.
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    const uint32_t page_bytes = get_tile_size(cb_in0);
    constexpr auto in0_args = TensorAccessorArgs<0>();
    const auto in0 = TensorAccessor(in0_args, in0_addr, page_bytes);
    // END BLANK reader_accessors

    // BLANK reader_loop: per tile, reserve a page of the circular buffer, read the tile into it, wait for the read,
    // and push the page to the compute kernel.
    for (uint32_t tile = 0; tile < n_tiles; ++tile) {
        cb_reserve_back(cb_in0, 1);
        uint32_t l1_addr = get_write_ptr(cb_in0);
        noc_async_read_page(tile, in0, l1_addr);
        noc_async_read_barrier();
        cb_push_back(cb_in0, 1);
    }
    // END BLANK reader_loop
}
