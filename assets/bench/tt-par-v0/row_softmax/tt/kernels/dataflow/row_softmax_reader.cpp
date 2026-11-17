// Written for LASSI-DF tt-par-v0 (task P17.68); a held-out item: row softmax, the reader kernel.
//
// Runs on one data-movement core of each working Tensix core. It first fills the reduce scaler tile in c_1 once,
// then moves this core's share of x from DRAM into c_0, one whole tile row (Wt tiles) per step.
//
// The scaler tile: the compute kernel's row reduces (MAX, then SUM) multiply what they reduce by this tile, and
// they read the first row of each 16 x 16 face. So every bf16 value is 0 except the 16 values that open each of
// the four faces, which are 1.0, and both reduces return the plain maximum and the plain sum.
//
// Compile args: 0 is Wt, the tiles in one tile row; from 1 on, x's TensorAccessorArgs.
// Runtime args: 0 is x's DRAM address, 1 the first tile row this core handles, 2 how many tile rows it handles.
// x's tiles lie in DRAM in row-major tile order, so tile row r holds tiles r * Wt to r * Wt + Wt - 1.
#include <cstdint>  // uint32_t

#include "api/dataflow/dataflow_api.h"  // circular buffers, runtime and compile args, TensorAccessor, NoC reads

namespace {

// A 32 x 32 tile of bf16 values is 512 four-byte words; each 16 x 16 face is 128 words, and the first row of a
// face (16 values) is its first 8 words.
constexpr uint32_t kWordsPerTile = 512;
constexpr uint32_t kWordsPerFace = 128;
constexpr uint32_t kWordsPerFaceRow = 8;
constexpr uint32_t kFaces = 4;
// Two bf16 values of 1.0 (0x3f80 each) in one word.
constexpr uint32_t kTwoOnes = 0x3f803f80u;

// Write the scaler tile into the next free slot of `cb` and hand it to the compute kernel.
void push_scaler_tile(uint32_t cb) {
    cb_reserve_back(cb, 1);
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb));
    for (uint32_t w = 0; w < kWordsPerTile; ++w) {
        words[w] = 0u;
    }
    for (uint32_t face = 0; face < kFaces; ++face) {
        for (uint32_t w = 0; w < kWordsPerFaceRow; ++w) {
            words[face * kWordsPerFace + w] = kTwoOnes;
        }
    }
    cb_push_back(cb, 1);
}

}  // namespace

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t first_tile_row = get_arg_val<uint32_t>(1);
    const uint32_t tile_row_count = get_arg_val<uint32_t>(2);

    constexpr uint32_t tiles_per_row = get_compile_time_arg_val(0);
    constexpr auto x_layout = TensorAccessorArgs<1>();

    constexpr uint32_t cb_x = tt::CBIndex::c_0;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_1;

    const uint32_t tile_bytes = get_tile_size(cb_x);
    const auto x_pages = TensorAccessor(x_layout, x_addr, tile_bytes);

    push_scaler_tile(cb_scaler);

    // One tile row per step: reserve room for all of it, start every tile's read, wait for them, and publish it.
    for (uint32_t r = first_tile_row; r < first_tile_row + tile_row_count; ++r) {
        cb_reserve_back(cb_x, tiles_per_row);
        uint32_t l1_addr = get_write_ptr(cb_x);
        for (uint32_t t = 0; t < tiles_per_row; ++t) {
            noc_async_read_page(r * tiles_per_row + t, x_pages, l1_addr);
            l1_addr += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, tiles_per_row);
    }
}
