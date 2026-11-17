// Written for LASSI-DF tt-par-v0 (task P17.68); a held-out item: col_max's reader, a scaler tile and then each owned tile column.
//
// Runs on every working core. Runtime args, in order:
//   0  the DRAM address of a, stored as tiles (one 32 x 32 bf16 tile per page, tiles in row-major order)
//   1  the first tile column this core owns
//   2  how many consecutive tile columns it owns
//   3  how many tiles make one tile column (the row count of a over 32)
//   4  how many tiles make one tile row (the column count of a over 32), the page stride down a tile column
// Compile args: a's TensorAccessorArgs, from index 0.
//
// First it fills one page of c_1 with the reduce scaler: bf16 1.0 in the first row of each of the tile's four
// 16 x 16 faces and zero everywhere else. The maximum reduce scales each element by its scaler's power of two
// before comparing, and 1.0 leaves every element as it is. Then, tile column by tile column, it hands the compute
// kernel the column's tiles through c_0, one tile at a time, from the top of the column down.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// The constants and the scaler helper below belong to this kernel alone.
namespace {

// Circular buffer slots shared with the compute kernel.
constexpr uint32_t kColumnTilesCb = tt::CBIndex::c_0;
constexpr uint32_t kUnitScaleCb = tt::CBIndex::c_1;

// Two bf16 1.0 values (bit pattern 0x3f80) side by side in one 32-bit word.
constexpr uint32_t kPairOfOnes = 0x3f803f80u;
// In L1 a bf16 tile is its four faces one after another; a face spans 128 words and one face row 8 words.
constexpr uint32_t kFaceSpanWords = 128u;
constexpr uint32_t kFaceRowSpanWords = 8u;
constexpr uint32_t kTileFaceCount = 4u;

// Write the scaler tile described above into the next free page of scale_cb and hand it on.
void push_unit_scale_tile(uint32_t scale_cb) {
    cb_reserve_back(scale_cb, 1);
    volatile tt_l1_ptr uint32_t* scale_words =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(scale_cb));
    const uint32_t scale_word_count = get_tile_size(scale_cb) / sizeof(uint32_t);
    for (uint32_t w = 0; w < scale_word_count; ++w) {
        scale_words[w] = 0u;
    }
    for (uint32_t face = 0; face < kTileFaceCount; ++face) {
        for (uint32_t w = 0; w < kFaceRowSpanWords; ++w) {
            scale_words[face * kFaceSpanWords + w] = kPairOfOnes;
        }
    }
    cb_push_back(scale_cb, 1);
}

}  // namespace

void kernel_main() {
    const uint32_t a_base_addr = get_arg_val<uint32_t>(0);
    const uint32_t first_tile_column = get_arg_val<uint32_t>(1);
    const uint32_t owned_tile_columns = get_arg_val<uint32_t>(2);
    const uint32_t tiles_down = get_arg_val<uint32_t>(3);
    const uint32_t tiles_per_tile_row = get_arg_val<uint32_t>(4);

    push_unit_scale_tile(kUnitScaleCb);

    // Pages of a are whole tiles of the same format as c_0, so c_0's page size is a's page size.
    constexpr auto a_page_layout = TensorAccessorArgs<0>();
    const auto a_pages = TensorAccessor(a_page_layout, a_base_addr, get_tile_size(kColumnTilesCb));

    for (uint32_t tile_column = first_tile_column; tile_column < first_tile_column + owned_tile_columns;
         ++tile_column) {
        // Going down a tile column skips one whole tile row of pages per step.
        for (uint32_t down = 0; down < tiles_down; ++down) {
            const uint32_t page = down * tiles_per_tile_row + tile_column;
            cb_reserve_back(kColumnTilesCb, 1);
            noc_async_read_tile(page, a_pages, get_write_ptr(kColumnTilesCb));
            noc_async_read_barrier();
            cb_push_back(kColumnTilesCb, 1);
        }
    }
}
