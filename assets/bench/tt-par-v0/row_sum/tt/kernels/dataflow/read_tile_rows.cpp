// Written for LASSI-DF tt-par-v0 (task P17.68); a held-out item: row_sum's reader, a scaler tile and then each owned tile row.
//
// Runs on every working core. Runtime args, in order:
//   0  the DRAM address of a, stored as tiles (one 32 x 32 bf16 tile per page, tiles in row-major order)
//   1  the first tile row this core owns
//   2  how many consecutive tile rows it owns
//   3  how many tiles make one tile row (the column count of a over 32)
// Compile args: a's TensorAccessorArgs, from index 0.
//
// First it fills one page of c_1 with the reduce scaler the compute kernel multiplies every element by: bf16 1.0
// in the first row of each of the tile's four 16 x 16 faces and zero everywhere else, so each element counts
// exactly once in its row's sum. Then it hands the compute kernel the tiles of its tile rows through c_0, one
// tile at a time, left to right along each tile row.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// The constants and the scaler helper below are local to this kernel.
namespace {

// Circular buffer slots shared with the compute kernel.
constexpr uint32_t kRowTilesCb = tt::CBIndex::c_0;
constexpr uint32_t kScalerCb = tt::CBIndex::c_1;

// Two bf16 1.0 values (bit pattern 0x3f80) side by side in one 32-bit word.
constexpr uint32_t kTwoOnes = 0x3f803f80u;
// In L1 a bf16 tile is its four faces one after another; a face spans 128 words and one face row 8 words.
constexpr uint32_t kWordsPerFace = 128u;
constexpr uint32_t kWordsPerFaceRow = 8u;
constexpr uint32_t kFacesPerTile = 4u;

// Write the scaler tile described above into the next free page of scaler_cb and hand it on.
void push_scaler_tile(uint32_t scaler_cb) {
    cb_reserve_back(scaler_cb, 1);
    volatile tt_l1_ptr uint32_t* scaler_words =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(scaler_cb));
    const uint32_t scaler_word_count = get_tile_size(scaler_cb) / sizeof(uint32_t);
    for (uint32_t w = 0; w < scaler_word_count; ++w) {
        scaler_words[w] = 0u;
    }
    for (uint32_t face = 0; face < kFacesPerTile; ++face) {
        for (uint32_t w = 0; w < kWordsPerFaceRow; ++w) {
            scaler_words[face * kWordsPerFace + w] = kTwoOnes;
        }
    }
    cb_push_back(scaler_cb, 1);
}

}  // namespace

void kernel_main() {
    const uint32_t a_dram_addr = get_arg_val<uint32_t>(0);
    const uint32_t first_tile_row = get_arg_val<uint32_t>(1);
    const uint32_t owned_tile_rows = get_arg_val<uint32_t>(2);
    const uint32_t tiles_across = get_arg_val<uint32_t>(3);

    push_scaler_tile(kScalerCb);

    // Pages of a are whole tiles of the same format as c_0, so c_0's page size is a's page size.
    constexpr auto a_layout = TensorAccessorArgs<0>();
    const auto a_tiles = TensorAccessor(a_layout, a_dram_addr, get_tile_size(kRowTilesCb));

    for (uint32_t tile_row = first_tile_row; tile_row < first_tile_row + owned_tile_rows; ++tile_row) {
        // The tiles of one tile row are consecutive pages of a.
        const uint32_t row_start_page = tile_row * tiles_across;
        for (uint32_t across = 0; across < tiles_across; ++across) {
            cb_reserve_back(kRowTilesCb, 1);
            noc_async_read_tile(row_start_page + across, a_tiles, get_write_ptr(kRowTilesCb));
            noc_async_read_barrier();
            cb_push_back(kRowTilesCb, 1);
        }
    }
}
