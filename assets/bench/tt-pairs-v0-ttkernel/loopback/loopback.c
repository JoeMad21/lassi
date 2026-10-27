/* Written for LASSI-DF tt-pairs-v0-ttkernel (task P17.41): the computation of the C++ reference
 * assets/bench/tt-pairs-v0/loopback/cpp/loopback.cpp as one function with C linkage, which Polygeist's cgeist
 * lowers to affine MLIR by its name for the generation prompt. Compute only: no I/O, no lassi_io.h, no main.
 *
 * Moves `tiles` whole tiles of 1024 16-bit values (each a bf16 bit pattern, copied unchanged) from src to dst
 * through the one-tile buffer scratch, tile by tile, as the reference's copy_tiles does with two memcpy calls per
 * tile; here each copy is a loop over the tile's values, which gives the same bytes.
 */

#define LASSI_TILE_VALUES 1024

#ifdef __cplusplus
extern "C" {
#endif

void lassi_loopback(const unsigned short *src, unsigned short *dst, long tiles, unsigned short *scratch) {
    for (long t = 0; t < tiles; ++t) {
        for (long e = 0; e < LASSI_TILE_VALUES; ++e) {
            scratch[e] = src[t * LASSI_TILE_VALUES + e];
        }
        for (long e = 0; e < LASSI_TILE_VALUES; ++e) {
            dst[t * LASSI_TILE_VALUES + e] = scratch[e];
        }
    }
}

#ifdef __cplusplus
}
#endif
