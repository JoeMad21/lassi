// Written for LASSI-DF tt-pairs-v0 (task P4.13); the CPU counterpart of the upstream loopback example from tt-metal 5280a9cf.
//
// Loopback on the host: the input is moved from a source array to a destination array one 32x32 bf16 tile
// (2048 bytes) at a time through a one-tile scratch buffer, as the device version moves it through L1.
//
// Usage: loopback <input> <output>
//   input   lassi_io file, bf16, rank 1, a positive multiple of 1024 elements (whole tiles)
//   output  lassi_io file written here, bf16, rank 1, as many elements as the input
// The elements are kept in file order and their bit patterns are copied unchanged.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "lassi_io.h"

namespace {

// A tile holds 32 x 32 bf16 values, each kept as its 16-bit pattern.
constexpr uint64_t kTileValues = 32u * 32u;

// Move `tiles` whole tiles from src to dst through the one-tile buffer `scratch`.
void copy_tiles(const uint16_t* src, uint16_t* dst, uint64_t tiles, uint16_t* scratch) {
    const size_t tile_bytes = static_cast<size_t>(kTileValues) * sizeof(uint16_t);
    for (uint64_t t = 0; t < tiles; ++t) {
        std::memcpy(scratch, src + t * kTileValues, tile_bytes);
        std::memcpy(dst + t * kTileValues, scratch, tile_bytes);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: loopback <input> <output>\n");
        return 2;
    }
    lassi_io_array src = {};
    if (lassi_io_read(argv[1], &src) != 0) {
        return 1;
    }
    if (src.dtype != LASSI_IO_BF16 || src.rank != 1u || src.count == 0u || src.count % kTileValues != 0u) {
        std::fprintf(stderr, "loopback: the input must be a rank-1 bf16 array of whole 32x32 tiles\n");
        lassi_io_free(&src);
        return 1;
    }
    const uint64_t count = src.count;

    std::vector<uint16_t> dst(count);
    std::vector<uint16_t> scratch(kTileValues);
    copy_tiles(static_cast<const uint16_t*>(src.data), dst.data(), count / kTileValues, scratch.data());

    const uint64_t dims[1] = {count};
    const int status = lassi_io_write(argv[2], "output", LASSI_IO_BF16, 1u, dims, dst.data());
    lassi_io_free(&src);
    return status;
}
