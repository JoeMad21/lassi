// The hand-written D2M module of tt-pairs-v0's eltwise_sfpu for the end-to-end checks of
// plans/spikes/ttmlir-runtime.md (2026-10-30): plans/spikes/ttir-d2m.md section 2's D2M eltwise_binary form with one
// input and d2m.tile_exp, as that section describes it: the input onto an 8x8 grid of 1x1-tile bf16 shards, one
// generic that loads each shard, takes exp tile by tile, and stores it, and the result back to a host tensor. Written
// by hand; never a model's output.
#layout = #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded, index_map = map(0)>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
!tile = !ttcore.tile<32x32, bf16>
!tiles = tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #layout>
!shard = tensor<1x1x!ttcore.tile<32x32, bf16>>
!cb = !d2m.cb<tensor<1x1x!ttcore.tile<32x32, bf16>>>
func.func @eltwise_sfpu(%src0: tensor<256x256xbf16>) -> tensor<256x256xbf16> {
  %a0 = d2m.empty() : !tiles
  %a1 = d2m.to_layout %src0, %a0 : tensor<256x256xbf16> into !tiles -> !tiles
  %c0 = d2m.empty() : !tiles
  %c1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%a1 : !tiles)
      outs(%c0 : !tiles) {
  ^unified0(%cb0: !cb, %cb1: !cb):
    %i = d2m.block_index(0) : index
    %j = d2m.block_index(1) : index
    %ea = tensor.empty() : !shard
    %la = d2m.remote_load %ea %a1[%i, %j] : !shard, !tiles -> !shard
    %ec = tensor.empty() : !shard
    %s = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%la : !shard) outs(%ec : !shard) {
    ^bb0(%x: !tile, %o: !tile):
      %t = "d2m.tile_exp"(%x) : (!tile) -> !tile
      linalg.yield %t : !tile
    } -> !shard
    %st = d2m.remote_store %c0[%i, %j] %s : !tiles, !shard -> !tiles
    d2m.yield %st : (!tiles)
  } : !tiles
  %h0 = d2m.empty() : tensor<256x256xbf16>
  %h1 = d2m.to_layout %c1, %h0 : !tiles into tensor<256x256xbf16> -> tensor<256x256xbf16>
  return %h1 : tensor<256x256xbf16>
}
