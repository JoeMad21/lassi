// The TT-NN probe of plans/spikes/ttmlir-runtime.md (2026-10-30; OQ-112, route A's first measurement): bale's histo
// as one ttir.scatter with SUM on 32-bit integers, zeros of 4096 counts scattered with ones at 16384 indices, in the
// form of the pinned test/ttmlir/Silicon/TTNN/n150/data_movement/scatter/scatter.mlir:16-17 at tt-mlir 5f396fd6.
func.func @histo(%arg0: tensor<4096xi32>, %arg1: tensor<16384xi32>, %arg2: tensor<16384xi32>) -> tensor<4096xi32> {
  %0 = "ttir.scatter"(%arg0, %arg1, %arg2) <{dim = 0 : i32, scatter_reduce_type = #ttcore.reduce_type<sum>}> : (tensor<4096xi32>, tensor<16384xi32>, tensor<16384xi32>) -> tensor<4096xi32>
  return %0 : tensor<4096xi32>
}
