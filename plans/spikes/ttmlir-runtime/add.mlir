// The TT-Metal probe of plans/spikes/ttmlir-runtime.md (2026-10-30): one ttir.add on f32 64x128, the module of the
// pinned test/ttmlir/Silicon/TTMetal/n150/simple_add.mlir:5-12 at tt-mlir 5f396fd6 without its CHECK lines.
func.func @add(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %1 = "ttir.add"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}
