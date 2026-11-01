// The bf16 TT-NN probe of plans/spikes/ttmlir-runtime.md (2026-10-30): add.mlir's ttir.add on bf16 64x128, a 16-bit
// format, to tell whether a TT-NN program runs on ttsim when no 32-bit input is tilized on the device.
func.func @add(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
  %1 = "ttir.add"(%arg0, %arg1) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
  return %1 : tensor<64x128xbf16>
}
