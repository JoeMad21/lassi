// The deliberately wrong TTIR module of tt-pairs-v0's eltwise_binary for the end-to-end check of
// plans/spikes/ttmlir-runtime.md (2026-10-30): eltwise_binary_ok.mlir with ttir.subtract in place of ttir.add, so it
// compiles and runs and its output c must fail the oracle. Written by hand for the check; never a model's output.
func.func @eltwise_binary(%a: tensor<256x256xbf16>, %b: tensor<256x256xbf16>) -> tensor<256x256xf32> {
  %diff = "ttir.subtract"(%a, %b) : (tensor<256x256xbf16>, tensor<256x256xbf16>) -> tensor<256x256xbf16>
  %c = "ttir.typecast"(%diff) : (tensor<256x256xbf16>) -> tensor<256x256xf32>
  return %c : tensor<256x256xf32>
}
