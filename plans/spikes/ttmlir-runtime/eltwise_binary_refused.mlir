// The deliberately mis-shaped TTIR module of tt-pairs-v0's eltwise_binary for the end-to-end check of
// plans/spikes/ttmlir-runtime.md (2026-10-30): eltwise_binary_ok.mlir on 128x256 tensors, half the 65536 elements of
// each input file, so it compiles and the driver must refuse its inputs before it opens the device. Written by hand
// for the check; never a model's output.
func.func @eltwise_binary(%a: tensor<128x256xbf16>, %b: tensor<128x256xbf16>) -> tensor<128x256xf32> {
  %sum = "ttir.add"(%a, %b) : (tensor<128x256xbf16>, tensor<128x256xbf16>) -> tensor<128x256xbf16>
  %c = "ttir.typecast"(%sum) : (tensor<128x256xbf16>) -> tensor<128x256xf32>
  return %c : tensor<128x256xf32>
}
