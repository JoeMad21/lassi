// The hand-written correct TTIR module of tt-pairs-v0's eltwise_binary for the end-to-end check of
// plans/spikes/ttmlir-runtime.md (2026-10-30): the two bf16 inputs of 65536 elements written as 256x256 tensors
// (the same elements in row-major order), a bf16 add on the device, and the sum widened to f32, which is what the
// item's tt program writes as its output c. Written by hand for the check; never a model's output.
func.func @eltwise_binary(%a: tensor<256x256xbf16>, %b: tensor<256x256xbf16>) -> tensor<256x256xf32> {
  %sum = "ttir.add"(%a, %b) : (tensor<256x256xbf16>, tensor<256x256xbf16>) -> tensor<256x256xbf16>
  %c = "ttir.typecast"(%sum) : (tensor<256x256xbf16>) -> tensor<256x256xf32>
  return %c : tensor<256x256xf32>
}
