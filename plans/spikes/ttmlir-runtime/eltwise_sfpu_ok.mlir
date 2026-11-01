// The hand-written TTIR module of tt-pairs-v0's eltwise_sfpu for the end-to-end checks of
// plans/spikes/ttmlir-runtime.md (2026-10-30): the bf16 input of 65536 elements as a 256x256 tensor and one ttir.exp,
// as plans/spikes/ttir-d2m.md section 2 lowered it; the item's output is bf16. Written by hand; never a model's output.
func.func @eltwise_sfpu(%src0: tensor<256x256xbf16>) -> tensor<256x256xbf16> {
  %result = "ttir.exp"(%src0) : (tensor<256x256xbf16>) -> tensor<256x256xbf16>
  return %result : tensor<256x256xbf16>
}
