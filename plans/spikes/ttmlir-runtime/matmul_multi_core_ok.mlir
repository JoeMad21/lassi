// The hand-written TTIR module of tt-pairs-v0's matmul_multi_core for the end-to-end checks of
// plans/spikes/ttmlir-runtime.md (2026-10-30): one ttir.matmul of the two bf16 640x640 inputs, as
// plans/spikes/ttir-d2m.md section 2 lowered it; the item's output is bf16 640x640. Written by hand; never a model's
// output.
func.func @matmul_multi_core(%a: tensor<640x640xbf16>, %b: tensor<640x640xbf16>) -> tensor<640x640xbf16> {
  %c = "ttir.matmul"(%a, %b) : (tensor<640x640xbf16>, tensor<640x640xbf16>) -> tensor<640x640xbf16>
  return %c : tensor<640x640xbf16>
}
