/* Written for LASSI-DF tt-pairs-v0-ttkernel (task P17.41): the computation of the C++ reference
 * assets/bench/tt-pairs-v0/eltwise_binary/cpp/eltwise_binary.cpp as one function with C linkage, which Polygeist's
 * cgeist lowers to affine MLIR by its name for the generation prompt. Compute only: no I/O, no lassi_io.h, no main.
 *
 * c[i] = a[i] + b[i] for i in [0, count). The reference widens each bf16 element of a and b exactly to the float
 * of the same value and adds in float, keeping the float sum; here a and b hold those widened values, so each sum
 * is the reference's. The bf16 widening belongs to the reference's file format and stays there.
 */

#ifdef __cplusplus
extern "C" {
#endif

void lassi_eltwise_binary(const float *a, const float *b, float *c, long count) {
    for (long i = 0; i < count; ++i) {
        c[i] = a[i] + b[i];
    }
}

#ifdef __cplusplus
}
#endif
