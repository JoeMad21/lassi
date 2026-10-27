/* Written for LASSI-DF tt-pairs-v0-ttkernel (task P17.41): the computation of the C++ reference
 * assets/bench/tt-pairs-v0/eltwise_sfpu/cpp/eltwise_sfpu.cpp as one function with C linkage, which Polygeist's
 * cgeist lowers to affine MLIR by its name for the generation prompt. Compute only: no I/O, no lassi_io.h, no main.
 *
 * result[i] = expf(src0[i]) for i in [0, count). The reference widens each bf16 element of src0 exactly to the
 * float of the same value, takes exp of that float in float (the float overload of exp, expf), and rounds it to bf16
 * to nearest, ties to even; here src0 holds the widened values and result the float values before that rounding,
 * which belongs to the reference's file format and stays there.
 */

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void lassi_eltwise_sfpu(const float *src0, float *result, long count) {
    for (long i = 0; i < count; ++i) {
        result[i] = expf(src0[i]);
    }
}

#ifdef __cplusplus
}
#endif
