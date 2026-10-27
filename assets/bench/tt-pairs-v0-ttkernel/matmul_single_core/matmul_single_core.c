/* Written for LASSI-DF tt-pairs-v0-ttkernel (task P17.41): the computation of the C++ reference
 * assets/bench/tt-pairs-v0/matmul_single_core/cpp/matmul_single_core.cpp as one function with C linkage, which
 * Polygeist's cgeist lowers to affine MLIR by its name for the generation prompt. Compute only: no I/O, no
 * lassi_io.h, no main.
 *
 * c = a b for the row-major m x k matrix a and k x n matrix b: each c[i][j] is a float sum over r in ascending
 * order of a[i][r] * b[r][j], started at 0, as the reference sums each element (it reads b through a transposed
 * copy, which moves values but changes none). The reference widens each bf16 element exactly to the float of the
 * same value and rounds each sum to bf16 to nearest, ties to even; here a and b hold the widened values and c the
 * float sums before that rounding, which belongs to the reference's file format and stays there.
 */

#ifdef __cplusplus
extern "C" {
#endif

void lassi_matmul_single_core(const float *a, const float *b, float *c, long m, long k, long n) {
    for (long i = 0; i < m; ++i) {
        for (long j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (long r = 0; r < k; ++r) {
                sum += a[i * k + r] * b[r * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

#ifdef __cplusplus
}
#endif
