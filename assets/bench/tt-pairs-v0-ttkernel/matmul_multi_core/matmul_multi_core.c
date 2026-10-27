/* Written for LASSI-DF tt-pairs-v0-ttkernel (task P17.41): the computation of the C++ reference
 * assets/bench/tt-pairs-v0/matmul_multi_core/cpp/matmul_multi_core.cpp (its function multiply) as one function
 * with C linkage, which Polygeist's cgeist lowers to affine MLIR by its name for the generation prompt. Compute
 * only: no I/O, no lassi_io.h, no main.
 *
 * c = a b for the row-major m x k matrix a and k x n matrix b, row by row as multiply builds it: a row of c starts
 * at 0, and for p in ascending order each c[i][j] gains a[i][p] * b[p][j], so every element gets the same float
 * additions in the same order as a dot product over k. The reference widens each bf16 element exactly to the
 * float of the same value, keeps the row's sums in floats, and rounds each to bf16 to nearest, ties to even; here
 * a and b hold the widened values and c the float sums before that rounding, which belongs to the reference's
 * file format and stays there.
 */

#ifdef __cplusplus
extern "C" {
#endif

void lassi_matmul_multi_core(const float *a, const float *b, float *c, long m, long k, long n) {
    for (long i = 0; i < m; ++i) {
        for (long j = 0; j < n; ++j) {
            c[i * n + j] = 0.0f;
        }
        for (long p = 0; p < k; ++p) {
            const float left = a[i * k + p];
            for (long j = 0; j < n; ++j) {
                c[i * n + j] += left * b[p * n + j];
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
