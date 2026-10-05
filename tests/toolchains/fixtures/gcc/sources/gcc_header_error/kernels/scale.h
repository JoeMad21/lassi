// Scales an array in place. factor is never declared, so every file that includes this one fails to build.
#pragma once

inline void scale(float *x, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] *= factor;
    }
}
