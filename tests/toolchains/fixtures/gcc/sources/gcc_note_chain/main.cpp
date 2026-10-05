// A note chain: the call matches neither overload of scale, so the error is followed by a note per candidate.
#include <cstdio>

void scale(float *x, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] *= 2.0f;
    }
}

void scale(double *x, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] *= 2.0;
    }
}

int main() {
    int values[4] = {1, 2, 3, 4};
    scale(values, 4);
    std::printf("%d\n", values[0]);
    return 0;
}
