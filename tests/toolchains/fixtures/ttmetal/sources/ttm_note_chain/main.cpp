// A TT host program that calls an overloaded function with an argument neither overload takes.
#include <tt-metalium/host_api.hpp>

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
    return values[0];
}
