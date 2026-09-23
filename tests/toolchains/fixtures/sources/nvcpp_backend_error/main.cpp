#include <cstdio>

void spread(int n, const float *x, float *y)
{
#pragma omp target teams distribute parallel for map(to: x[0:n]) map(from: y[0:n])
    for (int i = 0; i < n; i++) {
        float scratch[262144];
        for (int j = 0; j < 262144; j++) {
            scratch[j] = x[(i + j) % n];
        }
        float sum = 0.0f;
        for (int j = 0; j < 262144; j++) {
            sum += scratch[(j * 7919) % 262144];
        }
        y[i] = sum;
    }
}

int main()
{
    static float x[1024], y[1024];
    for (int i = 0; i < 1024; i++) {
        x[i] = 1.0f;
    }
    spread(1024, x, y);
    std::printf("%f\n", y[0]);
    return 0;
}
