#include <cstdio>

float twice(float value);

void scale(int n, const float *x, float *y)
{
#pragma omp target teams distribute parallel for map(to: x[0:n]) map(from: y[0:n])
    for (int i = 0; i < n; i++) {
        y[i] = twice(x[i]);
    }
}

int main()
{
    static float x[1024], y[1024];
    for (int i = 0; i < 1024; i++) {
        x[i] = 1.0f * i;
    }
    scale(1024, x, y);
    std::printf("%f\n", y[1023]);
    return 0;
}
