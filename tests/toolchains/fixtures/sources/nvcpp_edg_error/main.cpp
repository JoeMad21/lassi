#include <cstdio>

void saxpy(int n, float a, const float *x, float *y)
{
#pragma omp target teams distribute parallel for map(to: x[0:n]) map(tofrom: y[0:n])
    for (int i = 0; i < n; i++) {
        y[i] = a * x[i] + undefined_var;
    }
}

int main()
{
    static float x[1024], y[1024];
    for (int i = 0; i < 1024; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    saxpy(1024, 2.0f, x, y);
    std::printf("%f\n", y[0]);
    return 0;
}
