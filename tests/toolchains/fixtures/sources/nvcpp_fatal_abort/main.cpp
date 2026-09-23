#include <cstdio>

void outer(int n, float *y)
{
#pragma omp target teams distribute parallel for map(tofrom: y[0:n])
    for (int i = 0; i < n; i++) {
#pragma omp target
        y[i] += 1.0f;
    }
}

int main()
{
    static float y[1024];
    for (int i = 0; i < 1024; i++) {
        y[i] = 2.0f;
    }
    outer(1024, y);
    std::printf("%f\n", y[0]);
    return 0;
}
