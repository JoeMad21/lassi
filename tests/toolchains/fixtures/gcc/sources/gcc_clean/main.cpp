// A clean build: an OpenMP parallel loop with a reduction, built with the native flags (-O3 -fopenmp).
#include <cstdio>

int main() {
    double sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < 1000; ++i) {
        sum += 0.5 * i;
    }
    std::printf("%.1f\n", sum);
    return 0;
}
