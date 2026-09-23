#include <cstdio>
#include "kernels/scale.cuh"

__global__ void saxpy(int n, float a, const float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = scale(x[i], a) + y[i];
    }
}

int main()
{
    std::printf("built\n");
    return 0;
}
