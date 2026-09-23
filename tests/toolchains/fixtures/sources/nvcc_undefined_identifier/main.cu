#include <cstdio>

__global__ void saxpy(int n, float a, const float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + undefined_var;
    }
}

int main()
{
    std::printf("built\n");
    return 0;
}
