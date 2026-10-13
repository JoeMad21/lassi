// A clean HIP program for the hipcc-gfx942 fixtures and its remote test: a saxpy kernel; built, never run.
#include <hip/hip_runtime.h>

__global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    const int n = 1024;
    float* x = nullptr;
    float* y = nullptr;
    if (hipMalloc(&x, n * sizeof(float)) != hipSuccess) {
        return 1;
    }
    if (hipMalloc(&y, n * sizeof(float)) != hipSuccess) {
        return hipFree(x) == hipSuccess ? 1 : 2;
    }
    saxpy<<<(n + 63) / 64, 64>>>(n, 2.0f, x, y);
    hipError_t launched = hipGetLastError();
    hipError_t synced = hipDeviceSynchronize();
    hipError_t freed_x = hipFree(x);
    hipError_t freed_y = hipFree(y);
    if (launched != hipSuccess || synced != hipSuccess || freed_x != hipSuccess || freed_y != hipSuccess) {
        return 1;
    }
    return 0;
}
