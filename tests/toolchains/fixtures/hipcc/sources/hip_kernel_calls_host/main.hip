// Device-side error: the kernel calls host_scale, a plain host function.
#include <hip/hip_runtime.h>

float host_scale(float x) {
    return 2.0f * x;
}

__global__ void scale(int n, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = host_scale(y[i]);
    }
}

int main() {
    const int n = 256;
    float* y = nullptr;
    if (hipMalloc(&y, n * sizeof(float)) != hipSuccess) {
        return 1;
    }
    scale<<<(n + 63) / 64, 64>>>(n, y);
    hipError_t synced = hipDeviceSynchronize();
    hipError_t freed = hipFree(y);
    return synced == hipSuccess && freed == hipSuccess ? 0 : 1;
}
