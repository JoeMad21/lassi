// Device link: the kernel calls device_helper, which is declared __device__ and never defined.
#include <hip/hip_runtime.h>

__device__ int device_helper(int value);

__global__ void apply(int n, int* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = device_helper(y[i]);
    }
}

int main() {
    const int n = 256;
    int* y = nullptr;
    if (hipMalloc(&y, n * sizeof(int)) != hipSuccess) {
        return 1;
    }
    apply<<<(n + 63) / 64, 64>>>(n, y);
    hipError_t synced = hipDeviceSynchronize();
    hipError_t freed = hipFree(y);
    return synced == hipSuccess && freed == hipSuccess ? 0 : 1;
}
