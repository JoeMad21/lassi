#include <cstdio>

__global__ void bump(int *out)
{
    int value = threadIdx.x;
    asm volatile("add.bogus.s32 %0, %0, 1;" : "+r"(value));
    out[threadIdx.x] = value;
}

int main()
{
    int *out = nullptr;
    cudaMalloc(&out, 32 * sizeof(int));
    bump<<<1, 32>>>(out);
    cudaDeviceSynchronize();
    cudaFree(out);
    std::printf("built\n");
    return 0;
}
