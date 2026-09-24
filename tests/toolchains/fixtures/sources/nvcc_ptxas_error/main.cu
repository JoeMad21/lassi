#include <cstdio>

__global__ void reduce(const float *in, float *out, int n)
{
    __shared__ float buf[65536];
    for (int j = threadIdx.x; j < 65536; j += blockDim.x) {
        buf[j] = j < n ? in[j] : 0.0f;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int j = 0; j < 65536; j++) {
            sum += buf[j];
        }
        out[blockIdx.x] = sum;
    }
}

int main()
{
    std::printf("built\n");
    return 0;
}
