#include <cstdio>
#include <vector>

__global__ void fill(float *y, int n, float v)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = v;
    }
}

int main()
{
    std::vector<float> host(1024, 1.0f);
    float sum = 0.0f;
    for (int i = 0; i < host.size(); i++) {
        sum += host[i];
    }
    std::printf("%f\n", sum);
    return 0;
}
