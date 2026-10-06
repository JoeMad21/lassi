// Host-side error: main calls device_scale, which is declared __device__ only.
#include <hip/hip_runtime.h>

__device__ float device_scale(float x) {
    return 2.0f * x;
}

int main() {
    float value = device_scale(1.0f);
    return value > 0.0f ? 0 : 1;
}
