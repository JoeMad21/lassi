// P17.1 compile-only probe (plans/spikes/p17-frameworks/run.sh): one kernel, no host main and no runtime call,
// so `hipcc --offload-arch=gfx942 -c` needs only the compiler and the device libraries, never a device.
#include <hip/hip_runtime.h>

__global__ void p17_axpy(float a, const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}
