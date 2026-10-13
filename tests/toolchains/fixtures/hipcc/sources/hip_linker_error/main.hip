// Host linker error: main calls helper, which is declared and never defined.
#include <hip/hip_runtime.h>

int helper(int value);

int main() {
    return helper(1) == 2 ? 0 : 1;
}
