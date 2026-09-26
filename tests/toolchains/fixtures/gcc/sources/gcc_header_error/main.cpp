// An error in an included header: kernels/scale.h uses factor, which is never declared.
#include "kernels/scale.h"
#include <cstdio>

int main() {
    float y[4] = {0.0f, 1.0f, 2.0f, 3.0f};
    scale(y, 4);
    std::printf("%f\n", y[3]);
    return 0;
}
