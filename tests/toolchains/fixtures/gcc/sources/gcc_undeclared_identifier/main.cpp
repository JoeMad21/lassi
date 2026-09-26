// An undeclared identifier: the loop body uses undefined_var, which is never declared.
#include <cstdio>

int main() {
    float y[4] = {0.0f, 1.0f, 2.0f, 3.0f};
    for (int i = 0; i < 4; ++i) {
        y[i] = 2.0f * y[i] + undefined_var;
    }
    std::printf("%f\n", y[3]);
    return 0;
}
