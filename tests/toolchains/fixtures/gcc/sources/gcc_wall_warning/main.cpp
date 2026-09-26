// A warning that -Wall turns on: an unused variable. The native flags hold no -Wall, so the pragma below turns
// -Wunused-variable on for this file.
#pragma GCC diagnostic warning "-Wunused-variable"
#include <cstdio>

int main() {
    int unused = 0;
    std::printf("done\n");
    return 0;
}
