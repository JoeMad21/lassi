// A syntax error: the declaration of total has no semicolon.
#include <cstdio>

int main() {
    int total = 0
    for (int i = 0; i < 4; ++i) {
        total += i;
    }
    std::printf("%d\n", total);
    return 0;
}
