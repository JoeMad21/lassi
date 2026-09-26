// A TT host program with a syntax error: the declaration of total has no semicolon before the loop.
#include <tt-metalium/host_api.hpp>

int main() {
    int total = 0
    for (int i = 0; i < 4; ++i) {
        total += i;
    }
    return total;
}
