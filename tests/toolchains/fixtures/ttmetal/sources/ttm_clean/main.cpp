// A clean TT host program for the fixture capture: it includes the pinned host API and does no device work.
#include <tt-metalium/host_api.hpp>

int main() {
    int total = 0;
    for (int i = 0; i < 4; ++i) {
        total += i;
    }
    return total == 6 ? 0 : 1;
}
