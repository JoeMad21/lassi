// A TT host program whose helper never uses one parameter: a warning under the pinned flags, which lack -Werror.
#include <tt-metalium/host_api.hpp>

static int scale_value(int value, int factor) {
    return value * 2;
}

int main() {
    return scale_value(3, 4) == 6 ? 0 : 1;
}
