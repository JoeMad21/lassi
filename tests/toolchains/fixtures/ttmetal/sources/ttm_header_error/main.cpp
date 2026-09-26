// A TT host program whose own header holds an error.
#include <tt-metalium/host_api.hpp>

#include "host/scale.h"

int main() {
    return scale(3) == 6 ? 0 : 1;
}
