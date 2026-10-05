// A TT host program that calls a function it declares and never defines.
#include <tt-metalium/host_api.hpp>

int helper(int value);

int main() {
    return helper(3);
}
