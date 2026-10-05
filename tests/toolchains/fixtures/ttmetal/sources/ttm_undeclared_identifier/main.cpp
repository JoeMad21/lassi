// A TT host program that uses an identifier it never declares.
#include <tt-metalium/host_api.hpp>

int main() {
    int total = 0;
    for (int i = 0; i < 4; ++i) {
        total += undefined_var;
    }
    return total;
}
