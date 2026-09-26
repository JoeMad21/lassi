// A data-movement kernel for the fixture capture. The kernel JIT compiles it at first launch; the host compiler
// never does, and could not, since the device header below is not on the host include path.
#include "dataflow_api.h"

void kernel_main() {}
