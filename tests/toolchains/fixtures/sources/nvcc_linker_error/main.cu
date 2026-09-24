#include <cstdio>

void helper(float *data, int n);

int main()
{
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    helper(data, 4);
    std::printf("%f\n", data[0]);
    return 0;
}
