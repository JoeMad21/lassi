#pragma once

__device__ inline float scale(float v, float s)
{
    float unused = 0.0f;
    return v * s;
}
