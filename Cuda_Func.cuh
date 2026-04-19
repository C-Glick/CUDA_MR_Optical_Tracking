#pragma once

__global__ void GpuKernelVectorAdd (const float* in1, const float* in2, float* out);

__global__ void GpuKernelColorChange (cudaSurfaceObject_t image, int width, int height);
