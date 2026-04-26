#pragma once

__global__ void GpuKernelVectorAdd (const float* in1, const float* in2, float* out);

__global__ void GpuKernelColorChange (cudaSurfaceObject_t image, int width, int height);

__global__ void GpuKernelRemapImage (cudaSurfaceObject_t inputImage,
    cudaSurfaceObject_t outputImage, float* map1, float* map2, int width, int height);

__global__ void GpuKernelSimpleCopy (cudaSurfaceObject_t inputImage, cudaSurfaceObject_t outputImage, int width, int height);