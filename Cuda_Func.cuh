#pragma once

__global__ void GpuKernelVectorAdd (const float* in1, const float* in2, float* out);

__global__ void GpuKernelColorChange (cudaSurfaceObject_t image, int width, int height);

__global__ void GpuKernelRemapImage (cudaSurfaceObject_t inputImage,
    cudaSurfaceObject_t outputImage, float* map1, float* map2, int width, int height);

__global__ void GpuKernelSimpleCopy (cudaSurfaceObject_t inputImage, cudaSurfaceObject_t outputImage, int width, int height);

/**
 * Adaptive threshold that takes a greyscale 1 channel image and transforms it to a binary image.
 * This specifically implements the Mean C, inverse threshold.
 * The algorithm takse the mean of the neigbouring pixels, minus c (constant) as the threshold.
 * pixel values above that threshold = 0, below threshold = maxValue.
 * @param inputImage input greyscale image
 * @param outputImage output binary image
 * @param width
 * @param height
 * @param isContinuous if the image is continuous
 * @param step The byte offset between successive rows in the image to align to memory
 * @param maxValue the max value to set pixels below the threshold to
 * @param blockSize the neighborhood of pixels to look at, must be odd
 * @param C the constant to subtract from the mean
 */
__global__ void GpuKernelAdaptiveThresholdMeanCBinaryInv (char* inputImage, char* outputImage, int width, int height,
    bool isContinuous, int step, int maxValue, int blockSize, int C);