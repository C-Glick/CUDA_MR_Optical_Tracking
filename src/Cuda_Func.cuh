#pragma once

/**
 * Modifies the given image in place and changes the color of the image. Proof
 * of concept for memory management and how cuda surfaces work.
 * @param image Input image, Cuda surface pointer
 * @param width Image width
 * @param height Image height
 */
__global__ void GpuKernelColorChange (cudaSurfaceObject_t image, int width, int height);

/**
 * Copies data from one texture to another as a proof of concept for memory management and how
 * Cuda surfaces work.
 * @param inputImage Input image, Cuda surface pointer
 * @param outputImage Output image, Cuda surface pointer
 * @param width The width of the input / output image
 * @param height The height of the input / output image
 */
__global__ void GpuKernelSimpleCopy (cudaSurfaceObject_t inputImage, cudaSurfaceObject_t outputImage, int width, int height);


/**
 * Remaps pixels from the input image to the output image based on the map values provided
 * in map1 and map2. Effectively distorting the image to correct the fisheye lens effect
 * @param inputImage Input image, Cuda surface pointer
 * @param outputImage Output image, Cuda surface pointer
 * @param map1 Remap data for x coordinates (ie the pixel's new x value is given by map1[x][y])
 * @param map2 Remap data for y coordinates (ie the pixel's new y value is given by map2[x][y])
 * @param width The width of the input / output image
 * @param height The height of the input / output image
 */
__global__ void GpuKernelRemapImage (cudaSurfaceObject_t inputImage,
                                     cudaSurfaceObject_t outputImage, float* map1, float* map2, int width, int height);

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