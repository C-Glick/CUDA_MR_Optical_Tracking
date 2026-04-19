#include "Cuda_Func.cuh"


/**
 * Compute the frame's luminance values and send pixel results back to CPU
 * @param frame
 * @param frameNumber
 * @param luminance
 * @param width
 * @param height
 */
__global__ void GpuKernelVectorAdd (const float* in1, const float* in2, float* out)
{
    int threadIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

    out[threadIndex] = in1[threadIndex] + in2[threadIndex];
}


__global__ void GpuKernelColorChange (cudaSurfaceObject_t image, int width, int height)
{
    int threadIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
    // Calculate global row and column for the thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int numChannels = 4;
    if (col < width && row < height)
    {
        //set the first channel of each pixel to 0
        char b = 0;
        surf2Dwrite(b, image, col*numChannels, row);
        //image[threadIndex] = 0;
    }

}