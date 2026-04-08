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