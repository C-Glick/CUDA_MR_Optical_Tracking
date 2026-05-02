#include "Cuda_Func.cuh"

#include "opencv2/core/cuda_types.hpp"


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

__global__ void GpuKernelSimpleCopy (cudaSurfaceObject_t inputImage, cudaSurfaceObject_t outputImage, int width, int height)
{
    int threadIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
    // Calculate global row and column for the thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int numChannels = 4;
    if (col < width && row < height)
    {
        //copy each pixel from inputImage to outputImage
        uchar4 temp;
        surf2Dread(&temp, inputImage, col*numChannels, row);
        surf2Dwrite(temp, outputImage, col*numChannels, row);
    }
}

__global__ void GpuKernelRemapImage (cudaSurfaceObject_t inputImage,
    cudaSurfaceObject_t outputImage, float* map1, float* map2, int width, int height)
{
    int threadIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
    // Calculate global row and column for the thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;


    int numChannels = 4;
    if (col < width && row < height)
    {
        //compute which source pixel should be used at this output location
        //convert from float to int to get nearest pixel
        int srcX = round(map1[row*width + col]);
        int srcY = round(map2[row*width + col]);

        //todo set pixels outside range to black
        //clamp srcX and srcY to the image size
        srcX = max(0, min(srcX, width-1));
        srcY = max(0, min(srcY, height-1));

        //copy source pixel to destination pixel
        uchar4 temp;
        surf2Dread(&temp, inputImage, srcX*numChannels, srcY);
        surf2Dwrite(temp, outputImage, col*numChannels, row);
    }
}

__global__ void GpuKernelAdaptiveThresholdMeanCBinaryInv (char* inputImage, char* outputImage, int width, int height,
    bool isContinuous, int step, int maxValue, int blockSize, int C)
{
    int threadIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
    // Calculate global row and column for the thread
    int home_col = blockIdx.x * blockDim.x + threadIdx.x;
    int home_row = blockIdx.y * blockDim.y + threadIdx.y;

    int numChannels = 1;

    //copy source pixel to destination pixel
    //outputImage[row * step + col] = inputImage[row * step + col];

    //compute the mean of neighborhood of pixels
    int blockDelta = blockSize / 2;

    float mean = 0;
    for (int neighbor_x_delta = -blockDelta; neighbor_x_delta <= blockDelta; neighbor_x_delta++ )
    {
        for (int neighbor_y_delta = -blockDelta; neighbor_y_delta <= blockDelta; neighbor_y_delta++ )
        {

            //clamp pixel read on edges (same as edge replication)
            int neighbor_x = max(0, min(home_col + neighbor_x_delta, width-1));
            int neighbor_y = max(0, min(home_row + neighbor_y_delta, height-1));


            mean += inputImage[neighbor_y * step + neighbor_x];
        }
    }

    mean = mean / (blockSize * blockSize);

    //if home pixel value is less than the mean - c, set its value to max
    if (inputImage[home_row * step + home_col] < mean - C)
    {
        outputImage[home_row * step + home_col] = maxValue;
    }else
    {
        outputImage[home_row * step + home_col] = 0;
    }

}