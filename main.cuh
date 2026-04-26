#pragma once
#include <iostream>
#include <openvr.h>
#include <opencv2/core.hpp>

using namespace cv;
int main(const int argc, char** argv);

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


void openCvCameraTest();

void openCvImageTest(const std::string& imgPath);

void executeGpuTestKernel();

void openVRTest();


void saveCameraCalibrationToFile(Mat* kLeft, Mat* dLeft, Mat* kRight, Mat* dRight,
    Mat* newKLeft, Mat* newKRight, Mat* stereoCamTranslation, Mat* stereoCamRotation, std::vector<cv::Mat>* calibrationImages);

void readCameraCalibrationFromFile(Mat* kLeft, Mat* dLeft, Mat* kRight, Mat* dRight,
    Mat* newKLeft, Mat* newKRight, Mat* stereoCamTranslation, Mat* stereoCamRotation);

bool verifySavedCalibration(Mat* kLeft, Mat* dLeft, Mat* kRight, Mat* dRight,
    Mat* newKLeft, Mat* newKRight,  Mat* stereoCamTranslation, Mat* stereoCamRotation);


void cpuMarkerDetection(const Mat* image, Mat* camCalKLeft, Mat* camCalDLeft, Mat* camCalKRight, Mat* camCalDRight,
    Mat* camCalNewKLeft, Mat* camCalNewKRight);

/**
 * Signal CUDA has exclusive access to this resource
 * @return returns the surface object handle to the CUDA exclusive access
 */
cudaSurfaceObject_t setResourceCudaAccess(cudaGraphicsResource_t resource);
/**
 * Signal CUDA has exclusive access to these resources
 * @return returns the surface object handle to the CUDA exclusive access
 */
void setResourcesCudaAccess(std::vector<cudaGraphicsResource_t>* resources, std::vector<cudaSurfaceObject_t>* surfacesOut);

/**
 * Remove CUDA's exclusive access to this resource
 */
void unsetResourceCudaAccess(cudaSurfaceObject_t surface, cudaGraphicsResource_t resource);
/**
 * Remove CUDA's exclusive access to this set of resources
 */
void unsetResourcesCudaAccess(std::vector<cudaSurfaceObject_t>* surfaces, std::vector<cudaGraphicsResource_t>* resources);
