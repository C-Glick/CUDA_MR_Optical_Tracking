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