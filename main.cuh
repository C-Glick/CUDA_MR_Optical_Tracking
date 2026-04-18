#pragma once
#include <iostream>
#include <openvr.h>
#include <opencv2/core.hpp>

using namespace cv;
int main(const int argc, char** argv);

void openCvCameraTest();

void openCvImageTest(const std::string& imgPath);

void executeGpuTestKernel();

void openVRTest();

void readCameraCalibrationFromFile(Mat* kLeft, Mat* dLeft, Mat* kRight, Mat* dRight,
    Mat* newKLeft, Mat* newKRight);
bool verifySavedCalibration(Mat* kLeft, Mat* dLeft, Mat* kRight, Mat* dRight,
    Mat* newKLeft, Mat* newKRight);
