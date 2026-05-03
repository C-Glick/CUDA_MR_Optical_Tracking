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

/**
 * Main routine of OpenCV to display camera image, correct camera distortion, and find and track
 * aruco markers
 */
void openCvCameraRoutine();

/**
 * Initialize OpenVR systems
 */
void initOpenVR();
/**
 * Shutdown OpenVR systems
 */
void shutdownOpenVR();
/**
 * Capture the headset's camera feed through OpenVR systems. Note: This is currently
 * only implemented in Windows by SteamVR.
 */
void openVRCameraCapture();
/**
 * Returns the tracked pose of the head from SteamVR. Use this to translate
 * aruco tracked items from camera coordinates to world coordinates
 * @param pose Headpose from OpenVR
 */
void getHeadsetPose(vr::TrackedDevicePose_t pose);

/**
 * Save the provided camera parameters to file
 *
 */
void saveCameraCalibrationToFile(Mat* kLeft, Mat* dLeft, Mat* kRight, Mat* dRight,
                                 Mat* newKLeft, Mat* newKRight, Mat* stereoCamTranslation, Mat* stereoCamRotation, std::vector<cv::Mat>* calibrationImages);

/**
 * Read the camera calibration from file. Fills out the parameters passed in
 */
void readCameraCalibrationFromFile(Mat* kLeft, Mat* dLeft, Mat* kRight, Mat* dRight,
                                   Mat* newKLeft, Mat* newKRight, Mat* stereoCamTranslation, Mat* stereoCamRotation);

/**
 * Verifies that the camera calibration file was successfully saved to disk by
 * reading it back and checking against values in memory.
 * @return
 */
bool verifySavedCalibration(Mat* kLeft, Mat* dLeft, Mat* kRight, Mat* dRight,
                            Mat* newKLeft, Mat* newKRight,  Mat* stereoCamTranslation, Mat* stereoCamRotation);

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
