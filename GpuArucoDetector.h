//
// Created by colton-glick on 4/28/26.
//

#ifndef CUDA_MR_OPTICAL_TRACKING_GPUARUCODETECTOR_H
#define CUDA_MR_OPTICAL_TRACKING_GPUARUCODETECTOR_H
#include "opencv2/objdetect/aruco_detector.hpp"

using namespace cv;

class GpuArucoDetector : public cv::aruco::ArucoDetector
{

public:
    /**
     * Overrides the standard CV ArucoDetector detect markers, to use the gpu for adaptive threshold processing
     * @param image
     * @param corners
     * @param ids
     * @param rejectedImgPoints
     */
    void detectMarkers(InputArray image, OutputArrayOfArrays corners, OutputArray ids,
                       OutputArrayOfArrays rejectedImgPoints = noArray()) const;
private:


};


#endif //CUDA_MR_OPTICAL_TRACKING_GPUARUCODETECTOR_H
