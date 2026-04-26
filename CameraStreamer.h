//
// Created by colton-glick on 4/25/26.
//

#ifndef CUDA_MR_OPTICAL_TRACKING_CAMERASTREAMER_H
#define CUDA_MR_OPTICAL_TRACKING_CAMERASTREAMER_H

#include <atomic>
#include <thread>
#include <opencv2/opencv.hpp>

class CameraStreamer
{
public:
    cv::Mat latestFrame;
    std::mutex frameMutex;
    std::atomic<bool> isRunning;
    std::thread captureThread;

    CameraStreamer(int cameraID);

    void StopStream();
    ~CameraStreamer();

    /**
     * Thread safe Blocking call that will get the next nonempty frame
     * @param image
     */
    void getFrame(cv::Mat* image);

    /**
      * Thread safe NON Blocking call that will attempt to get the latest frame
      * @param image
      */
    bool tryGetFrame(cv::Mat* image);

private:
    cv::VideoCapture capture;
    void cameraCaptureThread();

};


#endif //CUDA_MR_OPTICAL_TRACKING_CAMERASTREAMER_H
