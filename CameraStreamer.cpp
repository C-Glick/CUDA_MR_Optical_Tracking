//
// Created by colton-glick on 4/25/26.
//

#include "CameraStreamer.h"
using namespace cv;

CameraStreamer::CameraStreamer(int cameraId)
{
    capture = cv::VideoCapture(cameraId);
    captureThread = std::thread(&CameraStreamer::cameraCaptureThread, this);
}

void CameraStreamer::cameraCaptureThread()
{
    Mat temp_frame;
    while (isRunning)
    {
        if (capture.read(temp_frame))
        {
            if (!temp_frame.empty())
            {
                std::lock_guard<std::mutex> lock(frameMutex);
                latestFrame = temp_frame.clone();
            }
        }
    }
}

void CameraStreamer::StopStream()
{
    isRunning = false;
}

CameraStreamer::~CameraStreamer()
{
    StopStream();
    captureThread.join();
    capture.release();
}
