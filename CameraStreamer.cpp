//
// Created by colton-glick on 4/25/26.
//

#include "CameraStreamer.h"
using namespace cv;

CameraStreamer::CameraStreamer(int cameraId)
{
    capture = cv::VideoCapture(cameraId);
    capture.set(CAP_PROP_BUFFERSIZE, 1);
    capture.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
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
        }else
        {
            std::cerr << "Failed to capture frame" << std::endl;
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
    latestFrame.release();
}
