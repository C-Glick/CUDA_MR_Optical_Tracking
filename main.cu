#include "main.cuh"

//#include <opencv2/opencv.hpp>
#include <fstream>
#include <thread>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include "Cuda_Func.cuh"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "external/nlohmann/json.hpp"

#define ESCAPE_KEY 27
#define SPACE_KEY 32

using namespace cv;
int main(const int argc, char** argv)
{

    int cudaDevices = cuda::getCudaEnabledDeviceCount();
    std::cout << cv::getBuildInformation() << std::endl;

    std::cout << "Number of cuda devices: " << cudaDevices << std::endl;

    if (cudaDevices == 0 )
    {
        std::cerr << "ERROR: The linked version of OpenCV is not compiled with CUDA support,"
                     "or no compatible cuda devices were found." << std::endl;
    }
    if (cudaDevices == -1)
    {
        std::cerr << "ERROR: CUDA driver is not installed or incompatible." << std::endl;
    }


    //openVRTest();
    //openCvImageTest(argv[1]);
    openCvCameraTest();
    //executeGpuTestKernel();

    return 0;
}

void openVRTest()
{

    vr::HmdError peError = vr::HmdError();
    //init open vr
    auto system = vr::VR_Init( &peError, vr::EVRApplicationType::VRApplication_Overlay);

    if (system == nullptr)
    {
        std::cout << peError << std::endl;
    }

    if (peError == vr::VRInitError_None)
    {
        std::cout << "Started OpenVR with no error!" << std::endl;
    }



    //auto OpenVrContext = vr::COpenVRContext();
    auto trackedCamera = vr::VRTrackedCamera();
    bool hasCamera;
    trackedCamera->HasCamera(vr::k_unTrackedDeviceIndex_Hmd, &hasCamera);

    vr::TrackedCameraHandle_t cameraHandle;
    auto image = Mat(2, 2, CV_8UC4);

    vr::EVRTrackedCameraError error;

    vr::HmdVector2_t focalLength;
    vr::HmdVector2_t center;

    error = vr::VRTrackedCamera()->GetCameraIntrinsics(vr::k_unTrackedDeviceIndex_Hmd,
        vr::Eye_Left, vr::VRTrackedCameraFrameType_Undistorted, &focalLength, &center);

    if (!hasCamera)
    {
        std::cout << "Error no camera detected in OpenVR" << std::endl;
    }
    error = trackedCamera->AcquireVideoStreamingService(vr::k_unTrackedDeviceIndex_Hmd, &cameraHandle);
    //size the image buffer
    vr::VRTextureBounds_t texture_bounds;
    uint32_t width = 1920;
    uint32_t height = 960;
    error = vr::VRTrackedCamera()->GetVideoStreamTextureSize(vr::k_unTrackedDeviceIndex_Hmd, vr::VRTrackedCameraFrameType_Distorted, &texture_bounds,
        &width, &height);
    if (error != vr::VRTrackedCameraError_None)
    {
        std::cout << error << std::endl;
        width = 1920;
        height = 960;
    }else
    {
        image.cols = width;
        image.rows = height;
    }



    uint32_t imageByteSize = width * height * image.channels();

    auto start = std::chrono::steady_clock::now();

    while (std::chrono::steady_clock::now() - start < std::chrono::duration<float>(60))
    {
        vr::TrackedDevicePose_t pose;
        system->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseSeated,
            0, &pose, 1 );

        if (hasCamera)
        {

            error = trackedCamera->GetVideoStreamFrameBuffer(cameraHandle, vr::VRTrackedCameraFrameType_Distorted,
                &image.data, imageByteSize, nullptr, 0);

            if (error != vr::VRTrackedCameraError_None)
            {
                std::cout << "error" << std::endl;
            }
        }


        float3 position = float3();
        position.x = pose.mDeviceToAbsoluteTracking.m[0][3];
        position.y = pose.mDeviceToAbsoluteTracking.m[1][3];
        position.z = pose.mDeviceToAbsoluteTracking.m[2][3];

        std::cout << position.x << " " << position.y << " " << position.z << std::endl;
        std::this_thread::sleep_for(std::chrono::duration<float>(0.1f));
    }


    //shutdown openvr
    vr::VR_Shutdown();

}

void openCvImageTest(const std::string& imgPath)
{
    const Mat image = imread(imgPath, IMREAD_COLOR);
    namedWindow("Display Test Image", WINDOW_AUTOSIZE);
    imshow("Display Test Image", image);
    if (!image.data) {
        printf("No image data\n");
        return;
    }

    waitKey(0);
    destroyAllWindows();
}


void saveCameraCalibrationToFile(Mat K, Mat D, Mat newK)
{
    nlohmann::json j;

    //convert OpenCV matrices to 2D arrays which work nicely with json library
    std::vector<std::vector<float>> kArray;
    for (int i = 0; i < K.rows; i++) {
        std::vector<float> row;
        K.row(i).copyTo(row);
        kArray.push_back(row);
    }

    std::vector<std::vector<float>> dArray;
    for (int i = 0; i < D.rows; i++) {
        std::vector<float> row;
        D.row(i).copyTo(row);
        dArray.push_back(row);
    }

    std::vector<std::vector<float>> newKArray;
    for (int i = 0; i < newK.rows; i++) {
        std::vector<float> row;
        newK.row(i).copyTo(row);
        newKArray.push_back(row);
    }

    j["K"] = kArray;
    j["D"] = dArray;
    j["newK"] = newKArray;

    std::ofstream out("camera_calibration.json");
    out << std::setw(4) << j << std::endl;
}


void readCameraCalibrationFromFile()
{

}

void openCvCameraTest()
{
    //TODO allow the index to change so we can select the correct camera(s)
    VideoCapture capture(0);
    Mat image;
    Mat leftImage, rightImage;


    if (capture.isOpened() == false)
    {
        std::cerr << "ERROR: Could not open camera." << std::endl;
    }
    namedWindow("Display Camera Image", WINDOW_AUTOSIZE);
    namedWindow("Left Camera Image", WINDOW_AUTOSIZE);
    namedWindow("Right Camera Image", WINDOW_AUTOSIZE);


    std::vector<cv::Mat> leftCalibrationImages;
    std::vector<cv::Mat> rightCalibrationImages;

    while (true)
    {
        capture >> image;
        leftImage = image(cv::Rect(0,0,image.cols/2,image.rows));
        rightImage = image(cv::Rect(image.cols/2,0,image.cols/2,image.rows));

        if (image.empty())
        {
            std::cerr << "ERROR: blank frame" << std::endl;
            break;
        }

        //imshow("Display Camera Image", image);
        imshow("Left Camera Image", leftImage);
        imshow("Right Camera Image", rightImage);

        int keyPress = waitKey(25);
        if (keyPress == ESCAPE_KEY) // escape key
        {
            //end camera capture
            break;
        }
        else if (keyPress == SPACE_KEY) // space bar
        {
            //save image to calibration array
            cv::Mat leftImageCopy;
            cv::Mat rightImageCopy;
            leftImage.copyTo(leftImageCopy);
            rightImage.copyTo(rightImageCopy);
            leftCalibrationImages.emplace_back(leftImageCopy);
            rightCalibrationImages.emplace_back(rightImageCopy);
        }

    }

    //debug show images
    // for (int i=0; i < leftCalibrationImages.size(); i++)
    // {
    //     std::string name = "Window " + std::to_string(i);
    //
    //     namedWindow(name, WINDOW_AUTOSIZE);
    //     imshow(name, leftCalibrationImages[i]);
    // }
    //
    // while (waitKey(25) == -1)
    // {
    //     for (int i=0; i < leftCalibrationImages.size(); i++)
    //     {
    //         std::string name = "Window " + std::to_string(i);
    //         imshow(name, leftCalibrationImages[i]);
    //     }
    // }

    //process calibration images

    const cv::Size CHECKERBOARD(9, 6);

    // Prepare object points (0,0,0), (1,0,0), ..., (5,8,0)
    std::vector<cv::Point3f> objp;
    for (int i = 0; i < CHECKERBOARD.height; i++)
    {
        for (int j = 0; j < CHECKERBOARD.width; j++)
        {
            objp.emplace_back(j, i, 0);
        }
    }

    std::vector<std::vector<cv::Point2f>> imagePoints; // 2d points in image plane
    std::vector<std::vector<cv::Point3f>> objPoints; // 3d points in world space

    Size imageSize;

    for (int i = 0; i < leftCalibrationImages.size(); i++)
    {
        imageSize = leftCalibrationImages[i].size();
        //convert image to greyscale
        Mat leftGrey;
        cvtColor(leftCalibrationImages[i], leftGrey, COLOR_BGR2GRAY);
        Mat rightGrey;
        cvtColor(rightCalibrationImages[i], rightGrey, COLOR_BGR2GRAY);


        //find the chess border corners
        std::vector<Point2f> corners;
        int result = cv::findChessboardCorners(leftGrey, CHECKERBOARD, corners,
            CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_FAST_CHECK + CALIB_CB_NORMALIZE_IMAGE);
        if (result != 0 && corners.size() == CHECKERBOARD.area())
        {
            //successfully found corners, refine result and add to calibration data
            cornerSubPix(leftGrey, corners, Size(3,3), Size(-1,-1),
                TermCriteria(TermCriteria::Type::EPS + TermCriteria::MAX_ITER, 30, 0.1));
            //place image corners into points array
            imagePoints.push_back(corners);
            //make space for corresponding world space points
            objPoints.push_back(objp);

            drawChessboardCorners(leftGrey, CHECKERBOARD, corners, result);
            imshow("coners", leftGrey);
            waitKey(50000);
        }
    }

    //solve calibration
    Mat K = Mat::zeros(3, 3, CV_64F);
    Mat D = Mat::zeros(4, 1, CV_64F);

    int NumImages = static_cast<int>(objPoints.size());

    std::vector<cv::Mat> rvecs, tvecs;

    double calibration_result = cv::fisheye::calibrate(objPoints, imagePoints, imageSize, K, D, rvecs, tvecs,
        fisheye::CALIB_RECOMPUTE_EXTRINSIC | fisheye::CALIB_CHECK_COND | fisheye::CALIB_FIX_SKEW,
        cv::TermCriteria(
            cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
            30,
            1e-6
        )
        );


    //print results
    std::cout << "Calibration result score (less than 1 = good): " << calibration_result << std::endl;
    std::cout << NumImages << " images used for calibration" << std::endl;
    std::cout << "K = " << K << std::endl;
    std::cout << "D = " << D << std::endl;

    //find the new camera intrinsic matrix for undistortion and rectification
    Mat newK;
    fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, imageSize, Matx33d::eye(), newK, 1.0);

    //use calibration results to undistort an image
    Mat correctedImage;
    fisheye::undistortImage(leftCalibrationImages[0], correctedImage, K, D, newK);

    namedWindow("Corrected Image");
    namedWindow("Live Uncorrected Image");
    namedWindow("Live Corrected Image");
    Mat liveCorrectedImage;
    while (waitKey(10) != ESCAPE_KEY)
    {
        imshow("Corrected Image", correctedImage);

        capture >> image;
        leftImage = image(cv::Rect(0,0,image.cols/2,image.rows));
        rightImage = image(cv::Rect(image.cols/2,0,image.cols/2,image.rows));

        imshow("Live Uncorrected Image", leftImage);
        fisheye::undistortImage(leftImage, liveCorrectedImage, K, D, newK);

        imshow("Live Corrected Image", liveCorrectedImage);
    }

    saveCameraCalibrationToFile(K, D, newK);

    capture.release();
    destroyAllWindows();
}

void executeGpuTestKernel()
{
    int arraySize = 5000;

    //allocate memory on CPU
    auto CpuIn1 = (float*) malloc(sizeof(float) * arraySize);
    auto CpuIn2 = (float*) malloc(sizeof(float) * arraySize);
    auto CpuResult = (float*) malloc(sizeof(float) * arraySize);

    //populate values.
    srand(time(NULL));
    for(int i=0; i < arraySize; i++){
        CpuIn1[i] = (rand() / float(RAND_MAX)) * 1000.0f;
        CpuIn2[i] = (rand() / float(RAND_MAX)) * 1000.0f;
    }

    //allocate memory on gpu
    float *GpuIn1;
    float *GpuIn2;
    float *GpuResult;

    cudaMalloc((void**)&GpuIn1, sizeof(float) * arraySize);
    cudaMalloc((void**)&GpuIn2, sizeof(float) * arraySize);
    cudaMalloc((void**)&GpuResult, sizeof(float) * arraySize);

    //copy data into GPU memory
    cudaMemcpy(GpuIn1, CpuIn1, sizeof(float) * arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(GpuIn2, CpuIn2, sizeof(float) * arraySize, cudaMemcpyHostToDevice);

    GpuKernelVectorAdd<<<256, 256, 0>>>(GpuIn1, GpuIn2, GpuResult);

    //copy results from gpu to cpu memory
    cudaMemcpy(CpuResult, GpuResult, sizeof(float) * arraySize, cudaMemcpyDeviceToHost);


    //print out results
    for (int i=0; i < arraySize; i++)
    {
        printf("%f + %f = %f \n", CpuIn1[i], CpuIn2[i], CpuResult[i]);
    }


    //free memory from GPU
    cudaFree(GpuIn1);
    cudaFree(GpuIn2);
    cudaFree(GpuResult);

    //free memory from CPU
    free(CpuIn1);
    free(CpuIn2);
    free(CpuResult);
}
