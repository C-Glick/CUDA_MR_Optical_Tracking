#include "main.cuh"

//#include <opencv2/opencv.hpp>
#include <fstream>
#include <thread>
#include <string>
#include <format>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include "Cuda_Func.cuh"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "external/nlohmann/json.hpp"
using json = nlohmann::json;

#define ESCAPE_KEY 27
#define SPACE_KEY 32

#define CALIBRATION_PATH "./camera_calibration"
#define CALIBRATION_FILE "calibration_params.json"
#define CALIBRATION_IMAGE_FILE "calibration_image_" //calibration_image_001.png

const cv::Size CHECKERBOARD(9, 6); // calibration pattern checkerboard size

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


void copyMatTo2dVector(const Mat* mat, std::vector<std::vector<float>>* vect)
{
    for (int i = 0; i < mat->rows; i++) {
        std::vector<float> row;
        mat->row(i).copyTo(row);
        vect->push_back(row);
    }
}

void saveCameraCalibrationToFile(Mat* kLeft, Mat* dLeft, Mat* kRight, Mat* dRight,
    Mat* newKLeft, Mat* newKRight, std::vector<cv::Mat>* calibrationImages)
{
    json j;
    //convert OpenCV matrices to 2D arrays which work nicely with json library
    std::vector<std::vector<float>> kLeftArray;
    copyMatTo2dVector(kLeft, &kLeftArray);
    std::vector<std::vector<float>> dLeftArray;
    copyMatTo2dVector(dLeft, &dLeftArray);

    std::vector<std::vector<float>> kRightArray;
    copyMatTo2dVector(kRight, &kRightArray);
    std::vector<std::vector<float>> dRightArray;
    copyMatTo2dVector(dRight, &dRightArray);

    std::vector<std::vector<float>> newKLeftArray;
    copyMatTo2dVector(newKLeft, &newKLeftArray);
    std::vector<std::vector<float>> newKRightArray;
    copyMatTo2dVector(newKRight, &newKRightArray);

    j["kLeft"] = kLeftArray;
    j["dLeft"] = dLeftArray;
    j["kRight"] = kRightArray;
    j["dRight"] = dRightArray;
    j["newKLeft"] = newKLeftArray;
    j["newKRight"] = newKRightArray;

    std::ofstream out((std::string)CALIBRATION_PATH + "/" + CALIBRATION_FILE);
    out << std::setw(4) << j << std::endl;

    std::cout << "Calibration saved to disk " << CALIBRATION_PATH << "/" << CALIBRATION_FILE << std::endl;

    if (calibrationImages != nullptr)
    {
        int i=0;
        for (Mat image : *calibrationImages)
        {
            std::string fileName = format((std::string(CALIBRATION_PATH) + "/" + CALIBRATION_IMAGE_FILE + "{:03}" + ".png").c_str(), i);
            std::vector<int> compression_params;
            compression_params.push_back(IMWRITE_PNG_COMPRESSION);
            compression_params.push_back(6);

            imwrite(fileName,image, compression_params);
        }

        std::cout << "Raw calibration images saved to disk " << CALIBRATION_PATH << std::endl;
    }
}


void readCameraCalibrationFromFile(Mat* kLeft, Mat* dLeft, Mat* kRight, Mat* dRight,
    Mat* newKLeft, Mat* newKRight)
{
    std::ifstream inFile((std::string)CALIBRATION_PATH + "/" + CALIBRATION_FILE);
    json jData = json::parse(inFile);

    std::vector<std::vector<double>> kLeftArray = jData["kLeft"];
    std::vector<std::vector<double>> dLeftArray = jData["dLeft"];
    std::vector<std::vector<double>> kRightArray = jData["kRight"];
    std::vector<std::vector<double>> dRightArray = jData["dRight"];
    std::vector<std::vector<double>> newKLeftArray = jData["newKLeft"];
    std::vector<std::vector<double>> newKRightArray = jData["newKRight"];

    //convert to OpenCV matrix
    *kLeft = Mat(kLeftArray.size(), kLeftArray[0].size(), CV_64F, kLeftArray.data());
    *dLeft = Mat(dLeftArray.size(), dLeftArray[0].size(), CV_64F, dLeftArray.data());
    *kRight = Mat(kRightArray.size(), kRightArray[0].size(), CV_64F, kRightArray.data());
    *dRight = Mat(dRightArray.size(), dRightArray[0].size(), CV_64F, dRightArray.data());
    *newKLeft = Mat(newKLeftArray.size(), newKLeftArray[0].size(), CV_64F, newKLeftArray.data());
    *newKRight = Mat(newKRightArray.size(), newKRightArray[0].size(), CV_64F, newKRightArray.data());

    std::cout << "Calibration read from disk " << CALIBRATION_PATH << "/" << CALIBRATION_FILE << std::endl;
}

bool checkForCameraCalibration()
{
    std::ifstream inFile((std::string)CALIBRATION_PATH + "/" + CALIBRATION_FILE);
    bool result = inFile.is_open();
    inFile.close();
    return result;
}


void calibrateCameras(Mat* kLeft, Mat* dLeft, Mat* kRight, Mat* dRight, Mat* newKLeft, Mat* newKRight)
{
    std::vector<cv::Mat> leftCalibrationImages;
    std::vector<cv::Mat> rightCalibrationImages;
    std::vector<cv::Mat> combinedCalibrationImages;


    std::cout << "Calibrating cameras ..." << std::endl;
    std::cout << "Load calibration images from disk? " << CALIBRATION_PATH << " (Y/N)" <<  std::endl;
    bool loadCalibrationFromFile = false;
    std::string response;
    std::cin >> response;
    loadCalibrationFromFile = (response == "y" || response == "Y");

    if (loadCalibrationFromFile)
    {

    }
    else
    {
        std::cout << "Starting camera ..." << std::endl;
        //todo can we do a partially obscure calibration pattern
        std::cout << "Hold calibration sheet in full view of both cameras." << std::endl;
        std::cout << "Press <space bar> to capture calibration frame." << std::endl;
        std::cout << "Move calibration sheet to different areas of the camera and tilt." << std:: endl;
        std::cout << "Capture between 10 and 30 calibration images" << std::endl;
        std::cout << "Press <ESC> once finished." << std::endl;

        //start up headset camera and capture calibration images
        //TODO allow the index to change so we can select the correct camera(s)
        VideoCapture capture(0);
        Mat image;
        Mat leftImage, rightImage;
        if (capture.isOpened() == false)
        {
            std::cerr << "ERROR: Could not open camera." << std::endl;
        }
        namedWindow("Display Camera Image", WINDOW_AUTOSIZE);

        while (true)
        {
            capture >> image;
            //break image in half for left and right eye
            leftImage = image(cv::Rect(0,0,image.cols/2,image.rows));
            rightImage = image(cv::Rect(image.cols/2,0,image.cols/2,image.rows));

            if (image.empty())
            {
                std::cerr << "ERROR: blank frame" << std::endl;
                continue;
            }

            imshow("Display Camera Image", image);

            int keyPress = waitKey(25);
            if (keyPress == ESCAPE_KEY) // escape key
            {
                //end camera capture
                break;
            }
            else if (keyPress == SPACE_KEY) // space bar
            {
                //save image to calibration array
                leftCalibrationImages.emplace_back(leftImage.clone());
                rightCalibrationImages.emplace_back(rightImage.clone());
                combinedCalibrationImages.emplace_back(image.clone());
            }
        }
        capture.release();
    }

    //process calibration images

    // Prepare object points (0,0,0), (1,0,0), ..., (5,8,0) These are the calibration points in the xy plane of the calibration sheet
    std::vector<cv::Point3f> objp;
    for (int i = 0; i < CHECKERBOARD.height; i++)
    {
        for (int j = 0; j < CHECKERBOARD.width; j++)
        {
            objp.emplace_back(j, i, 0);
        }
    }

    std::vector<std::vector<cv::Point2f>> imagePointsLeft; // 2d points in image plane of the calibration points
    std::vector<std::vector<cv::Point2f>> imagePointsRight;
    std::vector<std::vector<cv::Point3f>> objPoints; // 3d points in world space

    Size imageSize;

    for (int i = 0; i < combinedCalibrationImages.size(); i++)
    {
        imageSize = leftCalibrationImages[i].size();
        //convert images to greyscale
        Mat leftGrey;
        cvtColor(leftCalibrationImages[i], leftGrey, COLOR_BGR2GRAY);
        Mat rightGrey;
        cvtColor(rightCalibrationImages[i], rightGrey, COLOR_BGR2GRAY);

        //find the chess border corners
        std::vector<Point2f> cornersLeft;
        int cornersResultLeft = cv::findChessboardCorners(leftGrey, CHECKERBOARD, cornersLeft,
            CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_FAST_CHECK + CALIB_CB_NORMALIZE_IMAGE);
        std::vector<Point2f> cornersRight;
        int cornersResultRight = cv::findChessboardCorners(rightGrey, CHECKERBOARD, cornersRight,
            CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_FAST_CHECK + CALIB_CB_NORMALIZE_IMAGE);

        //TODO only accepting images that have the pattern fully visible in both camera images, do a partial calibration
        if (cornersResultLeft != 0 && cornersLeft.size() == CHECKERBOARD.area() &&
            cornersResultRight != 0 && cornersRight.size() == CHECKERBOARD.area())
        {
            //successfully found corners, refine result and add to calibration data
            cornerSubPix(leftGrey, cornersLeft, Size(3,3), Size(-1,-1),
                TermCriteria(TermCriteria::Type::EPS + TermCriteria::MAX_ITER, 30, 0.1));
            cornerSubPix(rightGrey, cornersRight, Size(3,3), Size(-1,-1),
                TermCriteria(TermCriteria::Type::EPS + TermCriteria::MAX_ITER, 30, 0.1));

            //place image corners into points array
            imagePointsLeft.push_back(cornersLeft);
            imagePointsRight.push_back(cornersRight);

            //allocate corresponding world space points
            objPoints.push_back(objp);

            //todo debug enable to show images
            drawChessboardCorners(leftGrey, CHECKERBOARD, cornersLeft, cornersResultLeft);
            drawChessboardCorners(rightGrey, CHECKERBOARD, cornersRight, cornersResultRight);

            imshow("Left image", leftGrey);
            imshow("Right image", rightGrey);

            waitKey(50000);
        }
    }

    //solve calibration
    *kLeft = Mat::zeros(3, 3, CV_64F);
    *kRight = Mat::zeros(3, 3, CV_64F);

    *dLeft = Mat::zeros(4, 1, CV_64F);
    *dRight = Mat::zeros(4, 1, CV_64F);

    int NumImages = static_cast<int>(objPoints.size());

    std::vector<cv::Mat> rvecs, tvecs;
    Mat stereoCameraRotation;
    Mat stereoCameraTranslation;

    // double calibration_result = cv::fisheye::calibrate(objPoints, imagePointsLeft, imageSize, K, D, rvecs, tvecs,
    //     fisheye::CALIB_RECOMPUTE_EXTRINSIC | fisheye::CALIB_CHECK_COND | fisheye::CALIB_FIX_SKEW,
    //     cv::TermCriteria(
    //         cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
    //         30,
    //         1e-6
    //     )
    //     );

    double calibration_result = cv::fisheye::stereoCalibrate(objPoints, imagePointsLeft, imagePointsRight,
        *kLeft, *dLeft, *kRight, *dRight, imageSize, stereoCameraRotation, stereoCameraTranslation, rvecs, tvecs,
        fisheye::CALIB_RECOMPUTE_EXTRINSIC | fisheye::CALIB_CHECK_COND | fisheye::CALIB_FIX_SKEW,
        cv::TermCriteria(
           cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
           30,
           1e-6
       )
       );

    //find the camera's new intrinsic matrix for undistortion and rectification
    fisheye::estimateNewCameraMatrixForUndistortRectify(*kLeft, *dLeft, imageSize, Matx33d::eye(), *newKLeft, 1.0);
    fisheye::estimateNewCameraMatrixForUndistortRectify(*kRight, *dRight, imageSize, Matx33d::eye(), *newKRight, 1.0);

    //print results
    std::cout << "Calibration result score (less than 1 = good): " << calibration_result << std::endl;
    std::cout << NumImages << " images used for calibration" << std::endl;
    std::cout << "K Left = " << *kLeft << std::endl;
    std::cout << "D Left = " << *dLeft << std::endl << std::endl;

    std::cout << "K Right = " << *kRight << std::endl;
    std::cout << "D Right = " << *dRight << std::endl << std::endl;

    std::cout << "Stereo camera rotation = " << stereoCameraRotation << std::endl;
    std::cout << "Stereo camera translation = " << stereoCameraTranslation << std::endl << std::endl;

    std::cout << "New K left camera intrinsics = " << *newKLeft << std::endl;
    std::cout << "New K right camera intrinsics = " << *newKRight << std::endl;


    std::cout << "Calibration complete. Save to disk? (Y/N)" << std::endl;
    std::cin >> response;

    if (response == "Y" || response == "y")
    {
        std::cout << "Include raw calibration images? (Y/N)" << std::endl;
        std::cin >> response;
        if (response == "Y" || response == "y")
        {
            saveCameraCalibrationToFile(kLeft, dLeft, kRight, dRight, newKLeft, newKRight, &combinedCalibrationImages);
        }else
        {
            saveCameraCalibrationToFile(kLeft, dLeft, kRight, dRight, newKLeft, newKRight, nullptr);
        }
    }


    destroyAllWindows();
}

void openCvCameraTest()
{

    //todo mechanise to use previous values.
    Mat camCalKLeft;
    Mat camCalDLeft;

    Mat camCalKRight;
    Mat camCalDRight;

    Mat camCalNewKLeft;
    Mat camCalNewKRight;

    bool doCameraCalibration = true;

    if (checkForCameraCalibration())
    {
        std::cout << "Camera calibration file found. Would you like to load calibration parameters from disk? (y/n)" << std::endl;
        std::string response;
        std::cin >> response;
        if (response == "y" || response == "Y")
        {
            doCameraCalibration = false;
        }else
        {
            doCameraCalibration = true;
        }
    }

    if (doCameraCalibration)
    {
        calibrateCameras(&camCalKLeft, &camCalDLeft, &camCalKRight, &camCalDRight,
            &camCalNewKLeft, &camCalNewKRight);
    }else
    {
        readCameraCalibrationFromFile(&camCalKLeft, &camCalDLeft, &camCalKRight, &camCalDRight,
            &camCalNewKLeft, &camCalNewKRight);
    }


    // VideoCapture capture(0);
    // Mat image;
    // Mat leftImage, rightImage;
    // if (capture.isOpened() == false)
    // {
    //     std::cerr << "ERROR: Could not open camera." << std::endl;
    // }
    // namedWindow("Display Camera Image", WINDOW_AUTOSIZE);
    //
    // while (true)
    // {
    //     capture >> image;

    //use calibration results to undistort an image
    // Mat correctedImage;
    // fisheye::undistortImage(leftCalibrationImages[0], correctedImage, K, D, newK);
    //
    // namedWindow("Corrected Image");
    // namedWindow("Live Uncorrected Image");
    // namedWindow("Live Corrected Image");
    // Mat liveCorrectedImage;
    // while (waitKey(10) != ESCAPE_KEY)
    // {
    //     imshow("Corrected Image", correctedImage);
    //
    //     capture >> image;
    //     leftImage = image(cv::Rect(0,0,image.cols/2,image.rows));
    //     rightImage = image(cv::Rect(image.cols/2,0,image.cols/2,image.rows));
    //
    //     imshow("Live Uncorrected Image", leftImage);
    //     fisheye::undistortImage(leftImage, liveCorrectedImage, K, D, newK);
    //
    //     imshow("Live Corrected Image", liveCorrectedImage);
    // }


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
