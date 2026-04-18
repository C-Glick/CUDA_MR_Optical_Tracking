#include "main.cuh"

//#include <opencv2/opencv.hpp>
#include <fstream>
#include <thread>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudacodec.hpp>
#include "Cuda_Func.cuh"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <format>

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

    cuda::setDevice(0);

#ifndef HAVE_OPENCV_CUDACODEC
    cout << "OpenCV was built without CUDA Video decoding support\n" << std::endl;
    exit(1);
#endif

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

std::vector<std::vector<double>> copyMatTo2dVector(const Mat* mat)
{
    auto vect = std::vector<std::vector<double>>();
    for (int i = 0; i < mat->rows; i++) {
        std::vector<double> row;
        mat->row(i).copyTo(row);
        vect.push_back(row);
    }
    return vect;
}

Mat copy2dVectorToMat(const std::vector<std::vector<double>>& vect)
{
    Mat mat(vect.size(), vect[0].size(), CV_64FC1);
    for (size_t i = 0; i < vect.size(); i++) {
        std::memcpy(mat.ptr<double>(i), vect[i].data(), vect[i].size() * sizeof(double));
    }
    return mat;
}



void saveCameraCalibrationToFile(Mat* kLeft, Mat* dLeft, Mat* kRight, Mat* dRight,
    Mat* newKLeft, Mat* newKRight, std::vector<cv::Mat>* calibrationImages)
{
    json j;
    //convert OpenCV matrices to 2D arrays which work nicely with json library
    auto kLeftArray = copyMatTo2dVector(kLeft);
    auto dLeftArray = copyMatTo2dVector(dLeft);

    auto kRightArray = copyMatTo2dVector(kRight);
    auto dRightArray = copyMatTo2dVector(dRight);

    auto newKLeftArray = copyMatTo2dVector(newKLeft);
    auto newKRightArray = copyMatTo2dVector(newKRight);

    j["kLeft"] = kLeftArray;
    j["dLeft"] = dLeftArray;
    j["kRight"] = kRightArray;
    j["dRight"] = dRightArray;
    j["newKLeft"] = newKLeftArray;
    j["newKRight"] = newKRightArray;

    std::filesystem::create_directories(CALIBRATION_PATH);
    std::ofstream out((std::string)CALIBRATION_PATH + "/" + CALIBRATION_FILE);
    if (!out.is_open())
    {
        std::cerr << "Failed to open file " << CALIBRATION_PATH << "/" << CALIBRATION_FILE<< std::endl;
    }
    out << std::setw(4) << j << std::endl;
    out.close();


    std::cout << "Wrote to disk, reading back to verify data integrity " << CALIBRATION_PATH << "/" << CALIBRATION_FILE << std::endl;

    if (verifySavedCalibration(kLeft, dLeft, kRight, dRight, newKLeft, newKRight))
    {
        std::cout << "Calibration successfully saved to disk " << CALIBRATION_PATH << "/" << CALIBRATION_FILE << std::endl;
    }else
    {
        std::cerr << "ERROR: Calibration failed to save to disk " << CALIBRATION_PATH << "/" << CALIBRATION_FILE << std::endl;
    }

    if (calibrationImages != nullptr)
    {
        std::cout << "Saving images to disk ..." << std::endl;

        int i=0;
        for (Mat image : *calibrationImages)
        {
            std::string fileName = std::string(CALIBRATION_PATH) + "/" + CALIBRATION_IMAGE_FILE + std::to_string(i) + ".png";
            std::vector<int> compression_params;
            compression_params.push_back(IMWRITE_PNG_COMPRESSION);
            compression_params.push_back(6);

            imwrite(fileName,image, compression_params);
            i++;
        }

        std::cout << "Raw calibration images saved to disk " << CALIBRATION_PATH << std::endl;
    }
}

bool verifySavedCalibration(Mat* kLeft, Mat* dLeft, Mat* kRight, Mat* dRight,
    Mat* newKLeft, Mat* newKRight)
{
    bool result = true;
    //read back file to make sure save was successful
    Mat kLeftCheck, dLeftCheck,  kRightCheck, dRightCheck, newKLeftCheck, newKRightCheck;
    readCameraCalibrationFromFile(&kLeftCheck, &dLeftCheck,  &kRightCheck, &dRightCheck, &newKLeftCheck, &newKRightCheck);

    Mat xorResult;
    bitwise_xor(kLeftCheck, *kLeft, xorResult);
    result = result && countNonZero(xorResult) == 0;
    bitwise_xor(dLeftCheck, *dLeft, xorResult);
    result = result && countNonZero(xorResult) == 0;
    bitwise_xor(kRightCheck, *kRight, xorResult);
    result = result && countNonZero(xorResult) == 0;
    bitwise_xor(dRightCheck, *dRight, xorResult);
    result = result && countNonZero(xorResult) == 0;
    bitwise_xor(newKLeftCheck, *newKLeft, xorResult);
    result = result && countNonZero(xorResult) == 0;
    bitwise_xor(newKRightCheck, *newKRight, xorResult);
    result = result && countNonZero(xorResult) == 0;

    return result;
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
    *kLeft = copy2dVectorToMat(kLeftArray);
    *dLeft = copy2dVectorToMat(dLeftArray);
    *kRight = copy2dVectorToMat(kRightArray);
    *dRight = copy2dVectorToMat(dRightArray);
    *newKLeft = copy2dVectorToMat(newKLeftArray);
    *newKRight = copy2dVectorToMat(newKRightArray);

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
    bool loadCalibrationImagesFromFile = false;
    if (std::filesystem::exists(CALIBRATION_PATH))
    {
        std::cout << "Load calibration images from disk? " << CALIBRATION_PATH << " (Y/N)" <<  std::endl;
        std::string response;
        std::cin >> response;
        loadCalibrationImagesFromFile = (response == "y" || response == "Y");
    }

    if (loadCalibrationImagesFromFile)
    {
        Mat image;
        Mat leftImage, rightImage;

        for (const auto& entry : std::filesystem::directory_iterator(CALIBRATION_PATH)) {
            // Check if the entry is a regular file and has the correct extension
            if (entry.is_regular_file() && entry.path().extension() == ".png") {
                imread(entry.path(), image);
                //break image in half for left and right eye
                leftImage = image(cv::Rect(0,0,image.cols/2,image.rows));
                rightImage = image(cv::Rect(image.cols/2,0,image.cols/2,image.rows));
                //save image to calibration array
                leftCalibrationImages.emplace_back(leftImage.clone());
                rightCalibrationImages.emplace_back(rightImage.clone());
                combinedCalibrationImages.emplace_back(image.clone());
            }
        }
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


        // cuda::GpuMat device_image;
        // Ptr<cudacodec::VideoReader> device_capture = cudacodec::
        if (capture.isOpened() == false)
        {
            std::cerr << "ERROR: Could not open camera." << std::endl;
        }
        namedWindow("Display Camera Image", WINDOW_NORMAL | WINDOW_OPENGL);
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

                std::cout << "Captured calibration frame #" << std::to_string(combinedCalibrationImages.size() - 1) << std::endl;
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

    destroyAllWindows();

    std::cout << "Calibration complete. Save to disk? (Y/N)" << std::endl;
    std::string response;
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


}

void openCvCameraTest()
{
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


    VideoCapture capture(0);
    UMat image;
    UMat leftImage, rightImage;
    if (capture.isOpened() == false)
    {
        std::cerr << "ERROR: Could not open camera." << std::endl;
    }

    //use calibration results to undistort live camera feed
    UMat correctedLeftImage;
    UMat correctedRightImage;

    namedWindow("Corrected Left Image");
    namedWindow("Corrected Right Image");
    while (waitKey(16) != ESCAPE_KEY)
    {
        capture >> image;
        leftImage = image(cv::Rect(0,0,image.cols/2,image.rows));
        fisheye::undistortImage(leftImage, correctedLeftImage, camCalKLeft, camCalDLeft, camCalNewKLeft);

        rightImage = image(cv::Rect(image.cols/2,0,image.cols/2,image.rows));
        fisheye::undistortImage(rightImage, correctedRightImage, camCalKRight, camCalDRight, camCalNewKRight);

        imshow("Corrected Left Image", correctedLeftImage);
        imshow("Corrected Right Image", correctedRightImage);
    }

    capture.release();
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
