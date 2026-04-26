#include "main.cuh"
#include "CameraStreamer.h"

//#include <opencv2/opencv.hpp>
#include <fstream>
#include <thread>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudaimgproc.hpp>
#include "Cuda_Func.cuh"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <format>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "external/nlohmann/json.hpp"
using json = nlohmann::json;

#define ESCAPE_KEY 27
#define SPACE_KEY 32

#define CALIBRATION_PATH "./camera_calibration"
#define CALIBRATION_FILE "calibration_params.json"
#define CALIBRATION_IMAGE_FILE "calibration_image_" //calibration_image_001.png

const cv::Size CHECKERBOARD(9, 6); // calibration pattern checkerboard size
float g_markerLength = 0.024f; //2.4 cm size marker side length
std::vector<Point3d> g_MarkerObjPoints(4);

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
    std::cout << "OpenCV was built without CUDA Video decoding support\n" << std::endl;
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

void onOpenGlDraw(void* param) {
    cv::ogl::Texture2D* tex = (cv::ogl::Texture2D*)param;
    // Enable texturing and bind the texture object
    //glEnable(GL_TEXTURE_2D);
    tex->bind();

    // Use OpenCV's helper to render the texture to the current window
    // This draws a screen-aligned quad by default
    cv::ogl::render(*tex);
}

void openCvImageTest(const std::string& imgPath)
{
    const Mat image = imread(imgPath, IMREAD_COLOR);
    namedWindow("OpenGL Display Test Image", WINDOW_OPENGL);
    ogl::Texture2D openGlTexture;
    openGlTexture.copyFrom(image);
    setOpenGlDrawCallback("OpenGL Display Test Image", onOpenGlDraw, &openGlTexture);
    //imshow("OpenGL Display Test Image", image);
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
    Mat* newKLeft, Mat* newKRight, Mat* stereoCamTranslation, Mat* stereoCamRotation,
    std::vector<cv::Mat>* calibrationImages)
{
    json j;
    //convert OpenCV matrices to 2D arrays which work nicely with json library
    auto kLeftArray = copyMatTo2dVector(kLeft);
    auto dLeftArray = copyMatTo2dVector(dLeft);

    auto kRightArray = copyMatTo2dVector(kRight);
    auto dRightArray = copyMatTo2dVector(dRight);

    auto newKLeftArray = copyMatTo2dVector(newKLeft);
    auto newKRightArray = copyMatTo2dVector(newKRight);

    auto stereoCamTranslationArray = copyMatTo2dVector(stereoCamTranslation);
    auto stereoCamRotationArray = copyMatTo2dVector(stereoCamRotation);


    j["kLeft"] = kLeftArray;
    j["dLeft"] = dLeftArray;
    j["kRight"] = kRightArray;
    j["dRight"] = dRightArray;
    j["newKLeft"] = newKLeftArray;
    j["newKRight"] = newKRightArray;
    j["stereoCamTranslation"] = stereoCamTranslationArray;
    j["stereoCamRotation"] = stereoCamRotationArray;

    std::filesystem::create_directories(CALIBRATION_PATH);
    std::ofstream out((std::string)CALIBRATION_PATH + "/" + CALIBRATION_FILE);
    if (!out.is_open())
    {
        std::cerr << "Failed to open file " << CALIBRATION_PATH << "/" << CALIBRATION_FILE<< std::endl;
    }
    out << std::setw(4) << j << std::endl;
    out.close();


    std::cout << "Wrote to disk, reading back to verify data integrity " << CALIBRATION_PATH << "/" << CALIBRATION_FILE << std::endl;

    if (verifySavedCalibration(kLeft, dLeft, kRight, dRight, newKLeft, newKRight, stereoCamTranslation,
        stereoCamRotation))
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
    Mat* newKLeft, Mat* newKRight,  Mat* stereoCamTranslation, Mat* stereoCamRotation)
{
    bool result = true;
    //read back file to make sure save was successful
    Mat kLeftCheck, dLeftCheck,  kRightCheck, dRightCheck, newKLeftCheck, newKRightCheck,
    stereoCamTranslationCheck, stereoCamRotationCheck;
    readCameraCalibrationFromFile(&kLeftCheck, &dLeftCheck,  &kRightCheck, &dRightCheck,
        &newKLeftCheck, &newKRightCheck, &stereoCamTranslationCheck, &stereoCamRotationCheck);

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
    bitwise_xor(stereoCamTranslationCheck, *stereoCamTranslation, xorResult);
    result = result && countNonZero(xorResult) == 0;
    bitwise_xor(stereoCamRotationCheck, *stereoCamRotation, xorResult);
    result = result && countNonZero(xorResult) == 0;


    return result;
}

void readCameraCalibrationFromFile(Mat* kLeft, Mat* dLeft, Mat* kRight, Mat* dRight,
    Mat* newKLeft, Mat* newKRight, Mat* stereoCamTranslation, Mat* stereoCamRotation)
{
    std::ifstream inFile((std::string)CALIBRATION_PATH + "/" + CALIBRATION_FILE);
    json jData = json::parse(inFile);

    std::vector<std::vector<double>> kLeftArray = jData["kLeft"];
    std::vector<std::vector<double>> dLeftArray = jData["dLeft"];
    std::vector<std::vector<double>> kRightArray = jData["kRight"];
    std::vector<std::vector<double>> dRightArray = jData["dRight"];
    std::vector<std::vector<double>> newKLeftArray = jData["newKLeft"];
    std::vector<std::vector<double>> newKRightArray = jData["newKRight"];
    std::vector<std::vector<double>> stereoCamTranslationArray = jData["stereoCamTranslation"];
    std::vector<std::vector<double>> stereoCamRotationArray = jData["stereoCamRotation"];


    //convert to OpenCV matrix
    *kLeft = copy2dVectorToMat(kLeftArray);
    *dLeft = copy2dVectorToMat(dLeftArray);
    *kRight = copy2dVectorToMat(kRightArray);
    *dRight = copy2dVectorToMat(dRightArray);
    *newKLeft = copy2dVectorToMat(newKLeftArray);
    *newKRight = copy2dVectorToMat(newKRightArray);
    *stereoCamTranslation = copy2dVectorToMat(stereoCamTranslationArray);
    *stereoCamRotation = copy2dVectorToMat(stereoCamRotationArray);

    std::cout << "Calibration read from disk " << CALIBRATION_PATH << "/" << CALIBRATION_FILE << std::endl;
}

bool checkForCameraCalibration()
{
    std::ifstream inFile((std::string)CALIBRATION_PATH + "/" + CALIBRATION_FILE);
    bool result = inFile.is_open();
    inFile.close();
    return result;
}


void calibrateCameras(Mat* kLeft, Mat* dLeft, Mat* kRight, Mat* dRight, Mat* newKLeft, Mat* newKRight,
    Mat* stereoCamTranslation, Mat* stereoCamRotation)
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
        CameraStreamer camStreamer = CameraStreamer(0);
        Mat image;
        Mat leftImage, rightImage;



        namedWindow("Display Camera Image", WINDOW_NORMAL);
        resizeWindow("Display Camera Image", 1920, 960);
        while (true)
        {
            if (!camStreamer.tryGetFrame(&image))
            {
                // camera not ready yet
                continue;
            }
            //break image in half for left and right eye
            leftImage = image(cv::Rect(0,0,image.cols/2,image.rows));
            rightImage = image(cv::Rect(image.cols/2,0,image.cols/2,image.rows));

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

    double calibration_result = cv::fisheye::stereoCalibrate(objPoints, imagePointsLeft, imagePointsRight,
        *kLeft, *dLeft, *kRight, *dRight, imageSize, *stereoCamRotation, *stereoCamTranslation, rvecs, tvecs,
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

    //TODO
    //fisheye::stereoRectify()

    //print results
    std::cout << NumImages << " images used for calibration" << std::endl;
    std::cout << "K Left = " << *kLeft << std::endl;
    std::cout << "D Left = " << *dLeft << std::endl << std::endl;

    std::cout << "K Right = " << *kRight << std::endl;
    std::cout << "D Right = " << *dRight << std::endl << std::endl;

    std::cout << "Stereo camera rotation = " << *stereoCamRotation << std::endl;
    std::cout << "Stereo camera translation = " << *stereoCamTranslation << std::endl << std::endl;

    std::cout << "New K left camera intrinsics = " << *newKLeft << std::endl;
    std::cout << "New K right camera intrinsics = " << *newKRight << std::endl;

    std::cout << "Calibration result score (RMS reprojection error, less than 1 = good): " << calibration_result << std::endl;


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
            saveCameraCalibrationToFile(kLeft, dLeft, kRight, dRight, newKLeft, newKRight,
                stereoCamTranslation, stereoCamRotation, &combinedCalibrationImages);
        }else
        {
            saveCameraCalibrationToFile(kLeft, dLeft, kRight, dRight, newKLeft, newKRight,
                stereoCamTranslation, stereoCamRotation, nullptr);
        }
    }
}

void openCvCameraTest()
{
    Mat camCalKLeft, camCalDLeft;
    Mat camCalKRight, camCalDRight;
    Mat camCalNewKLeft, camCalNewKRight;
    Mat stereoCamTranslation;
    Mat stereoCamRotation;

    Mat remapXLeft, remapYLeft, remapXRight, remapYRight;

    Size imageSize = Size(960, 960);


    // set coordinate system for markers
    g_MarkerObjPoints[0] = Point3d(-g_markerLength/2.f,g_markerLength/2.f,0);
    g_MarkerObjPoints[1] = Point3d(g_markerLength/2.f,g_markerLength/2.f,0);
    g_MarkerObjPoints[2] = Point3d(g_markerLength/2.f,-g_markerLength/2.f,0);
    g_MarkerObjPoints[3] = Point3d(-g_markerLength/2.f,-g_markerLength/2.f,0);

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
            &camCalNewKLeft, &camCalNewKRight, &stereoCamTranslation, &stereoCamRotation);
    }else
    {
        readCameraCalibrationFromFile(&camCalKLeft, &camCalDLeft, &camCalKRight, &camCalDRight,
            &camCalNewKLeft, &camCalNewKRight, &stereoCamTranslation, &stereoCamRotation);
    }

    //compute maps for image correction
    fisheye::initUndistortRectifyMap(camCalKLeft, camCalDLeft, cv::Matx33d::eye(),
        camCalNewKLeft, imageSize, CV_32FC1, remapXLeft, remapYLeft);
    fisheye::initUndistortRectifyMap(camCalKRight, camCalDRight, cv::Matx33d::eye(),
      camCalNewKRight, imageSize, CV_32FC1, remapXRight, remapYRight);

    CameraStreamer camStreamer = CameraStreamer(0);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // wait for stream to start

    Mat image;
    Mat leftImage, rightImage;

    //use calibration results to undistort live camera feed
    Mat correctedLeftImage;
    Mat correctedRightImage;

    // namedWindow("Corrected Left Image");
    // namedWindow("Corrected Right Image");

    namedWindow("GPU Image", WINDOW_NORMAL | WINDOW_OPENGL);

    camStreamer.getFrame(&image);

    leftImage = image(cv::Rect(0,0,image.cols/2,image.rows)).clone();
    rightImage = image(cv::Rect(image.cols/2,0,image.cols/2,image.rows)).clone();


    //upload image to GPU
    int imageBytes = image.step * image.rows;
    int imageWidth = 1920 / 2;
    int imageHeight = 960;

    //compute optimal blocks and threads based on image size
    dim3 threadsPerBlock(32,32);
    dim3 numBlocks((imageWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
           (imageHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

    resizeWindow("GPU Image", imageWidth, imageHeight);

    //allocate memory on gpu
    uchar *gpuImage;
    gpuErrchk(cudaMalloc((void**)&gpuImage, imageBytes));

    ogl::Texture2D openGlTextureDistortedLeft;
    ogl::Texture2D openGlTextureCorrectedLeft;
    ogl::Texture2D openGlTextureDistortedRight;
    ogl::Texture2D openGlTextureCorrectedRight;
    openGlTextureDistortedLeft.copyFrom(leftImage);
    openGlTextureCorrectedLeft.copyFrom(leftImage);
    openGlTextureDistortedRight.copyFrom(rightImage);
    openGlTextureCorrectedRight.copyFrom(rightImage);

    //setOpenGlDrawCallback("GPU Image", onOpenGlDraw, &openGlTextureDistorted);
    setOpenGlDrawCallback("GPU Image", onOpenGlDraw, &openGlTextureCorrectedLeft);


    //setup texture so CUDA and OpenGL can talk to each other

    std::vector<cudaGraphicsResource_t> cudaResources;
    cudaGraphicsResource_t cudaDistortedLeftImageHandle;
    cudaGraphicsResource_t cudaCorrectedLeftImageHandle;
    cudaGraphicsResource_t cudaDistortedRightImageHandle;
    cudaGraphicsResource_t cudaCorrectedRightImageHandle;
    gpuErrchk(cudaGraphicsGLRegisterImage(&cudaDistortedLeftImageHandle, openGlTextureDistortedLeft.texId(),
        GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    gpuErrchk(cudaGraphicsGLRegisterImage(&cudaCorrectedLeftImageHandle, openGlTextureCorrectedLeft.texId(),
        GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    gpuErrchk(cudaGraphicsGLRegisterImage(&cudaDistortedRightImageHandle, openGlTextureDistortedRight.texId(),
      GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    gpuErrchk(cudaGraphicsGLRegisterImage(&cudaCorrectedRightImageHandle, openGlTextureCorrectedRight.texId(),
        GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore)); 

    cudaResources.push_back(cudaDistortedLeftImageHandle);
    cudaResources.push_back(cudaCorrectedLeftImageHandle);
    cudaResources.push_back(cudaDistortedRightImageHandle);
    cudaResources.push_back(cudaCorrectedRightImageHandle);

    gpuErrchk(cudaPeekAtLastError());

    //upload distortion parameters to GPU
    if (!remapXLeft.isContinuous() || !remapYLeft.isContinuous())
    {
        std::cerr << "Left map is not continuous!" << std::endl;
    }
    if (!remapXRight.isContinuous() || !remapYRight.isContinuous())
    {
        std::cerr << "Right map is not continuous!" << std::endl;
    }

    //allocate memory on gpu
    float *gpuMapXLeft, *gpuMapYLeft, *gpuMapXRight, *gpuMapYRight;
    int mapSizeBytes = remapXLeft.cols * remapXLeft.rows * sizeof(float);
    gpuErrchk(cudaMalloc((void**)&gpuMapXLeft, mapSizeBytes));
    gpuErrchk(cudaMalloc((void**)&gpuMapYLeft, mapSizeBytes));
    gpuErrchk(cudaMalloc((void**)&gpuMapXRight, mapSizeBytes));
    gpuErrchk(cudaMalloc((void**)&gpuMapYRight, mapSizeBytes));

    //copy data to GPU
    gpuErrchk(cudaMemcpy(gpuMapXLeft, remapXLeft.data, mapSizeBytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(gpuMapYLeft, remapYLeft.data, mapSizeBytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(gpuMapXRight, remapXRight.data, mapSizeBytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(gpuMapYRight, remapYRight.data, mapSizeBytes, cudaMemcpyHostToDevice));

    std::vector<cudaSurfaceObject_t> cudaSurfaces;

    while (waitKey(1) != ESCAPE_KEY)
    {
        if (! camStreamer.tryGetFrame(&image))
        {
            //image not ready
            continue;
        }

        //use clone to make the image continuous
        leftImage = image(cv::Rect(0,0,image.cols/2,image.rows)).clone();
        rightImage = image(cv::Rect(image.cols/2,0,image.cols/2,image.rows)).clone();

        ////////////// GPU and CUDA processing

        //TODO look into using multiple cuda streams and multiple images in pipeline
        //send image to gpu
        openGlTextureDistortedLeft.copyFrom(leftImage);
        openGlTextureDistortedRight.copyFrom(rightImage);

        //give cuda control of the textures
        cudaSurfaceObject_t distortedLeftSurface = cudaSurfaces.emplace_back(setResourceCudaAccess(cudaDistortedLeftImageHandle));
        cudaSurfaceObject_t correctedLeftSurface = cudaSurfaces.emplace_back(setResourceCudaAccess(cudaCorrectedLeftImageHandle));
        cudaSurfaceObject_t distortedRightSurface = cudaSurfaces.emplace_back(setResourceCudaAccess(cudaDistortedRightImageHandle));
        cudaSurfaceObject_t correctedRightSurface = cudaSurfaces.emplace_back(setResourceCudaAccess(cudaCorrectedRightImageHandle));

        //launch kernels


        GpuKernelColorChange<<<numBlocks, threadsPerBlock>>>(distortedLeftSurface, imageWidth, imageHeight);
        gpuErrchk(cudaPeekAtLastError());
        GpuKernelRemapImage<<<numBlocks, threadsPerBlock>>>(distortedLeftSurface, correctedLeftSurface,
         gpuMapXLeft, gpuMapYLeft, imageWidth, imageHeight);
        gpuErrchk(cudaPeekAtLastError());

        //give control of the texture back to opengl to display
        unsetResourcesCudaAccess(&cudaSurfaces, &cudaResources);

        //wait for cuda to finish processing
        gpuErrchk( cudaDeviceSynchronize());

        //trigger opengl to display
        updateWindow("GPU Image");



        //////////// CPU only image processing

        //fisheye::undistortImage(leftImage, correctedLeftImage, camCalKLeft, camCalDLeft, camCalNewKLeft);
        //fisheye::undistortImage(rightImage, correctedRightImage, camCalKRight, camCalDRight, camCalNewKRight);
        remap(leftImage, correctedLeftImage, remapXLeft, remapYLeft, INTER_LINEAR, BORDER_CONSTANT);
        remap(rightImage, correctedRightImage, remapXRight, remapYRight, INTER_LINEAR, BORDER_CONSTANT);

        //find markers
        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
        aruco::DetectorParameters detectorParams = aruco::DetectorParameters();
        cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
        cv::aruco::ArucoDetector detector(dictionary, detectorParams);
        detector.detectMarkers(correctedLeftImage, markerCorners, markerIds, rejectedCandidates);

        size_t nMarkers = markerCorners.size();
        std::vector<Vec3d> rvecs(nMarkers), tvecs(nMarkers);

        std::vector<float> empty_vec;

        // Calculate pose for each marker
        for (size_t i = 0; i < nMarkers; i++) {
            solvePnP(g_MarkerObjPoints, markerCorners.at(i), camCalKLeft,
                empty_vec, rvecs.at(i), tvecs.at(i));
        }

        // draw results
        Mat imageCopy;
        correctedLeftImage.copyTo(imageCopy);
        if(!markerIds.empty()) {
            cv::aruco::drawDetectedMarkers(imageCopy, markerCorners, markerIds);


            for(unsigned int i = 0; i < markerIds.size(); i++)
                cv::drawFrameAxes(imageCopy, camCalKLeft, empty_vec, rvecs[i],
                    tvecs[i], g_markerLength * 1.5f, 2);

        }

        imshow ("marker detection", imageCopy);

        imshow("Corrected Left Image", correctedLeftImage);
        imshow("Corrected Right Image", correctedRightImage);

        imshow ("Raw image", image);



        ///////// end cpu only image processing

    }

    cudaFree(gpuImage);
    destroyAllWindows();
}

cudaSurfaceObject_t setResourceCudaAccess(cudaGraphicsResource_t resource)
{
    gpuErrchk(cudaGraphicsMapResources(1, {&resource}));
    cudaArray_t arrayHandle;
    gpuErrchk(cudaGraphicsSubResourceGetMappedArray(&arrayHandle, resource, 0, 0));
    cudaResourceDesc resourceDesc = cudaResourceDesc();
    resourceDesc.resType = cudaResourceTypeArray;
    resourceDesc.res.array.array = arrayHandle;
    cudaSurfaceObject_t surface;
    gpuErrchk(cudaCreateSurfaceObject(&surface, &resourceDesc));

    return surface;
}

void setResourcesCudaAccess(std::vector<cudaGraphicsResource_t>* resources,
    std::vector<cudaSurfaceObject_t>* surfacesOut)
{
    for (cudaGraphicsResource_t resource : *resources)
    {
        surfacesOut->emplace_back(setResourceCudaAccess(resource));
    }
}


void unsetResourceCudaAccess(cudaSurfaceObject_t surface, cudaGraphicsResource_t resource)
{
    gpuErrchk( cudaDestroySurfaceObject(surface));
    gpuErrchk( cudaGraphicsUnmapResources(1, {&resource}));
}

void unsetResourcesCudaAccess(std::vector<cudaSurfaceObject_t>* surfaces, std::vector<cudaGraphicsResource_t>* resources)
{
    int size = surfaces->size();
    for (int i = 0; i < size; i++)
    {
        unsetResourceCudaAccess(surfaces->at(0), resources->at(i));
        //remove reference to surface in vector after delete
        surfaces->erase(surfaces->begin());
    }
}



void cpuMarkerDetection(const Mat* image, Mat* camCalKLeft, Mat* camCalDLeft, Mat* camCalKRight, Mat* camCalDRight,
    Mat* camCalNewKLeft, Mat* camCalNewKRight)
{


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
