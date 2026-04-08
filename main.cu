#include "main.cuh"

//#include <opencv2/opencv.hpp>
#include <thread>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include "Cuda_Func.cuh"
#include <opencv2/calib3d.hpp>

using namespace cv;
int main(const int argc, char** argv)
{
    if (argc != 2) {
        printf("usage: <exe> <Image_Path>\n");
        return -1;
    }

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


    openVRTest();
    //openCvImageTest(argv[1]);
    //openCvCameraTest();
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


    if (!hasCamera)
    {
        std::cout << "Error no camera detected in OpenVR" << std::endl;
    }
    auto error = trackedCamera->AcquireVideoStreamingService(vr::k_unTrackedDeviceIndex_Hmd, &cameraHandle);
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

void openCvCameraTest()
{
    //TODO allow the index to change so we can select the correct camera(s)
    VideoCapture capture(0);
    Mat image;
    Mat leftImage, rightImage;

    //cv::fisheye::calibrate()

    if (capture.isOpened() == false)
    {
        std::cerr << "ERROR: Could not open camera." << std::endl;
    }
    namedWindow("Display Camera Image", WINDOW_AUTOSIZE);
    namedWindow("Left Camera Image", WINDOW_AUTOSIZE);
    namedWindow("Right Camera Image", WINDOW_AUTOSIZE);


    while (waitKey(25) == -1)
    {
        capture >> image;
        leftImage = image(cv::Rect(0,0,image.cols/2,image.rows));
        rightImage = image(cv::Rect(image.cols/2,0,image.cols/2,image.rows));

        if (image.empty())
        {
            std::cerr << "ERROR: blank frame" << std::endl;
            break;
        }

        imshow("Display Camera Image", image);
        imshow("Left Camera Image", leftImage);
        imshow("Right Camera Image", rightImage);

    }

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
