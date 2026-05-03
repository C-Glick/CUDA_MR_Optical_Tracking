# Course Project: 3D Object Camera Tracking in VR / MR Headsets

This application demonstrates uses of CUDA to augment Aruco marker tracking for mixed reality applications.

Simplification: 
- track an aruco tag in camera space and transform to world space using head pose
- get raw image from cameras, pass onto GPU to do the warp
- write CUDA kernels to process the aruco code detection and tracking
- also send headset pose to GPU with time code
- transform camera relative position to world position
- send position back to application and use to render item

## Key features
- Interaction between multiple libraries and technologies, OpenCV, CUDA, OpenVR, and OpenGL
- CMake automation 
- OpenCV custom compilation with CUDA support for hardware accelerated basic operations
- Camera calibration
    - Calibration process to capture camera intrinsic properties used for distortion correction
    - Saves camera calibration to disk for later usage
- Reads high definition video feed using OpenCV in real time 
    - CameraStreamer class with seperate capture thread to capture images in real time asynchronous of processing, ensures
      always working on the latest data if algorithm runs slower than frame rate
- Integrates with OpenGL, Places captured data into OpenGL texture
- Passes OpenGL texture access to CUDA and runs image modification kernels on texture
    - Implemented remapping function in CUDA to correct image distortion
- Displays the corrected image directly to screen without copying back to CPU, CUDA hands access back to OpenGL for rendering
- Implements adaptive thresholding in GPU during aruco code detection
    - Extended the ArucoDetector class from OpenCV with new class GpuArucoDetector. 
    - GpuArucoDetector uses adaptive thresholding in CUDA to speed up parts of the tracking
- Uses OpenCV functions to find aruco codes in the image and tracks them in 3D space relative to the camera
- Communicates with OpenVR to track VR headset position and translate aruco markers into world space coordinates

## Prerequisite Dependencies
- Standard C++ toolchain
- cmake
- CUDA https://developer.nvidia.com/cuda/toolkit
- cuDNN https://developer.nvidia.com/cudnn
- OpenGL
- GTK 2
    - gtkglext-1.0 (apt install libgtkglext1-dev)
    - appmenu-gtk2-module 
    - appmenu-gtk3-module 
    - libcanberra-gtk-module 

# To Build

## Ensure all gitmodules are checked out
- run `git submodule init` and `git submodule update`


## compile OpenCV with CUDA support
- build OpenCV with CUDA support and cuDNN support (python support disabled)
(CUDA and cuDNN need to be installed before hand)
- OpenCV takes a while to compile with all the extra modules so be aware this can take some time

- run `./compile_opencv.sh` from the root of the project, "CUDA_MR_Optical_Tracking". This will configure and compile OpenCV in the external/opencv folder.

## compile project
- Run `run.sh` in the root of the project, this uses cmake to configure and compile the project. It will then execute with recommended command line arguments.
- The built binary is located in the "cmake-build-*" folder and can be run with custom command line arguments
- To compile manually instead of using `run.sh` :
    - Run cmake on the root of the project, `cmake -S ./ -B ./cmake-build-release -D CMAKE_BUILD_TYPE=Release` or `cmake -S ./ -B ./cmake-build-debug -D CMAKE_BUILD_TYPE=Debug` for a debug build
    - Next run `make -C ./cmake-build-release` to compile project
    - If compilation is successful run project with command `./cmake-build-release/CUDA_MR_Optical_Tracking`


# clean project
- From the root of the project run `./clean_root.sh` this will remove the root project clean  
- From the root of the project run `./clean_opencv.sh` this will remove the opencv build directory


## Challenges
- CPP verbose language 
- Learning CMake 
- OpenXR and OpenVR lack of support for camera integration
- OpenCV systems and image processing
- OpenCV interaction with CUDA
- OpenCV compilation inside of a larger project
- High bandwidth usage for HD video steaming, encountered odd visual glitches
- Cuda surfaces for texture memory storage and modifications
- 3D graphics coordinate system transformations