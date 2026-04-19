# Course Project: 3D Object Camera Tracking in VR / MR Headsets

Simplification: 
- track an aruco tag in camera space and transform to world space using head pose
- get raw image from cameras, pass onto GPU to do the warp
- write CUDA kernels to process the aruco code detection and tracking
- also send headset pose to GPU with time code
- transform camera relative position to world position
- send position back to application and use to render item

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