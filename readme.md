# Course Project: 3D Object Camera Tracking in VR / MR Headsets

Simplification: 
- traseck a aruco tag in camera space and transform to world space using headt pose
- get raw image from cameras, pass onto GPU to do the warp
- write CUDA kernels to process the aruco code detection and tracking
- also send headset pose to GPU with time code
- transform camera relative position to world position
- send position back to application and use to render item

## Dependencies
- C++ toolchain
- opengl (using GLAD to generate (https://gen.glad.sh/#generator=c&api=gl%3D4.6%2Cglx%3D1.4&profile=gl%3Dcore%2Cgles1%3Dcommon&extensions=GL_EXT_depth_bounds_test%2CGL_OVR_multiview&options=LOADER))
- cmake
- OpenXR SDK: https://github.com/KhronosGroup/OpenXR-SDK (TODO this might be pulled dynamically and built by the cmake script)
- OpenCV: https://docs.opencv.org/4.x/df/d65/tutorial_table_of_content_introduction.html https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html or install on linux using `apt install libopencv-dev`


# To Build
run:

```
mkdir build
cd build
cmake -G "Unix Makefiles" ../
make 
```



## compile OpenCV with CUDA support

new approach, compile openCV with Cuda support and include built files in project. Currently built under git/opencv using below command. Cmake will then use find_package command an point to build folder to include.



https://gist.github.com/minhhieutruong0705/8f0ec70c400420e0007c15c98510f133

build OpenCV with CUDA support and cuDNN support (python support disabled)

(CUDA and cuDNN need to be installed before hand)

```
cmake -D CMAKE_BUILD_TYPE=Release \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules/ \
-D BUILD_opencv_python3=OFF \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_PC_FILE_NAME=opencv.pc \
-D WITH_CUDA=ON \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=ON \
-D CUDA_ARCH_BIN=7.5 \
-D ENABLE_FAST_MATH=ON \
-D CUDA_FAST_MATH=ON \
-D WITH_CUFFT=ON \
-D WITH_CUBLAS=ON \
-D WITH_V4L=ON \
-D WITH_OPENCL=ON \
-D WITH_OPENGL=ON \
-D WITH_GSTREAMER=ON \
-D WITH_TBB=ON ../
```
