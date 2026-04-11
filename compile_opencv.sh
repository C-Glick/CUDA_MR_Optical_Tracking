#!/bin/bash

cd ./external/opencv

cmake -S ./ -B ./build \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules/ \
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
    -D WITH_TBB=ON

echo "=============Building OpenCV... ============="

cd ./build

make -j $(nproc)