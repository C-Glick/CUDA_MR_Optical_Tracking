#!/bin/bash

echo "===== Configuring with cmake... ====="
cmake -S ./ -B ./cmake-build-debug -D CMAKE_BUILD_TYPE=Debug

echo "===== Compiling with cmake... ====="
cmake --build ./cmake-build-debug --config Debug --target CUDA_MR_Optical_Tracking -j $(nproc)

echo "—————===== Running the executable with recommended settings... =====—————"
./cmake-build-debug/CUDA_MR_Optical_Tracking --help
./cmake-build-debug/CUDA_MR_Optical_Tracking --calibration=./camera_calibration --video=./GlickArucoSampleVideo.mp4 --noSteamVR
