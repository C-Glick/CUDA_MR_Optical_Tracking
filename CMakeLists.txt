cmake_minimum_required(VERSION 4.1)
project(CUDA_MR_Optical_Tracking CXX CUDA)

#execute_process(COMMAND ../compile_opencv.sh
#        COMMAND_ECHO STDOUT WORKING_DIRECTORY ../)

add_executable(CUDA_MR_Optical_Tracking main.cu
        Cuda_Func.cu
        Cuda_Func.cuh
        main.cuh)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")


set(CMAKE_CXX_STANDARD 20)

set(OpenVR_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/openvr")
find_package(OpenVR REQUIRED)

# Link the library to your target
if(OpenVR_FOUND)
    message(STATUS "OpenVR include dirs: ${OpenVR_INCLUDE_DIRS}")
    message(STATUS "OpenVR libs: ${OpenVR_LIBRARIES}")
    target_link_libraries(CUDA_MR_Optical_Tracking PRIVATE ${OpenVR_LIBRARIES})
    target_include_directories(CUDA_MR_Optical_Tracking PRIVATE ${OpenVR_INCLUDE_DIRS})
    # The include directories are usually automatically set when using an imported target
endif()

# satisfy openCV's outdated way of include CUDA in cmake
set(CUDA_FOUND TRUE)
set(CUDA_VERSION_STRING "13.1")
message("Assuming CUDA version is 13.1")
# TODO make openCV a part of this project as a git submodule
find_package(OpenCV PATHS external/opencv/opencv_build NO_DEFAULT_PATH)
if(NOT OpenCV_FOUND)
    #build opencv with CUDA support
    message(WARNING "Could not find OpenCV with CUDA support, please run compile_opencv.sh from project root.")
    find_package(OpenCV REQUIRED PATHS external/opencv/opencv_build NO_DEFAULT_PATH)
endif()

message(STATUS "opencv include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "opencv libs: ${OpenCV_LIBS}")

include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(CUDA_MR_Optical_Tracking PRIVATE ${OpenCV_LIBS})
