cmake_minimum_required(VERSION 3.28)
project(CUDA_MR_Optical_Tracking CXX CUDA)

#execute_process(COMMAND ../compile_opencv.sh
#        COMMAND_ECHO STDOUT WORKING_DIRECTORY ../)

add_executable(${PROJECT_NAME} main.cu
        Cuda_Func.cu
        Cuda_Func.cuh
        main.cuh
        external/nlohmann/json.hpp
        CameraStreamer.cpp
        CameraStreamer.h
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")


set(CMAKE_CXX_STANDARD 20)

#
#cmake_policy(SET CMP0167 NEW)
#find_package(Boost REQUIRED)
#target_link_libraries(${PROJECT_NAME} PRIVATE ${Boost_LIBRARIES})
#target_include_directories(${PROJECT_NAME} PRIVATE ${Boost_INCLUDE_DIRS})

# ==== OpenVR ====

set(OpenVR_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/openvr")
find_package(OpenVR REQUIRED)

# Link the library to your target
if(OpenVR_FOUND)
    message(STATUS "OpenVR include dirs: ${OpenVR_INCLUDE_DIRS}")
    message(STATUS "OpenVR libs: ${OpenVR_LIBRARIES}")
    target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenVR_LIBRARIES})
    target_include_directories(${PROJECT_NAME} PRIVATE ${OpenVR_INCLUDE_DIRS})
    # The include directories are usually automatically set when using an imported target
endif()


# ==== OpenCV ====

# satisfy openCV's outdated way of include CUDA in cmake
set(CUDA_FOUND TRUE)
set(CUDA_VERSION_STRING "13.1")
message("Assuming CUDA version is 13.1")

find_package(OpenCV PATHS external/opencv/build NO_DEFAULT_PATH)
if(NOT OpenCV_FOUND)
    #build opencv with CUDA support
    message(FATAL_ERROR "Could not find OpenCV with CUDA support, please run compile_opencv.sh from project root.")
endif()

message(STATUS "opencv include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "opencv libs: ${OpenCV_LIBS}")

target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

set(CMAKE_CUDA_FLAGS ${CUDA_CUDA_FLAGS} "-g -G")
