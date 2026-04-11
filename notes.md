## approach
Simplification: 
- track an aruco tag in camera space and transform to world space using head pose
- get raw image from cameras, pass onto GPU to do the warp
- write CUDA kernels to process the aruco code detection and tracking
- also send headset pose to GPU with time code
- transform camera relative position to world position
- send position back to application and use to render item

## openXR approach
- needs opengl 
    - opengl (using GLAD to generate (https://gen.glad.sh/#generator=c&api=gl%3D4.6%2Cglx%3D1.4&profile=gl%3Dcore%2Cgles1%3Dcommon&extensions=GL_EXT_depth_bounds_test%2CGL_OVR_multiview&options=LOADER))
    - OpenXR SDK: https://github.com/KhronosGroup/OpenXR-SDK (TODO this might be pulled dynamically and built by the cmake script)

## stock openCV
- OpenCV: https://docs.opencv.org/4.x/df/d65/tutorial_table_of_content_introduction.html https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html or install on linux using `apt install libopencv-dev`

## compilation

new approach, compile openCV with Cuda support and include built files in project. Currently built under git/opencv using below command. Cmake will then use find_package command an point to build folder to include.


https://gist.github.com/minhhieutruong0705/8f0ec70c400420e0007c15c98510f133