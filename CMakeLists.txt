cmake_minimum_required(VERSION 3.23)
project(mnist_blas)

set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CUDA_STANDARD 14)


add_executable(mnist_blas main.cpp read_mnist.cpp read_mnist.h matrix.cpp matrix.h utils.cpp utils.h optimizer.cpp optimizer.h)
#set_target_properties(mnist_blas PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

include_directories(C:/HMS/_CODE/Envs/OpenBLAS/include/)
link_directories(C:/HMS/_CODE/Envs/OpenBLAS/lib/)
set(BLAS_LIBS libopenblas)
target_link_libraries(mnist_blas libopenblas)

set(OpenCV_DIR C:/HMS/_CODE/Envs/OpenCV/build/x64/vc16/lib/)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)
target_link_libraries(mnist_blas ${OpenCV_LIBS})


