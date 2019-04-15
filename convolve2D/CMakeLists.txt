cmake_minimum_required (VERSION 3.7)

project (OpenCL_convolve2D)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
set (CMAKE_CXX_STANDARD 17)

set (SOURCE_FILES main.cpp)

find_package (OpenCL REQUIRED)

if (${OPENCL_FOUND})
    include_directories (${OpenCL_INCLUDE_DIR})

    add_executable (convolve2D ${SOURCE_FILES})
    target_link_libraries (convolve2D LINK_PRIVATE ${OpenCL_LIBRARY})

    message ("${OpenCL_LIBRARY}")
endif()