cmake_minimum_required (VERSION 3.7)

project (OpenCL_pefix_sum)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
set (CMAKE_CXX_STANDARD 17)

set (SOURCE_FILES main.cpp cl.hpp)

find_package (OpenCL REQUIRED)

if (${OPENCL_FOUND})
    include_directories (${OpenCL_INCLUDE_DIR})

    add_executable (prefix-sum ${SOURCE_FILES})
    target_link_libraries (prefix-sum LINK_PRIVATE ${OpenCL_LIBRARY})

    message ("${OpenCL_LIBRARY}")
endif()