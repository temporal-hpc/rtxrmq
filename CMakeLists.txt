cmake_minimum_required(VERSION 3.17)
project (RTX-RMQ LANGUAGES CXX CUDA)

# Set default build type to 'Release'
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# add path to our CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")

# turn on C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# for common headers
include_directories(./)
include_directories(./include)

# ==============================================================================
# Utility function to get PTX compilation & copying working
# ==============================================================================
include(PTXUtilities)

# ==============================================================================
# To link against CUDA::cudart and/or CUDA::cuda_driver
# ==============================================================================
find_package(CUDAToolkit REQUIRED)

# ==============================================================================
# Find Optix7
# ==============================================================================
find_package(OptiX7 REQUIRED)
include_directories(${OptiX7_INCLUDE_DIRS})


# ==============================================================================
# Find OpenMP Library
# ==============================================================================
find_package(OpenMP REQUIRED)
IF (OPENMP_FOUND)
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()


# ==============================================================================
# Main Executable
# ==============================================================================
find_library(RMQRMM rmqrmmBP.a hrmq/)

add_compile_options(-fopenmp)
add_executable(rtxrmq src/main.cu common/common.cpp)
target_link_libraries(rtxrmq PRIVATE CUDA::cudart OptiX7 OpenMP::OpenMP_CXX ${RMQRMM})
set_target_properties(rtxrmq PROPERTIES CUDA_ARCHITECTURES native)

# ==============================================================================
# Add any needed PTX compilations here
# ==============================================================================
add_ptx_targets(rtxrmq rtx_kernels)
