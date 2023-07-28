cmake_minimum_required(VERSION 3.20)
project(onetla LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_C_COMPILER icx)
set (CMAKE_CXX_COMPILER icpx)

# option(WITH_PREFETCH "use esimd" OFF)

# if (WITH_PREFETCH)
#     add_definitions(-DUSE_PREFETCH)
# endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fopenmp")
include_directories(include)
include_directories(examples)


add_subdirectory(examples)
