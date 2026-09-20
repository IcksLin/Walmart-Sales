cmake_minimum_required(VERSION 3.15)
project(walmart_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

find_package(OpenMP REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(walmart_cpp src/bindings.cpp src/walmart_core.cpp)
target_link_libraries(walmart_cpp PRIVATE OpenMP::OpenMP_CXX)
set_target_properties(walmart_cpp PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src)
