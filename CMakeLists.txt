cmake_minimum_required(VERSION 3.15)

if(NOT SKBUILD_PROJECT_NAME)
    set(SKBUILD_PROJECT_NAME "pys2index")
endif()

if(NOT SKBUILD_PROJECT_VERSION)
    set(SKBUILD_PROJECT_VERSION 9999)
endif()

project(
  ${SKBUILD_PROJECT_NAME}
  LANGUAGES CXX)

# https://gitlab.kitware.com/cmake/cmake/-/issues/16716
set(PROJECT_VERSION ${SKBUILD_PROJECT_VERSION})

set(CMAKE_MODULE_PATH
  ${PROJECT_SOURCE_DIR}/cmake
  ${CMAKE_MODULE_PATH})

set(pys2index_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-parameter -Wextra -Wreorder -Wconversion -fvisibility=hidden")
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MP /bigobj /J")
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
    add_definitions(-D_USE_MATH_DEFINES)
    add_definitions(-DNOMINMAX)
endif()

# Dependencies
# ============

find_package(xtensor REQUIRED)
message(STATUS "Found xtensor: ${xtensor_INCLUDE_DIRS}")

find_package(xtensor-python REQUIRED)
message(STATUS "Found xtensor-python: ${xtensor-python_INCLUDE_DIRS}")

find_package (Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)
message(STATUS "Found python v${Python_VERSION}: ${Python_EXECUTABLE}")
message(STATUS "Found numpy v${Python_NumPy_VERSION}: ${Python_NumPy_INCLUDE_DIRS}")

find_package(pybind11 REQUIRED)
message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")

find_package(s2 REQUIRED)

# Build
# =====

add_library(pys2index MODULE src/main.cpp)

target_include_directories(pys2index PUBLIC ${pys2index_INCLUDE_DIR})

target_compile_features(pys2index PUBLIC cxx_std_17)
target_compile_definitions(pys2index PRIVATE VERSION_INFO=${PROJECT_VERSION})

target_link_libraries(pys2index PRIVATE
  xtensor
  xtensor-python
  pybind11::module
  Python::NumPy
  s2
  )

pybind11_extension(pys2index)
if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(pys2index)
endif()

# Installation
# ============

install(TARGETS pys2index LIBRARY DESTINATION .)
