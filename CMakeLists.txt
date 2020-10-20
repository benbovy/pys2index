cmake_minimum_required(VERSION 3.12)
project(pys2index)

set(CMAKE_MODULE_PATH
  ${PROJECT_SOURCE_DIR}/cmake
  ${CMAKE_MODULE_PATH})

set(pys2index_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CMAKE_CXX_STANDARD 14)
set(PYBIND11_CPP_STANDARD -std=c++14)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wunused-parameter -Wextra -Wreorder -Wconversion -fvisibility=hidden")
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MP /bigobj")
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

# Dependencies
# ============

find_package(xtensor REQUIRED)
message(STATUS "Found xtensor: ${xtensor_INCLUDE_DIRS}")

find_package(xtensor-python REQUIRED)
message(STATUS "Found xtensor-python: ${xtensor-python_INCLUDE_DIRS}")

find_package (Python COMPONENTS Interpreter NumPy)

set(PYBIND11_PYTHON_VERSION ${Python_VERSION} CACHE STRING "")
find_package(pybind11 REQUIRED)

find_package(s2 REQUIRED)

# Build
# =====

add_library(pys2index MODULE src/main.cpp)

target_include_directories(pys2index PUBLIC
  ${pys2index_INCLUDE_DIR}
  ${xtensor-python_INCLUDE_DIRS}
  ${Python_NumPy_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIRS}
  ${s2_INCLUDE_DIRS}
  )

target_link_libraries(pys2index PUBLIC
  xtensor
  xtensor-python
  pybind11::module
  s2
  )

set_target_properties(pys2index PROPERTIES
  PREFIX "${PYTHON_MODULE_PREFIX}"
  SUFFIX "${PYTHON_MODULE_EXTENSION}")
