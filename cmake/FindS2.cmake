# The MIT License (MIT)
#
# Copyright (c) 2020 Benoit Bovy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#
# FindS2
# ------
#
# Find S2 (S2Geometry) include directories and libraries.
#
# This module will set the following variables:
#
#  S2_FOUND          - System has S2
#  S2_INCLUDE_DIRS   - The S2 include directories
#  S2_LIBRARIES      - The libraries needed to use S2
#
# This module will also create the "tbb" target that may be used when building
# executables and libraries.

include(FindPackageHandleStandardArgs)

if(NOT S2_ROOT AND DEFINED ENV{S2_DIR})
  set(S2_ROOT "$ENV{S2_DIR}")
endif()

find_path(s2_INCLUDE_DIR s2/s2cell.h
  HINTS ${S2_ROOT}
  PATH_SUFFIXES include
  )

find_library(s2_LIBRARY
  NAMES s2
  HINTS ${S2_ROOT}
  )

find_package_handle_standard_args(s2
  REQUIRED_VARS s2_INCLUDE_DIR s2_LIBRARY
  )

if(NOT CMAKE_VERSION VERSION_LESS 3.0 AND S2_FOUND)
  set(s2_INCLUDE_DIRS ${s2_INCLUDE_DIR})
  set(s2_LIBRARIES ${s2_LIBRARY})

  add_library(s2 SHARED IMPORTED)
  set_target_properties(s2 PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${s2_INCLUDE_DIRS}
    IMPORTED_LOCATION ${s2_LIBRARIES}
    )

  mark_as_advanced(s2_INCLUDE_DIRS s2_LIBRARIES)
endif()
