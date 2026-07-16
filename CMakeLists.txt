cmake_minimum_required(VERSION 3.15...3.29)
project(hnsw_healer VERSION 0.3.2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Relocatable extension (important for macOS delocate / auditwheel)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
if(APPLE)
  set(CMAKE_INSTALL_RPATH "@loader_path")
  set(CMAKE_MACOSX_RPATH ON)
else()
  set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

# ---------------------------------------------------------------------------
# Python + pybind11
# Development.Module is required for manylinux/cibuildwheel (no full libpython).
# Fall back to Development for older CMake/toolchains.
# ---------------------------------------------------------------------------
set(_hnsw_python_found FALSE)
find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)
if(Python3_Development.Module_FOUND)
  set(_hnsw_python_found TRUE)
else()
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  set(_hnsw_python_found TRUE)
endif()

if(NOT pybind11_DIR)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE _pybind11_cmakedir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pybind11_result
    ERROR_QUIET
  )
  if(_pybind11_result EQUAL 0 AND _pybind11_cmakedir)
    set(pybind11_DIR "${_pybind11_cmakedir}")
  endif()
endif()

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(hnsw_healer MODULE src/healer.cpp)

target_include_directories(hnsw_healer
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(hnsw_healer
  PRIVATE
    VERSION_INFO="${PROJECT_VERSION}"
)

if(MSVC)
  target_compile_options(hnsw_healer PRIVATE /O2 /W4 /EHsc)
else()
  # Keep warnings useful but avoid -Werror so CI toolchains don't flake.
  target_compile_options(hnsw_healer PRIVATE -O3 -Wall -Wextra)
endif()

# IMPORTANT: Do NOT override LIBRARY_OUTPUT_DIRECTORY here.
# setup.py / cibuildwheel pass -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=<extdir>
# so the built .so/.pyd lands where setuptools packages the wheel.
# Overriding to CMAKE_BINARY_DIR produced empty/broken wheels in CI.

install(TARGETS hnsw_healer
  LIBRARY DESTINATION .
  RUNTIME DESTINATION .
)
