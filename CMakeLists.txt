cmake_minimum_required(VERSION 3.15...3.29)
project(hnsw_healer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Prefer a local pybind11 (e.g. from pip) via Python, fall back to find_package.
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Allow overriding with -Dpybind11_DIR=...
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

# Hardened release defaults suitable for a memory-touching unlearning helper.
if(MSVC)
  target_compile_options(hnsw_healer PRIVATE /O2 /W4 /EHsc)
else()
  target_compile_options(hnsw_healer PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

# Place the extension next to the project root for easy import during dev.
set_target_properties(hnsw_healer PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

install(TARGETS hnsw_healer
  LIBRARY DESTINATION .
  RUNTIME DESTINATION .
)
