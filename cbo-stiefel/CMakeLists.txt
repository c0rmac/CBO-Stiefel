cmake_minimum_required(VERSION 3.12)
project(cbo-stiefel_module VERSION 1.0.0 LANGUAGES CXX)

# --- 1. CONFIGURATION AND DEPENDENCY FINDING ---

# --- Force Find Conda Python Paths Manually ---
# Get executable hint (MUST be passed via -DPYTHON_EXECUTABLE=... from build script)
if(NOT DEFINED PYTHON_EXECUTABLE)
    message(FATAL_ERROR "PYTHON_EXECUTABLE must be set via CMake command line (-D...).")
endif()
message(STATUS "Using PYTHON_EXECUTABLE hint: ${PYTHON_EXECUTABLE}")

# Derive necessary paths directly from the executable path
get_filename_component(PYTHON_BIN_DIR ${PYTHON_EXECUTABLE} DIRECTORY)
get_filename_component(PYTHON_ROOT_DIR ${PYTHON_BIN_DIR} DIRECTORY)

# Extract Python version to construct paths
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    OUTPUT_VARIABLE PYTHON_VERSION_STRING
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT PYTHON_VERSION_STRING)
    message(FATAL_ERROR "Could not determine Python version from executable.")
endif()
message(STATUS "Derived Python Version: ${PYTHON_VERSION_STRING}")

set(PYTHON_INCLUDE_DIR "${PYTHON_ROOT_DIR}/include/python${PYTHON_VERSION_STRING}")
set(PYTHON_LIBRARY "${PYTHON_ROOT_DIR}/lib/libpython${PYTHON_VERSION_STRING}.dylib")

message(STATUS "Manually Set Python Include Dir: ${PYTHON_INCLUDE_DIR}")
message(STATUS "Manually Set Python Library: ${PYTHON_LIBRARY}")

# Verify paths exist
if(NOT EXISTS "${PYTHON_INCLUDE_DIR}/Python.h")
    message(FATAL_ERROR "Python.h not found at manually set path: ${PYTHON_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${PYTHON_LIBRARY}")
    set(PYTHON_LIBRARY_ALT "${PYTHON_ROOT_DIR}/lib/libpython${PYTHON_VERSION_STRING}.dylib")
    if(NOT EXISTS "${PYTHON_LIBRARY_ALT}")
         message(FATAL_ERROR "Python library not found at ${PYTHON_LIBRARY} or ${PYTHON_LIBRARY_ALT}")
    else()
        set(PYTHON_LIBRARY ${PYTHON_LIBRARY_ALT})
        message(STATUS "Using alternate Python library name: ${PYTHON_LIBRARY}")
    endif()
endif()

include_directories(SYSTEM ${PYTHON_INCLUDE_DIR})
# --- End Manual Python Path Setup ---

# Find Pybind11 and Eigen3
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# --- 2. TORCH/LIBTORCH (CONDITIONAL BACKEND) ---
set(EXTRA_INCLUDE_DIRS "")
set(EXTRA_LINK_LIBS "")
if (DEFINED ENV{LIBTORCH_DIR})
    set(LIBTORCH_DIR $ENV{LIBTORCH_DIR})
    find_package(Torch QUIET PATHS ${LIBTORCH_DIR} ${PYTHON_ROOT_DIR})

    if (Torch_FOUND)
        message(STATUS "Found Torch (LibTorch). Enabling PyTorch Backend.")
        add_compile_definitions(USE_TORCH_BACKEND)
        list(APPEND EXTRA_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
        list(APPEND EXTRA_LINK_LIBS ${TORCH_LIBRARIES})
    else()
        message(WARNING "LibTorch not found. PyTorch backend disabled.")
    endif()
endif()

# --- 3. DEFINE THE PYTHON EXTENSION MODULE ---
set(CBO_SOURCES
    # Paths updated to reflect the 'src/' root directory
    src/bindings/cbo_module.cpp
    src/solvers/base_solver.h
    src/solvers/base_solver.cpp
    src/solvers/kim_solver.h
    src/solvers/kim_solver.cpp
    src/solvers/cormac_solver.h
    src/solvers/cormac_solver.cpp
    src/objectives/base_objective.h
    src/objectives/ackley_objective.h
)
pybind11_add_module(cbo_module SHARED ${CBO_SOURCES})

# --- 3.5. FORCE OUTPUT FILENAME ---
# Forces output name to cbo_module.so (overriding Python's PEP 3149 standard)
set_target_properties(cbo_module PROPERTIES
    LIBRARY_OUTPUT_NAME "cbo_module"
    SUFFIX ".so"
)

# --- 4. TARGET PROPERTIES AND LINKAGE ---
target_include_directories(cbo_module PUBLIC
    src # Public includes now point to the 'src' directory
    ${pybind11_INCLUDE_DIRS}
    ${EXTRA_INCLUDE_DIRS}
)

target_link_libraries(cbo_module PUBLIC
    pybind11::module
    ${EXTRA_LINK_LIBS}
)

set_target_properties(cbo_module PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)

target_compile_options(cbo_module PRIVATE $<$<CONFIG:Release>:-O3 -DNDEBUG>)
target_compile_features(cbo_module PRIVATE cxx_std_17)

# -----------------------------------------------------------
# --- 5. CPACK SOURCE DISTRIBUTION (Process 2: tar.gz) ---
# -----------------------------------------------------------
# Enables the 'package_source' target to create the distribution tarball.

set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_SOURCE_GENERATOR "TGZ")

# The CPack output filename
set(CPACK_SOURCE_PACKAGE_FILE_NAME
    "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-Source")

# Ignore common files and the release/export directories
set(CPACK_SOURCE_IGNORE_FILES
    "/.git/" "/.github/" "/.idea/" "/.vs/" "/build/"
    "/cmake-build-.*" "*.pyc" "*.so" "*.dylib" "*.dll"
    "/export/" "/releases/", "/demo/"
)

include(CPack)