# filename: CMakeLists.txt

cmake_minimum_required(VERSION 3.12)
project(cbo-stiefel_module LANGUAGES CXX)

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

# Extract Python version to construct paths (Requires CMake 3.12+)
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
set(PYTHON_LIBRARY "${PYTHON_ROOT_DIR}/lib/libpython${PYTHON_VERSION_STRING}.dylib") # Adjust extension for Linux (.so)

message(STATUS "Manually Set Python Include Dir: ${PYTHON_INCLUDE_DIR}")
message(STATUS "Manually Set Python Library: ${PYTHON_LIBRARY}")

# Verify paths exist
if(NOT EXISTS "${PYTHON_INCLUDE_DIR}/Python.h")
    message(FATAL_ERROR "Python.h not found at manually set path: ${PYTHON_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${PYTHON_LIBRARY}")
    # Try alternate library name without dot
    set(PYTHON_LIBRARY_ALT "${PYTHON_ROOT_DIR}/lib/libpython${PYTHON_VERSION_STRING}.dylib") # Check this too
    if(NOT EXISTS "${PYTHON_LIBRARY_ALT}")
         message(FATAL_ERROR "Python library not found at ${PYTHON_LIBRARY} or ${PYTHON_LIBRARY_ALT}")
    else()
        set(PYTHON_LIBRARY ${PYTHON_LIBRARY_ALT})
        message(STATUS "Using alternate Python library name: ${PYTHON_LIBRARY}")
    endif()
endif()

# Explicitly add the include directory
include_directories(SYSTEM ${PYTHON_INCLUDE_DIR})
# --- End Manual Python Path Setup ---

# Find Pybind11 (Should work now as Python paths are set)
find_package(pybind11 REQUIRED)

# Find Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR}) # Add Eigen includes

# --- GIL FIX ---
# add_compile_definitions(EIGEN_NO_AUTOMATIC_THREADING)

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
    # Bindings file (Pybind11 entry point)
    kernel/bindings/cbo_module.cpp

    # Base Solver files
    kernel/solvers/base_solver.h
    kernel/solvers/base_solver.cpp # Implementation moved here

    # Kim Solver files
    kernel/solvers/kim_solver.h
    kernel/solvers/kim_solver.cpp

    # Cormac Solver files
    kernel/solvers/cormac_solver.h
    kernel/solvers/cormac_solver.cpp

    # Objective files (Headers are sufficient if definitions are inline or included)
    kernel/objectives/base_objective.h
    kernel/objectives/ackley_objective.h
    # Add other objective headers here as you create them (e.g., wopp_objective.h)
)
pybind11_add_module(cbo_module SHARED ${CBO_SOURCES})

# --- 4. TARGET PROPERTIES AND LINKAGE ---
target_include_directories(cbo_module PUBLIC
    kernel
    # Python include dir already added globally
    ${pybind11_INCLUDE_DIRS}
    ${EXTRA_INCLUDE_DIRS}
)

# Link explicitly against the found Python library and Pybind11 targets
target_link_libraries(cbo_module PUBLIC
    pybind11::module # Pybind11 target handles Python linkage implicitly in newer versions
    # Link Python library manually for robustness if pybind11::module isn't sufficient
    # ${PYTHON_LIBRARY}
    ${EXTRA_LINK_LIBS}
)

# Set output directory
set_target_properties(cbo_module PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)

# Optional: Add compiler flags
target_compile_options(cbo_module PRIVATE $<$<CONFIG:Release>:-O3 -DNDEBUG>)
target_compile_features(cbo_module PRIVATE cxx_std_17)