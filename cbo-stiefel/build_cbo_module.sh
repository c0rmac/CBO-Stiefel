#!/bin/bash

# Script for configuring and building the Pybind11 C++ CBO module.
# Includes CMAKE_PREFIX_PATH fix for Conda environments, explicit Python executable hint,
# and post-build linkage fix for macOS.

# --- 1. CONFIGURATION AND PATH SETUP ---
PROJECT_ROOT=$(pwd)
BUILD_DIR="${PROJECT_ROOT}/build"
MODULE_NAME="cbo_module"
MODULE_FILE="${MODULE_NAME}.so"

echo "=================================================="
echo "Tessera CBO Module Builder (C++)"
echo "Project Root: ${PROJECT_ROOT}"
echo "=================================================="

# --- Check Conda Environment and Get Python Paths ---
PYTHON_EXEC_PATH=$(which python)
if [ -z "$PYTHON_EXEC_PATH" ] || [[ "$PYTHON_EXEC_PATH" != *"/envs/"* && "$PYTHON_EXEC_PATH" != *"/miniconda"* && "$PYTHON_EXEC_PATH" != *"/anaconda"* && "$PYTHON_EXEC_PATH" != *"/mambaforge"* ]]; then
    echo "ERROR: Could not find Python interpreter in a Conda/Mamba env. Activate 'cbo'."
    exit 1
fi
PYTHON_ROOT_DIR_HINT=$(dirname $(dirname "$PYTHON_EXEC_PATH"))
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_LIBRARY_PATH="${PYTHON_ROOT_DIR_HINT}/lib/libpython${PYTHON_VERSION}.dylib"

echo "[INFO] Using Python Interpreter: $PYTHON_EXEC_PATH (Version: $PYTHON_VERSION)"
echo "[INFO] Derived Python Root Dir: $PYTHON_ROOT_DIR_HINT"
echo "[INFO] Target Python Library: $PYTHON_LIBRARY_PATH"

# --- 2. Handle LibTorch/PyTorch Backend ---
# (Logic remains the same)
if [ -z "$LIBTORCH_DIR" ]; then
    echo "[INFO] LIBTORCH_DIR not set. PyTorch backend disabled."
else
    echo "[INFO] Found LIBTORCH_DIR=$LIBTORCH_DIR. Enabling PyTorch backend."
fi
echo "--------------------------------------------------"

# --- 3. Clean and Create Build Directory ---
echo "Cleaning previous build..."
rm -rf "$BUILD_DIR"
rm -f "$PROJECT_ROOT/$MODULE_FILE"

echo "Creating build directory: $BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || { echo "ERROR: Could not enter build directory."; exit 1; }

# --- 4. Configure Project with CMake (Forcing Conda Prefix Path AND Executable) ---
echo "Configuring project with CMake..."

# Set CMAKE_PREFIX_PATH to force CMake to prioritize the Conda environment root.
export CMAKE_PREFIX_PATH="$PYTHON_ROOT_DIR_HINT:$CMAKE_PREFIX_PATH"
echo "[INFO] Temporarily prepended Conda root to CMAKE_PREFIX_PATH: $CMAKE_PREFIX_PATH"

# Run CMake, now also providing the explicit executable path.
# CMakeLists.txt uses this to verify find_package worked correctly.
cmake -DCMAKE_BUILD_TYPE=Release \
      -DPYBIND11_CPP_STANDARD=-std=c++17 \
      -DPYTHON_EXECUTABLE="$PYTHON_EXEC_PATH" \
      ..

# Unset the temporary path modification
export CMAKE_PREFIX_PATH=$(echo "$CMAKE_PREFIX_PATH" | sed "s|^${PYTHON_ROOT_DIR_HINT}:||")
echo "[INFO] Restored CMAKE_PREFIX_PATH: $CMAKE_PREFIX_PATH"

# Check CMake configuration success
if [ $? -ne 0 ]; then
    echo "=================================================="
    echo "ERROR: CMake configuration failed."
    echo "       Check Conda env and Python dev files."
    echo "=================================================="
    cd "$PROJECT_ROOT"
    exit 1
fi

# --- 5. Build the Module ---
echo "Configuration successful. Starting build..."
NUM_CORES=$(sysctl -n hw.ncpu)
cmake --build . --target "$MODULE_NAME" --config Release -- -j${NUM_CORES}

# Check build success
if [ $? -ne 0 ]; then
    echo "=================================================="
    echo "ERROR: Compilation failed."
    echo "=================================================="
    cd "$PROJECT_ROOT"
    exit 1
fi

# --- 6. Post-Build Linkage Fix (macOS only) ---
# (Logic remains the same)
if [[ "$OSTYPE" == "darwin"* ]]; then
    FINAL_MODULE_PATH="$PROJECT_ROOT/$MODULE_FILE"
    if [ ! -f "$FINAL_MODULE_PATH" ]; then
        if [ -f "$BUILD_DIR/$MODULE_FILE" ]; then FINAL_MODULE_PATH="$BUILD_DIR/$MODULE_FILE"; fi
    fi

    if [ -f "$FINAL_MODULE_PATH" ]; then
        echo "Performing install_name_tool check/fix..."
        LINKED_PYTHON_LIB=$(otool -L "$FINAL_MODULE_PATH" | grep -E 'Python.framework|libpython' | awk '{print $1}' | head -n 1)

        if [ -n "$LINKED_PYTHON_LIB" ] && [[ "$LINKED_PYTHON_LIB" != *"$PYTHON_ROOT_DIR_HINT"* ]]; then
            echo "[FIX] Incorrect Python library linked: $LINKED_PYTHON_LIB"
            echo "[FIX] Changing link path to: $PYTHON_LIBRARY_PATH"
            install_name_tool -change "$LINKED_PYTHON_LIB" "$PYTHON_LIBRARY_PATH" "$FINAL_MODULE_PATH"
            if [ $? -ne 0 ]; then echo "ERROR: install_name_tool failed."; else echo "✅ Link redirect attempted."; fi
        elif [ -n "$LINKED_PYTHON_LIB" ]; then
            echo "[INFO] Correct Python library seems linked: $LINKED_PYTHON_LIB"
        else
            echo "WARNING: Could not identify linked Python library."
        fi
    else
        echo "WARNING: Module file '$MODULE_FILE' not found for linkage check."
    fi
fi

# --- 7. Final Verification ---
cd "$PROJECT_ROOT"
if [ -f "$MODULE_FILE" ]; then
    echo "=================================================="
    echo "✅ BUILD SUCCESSFUL."
    echo "Module path: $PROJECT_ROOT/$MODULE_FILE"
    echo "Verifying linked Python library:"
    otool -L "$MODULE_FILE" | grep 'python'
    echo "=================================================="
else
    echo "=================================================="
    echo "ERROR: Build finished, but module file '$MODULE_FILE' not found."
    echo "=================================================="
    exit 1
fi

exit 0