#!/bin/bash

# Script for configuring and building the Pybind11 C++ CBO module
# OR packaging the source code AND the Homebrew formula.

# --- 0. DETERMINE BUILD MODE ---
# Usage: ./build_cbo_module.sh          -> Builds Python Module (default)
#        ./build_cbo_module.sh package  -> Creates Source Package (.tar.gz) & Formula
MODE="build" # Default mode
if [[ "$1" == "package" || "$1" == "tar" ]]; then
    MODE="package"
fi

# --- 1. CONFIGURATION AND PATH SETUP ---
PROJECT_ROOT=$(pwd)
BUILD_DIR="${PROJECT_ROOT}/build"
RELEASE_DIR="${PROJECT_ROOT}/releases" # <-- Final output directory for packages
MODULE_NAME="cbo_module"
MODULE_FILE="${MODULE_NAME}.so"
TEMPLATE_FILE="${PROJECT_ROOT}/cbo-stiefel.rb.template"

echo "=================================================="
echo "Tessera CBO Module Builder (C++)"
if [[ "$MODE" == "build" ]]; then
    echo "MODE: Building Python Module"
else
    echo "MODE: Creating Source Package & Homebrew Formula"
fi
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
if [[ "$MODE" == "package" ]]; then
    echo "Cleaning previous package directory..."
    rm -rf "$RELEASE_DIR"
fi

echo "Creating build directory: $BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || { echo "ERROR: Could not enter build directory."; exit 1; }

# --- 4. Configure Project with CMake (Common for Both Modes) ---
echo "Configuring project with CMake..."
export CMAKE_PREFIX_PATH="$PYTHON_ROOT_DIR_HINT:$CMAKE_PREFIX_PATH"
echo "[INFO] Temporarily prepended Conda root to CMAKE_PREFIX_PATH: $CMAKE_PREFIX_PATH"

cmake -DCMAKE_BUILD_TYPE=Release \
      -DPYBIND11_CPP_STANDARD=-std=c++17 \
      -DPYTHON_EXECUTABLE="$PYTHON_EXEC_PATH" \
      ..

export CMAKE_PREFIX_PATH=$(echo "$CMAKE_PREFIX_PATH" | sed "s|^${PYTHON_ROOT_DIR_HINT}:||")
echo "[INFO] Restored CMAKE_PREFIX_PATH: $CMAKE_PREFIX_PATH"

if [ $? -ne 0 ]; then
    echo "=================================================="
    echo "ERROR: CMake configuration failed."
    echo "=================================================="
    cd "$PROJECT_ROOT"
    exit 1
fi
echo "Configuration successful. Starting build..."


# --- 5. BUILD THE SELECTED TARGET ---

if [[ "$MODE" == "build" ]]; then
    # --- PROCESS 1: Build the Python Module ---

    NUM_CORES=$(sysctl -n hw.ncpu)
    cmake --build . --target "$MODULE_NAME" --config Release -- -j${NUM_CORES}

    if [ $? -ne 0 ]; then
        echo "=================================================="
        echo "ERROR: Compilation failed."
        echo "=================================================="
        cd "$PROJECT_ROOT"
        exit 1
    fi

    # --- 6. Post-Build Linkage Fix (macOS only) ---
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

else
    # --- PROCESS 2: Build Source Archive & Formula ---

    cd "$PROJECT_ROOT"

    # 1. Read Project Version from CMakeLists.txt
    PROJECT_VERSION=$(grep "project(cbo-stiefel_module VERSION" CMakeLists.txt | sed -E 's/.*VERSION ([0-9\.]+).*/\1/')

    if [ -z "$PROJECT_VERSION" ]; then
        echo "ERROR: Could not automatically determine PROJECT_VERSION from CMakeLists.txt."
        exit 1
    fi
    echo "[INFO] Found Project Version: $PROJECT_VERSION"

    # 2. Define filenames and URLs
    TAR_FILE_NAME="cbo-stiefel_module-${PROJECT_VERSION}-Source.tar.gz"
    TAR_FILE_PATH_BUILD="${BUILD_DIR}/${TAR_FILE_NAME}"
    TAR_FILE_PATH_RELEASE="${RELEASE_DIR}/${TAR_FILE_NAME}"

    GITHUB_USER="c0rmac"
    REPO_NAME="CBO-Stiefel"

    RELEASE_URL="https://github.com/${GITHUB_USER}/${REPO_NAME}/releases/download/v${PROJECT_VERSION}/${TAR_FILE_NAME}"

    FORMULA_TEMPLATE="${PROJECT_ROOT}/cbo-stiefel.rb.template"
    FORMULA_OUTPUT="${RELEASE_DIR}/cbo-stiefel.rb"

    # 3. Run the CPack build
    cd "$BUILD_DIR" || exit 1
    echo "[INFO] Building source package..."
    cmake --build . --target "package_source" --config Release
    if [ $? -ne 0 ]; then
        echo "ERROR: CPack (package_source) failed."
        cd "$PROJECT_ROOT"
        exit 1
    fi
    cd "$PROJECT_ROOT"

    # 4. Move tarball to releases/
    echo "[INFO] Moving package to releases/..."
    mkdir -p "$RELEASE_DIR"
    mv "$TAR_FILE_PATH_BUILD" "$TAR_FILE_PATH_RELEASE"
    if [ ! -f "$TAR_FILE_PATH_RELEASE" ]; then
        echo "ERROR: Packaged tarball not found at $TAR_FILE_PATH_BUILD"
        exit 1
    fi

    # 5. Calculate SHA256 Hash
    echo "[INFO] Calculating SHA256 hash..."
    FORMULA_SHA256=$(shasum -a 256 "$TAR_FILE_PATH_RELEASE" | awk '{print $1}')
    echo "[INFO] Hash: $FORMULA_SHA256"

    # 6. Generate the Formula file
    echo "[INFO] Generating Homebrew formula at $FORMULA_OUTPUT..."
    if [ ! -f "$FORMULA_TEMPLATE" ]; then
        echo "ERROR: Template file not found: $FORMULA_TEMPLATE"
        exit 1
    fi

    cp "$FORMULA_TEMPLATE" "$FORMULA_OUTPUT"

    # Use 'sed' to replace placeholders
    sed -i '' "s|__URL__|${RELEASE_URL}|g" "$FORMULA_OUTPUT"
    sed -i '' "s|__VERSION__|${PROJECT_VERSION}|g" "$FORMULA_OUTPUT"
    sed -i '' "s|__SHA256__|${FORMULA_SHA256}|g" "$FORMULA_OUTPUT"

    echo "=================================================="
    echo "✅ PACKAGE & FORMULA CREATION SUCCESSFUL."
    echo "Files created in the clean release directory: $RELEASE_DIR/"
    ls -l "$RELEASE_DIR"
    echo "=================================================="
    echo "ACTION REQUIRED:"
    echo "1. Create a GitHub release on 'CBO-Stiefel' repo tagged 'v${PROJECT_VERSION}'."
    echo "2. Upload '${TAR_FILE_NAME}' (from releases/) to that release."
    echo "3. The formula '${FORMULA_OUTPUT}' is now ready to be used."
fi

exit 0