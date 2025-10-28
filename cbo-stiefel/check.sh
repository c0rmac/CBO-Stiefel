PYTHON_EXEC_PATH=$(which python)
PYTHON_ROOT_DIR=$(dirname $(dirname "$PYTHON_EXEC_PATH"))
PYTHON_VERSION_MAJOR=$(python -c "import sys; print(sys.version_info.major)")
PYTHON_VERSION_MINOR=$(python -c "import sys; print(sys.version_info.minor)")
PYTHON_INCLUDE_DIR="$PYTHON_ROOT_DIR/include/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}"

ls "$PYTHON_INCLUDE_DIR/Python.h"