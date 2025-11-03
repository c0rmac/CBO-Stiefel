# Filename: cbo-stiefel.rb.template

class CboStiefel < Formula
  # 1. METADATA
  desc "A gradient free algorithm that optimises highly non-convex functions over Stiefel Manifolds (Orthogonal matrices of dimension n*k)"
  homepage "https://github.com/c0rmac/CBO-Stiefel/tree/main/cbo-stiefel"

  # These lines will be replaced by the build script (build_cbo_module.sh)
  url "https://github.com/c0rmac/CBO-Stiefel/releases/download/v1.0.0/cbo-stiefel_module-1.0.0-Source.tar.gz"
  version "1.0.0"
  sha256 "52c34e998c5208d2743d331506d9500b4c469679ea983b624d658feed150d874"

  # 2. DEPENDENCIES
  depends_on "cmake" => :build
  depends_on "eigen"
  depends_on "pybind11"
  # Ensures the Homebrew installation finds and uses its own Python interpreter
  depends_on "python"

  # 3. INSTALLATION
  def install
    # Get paths for the Homebrew-installed Python
    py_ver = Formula["python"].version.major_minor
    py_exec = Formula["python"].opt_bin/"python3.#{py_ver}"
    # Define the target site-packages directory
    site_packages = prefix/lib/"python#{py_ver}/site-packages"

    # Set up CMake arguments, passing the explicit Python executable needed
    # by the project's CMakeLists.txt
    args = std_cmake_args
    args << "-DPYTHON_EXECUTABLE=#{py_exec}"

    # Ensure CMake finds Homebrew's installed dependencies
    ENV.prepend_path "CMAKE_PREFIX_PATH", Formula["eigen"].opt_share/"eigen3/cmake"
    ENV.prepend_path "CMAKE_PREFIX_PATH", Formula["pybind11"].cmake_dir
    ENV.prepend_path "CMAKE_PREFIX_PATH", Formula["python"].opt_prefix

    # Configure and build the module
    system "cmake", "-S", ".", "-B", "build", *args
    system "cmake", "--build", "build"

    # Install the compiled module (cbo_module.so) into the user's site-packages folder
    mkdir_p site_packages
    site_packages.install "cbo_module.so"
  end

  # 4. TEST BLOCK
  test do
    # Simple test to confirm the module can be imported by the system's Python
    system Formula["python"].opt_bin/"python3", "-c", "import cbo_module; print(cbo_module)"
  end
end