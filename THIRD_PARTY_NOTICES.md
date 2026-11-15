# Third-party notices

- `third_party/stb_image.h` and `third_party/stb_image_write.h`: stb single-file
  public-domain libraries by Sean Barrett and contributors.
- `third_party/nlohmann/json.hpp`: JSON for Modern C++ 3.11.3, MIT license.
- `third_party/glad/gl.h`, `third_party/glad/gl.c`: generated Glad 2 OpenGL 4.5
  loader. The generated files retain their upstream SPDX/license header.
- `third_party/EGL/` and `third_party/KHR/khrplatform.h`: Khronos EGL 1.5 and
  platform headers, Apache-2.0/MIT-style Khronos licensing as stated in the files.
- GLFW is not vendored to avoid shipping platform binaries. Install GLFW 3.3+ with
  vcpkg, place its source in `third_party/glfw`, or pass the local include/library
  paths to CMake. GLFW uses the zlib/libpng license.
