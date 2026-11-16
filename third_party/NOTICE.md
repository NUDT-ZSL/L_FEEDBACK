# Third-Party Notices

The repository vendors header-only or generated sources so the renderer can build and be validated without network access:

- **GLM** (`third_party/glm`): MIT license, https://github.com/g-truc/glm
- **nlohmann/json** (`third_party/nlohmann`): MIT license, https://github.com/nlohmann/json
- **stb_image** (`third_party/stb/stb_image.h`): public domain / MIT dual license, https://github.com/nothings/stb
- **GLAD-generated OpenGL loader** (`third_party/glad`): generated from Khronos OpenGL Registry data; Khronos headers are Apache-2.0/MIT compatible. https://github.com/Dav1dde/glad
- **Khronos EGL / KHR headers** (`third_party/EGL`, `third_party/KHR`): Apache-2.0 / MIT compatible, https://registry.khronos.org/EGL/

The primary supported dependency flow remains the `vcpkg.json` manifest (`glfw3`, `glm`, `stb`, `nlohmann-json`). The vendored copies are fallback build inputs.

