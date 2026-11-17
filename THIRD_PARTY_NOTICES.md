# Third-Party Notices

This project vendors dependency source/header material to build offline.

| Component | Location | Version | License |
|---|---|---|---|
| GLFW | `third_party/glfw` | 3.4 | zlib/libpng-like MIT-style license in `third_party/glfw/LICENSE.md` |
| GLM | `third_party/glm` | 1.0.1 headers | MIT license in `third_party/glm/COPYING.txt` |
| GLAD generated loader | `third_party/glad` | Glad 2.0.8, OpenGL 4.3 core | WTFPL OR CC0-1.0 AND Apache-2.0; see `third_party/glad/LICENSE` and generated header notices |
| stb_image_write | `third_party/stb/stb_image_write.h` | 1.18 | Public domain / MIT alternative, stated in the header |
| Mesa `GL/gl.h` compatibility header | `third_party/glvnd/GL/gl.h` | GLVND/Mesa header | MIT-style Mesa license retained in the file header |

The application source itself contains embedded GLSL and deterministic scene data. It does not
download dependencies or assets at runtime.
