# Third-party components

The repository includes pinned headers/generated sources so the verifier can be
built in an offline CI environment. Runtime material validation itself uses no
third-party assets.

- `third_party/glad`: GLAD 2.0.2-generated OpenGL 4.5 Core loader. The generated
  header carries the SPDX expression
  `(WTFPL OR CC0-1.0) AND Apache-2.0`.
- `third_party/glm`: GLM 1.0.1 headers, MIT license text in
  `third_party/glm/COPYRIGHT`.
- `third_party/nlohmann`: nlohmann/json 3.11.3 headers, MIT license text in the
  headers.
- GLFW is resolved from vcpkg and is not vendored.
