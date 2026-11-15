# 实时骨骼与材质同步 OpenGL 演示

这是一个 C++17、GLFW、GLM、OpenGL 4.5 Core 的本地桌面演示。程序不读取任何外部模型、纹理或着色器文件：骨骼层级、网格、材质预设、动画关键帧和着色器全部内置。运行时每个渲染帧将动画关节世界矩阵与当前材质参数打包进同一个 std140 UBO，并通过一次 `glBufferSubData` 同步到 GPU。

![自测截图](artifacts/self-test.png)

截图来自 `--self-test` 第 100 帧：当前为 `Wave` 动画、`Neon Green` 材质。图中右臂抬起并处于挥手关键帧。

强制编译失败后的安全着色器输出见 `artifacts/shader-fallback.png`：几何和骨骼动画仍正常，材质退化为纯漫反射色，不再执行光照。

## 功能与实现

- 1280x720、OpenGL 4.5 Core、双缓冲窗口；标题显示材质、动画、FPS 和 VSync 状态。
- 7 个层级关节：`Root -> Torso -> Head`，以及左右上臂/前臂。每个关节有一个 `MeshSection`，包含位置、法线、UV 和关节索引。
- 3 个材质预设：`Ceramic Blue`、`Copper`、`Neon Green`，每个包含漫反射色、粗糙度、自发光强度。
- 3 个循环动画：`Idle`、`Walk`、`Wave`。关键帧同时保存关节局部位置和四元数旋转。
- 动画时间以秒计算，用单调时间差累加，不依赖帧计数；空格键循环切换片段。
- 每帧构建一个 704 字节 `FrameUniforms` 包，其中同时包含材质和 8 个关节矩阵槽位。`1/2/3` 切材质，`Space` 切动画。
- 顶点着色器使用关节矩阵做确定性的刚性蒙皮；片段着色器包含环境光、方向光、粗糙度影响的镜面项、自发光和 UV 微纹理。
- 顶点/索引使用 VBO/EBO；运行时 `updateMeshVertices()` 使用 `glBufferSubData` 更新 VBO，自测会实际执行一次。
- UBO 使用 `glBufferSubData` 部分更新：128 字节矩阵头仅在矩阵变化时更新；每帧固定从偏移 128 上传 576 字节的材质/灯光/关节区域，并用 indexed binding 绑定到着色器 `binding = 0`。
- 自定义着色器编译或链接失败时，完整错误日志写入 stderr，并切换到内置纯色安全着色器继续运行。
- `V` 切换 VSync；关闭时使用 5 ms 帧预算将上限限制为 200 FPS。
- 标题每 0.5 秒更新一次 FPS。窗口关闭或 `ESC` 正常释放程序、VAO、VBO、EBO、UBO，然后返回 0。
- 每帧末尾检查 `glGetError`。若连续 3 个渲染帧都有错误，记录 `E1010` 并非零退出。

## 操作

| 按键 | 行为 |
|---|---|
| `1` / `2` / `3` | 选择材质预设 |
| `Space` | `Idle -> Walk -> Wave -> Idle` |
| `V` | 开/关 VSync，关闭时 200 FPS 上限 |
| `ESC` | 清理资源并退出 |

## Windows 构建

推荐使用 Visual Studio 2022 + vcpkg。安装 64 位静态依赖可以避免运行目录中的 GLFW DLL：

```powershell
vcpkg install glfw3:x64-windows-static glm:x64-windows-static

cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake `
  -DVCPKG_TARGET_TRIPLET=x64-windows-static `
  -DDEMO_ENABLE_GPU_TESTS=ON
cmake --build build --config Release
```

手动安装 GLFW 3.3.8+ 和 GLM 0.9.9+ 时，将两者的 CMake 包前缀加入 `CMAKE_PREFIX_PATH`：

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_PREFIX_PATH="C:/deps/glfw;C:/deps/glm" `
  -DDEMO_ENABLE_GPU_TESTS=ON
cmake --build build --config Release
```

MinGW-w64 GCC 13+ 可使用 `-G "MinGW Makefiles"`，参数与上面相同。程序自身不依赖工作目录或资源文件；运行：

```powershell
.\build\Release\skeletal_material_demo.exe
```

Windows 需要显卡驱动提供 OpenGL 4.5 Core。远程桌面或虚拟机若只暴露 OpenGL 1.1，会按设计输出 `E1003` 并以非零码退出，这是环境能力不足而非程序回退路径。

## 自动验证

CPU 测试不创建窗口。GPU 集成测试创建隐藏窗口、渲染 150 个固定 1/60 秒帧、切换材质与动画、执行动态 VBO 更新和 UBO 回读，并统计非背景像素。

```powershell
ctest --test-dir build -C Release --output-on-failure
.\build\Release\skeletal_material_demo.exe --self-test `
  --screenshot artifacts/self-test.ppm
.\scripts\verify_windows.ps1 -BuildDir build -Config Release
```

`verify_windows.ps1` 会逐一检查正常路径、着色器编译失败、链接失败、上下文失败、资源失败和连续 GPU 错误退出码。

Linux 上已有构建目录时：

```bash
./scripts/verify_linux.sh build
```

也可单独运行故障注入：

| 参数 | 预期结果 |
|---|---|
| `--inject-shader-compile-failure` | stderr 记录 `E2001`，使用安全着色器，最终返回 0 |
| `--inject-shader-link-failure` | stderr 记录 `E2002`，使用安全着色器，最终返回 0 |
| `--simulate-context-failure` | stderr 记录 `E1003`，返回 1003 |
| `--simulate-resource-failure` | stderr 记录 `E3001`，返回 3001 |
| `--inject-gl-error` | 连续 3 帧错误后记录 `E1010`，返回 1010 |

`--benchmark-frames 120 --vsync-off` 通过正常渲染循环测量关闭 VSync 后的 FPS，用于核验 200 FPS 上限；隐藏窗口不会干扰桌面。

## UBO std140 契约

CPU `FrameUniforms` 使用 `offsetof` 静态断言；GLSL 的 `FrameData` 块声明必须保持同样顺序。

| 字段 | 字节偏移 | 大小 |
|---|---:|---:|
| `mat4 viewProjection` | 0 | 64 |
| `mat4 model` | 64 | 64 |
| `vec4 diffuseColor` | 128 | 16 |
| `vec4 materialParams` | 144 | 16 |
| `vec4 lightDirection` | 160 | 16 |
| `vec4 viewPosition` | 176 | 16 |
| `mat4 jointWorld[8]` | 192 | 512 |
| 总计 | | 704 |

`materialParams.x/y/z` 分别为粗糙度、自发光强度、动画秒数。CPU 和 GLSL 都声明 8 个关节槽位，实际使用 7 个；第 8 个保持单位矩阵。

每帧动态上传区间为字节 `[128, 704)`；矩阵头为字节 `[0, 128)`，仅在投影或模型矩阵变化时上传。

## 错误码

| 码 | 名称 | 行为 |
|---:|---|---|
| 1001 | `GLFW_INITIALIZATION_FAILED` | 初始化失败，非零退出 |
| 1003 | `OPENGL_45_UNAVAILABLE` | 窗口/GL 4.5 上下文创建或版本检查失败 |
| 1004 | `OPENGL_LOADER_FAILED` | 缺少 4.5 函数入口 |
| 1010 | `PERSISTENT_GL_ERROR` | 连续 3 帧检测到 GL 错误 |
| 2001 | `SHADER_COMPILE_FALLBACK` | 编译失败，安全着色器继续 |
| 2002 | `SHADER_LINK_FALLBACK` | 链接失败，安全着色器继续 |
| 2003 | `SAFE_SHADER_FAILED` | 连安全着色器都失败，非零退出 |
| 3001 | `SCENE_RESOURCE_INVALID` | 程序化网格校验失败 |
| 3002 | `GPU_BUFFER_UPLOAD_FAILED` | VAO/VBO/EBO/UBO 初始化失败 |
| 3003 | `UNIFORM_BUFFER_SYNC_FAILED` | UBO 上传/回读失败 |
| 3004 | `MESH_DYNAMIC_UPDATE_FAILED` | 动态 VBO 更新失败 |

## 代码结构

- `src/scene.*`：关节层级、网格生成与校验、材质数组、关键帧和动画状态机。
- `src/frame_data.*`：std140 帧数据布局与打包。
- `src/gl_loader.*`：最小 OpenGL 4.5 入口加载器，避免额外运行库。
- `src/renderer.*`：着色器编译/回退、VAO/VBO/EBO/UBO、帧同步和绘制。
- `src/application.*`：GLFW 窗口、输入、主循环、FPS/标题、故障注入和自测。
- `tests/cpu_tests.cpp`：确定性 CPU 契约测试。
- `scripts/verify_*.sh|ps1`：端到端错误路径验证。

## 本机实际验证结果

验证环境：WSL2 Ubuntu 26.04、GCC 15.2.0、GLFW 3.4（X11）、Mesa 26.0.8 llvmpipe，OpenGL 4.5 Core、GLSL 4.50。Debug 和 Release 配置均完成完整测试。该环境不是 Windows，故 Windows 图形驱动路径仍需在目标机执行上面的 PowerShell 脚本确认。

实际结果：

- `cpu_tests`：3691 项检查通过。
- GPU 自测：150 帧、150 次逐帧动态 UBO 上传、1 次 UBO 矩阵头上传、1 次动态 VBO 更新、UBO 回读与 CPU 包逐字节一致。
- 自测渲染抽查：62,152 个非背景像素；截图见 `artifacts/self-test.png`。
- 正常自定义着色器：`CustomLighting`，退出码 0。
- 编译失败注入：stderr 包含 Mesa GLSL 预处理器错误并记录 `E2001`，`SafeFallback` 完成 150 帧和 62,152 个可见像素，退出码 0；纯色截图见 `artifacts/shader-fallback.png`。
- 链接失败注入：stderr 包含 varying 类型不匹配日志并记录 `E2002`，`SafeFallback` 完成 150 帧，退出码 0。
- 上下文失败：记录 `E1003` 并非零退出。
- 程序化资源失败：记录 `E3001` 并非零退出。
- 连续 GL 错误：第 1/2/3 帧依次报告 `GL_INVALID_ENUM`，随后记录 `E1010` 并非零退出。
- VSync 关闭基准：120 帧在 llvmpipe 上多次运行实测 `183.52–194.75 FPS`，均低于 200 FPS 上限；脚本验证阈值为 205 FPS，仅吸收计时/调度抖动。

完整输出由 `scripts/verify_linux.sh` 写入 `artifacts/verify-linux/`。
