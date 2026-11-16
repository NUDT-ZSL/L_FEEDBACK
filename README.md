# Offline Multi-light HDR Preview Renderer

面向 Windows 10/11 的小型离线静帧预览工具。程序从 JSON 读取相机、三角网格、材质和最多 32 个光源，使用 OpenGL 4.5 Core 的延迟着色管线输出 8 位 PNG。

## 已实现内容

- 严格 JSON 场景解析与字段校验，业务错误写入 stderr 并返回退出码 1。
- Geometry Pass 使用 4 个 MRT 附件：
  - `RGBA32F` 世界位置与前景掩码
  - `RGBA16F` 世界法线
  - `RGBA16F` 线性反照率
  - `RG16F` 金属度/粗糙度
- 光照 Pass 逐光源计算 Cook-Torrance GGX 直接光照，覆盖点光、方向光、聚光。
- 所有 HDR 颜色保持在线性浮点纹理中，最终执行 ACES fitted 或 Reinhard 色调映射。
- 阴影图集按 `columns x rows` 行优先分配：
  - 方向光和聚光各占 1 个 tile
  - 点光占 6 个 tile，对应立方体六个面
  - 所有类型共用同一个分配器和图集深度纹理
- PCF 软阴影；核尺寸、采样步长、采样数、tile 分辨率和图集布局全部来自场景配置。
- 光照后对聚合阴影可见度做配置化 2D 空间滤波。默认权重为 5 抽头可分离高斯核，执行两次可得到至少 `5x5` 支撑域；只修改原始可见度 `< 0.999` 的阴影像素。
- 记录滤波前后的阴影可见度最小/平均/最大值、阴影像素数和前景百分比。
- GL 资源计数器、RAII 清理和 `--self-test-resources` 重复初始化/清理测试。
- `fast` / `high` 两种模式共用解析和管线，仅使用配置中的模式参数。

## 目录

```text
include/renderer/     场景、数学、日志、渲染器接口
src/                  解析器、OpenGL 管线、CLI、PNG 输出
tests/                解析单测与端到端 CTest
third_party/glad/     Glad 2.0.2，OpenGL 4.5 Core
third_party/stb/      stb_image 与 stb_image_write
include/nlohmann/     JSON for Modern C++ 3.11.3
```

## Windows 构建

要求 Visual Studio 2022（C++17）、CMake 3.20+ 和支持 OpenGL 4.5 的驱动。

推荐用 vcpkg 提供 GLFW：

```bat
vcpkg install glfw3:x64-windows
cmake -S . -B build -A x64 ^
  -DCMAKE_TOOLCHAIN_FILE=C:/src/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

如果没有安装 GLFW，CMake 默认通过 `FetchContent` 下载 GLFW 3.3.9。完全离线构建时应关闭下载：

```bat
cmake -S . -B build -A x64 -DRENDER_FETCH_GLFW=OFF ^
  -Dglfw3_DIR=C:/deps/glfw/lib/cmake/glfw3
```

Glad 4.5 loader、stb 和 JSON 头文件已随仓库提供。

## 生成样本并运行

样本和错误样本不是附件，全部由可执行文件确定性生成：

```bat
build\Release\render.exe --generate-samples samples
```

生成：

- `samples\scene.json`：4 个网格、4 种材质、方向光/点光/聚光各一个
- `samples\invalid_missing_camera.json`
- `samples\invalid_atlas_capacity.json`

快速预览：

```bat
build\Release\render.exe --scene samples\scene.json --mode fast ^
  --log out\fast.log --output out\fast.png
```

高质量：

```bat
build\Release\render.exe --scene samples\scene.json --mode high ^
  --log out\high.log --output out\high.png
```

只校验场景，不创建 OpenGL 上下文：

```bat
build\Release\render.exe --scene samples\scene.json --validate --log out\validate.log
```

重复初始化/清理资源自检：

```bat
build\Release\render.exe --scene samples\scene.json --mode fast ^
  --log out\resources.log --self-test-resources
```

未指定 `--output` 时，PNG 路径取场景 `settings.output`。相对路径相对于场景文件所在目录；命令行 `--output` 相对于当前工作目录。

## 场景 JSON

完整字段如下。所有数值和拓扑均来自该外部文件，程序内不存在样本拓扑。

```json
{
  "version": 1,
  "camera": {
    "position": [4.6, 3.2, 7.5],
    "target": [0.0, 1.05, -0.2],
    "up": [0.0, 1.0, 0.0],
    "fovYDegrees": 43.0,
    "near": 0.1,
    "far": 60.0
  },
  "materials": [
    {
      "name": "copper",
      "albedo": [0.72, 0.28, 0.12],
      "metallic": 0.92,
      "roughness": 0.24
    }
  ],
  "meshes": [
    {
      "name": "mesh0",
      "material": 0,
      "transform": [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
      ],
      "vertices": [
        [0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0]
      ],
      "indices": [0, 1, 2]
    }
  ],
  "lights": [
    {
      "type": "point",
      "position": [2.8, 2.25, 1.25],
      "color": [0.18, 0.5, 1.0],
      "intensity": 22.0,
      "range": 9.0,
      "castShadow": true,
      "near": 0.1,
      "far": 10.0,
      "shadowBias": 0.0025
    },
    {
      "type": "directional",
      "direction": [0.5, -1.0, 0.25],
      "color": [1.0, 0.9, 0.72],
      "intensity": 2.4,
      "orthoSize": 8.0,
      "castShadow": true,
      "near": 0.1,
      "far": 30.0,
      "shadowBias": 0.0015
    },
    {
      "type": "spot",
      "position": [-2.8, 3.1, 0.5],
      "direction": [0.65, -1.0, -0.15],
      "color": [1.0, 0.38, 0.12],
      "intensity": 38.0,
      "range": 12.0,
      "innerConeDegrees": 15.0,
      "outerConeDegrees": 28.0,
      "castShadow": true,
      "near": 0.1,
      "far": 13.0,
      "shadowBias": 0.0018
    }
  ],
  "settings": {
    "width": 640,
    "height": 360,
    "output": "preview.png",
    "background": [0.008, 0.01, 0.018],
    "exposure": 1.15,
    "toneMapping": "aces",
    "shadowAtlas": {
      "tileResolution": 256,
      "columns": 4,
      "rows": 4,
      "pcfKernel": 5,
      "pcfStep": 1.0
    },
    "denoise": {
      "filterRadius": 2,
      "weights": [0.06136, 0.24477, 0.38774, 0.24477, 0.06136]
    },
    "modes": {
      "fast": {
        "shadowSamples": 1,
        "postprocessIterations": 1,
        "atlasScale": 0.5,
        "gaussianFilter": false
      },
      "high": {
        "shadowSamples": 25,
        "postprocessIterations": 2,
        "atlasScale": 1.0,
        "gaussianFilter": true
      }
    }
  }
}
```

约束：

- `lights` 数量为 1..32。
- 点光阴影占用 6 个 tile；所有投影光源的 tile 需求不得超过 `columns * rows`。
- `vertices` 每项为 `[x,y,z,nx,ny,nz]`，`indices` 为逆时针三角形索引。
- `transform` 是 16 个行主序数值。
- `filterRadius` 必须等于 `(weights.size()-1)/2`，并且至少有 3 个奇数长度权重。
- `pcfKernel` 必须是 1..15 的奇数；模式采样数不得超过 `pcfKernel^2`。
- `fast` 必须关闭高斯滤波；`high` 必须开启，并增加采样数、tile 比例和后处理迭代次数。

两种模式的有效 shadow tile 分辨率为：

```text
round(shadowAtlas.tileResolution * modes.<mode>.atlasScale)
```

样本中快速模式为 128，高质量模式为 256；图集分别扩大到 `512x512` 和 `1024x1024`。

## 渲染与后处理顺序

```text
Geometry Pass (4x MRT + depth)
    -> Shadow Atlas Pass (1 tile per directional/spot, 6 tiles per point)
    -> Lighting Pass (Cook-Torrance HDR + shadow visibility)
    -> optional iterative shadow visibility filter
    -> composite shadow correction
    -> ACES/Reinhard + sRGB
    -> RGBA8 readback -> PNG
```

降噪使用独立的 `R16F` 阴影可见度纹理。合成着色器仅在原始可见度 `< 0.999` 时应用 `filtered / original` 修正，因此无阴影区域的 HDR 颜色不被修改。滤波前后都在前景掩码内统计最小、平均和最大阴影强度。

## 命令行与错误码

```text
render --scene <scene.json> --mode <fast|high> --log <log> [--output <png>]
       [--self-test-resources]
render --scene <scene.json> --validate [--log <log>]
render --generate-samples <directory>
render --help
```

错误使用一行可解析记录写入 stderr，并返回退出码 1：

```text
timestamp=2026-11-16T16:35:06.962Z level=ERROR event=fatal error=E_INVALID_SCENE message="root.camera: missing required field"
```

主要错误码：

- `E_USAGE`：命令行错误
- `E_SCENE_IO` / `E_SAMPLE_IO`：场景或样本读写失败
- `E_INVALID_SCENE`：JSON 语法、字段、范围、拓扑或图集容量非法
- `E_GL_CONTEXT`：GLFW/OpenGL 4.5 上下文失败
- `E_GL_RESOURCE`：着色器或 GL 对象创建失败
- `E_RENDER`：渲染 Pass 或 GL 查询失败
- `E_OUTPUT_IO`：PNG 写入失败
- `E_LOG_IO`：日志写入失败
- `E_RESOURCE_LEAK`：清理后资源计数不为零
- `E_INTERNAL`：未归类的运行时异常

## 日志格式

所有日志行都是 `key=value`，每行包含 UTC 时间戳和 `error` 字段。关键记录：

```text
timestamp=... level=INFO event=light_atlas error=OK light_index=1 light_type=point atlas_slot=1 face_count=6 face_slots="1,2,3,4,5,6"
timestamp=... level=INFO event=gpu_timings error=OK geometry_gpu_ms=4.978 shadow_gpu_ms=3.169 lighting_gpu_ms=7.845
timestamp=... level=INFO event=shadow_stats error=OK filter_applied=1 filter_iterations=2 filter_radius=2 before_min=0.000000 before_mean=0.962340 before_max=1.000000 after_min=0.000000 after_mean=0.962282 after_max=1.000000 geometry_pixels=111914 shadowed_pixels=10996
timestamp=... level=INFO event=image_stats error=OK non_background_pixels=111914 total_pixels=230400 non_background_percent=48.573785 output=out/fast.png
timestamp=... level=INFO event=resource_balance error=OK stage=render_shutdown shader_programs=0 textures=0 framebuffers=0 renderbuffers=0 buffers=0 vertex_arrays=0 queries=0 total=0
```

GPU 耗时来自 `GL_TIME_ELAPSED` 查询，查询结果在 `glFinish()` 后读取；实际数值会随机器和驱动变化。

## 实际验证结果

本次在 WSL2 Ubuntu 26.04、Mesa llvmpipe（OpenGL 4.5 Core Profile，GLSL 4.50）上完成真实 OpenGL 管线运行。Windows 构建使用相同的 CMake/C++17/GLFW/Glad 路径，但当前容器未执行 MSVC 本地构建。

执行：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

结果：

- `scene_validation`：通过；样本生成两次逐字节一致；方向光/点光/聚光解析正确；tile 需求 `8/16`。
- `render_integration`：通过。
- 快速模式两次运行 PNG SHA-256 均为：
  `c632c72796937e4c92dd2af3e11da7410c6a7d6cc21080a5dac8b829cc2393f2`
- 高质量模式两次运行 PNG SHA-256 均为：
  `eb0ff109526d458a79df5fc199b91d6c0be01a16168d49dd452c444a6f012a01`
- 快速/高质量输出不同，证明模式参数实际生效。
- 两张 PNG 均为 `640x360`、8 位 RGB PNG。
- 前台像素统计：`111914 / 230400 = 48.573785%`。
- 图集槽位：方向光 `0`，点光 `1..6`，聚光 `7`。
- 非法缺相机场景和非法图集容量场景均返回 `1`，stderr 包含 `E_INVALID_SCENE`。
- 资源自检连续初始化/清理两次，每次初始化 40 个 GL 对象，清理后资源计数为 `0`。
- 日志包含时间戳、错误码、图集槽位、`geometry_gpu_ms`、`lighting_gpu_ms`、滤波前后阴影统计和前景百分比。
- 额外以 `-fsanitize=undefined` 构建后，解析测试和端到端渲染测试均通过。

确定性说明：相同机器、相同驱动、相同场景和参数下，管线不使用随机数、时间相关输入或非确定采样序列，PNG 像素完全一致。不同 GPU/驱动之间不承诺位级一致。
