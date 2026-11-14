# Deferred Fog

一个从零实现的 OpenGL 4.5 延迟渲染管线示例，包含：

- 4 个 `RGBA32F` MRT G-Buffer（位置、法线、反照率/粗糙度、金属度）；
- 基于深度纹理重建世界坐标的等距步进体积雾；
- 屏幕空间半透明物体远到近排序、前乘 Alpha 混合和透明深度隔离；
- 版本化光源快照，仅在位置、颜色或强度变化时重新上传 GPU；
- JSON 配置、确定性内置场景、离线 PNG/PFM/JSON 验证产物；
- CPU 确定性参考后端，用于无 GPU 环境下的逻辑和像素级回归测试。

## 渲染流程

1. **几何阶段**：不透明物体写入 4 个 `GL_RGBA32F` 颜色附件和
   `GL_DEPTH_COMPONENT32F` 深度纹理。任何附件或 FBO 不完整都会抛出
   `E_GBUFFER_FAIL`。
2. **延迟光照**：全屏 Pass 从 G-Buffer 读取材质和世界坐标，使用版本化 UBO
   中的点光/方向光计算 HDR 颜色。
3. **体积雾**：从深度纹理重建世界坐标，沿相机射线等距采样确定性 3D FBM
   密度场；每个样本计算点光/方向光散射并累积透射率。雾以
   `ONE, ONE_MINUS_SRC_ALPHA` 前向混合到线性 HDR 场景。
4. **半透明阶段**：每帧按视空间深度从远到近排序，关闭深度写入并以前乘 Alpha
   绘制。共享的雾深度纹理保留不透明场景深度，透明像素不会写入其中。
5. **显示/读回**：Reinhard 色调映射和 sRGB 编码后读回颜色；深度 PFM 使用
   视空间线性深度，背景为相机远平面。

## 构建（Windows / MSVC）

要求 CMake 3.20+、VS 2019/2022、OpenGL 4.5 驱动。GLAD 2.0、stb 和
nlohmann/json 3.11.2 已放在 `third_party/`，无需下载。

使用 vcpkg 安装 GLFW：

```powershell
vcpkg install glfw3:x64-windows
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

也可以使用本地 GLFW：

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DGLFW_ROOT=C:\libs\glfw
```

`GLFW_ROOT` 下应包含 `include/GLFW/glfw3.h` 和 `lib/glfw3.lib`。

Linux/无显示环境可显式关闭 GPU 后端：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDEFERRED_ENABLE_GPU=OFF
cmake --build build -j
```

## 运行

GPU 构建中 `auto` 离线模式优先使用 OpenGL；`--backend cpu` 可运行确定性参考实现。

```powershell
# 离线 OpenGL 验证，固定 32 步体积雾
.\build\Release\deferred_fog.exe --config config\render.json --offline `
  --backend auto --output artifacts\offline_gpu

# CPU 参考产物
.\build\Release\deferred_fog.exe --config config\render.json --offline `
  --backend cpu --output artifacts\offline_cpu

# 验证并可选比较两个后端
.\build\Release\deferred_verify.exe artifacts\offline_gpu artifacts\offline_cpu

# 交互模式：相机路径和 10 Hz 光源状态变化
.\build\Release\deferred_fog.exe --config config\render.json --interactive
```

离线产物：

- `color.png`：颜色缓冲区；
- `depth.pfm`：单通道 32 位浮点视空间深度，PFM 行序符合规范（自底向上）；
- `transparent_objects.json`：排序模式、近似标志、rank、id、视深、透明度和位置；
- `frame_metadata.json`：相机、光源、快照版本/上传次数、G-Buffer 布局、哈希；
- `gbuffer_layout.txt`：固定 MRT 布局。

## 配置

启动时读取一次，运行中不热重载。未知字段、类型错误、越界值都会以
`E_CONFIG_PARSE` 终止。

| 字段 | 范围 | 说明 |
|---|---:|---|
| `render.width/height` | 64-4096 | 输出分辨率 |
| `render.maxFrames` | 0-100000 | 交互模式帧数上限，0 为无限 |
| `volumetric.enabled` | bool | 是否启用体积雾 |
| `volumetric.steps` | 16-64 | 交互模式步数；离线固定覆盖为 32 |
| `volumetric.densityScale` | 0-2 | 密度缩放 |
| `volumetric.maxDistance` | 1-200 | 最大积分距离 |
| `volumetric.noiseScale` | 0.001-2 | 3D 噪声频率 |
| `volumetric.absorption/scattering` | 0-8 | 吸收/散射系数 |
| `transparent.sortingEnabled` | bool | 关闭时保持场景顺序 |
| `transparent.exactSortLimit` | 1-1000000 | 超过后切换桶排序，默认 1024 |
| `transparent.bucketCount` | 16-16384 | 近似桶数量 |
| `lights.maxLights` | 1-32 | 上传和着色最大光源数 |

## 光源快照

`LightSnapshot::sync()` 仅在数组长度或光源位置、方向、颜色、强度发生变化时递增
版本。延迟光照与体积雾分别检查版本，仅在需要时上传 UBO。交互模式每 100 ms
改变一次内置光源状态，因此大多数渲染帧不会重复上传。

## 半透明排序

- `count <= exactSortLimit`：`std::stable_sort` 精确按视深降序；
- `count > exactSortLimit`：O(n + bucketCount) 计数式桶排序，桶内保持稳定场景
  顺序，允许一个桶宽以内的深度反序；
- 两种模式都输出前乘 Alpha 标记与 `doesWriteFogDepth=false`；
- 单元测试对 1025 个透明物体同时渲染精确结果与 64 桶近似结果，阈值为平均差
  `<= 1/255`、P99 `<= 8/255`。

## 错误码

错误写入 stderr，格式为 `业务码: 具体信息`，进程退出码在 1-255。

| 业务码 | 退出码 | 场景 |
|---|---:|---|
| `E_CONFIG_PARSE` | 2 | 配置缺失、JSON 语法/类型/范围错误 |
| `E_GBUFFER_FAIL` | 3 | G-Buffer 纹理或 FBO 创建失败 |
| `E_SHADER_COMPILE` | 4 | 着色器编译或链接失败 |
| `E_IO` | 5 | 输入/输出文件失败 |
| `E_GL_CONTEXT` | 6 | GLFW/OpenGL 4.5 上下文失败 |
| `E_ARGS` | 7 | 命令行参数错误 |
| `E_RENDER_FAIL` | 8 | 其他 GL/FBO 运行失败 |

`--inject-gbuffer-failure` 与 `--inject-shader-failure` 仅用于验证错误路径。

## 自动验证

```powershell
.\scripts\run_validation.ps1 -Gpu
```

```bash
./scripts/run_validation.sh
GPU_BACKEND=1 ./scripts/run_validation.sh
```

脚本运行核心测试、离线渲染、产物结构/哈希/排序校验，并注入配置、G-Buffer
和着色器错误检查退出码。结果写入 `artifacts/`。

## 实际验证结果

验证环境：WSL2 Ubuntu、Mesa `llvmpipe (LLVM 21.1.8, 256 bits)`、OpenGL 4.5
Core（GPU 代码实际创建上下文并离屏渲染）。项目同时提供 MSVC/GLFW 构建配置，
但本次执行环境没有 Windows/MSVC。

- 核心测试：全部通过；1025 个透明物体桶排序相对精确排序平均像素差
  `0.00253987/255`，P99 `0`，满足 `1.0/8.0` 阈值。
- CPU 离线：`640x360`，32 步，4 个透明排序对象，颜色哈希
  `17506978379173897026`，深度哈希 `8512139502578907713`。
- OpenGL 离线：同一场景 32 步，排序顺序 `[24, 22, 21, 23]` 与 CPU 一致；
  颜色哈希 `7313631885968731777`，深度哈希 `4824537670985800409`。
- CPU/OpenGL 比较：颜色平均差 `2.673/255`、P99 `19/255`；深度平均差
  `0.1422`、P99 `0.0060`。差异来自光线求交参考实现与光栅化的轮廓/采样差别。
- 错误注入：`E_CONFIG_PARSE -> 2`、`E_GBUFFER_FAIL -> 3`、
  `E_SHADER_COMPILE -> 4`，stderr 与退出码均符合约定。

完整机器可读结果见 `artifacts/verification_report.json`、
`artifacts/verification_gpu_vs_cpu.json` 和 `artifacts/error_codes.json`。
