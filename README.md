# Indoor Deferred Renderer

面向室内大场景的 OpenGL 4.5 延迟渲染器。场景由固定种子生成，包含数百个静态点光源、静态室内构件和可移动物体。实现覆盖分层程序化材质、G-Buffer、逐光源立方体阴影图、增量阴影更新、版本账本、LRU 阴影缓存、窗口缩放、OpenGL 上下文丢失恢复以及线程安全资源迁移队列。

项目不依赖外部模型或纹理。GLAD 已随仓库提供，所有材质纹理由 CPU 确定性生成。

## 构建与运行

### Windows / Visual Studio 2022

依赖：CMake 3.20+、支持 C++17 的 MSVC、vcpkg 安装的 `glfw3` 和 `glm`。

```powershell
vcpkg install glfw3 glm

cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=<VCPKG_ROOT>/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release

.\build\Release\indoor_deferred.exe --seed 1337 --lights 256
```

### Linux

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/indoor_deferred --seed 1337 --lights 256
```

无头 GPU 验证还需要 EGL 开发包，例如 Ubuntu 上的 `libegl-dev`。主程序本身只依赖 GLFW、GLM 和桌面 OpenGL 4.5 驱动。

### 命令行

```text
--seed N          固定场景种子，默认 1337
--lights N        点光源数量，默认 256
--width N         初始宽度，默认 1280
--height N        初始高度，默认 720
--cache N         阴影缓存容量，默认等于光源数量
--shadow-size N   固定立方体阴影图边长；缺省时随窗口尺寸计算
--test-context-loss  真实 GLFW 窗口渲染一帧后自动丢失/重建上下文、
                     逐像素比较并退出
```

相同 `--seed`、`--lights` 和场景代次必然生成相同的光源、物体、材质参数和动画轨迹。程序启动时会打印场景摘要：

```text
Initial: seed=1337 lights=24 objects=161 digest=0de3a062f5270c7e ...
```

## 操作

| 输入 | 功能 |
| --- | --- |
| `1`..`5` | 直接选择五个材质层组合预设 |
| `T` | 循环切换材质层组合预设 |
| `F1` | 最终图像 |
| `F2` | 世界坐标 G-Buffer |
| `F3` | 视空间法线 G-Buffer |
| `F4` | 合成后的基础色 G-Buffer |
| `F5` | 粗糙度/金属度 G-Buffer |
| `F6` | 光源阴影版本热力图 |
| `F7` | 阴影缓存占用率条 |
| `Tab` | 循环调试通道 |
| `P` | 暂停/恢复确定性物体动画 |
| `M` | 单步移动 1/30 秒 |
| `X` | 通过资源管理线程请求删除一个光源 |
| `R` | 通过资源管理线程请求确定性场景切换 |
| `L` | 模拟 OpenGL 上下文丢失并自动恢复、逐像素比较 |
| 鼠标左键拖动/滚轮 | 轨道相机旋转/缩放 |

窗口标题实时显示光源总数、有效阴影图数、缓存占用、累计与当前帧版本迁移数、材质预设、调试通道、当前脏光源数、本帧重绘阴影图数和上下文恢复比较结果。

## 渲染管线

### G-Buffer

G-Buffer 使用四个颜色附件：

| 附件 | 格式 | 内容 |
| --- | --- | --- |
| 0 | `RGBA16F` | 世界坐标 XYZ，视空间正深度 |
| 1 | `RGBA16F` | 视空间法线 XYZ，对象 ID |
| 2 | `RGBA8` | 组合后的线性基础色 RGB，AO |
| 3 | `RGBA16F` | 粗糙度、金属度、材质预设、对象 ID |

布局由 CMake 同时生成 `src/render/GBufferLayout.hpp.in` 对应的 C++ 头和 `shaders/gbuffer_layout.glsl.in` 对应的 GLSL 常量。`gbuffer.frag` 和 `composite.frag` 不使用手写重复的附件序号。C++ 结构体通过 `static_assert` 检查 16 字节布局，CMake 生成过程保证两端的附件顺序一致。

完整帧流程：

1. 对物体实例执行程序化材质三层采样与组合，写入 G-Buffer。
2. 全屏光照阶段按每批 11 个光源遍历整个帧光源快照。OpenGL 4.5 保证的 16 个片元纹理单元中，0..3 用于 G-Buffer，5..15 用于 11 张立方体阴影图。
3. HDR 累加完成后进行 ACES 色调映射和 gamma 合成。
4. 最终图像先写入独立的 `RGBA8` 目标，再 blit 到窗口。这样无头测试、缩放比较和上下文恢复比较使用完全相同的像素路径。

### 分层材质

材质纹理是 8 层、256×256 的 `GL_TEXTURE_2D_ARRAY`，分为基础色层、法线层、粗糙度/金属度层。每个对象选择三层索引、UV 缩放、法线强度、基础色、粗糙度和金属度覆盖值。

五个预设只改变三层在 G-Buffer 写入前的组合方式：

| 预设 | 组合 |
| --- | --- |
| 0 BaseOnly | 仅基础色层 |
| 1 NormalDetail | 基础色 + 法线层 |
| 2 RoughMetal | 基础色 + 粗糙度/金属度层 |
| 3 BaseNormal | 细节基础色混合 + 半强度法线 |
| 4 FullLayered | 细节基础色 + 全强度法线 + 粗糙度/金属度 |

### 阴影版本与增量更新

- 每个光源拥有独立立方体深度纹理和 `uint64` 版本号。
- 每帧在 G-Buffer/光照前捕获 `FrameSnapshot`，包含全部光源、阴影版本、缓存驻留状态和 LRU 顺序。
- 动态物体移动时，用旧/新包围球路径与光源影响半径做保守扫掠检测，只产生受影响的脏光源集合。
- 仅脏光源的阴影图重新渲染一次并令版本号 `+1`；未受影响光源版本完全不变。
- 光照着色器同时接收光源快照版本和已绑定纹理的缓存版本。只有 `snapshotVersion == mapVersion` 时才采样阴影，否则视为无效阴影。
- 窗口缩放不重置版本；上下文恢复也从持久化快照恢复版本。二者之后的移动仍从原版本继续递增。

### 阴影缓存与 LRU

`ShadowCachePolicy` 维护逻辑槽位和 LRU 顺序，GL 层一一对应实际立方体纹理。

- 同一光源重复请求返回已有槽位，不会重复分配纹理。
- 容量不足时淘汰最久未使用的纹理并立即 `glDeleteTextures`。
- 删除光源时释放其缓存槽位、删除 GL 纹理并移除版本账本条目。
- 场景切换释放全部阴影纹理并重建新场景账本。
- 默认容量等于光源数量；使用 `--cache 12 --lights 32` 可直接观察 LRU 淘汰和低占用率。

### 窗口缩放与上下文丢失

窗口缩放只发生在帧边界。G-Buffer、HDR、最终目标和全部阴影纹理按新尺寸重建，版本账本保持不变。测试中从 320×180 缩放到 640×360 再恢复，最终图像与缩放前逐像素一致。

模拟上下文丢失流程：

1. 读取最终目标像素。
2. 不调用任何旧上下文的 `glDelete*`，仅丢弃句柄。
3. 销毁 GLFW 窗口和 OpenGL 上下文并创建新上下文。
4. 从 CPU 持久化的材质纹素、光源快照、对象状态和版本账本重建所有 GL 对象。
5. 按快照的 LRU 顺序重建缓存驻留阴影图。
6. 用恢复后的快照渲染首帧，与丢失前像素逐像素比较。

### 线程模型

`ResourceManager` 工作线程只处理 `ResourceRequest` 并生成普通命令，不包含任何 OpenGL 调用。渲染线程只在 `pollAtFrameBoundary()` 中取出已到达应用帧的 `ResourceCommand`。资源创建、阴影图更新、纹理解除和 FBO 重建全部在渲染线程执行。

`RenderThreadGuard` 会在渲染器公开 GL 操作入口检查当前线程。CPU 测试验证以下栅栏行为：

```text
frame0=0 commands frame1=1 frame5=0 frame6=1
```

即第 0 帧提交的迁移只能在第 1 帧边界生效，渲染中途不会应用。

## 验证

### CPU 测试

```bash
./build/indoor_core_tests
```

本机实际结果：

```text
digest=981aebcbd5cd508a lights=300 objects=161
movedObjects=16 dirtyLights=106 totalLights=256 migrations=106
capacity=3 evicted=202 residentAfterRelease=2
frame0=0 commands frame1=1 frame5=0 frame6=1
All core tests passed.
```

覆盖相同种子完全一致、不同种子不同、300 光源请求、五个预设循环、增量脏集合、未受影响版本不变、受影响版本恰好 `+1`、LRU 淘汰顺序、重复分配阻止、物理释放语义、删除光源和帧边界队列。

### 无头 OpenGL 测试

```bash
LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe \
  ./build/indoor_gpu_tests --artifacts artifacts
```

本机实际结果（WSL2 + Mesa llvmpipe，真实 OpenGL 4.5 Core）：

```text
OpenGL version=4.5 (Core Profile) Mesa 26.0.8
scene digest=ba5054e3580b7ce7 lights=48 objects=161
five material presets produced five distinct final images
movedObjects=16 dirtyLights=15 rerenderedMaps=15
resize round trip restored at 320x180 diff=0
context recovery pixel diff=0 shadowMaps=48
post-recovery version continuity verified
deleted light id=47 cache=47/48
LRU resident=12 evictionsThisFrame=12 totalLights=32
All headless GPU tests passed.
```

GPU 测试实际创建纹理、FBO、VAO、着色器和立方体阴影图，并执行 `glReadPixels` 比较。它还通过 `glIsTexture` 验证删除光源后原纹理句柄已经释放。

### 窗口运行冒烟

```text
OpenGL version: 4.5 (Core Profile) Mesa 26.0.8
Initial: seed=1337 lights=24 objects=161 digest=0de3a062f5270c7e shadowSize=40 cacheCapacity=24
```

GLFW 窗口在 WSLg 下持续渲染 6 秒，由 `timeout` 结束，进程未崩溃。

真实 GLFW 窗口的上下文销毁/重建路径也由自动模式验证：

```bash
./build/indoor_deferred --seed 1337 --lights 16 --width 480 --height 270 \
  --test-context-loss
```

本机实际结果：

```text
Simulating OpenGL context loss...
Context recovery: OK diff=0,
  digest_before=0x590b366c6e82f747, digest_after=0x590b366c6e82f747
```

进程以 0 退出；该模式会真实执行 `glfwDestroyWindow`、新建窗口和重新加载 GL 函数。

### 验证图像

`artifacts/` 中保存了实际运行生成的 PPM 和转换后的 PNG：

```bash
python3 tools/ppm_to_png.py artifacts
```

- `final_initial.png`：初始最终帧。
- `gbuffer_world.png`、`gbuffer_normal.png`、`gbuffer_albedo.png`、`gbuffer_roughmetal.png`：G-Buffer 调试通道。
- `material_preset_0.png` 至 `material_preset_4.png`：五种材质层组合。
- `shadow_versions.png`：版本热力图。
- `shadow_versions_after_move.png`：增量更新后版本号已分化的热力图。
- `cache_occupancy.png`：缓存占用率条。
- `cache_occupancy_after_delete.png`：删除光源后 47/48 的实际占用条。
- `moved_incremental.png`：移动后仅增量更新 15 张阴影图的帧。
- `resize_640x360.png`、`resize_restored.png`：缩放往返验证。
- `context_before.png`、`context_after.png`：上下文丢失前后逐像素一致。
- `lru_cache.png`：容量 12、32 个光源下的实际 LRU 运行结果。
- `validation.txt`：本机测试输出摘要。

## 主要代码位置

- `src/core/DeterministicScene.cpp`：固定种子场景、材质参数、动画和扫掠脏集合。
- `src/core/ShadowSystems.cpp`：版本账本和 LRU 缓存策略。
- `src/core/ResourceManager.cpp`：工作线程和帧边界命令队列。
- `src/render/DeferredRenderer.cpp`：G-Buffer、阴影缓存、分批光照、合成、缩放和恢复。
- `src/render/ProceduralMaterials.cpp`：程序化材质纹理和像素比较工具。
- `src/app/Application.cpp`：GLFW 窗口、输入、标题栏、资源命令应用和上下文丢失模拟。
- `shaders/`：G-Buffer、阴影、光照和调试合成 GLSL。
- `tests/core_tests.cpp`、`tests/gpu_tests.cpp`：CPU 与真实 OpenGL 验证。

## 环境说明

当前执行环境是 WSL2/Linux，未安装 Visual Studio/MSVC，因此本仓库中的实际运行验证使用 GCC 和 Mesa llvmpipe。CMake 与源码已按 MSVC/vcpkg 路径编写；Windows 构建需要在实际 Windows + VS2022 环境中执行上述命令。
