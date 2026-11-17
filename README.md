# Offline Deterministic Software Character Rasterizer

一个从零实现的 Windows 离线软光栅化三维角色渲染器，用于在无 GPU 的低配目标机上验证最终画面。项目仅依赖 C++20 标准库与 CMake，不使用 DirectX、OpenGL、Vulkan、D3D、DirectXMath 或任何第三方图形/JSON 库。

渲染器读取场景 JSON，输出二进制 P6 PPM 与渲染统计 JSON。仓库内的角色场景由程序确定性构造，也可用同一场景格式提供其他网格与材质数据。

## 功能

- C++20/CMake 控制台程序，无 GPU、无外部资源。
- 模型/视图/透视/视口变换、六平面齐次裁剪、背面剔除、32 位浮点 Z-Buffer。
- 逐三角形 UV 微分构造切线空间，再按顶点累加并正交化；法线非单位、法线为零及 UV 退化时有确定性回退。
- 程序内嵌切线空间法线贴图，双线性采样与 UV wrap。
- 定向光与最多 8 个点光源的 Lambert 漫反射和 Phong 高光；点光源按到片元的真实欧氏距离做平方反比衰减。
- 定向光正交阴影贴图，百分比过滤（PCF），深度偏移与法线偏移共同抑制阴影痤疮。
- 不透明物体先绘制并写深度；透明三角形按相机空间质心深度全局从远到近排序，在线性空间 alpha 混合。
- 输出 gamma 2.2 校正后的 PPM、SHA-256、完整渲染统计和确定性内存估算。
- 严格输入校验；失败时只在 stderr 输出 JSON 错误，不创建 PPM 或统计文件。
- 自带 C++ 单元测试与 Python 端到端验收测试。

## Windows 构建

前置：

- Windows 10/11
- Visual Studio 2022 或 Build Tools，安装“使用 C++ 的桌面开发”
- CMake 3.25+

在 “x64 Native Tools Command Prompt for VS 2022” 中执行：

```bat
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

也可以使用 Ninja：

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

本实现自带行主序 `Mat4`/`Vec3` 数学库，因此不需要 DirectXMath，也就不存在 NuGet 或本地头文件版本固定问题。若要改用 DirectXMath，可替换 `src/math.*`，但当前交付版本刻意不引入任何额外依赖。

## 运行

生成程序内确定性角色场景：

```bat
build\Release\softraster.exe --generate-sample sample.json
```

渲染：

```bat
build\Release\softraster.exe sample.json sample.ppm sample.stats.json
```

省略统计路径时使用 `<output.ppm>.stats.json`：

```bat
build\Release\softraster.exe sample.json sample.ppm
```

退出码为 0 表示成功。任何失败都会向 stderr 写一行 JSON，例如：

```json
{"error":{"code":3,"name":"MISSING_FIELD","message":"missing required field 'target'","path":"camera.target"}}
```

失败时进程退出码与 JSON 中的 `code` 相同，且不会生成半成品 PPM/统计文件。

## 场景与约定

完整字段说明见 [docs/SCENE_FORMAT.md](docs/SCENE_FORMAT.md)。核心约定：

- 右手坐标系；相机看向自身局部 `-Z`；世界空间逆时针三角形为正面。
- 相机空间深度越远越小；透明三角形按质心相机 Z 升序绘制。
- 阴影深度和屏幕 Z-Buffer 都存为 32 位浮点，范围为 `[0, 1]`。
- 定向光的 `direction` 是“从表面指向光源”的单位化方向。
- 材质 `opacity == 1` 为不透明，`opacity < 1` 为半透明。
- 当透明三角形 `opacity < transparency.depthWriteMaxOpacity` 时才写 Z-Buffer；否则只做深度测试和混合。这是任务规定的透明深度写入语义。
- 任意两个半透明三角形具有完全相同的 float 相机空间质心深度时，返回 `DUPLICATE_TRANSPARENT_DEPTH`，避免不确定排序。
- 法线贴图 `data` 是按行排列的切线空间向量，长度必须为 `width * height * 3`；采样后归一化，零向量回退到几何法线。

## 统计口径

统计 JSON 的必需字段及口径：

- `rasterizedTriangles`：主相机 pass 中经过裁剪与背面剔除后至少覆盖一个采样点的源三角形数（每个源三角形最多计一次，不含阴影 pass）。
- `clippedTriangles`：主相机 pass 中至少一个顶点位于六个裁剪平面之外的源三角形数。
- `shadowSampleCount`：主相机片元着色时实际读取阴影贴图的 texel 次数；一次 `pcfRadius = r` 的 PCF 最多 `(2r+1)^2` 次。
- `transparentSortSwaps`：把半透明三角形从输入顺序变为“远到近”顺序所需的最少相邻交换数（逆序对数），由 Fenwick 树确定性计算。
- `peakMemoryEstimatedBytes`：场景 DOM、网格/切线与实例副本、颜色/32 位深度缓冲、阴影深度、透明排序数组、PPM 和固定开销的保守峰值估算。
- `shadowMapSize`：当前运行的阴影贴图分辨率，必须与场景 `directionalLight.shadow.mapSize` 一致。
- `outputImageSha256`：完整 PPM 文件（含 P6 头）的 SHA-256。

另有 `opaqueFragments`、`transparentFragments`、`alphaBlendedFragments`、`transparentDepthWrites`、`transparentTriangles`、`ppmBytes` 和视口信息，便于核对一致性。

## 错误码

| Code | Name | 含义 |
| ---: | --- | --- |
| 1 | `IO_ERROR` | 场景输入文件无法打开或读取 |
| 2 | `JSON_PARSE_ERROR` | JSON 语法错误、重复键等 |
| 3 | `MISSING_FIELD` | 缺少必填字段 |
| 4 | `TYPE_MISMATCH` | JSON 类型不匹配 |
| 5 | `INVALID_VALUE` | 越界、非整数、不可逆矩阵等 |
| 6 | `NON_FINITE_FLOAT` | `NaN`/无穷或超出 32 位浮点范围 |
| 7 | `INDEX_OUT_OF_RANGE` | 顶点/网格/材质索引越界 |
| 8 | `SHADOW_MAP_TOO_LARGE` | 阴影贴图尺寸大于 8192 |
| 9 | `DUPLICATE_TRANSPARENT_DEPTH` | 半透明排序深度完全相同 |
| 10 | `MEMORY_LIMIT_EXCEEDED` | 估算峰值超过 `limits.maxMemoryBytes` |
| 11 | `OUTPUT_WRITE_ERROR` | PPM 或统计文件写入失败 |

## 验证

当前环境实际验证了 Release 构建、`-Werror` 构建、双次逐字节一致渲染、SHA-256、透明排序统计、阴影采样统计、法线/UV 回退，以及所有主要错误路径。结果与命令见 [docs/VERIFICATION.md](docs/VERIFICATION.md)。

本机验证环境为 WSL2/Linux + GCC 15 + CMake 3.30；代码与 CMake 配置面向 MSVC/Windows，但当前容器中没有 MSVC，Windows 构建步骤需在目标机执行。

## 代码结构

- `src/main.cpp`：CLI、原子写出、统计 JSON。
- `src/scene.*`：场景模型、JSON 字段解析和输入校验。
- `src/renderer.*`：裁剪、光栅化、切线空间、光照、阴影和透明排序。
- `src/math.*`：自成体系的向量/矩阵/投影数学。
- `src/json.*`：无第三方依赖的 JSON DOM 解析与确定性写出。
- `src/sha256.*`：PPM SHA-256。
- `src/sample_scene.*`：程序内确定性角色场景。
- `tests/unit_tests.cpp`：数学/JSON/SHA/双次渲染单元测试。
- `tests/integration.py`：CLI、统计、错误码和无半成品文件测试。
