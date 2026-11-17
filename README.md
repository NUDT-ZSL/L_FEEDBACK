# CPU Terrain Renderer

一个不依赖 GPU 或图形库的离线软件光栅化地形渲染原型。程序从 JSON 配置读取确定性高度场、相机、LOD、光照和输出参数，在纯 CPU 上执行完整的地形渲染管线，输出 24-bit BMP 或 binary PPM 图像及 JSON 统计报告。

## 构建

要求 CMake 3.20+ 和 C++17 编译器。项目没有第三方依赖。

Linux / MinGW-w64：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Windows + Visual Studio 2019：

```powershell
cmake -S . -B build -G "Visual Studio 16 2019" -A x64
cmake --build build --config Release
```

## 运行

```bash
./build/terrain_renderer --config configs/default.json
```

默认输出：

- `renders/terrain.bmp`
- `renders/report.json`

只校验配置而不渲染：

```bash
./build/terrain_renderer --config configs/default.json --validate-only
```

非法配置示例：

```bash
./build/terrain_renderer --config configs/invalid_fov.json
# stderr: error: camera.fov_degrees must be in (5, 175)
# exit code: 2
```

输出路径相对于配置文件所在目录解析。`output.image` 必须是 `.bmp` 或 `.ppm`，`output.report` 必须是 `.json`。配置错误返回 2，网格校验失败返回 3，渲染失败返回 4，文件写入失败返回 5；所有错误均写入 stderr。

## 管线

1. **确定性高度场**：`64x64` 至 `512x512` 高度矩阵，由以 `seed` 驱动的多倍频正弦/余弦叠加生成。地形顶点高度使用双线性采样，法线使用中心差分，纹理坐标和材质 ID 由程序生成。
2. **地形网格与 LOD**：每个完整块为 `32x32` 顶点，LOD 间距为 1、2、4 个高度场单位。块级选择同时考虑屏幕投影面积和相机距离；相邻不同 LOD 块之间生成过渡三角带闭合高度裂缝。索引按顶点属性去重，并校验重复三角形、退化三角形和非流形边。
3. **裁剪与剔除**：在齐次裁剪空间依次对 `x±w`、`y±w`、`z±w` 六个视锥平面执行 Sutherland-Hodgman 裁剪。部分裁剪时线性插值世界坐标、法线、UV 和材质 ID，随后与光栅化阶段共用透视正确插值规则。背面剔除发生在视锥裁剪之后，过渡三角带按双面处理。
4. **G-Buffer 与光栅化**：以重心坐标覆盖像素，写入位置、法线、UV、UV 屏幕导数、线性 `1/z` 深度和材质 ID。深度比较使用 `1/z`，属性插值使用 `(attribute/w)` 的透视正确权重。支持 `1x`、`2x`、`4x` 超采样及盒式降采样。
5. **各向异性采样**：由 UV 屏幕导数得到长轴/短轴方向和尺度，并根据法线与视线夹角拉伸长轴。在不超过 `4x4=16` 个加权 tap 的纹理足迹内采样程序纹理，统计实际平均采样次数。
6. **延迟光照**：G-Buffer 完成后统一计算环境光、至少 4 个方向光和至少 8 个带距离衰减的点光，包含 Lambert 漫反射与 Blinn-Phong 高光。
7. **输出与统计**：生成 BMP/PPM 和确定性 JSON 报告。报告不包含时间戳或运行耗时。
8. **消息循环**：Windows 构建创建一个隐藏 Win32 窗口并处理 `PeekMessage` / `TranslateMessage` / `DispatchMessage`，在渲染前后泵送消息；POSIX 构建保留相同的生命周期钩子，但离线模式不创建原生窗口。

## 配置

完整模板见 `configs/default.json`。主要字段：

| 字段 | 说明 |
| --- | --- |
| `heightfield.size` | 高度场边长，`[64, 512]` |
| `heightfield.frequency` | 基础空间频率，`(0, 0.25]` |
| `heightfield.amplitude` | 高度振幅，`(0, 200]` |
| `heightfield.seed` | 32-bit 确定性种子 |
| `heightfield.octaves` | 倍频数，`[1, 8]` |
| `camera.position/target/up` | 相机参数 |
| `camera.fov_degrees` | 垂直视场角，`(5, 175)` |
| `camera.near/far` | `0 < near < far` |
| `render.width/height` | 输出分辨率 |
| `render.supersampling` | 仅允许 `1`、`2`、`4` |
| `lod.block_size` | 每块顶点数，默认 `32`，允许 4 的倍数 |
| `lod.*_screen_area` | 投影面积占比阈值 |
| `lod.*_distance` | 相机距离阈值 |
| `texture.size/scale` | 程序纹理尺寸与世界空间缩放 |
| `lighting.directional` | 方向光数组，至少 4 项 |
| `lighting.point` | 点光数组，至少 8 项 |
| `output.image/report` | 输出文件路径 |

方向光的 `direction` 是“指向光源”的单位化前向量。点光使用 `range` 截断，并在范围内按平方反比加线性项衰减。

## 验证

项目包含 `tests/verify.py` 端到端测试，检查：

- 同一配置连续执行两次，BMP 与 JSON 的 SHA-256 分别一致。
- BMP 头、文件长度、尺寸、24-bit 像素载荷合法。
- PPM 头部、尺寸和像素载荷合法。
- 三角形计数关系、裁剪统计、三种 LOD 分布、G-Buffer 内存估算、各向异性采样预算合法。
- 非法 FOV 返回非零退出码，stderr 含具体字段名，且不生成输出文件。
- 缺失点光列表返回非零退出码和明确错误。
- 64x64/1x、512x512/1x 和 128x128/4x 配置矩阵可完成渲染，且拓扑审计无悬挂边/非流形边。

运行：

```bash
python3 tests/verify.py ./build/terrain_renderer configs/default.json build/test-output
```

如果环境提供 `ctest`，CMake 也会注册同名的 `end_to_end_verification` 测试。

### 本机实测记录

环境：Ubuntu 26.04、GCC 15.2.0、CMake 3.30.5，Release 构建。

```text
rendered 800x450 at 2x SSAA | triangles=52218 | lod blocks=31/7/11 | avg aniso taps=10.231
image: renders/terrain.bmp (1,080,054 bytes)
report: renders/report.json (7,676 bytes)
```

关键报告值：

```text
LOD blocks:                 31 / 7 / 11
terrain / transition tris:  51996 / 222
frustum culled:             7369
partially clipped:          255
after frustum clip:         45105
backface culled:            11373
after clipping:             33732
covered G-buffer samples:   689226
G-Buffer:                   56 bytes/sample, 80,640,000 bytes (76.90 MiB)
anisotropic taps:           average 10.231322, maximum 16
topology edges:             78,560 total, 78,094 manifold, 466 outer boundary,
                            0 non-manifold, 0 hanging
```

确定性测试中，同一配置两次运行得到：

```text
BMP SHA-256:    d9042e035913e02d4a7185b20a06cfab77948da46c0db4ecd040fc4cb3ae6de8
Report SHA-256: 3aedc1c20fb6eecd1365fef70de26e76c79b7ea3301a1f57fa2f6152af3ad2a7
```

生成的 `renders/report.json` 还包含每个 LOD 块的距离、投影面积占比、三角形数和实际索引拓扑校验结果。

详细实现说明见 `TECHNICAL.md`。
