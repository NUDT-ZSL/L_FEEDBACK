# 离线风蚀峡谷软件光栅化渲染器

这是一个 C++17、零图形库、零 GPU、零网络依赖的软件渲染器。它在空白工作区中完成
JSON 场景读取、确定性高度场生成、分层材质、层级浮石变换、三角形光栅化、24 位定点
深度、三光源模板阴影体，以及基于瓦片依赖的增量重绘。

## 构建（Windows 10/11）

项目只依赖 CMake、C++17 编译器和操作系统库。Visual Studio 2022 示例：

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
.\build\Release\canyon_renderer.exe --help
```

MinGW-w64 示例：

```powershell
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
.\build\canyon_renderer.exe --help
```

也可直接复用 `scripts/build_windows.ps1`。渲染器不访问网络，所有输出均写入
`--out` 指定目录（本仓库示例使用 `build` 下的目录）。

## 运行

```powershell
# 对“应用全部变更后的最终场景”做全量渲染
.\build\Release\canyon_renderer.exe `
  --scene scenes\canyon_demo.json --out build\out_full --mode full

# 先渲染初始场景，再按 JSON changes 顺序做瓦片增量重绘，最后与全量结果比较
.\build\Release\canyon_renderer.exe `
  --scene scenes\canyon_demo.json --out build\out_inc --mode incremental
```

输出：

- `color.ppm`：640×360 P6 RGB 颜色图。
- `depth.ppm`：按非背景深度范围拉伸显示的 8 位深度可视化图。
- `depth_raw.bin`：逐像素 32 位小端原始定点深度，未做可视化拉伸。
- `stats.json`：整帧/深度/组合哈希、瓦片哈希、受影响瓦片、依赖表、阴影体积三角形数、
  模板计数、溢出次数、耗时和错误信息。
- `render.log`：只包含确定性内容的文本日志，逐变更列出受影响瓦片序列和逐步帧哈希。

哈希算法为 FNV-1a 64。`frameHash` 只覆盖颜色字节；`depthHash` 覆盖原始深度；
`combinedHash` 依次覆盖颜色和原始深度。耗时只写统计，不参与任何哈希。

## 场景格式

`scenes/canyon_demo.json` 是完整样例，主要字段：

- `seed`：确定性高度场/岩石扰动种子。
- `render`：分辨率、`tileSize`、`depthBits`、`stencilBits`、近远裁剪面、阴影体拉伸距离
  和阴影深度偏置。
- `camera`：位置、目标、上方向和垂直视场角。
- `heightfield`：分辨率、X/Z 范围、非负高度范围、峡谷壁起始/结束距离、噪声八度和
  侵蚀强度。
- `materials.layers`：至少三层高度区间材质；每层有颜色和坡向着色。采样时同时考虑
  世界高度、坡向和确定性岩层条带。
- `lights`：强制恰好两盏方向光和一个点光；强度必须有限且为正。
- `rocks`：扁平节点数组，`id` 唯一，`parent` 可为 `null` 或父节点 ID。局部变换按
  `T * Rz * Ry * Rx * S` 组合，父节点世界矩阵乘以子节点局部矩阵。父子环按
  `E_SCENE_INVALID` 拒绝。
- `changes`：按顺序应用。`kind=transform` 支持 `translation`、`rotationDeg`、`scale`
  的部分更新；`kind=material` 支持 `baseColor`、`roughness` 的部分更新。

数值 `1e309` 会解析为无穷大，并在变换校验阶段返回 `E_TRANSFORM_INVALID`，不会被
误报成普通 JSON 语法错误。

## 错误码与退出码

| 业务码 | 退出码 | 含义 |
|---|---:|---|
| `E_SCENE_INVALID` | 2 | JSON/场景字段非法、缺字段、高度范围非法、父子环等 |
| `E_LIGHT_INVALID` | 2 | 光源类型/字段/光强/颜色非法，或方向光/点光数量不对 |
| `E_TRANSFORM_INVALID` | 2 | 变换缺失、尺度非正、矩阵出现非有限值 |
| `E_STENCIL_OVERFLOW` | 2 | 模板通过/失败计数超出配置的 `stencilBits` 通道容量 |
| `E_INCREMENTAL_DIVERGED` | 3 | 增量最终帧与同场景全量渲染逐字节不一致 |

出错时仍会写 `stats.json` 和 `render.log`，并返回非零退出码。

## 实现要点

- **确定性峡谷**：整数哈希种子 + 平滑值噪声 + 多八度 FBM + 峡谷剖面和侵蚀沟槽。
  顶点法线由相邻高度差分计算；给定种子重复运行结果一致。
- **完整光栅管线**：齐次坐标六平面裁剪（近、远、左右、上下）、屏幕空间重心坐标、
  透视校正的世界位置/法线插值、背面剔除、定点深度测试和原始 primitive 编号平局裁决。
- **深度**：默认 24 位整数深度（`0..16777215`），支持 16–30 位。样例中约 13.4 万
  个非背景像素、约 7 万个不同深度值；近远物体不会仅因量化而无序覆盖。
- **阴影体**：每个浮石三角形生成封闭三棱柱阴影体；方向光沿光线方向拉伸，点光按
  顶点到光源方向拉伸。每盏光独立清零模板，分别累计深度测试通过/失败两个逻辑通道。
  几何像素上 `passCount != 0 || failCount != 0` 判为阴影；背景像素不写入模板，因此
  阴影体积边界外保持天空/受光状态。每次计数越界即停止并上报 `E_STENCIL_OVERFLOW`。
- **层级浮石**：世界矩阵递归求解，祖先旋转会改变所有后代的位置与朝向；增量变更
  会递归重算该子树几何和阴影体，而非只重算目标节点。
- **瓦片增量**：每块瓦片记录影响它的浮石集合，足迹覆盖可见岩石与三个光源阴影体。
  变换变更用旧依赖表加新旧足迹求脏瓦片；纯材质变更只重绘岩石可见足迹，不重建阴影体。
  所有脏瓦片按行优先排序，逐瓦片重绘并记录颜色哈希。
- **强制自校验**：增量完成后，从初始场景独立应用全部变更并做全量渲染，比较颜色、
  原始深度和全部瓦片哈希；任何不一致返回 `E_INCREMENTAL_DIVERGED`。

## 本地实测

本仓库在 Linux 使用 GCC 15.2 和 CMake 3.30 做了实际编译/运行；代码未使用平台专属
API，Windows 使用相同 CMake 目标。验收脚本只用 Python 标准库，渲染器本身不依赖
Python。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
python3 tests/acceptance.py build/canyon_renderer scenes build/acceptance_out
# 或：ctest --test-dir build --output-on-failure
```

实测通过项：

- 全量与增量 `color.ppm`、`depth.ppm`、`depth_raw.bin` 逐字节一致。
- 样例最终 `frameHash=fa1326d8f7556db4`，`depthHash=005f514e2edbd2e4`。
- 4 个变更累计影响 680/920 个瓦片；两次独立增量运行的帧哈希、瓦片哈希和
  `affectedTiles` 列表完全一致。4 个变更分别影响 640、4、680、4 块瓦片；纯材质变更
  不重建阴影体。
- 最终帧共提交 5,350,400 个阴影体三角形栅格化请求；模板通过/失败测试计数约 139 万/
  341 万次，最大绝对计数为 7（配置容量 255）。
- 父节点 `canyon_spire` 旋转后，`spire_shard` 和 `spire_chip` 的世界中心都移动
  超过验收阈值 0.2，验证了层级链路。
- 覆盖 `E_SCENE_INVALID`（父子环、负高度、缺字段）、`E_LIGHT_INVALID`（光强为 0）、
  `E_TRANSFORM_INVALID`（`1e309` 变换）、`E_STENCIL_OVERFLOW`（1 位模板 + 重叠浮石）。
- 模板溢出计数大于 0，合法场景溢出计数为 0，且 `shadowMaskedBackgroundPixels=0`。
