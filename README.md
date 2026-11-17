# 地形离屏软光栅渲染器

这是一个面向关卡美术离线预览的 CPU 地形渲染工具。渲染管线完全由 Python 与 NumPy 实现，不依赖 GPU、驱动或容器，可在 Windows 命令行重复执行并生成字节一致的 PNG。

## 功能

- 由固定种子生成二维高度场，按单位间距和高度放大系数重建顶点与逐顶点法线。
- 完整齐次裁剪：近/远裁剪面、左右上下视锥面，以及完全位于近平面后方、跨近平面和屏幕外三角形的处理。
- 瓦片化软件光栅：顶点属性透视校正、背面剔除、顶左填充规则、深度缓冲和线性相机深度输出。
- 三参数级联划分、正交光方向视图、级联重叠区可见度混合、3x3 PCF、常数深度偏置、斜率偏置和法线偏移。
- 漫反射/法线扰动/粗糙度按图层计算，归一化权重线性合成；方向光、环境光和两个以上带距离衰减的点光源。
- 线性颜色空间的指数色调映射、gamma 编码和确定性 8 位量化。
- 高度场或权重局部变化时，只更新受影响的阴影纹素、主视图脏瓦片；相机、分辨率、光照、材质或尺度变化时整体重建。
- 内部 PNG 编码器固定 zlib 参数且不写时间戳；JSON 清单记录参数摘要、告警、级联尺寸、阶段耗时和增量统计。

## 环境

目标环境为 Windows 10/11、Python 3.11。运行依赖：

```bat
py -3.11 -m pip install -r requirements.txt
```

代码只要求 NumPy 运行时；Pillow 用于验收时读取和检查 PNG，不作为内部编码依赖。

## 快速开始

在仓库根目录执行：

```bat
py -3.11 render_terrain.py ^
  --height-size 65 ^
  --resolution 320 240 ^
  --cascades 3 --shadow-size 512 ^
  --unit-scale 0.25 --height-scale 2.0 ^
  --output-dir preview
```

也可以使用：

```bat
render_terrain.bat --height-size 65 --resolution 320 240 --output-dir preview
py -3.11 -m terrain_softrast --height-size 65 --resolution 320 240 --output-dir preview
```

输出目录包含：

- `color.png`：RGB8，线性光照结果经确定性指数色调映射和 gamma 编码。
- `depth.png`：Gray16，线性相机深度，`0=near`、`65535=far`，背景为 far。
- `shadow_visibility.png`：Gray8，`0=完全遮挡`、`255=完全受光`，背景为 0。
- `manifest.json`：完整渲染清单。

## 输入文件

高度场可通过 `--height-file` 提供 `.npy`、`.txt` 或 `.csv`；默认由 `--seed` 确定性生成。权重图通过 `--weights-file` 提供，形状必须严格为 `(图层数, 高度, 宽度)`：

```bat
py -3.11 render_terrain.py ^
  --height-file terrain.npy ^
  --weights-file weights.npy ^
  --resolution 640 360 --output-dir preview
```

默认材质为四层 `meadow/rock/sand/snow`。权重不必预先归一化：每个像素先按通道和归一化，再在三角形内透视校正插值。若输入未归一化或含负权重，渲染继续，但清单中会出现业务告警码 `MATERIAL_WEIGHTS_NOT_NORMALIZED` / `MATERIAL_WEIGHTS_NEGATIVE`。

## 增量重渲染

首次运行时指定缓存文件：

```bat
py -3.11 render_terrain.py ^
  --height-file height_a.npy ^
  --resolution 640 360 --cache-file preview.cache.npz ^
  --output-dir preview
```

修改局部高度后，用同一缓存和高度场网格坐标脏区 `x0 y0 x1 y1` 再次运行：

```bat
py -3.11 render_terrain.py ^
  --height-file height_b.npy ^
  --resolution 640 360 --cache-file preview.cache.npz ^
  --dirty-rect 20 18 32 30 ^
  --output-dir preview
```

脏区是半开区间，按高度场顶点索引解释。缓存只有在以下结构完全一致时才复用：分辨率、瓦片尺寸、相机、方向光、点光源、材质、色调映射、阴影参数、级联参数、单位间距和高度放大系数。相机/分辨率等变化会整体重建。若提供的脏区没有覆盖实际输入差异，工具安全回退到全量渲染，并写入 `DIRTY_RECT_INCOMPLETE` 告警。

增量路径会：

1. 重新构建顶点与法线，定位受脏区影响的一环三角形。
2. 仅清除并重画各相关级联的阴影纹素矩形，矩形内仍用新场景全部几何重建深度。
3. 将改动几何及其沿方向光传播的保守世界包围盒投影到主视图，选出脏瓦片。
4. 只重置并光栅化这些主视图瓦片，其余像素沿用缓存。

## 参数与退出码

常用参数：

- `--height-size N`：生成 `N x N` 高度场。
- `--resolution WIDTH HEIGHT`：输出分辨率，也可分别用 `--width/--height`。
- `--cascades N`、`--shadow-size S`：级联数及每级阴影贴图边长。
- `--tile-size S`：主视图瓦片尺寸。
- `--unit-scale`、`--height-scale`：世界间距与高度放大。
- `--camera-eye` 等：相机位置、目标、上方向、FOV、近远裁剪面。
- `--light-direction/color/intensity`：方向光。
- `--point-light x,y,z,range,intensity[,r,g,b]`：可重复，默认提供两个点光源。
- `--dirty-rect x0 y0 x1 y1`：增量脏区。
- `--cache-file path.npz`：增量缓存。

退出码：

- `0`：成功。
- `2`：参数/输入校验失败，原因写入 stderr；不会创建或写入输出文件。
- `1`：渲染或文件系统错误。

## 设计取舍

- 阴影分辨率与覆盖：每级使用围绕该视锥分片包围球的固定正交范围。数值稳定、纹理密度可预测，但远景级联的每纹素世界尺寸会随覆盖范围增大。可通过级联数、`split_lambda` 和阴影尺寸调节。
- 阴影偏置：常数偏置抵抗基础深度误差，斜率偏置补偿掠射角，法线偏移进一步减少痤疮；偏置过大可能产生 peter-panning，因此没有无上限放大。
- 增量正确性：阴影污染可能离开几何改动的屏幕包围盒，因此更新范围额外包含沿光方向到场景边界的保守投影。必要时宁可多更新瓦片或整体重建，也不复用可能失效的数据。
- 浮点行为：所有路径为单线程 NumPy/Python 确定性计算，不做随机采样、多线程归约或时间相关抖动。PNG 编码不写时间戳。

## 测试

```bat
py -3.11 -m pip install pytest
py -3.11 -m pytest -q
```

测试覆盖：法线/尺度、近远裁剪、共享边、重复字节一致性、级联重叠、投影阴影、材质加权等价、权重告警、增量与全量等价、脏区回退、相机/尺度缓存失效、CLI 重复输出和非法参数退出码。

实际执行记录和哈希见 `VALIDATION.md`。
