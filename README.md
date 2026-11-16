# 离线地形预览渲染器

这是一个纯本地、确定性的 Windows 可运行地形预览器，面向关卡美术在提交引擎前检查高度场、分层材质、方向光软阴影和轨道相机取景。运行时仅依赖 `numpy` 与 `pillow`，不读取外部数据集，也不访问网络。

## 快速开始

Windows 10/11，Python 3.11 或 3.12：

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python render_terrain.py --config sample_config.json
```

也可以直接运行：

```bat
run_sample.bat
```

强制忽略缓存并全量重绘：

```bat
python render_terrain.py --config sample_config.json --force-full
```

运行自动化测试：

```bat
python -m unittest discover -s tests -v
```

## 生成物

默认样本会生成：

| 路径 | 内容 |
| --- | --- |
| `data/sample_terrain.bin` | 自校验、确定性的高度场二进制文件 |
| `outputs/sample/preview.png` | 材质、直接光、半球环境光和 PCF 阴影合成图 |
| `outputs/sample/normal.png` | 法线可视化调试图 |
| `outputs/sample/shadow.png` | PCF 阴影可见度掩码调试图 |
| `outputs/sample/tiles.json` | 可见世界瓦片、剔除统计、屏幕瓦片统计和版本信息 |
| `outputs/sample/cache/` | 上次渲染的内容寻址瓦片缓存、阴影图和清单 |

所有路径都相对于 JSON 配置文件所在目录解析。

## 实现结构

| 模块 | 职责 |
| --- | --- |
| `render_terrain.py` / `terrain_preview/cli.py` | 入口、参数解析、稳定错误码 |
| `terrain_preview/config.py` | JSON 读取与严格参数校验 |
| `terrain_preview/terrain.py` | SplitMix64 + 分形值噪声、法线、确定性二进制格式 |
| `terrain_preview/scene.py` | 透视轨道相机、世界瓦片、AABB 视锥与背面剔除 |
| `terrain_preview/raster.py` | CPU 三角形光栅化、近裁剪、透视校正、Z-buffer、阴影深度图 |
| `terrain_preview/material.py` | 海拔分层权重、坡度岩石混合、sRGB/线性色转换 |
| `terrain_preview/light.py` | 方向光正交拟合、深度图、固定样本数 PCF |
| `terrain_preview/shading.py` | 漫反射、半球环境光、阴影和色调映射 |
| `terrain_preview/cache.py` | 版本签名、瓦片缓存、清单原子提交、阴影缓存 |
| `terrain_preview/png.py` | 逐扫描线 PNG 写入，避免整图暂存并保证文件原子替换 |
| `terrain_preview/renderer.py` | 增量判定、分块渲染、统计和输出编排 |

## 关键行为

### 确定性地形

- 每个倍频使用整数格点哈希和五次平滑插值，不依赖随机数生成器状态。
- 高度值最终量化为 little-endian `float32`，二进制文件头包含维度、种子、载荷长度和 SHA-256。
- 同一配置重复生成会得到逐字节相同的 `.bin` 文件；读取时会再次校验哈希。

### 相机与瓦片剔除

- 相机参数为方位角、俯仰角、距离、垂直视场角、近远裁剪面和归一化目标点。
- 地形按 `terrain.tile_cells` 切分为固定世界瓦片。
- 每个世界瓦片先做六平面 AABB 视锥测试，再做瓦片角点背面测试。
- 可见瓦片 ID 列表同时打印到控制台并写入 `tiles.json`。

### 材质

- 四层顺序固定为 `sand`、`grass`、`rock`、`snow`。
- 三个海拔阈值与 `elevation_transition` 控制相邻层软过渡。
- `slope_start`、`slope_end` 和 `slope_rock_strength` 会把陡坡逐步混合到岩石层。
- 因此“同海拔的平坦雪坡”和“陡峭岩壁”会得到不同权重和颜色。
- 每层颜色与全局权重均来自配置文件。

### 阴影与光照

- 方向光使用拟合地形包围盒的正交光源相机。
- 阴影深度图在光源空间对全部地形三角形光栅化。
- 可见片元使用固定 `pcf_samples` 数量的确定性向日葵圆盘采样。
- `depth_bias`、`normal_offset` 和 `filter_radius` 均由配置提供。
- 采样位置超出阴影图边界时按“完全受光”计入，因此地形边界不会出现错误黑边。
- 最终颜色 = 分层反照率 ×（方向光可见漫反射 + 半球环境光），再做曝光和 gamma 编码。

### 增量重绘

- 每个屏幕瓦片缓存最终 RGB、法线调试图、阴影掩码和覆盖率。
- 缓存同时保存透视校正后的世界坐标与法线，用于判断新阴影图是否会改变该瓦片的 PCF 结果。
- 清单为每个屏幕瓦片保存贡献世界瓦片的 `height_versions`、`material_versions`、`lighting_version` 和 `shadow_version`。
- 瓦片文件按内容哈希命名；缓存清单最后原子替换，因此失败时不会提交半份清单。
- 相机、观察窗口、光照或阴影参数变化时全量重绘。
- 材质变化通过逐世界瓦片的材质签名比较；只有实际材质输出变化且与脏世界瓦片相交的屏幕瓦片会重绘。
- 局部高度变化会重绘直接受影响瓦片，并对其余复用候选瓦片重新计算 PCF 掩码；只有阴影掩码实际变化的瓦片才追加为脏瓦片。
- 统计固定打印为 `redrawn_tiles=... reused_tiles=...`，并写入 `tiles.json`。

### 资源限制

- `limits.max_total_samples` 是本次需要重绘瓦片的主采样与 PCF 采样上限，超限返回稳定错误码 `7`。
- `limits.max_memory_mb` 计入高度场、法线、阴影深度图和当前 G-buffer 批次。
- 瓦片颜色缓存保存在磁盘，不整批驻留内存；PNG 使用逐行流式写入。
- 全帧 G-buffer 超限时自动切换到按屏幕瓦片列分块，并打印 `memory_notice`。

## 配置摘要

完整字段见 `sample_config.json`。主要字段：

| 字段 | 含义 |
| --- | --- |
| `terrain.rows/cols` | 高度场顶点行列数 |
| `terrain.seed` | 确定性噪声种子 |
| `terrain.noise.*` | 基频、倍频数、持续度、频率倍率 |
| `terrain.tile_cells` | 世界瓦片边长（地形单元数） |
| `terrain.height_patches` | 可选局部高度补丁：归一化中心、半径和高度增量 |
| `camera.target` | 世界坐标归一化目标 `[x, y, z]` |
| `materials.elevation_thresholds` | 沙滩/草地、草地/岩石、岩石/雪线三个阈值 |
| `materials.elevation_transition` | 海拔层间过渡宽度 |
| `light.direction.*` | 指向光源的方位角与仰角 |
| `light.shadow.*` | 开关、2 的幂分辨率、偏移、PCF 样本数和过滤半径 |
| `render.tile_size` | 屏幕增量缓存瓦片尺寸 |
| `limits.*` | 总采样、内存和最大地形单元数 |

## 退出码

| 退出码 | 错误前缀 | 含义 |
| --- | --- | --- |
| `0` | - | 成功 |
| `1` | `E_UNEXPECTED` | 未预期异常，不输出 Python 堆栈 |
| `2` | `E_CONFIG_PARSE` | 配置文件缺失、无法读取或 JSON 非法 |
| `3` | `E_CONFIG` | 配置字段非法 |
| `4` | `E_TERRAIN` | 地形生成、读写或校验失败 |
| `5` | `E_RENDER` | 渲染流程失败 |
| `6` | `E_CACHE` | 缓存写入或路径校验失败 |
| `7` | `E_RESOURCE` | 采样或内存预算无法满足 |
| `8` | `E_OUTPUT` | PNG 或输出文件写入失败 |

已验证的非法配置包括尺寸越界、材质阈值倒置、阴影分辨率非 2 的幂和相机距离为零；均在创建输出前返回错误码 `3`。

## 实际验证记录

本次实现实际运行环境为 Python 3.14.4、NumPy 2.5.4、Pillow 13.0.0；源码只使用 Python 3.11 起可用的语法，不依赖 3.14 特性。

样本首次全量渲染：

```text
terrain_sha256=c7ab05c09897b14c5e9218b766ad63fbf660aef7174c619a0ebbf2d67c44328e
tile_culling: total=64 visible=64 culled_frustum=0 culled_backface=0
screen_tiles: total=80 redrawn_tiles=80 reused_tiles=0
shadow_map: enabled=True resolution=512 generated=True reused=False
samples: estimated=5222400 actual=1463700
memory: limit_mb=192.0 estimated_full_gbuffer_mb=13.26 chunked=False batch_columns=10
```

同一配置再次渲染：

```text
cache_decision=incremental signatures compared
screen_tiles: total=80 redrawn_tiles=0 reused_tiles=80
shadow_map: enabled=True resolution=512 generated=False reused=True
```

另外将雪层颜色单独修改后，样本得到 `redrawn_tiles=31 reused_tiles=49`；未变化的阴影图被复用，说明材质签名不是简单的全局失效。

在开启阴影的自动化小样本中加入一个局部高度补丁后，得到 `redrawn_tiles=4 reused_tiles=26`；阴影检查覆盖 7 个复用候选瓦片，未发生阴影误变，因此没有额外脏瓦片。

自动化测试共 17 项，实际结果：

```text
Ran 17 tests in 3.290s
OK
```

覆盖内容包括：

- 同配置高度场载荷和二进制文件逐字节一致，种子变化会改变结果，篡改载荷可被校验发现。
- 平坦高海拔与陡坡同海拔的材质权重和反照率不同，所有材质权重归一化。
- 窄视场相机能剔除部分世界瓦片。
- 阴影图外片元返回完全受光。
- 全量 → 零重绘复用 → 仅雪材质部分重绘。
- 开启阴影时的局部高度补丁只重绘直接受影响和阴影实际变化区域。
- `4 MB` 内存上限下 `512×512` 输出自动分列渲染。
- 缺文件、尺寸越界、阈值倒置、非 2 的幂阴影图、零相机距离、采样超限和输出写入失败均返回稳定错误码且不泄漏堆栈。
