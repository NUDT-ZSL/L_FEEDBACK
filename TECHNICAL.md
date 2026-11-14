# 技术说明

## 1. 数据流

```text
场景 JSON
  -> 严格校验与规范化
  -> 几何哈希 / 光照哈希
  -> 三角形体素化 + 空间哈希
  -> 探针布局 + 布局哈希
  -> 单线程或多线程探针烘焙
  -> 能量上界钳制
  -> float32 记录 + 版本化二进制缓存
  -> JSON 烘焙/校验报告
```

场景 JSON 中的网格均为世界坐标三角形。样本场景包含地面、顶面、三面墙、盒状遮挡物、
两级细分二十面体球和两个圆盘面积光。所有数据由 `voxel_gi.sample_scene` 生成。

## 2. 体素化与空间哈希

设遮挡网格包围盒为 `Bmin/Bmax`，最长轴长度为 `E`，配置分辨率为 `R`，边界填充为
`P`，则：

```text
voxel_size = E / (R - 2P)
dims       = ceil((Bmax - Bmin) / voxel_size) + 2P
origin     = Bmin - P * voxel_size
```

每个三角形先按其 AABB 枚举候选体素，再执行 13 轴分离轴测试（3 个盒轴、9 个边叉积
轴、1 个三角形法线轴）。相交体素标记为占用，并将三角形 ID 插入
`SpatialHash[(ix, iy, iz)]`。

主射线使用 3D DDA 遍历体素。进入占用体素后，通过空间哈希取出候选三角形，使用
Möller–Trumbore 算法求精确首次命中。遮挡射线只需体素占用查询，因此不对每个阴影
样本执行三角形求交。

自遮挡处理：阴影射线起点沿命中面法线偏移，并跳过起点所在的表面体素。只有当射线
起点本来就在网格内部时才跳过；从网格外进入的射线不会跳过第一个占用体素。

## 3. 探针布局与方向采样

给定布局包围盒和间距 `s`，每个轴的数量与起点为：

```text
count = floor(extent / s) + 1
start = min + 0.5 * (extent - (count - 1) * s)
```

探针按 x、y、z 的笛卡尔顺序展开。位置以 float64 计算并纳入布局哈希。

半球方向使用 Fibonacci 低差异序列。局部 `z=(i+0.5)/N` 保证单位半球均匀立体角；
固定种子只为每个探针生成一个方位相位旋转。采样顺序固定为 `i=0..N-1`。

## 4. 直接光照与 SH 投影

每个探针向固定 `up` 的半球发射 `ray_count` 条射线：

- 未命中：入射辐射亮度为常量 `ambient_radiance`。
- 命中：使用命中三角形的双面定向法线，与材质反照率计算朗伯反射。

点光按各向同性功率计算：

```text
E = P / (4*pi*r^2) * max(0, n·l) * visibility
```

面积光使用固定数量、等面积的圆盘采样点。样本以解析面积权重累积：

```text
Le    = color * P / (pi * area)
dE    = Le * max(0,n·l) * max(0,n_light·-l) * (area/N) / r^2
L_out = albedo/pi * sum(dE)
```

所有阴影可见性均由体素 DDA 遮挡遍历计算。

每个方向样本构造方向辐照度标量场：

```text
f_i = L_i * max(0, up·d_i)
```

在均匀半球立体角下，SH 投影为：

```text
c_lm = (2*pi/N) * sum_i f_i * Y_lm(d_i)
```

系数形状为 `(9, 3)`。`Y00 = 0.28209479`，因此每通道积分辐照度为：

```text
E_channel = c_00 * sqrt(4*pi)
```

残差为重建方向辐照度与采样值的 RGB 联合相对 RMS：

```text
residual = rms(reconstruct(f) - f) / max(rms(f), 1e-15)
```

三阶 SH 对带半球截断的方向辐照度函数通常会有明显残差；该值被逐探针保存并在报告中
统计，而不是被隐藏。

## 5. 能量守恒

设光源总辐射功率为：

```text
P_total = sum(light.radiant_power)
```

场景反照率上界为所有材质 RGB 通道最大值 `A_max`。每个探针的“总辐照度”定义为
SH 的 `c00*sqrt(4*pi)` 在 RGB 三个通道上的求和。若它超过：

```text
E_bound = P_total * A_max
```

则对该探针的全部 9×3 个系数施加统一比例衰减，保留通道比例。实现额外使用
`1e-6` 的安全边界，避免 float32 序列化后的舍入让总能量略微超过声明上界。

因此每个探针满足：

```text
sum_rgb(E_channel) <= P_total * A_max
```

遮挡只会减少可见光贡献；漏射只使用环境常量；随后再执行上界钳制。

## 6. 并行确定性

探针任务彼此独立。每个任务只读取场景、体素网格和探针布局，写入自己的结果元组，
不共享累加器。线程池使用 `ThreadPoolExecutor.map(..., chunksize=1)` 动态领取任务，
结果按探针索引回收。

每个探针内部保持固定方向顺序、固定球谐基顺序和 float64 累加；最终才转换为
float32。因此单线程和多线程系数、残差和缓存载荷逐位一致。

## 7. 缓存格式

固定头为 little-endian：

```text
8s  magic = b"VXGI9SH1"
u32 format_version = 1
u32 metadata_length
u32 payload_length
```

其后是 canonical JSON 元数据和连续载荷。每条探针记录 112 字节：

```text
coefficients: <f4[9,3]  (108 bytes)
residual:     <f4       (4 bytes)
```

元数据包含：

- 格式版本、算法版本、场景 ID。
- 几何哈希、光照哈希、布局哈希、完整场景哈希。
- 探针包围盒、间距、up、三维数量、位置 SHA-256。
- 烘焙种子、射线数、体素分辨率、环境光。
- 能量上界、钳制探针数、最大逐探针辐照度。
- 平均/最大残差、体素统计、载荷长度与 SHA-256。

加载时依次检查：文件长度、magic、格式版本、元数据版本、载荷描述、probe 数量、
记录大小、载荷 SHA-256、系数与残差有限性。校验模式随后比较几何哈希、场景哈希和
布局哈希；几何不匹配返回 `CACHE_GEOMETRY_MISMATCH`。

缓存先完整生成于内存，再写入同目录临时文件，`fsync` 后通过 `os.replace` 原子替换。
因此任何输入错误或烘焙异常都不会截断已有缓存。

## 8. 错误码

主要业务码：

```text
INVALID_JSON
INVALID_SCHEMA
INVALID_MESH_NAN
INVALID_MESH_DEGENERATE
INVALID_MATERIAL
INVALID_LIGHT
INVALID_PROBE_SPACING
INVALID_PROBE_BOUNDS
INVALID_BAKE_CONFIG
PROBE_INSIDE_OCCLUDER
CACHE_NOT_FOUND
CACHE_BAD_MAGIC
CACHE_VERSION_UNSUPPORTED
CACHE_CORRUPT
CACHE_CHECKSUM_MISMATCH
CACHE_GEOMETRY_MISMATCH
CACHE_SCENE_MISMATCH
CACHE_LAYOUT_MISMATCH
IO_ERROR
INTERNAL_ERROR
```

## 9. 已知边界

- 当前交付是单次命中的直接光照加环境漏射，不递归计算多次漫反射弹射。
- 体素占用是保守表面壳，用于遮挡与加速；最终主射线命中仍使用精确三角形求交。
- 样本较小，因此线程池主要提供任务级动态均衡；更大的场景可直接沿用同一工作队列。
