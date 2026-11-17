# 离线分层材质 CPU 光线追踪器

这是一个从零实现的、仅使用 CPU 的确定性路径追踪器。它面向图形渲染教学和素材预生成，重点演示：

- 基底（漫反射/金属）+ 透明介质涂层的分层材质；
- 矩形与球形区域光源的面积采样软阴影；
- 球体、平面、轴对齐盒（AABB）和三角形的解析求交；
- 固定像素抖动抗锯齿、可复现的多次弹射、PNG 与 JSON 统计输出。

实现不调用图形 API、不依赖 GPU、不启动并行线程。渲染主循环是纯 Python 标量运算，NumPy 仅用于图像累积和后处理，Pillow 仅用于 PNG 编码。

## 环境与运行

Windows 10/11 64 位、Python 3.9+：

```powershell
pip install numpy pillow
python -m lrt --output render.png --stats render.json
```

默认场景为 256×256，使用种子 `158` 生成，包含球体、地面平面、AABB 基座、三角背景墙，以及矩形光和球形光。指定 JSON 场景：

```powershell
python -m lrt --scene scenes/minimal_layered.json --output render.png --stats stats.json
```

常用覆盖参数：

```powershell
python -m lrt --width 256 --height 256 --seed 158 --spp 8 --aa 4 --shadow-samples 32
python -m lrt --scene scene.json --calibrate --output render.png --stats stats.json
```

`--aa` 只接受 `1`、`2`、`4`，表示每轴子像素数，即每像素 `1×1`、`2×2`、`4×4` 个子像素样本。

## 场景格式

完整可运行示例见 `scenes/minimal_layered.json`。顶层字段：

| 字段 | 说明 |
| --- | --- |
| `width`, `height` | 输出分辨率，默认 256×256 |
| `seed` | 确定性采样种子 |
| `camera` | `position`、`look_at`、可选 `up` 与 `fov` |
| `materials` | 材质名到材质定义的对象 |
| `geometry` | 几何体数组，`material` 引用材质名 |
| `lights` | 非空光源数组 |
| `render` | `samples_per_pixel`、`aa_samples`、`shadow_samples`、`max_depth`、`exposure`、`self_intersection_epsilon` |
| `background` | 主光线或未命中路径看到的 RGB 环境值 |

几何类型：

```json
{"type": "sphere", "center": [0, 1, 0], "radius": 1, "material": "mat"}
{"type": "plane", "point": [0, 0, 0], "normal": [0, 1, 0], "material": "mat"}
{"type": "aabb", "min": [-1, 0, -1], "max": [1, 1, 1], "material": "mat"}
{"type": "triangle", "v0": [-1, 0, 0], "v1": [1, 0, 0], "v2": [0, 2, 0], "material": "mat"}
```

分层材质：

```json
{
  "type": "layered",
  "base": {"type": "metal", "color": [0.9, 0.62, 0.2], "roughness": 0.05},
  "coating": {
    "ior": 1.52,
    "thickness": 0.001,
    "tint": [1.0, 1.0, 1.0],
    "absorption": [0.0, 0.0, 0.0]
  }
}
```

`base.type` 为 `diffuse` 或 `metal`。也支持不带涂层的顶层 `diffuse`/`metal` 调试材质。涂层折射率必须在 `[1.0001, 2.5]`，厚度必须在 `[1e-6, 0.05]`；`thickness` 通过 Beer-Lambert 衰减参与能量计算，而不是仅作元数据。

光源：

```json
{"type": "rect", "position": [0, 3, 0], "right": [1, 0, 0], "up": [0, 0, 1],
 "width": 2, "height": 1.5, "intensity": [3, 3, 3]}
{"type": "sphere", "center": [2, 2, 1], "radius": 0.3, "intensity": [2, 2.5, 3]}
```

矩形光由中心、两个正交单位方向、宽高和辐射亮度定义，双面发光。球形光按接收点可见球冠面积均匀采样。

## 核心算法

### 分层能量合成

对入射角余弦 `cos_i`，先计算介质界面的非偏振菲涅尔反射率 `F`，并令涂层内路径长度为 `thickness / cos_i`。基底分支的反射率包含出射界面菲涅尔项、涂层吸收和着色：

```text
T_eff = (1 - F)^2 * exp(-absorption * thickness / cos_i) * tint
R_eff = F + T_eff * Rb
```

当 `Rb`（漫反射反照率或金属颜色）位于 `[0,1]` 时，`R_eff` 必然不超过 1。路径追踪中的反射/基底事件使用 `p = max(F, 0.12)` 做事件选择，反射权重 `F/p`、基底条件权重 `T_eff*Rb/(1-p)` 均经过补偿，因此期望值仍是上式；提高 `p` 只是降低低样本数时的镜面火点方差。

`--calibrate` 会在固定角度 `0..85°` 下对每个材质检查 `R_eff` 的每个 RGB 通道。越界值会写入警告和统计，不会被静默截断；最终 PNG 的色调映射只发生在显示阶段。

### 区域光、软阴影与反射/折射

- 矩形光：在矩形面积上均匀采样，按面积 pdf 计算几何项。
- 球形光：在接收点可见球冠上均匀面积采样，采样面积只包含可见部分。
- 阴影射线：从基底着色点发出，使用随命中距离缩放的偏移，避免自相交。
- 涂层折射：在基底分支中依次对入射和出射界面执行 Snell 折射，并使用层内路径长度计算吸收。两条平行界面会使最终传播方向恢复，符合薄层解析模型。
- 路径：主光线、涂层/金属反射、漫反射间接弹射和折射段均有独立计数；达到 `max_depth` 时记录截断。

### 确定性与采样序列隔离

所有随机选择都由整数计数器哈希得到，不使用全局 RNG：

```text
(seed, pixel_id, subpixel_id, path_id, bounce, stream_id) -> [0,1)
```

各采样职责使用不同 stream：

- 子像素固定抖动：`2×2`/`4×4` 分层网格，每层内确定性抖动；
- 区域光：像素局部旋转的 2D Halton 低差异序列；
- 间接散射：按 `path_id` 索引的低差异序列；
- 反射/基底事件：独立基数的 radical inverse 加像素局部旋转。

因此 AA、区域光和路径采样不会共享同一序列。重复运行同一输入时，图像像素和 JSON 字段逐项一致。

## 错误码

CLI 将用户输入错误输出到 stderr，并以整数状态码退出：

| 代码 | 含义 |
| ---: | --- |
| 2 | 场景文件不可读或 JSON 语法非法 |
| 3 | 缺字段、字段类型错误、材质引用无效 |
| 4 | 几何参数非法（如负半径、退化三角形、空几何） |
| 5 | 未知材质类型 |
| 6 | 光源数组为空或光源类型非法 |
| 7 | 涂层折射率或厚度越界 |
| 8 | 渲染参数越界（如 aa 不是 1/2/4） |
| 9 | 输出文件写入失败 |

渲染过程中若采样颜色出现 NaN，只把该像素写成洋红色 `(255,0,255)`，增加 `nan_pixels`/`nan_samples` 后继续渲染其他像素。

## JSON 统计

主要字段：

- `total_primary_rays`、`total_rays`、`average_rays_per_pixel`；
- `shadow_samples_per_light`、`shadow_rays`、`average_shadow_rays_per_pixel`；
- `reflection_rays`、`refraction_rays`、`indirect_rays`；
- `depth_clip_count`、`nan_pixels`、`nan_samples`；
- `max_layer_reflectance`、`min_layer_reflectance`、`energy_warnings`；
- `shadow_noise`（主命中直接光亮度图的 3×3 邻域方差均值）；
- `edge_gradient`（相邻像素色调映射亮度绝对差的均值）；
- `scene_sha256`、采样参数和采样序列说明。

统计中不含时间戳、随机调度信息或输出路径，因此同一场景与参数可逐字段比较。

## 测试与验收

```powershell
python -m unittest discover -s tests -v
python tools/verify_acceptance.py
```

完整验收脚本会生成：

- `artifacts/default_256.png`：默认场景 256×256 PNG；
- `artifacts/default_256.json`：对应统计；
- `artifacts/acceptance_report.json`：能量、错误码、阴影收敛、AA 与确定性结果。

测试覆盖四种求交、分层能量上界、校准越界警告、非法输入错误码、NaN 像素隔离、阴影噪声下降、AA 梯度下降、PNG 输出及同输入重复运行一致性。

## 实际验证记录

当前执行环境为 Python 3.14.4、NumPy 2.5.4、Pillow 13.0.0；实现未使用平台相关路径或外部命令。以下结果由 `python tools/verify_acceptance.py` 实际生成，并保存在 `artifacts/acceptance_report.json`：

- 单元/集成测试：`21/21` 通过。
- 默认 256×256：8 paths/subpixel、1x AA、每光 8 个阴影样本、最大深度 5、种子 158。
- 默认渲染统计：`7,980,914` 条光线，`121.7791` 条/像素；阴影射线 `97.8949` 条/像素；深度截断 `10,984`；NaN 像素 `0`。
- 能量：所有材质、所有校准角度的反射率范围 `[0.131778, 0.961492]`，全部位于 `[0,1]`。
- 默认图像两次渲染逐像素一致，统计逐字段一致。RGB 像素缓冲区 SHA-256：`4ec8037d1cc1566aef07061e255f3b90d7e01adc880956058aa5eecae8b972e0`；规范化统计对象 SHA-256：`c58b292ccb344a99528c81f192ae8c0aa3073716fdb0ccd6230a06e74cd997f3`。
- 阴影收敛（96×96，1→16 个区域光样本）：邻域方差 `0.000947506 → 0.000220547`，下降约 `76.7%`；平均阴影射线 `3.0698 → 49.1391` 条/像素。
- 抗锯齿（64×64，1x→4x）：边缘梯度 `0.0257236 → 0.00900325`，降至 `0.3500` 倍；4x 两次串行运行的图像与统计一致。
- 非法输入错误码实测：缺字段 `3`、负半径 `4`、未知材质 `5`、空光源 `6`、IOR 越界 `7`、厚度越界 `7`、非法 AA `8`。

完整验收脚本串行执行两次 256×256 默认渲染及全部分项检查，本次耗时约 138 秒，峰值常驻内存约 58 MB。生成文件：

- `artifacts/default_256.png`
- `artifacts/default_256.json`
- `artifacts/acceptance_report.json`

