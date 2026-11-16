# CPU Deferred Renderer

一个纯 NumPy、离屏运行的软件延迟着色渲染器。程序不使用 GPU、OpenGL 或图形窗口，完成顶点变换、三角形光栅化、GBuffer 写入、三类阴影贴图、延迟光照、各向异性高光与统计输出。

## 环境与安装

- Python 3.9+
- NumPy >= 1.21
- Pillow >= 9.0

Windows PowerShell：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux/macOS：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 运行确定性样本

仓库已包含确定性样本 `sample/sample_scene.json` 及其 NPZ 网格、JSON 材质。直接运行：

```bash
.venv/bin/python render.py --scene sample/sample_scene.json --output output
```

如需重新生成完全相同的样本资产：

```bash
.venv/bin/python tools/make_sample_scene.py --output sample
```

成功后退出码为 `0`。输出目录必须尚不存在；替换旧输出时使用 `--overwrite`。所有文件先写入隐藏暂存目录，全部成功后再原子替换目标目录，失败路径会清理暂存内容。

命令行覆盖项：

```text
--width N                 输出宽度
--height N                输出高度
--shadow-resolution N     所有阴影贴图分辨率
--depth-tolerance F       阴影深度比较容差
--depth-bias F            阴影渲染常数深度偏移
--slope-scale F           阴影渲染斜率缩放偏移
--cascade-count N         方向光级联数，至少为 2
--overwrite               原子替换已有输出目录
```

配置或运行错误返回非零退出码，原因写入 `stderr`。例如：

```bash
python render.py --scene broken.json --output out
# error: invalid JSON in scene file ...
```

## 输出

```text
output/
  hdr_color.npy                 float32 HxWx3，未做色调映射
  depth.png                     主相机深度可视化
  report.json                   统计与一致性元数据
  gbuffer/
    layout.json                 显式通道布局、dtype、语义和文件名
    position.npy                float32 HxWx3 世界坐标
    normal.npy                  float32 HxWx3 世界法线
    tangent.npy                 float32 HxWx3 世界切线
    bitangent.npy               float32 HxWx3 世界副切线
    uv.npy                      float32 HxWx2 UV
    material_index.npy          int32 HxW 材质索引
    depth.npy                   float32 HxW 窗口深度 [0, 1]
    coverage.npy                uint8 HxW 覆盖掩码
  shadow_maps/
    directional_sun_cascade_*.png
    point_<light>_posX/negX/posY/negY/posZ/negZ.png
    spot_<light>.png
```

`GBuffer.load(path)` 直接读取 `.npy` 和 `layout.json`，延迟阶段不需要再次执行顶点变换。

## 场景格式

场景 JSON 的核心字段：

```json
{
  "width": 480,
  "height": 320,
  "materials": "materials.json",
  "camera": {
    "position": [6.2, 4.4, 9.2],
    "target": [0.0, 1.15, -0.2],
    "up": [0.0, 1.0, 0.0],
    "fov_y_deg": 48.0,
    "near": 0.1,
    "far": 50.0
  },
  "shadows": {
    "resolution": 256,
    "cascade_count": 3,
    "split_lambda": 0.62,
    "shadow_distance": 35.0,
    "pcf_radius": 1
  },
  "depth": {
    "bias": 0.0015,
    "slope_scale": 3.0,
    "compare_tolerance": 0.006
  },
  "lights": [
    {
      "id": "sun",
      "type": "directional",
      "direction": [0.30, -0.85, -0.42],
      "color": [1.0, 0.93, 0.80],
      "intensity": 2.8
    }
  ],
  "meshes": [
    {
      "name": "ground",
      "path": "meshes/ground.npz",
      "transform": {"translate": [0.0, 0.0, 0.0]},
      "cast_shadow": true
    }
  ]
}
```

光源字段：

- `directional`：`direction` 是光线传播方向，颜色与强度独立。
- `point`：`position`、`range`、颜色、强度；衰减为 `intensity / distance^2`，`range` 外不参与着色。
- `spot`：在点光字段上增加 `direction`、`inner_angle_deg`、`outer_angle_deg`；锥角过渡使用余弦 `smoothstep`。

网格是 `.npz`，必需数组：

```text
positions          (N, 3) float
normals            (N, 3) float
tangents           (N, 3) float
bitangents         (N, 3) float，可省略并由 cross(normal, tangent) 推导
uvs                 (N, 2) float
indices             (T, 3) int
material_indices    (T,) int
```

`transform` 可以是 4x4 行主序矩阵，或 `translate`、`rotate_xyz_deg`、`scale` 对象。`material_index` 可选，用于覆盖整个节点的三角形材质索引。

材质 JSON 是 `{"materials": [...]}`。每个材质支持：

```text
id, base_color, specular_color, metallic, roughness
base_specular_exponent
tangent_specular_exponent, bitangent_specular_exponent
anisotropy, anisotropy_axis ("tangent" | "bitangent")
anisotropy_rotation_deg
```

若未显式给出切线/副切线指数，则依据 `base_specular_exponent`、`anisotropy` 和拉伸轴自动生成。GBuffer 的材质索引在延迟阶段查表。

## 管线实现

1. 几何准备：按网格计算世界包围球，与相机视锥六个平面测试；被剔除网格的所有三角形不进入 GBuffer 光栅化。保留三角形做世界空间背面剔除、近/远裁剪平面裁剪和屏幕包围盒光栅化。
2. GBuffer：使用透视正确插值写入世界位置、法线、切线、副切线与 UV；法线框架逐像素正交归一化；深度使用 NDC 深度映射到 `[0, 1]`。
3. 阴影：
   - 方向光：按指数/均匀混合分割生成至少两级级联，拟合相机子视锥的 Light-space AABB 并做 texel 对齐，使用正交投影。
   - 点光：六个 90° 透视面，着色时按从光源到片元的主轴选择面。
   - 聚光灯：单张透视阴影图，FOV 覆盖外锥角。
   - 阴影渲染使用 `bias = depth_bias + slope_scale * max(|dD/dx|, |dD/dy|)`；查询使用同一配置对象中的偏移、斜率缩放和 `compare_tolerance`，并对 3x3 PCF 的每个邻域样本做局部斜率校正。
4. 延迟光照：从 GBuffer 读取并归一化位置/法线/切线框架，对每个光源先做影响范围/锥体裁剪，只对候选像素计算阴影与着色，贡献累加到 float32 HDR 缓冲。
5. 各向异性高光：将半角向量变换到切线空间 `(T, B, N)`，使用 Ashikhmin-Shirley 型分布：

   ```text
   D = sqrt((nu + 2)(nv + 2)) / (8*pi) * NdotH^(nu*TdotH^2 + nv*BdotH^2)
   ```

   `nu`、`nv` 分别为切线/副切线指数，`anisotropy_rotation_deg` 在光照阶段旋转切线与副切线。

统计定义：

- `triangles_before_culling`：输入三角形总数。
- `mesh_frustum_culled_triangles`：包围球被相机视锥剔除的网格所含三角形数。
- `triangles_after_culling`：进入主相机光栅化的三角形数。
- `backface_culled_triangles` / `offscreen_culled_triangles`：主相机阶段后续剔除数。
- `light_pixel_interactions`：光源影响范围内（点光半径、聚光半径与外锥）的 GBuffer 像素数；方向光为全部覆盖像素。它不等于最终非阴影亮像素数。

`report.json` 还记录每个光源的 `candidate_pixels`、实际阴影查询数、完全阴影像素数、亮像素数、各阶段单次 `perf_counter_ns` 测量耗时，以及峰值内存估算。

## 验证

```bash
.venv/bin/python -m unittest -v tests.test_pipeline
.venv/bin/python tools/verify_outputs.py output
```

`tests/test_pipeline.py` 覆盖：

- GBuffer 通道 dtype/shape、法线/切线/副切线归一正交、材质索引与深度背景。
- HDR 形状/有限性/非负性、无色调映射、9 张以上阴影贴图与报告文件一致性。
- 方向光/点光/聚光灯交互计数、剔除计数和阶段耗时字段。
- 两次同参数渲染的 HDR、深度、GBuffer 和阴影贴图字节级确定性。
- 相机完全背向场景时的网格级视锥剔除和空画面统计。
- 非法 JSON、非法光源类型返回非零退出码，且不创建输出或残留暂存目录。

## 实测结果

环境：Python 3.11.18、NumPy 2.4.6、Pillow 13.0.0，Linux x86-64。

默认样本运行：

```text
.venv/bin/python render.py --scene sample/sample_scene.json --output output
```

结果：

```text
exit code                 0
分辨率                    480 x 320
覆盖像素                  115647
方向光级联                3（另有点光 6 面 + 聚光 1 面，共 10 张）
输入/主相机后三角形       2860 / 2860
背面剔除 / 屏幕外剔除     1340 / 280
实际光栅化三角形          1240
光源-像素交互             方向光 115647，点光 82635，聚光灯 13572
GBuffer 光栅化            6927.72 ms
阴影贴图生成              27935.28 ms
延迟着色                  735.30 ms
输出写入                  128.76 ms
整条管线                  35727.99 ms
峰值内存估算              90.71 MB（RSS 峰值 136.44 MB）
HDR                       有限，非负，最大值 9.022，无色调映射
```

`tools/verify_outputs.py output` 输出：

```text
verification OK: 480x320, 115647 covered pixels, 10 shadow maps, 211854 interactions
```

错误配置测试返回非零并保持输出目录不存在；正确样本返回 `0`。

## 范围说明

- 只实现不透明几何，不含纹理采样、透明混合、后期色调映射或抗锯齿。
- 网格与光栅化使用 NumPy 批量数组，但外层按三角形迭代，适合确定性验证而非交互帧率。
- 点光六面阴影与聚光阴影为单次深度采样加 PCF，不包含硬件立方体贴图过滤。
