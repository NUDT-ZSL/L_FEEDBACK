# Deterministic Skinned Mesh Renderer

这是一个纯 Python/NumPy 的离线 CPU 渲染器。它读取确定性 JSON 场景，逐帧执行线性混合蒙皮、透视投影、共享深度缓冲的多材质 Blinn-Phong 光照、三级级联方向光阴影，输出 PNG 序列、逐帧统计 JSON 和增量版本清单。

实现不依赖 Docker、GPU、WebGL 或外部渲染服务。核心依赖为：

- Python 3.11+
- NumPy 1.26+（代码在当前 NumPy 2.5.4 上验证）
- Pillow 10+（测试中用 Pillow 校验 PNG；渲染器自身使用固定 zlib 编码以稳定字节）

## 安装

Windows PowerShell 或命令提示符：

```powershell
python --version
python -m pip install -r requirements.txt
```

Linux/macOS：

```bash
python3 --version
python3 -m pip install -r requirements.txt
```

## 快速开始

生成确定性动画样本：

```powershell
python tools/generate_scene.py --output examples/demo_scene.json --frames 4 --width 640 --height 360 --shadow-resolution 512
```

全量渲染：

```powershell
python renderer.py --scene examples/demo_scene.json --output out_demo --threads 1
```

多线程渲染：

```powershell
python renderer.py --scene examples/demo_scene.json --output out_demo_mt --threads 4
```

命令行同时接受 `--input/--out/--jobs/--append` 别名。线程数上限为 `os.cpu_count()`；默认值为 1。

## 输出

默认输出：

```text
out_demo/
  frame_0000.png
  frame_0000.json
  frame_0001.png
  frame_0001.json
  ...
  manifest.json
```

PNG 为 8 位 RGB，不带时间戳或可变文本块。逐帧 JSON 示例：

```json
{
  "depth_max": 14.766991,
  "depth_min": 4.578458,
  "frame": 0,
  "frame_hash": "…",
  "invalid_pixel_count": 0,
  "issues": { "...": 0 },
  "material_pixel_counts": [102742, 2830, 2247, 1821],
  "nan_pixel_count": 0,
  "rendered_pixel_count": 109640,
  "shadow_bias": 0.003930,
  "shadow_cascade_pixel_counts": [42880, 62803, 3957],
  "shadow_cascade_valid_ratio": [0.530083, 0.237720, 0.048454],
  "shadow_cascades": [ ... ],
  "shadow_resolution": 512,
  "structure_hash": "…",
  "triangle_count": 370,
  "vertex_count": 192,
  "width": 640,
  "height": 360
}
```

所有统计中的浮点数均固定输出 6 位小数。深度是相机空间正深度；阴影有效占比定义为该级深度图中写入过几何的纹素数除以 `resolution²`。

`manifest.json` 保存完整场景哈希、稳定结构哈希、逐帧姿态哈希以及每个 PNG/JSON 的 SHA-256。清单不包含时间、线程数或渲染耗时，所以同一最终场景经“增量追加”或“全量重渲”得到完全相同的清单和帧文件。

## 场景格式

完整样本见 `examples/demo_scene.json`。核心结构：

```json
{
  "schema_version": 1,
  "scene_id": "deterministic_skinned_demo",
  "output": {
    "width": 640,
    "height": 360,
    "background": [0.025, 0.03, 0.045],
    "frame_pattern": "frame_{frame:04d}.png",
    "stats_pattern": "frame_{frame:04d}.json",
    "manifest_name": "manifest.json"
  },
  "camera": {
    "position": [4.6, 2.75, 6.1],
    "target": [0.0, 1.05, 0.0],
    "up": [0.0, 1.0, 0.0],
    "fov_y_degrees": 43.0,
    "near": 0.1,
    "far": 30.0
  },
  "lights": {
    "ambient": [0.11, 0.12, 0.15],
    "directional": {
      "direction": [0.42, 0.88, 0.32],
      "color": [1.0, 0.93, 0.82],
      "intensity": 1.15,
      "cast_shadow": true
    },
    "point": {
      "position": [4.6, 2.75, 6.1],
      "color": [0.55, 0.68, 1.0],
      "intensity": 2.0,
      "falloff": 0.045
    }
  },
  "shadow": {
    "resolution": 512,
    "splits": [6.0, 11.0, 24.0],
    "bias": 0.001,
    "slope_bias": 0.004,
    "normal_offset": 0.014,
    "extent_padding": 1.14,
    "depth_padding": 2.5
  },
  "materials": [
    {
      "name": "floor",
      "diffuse": [0.42, 0.47, 0.52],
      "specular": [0.08, 0.08, 0.09],
      "shininess": 24.0
    }
  ],
  "skeleton": {
    "bones": [
      {
        "name": "ground",
        "parent": -1,
        "inverse_bind_matrix": [16 numbers],
        "local_frames": [
          {"frame": 0, "matrix": [16 numbers]}
        ]
      }
    ]
  },
  "animation": {"fps": 24, "frames": [0, 1, 2]},
  "mesh": {
    "vertices": [[x, y, z], ...],
    "normals": [[x, y, z], ...],
    "skin_indices": [[b0, b1, b2, b3], ...],
    "skin_weights": [[w0, w1, w2, w3], ...],
    "triangles": [{"indices": [i, j, k], "material": 0}, ...]
  }
}
```

约定与限制：

- 矩阵使用列向量约定和行主序展开；接受 16 个数的扁平数组或 4×4 嵌套数组。
- `parent` 为 `-1` 或同骨骼数组中的父索引；父子链必须无环。
- 每根骨骼的 `local_frames` 必须恰好覆盖 `animation.frames` 中的所有帧。
- 权重数量为 1 到 4；每个顶点的权重必须非负且总和为 1，容差为 `1e-6`。
- 骨骼索引必须落在 `[0, bone_count)`，包括权重为 0 的填充槽。
- 每个三角形必须指定有效材质槽；所有面共享同一深度缓冲。
- 顶点数上限 5000，输出上限 1280×720，阴影分辨率上限 2048。
- 方向光的 `direction` 表示从表面指向光源的方向。点光可位于相机处；缺省位置取相机位置。
- `shadow.splits` 必须恰好有 3 个严格递增的正距离。省略时按 `distance`/相机远平面和 `split_lambda` 计算三级拆分。

## 渲染算法

每帧流程：

1. 按父先子后的拓扑顺序计算骨骼世界矩阵 `W_b`，蒙皮矩阵为 `S_b = W_b · inverse_bind_b`。
2. 每个顶点使用最多 4 个权重执行 `p' = Σ w_i · (S_i · [p, 1])`；法线使用蒙皮矩阵 3×3 部分的逆转置，逐骨骼变换后混合归一化。
3. 相机视图变换、近/远平面裁剪、透视投影，将三角形光栅化到深度 G-buffer（世界坐标、法线、材质槽、正深度）。
4. 在光空间按三级相机距离切片拟合正交视锥，分别生成三级深度图。分辨率 ≤1024 时增加分辨率相关深度偏移、斜率偏移和法线偏移，并使用 3×3 PCF 采样。
5. 按像素法线计算环境光、方向光漫反射/高光（受阴影可见性调制）和点光漫反射/高光。使用 Blinn-Phong 半程向量。
6. 线性光照结果做 sRGB 转换、固定舍入并写出 8 位 RGB。所有几何与光照计算统一使用 `float64`。

光栅化和着色均按行条带处理：三角形光栅每块最多 32 行，着色每块最多 64 行。因此单个临时像素数组远小于 256 MB。

## 增量导出

`--incremental` 的恢复流程：

1. 读取输出目录中的 `manifest.json`，校验格式版本、输出配置、稳定结构哈希和逐帧姿态哈希。
2. 对清单内每个已导出 PNG/JSON 重新计算 SHA-256；任一文件缺失或被修改都视为输出不可信。
3. 识别已验证已导出帧的最大帧号，只渲染编号更大的新帧。
4. 若静态场景、已有帧姿态、文件内容或输出配置发生变化，则删除旧的生成文件并整体重建。

为支持“只追加关键姿态”同时保留内容哈希保护，清单同时保存：

- `scene_hash`：完整场景 JSON 的规范哈希。
- `structure_hash`：排除 `animation.frames` 和各骨骼 `local_frames` 后的静态场景哈希。
- 每帧 `frame_hash`：该帧全部骨骼局部矩阵的规范哈希。

因此追加新帧会让完整 `scene_hash` 改变，但若结构哈希和所有旧帧哈希保持不变，增量模式仍将其识别为纯追加。任何旧帧或静态内容的修改都会触发全量重建。

追加示例：

```powershell
# 先生成并渲染 2 帧
python tools/generate_scene.py --output examples/incremental.json --frames 2 --width 640 --height 360 --shadow-resolution 512
python renderer.py --scene examples/incremental.json --output out_incremental --incremental

# 同一场景追加为 4 帧，只处理第 2、3 帧
python tools/generate_scene.py --output examples/incremental.json --frames 4 --width 640 --height 360 --shadow-resolution 512
python renderer.py --scene examples/incremental.json --output out_incremental --incremental --threads 2
```

## 错误码

所有业务错误以单行 JSON 写入 stderr，并返回非零整数退出码：

```json
{"error":{"code":"E_BONE_CYCLE","message":"bone parent chain contains a cycle","path":"skeleton.bones[2].parent"}}
```

| 错误码 | 退出码 | 含义 |
|---|---:|---|
| `E_IO` | 3 | 场景无法读取 |
| `E_SCHEMA` | 10 | JSON 结构、类型、缺失字段或非法值 |
| `E_MATRIX_DIM` | 11 | 矩阵不是 16 元素或 4×4 |
| `E_BONE_PARENT` | 12 | 父骨骼索引越界 |
| `E_BONE_CYCLE` | 13 | 骨骼父子链有环或自父 |
| `E_WEIGHT_COUNT` | 14 | 权重数量超过 4 或为空 |
| `E_WEIGHT_SUM` | 15 | 权重总和不是 1 |
| `E_WEIGHT_BONE_RANGE` | 16 | 权重引用的骨骼索引越界 |
| `E_WEIGHT_RANGE` | 17 | 权重为负 |
| `E_INDEX` | 18 | 三角形顶点索引越界 |
| `E_MATERIAL_RANGE` | 19 | 三角形材质槽越界 |
| `E_CAMERA` | 20 | 相机参数或基向量非法 |
| `E_SHADOW` | 21 | 级联拆分/阴影参数非法 |
| `E_LIMIT` | 22 | 顶点、分辨率或阴影分辨率超限 |
| `E_KEYFRAME` | 23 | 帧号重复、缺失或局部姿态不完整 |
| `E_OUTPUT` | 24 | 输出目录或文件写入错误 |
| `E_THREADS` | 31 | 线程数不在 `[1, CPU 核数]` |
| `E_USAGE` | 32 | 命令行参数错误 |
| `E_RENDER` | 40 | 渲染阶段异常 |

运行时出现的非有限顶点、被跳过的无效三角形、非有限着色结果等不会中止剩余帧：计数写入该帧 `issues`、`invalid_pixel_count`、`nan_pixel_count`，无效像素以背景色写出，其他帧继续处理。

## 确定性与线程一致性

- 单线程和多线程均只按帧并行；每一帧有独立的 G-buffer、阴影图、统计对象和输出文件，不存在共享可写累加器。
- PNG 使用固定扫描线 `filter=0`、zlib level 9、无辅助块；相同输入重复运行得到逐字节相同 PNG。
- 统计 JSON 使用排序键和固定 6 位小数，不包含运行时间、线程数、绝对路径等易变字段。
- 清单按帧号排序，文件内容哈希来自实际写出的字节；全量渲染与正确的增量追加结果逐字节一致。

## 测试与验证

```powershell
python -m unittest discover -s tests -v
```

当前环境（Python 3.14.4、NumPy 2.5.4、Pillow 13.0.0、24 CPU）的实际结果：

```text
Ran 8 tests in 9.228s
OK
```

自动化测试覆盖：

- 3 帧多材质蒙皮动画；单线程、4 线程和重复运行全部输出逐字节一致。
- 3 级阴影有效占比、固定小数位、材质像素覆盖、深度范围。
- 增量 2→4 帧追加只渲染新帧；旧帧 mtime 和字节不变；结果与全量渲染逐字节一致。
- 同场景再次增量运行全部跳过；结构变化、旧帧被篡改或输出目录含陈旧帧时整体重建。
- `E_SCHEMA`、`E_BONE_CYCLE`、`E_MATRIX_DIM`、`E_WEIGHT_COUNT`、`E_WEIGHT_SUM`、`E_WEIGHT_BONE_RANGE`、`E_THREADS`。
- 运行时非有限几何写入统计但不中止其余帧。

1280×720、1024 阴影图的单帧实测：

```text
Elapsed: 0.78 s
Maximum resident set size: 172552 KB
rendered_pixel_count > 0
shadow_cascade_valid_ratio: 3 级均非零
```

资源边界样本使用 5000 顶点、9702 个三角形、两个材质槽、1280×720 输出和 1024 阴影图：

```powershell
python tools/generate_stress_scene.py --output examples/stress_5000.json --width 1280 --height 720 --shadow-resolution 1024
python renderer.py --scene examples/stress_5000.json --output out_stress --threads 1
```

实测结果：

```text
Elapsed: 2.88 s
Maximum resident set size: 142672 KB
rendered_pixel_count: 154566
material_pixel_counts: [77292, 77274]
invalid_pixel_count: 0
nan_pixel_count: 0
shadow_cascade_valid_ratio: [0.150950, 0.057425, 0.015501]
```
