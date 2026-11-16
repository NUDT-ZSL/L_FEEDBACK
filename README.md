# 离线材质校验工具

这是一个面向游戏材质流水线的确定性离线校验器。它读取 JSON 材质清单和成对的 GLSL
顶点/片元着色器，在无窗口 OpenGL 3.3+ 上下文中完成编译、四目标 G-Buffer 渲染、
延迟着色与正向渲染对比，并输出结构化报告和可直接逐像素检查的 `float32` 缓冲。

工具不依赖外部贴图、模型或场景素材。内置的立方体、平面、零法线平面、相机矩阵、
光源和测试清单都由代码固定生成。

## 安装与快速运行

目标解释器为 Python 3.10/3.11；当前实现也已在 Python 3.14 上验证。

```powershell
py -3.11 -m pip install "numpy>=1.23" "moderngl>=5.8"
py -3.11 -m pip install -e .
py -3.11 -m material_validator check-gl
```

若 ModernGL 的独立上下文组件未随 wheel 安装：

```powershell
py -3.11 -m pip install "moderngl[headless]>=5.8"
```

生成并运行确定性有效样本：

```powershell
py -3.11 -m material_validator generate-sample --out examples --kind valid
py -3.11 -m material_validator --manifest examples/valid_manifest.json --out outputs\valid --parallel 4 --render-parallel 4
```

运行故意包含错误的样本：

```powershell
py -3.11 -m material_validator generate-sample --out examples --kind invalid
py -3.11 -m material_validator --manifest examples/invalid_manifest.json --out outputs\invalid --parallel 4
```

运行透明混合对比样本：

```powershell
py -3.11 -m material_validator generate-sample --out examples --kind blend
py -3.11 -m material_validator --manifest examples/blend_manifest.json --out outputs\blend
```

同一命令重复执行会产生相同的报告、浮点缓冲和 SHA-256。`--parallel` 控制材质编译线程数，
`--render-parallel` 控制场景渲染线程数；每个工作线程使用独立的 EGL/WGL 上下文。

## CLI

```text
material-validator --manifest PATH --out DIR
                    [--parallel N] [--render-parallel N]
                    [--abs-tol X] [--rel-tol X]
                    [--unused-uniform-policy manifest|warn|error|ignore]
                    [--no-render] [--write-raw] [--fail-on-warning]

material-validator generate-sample --out DIR --kind valid|invalid|blend
material-validator check-gl
```

退出码：

- `0`：所有检查通过且所有场景对比通过。
- `1`：存在编译、装配、状态、G-Buffer 或像素阈值错误。报告仍会写出。
- `2`：清单/IO/上下文等无法继续执行的初始化错误。若输出目录可写，也会写出最小错误报告。

主要输出：

- `report.json`：完整结构、材质/通道聚合诊断、G-Buffer 统计、对比结果、文件校验值。
- `summary.txt`：固定格式的短摘要。
- `buffers/*.npy`：`float32`、C 顺序、形状为 `(height, width, channels)`。
- `buffers/*.difference.npy`：有符号 `deferred - forward`。
- `--write-raw` 时额外写出 `.f32` 原始小端浮点缓冲。

GL 上下文可用 `MATERIAL_VALIDATOR_GL_BACKEND=egl` 或 `wgl` 强制指定；默认依次尝试
EGL、WGL、ModernGL 自动选择。

## 清单结构

完整样本见 `examples/valid_manifest.json`。最小结构如下：

```json
{
  "schema_version": "1.1",
  "render": {
    "width": 160,
    "height": 120,
    "background": [0.012, 0.016, 0.024, 1.0],
    "normal_fallback": [0.0, 0.0, 1.0],
    "normal_tolerance": 0.0001
  },
  "comparison": {
    "absolute": 0.0002,
    "relative": 0.002,
    "channels": ["R", "G", "B"]
  },
  "uniform_policy": {"unused_declared": "warn"},
  "materials": [
    {
      "name": "stone",
      "vertex_shader": "shaders/standard.vert",
      "fragment_shader": "shaders/standard.frag",
      "uniforms": {
        "u_base_color": {"type": "vec3", "value": [0.52, 0.61, 0.68]},
        "u_metallic": {"type": "float", "value": 0.05},
        "u_roughness": {"type": "float", "value": 0.72},
        "u_tint_matrix": {
          "type": "mat3",
          "value": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        },
        "u_enable_tint": {"type": "bool", "value": true},
        "u_opacity": {"type": "float", "value": 1.0},
        "u_vertex_offset": {"type": "float", "value": 0.0}
      },
      "blend": {
        "enabled": false,
        "src_factor": "one",
        "dst_factor": "zero",
        "equation": "add"
      },
      "depth": {"test": true, "write": true, "func": "lequal"}
    }
  ],
  "channels": [
    {"name": "normal", "kind": "normal", "materials": ["stone"]},
    {"name": "depth", "kind": "depth_prepass", "materials": ["stone"]},
    {"name": "deferred", "kind": "deferred", "materials": ["stone"]}
  ],
  "scenes": [
    {
      "name": "studio",
      "camera": {
        "position": [1.65, 1.25, 4.25],
        "target": [0, 0, 0],
        "up": [0, 1, 0],
        "fov_y_deg": 42,
        "near": 0.1,
        "far": 50
      },
      "lights": [
        {
          "name": "key",
          "type": "directional",
          "direction": [0.35, -0.8, -0.5],
          "color": [1.0, 0.95, 0.88],
          "intensity": 1.15
        }
      ],
      "objects": [
        {
          "name": "cube",
          "mesh": "cube",
          "material": "stone",
          "position": [0, 0, 0],
          "rotation": [12, 24, 0],
          "scale": [0.5, 0.5, 0.5]
        }
      ]
    }
  ]
}
```

材质可写成数组，也可写成 `{ "stone": {...}, "metal": {...} }` 对象映射。`passes` 可省略；
若没有顶层 `channels`，材质自己的 `passes` 用于推导三个通道。

### 版本兼容

| 版本 | 缺省宽度/高度 | 缺省 abs/rel | 缺省法线容差 |
| --- | --- | --- | --- |
| `1.0` | `128 x 96` | `5e-4 / 2e-3` | `2e-4` |
| `1.1` | `160 x 120` | `1e-4 / 1e-3` | `1e-4` |

缺少 `schema_version` 时按 `1.0` 处理并产生警告。未知版本会报错，同时使用 `1.1` 默认值完成
可诊断的报告生成。未知字段不会被解释，但会原值保留在报告的 `unknown_fields` 中并产生警告。

### 通道与参数共享

三个必需通道是 `depth_prepass`、`normal`、`deferred`。执行顺序固定为
`depth_prepass -> normal(G-Buffer) -> deferred -> forward`，与清单中的声明顺序无关。
三个通道的材质集合必须一致，否则报告 `channel_material_mismatch`。

同一材质可被多个通道引用。材质 uniform 只能有一份共享值；通道引用可以用
`{"name": "...", "uniforms": {...}}` 提供覆盖声明。若同一 uniform 在基础材质或不同通道间出现
不同值，报告 `channel_uniform_conflict`，`details.sources` 列出全部来源路径，该材质被拒绝渲染，
不会静默采用最后一次声明。

## 着色器契约

顶点属性：

- `in vec3 in_position`
- `in vec3 in_normal`

片元输出必须显式或按顺序占用以下槽位：

| Location | 输出 | 含义 |
| --- | --- | --- |
| 0 | `frag_world_position` | 世界空间位置，alpha 为 1 |
| 1 | `frag_world_normal` | 归一化世界空间法线，alpha=1 表示有效；0 表示零长度回退 |
| 2 | `frag_albedo` | RGB 反照率；A=1 表示该像素被几何覆盖 |
| 3 | `frag_metallic_roughness` | RG 为夹取后的金属度/粗糙度；B/A 为原始值越界标记 |
| 4 | `frag_color` | 正向渲染最终颜色 |

渲染器提供的保留 uniform 不要求在材质清单中重复声明：

```text
u_model, u_view_proj, u_normal_matrix,
u_camera_position, u_ambient_color, u_normal_fallback, u_background_color,
u_light_count, u_light_position[4], u_light_color[4],
u_light_intensity[4], u_light_radius[4]
```

矩阵 uniform 的 JSON 值按行主序书写。上传时工具会转置为 OpenGL 列主序字节。支持标量
`float/int/uint`、布尔、`vecN/ivecN/uvecN/bvecN` 和 `mat2/3/4`。材质 uniform 数组不支持。

`uniform_policy.unused_declared` 或 CLI 覆盖值决定“清单声明但链接后未使用”的策略：
`warn`（默认）、`error`、`ignore`。着色器实际读取但清单未声明的 uniform 始终是材质装配错误，
报告包含着色器源文件、阶段和声明行号。

## G-Buffer 与场景

正常通道把几何写入四张 `RGBA32F` 纹理。渲染前视口、清除值、深度状态和对象顺序固定；
报告中的每张纹理都是 `(height, width, 4)`。法线纹理 alpha 同时作为回退标记，因此仍保持
“四张 G-Buffer 纹理”的约束。

零长度法线由材质片元着色器写为清单中的 `render.normal_fallback`（默认 `(0,0,1)`），
alpha 写 0。渲染后工具统计覆盖像素中的回退数量，并验证回退值与单位长度。金属度/粗糙度
先夹取到 `[0,1]`，越界原始输入由纹理的 B/A 通道记录。清单中的 `u_metallic`/`u_roughness`
越界也会在解析阶段直接报错。

内置网格名为 `cube`、`plane`、`zero_normal_plane`。场景对象使用世界位置、Z-Y-X 顺序的
欧拉角旋转和缩放；相机采用右手系 look-at 与透视投影。光源支持 `point` 和 `directional`，
最多四个；方向光被确定性地转换为远距离光源。

## 状态校验

混合因子支持：

```text
zero, one, src_color, one_minus_src_color, src_alpha,
one_minus_src_alpha, dst_alpha, one_minus_dst_alpha,
dst_color, one_minus_dst_color, src_alpha_saturate
```

混合方程支持 `add`、`subtract`、`reverse_subtract`、`min`、`max`。深度函数支持
`never/less/equal/lequal/greater/notequal/gequal/always`。

开启混合同时开启 `depth.write` 会被视为不自洽并拒绝材质，报告
`blend_with_depth_write`、材质名和清单路径。不支持的因子/方程/深度函数也会带路径报错。
深度测试和深度写入在三个几何相关阶段生效；正向阶段还会应用材质的混合状态。G-Buffer
本身禁用颜色混合，以避免位置/法线等语义通道被透明合成污染；混合差异由正向/延迟对比和
`floating_blend_order` 分类显式暴露。

## 对比与分类

对配置通道逐像素检查：

```text
abs(deferred - forward) <= abs_tol + rel_tol * max(abs(deferred), abs(forward))
```

报告包含超阈值像素数、每通道超阈值数、全局最大绝对偏差、所在通道和像素坐标。缓冲由 GL
回读，报告坐标的 row 0 是缓冲第一行。`difference.npy` 保留有符号差值。

超阈值像素按以下优先级互斥分类：

1. `material_parameter_range`：像素的金属度/粗糙度越界标记或原始值越界。
2. `normal_encoding_precision`：像素使用了零长度法线回退。
3. `floating_blend_order`：像素被至少一个开启混合的材质覆盖。
4. `unclassified`：无法归入上述原因。

分类是基于 G-Buffer 标记和混合覆盖掩码的确定性诊断，不替代具体引擎的数值根因分析。

## 确定性与并行

- 材质结果和场景结果按清单顺序合并，与任务完成顺序无关。
- 每个 GL 工作线程创建并销毁自己的上下文，不共享可变 GL 状态。
- 固定对象遍历顺序、固定光照数组、关闭抖动/裁剪/多重采样、使用 `float32` 纹理和无窗口 FBO。
- 报告不含时间戳，JSON 键排序固定。
- 测试分别以编译/渲染并行度 `1` 与 `4` 渲染，所有输出缓冲 SHA-256 相同。

## 测试

```powershell
py -3.11 -m pytest -q
```

本次实现的实际验证结果见 `VERIFICATION.md`。

## 当前边界

- 不加载外部贴图、法线贴图或复杂 PBR 预滤波环境；样例使用共享程序化 BRDF。
- 混合状态只有一组 RGB 因子，不单独配置 alpha 因子或 `glBlendEquationSeparate`。
- G-Buffer 混合不参与透明合成，透明材质适合在正向阶段验证；跨路径差异会显式分类。
- 分类使用像素覆盖标签，不追踪材质内部的具体浮点运算指令。
