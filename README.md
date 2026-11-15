# 离线材质图编译器与 CPU 渲染验证器

这是一个纯本地、无 GPU、无网络的确定性材质渲染管线。输入为 JSON 材质节点图和
JSON 场景，编译为带稳定指令顺序的内存 IR，再由 NumPy/Pillow 在 CPU 上完成解析
光线求交、G-Buffer 构建、材质指令求值、直接光照与通道导出。

## 快速开始

目标依赖为 Python 3.10+、`numpy==1.26.4`、`pillow==10.4.0`：

```bash
python -m pip install -r requirements.txt
python run.py generate-samples --out samples
python run.py validate --graph samples/material_graph.json
python run.py render \
  --graph samples/material_graph.json \
  --scene samples/scene.json \
  --out outputs/sample
```

Windows PowerShell 用法相同：

```powershell
py -3.10 -m pip install -r requirements.txt
py -3.10 run.py generate-samples --out samples
py -3.10 run.py render --graph samples/material_graph.json --scene samples/scene.json --out outputs/sample
```

`render` 会生成：

| 文件 | 位深/编码 | 含义 |
|---|---|---|
| `base_color.png` | 8-bit sRGB RGB | 材质基础色 |
| `normal.png` | 16-bit linear RGB | 世界空间单位法线，按 `n*0.5+0.5` 编码 |
| `roughness.png` | 16-bit linear gray | 粗糙度 |
| `metallic.png` | 16-bit linear gray | 金属度 |
| `emissive.png` | 16-bit linear RGB，钳制到 0..1 | 自发光通道 |
| `final.png` | 8-bit sRGB RGB | 方向光 + 半球环境光后的最终 LDR 图像 |
| `report.json` | JSON | 像素计数、通道统计、G-Buffer 统计、编译信息 |
| `material_ir.json` | JSON | 稳定指令序列与 IR SHA-256 |

非法输入不会创建或改写输出目录。CLI 在 stderr 输出结构化错误并返回退出码 2。

## 项目结构

```text
run.py                     统一入口
verify.py                  完整本地验证与哈希记录脚本
matgraph/compiler.py       JSON 校验、类型推导、循环/深度/死节点检测、折叠、IR
matgraph/texture.py        本地 PNG 加载、sRGB 解码、确定性采样
matgraph/scene.py          场景、相机、光源和几何校验
matgraph/raster.py         解析球/平面求交与 G-Buffer
matgraph/evaluate.py       IR 逐像素求值、切线帧与法线映射
matgraph/lighting.py       GGX 直接光、半球环境光、ACES 与 sRGB
matgraph/renderer.py       渲染编排、报告和六张 PNG
matgraph/outputs.py        原子输出目录发布
matgraph/samples.py        确定性样本图和纹理生成
tests/                     标准库 unittest 测试
samples/                   已生成样本
verification/              两次独立渲染和实际验证记录
```

## 材质图格式

节点索引是 `nodes` 数组中的从 0 开始的下标。所有错误都会尽量附带
`node_index`、`node_id` 和字段路径。

```json
{
  "schema": "material-graph/v1",
  "name": "example",
  "textures": {
    "albedo": {
      "path": "textures/albedo.png",
      "color_space": "srgb",
      "wrap": "repeat",
      "filter": "bilinear"
    }
  },
  "nodes": [
    {"id": "uv", "type": "uv"},
    {
      "id": "base_tex",
      "type": "texture_sample",
      "texture": "albedo",
      "inputs": {"uv": "uv"}
    },
    {
      "id": "normal",
      "type": "normal_map",
      "inputs": {"normal": "normal_tex", "strength": "strength"}
    }
  ],
  "outputs": {
    "base_color": "base_tex",
    "normal": "normal",
    "roughness": "roughness",
    "metallic": "metallic",
    "emissive": "emissive"
  }
}
```

支持节点：

| `type` | 输入 | 输出维度 | 说明 |
|---|---|---:|---|
| `constant` | 无；`value` 为标量、vec2 或 vec3 | 1/2/3 | 常量必须在 `[-1e6, 1e6]` 内 |
| `uv` | 无 | 2 | 当前像素的几何 UV |
| `texture_sample` | `uv`: vec2；`texture` 指向纹理表 | 3 | 双线性/最近邻采样，按需 sRGB 转线性 |
| `add` | `a`, `b` | 最大输入维度 | 支持标量广播和同维向量相加 |
| `multiply` | `a`, `b` | 最大输入维度 | 支持标量广播和同维向量逐分量乘 |
| `mix` | `a`, `b`: 同维；`factor`: 标量 | 与 a/b 相同 | `a*(1-factor)+b*factor` |
| `fresnel` | `normal`: vec3；`ior`: 标量 | 1 | Schlick 菲涅尔，IOR 必须为 1..4 |
| `normal_map` | `normal`: 编码切线法线 vec3；`strength`: 标量 | 3 | 使用当前像素 UV 偏导构造 TBN，输出世界法线 |

五个必需输出：

- `base_color`：vec3
- `normal`：vec3，最终会被归一化
- `roughness`：标量
- `metallic`：标量
- `emissive`：vec3

纹理路径必须相对于材质图文件，只允许 PNG，禁止绝对路径和 `..` 越界。
`color_space` 为 `srgb` 或 `linear`，`wrap` 为 `repeat` 或 `clamp`，
`filter` 为 `bilinear` 或 `nearest`。

### 编译规则

1. 先校验根对象、schema、节点类型、ID、输入集合和纹理声明。
2. 校验引用、类型和维度；`add`/`multiply` 只允许同维或标量广播。
3. DFS 检测环和最大深度；随后检查每个输出引用和所有节点是否可达。
4. 使用以节点 ID 字典序为平局规则的 Kahn 拓扑排序，保证稳定指令顺序。
5. 所有输入均为常量的 `add`、`multiply`、`mix` 以及常量 UV 的纹理采样会在编译期
   求值并消除指令；常量直接内联到消费者操作数中。
6. `uv`、`fresnel`、`normal_map` 等依赖像素上下文的节点不会被错误折叠。

`material_ir.json` 中每个操作数都是常量或临时值 `tN`，并包含维度、节点索引、
纹理元数据和 `limits`。IR 的 SHA-256 对不含该哈希字段的规范 JSON 计算。

## 场景格式

```json
{
  "schema": "scene/v1",
  "name": "sphere-on-plane",
  "camera": {
    "position": [0.0, 1.38, 3.45],
    "target": [0.0, 0.70, 0.0],
    "up": [0.0, 1.0, 0.0],
    "fov_degrees": 43.0
  },
  "render": {
    "width": 320,
    "height": 240,
    "exposure": 1.1,
    "background": [0.012, 0.017, 0.028],
    "ambient": {
      "sky": [0.18, 0.23, 0.32],
      "ground": [0.03, 0.025, 0.02]
    }
  },
  "lights": [
    {
      "type": "directional",
      "direction": [0.45, 0.90, 0.35],
      "color": [3.1, 2.8, 2.5]
    }
  ],
  "objects": [
    {"type": "sphere", "center": [0.0, 0.72, 0.0], "radius": 0.72},
    {
      "type": "plane",
      "point": [0.0, 0.0, 0.0],
      "normal": [0.0, 1.0, 0.0],
      "size": [8.0, 8.0]
    }
  ]
}
```

`direction` 是从表面指向光源的方向。`size` 是有限平面的两个边长。所有物体共享
同一个编译材质。可用 `--width`、`--height` 覆盖分辨率，仍受资源上限约束。

## 渲染语义

- 相机使用水平视场角、透视投影和像素中心主光线。
- 球体使用解析二次方程；有限平面使用解析平面方程。深度是正的光线距离。
- 所有命中像素，包括轮廓边缘像素，深度直接来自解析求交，不用插值深度或近似。
- G-Buffer 保存世界坐标、几何法线、UV、深度、对象 ID、有效掩码和轮廓边缘掩码。
- 切线帧由屏幕相邻像素的世界位置差分和 UV 差分求解 `dP/du`、`dP/dv`；
  UV 周期差会先展开。行列式退化、零长度切线或非有限值时，使用由几何法线构造的
  确定性正交基，并计入 `degenerate_tangent_pixels`。
- 法线贴图输入解释为编码切线空间法线 `[0,1]^3`，解码为 `[-1,1]^3` 后按强度缩放、
  经 TBN 变换并重归一化。最终材质法线再次归一化；零向量回退到几何法线并计入
  `normal_fallback_pixels`。
- 直接光使用 GGX 法线分布、Smith-Schlick 几何项和 Schlick 菲涅尔。基础色按金属度
  在漫反射与 F0 之间分配，粗糙度同时约束高光。半球环境光在天空色与地面色间插值。
- 最终辐射亮度先乘曝光，再用 ACES Narkowicz 曲线映射到 `[0,1]`，最后执行 sRGB
  传递函数和 8-bit 量化。最终通道严格位于 0..255。

## 报告

`report.json` 的 `counts` 包含：

- `total_pixels`
- `valid_pixels`
- `shaded_pixels`
- `degenerate_tangent_pixels`
- `normal_fallback_pixels`
- `edge_pixels`

`channels` 为五个材质通道和 `final_ldr` 提供 `mean`、`min`、`max`、`count`。
其中材质通道只统计有效像素，法线和颜色统计按分量给出。`g_buffer` 提供世界位置、
几何法线和深度的统计，并记录深度约定与边缘深度方法。报告浮点数统一舍入到
10 位小数，JSON 使用排序键和固定缩进。

## 错误模型

所有可预期失败返回：

```json
{
  "ok": false,
  "error": {
    "code": "E_MISSING_INPUT",
    "message": "missing required input 'uv' for node type 'texture_sample'",
    "stage": "graph",
    "node_index": 1,
    "node_id": "base_tex",
    "field": "inputs.uv"
  }
}
```

主要错误码：

| 错误码 | 含义 |
|---|---|
| `E_INVALID_JSON`, `E_ROOT_TYPE`, `E_SCHEMA` | JSON/根对象/schema 非法 |
| `E_NODE_SCHEMA`, `E_NODE_TYPE`, `E_NODE_ID`, `E_DUPLICATE_ID` | 节点结构非法 |
| `E_INPUT_UNKNOWN`, `E_MISSING_INPUT`, `E_INPUT_NOT_FOUND` | 输入集合或引用非法 |
| `E_CONSTANT_TYPE`, `E_INPUT_TYPE`, `E_INPUT_DIMENSION` | 常量/类型/维度非法 |
| `E_CYCLE`, `E_GRAPH_TOO_DEEP`, `E_NO_OUTPUT_PATH` | 图结构非法 |
| `E_OUTPUT_MISSING`, `E_OUTPUT_REF`, `E_OUTPUT_TYPE`, `E_OUTPUT_UNKNOWN` | 输出非法 |
| `E_TEXTURE_REF`, `E_TEXTURE_PATH`, `E_TEXTURE_FORMAT`, `E_TEXTURE_IO` | 纹理非法 |
| `E_SCENE_*` | 场景字段、相机、光源或几何非法 |
| `E_NUMERIC` | 编译或求值产生非有限值/超出定义域 |
| `E_LIMIT` | 节点、深度、分辨率、像素、纹理或 JSON 大小超限 |
| `E_OUTPUT_IO` | 输出发布失败 |

节点级错误带节点索引；根级、输出级或场景级错误没有可归属节点时
`node_index` 为 `null`。CLI 还会用 `--error-out` 原子写出同一错误对象。

默认资源上限：

```text
max_nodes=512
max_graph_depth=64
max_width=1024
max_height=1024
max_pixels=1048576
max_texture_dimension=2048
max_texture_bytes=16 MiB
max_json_bytes=8 MiB
max_objects=64
max_lights=8
```

## 确定性与原子输出

- 图、场景和纹理全部来自本地文件，无网络、无 GPU、无真实素材。
- 指令拓扑平局按节点 ID 排序，纹理和输出文件按名称排序。
- 所有材质计算使用 `float64`；PNG 使用固定过滤类型 0 和固定 zlib level 9 自写编码器，
  不写入时间戳或可变元数据。
- 报告和 IR 使用排序键 JSON，浮点统计固定舍入。
- 所有六个 PNG、报告和 IR 先完全生成在内存，再写进同父目录的暂存目录；成功后整体
  替换目标目录。失败时不发布暂存内容，已有目标目录也会回滚。
- 两次独立 CLI 渲染的八个文件在当前样本上逐字节相同。

## 测试与实测结果

运行：

```bash
python -m unittest discover -s tests -v
python verify.py
```

`verify.py` 会重建样本、执行两次独立 CLI 渲染、比较所有输出文件 SHA-256、验证非法图
不产生输出目录，并运行测试；结果写入 `verification/verification.json`。

本次实际验证（当前执行环境 Python 3.14.4、NumPy 2.5.4、Pillow 13.0.0）：

- 18 个 `unittest` 测试全部通过，结果 `OK`。
- 样本分辨率 320x240，总像素 76800，有效/着色像素 45594，轮廓边缘像素 932。
- 样本材质 IR SHA-256：`abb9f09bd74bd62e4f9007872b40b8350e37c13dcfe3bdddfcd507214b7497d6`。
- 两次独立 CLI 渲染的 `base_color.png`、`normal.png`、`roughness.png`、
  `metallic.png`、`emissive.png`、`final.png`、`report.json`、`material_ir.json`
  逐文件 SHA-256 完全一致。
- 非法图（删除 `base_tex` 的 `uv` 输入）返回 `E_MISSING_INPUT`、
  `node_index=1`、`node_id="base_tex"`，退出码 2，且未创建输出目录。
- 合成退化 UV 测试覆盖全部 16 个像素，全部使用确定性切线回退并记录，
  输出法线仍为单位向量。

完整逐文件哈希和样本输入哈希见 `verification/verification.json`；实际渲染可在
`verification/run_a` 与 `verification/run_b` 查看。

## 当前边界

- 所有物体共享一个材质图，尚未实现逐物体材质槽。
- 使用像素中心单主光线，没有 MSAA/超采样；边缘深度仍由解析求交精确计算。
- 直接光不投射阴影，环境光为半球近似。
- 法线映射使用几何 UV 推导切线帧，不接受单独的自定义切线 UV。
- 纹理仅支持本地 8-bit PNG；正常样本和贴图均由项目脚本确定性生成。
