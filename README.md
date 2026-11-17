# CPU 软光栅角色预览渲染器

这是一个不依赖 GPU、Docker 或外部图像资源的确定性软件渲染器。它读取场景 JSON，执行完整的顶点变换、近裁剪、透视投影、背面剔除、深度预通道、Hi-Z 遮挡剔除、逐像素光照、方向光/点光阴影和透明混合，输出 PNG 与可复核的 JSON 统计日志。

内置样本是一套程序化生成的角色、平台和透明面罩，包含 5,241 个顶点、7,504 个三角形、6 个材质段和 2 个光源。

## 快速运行

要求 Python 3.10+、NumPy 1.24+、Pillow 9.0+。

Windows CMD：

```bat
python -m pip install -r requirements.txt
python render.py --scene samples\character_scene.json --output artifacts\acceptance --stats artifacts\acceptance\stats.json
```

PowerShell 与 Linux/macOS：

```bash
python -m pip install -r requirements.txt
python render.py --scene samples/character_scene.json --output artifacts/acceptance --stats artifacts/acceptance/stats.json
```

也可以使用模块入口：

```bash
python -m softrender --scene samples/character_scene.json --output out --stats out/render_stats.json
```

成功时 stdout 输出一行摘要 JSON；`--stats` 省略时默认写到 `<output>/render_stats.json`。输出文件采用临时文件加替换的原子写入方式：

- `<output>/render.png`
- 指定的统计日志（默认 `<output>/render_stats.json`）

重新生成内置场景：

```bash
python samples/build_character_scene.py
```

## 场景格式

场景是 UTF-8 JSON 对象。核心结构如下：

```json
{
  "schemaVersion": 1,
  "camera": {
    "position": [3.0, 2.5, 5.3],
    "target": [0.0, 1.3, 0.0],
    "up": [0.0, 1.0, 0.0],
    "fovY": 36.0,
    "near": 0.05,
    "far": 40.0,
    "width": 512,
    "height": 512,
    "background": [0.018, 0.023, 0.034],
    "ambient": [0.11, 0.13, 0.17],
    "exposure": 2.25,
    "cullMode": "back"
  },
  "mesh": {
    "vertices": [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 1.0, 0.0]],
    "normals": [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
    "uvs": [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
    "tangents": [[1.0, 0.0, 0.0, 1.0]],
    "transform": {
      "translation": [0.0, 0.0, 0.0],
      "rotationDegrees": [0.0, 0.0, 0.0],
      "scale": [1.0, 1.0, 1.0]
    },
    "primitives": [
      {"material": "armor", "indices": [0, 1, 2]}
    ]
  },
  "materials": [],
  "lights": [],
  "debug": {}
}
```

约定：

- `mesh.primitives` 按材质分段提交；`indices` 是全局顶点索引，每组长度为 3 的倍数，三角形使用逆时针正面绕序。
- `normals`、`tangents`、`uvs` 的数组长度必须与顶点数一致。`normals` 可省略并由面积加权法线生成；`tangents` 可省略并按 UV 梯度生成，第四分量是副切线手性。
- `mesh.transform.rotationDegrees` 按 X、Y、Z 顺序应用。法线使用模型矩阵逆转置，切线使用模型线性部分。
- 相机是右手坐标系，看向 `-Z`；`fovY` 为角度，`near < far`，`width/height` 在 1 到 4096。
- JSON 中的颜色按 sRGB 解释并转换到线性空间。漫反射贴图默认也按 sRGB 解码；法线贴图按切线空间编码 `[0,1]` 读取；高光掩码使用线性值。
- 光照在线性空间累加，乘曝光后使用 IEC 61966-2-1 sRGB 传输函数编码为 8-bit PNG。

### 材质与贴图

每个材质必须提供 `name`、`baseColor`、`diffuseMap`、`normalMap`、`specularMap`。可选字段包括 `specular`、`shininess`、`opacity`、`blend`、`doubleSided`、`normalStrength`、`emission`。

贴图支持三种描述方式。

本地生成标识：

```json
{"kind": "generated", "id": "armor_normal", "width": 96, "height": 96}
```

内置标识包括 `armor_diffuse`、`armor_normal`、`armor_spec`、`cloth_*`、`skin_*`、`visor_*`、`platform_*`。

程序化图案：

```json
{
  "kind": "procedural",
  "pattern": "checker",
  "width": 64,
  "height": 64,
  "scale": 8.0,
  "colors": [[0.1, 0.2, 0.3], [0.8, 0.7, 0.5]],
  "srgb": true,
  "wrap": "repeat"
}
```

`pattern` 支持 `checker`、`stripes`、`waves`；法线贴图角色会从图案高度场生成切线空间法线。

Base64 PNG：

```json
{"kind": "base64", "format": "png", "srgb": true, "data": "iVBORw0KGgo..."}
```

### 光源与阴影

至少需要两个光源，最多八个，类型为 `directional` 或 `point`：

```json
{
  "name": "warm-key",
  "type": "directional",
  "direction": [0.48, 0.80, 0.55],
  "color": [1.0, 0.83, 0.64],
  "intensity": 3.4,
  "shadow": {
    "enabled": true,
    "resolution": 512,
    "bias": 0.0012,
    "normalBias": 0.012,
    "slopeBias": 1.4,
    "pcfRadius": 1
  }
}
```

`direction` 是“从表面指向光源”的方向。点光还需 `position`、`range`，可选 `attenuation` 的 `constant/linear/quadratic`。每个启用阴影的光源都会生成深度图：方向光使用拟合场景包围球的正交阴影图，点光使用六面立方体阴影图。`pcfRadius=1` 表示固定 3×3 核。

### 调试叠加

```json
{
  "debug": {
    "wireframe": true,
    "wireframeColor": [0.15, 1.0, 0.35],
    "normals": true,
    "normalColor": [1.0, 0.18, 0.72],
    "normalLength": 0.11
  }
}
```

叠加在基础着色完成后执行，不会改变基础着色计算；线框和法线线段仍然通过深度缓冲测试。统计日志分别记录 `wireframe_pixels` 和 `normal_overlay_pixels`。

## 渲染管线

1. **场景校验**：检查字段、类型、有限数值、索引范围、材质/光源引用、阴影参数和图像尺寸。所有业务错误写入 stderr 的 JSON。
2. **几何变换**：模型矩阵到世界空间，相机矩阵到视图空间，透视矩阵到裁剪空间；执行齐次除法、透视正确属性插值和视口映射。
3. **背面剔除与近裁剪**：按相机空间三角形朝向剔除；跨越 `z=-near` 的三角形使用 Sutherland-Hodgman 生成 3/4 顶点多边形，再扇形三角化，不丢弃跨越图元。
4. **深度预通道**：不透明三角形按材质段和前后深度提交，写入全分辨率最近深度。透明三角形不进入预通道。
5. **Hi-Z 构建与查询**：对全分辨率深度做 2×2 最大值金字塔。主通道对每个屏幕包围矩形执行保守层次查询；证明确实完全被遮挡的三角形跳过着色，其深度已由预通道保留。
6. **阴影**：从每个光源视角渲染不透明投影物；接收端按几何法线偏移、斜率缩放偏置和固定核 PCF 查询。
7. **主着色**：不透明像素通过预通道深度 Early-Z；透明材质按段和三角形从后向前混合，且不写深度。切线空间法线经 TBN 变换，至少两个光源分别计算漫反射与 Blinn-Phong 高光，高光乘材质掩码。
8. **输出**：线性颜色累加、曝光、sRGB Gamma 编码、8-bit 量化，然后原子写出 PNG 与 JSON。

## 错误码

stderr 统一格式：

```json
{"ok":false,"error":{"code":"E_SCHEMA","message":"missing required field 'camera'","path":"$.camera"}}
```

| 代码 | 退出码 | 含义 |
|---|---:|---|
| `E_INPUT_READ` | 2 | 场景/资源读取失败 |
| `E_JSON_PARSE` | 3 | JSON 或 UTF-8 解析失败 |
| `E_SCHEMA` | 4 | 必需字段缺失或类型错误 |
| `E_VALUE` | 5 | 数值越界、非有限值或引用非法 |
| `E_TEXTURE` | 6 | Base64 或图像解码失败 |
| `E_OUTPUT` | 7 | 输出目录/文件不可写或路径冲突 |
| `E_RENDER` | 8 | 未预期的渲染失败 |

## 测试与实际验证

运行全部自动化测试：

```bash
python -m unittest discover -s tests -v
```

本机实际环境：Python 3.14.4、NumPy 2.5.5、Pillow 13.0.0。7 个测试全部通过，覆盖完整样本、字节级可重复性、近裁剪、Hi-Z、调试叠加和错误码。

内置样本实际渲染命令：

```bash
python render.py --scene samples/character_scene.json --output artifacts/acceptance --stats artifacts/acceptance/stats.json
```

实测结果：

- 分辨率：512×512，RGB PNG，可重复 SHA-256：`501065327a5d52e25923a7d99cf87f6512d5414e31e320d71c44fb97cb21f9e8`
- 输入场景 SHA-256：`7ba05abb86d8dd74da9e8ccf455fa4aaf3aa5fca14fc8ad538a9c1642684df28`
- 三角形总数：7,504
- 背面剔除：3,707
- Hi-Z 完全遮挡剔除：948
- 总剔除数：4,655
- 深度预通道提交/实际写入：3,413 / 2,630
- 着色像素：72,425；着色片元：74,131
- 透明片元：1,749；透明绘制段：1
- 方向光 `warm-key` 阴影图：512×512×1
- 点光 `cool-fill` 阴影图：256×256×6

连续两次独立渲染的 `render.png` 和统计 JSON 均字节一致；统计 JSON 不含时间戳或绝对输出路径。交付样本位于 `artifacts/acceptance/`，第二次复核结果位于 `artifacts/repro_check/`。

错误路径实测记录见 `artifacts/error_checks/report.txt`：缺失资源退出码 2、缺失字段 4、非法数值 5、输出路径冲突 7，均同时输出上述统一 JSON。
