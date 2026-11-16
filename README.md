# 本地离线条带渲染器

这是一个仅依赖 Python 3.10+、NumPy 和 Pillow 的确定性球体渲染器。它读取
JSON 场景和本地 PNG 贴图，输出四张 8 位分层 PNG 以及一份 `self_check.json`。
整个流程不使用 GPU、网络、容器或外部素材。

## 快速开始

```bash
python -m pip install -r requirements.txt

# 可选：重新生成确定性贴图和场景
python make_demo.py --out examples/demo --size 128 --texture-size 64

# 双线性采样、32 行分块
python render.py --scene examples/demo/scene_bilinear_32_row_forward.json

# 最近邻采样
python render.py --scene examples/demo/scene_nearest_32_row_forward.json

# 输出目录可临时覆盖
python render.py \
  --scene examples/demo/scene_bilinear_16_column_reverse.json \
  --output-dir /tmp/strip-render-column
```

Windows 下可将 `python` 换成 `py -3`。仓库内的 `examples/demo` 已包含生成好的
贴图、五个场景、五组实际输出和报告。

成功的命令返回整数退出码 `0`。场景字段错误返回 `2`；贴图、几何、分块或输出
环境错误返回 `3`。错误码同时写入 stderr 和可写入位置下的错误自检报告。

## 输出

每次成功渲染在场景 `output.directory`（或 `--output-dir`）下写入：

| 文件 | 内容 |
| --- | --- |
| `base_color.png` | 基础色层 |
| `direct_light.png` | 方向光直接照亮层 |
| `ambient_light.png` | 解析环境光层 |
| `composite.png` | 前三个浮点层按权重相加后的最终合成 |
| `self_check.json` | 命中数、NaN/Inf、法线回退、均值、合成误差等 |

所有 PNG 都是 RGB 8 位。量化公式为
`uint8(rint(clip(value, 0, 1) * 255))`，其中 `rint` 使用 IEEE 最近偶数舍入。

`self_check.json` 的主要字段：

- `render.hit_pixels`、`render.miss_pixels`、`render.pixel_count`
- `render.nan_count`、`render.inf_count` 及按通道拆分的计数
- `render.normal_fallback_count` 和缺失/零长度/无效结果的拆分数
- `render.channel_means` 与仅命中像素的 `render.channel_means_hit`
- `render.composite_check.max_abs_error`，检查阈值为 `1e-6`
- `outputs.channel_quantized_sha256`，量化后像素缓冲的 SHA-256
- `textures.*` 与当前过滤、分块和权重配置

## 场景格式

场景 JSON 采用严格 schema：不接受未知字段；缺失字段、字段类型错误、颜色越界、
非法枚举等都以 `E_SCENE_FIELD` 和退出码 `2` 拒绝。贴图路径和输出目录如果是
相对路径，都相对于场景 JSON 所在目录解析。

示例见
[`examples/demo/scene_bilinear_32_row_forward.json`](examples/demo/scene_bilinear_32_row_forward.json)。

```json
{
  "schema_version": 1,
  "camera": {
    "position": [0.0, 0.15, 3.8],
    "target": [0.0, 0.0, 0.0],
    "up": [0.0, 1.0, 0.0],
    "vertical_fov_degrees": 42.0,
    "near": 0.05,
    "far": 100.0
  },
  "light": {
    "direction": [-0.45, 0.70, 0.60],
    "color": [1.0, 0.96, 0.88],
    "intensity": 2.0
  },
  "geometry": {
    "type": "sphere",
    "center": [0.0, 0.0, 0.0],
    "radius": 1.0
  },
  "material": {
    "base_color_texture": "textures/base_color.png",
    "normal_texture": "textures/normal.png",
    "roughness_texture": "textures/roughness.png",
    "base_color_tint": [1.0, 1.0, 1.0],
    "specular_strength": 0.35,
    "normal_strength": 0.9
  },
  "environment": {
    "sky_color": [0.36, 0.46, 0.62],
    "ground_color": [0.10, 0.09, 0.08],
    "intensity": 0.9
  },
  "output": {
    "width": 128,
    "height": 128,
    "background_color": [0.015, 0.02, 0.03],
    "texture_filter": "bilinear",
    "block_size": 32,
    "block_axis": "row",
    "block_order": "forward",
    "directory": "outputs/bilinear_32_row_forward",
    "weights": {
      "base_color": 0.25,
      "direct": 0.55,
      "ambient": 0.2
    }
  }
}
```

约束：

- `light.direction` 是从表面指向光源的单位化方向输入，不能是零向量。
- `geometry.type` 当前只支持 `"sphere"`，半径必须大于 0。
- `normal_texture` 可省略或设为 `null`；路径指向不存在的文件时按缺失回退。
- 基础色和粗糙度贴图必须是可读取文件；三张贴图分辨率各自在 `2..4096`。
- `width`/`height` 在 `64..1024`，`block_size` 必须同时整除宽和高。
- `block_axis` 为 `"row"` 或 `"column"`，`block_order` 为 `"forward"` 或
  `"reverse"`。
- 三个合成权重均在 `[0,1]`，且和必须为 `1`。这保证所有未命中像素在合成层也
  仍然是背景色。

## 渲染与采样

相机采用右手透视投影，主光线来自像素中心。球体的包围盒先保守投影到屏幕空间，
用于限定光栅化分块范围；每个候选像素再与球体做解析求交，因此命中坐标、几何
法线、着色法线和 UV 都由解析结果产生，不依赖三角形插值误差。`RenderBuffers`
同时保留 `world_normal` 与法线贴图扰动后的 `shading_normal`。

球面 UV 定义为：

```text
u = 0.5 + atan2(n.x, n.z) / (2*pi)
v = acos(clamp(n.y, -1, 1)) / pi
```

最近邻采样取 `floor(u*W)`、`floor(v*H)` 并钳制索引。双线性采样在纹素中心之间
插值，使用 `u*W-0.5` 与 `v*H-0.5`，四个角索引都独立钳制到贴图边界，不会环绕。
基础色、粗糙度和法线贴图统一使用场景的 `texture_filter`。

PNG 的 8 位数值按线性 `[0,1]` 读取，不做 sRGB 转换；粗糙度使用转换后 RGB
图像的红色通道，单通道灰度 PNG 因此可直接使用。

TBN 以球面几何法线构造，切向 `+X` 对应 UV 的 `+U`，切向 `+Y` 对应 UV 的 `+V`。
极点处使用稳定的备用切线。法线贴图解码为 `2*rgb-1`，长度小于 `1e-2` 的近似
零向量按几何法线回退；所有分量先钳制到 `[-1,1]`，随后归一化，因此负值和超范围
输入不会产生 NaN/Inf。

直接光采用 Lambert 漫反射和 Blinn-Phong 简化高光：

```text
NdotL = max(dot(N, L), 0)
NdotH = max(dot(N, normalize(L+V)), 0)
shininess = clip(2 / max(roughness^4, 1e-6) - 2, 2, 2048)
diffuse = NdotL / pi
specular = specular_strength * (1-roughness)^2 * NdotH^shininess * NdotL
direct = clamp(albedo * light_color * intensity * (diffuse + specular), 0, 1)
```

环境光使用上下半球线性辐射亮度分布
`L(w)=0.5*(sky+ground)+0.5*(sky-ground)*(w dot Y)`。Lambert 半球积分给出：

```text
ambient = albedo * env_intensity *
          (0.5*(sky+ground) + (1/3)*(sky-ground)*(N dot Y))
```

该结果是解析闭式近似，不是逐像素采样。所有浮点层最后钳制到 `[0,1]`，合成层
严格按下式生成：

```text
composite = base_color*w_base + direct*w_direct + ambient*w_ambient
```

`self_check.json` 会在同一份浮点结果上重新计算并记录最大绝对误差。

## 一致性与输出事务

分块只决定着色 tile 的遍历顺序。所有通道先写入固定形状的完整浮点数组，合成层
和统计量在整幅图完成后按固定全图顺序计算，不做跨 tile 累加。因此行分块、列分块、
正序、逆序、整幅以及任意可整除分块尺寸得到相同像素值。

写出流程先探测目录可写性，再把四张 PNG 和报告写入同目录的临时 staging 目录。
提交时先备份同名旧文件，再逐个原子替换；任何提交异常都会回滚旧文件并清空
staging，不会留下半成品通道。成功提交会覆盖上一轮五个固定输出文件。

## 错误码

| 错误码 | 退出码 | 含义 |
| --- | ---: | --- |
| `E_SCENE_FIELD` | 2 | 缺失字段、类型错误、未知字段或字段值非法 |
| `E_SCENE_JSON` | 2 | 场景 JSON 语法错误 |
| `E_SCENE_IO` | 2 | 场景文件不可读 |
| `E_TEXTURE_SIZE` | 3 | 贴图尺寸不在 `2..4096` |
| `E_TEXTURE_READ` | 3 | 基础色/粗糙度贴图缺失或无法读取 |
| `E_CAMERA_INSIDE_SPHERE` | 3 | 相机位于球体内部 |
| `E_BLOCK_NOT_DIVISIBLE` | 3 | 分块尺寸不能同时整除宽和高 |
| `E_OUTPUT_NOT_WRITABLE` | 3 | 输出路径不可创建、不可写或提交失败 |
| `E_INTERNAL` | 3 | 未预期的内部异常 |

错误报告优先写入预期输出目录的 `self_check.json`。若该目录本身不可写，则尝试
写入场景目录旁的 `render_error_output/render_error_report.json`，再回退到当前
工作目录。

## 测试与验证

```bash
python -m unittest discover -s tests -v
```

测试覆盖：

- 最近邻/双线性定义、边缘钳制
- 四通道合成误差
- 行/列、正序/逆序及整幅分块逐像素一致性
- 缺失和零法线回退、极端法线有限性
- 最近偶数量化
- 成功 CLI、报告统计、重复运行覆盖
- 字段错误退出 2，四类环境/资产错误退出 3
- 注入提交失败后的旧输出回滚

实际验收数据见 [`VALIDATION.md`](VALIDATION.md)。
