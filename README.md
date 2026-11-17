# NumPy 离线分层材质软光栅渲染器

这是一个完全由 CPU 和 NumPy 计算的离线渲染器。它从 JSON 读取透视相机、
多个点光源/方向光源、环境光、三角网格和多层材质，输出线性 RGB 的
32 位浮点 HDR 图像。渲染过程不打开窗口、不依赖 GPU、不读取模型或纹理资产。

## 环境

- Python >= 3.10
- NumPy >= 1.24

```bash
python -m pip install "numpy>=1.24"
# 可选：安装 softraster 命令
python -m pip install -e .
```

## 快速开始

```bash
# 确定性生成样本场景
python examples/generate_sample.py

# 线性 HDR，连续渲染两次并逐像素自检
python renderer.py examples/studio_scene.json \
  -o artifacts/sample_hdr.npy --self-check

# 仅色调映射
python renderer.py examples/studio_scene.json \
  -o artifacts/tone_mapped.npy --tone-map

# 仅 sRGB 伽马，不隐式执行色调映射
python renderer.py examples/studio_scene.json \
  -o artifacts/gamma_only.npy --gamma

# 两个后处理步骤同时启用
python renderer.py examples/studio_scene.json \
  -o artifacts/display.npy --tone-map aces --gamma

# 也可以作为模块运行
python -m softraster examples/studio_scene.json -o artifacts/out.npy
```

样本场景包含 3 个点光源、1 个方向光源、环境光、6 个网格和多层材质。
其中一个网格完全位于相机视锥外，但仍参与所有光源的深度图，因此它的阴影
可以落到可见表面上。

## 命令行

```text
python renderer.py SCENE.json -o OUTPUT [--tone-map [aces|reinhard]]
                                   [--gamma] [--self-check]
```

- `SCENE.json`：输入场景路径。
- `-o/--output`：输出路径，支持 `.npy` 和 `.pfm`。
- 不传后处理开关时输出线性浮点 HDR。
- `--tone-map`：启用色调映射；省略参数时使用 `aces`，也可选 `reinhard`。
- `--gamma`：只启用 sRGB 传输函数。它可以单独使用，不会触发色调映射。
- `--self-check`：使用同样参数连续渲染两次，并逐像素比较最终 `float32`
  图像。通过时退出码为 `0`；差异非零时在标准输出打印 JSON 报告并以非零
  退出码终止。

输出数组形状始终为 `H x W x 3`，dtype 为 `float32`。`.npy` 保留数值；
`.pfm` 使用 32 位小端浮点和标准自底向上的行顺序。

## JSON 场景格式

完整示例见 `examples/studio_scene.json`。根节点可以是场景对象本身，也可以
使用下例中的 `"scene"` 包装；渲染尺寸既可以放在根节点 `"render"`，也可以
放在包装内的 `"scene"` 中。

```json
{
  "scene": {
    "camera": {
      "position": [0, 2.25, 7.5],
      "target": [0, 1.2, 0],
      "up": [0, 1, 0],
      "fov_y_deg": 44,
      "near": 0.15,
      "far": 60
    },
    "ambient": {"color": [0.035, 0.045, 0.065], "intensity": 1.0},
    "background": [0.006, 0.008, 0.012],
    "lights": [
      {
        "name": "key",
        "type": "point",
        "position": [4.2, 4.8, 3.2],
        "color": [1.0, 0.76, 0.5],
        "intensity": 72,
        "range": 25,
        "shadow": {
          "enabled": true,
          "map_size": 128,
          "radius": 2.6,
          "samples": 12,
          "bias": 0.003,
          "normal_bias": 0.025
        }
      },
      {
        "name": "sun",
        "type": "directional",
        "direction": [-0.32, -1.0, -0.22],
        "color": [1, 0.93, 0.82],
        "intensity": 2.1,
        "shadow": {"enabled": true, "map_size": 256, "radius": 3.2}
      }
    ],
    "materials": [
      {
        "name": "coated_floor",
        "layers": [
          {
            "base_color": [0.18, 0.21, 0.26],
            "metallic": 0.0,
            "roughness": 0.62,
            "clearcoat": 0.0,
            "normal_strength": 0.7,
            "normal": [0.22, 0.08, 1.0],
            "opacity": 1.0
          },
          {
            "base_color": [0.08, 0.1, 0.13],
            "metallic": 0.0,
            "roughness": 0.1,
            "clearcoat": 1.0,
            "normal_strength": 0.25,
            "normal": [0.02, -0.03, 1.0],
            "opacity": 0.24
          }
        ]
      }
    ],
    "meshes": [
      {
        "name": "floor",
        "vertices": [[-7, 0, -6], [-7, 0, 5], [7, 0, 5], [7, 0, -6]],
        "normals": [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
        "triangles": [[0, 1, 2], [0, 2, 3]],
        "material": 0,
        "double_sided": false
      }
    ]
  },
  "render": {
    "width": 320,
    "height": 240,
    "exposure": 1.0,
    "max_layers": 8
  }
}
```

注意：实际文件必须包含至少 3 个点光源和 1 个方向光源；上面的片段只展示
了两个光源的字段结构。

### 几何与材质约束

- 顶点坐标位于世界空间，三角面按从外部观察的逆时针方向定义；法线朝向
  相机时通过背面剔除。`double_sided: true` 可关闭该网格的背面剔除。
- `triangles` 可以是三元组数组，也可以是与 `indices` 等价的扁平索引数组。
- 每个网格通过 `material` 整数引用 `materials`；网格顶点索引和材质索引越界
  会触发 `RENDER_ERR_INDEX`。
- `normals` 可省略，此时按面积加权计算逐顶点法线。
- 材质层按数组顺序从底层到顶层合成。每层都可独立配置 `base_color`、
  `metallic`、`roughness`、`clearcoat`、`normal_strength`、切空间法线扰动
  `normal` 和 `opacity`。
- 默认最多 8 层，可由 `render.max_layers` 在 `[1, 32]` 内调整；超限触发
  `RENDER_ERR_LAYER_LIMIT`。
- 点光源的 `range` 是有限作用范围；方向光源的 `direction` 表示光线传播
  方向，着色时使用它的反向量作为入射方向。
- 阴影 `radius` 的单位是深度图像素，`samples` 是 PCF 采样数。

## 实现说明

### 光栅化

1. 每个网格先用包围球与相机视锥做整体判定，完全不可见的网格不进入主相机
   光栅化；阴影阶段仍使用全部网格。
2. 三角形在视图空间对近平面和远平面依次执行 Sutherland-Hodgman 裁剪。
   裁剪交点同时插值世界坐标与法线，再以裁剪后的多边形顶点扇形三角化。
3. 投影后使用像素中心、边函数和透视正确重心坐标做屏幕空间覆盖测试。
   退化面、投影面积为零的面和裁剪后退化的子三角形直接跳过。
4. 主相机写深度、网格编号、世界坐标和插值法线；深度测试只接受更近的
   fragment，因此重叠面的结果确定。

### 光源与软阴影

- 方向光源使用正交深度图；点光源使用包含 6 个面的透视立方体深度图。
- 深度图始终包含所有网格，包括被主相机视锥剔除的网格，且不按主相机背面
  剔除，以保证开放或薄网格也能投射阴影。
- 每个光源单独采样自己的深度图。PCF 使用固定黄金角圆盘核、双线性深度采样、
  法线偏移和斜率相关深度偏移。`radius` 控制核半径。
- 对每个光源只计算一次可见度；漫反射、普通高光和清漆高光都共享该值，多个
  光源之间直接求和，不会把同一阴影因子重复乘入。

### 分层材质

- 每层基于同一几何法线和该层自己的切空间扰动生成法线。上层法线计算函数
  不接收下层法线，因此底层扰动不会反向传播到清漆层。
- 每层使用金属度/粗糙度 GGX 微表面 BRDF；清漆使用独立低粗糙度电介质高光
  瓣，并按该层 `clearcoat` 权重加入该层反射。
- 合成从背景开始，按层数组顺序执行
  `result = result * (1 - opacity) + layer_radiance * opacity`。因此高层的
  清漆反射覆盖在下层结果之上。

### 后处理与输出

- 渲染、曝光和 HDR 输出都在线性 RGB 中进行。
- ACES/Reinhard 色调映射与 sRGB 传输函数分别由两个独立开关控制。只有显式
  传入的步骤会执行。
- 输出先写同目录临时文件，完成后原子替换目标路径。解析、校验或渲染失败时
  不会写出部分图像。

## 错误码

业务错误以 JSON 写到标准错误，并返回非零整数：

```json
{"code":"RENDER_ERR_SCHEMA","message":"missing required field 'camera'","details":{"source":"..."}}
```

| 错误码 | 退出码 | 触发条件 |
| --- | ---: | --- |
| `RENDER_ERR_SCHEMA` | 2 | JSON 无法解析、必要字段缺失或字段类型/范围非法 |
| `RENDER_ERR_INDEX` | 3 | 网格顶点索引或材质索引越界 |
| `RENDER_ERR_LAYER_LIMIT` | 4 | 材质层数超过限制 |
| `RENDER_ERR_IO` | 5 | 输出格式不支持或写出失败 |
| `RENDER_ERR_DETERMINISM` | 6 | 自检发现两次输出不完全相同 |

## 实测验证

测试环境：Python `3.14.4`，NumPy `2.5.5`，Ubuntu Linux，无 GPU。

```text
$ python -m unittest discover -s tests -v
test_business_errors_and_no_partial_output ... ok
test_frustum_culling_and_offscreen_shadow_caster ... ok
test_hdr_output_and_two_render_exact_self_check ... ok
test_layer_normals_do_not_feed_into_upper_layers ... ok
test_near_plane_clip_interpolates_only_clipped_vertices ... ok
test_pcf_radius_changes_soft_visibility ... ok
test_tonemap_and_gamma_switches_are_independent ... ok

Ran 7 tests in 11.999s
OK
```

样本输出自检：

```text
$ python renderer.py examples/studio_scene.json \
    -o artifacts/sample_hdr.npy --self-check
{"deterministic":true,"different_pixels":0,"max_abs_diff":0.0}
```

实际样本图像为 `240 x 320 x 3`、`float32`，线性 HDR 数值范围
`0.006000000052154064` 到 `294.0885009765625`。启用 ACES 与 sRGB 后的
数值范围为 `0.024320214986801147` 到 `1.0`。预览文件为
`artifacts/sample_preview.png`，HDR 和显示空间结果分别保存在
`artifacts/sample_hdr.npy` 与 `artifacts/sample_aces_gamma.npy`。

测试还确认：

- 三类错误输入（含材质索引和顶点索引两种越界）分别返回
  `RENDER_ERR_SCHEMA`/`2`、`RENDER_ERR_INDEX`/`3`、
  `RENDER_ERR_LAYER_LIMIT`/`4`，且目标文件不存在。
- 样本中视锥外的阴影投射网格可见性为 `false`，主相机不写它的像素；移除
  该网格后，方向光与一个点光源对可见表面的可见度结果发生改变。
- 近平面裁剪碎片深度均不小于 `near`，全部属性有限；完全位于近平面后和
  退化的三角形不产生 fragment。
- PCF `radius=0` 与 `radius=4` 对同一接收面产生不同可见度，宽核包含部分
  可见值。
- 仅色调映射、仅 gamma、两者同时启用分别与对应数学组合一致，互不隐式触发。
