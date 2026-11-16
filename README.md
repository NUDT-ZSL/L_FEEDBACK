# Deferred Validation

基于 Python 与 OpenGL 4.5 Core 的离线延迟渲染验证工具。场景、相机、三角网格、
材质和光源全部来自本地 JSON；渲染通过 EGL surfaceless 上下文和 FBO 多通道完成，
输出 G-buffer、直接光、间接光、最终 PNG 及 JSON 通道一致性报告。

## 功能

- OpenGL 4.5 Core，EGL surfaceless 离屏上下文；无窗口、无需 X server。
- 层级场景图、透视相机、三角形网格、切线空间法线贴图。
- G-buffer：世界位置/深度、世界法线、线性反照率、金属度、粗糙度。
- 方向光、点光、聚光灯；Cook-Torrance GGX/Smith/Schlick BRDF。
- 方向光和聚光灯使用 2D 阴影贴图；点光使用立方体阴影贴图。
- 可配置阴影分辨率、深度偏移、法线偏移、PCF 半径与 tap 数。
- 可配置 SSAA（1x–4x）；GL 资源在分辨率/SSAA/场景变化时显式重建。
- ACES 或 Reinhard HDR 色调映射，支持曝光参数。
- 每个 G-buffer 通道、直接光、间接光、最终图像均输出 PNG。
- 报告包含像素均值/方差/最小/最大、通道差异、覆盖率及几何/材质/光照/渲染校验和。
- `verify` 模式重新渲染并与参考报告和参考 PNG 比较，区分实质差异与浮点噪声。

## 环境与安装

要求 Python 3.9+、NumPy、PyOpenGL、PyGLM、Pillow，以及可创建 OpenGL 4.5
Core 上下文的 Mesa/EGL。软件渲染后备使用 Mesa `llvmpipe`。

```bash
# 二选一
python3 -m venv .venv
. .venv/bin/activate
pip install -e .

# 或使用 uv
uv venv --python python3 .venv
uv pip install --python .venv/bin/python -e .
```

Linux/Mesa 软件后备示例：

```bash
sudo apt install libegl1-mesa libgl1-mesa-dri
EGL_PLATFORM=surfaceless MESA_LOADER_DRIVER_OVERRIDE=llvmpipe \
  .venv/bin/deferred-validation render \
  --scene examples/default_scene.json -o artifacts/baseline
```

程序自身已默认设置 `PYOPENGL_PLATFORM=egl` 和 `EGL_PLATFORM=surfaceless`。无论使用
硬件驱动还是 llvmpipe，启动后都会校验实际上下文版本不低于 4.5。

## 快速开始

```bash
# 1. 生成内置多材质、多光源示例场景（不依赖外部资源）
.venv/bin/deferred-validation generate -o examples/default_scene.json

# 2. 离线渲染
.venv/bin/deferred-validation render \
  --scene examples/default_scene.json \
  -o artifacts/baseline

# 3. 同场景复核：退出码 0
.venv/bin/deferred-validation verify \
  --scene examples/default_scene.json \
  --report artifacts/baseline/report.json \
  -o artifacts/verify_same

# 4. 修改光照后再复核：退出码 3，并在 stderr 给出具体原因
python3 - <<'PY'
import json
s = json.load(open("examples/default_scene.json"))
s["lights"][0]["intensity"] *= 1.2
json.dump(s, open("/tmp/changed_scene.json", "w"))
PY
.venv/bin/deferred-validation verify \
  --scene /tmp/changed_scene.json \
  --report artifacts/baseline/report.json \
  -o artifacts/verify_changed
```

示例场景生成脚本也可直接执行：

```bash
.venv/bin/python examples/generate_scene.py -o examples/default_scene.json
```

## 输出

`render` 在输出目录写入：

```text
final.png          ACES/Reinhard + gamma 的最终图像
direct.png         直接光照通道的色调映射预览
indirect.png       间接光照通道的色调映射预览
g_position.png     世界位置，按场景包围盒归一化后可视化
g_normal.png       世界法线，[-1,1] 映射到 [0,1]
g_albedo.png       线性反照率转 sRGB
g_metallic.png     金属度
g_roughness.png    粗糙度
g_depth.png        相机距离按 near/far 归一化
report.json        通道布局、统计、差异、校验和及报告自检结果
```

PNG 是可视化产物；`report.json` 中的直接/间接/合成统计来自浮点 HDR 通道。报告
`gbuffer_layout` 与渲染器共用 `deferred_validation/gbuffer.py` 中的同一份附件定义。

## JSON 场景格式

完整示例见 `examples/default_scene.json`。最小结构如下：

```json
{
  "version": 1,
  "camera": {
    "position": [4.6, 3.25, 6.8],
    "target": [0.0, 0.85, 0.0],
    "up": [0.0, 1.0, 0.0],
    "fov_y_deg": 42.0,
    "near": 0.05,
    "far": 50.0
  },
  "rendering": {
    "width": 400,
    "height": 300,
    "ssaa": 2,
    "exposure": 1.05,
    "tone_mapper": "aces",
    "shadows": {
      "enabled": true,
      "directional_resolution": 512,
      "spot_resolution": 512,
      "point_resolution": 256,
      "bias": 0.0015,
      "normal_bias": 0.004,
      "filter_radius": 1,
      "filter_taps": 3
    }
  },
  "indirect": {
    "enabled": true,
    "probe": {"type": "hemisphere", "sky_color": [0.18, 0.26, 0.38],
              "ground_color": [0.045, 0.035, 0.03]},
    "intensity": 0.8
  },
  "meshes": {
    "cube": {
      "positions": [0,0,0, 1,0,0, 0,1,0],
      "normals": [0,0,1, 0,0,1, 0,0,1],
      "uvs": [0,0, 1,0, 0,1],
      "indices": [0,1,2]
    }
  },
  "materials": {
    "copper": {"albedo": [0.94, 0.40, 0.18],
               "metallic": 1.0, "roughness": 0.27}
  },
  "nodes": [
    {"name": "cube", "mesh": "cube", "material": "copper",
     "translation": [0,0,0], "rotation_deg": [0,0,0], "scale": [1,1,1]}
  ],
  "lights": [
    {"name": "key", "type": "directional",
     "direction": [0.38, -0.8, -0.46],
     "color": [1,0.91,0.78], "intensity": 3.2, "casts_shadow": true}
  ]
}
```

约定：

- `position` 中平移在旋转和缩放之前；节点矩阵为 `T * Rz * Ry * Rx * S`。
- 网格可用显式 `positions/normals/uvs/indices`，也可用
  `{"primitive": {"type": "plane" | "box" | "sphere", ...}}`。
- 光源 `direction` 表示光线传播方向，着色器使用 `-direction` 作为指向光源的方向。
- 点光/聚光必须提供 `position`；聚光要求 `0 <= inner_deg < outer_deg < 90`。
- `range` 用于点光/聚光的距离衰减窗口与阴影远平面。
- 法线贴图可内联为 `width/height/channels/data`，也可提供相对场景文件的 `file`。
- 非法字段会带 JSON 路径输出到 stderr，退出码为 `2`。

## 验证语义与退出码

`verify` 会先重新渲染当前场景，再与参考 `report.json` 比较：

1. 比较场景/几何/材质/光照/渲染设置校验和。
2. 比较 G-buffer 布局和尺寸/SSAA/色调映射设置。
3. 读取参考报告目录和当前输出目录中的 PNG，逐像素比较 RGB。
4. 任一校验和或布局变化视为实质差异；像素差超过阈值且显著像素比例超过
   `0.0005` 也视为实质差异。
5. 只有低于阈值的图像变化且校验和一致时归类为 `floating_point_noise`。

默认像素阈值为归一化 sRGB 空间 `0.01`，可用 `--threshold` 调整。汇总 JSON 可写到
`--json-out`。

退出码：

| 代码 | 含义 |
|---:|---|
| 0 | 成功；验证为一致或仅有浮点噪声 |
| 2 | 场景/报告缺失、JSON 或字段非法 |
| 3 | 验证发现实质差异 |
| 4 | 渲染、FBO、着色器或输出失败 |
| 130 | 用户中断 |

## 测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```

测试覆盖配置错误路径、完整多通道渲染、报告统计、SSAA/色调映射状态切换、
同场景验证、参数变更退出码 3、亚阈值图像变化归类为浮点噪声。

更多实现细节见 [docs/TECHNICAL.md](docs/TECHNICAL.md)，本机实测记录见
[VALIDATION.md](VALIDATION.md)。
