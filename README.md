# Fog City Preview

离线、确定性的 OpenGL 4.1 关键帧预览渲染器，用于验证雾天早晨城市关卡的体积光、建筑遮挡和半透明玻璃表现。场景、摄像机关键帧和几何均由程序确定性生成，不读取贴图或其他美术资源。

## 功能

- 从 JSON 加载摄像机关键帧、平行光、雾、凸多面体建筑和玻璃面板。
- 每帧执行 24 位深度 + `R32UI` 建筑 ID 的相机预通道，以及用于体积光遮挡的光空间 ID/深度预通道。
- 在独立的全屏通道中完成屏幕空间雾散射积分。
- 体积光先与不透明底色合成，再按视深从远到近、以源 alpha 混合玻璃面板。
- 每帧输出 PNG、逐帧统计 JSON，并输出聚合 `stats.json`。
- 提供只读校验模式，可重新渲染并逐字节比较 PNG、逐字段比较统计 JSON。
- 支持 1–32 个并行帧上限；上下文数量不超过该上限，超出的帧排队。

## 环境

- Windows 10/11
- Python 3.11+
- OpenGL 4.1 core 或更高
- `numpy`、`moderngl`、`PyGLM`、`Pillow`

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

也可以安装本项目并提供 `fogcity` 命令：

```powershell
python -m pip install -e .
```

Windows 上默认使用 moderngl 的本地 WGL standalone context。若需要明确指定：

```powershell
python -m fogcity render --scene scenes\sample_city.json --out out --backend wgl
```

Linux/无显示环境可使用 `--backend egl`；`auto` 在 Linux 上默认选择 EGL，在 Windows 上使用默认 WGL 路径。

## 快速开始

生成确定性的 7 帧样本场景：

```powershell
python -m fogcity generate-sample --path scenes\sample_city.json
```

连续渲染两次：

```powershell
python -m fogcity render --scene scenes\sample_city.json --out out\run1
python -m fogcity render --scene scenes\sample_city.json --out out\run2
```

只读校验已有输出（不写任何文件）：

```powershell
python -m fogcity render --scene scenes\sample_city.json --out out\run1 --validate-only
```

并行帧上限为 2：

```powershell
python -m fogcity render --scene scenes\sample_city.json --out out\parallel --parallel-frames 2
```

输出目录结构：

```text
out\run1\
  frame_0000.png
  frame_0000.json
  frame_0001.png
  frame_0001.json
  ...
  stats.json
```

`stats.json` 只包含一个可变化的时间戳字段 `generated_at`；帧统计中没有时间戳。验证模式会忽略该字段。

统计和错误 context 中的 `frame_index` 为 0 基编号；命令行进度文本使用 1 基编号。

## 场景格式

最小结构：

```json
{
  "schema_version": 1,
  "name": "scene-name",
  "render": {
    "width": 480,
    "height": 270,
    "fps": 1.0,
    "start_time": 0.0,
    "end_time": 6.0
  },
  "camera_keyframes": [
    {
      "time": 0.0,
      "position": [7.0, 2.25, 18.0],
      "orientation": [0.0, 0.0, 0.0, 1.0],
      "fov_y_deg": 50.0,
      "near": 0.1,
      "far": 90.0
    }
  ],
  "directional_light": {
    "direction": [-0.42, -0.62, 0.66],
    "color": [1.0, 0.82, 0.58],
    "intensity": 4.6
  },
  "fog": {
    "color": [0.52, 0.65, 0.74],
    "density": 0.018,
    "extinction": 1.25,
    "scattering": 0.95,
    "anisotropy": 0.62,
    "ambient": 0.50,
    "steps": 32
  },
  "shadow": {
    "resolution": 768,
    "bias": 0.0015
  },
  "buildings": [
    {
      "id": 1,
      "color": [0.34, 0.39, 0.43],
      "vertices": [[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1], [-1, 2, -1], [1, 2, -1], [1, 2, 1], [-1, 2, 1]],
      "faces": [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    }
  ],
  "glass_panels": [
    {
      "id": 101,
      "color": [0.23, 0.62, 0.68],
      "transmittance": 0.64,
      "vertices": [[-3, 0, 10], [3, 0, 10], [3, 3, 10], [-3, 3, 10]]
    }
  ]
}
```

约束与约定：

- `camera_keyframes[].orientation` 为 `[x, y, z, w]`，必须是可归一化的非零四元数。
- 关键帧时间必须严格递增；重复和倒退分别返回独立业务错误码。
- 建筑顶点/面必须构成闭合、凸、平面的多面体；面使用扇形三角化。
- 玻璃面板为凸平面多边形；顶点绕序决定面法线，法线背向摄像机的面板不参与混合。
- 平行光 `direction` 表示光线传播方向，加载时归一化。

完整说明见 [docs/TECHNICAL.md](docs/TECHNICAL.md)。

## 错误输出

所有预期业务错误都向 stderr 写一行 JSON，例如：

```json
{"error":{"code":"E_SCENE_NOT_FOUND","context":{"path":"missing.json"},"exit_code":2,"message":"scene JSON file does not exist"},"ok":false}
```

常见退出码：

| 退出码 | 错误码 | 含义 |
|---:|---|---|
| 2 | `E_SCENE_NOT_FOUND` | 场景文件不存在 |
| 3 | `E_SCENE_IO` | 场景读取失败 |
| 4 | `E_SCENE_JSON` | JSON 语法错误 |
| 10 | `E_SCENE_FIELD` | 字段类型、范围或几何非法 |
| 11 | `E_KEYFRAME_TIME_ORDER` | 关键帧时间倒退 |
| 12 | `E_KEYFRAME_TIME_DUPLICATE` | 关键帧时间重复 |
| 20 | `E_GL_CONTEXT` | 无法建立 OpenGL 4.1 上下文 |
| 21 | `E_GL_RESOURCE` | Shader/GL 资源初始化失败 |
| 22 | `E_FRAME_RENDER` | 单帧渲染失败，context 中含 `frame_index` |
| 30 | `E_OUTPUT_WRITE` | 输出写入失败 |
| 40 | `E_VERIFY_OUTPUT_MISSING` | 校验所需输出缺失 |
| 41 | `E_VERIFY_MISMATCH` | 校验结果不一致 |
| 99 | `E_INTERNAL` | 未预期内部错误 |

## 测试

```powershell
python -m unittest discover -s tests -v
```

本次实际验证结果见 [docs/VALIDATION.md](docs/VALIDATION.md)。

