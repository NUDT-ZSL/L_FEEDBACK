# Character Preview

一个面向游戏角色预览的确定性离线渲染工具。项目只使用 NumPy、PyOpenGL 和 Pillow，在
Windows 上通过隐藏窗口 WGL 创建 OpenGL 3.3 Core 上下文；本仓库的 Linux 验收则使用
EGL surfaceless。渲染结果包含 PNG 帧序列、逐帧 JSON 元数据和总清单，不依赖模型、贴图或
网络资源。

## 已实现功能

- 确定性生成器输出 12 个网格节点、6 个 PBR 材质和 5 个点光源。
- 自描述 JSON 模型校验；所有格式错误返回稳定错误码。
- FBO 颜色纹理 + 深度纹理离屏渲染，`glReadPixels` 读取后写 PNG。
- 方向光阴影贴图，3x3 PCF，可配置常量偏移、斜率偏移和世界空间法线偏移。
- 每个点光源六面立方体深度阴影，最多同时启用 8 个点光源。
- 点光选择按 `intensity / distance(target)^2` 降序，使用名称和源索引稳定破同分。
- GGX 法线分布、Smith 几何项、Schlick 菲涅尔、金属度/粗糙度能量分配。
- 线性空间光照、曝光、Reinhard tone mapping、gamma 2.2 输出。
- 初始化、渲染、初始化失败和重复初始化路径均有显式 GL 资源回收与错误检查。
- 配置缺失、类型错误或越界时回退默认值并输出警告。
- 多帧相机环绕动画；逐帧输出图片、相机矩阵、光源选择状态与阴影参数。
- 某一帧失败时立即停止，不删除已完成帧，并写入失败报告。

## Windows 安装

要求 Windows 10/11、Python 3.10 或 3.11，以及支持 OpenGL 3.3+ 的显卡驱动或软件渲染器。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`PyOpenGL_accelerate` 是可选加速包：

```powershell
python -m pip install PyOpenGL_accelerate
```

上下文层不依赖 GLFW、Docker 或窗口系统：Windows 通过 `ctypes` 创建隐藏 HWND，选择
像素格式，使用临时 WGL 上下文取得 `wglCreateContextAttribsARB`，再创建 3.3 Core
上下文。若驱动不支持 3.3，工具以 `E_CONTEXT_WGL` 或 `E_GL_VERSION` 退出。

## 运行

在仓库根目录执行：

```powershell
# 1. 确定性生成样本模型
python tools/generate_character.py --output samples/character.json

# 2. 仅校验模型和配置
python -m character_preview validate `
  --model samples/character.json `
  --config configs/sample_render.json

# 3. 渲染 4 帧
python -m character_preview render `
  --model samples/character.json `
  --config configs/sample_render.json `
  --output output/sample
```

也可以覆盖帧数和时间步长：

```powershell
python -m character_preview render `
  --model samples/character.json `
  --config configs/sample_render.json `
  --frames 8 `
  --time-step 0.125 `
  --output output/animation
```

输出结构：

```text
output/sample/
  frame_0000.png
  frame_0000.json
  frame_0001.png
  frame_0001.json
  ...
  manifest.json          # 全部成功时生成
  render_report.json     # 仅失败时生成，保留成功帧列表
```

## 模型格式

样本见 `samples/character.json`。矩阵为 16 个列主序浮点数，坐标右手系、`+Y` 向上、
`-Z` 向前。材质颜色和光源颜色均声明为线性空间。

```json
{
  "schema_version": 1,
  "units": "meters",
  "color_space": "linear",
  "materials": [
    {
      "name": "suit_blue",
      "diffuse": [0.035, 0.12, 0.34],
      "specular": [0.85, 0.9, 1.0],
      "roughness": 0.32,
      "metallic": 0.18
    }
  ],
  "nodes": [
    {
      "name": "torso",
      "material": "suit_blue",
      "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1.39, 0, 1],
      "mesh": {
        "positions": [[-0.31, -0.34, -0.18], [0.31, -0.34, -0.18], [0, 0.34, -0.18]],
        "normals": [[0, 0, -1], [0, 0, -1], [0, 0, -1]],
        "indices": [0, 1, 2]
      }
    }
  ],
  "scene": {
    "ambient": [0.035, 0.045, 0.07],
    "directional": {
      "name": "key_sun",
      "direction": [-0.45, -1.0, -0.35],
      "color": [1.0, 0.94, 0.84],
      "intensity": 3.0,
      "cast_shadow": true
    },
    "point_lights": [
      {
        "name": "warm_key_left",
        "position": [-1.8, 2.55, 2.0],
        "color": [1.0, 0.56, 0.28],
        "intensity": 24.0,
        "range": 9.0,
        "cast_shadow": true
      }
    ],
    "camera": {
      "position": [3.1, 2.05, 4.15],
      "target": [0.0, 1.08, 0.0],
      "up": [0.0, 1.0, 0.0],
      "fov_deg": 38.0
    }
  }
}
```

校验要求：

- `schema_version` 必须为 `1`，单位必须为米，颜色空间必须为 linear。
- 至少 3 个节点；节点名和材质名必须唯一。
- 每个节点必须引用已存在材质，矩阵必须有限、仿射且非奇异。
- 法线必须与顶点数量一致且为单位长度；索引必须是有效三角形整数索引。
- `roughness` 和 `metallic` 必须在 `[0, 1]`；颜色和强度不得为负。

## 渲染配置

完整样例见 `configs/sample_render.json`。

| 字段 | 默认值 | 有效范围 | 作用 |
|---|---:|---:|---|
| `width` / `height` | 640 | 64–4096 | 输出分辨率 |
| `shadow_map_size` | 1024 | 256–4096 | 方向光阴影贴图边长 |
| `point_shadow_size` | 256 | 64–2048 | 每个点光立方体面的边长 |
| `max_point_lights` | 4 | 0–8 | 同时生效的点光源上限 |
| `directional_light_scale` | 1.0 | 0–100 | 方向光强度倍率 |
| `point_light_scale` | 1.0 | 0–100 | 点光强度倍率 |
| `exposure` | 0.8 | 0.05–16 | 线性 HDR 曝光 |
| `shadow_bias` | 0.0015 | 0–0.05 | 方向光阴影深度偏移 |
| `shadow_normal_bias` | 0.025 | 0–0.25 | 方向光法线偏移，世界单位 |
| `point_shadow_bias` | 0.012 | 0–0.2 | 点光阴影深度偏移，世界单位 |
| `point_shadow_normal_bias` | 0.025 | 0–0.25 | 点光法线偏移，世界单位 |
| `frame_count` | 3 | 1–10000 | 动画帧数 |
| `time_step` | 0.25 | 0–120 | 相邻帧秒数 |
| `camera_orbit_speed_deg` | 12.0 | -360–360 | 相机绕目标每秒旋转角度 |
| `output_dir` | `output` | 非空字符串 | 默认输出目录 |

缺失字段、错误类型和越界值都不会导致崩溃：对应字段回退默认值，并在 stderr 输出
`[WARN]` 信息。JSON 本身损坏时返回 `E_CONFIG_JSON`。配置不会静默截断到边界值。

## 渲染实现

### 阴影

方向光使用正交视锥覆盖模型世界包围盒，深度纹理附加到专用 FBO，主着色器执行 3x3 PCF。
主着色器先沿法线把采样位置推出 `shadow_normal_bias`，再通过斜率相关深度偏移抵抗痤疮。
阴影 pass 还启用了固定的 polygon offset，最终深度偏移由配置进一步调节。

每个被选中的点光源有独立的 `GL_DEPTH_COMPONENT24` 立方体贴图。渲染时依次把六个面附加到
同一个 depth-only FBO，使用 90° 透视投影和光源 `range` 作为远平面。主着色器按 GL 立方体
面规则采样，并把光源距离换算为透视深度后与阴影深度比较；点光偏移同时使用世界空间法线
偏移和深度导数补偿。

点光源选择策略固定为：

```text
score = intensity / max(distance(light.position, camera.target)^2, 1e-6)
按 score 降序，score 相同则按 name，再按源 JSON 索引
```

该策略兼顾亮度和距角色距离，完全确定。未选中的点光源名称会写入逐帧元数据
`unselected_point_lights`。

### PBR 与颜色

直接光使用 GGX + Smith + Schlick：

```text
F0       = mix(0.04 * specular, diffuse * specular, metallic)
diffuse  = (1 - F) * (1 - metallic) * albedo / PI
specular = D(GGX) * G(Smith) * F / (4 * NdotV * NdotL)
```

环境光使用半球近似，并用粗糙度和 Fresnel 调制环境高光。点光采用距离平方衰减和
`range` 窗函数。所有材质、环境光和直接光计算都在线性空间；最后执行曝光、Reinhard
tone mapping 和 gamma 2.2。

### 状态与资源

`CharacterRenderer.initialize()` 和 `close()` 都是幂等的。初始化前后、每帧前后都会保存并
恢复：

- 视口、深度测试、深度函数、深度写掩码。
- 面剔除开关、剔除面、正面方向、blend/scissor/stencil、sRGB 和 rasterizer discard。
- 当前/读/写 FBO、program、VAO、活动纹理单元及 0–8 号单元纹理绑定。
- color mask、深度范围、清屏颜色/深度、polygon offset、pack alignment。

纹理、Buffer、VAO、FBO 和 Program 都由类内资源表登记，正常结束按逆序显式释放；初始化
异常时走同一清理路径。关键阶段调用 `check_gl_error`，GL 失败会报告 `E_GL` 和十六进制
错误码（例如 `0x0502 GL_INVALID_OPERATION`）。

### 元数据

每帧 JSON 包含：

- `frame_index`、`time_seconds`、图片文件名和 PNG SHA-256。
- 相机位置、目标、FOV、投影矩阵、视图矩阵、视图投影矩阵（列主序）。
- 方向光、被选中点光、未选中点光、选择策略和数量上限。
- 阴影尺寸、偏移、PCF 规则和颜色管线说明。
- 模型节点数、材质数、单位与颜色空间。

失败时 `render_report.json` 包含错误码、消息、已完成帧数量和已完成帧列表。CLI 同时把错误
码写到 stderr，退出码为 `2`。

## 错误码

| 错误码 | 含义 |
|---|---|
| `E_MODEL_READ` / `E_MODEL_JSON` | 模型不可读 / JSON 语法错误 |
| `E_MODEL_SCHEMA` / `E_MODEL_VERSION` | 顶层结构错误 / schema 版本不支持 |
| `E_MODEL_NODE_COUNT` / `E_MODEL_NAME` | 节点不足 3 个 / 节点名重复或为空 |
| `E_MODEL_TRANSFORM` | 矩阵尺寸、有限性、仿射性或奇异性错误 |
| `E_MODEL_GEOMETRY` | 顶点、法线、索引错误 |
| `E_MODEL_MATERIAL` / `E_MODEL_MATERIAL_REF` | 材质参数错误 / 材质引用不存在 |
| `E_MODEL_LIGHT` | 光源字段或数值错误 |
| `E_CONFIG_READ` / `E_CONFIG_JSON` / `E_CONFIG_SCHEMA` | 配置文件错误 |
| `E_CONTEXT_LOAD` / `E_CONTEXT_EGL` / `E_CONTEXT_WGL` | 上下文创建失败 |
| `E_GL_VERSION` | OpenGL 版本低于 3.3 |
| `E_GL` | 运行时 OpenGL 错误，附 GL 错误码 |
| `E_FRAMEBUFFER` / `E_SHADER_COMPILE` / `E_SHADER_LINK` | FBO 或着色器错误 |
| `E_RENDER_INIT` / `E_RENDER_ARGUMENT` | 渲染器未初始化 / 参数错误 |
| `E_OUTPUT` / `E_RENDER_UNEXPECTED` | 输出失败 / 未分类运行失败 |

## 测试

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

测试包括模型校验错误码、确定性排序、矩阵往返、配置回退警告、真实 OpenGL 双帧序列、
PNG/元数据哈希一致性、阴影只减光不加光、状态恢复、重复初始化/关闭，以及第二帧注入故障
后第一帧输出仍保留。

本机实际验收结果见 `artifacts/verification_report.md`，样本帧位于
`artifacts/verification/sample_sequence/`。

## 已知边界

- 点光源上限固定为 8，因为 OpenGL 3.3 没有立方体贴图数组；主着色器用固定 8 个
  `samplerCubeShadow` 分支。配置允许 0–8。
- 当前动画只驱动相机环绕；网格节点没有骨骼或关键帧动画。
- 阴影质量受贴图尺寸和光源 `range` 影响，场景尺度应使用米并能被方向光正交包围盒覆盖。
- Windows WGL 路径按 OpenGL 3.3 Core 规范实现，但本次执行环境为 WSL/Linux，未在真实
  Windows GPU 驱动上进行二进制验证；Linux EGL 路径已完整实跑。
