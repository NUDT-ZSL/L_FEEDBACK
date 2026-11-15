# ModernGL 离线渲染校验器

一个无窗口的 OpenGL 4.1 Core 渲染校验器。它读取内置材质 JSON，编译三套
GLSL 程序，按种子生成并批量渲染立方体和球体，为每个实例输出
Color、Normal、Depth、ObjectID 四个 `GL_RGBA32F` 通道，并生成 CSV、状态快照和
JSON 摘要。

运行时不访问网络，不使用 Docker，也不创建可见窗口。程序在 Windows 上使用
WGL 离屏上下文；在 Linux 验证环境中使用 EGL。

## 快速开始（Windows）

要求 64 位 Python 3.10 或 3.11，以及支持 OpenGL 4.1 Core 的显卡驱动。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --no-deps -r requirements.txt

.\.venv\Scripts\python.exe -m render_check --instances 6 --seed 1337
# 等价入口：
.\.venv\Scripts\python.exe run_validation.py --instances 6 --seed 1337
```

也可以使用脚本：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1
scripts\run_windows.bat --instances 6 --seed 1337
```

### 精确版本依赖说明

PyPI 上 `moderngl==5.10.0` 的依赖元数据声明 `glcontext>=2.5.0`，与任务指定的
`glcontext==2.3.7` 冲突。直接执行
`pip install moderngl==5.10.0 numpy==1.26.4 glcontext==2.3.7` 会解析失败。
本项目按任务要求保留这两个精确版本，并将 ModernGL 用 `--no-deps` 安装；本机
OpenGL 4.5 Core 环境下该组合已通过完整渲染与测试。`scripts/install_windows.ps1`
采用同样的安装顺序。

## 命令行

```text
python -m render_check [options]

--instances N          实例数，1..64，默认 3
--seed S               确定性种子，默认 1337
--out DIR              CSV、摘要与 .npy 输出目录，默认 render_output
--materials FILE       材质 JSON；默认使用内置 assets/materials.json
--width PX             输出宽度，16..2048，默认 256
--height PX            输出高度，16..2048，默认 256
--backend auto|wgl|egl|glx|cgl
--memory-limit-mb MB   默认 512
--time-limit-sec SEC   默认 30
--fault-inject none|state_residue|diff
--debug                内部异常时打印堆栈
```

Windows 的 `auto` 后端为 `wgl`，会在无可见窗口的情况下创建 standalone 上下文。
如果驱动无法提供 OpenGL 4.1 Core，程序输出 `E_GL_003` 并以 3 退出。

## 内置样本

- `assets/materials.json`：PBR 金属、粗糙塑料、自发光三种材质。
- `assets/shaders/`：每种材质独立的顶点/片段 GLSL 程序。
- `assets/textures/`：由 `tools/generate_sample_assets.py` 确定性生成的 `.npy` 纹理。
- `render_check/geometry.py`：内置立方体与 UV 球体生成器。

批次中的实例按 `instance_index % material_count` 循环使用材质。几何在立方体与球体
之间交替，具体起点由种子决定。相机、光源、近远裁剪面和投影参数固定：

```text
camera eye       (0, 0, 6)
camera target    (0, 0, 0)
light position   (3, 4, 5)
vertical FOV     45 degrees
near / far       0.1 / 100.0
```

实例的旋转、缩放和微小位移只由 `--seed` 与实例索引产生，使用稳定的整数种子，
不依赖 Python 的随机哈希。

## 输出

成功运行后，`--out` 目录包含：

```text
instance_000_pbr_metal_color.npy
instance_000_pbr_metal_normal.npy
instance_000_pbr_metal_depth.npy
instance_000_pbr_metal_object_id.npy
...
render_log.csv
channel_map.json
state_snapshots.jsonl
summary.json
```

所有通道均为 `float32`、形状 `[height, width, 4]`，第一行对应图像顶部。Color、
Normal、Depth、ObjectID 使用四个 `GL_RGBA32F` 颜色附件；另外挂载内部深度纹理
做真实深度测试。

### 通道语义

- Color：PBR 光照结果，RGB 为线性颜色，A 为 1。
- Normal：世界空间单位法线映射到 `[0, 1]`，即 `normal * 0.5 + 0.5`，A 为 1。
- Depth：R 为归一化线性视深度 `view_depth / 100`，G 为 `gl_FragCoord.z`，
  B 为世界单位视深度，A 为 1。
- ObjectID：R 为 1-based 实例 ID，G 为材质索引，B 为几何 ID（0=立方体，
  1=球体），A 为 1。背景四个通道均为 0。

`render_log.csv` 的列为：

```text
instance_index,material_name,color_file,normal_file,depth_file,object_id_file,render_time_ms,state_hash
```

`channel_map.json` 维护实例、材质、ObjectID、四个通道文件和状态哈希的映射。
`summary.json` 包含成功/失败实例数、峰值内存、总耗时、上下文信息、驱动信息、
固定场景参数、通道映射和确定性检查结果。

## 材质 JSON 模式

每个材质必须包含 `name`、`vertex_shader`、`fragment_shader`、`texture` 和
`uniforms`。相对路径以 JSON 文件所在目录为基准。

```json
{
  "name": "pbr_metal",
  "vertex_shader": "shaders/pbr_metal.vert",
  "fragment_shader": "shaders/pbr_metal.frag",
  "texture": {
    "path": "textures/pbr_metal_albedo.npy",
    "format": "RGBA8",
    "unit": 0
  },
  "uniforms": {
    "base_color": [0.95, 0.78, 0.42],
    "metallic": 1.0,
    "roughness": 0.18,
    "emissive_strength": 0.0,
    "uv_scale": 2.0
  }
}
```

- 材质名必须非空且唯一。
- 着色器文件和纹理文件必须存在且是普通文件。
- `uniforms.base_color` 必须是 3 个 `[0,1]` 数值。
- `metallic`、`roughness` 必须在 `[0,1]`；`emissive_strength`、`uv_scale`
  必须非负。
- 纹理只允许声明 `RGBA8` 或 `RGBA32F`。源文件必须是 `HxWx4` 且分别为
  `uint8` 或 `float32`；程序比较实际 dtype、通道数与声明格式。
- GLSL 统一变量可写作 `metallic` 或 `u_metallic` 查找默认值；内置 JSON 使用
  不带 `u_` 前缀的形式。

## 校验机制

1. **材质描述**：字段缺失、类型错误、重复名称、引用文件不存在等统一报
   `E_MAT_001`。
2. **着色器管道**：顶点/片段程序先分别做编译预检，再完整链接并检查必需
   attribute/uniform 接口。编译错误与链接错误使用不同业务码，且失败后不渲染。
3. **状态隔离**：每个实例、每个确定性 pass 渲染前都重置所有活动统一变量。
   渲染后读取实际 uniform 值形成快照，同时由材质默认值、固定相机/光照参数和
   实例输入独立重算期望快照。两者哈希不一致即报 `E_STATE_001`。
4. **确定性**：每个实例在相同上下文中独立渲染两遍，四通道逐元素比较，
   最大绝对差必须 `<= 1e-6`，否则报 `E_DIFF_001`。同一输入跨进程重复运行
   也使用相同相机、光照、种子和浮点数据路径。
5. **性能约束**：每个实例前后和最终写出前检查进程峰值工作集与总耗时。
   默认超过 512 MiB 或 30 秒报 `E_PERF_001`。峰值内存是宿主进程工作集，
   不是 GPU 显存。

`--fault-inject state_residue` 和 `--fault-inject diff` 仅用于验收自测，分别
模拟统一变量残留和第二遍渲染读数偏差。默认值 `none` 不注入任何故障。

## 业务码与退出码

| 业务码 | 退出码 | 含义 |
|---|---:|---|
| `E_MAT_001` | 2 | 材质 JSON、字段、类型或引用文件校验失败 |
| `E_GL_001` | 3 | GLSL 编译失败 |
| `E_GL_002` | 3 | GLSL 链接或程序接口校验失败 |
| `E_GL_003` | 3 | 无法创建 OpenGL 4.1 Core 离窗口上下文 |
| `E_STATE_001` | 4 | 统一变量快照与独立重算状态不一致 |
| `E_DIFF_001` | 5 | 两遍渲染通道差异超过 `1e-6` |
| `E_TEX_001` | 6 | 纹理声明/实际格式不匹配或格式非法 |
| `E_PERF_001` | 7 | 峰值内存或总耗时超过限制 |
| `E_ARG_001` | 2 | 命令行参数非法 |
| `E_IO_001` | 1 | 输出目录或日志写入失败 |
| `E_INTERNAL_001` | 1 | 未分类内部错误，`--debug` 可显示堆栈 |

失败时业务码写入 stderr，便于命令行和 CI 直接匹配；可用输出目录中的
`summary.json` 读取结构化错误、已完成实例数和性能数据。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试覆盖：

- 四实例成功渲染、四通道形状/dtype、CSV/映射/状态哈希/摘要一致性。
- 相同输入两次独立运行的 `.npy` 逐元素确定性。
- 材质字段、GLSL 编译、GLSL 链接、纹理格式、状态残留、确定性偏差、
  内存/时间超限和实例数边界。

本次实际验证环境、命令与结果见 [`docs/VERIFICATION.md`](docs/VERIFICATION.md)。

