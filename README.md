# 离线帧渲染调度与合成工具

这是一个从零实现的确定性离线帧渲染器，面向 Windows，也可在 Linux/macOS
运行。它只依赖 Python 3.11、numpy 1.26 和 Pillow 10.2，不使用 GPU、容器、
网络资源或第三方渲染器。

实现包括：

- 向量化光线与球体求交、最近命中、主光线着色和点光源阴影判定；
- Lambert 漫反射与 Blinn-Phong 高光，线性颜色钳位后编码为 8 位 sRGB；
- 固定的 `N x N` 子采样网格，像素内偏移由每帧 JSON 种子生成；
- 帧块任务、操作系统子进程并发、原子检查点和确定性重试；
- 逐帧流式写 PNG，拼装时逐文件复制和校验，不把整段序列装入内存；
- 检查点、帧块清单和最终序列清单三个层次的指纹/校验和绑定。

## 环境准备

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Linux/macOS 或本仓库当前验证环境：

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

仓库内已包含确定性生成的 `samples/scene_240.json`。也可以重新生成：

```bash
.venv/bin/python -m frame_render.sample \
  --output samples/scene_240.json \
  --frames 240 --width 128 --height 72 --samples-per-axis 4
```

Windows 下将 `.venv/bin/python` 换成 `.venv\Scripts\python`。

## 运行

完整渲染并拼装：

```bash
.venv/bin/python -m frame_render run \
  --input samples/scene_240.json \
  --work-dir var/render_work \
  --output var/render_frames \
  --chunk-size 20 \
  --jobs 4 \
  --max-retries 1
```

只恢复/渲染帧块，不执行最终拼装：

```bash
.venv/bin/python -m frame_render render \
  --input samples/scene_240.json \
  --work-dir var/render_work \
  --chunk-size 20 --jobs 4
```

从已有工作目录拼装：

```bash
.venv/bin/python -m frame_render assemble \
  --input samples/scene_240.json \
  --work-dir var/render_work \
  --output var/render_frames \
  --chunk-size 20
```

验证最终序列清单和所有帧文件：

```bash
.venv/bin/python -m frame_render verify --output var/render_frames
```

安装为本地命令后也可以使用 `frame-render`：

```bash
uv pip install --python .venv/bin/python -e .
```

正常成功摘要写到 stdout。业务失败摘要写到 stderr，格式为 JSON。
工作目录和输出目录必须互不包含，避免替换最终目录时删除工作状态。

## 输入 JSON

输入格式是严格 JSON，不接受重复键或 `NaN`/`Infinity`。顶层结构如下：

```json
{
  "schema_version": 1,
  "scene": {
    "spheres": [
      {
        "id": "center",
        "center": [0.0, 0.0, 0.0],
        "radius": 1.0,
        "albedo": [0.8, 0.2, 0.1],
        "specular": 0.6,
        "shininess": 64.0
      }
    ],
    "light": {
      "position": [4.0, 6.0, 4.0],
      "color": [1.0, 0.95, 0.9],
      "intensity": 1.15
    },
    "camera": {
      "width": 128,
      "height": 72,
      "fov_y_deg": 43.0,
      "position": [6.4, 1.15, 0.0],
      "target": [0.0, 0.18, 0.0],
      "up": [0.0, 1.0, 0.0]
    },
    "ambient": [0.055, 0.065, 0.085],
    "background": [0.012, 0.018, 0.032]
  },
  "animation": {
    "frame_start": 0,
    "frame_end": 1,
    "frames": [
      {
        "frame": 0,
        "position": [6.4, 1.15, 0.0],
        "target": [0.0, 0.18, 0.0],
        "up": [0.0, 1.0, 0.0],
        "seed": 1592677427
      },
      {
        "frame": 1,
        "position": [6.35, 1.2, 0.5],
        "target": [0.02, 0.18, 0.0],
        "up": [0.0, 1.0, 0.0],
        "seed": 1592677428
      }
    ]
  },
  "sampling": {
    "samples_per_axis": 4,
    "jitter": true,
    "seed_algorithm": "pcg64-grid-jitter-v1"
  }
}
```

球体可以使用 `albedo`，也可以使用同义字段 `color`，但不能同时提供。
`scene.camera` 提供固定的内参和初始外参；每帧 `position`、`target` 和可选
`up` 覆盖相机外参，从而表达平移和旋转轨迹。帧号必须从 `frame_start` 连续
递增到 `frame_end`，且每帧必须提供 uint64 范围内的 `seed`。

## 渲染与确定性

每个像素在固定 `samples_per_axis x samples_per_axis` 网格中生成样本。每帧
使用该帧 seed 初始化一个局部 `numpy.random.PCG64`，不读取或修改 numpy
全局随机状态。样本偏移、相机基向量、求交顺序、着色和量化顺序全部固定，
因此输出不依赖帧块大小或并发度。

主光线对全部球体求出最近正根；命中点使用 Lambert 漫反射和 Blinn-Phong
高光计算线性 RGB，并额外发射阴影光线判断点光源可见性。线性颜色钳位到
`[0, 1]` 后用分段 sRGB 传递函数编码，再四舍五入到 8 位。

## 帧块、指纹与检查点

连续帧按 `--chunk-size` 切片。每个帧块指纹由以下规范化 JSON 计算：

- 渲染器/schema/sampler 版本；
- 球体、光源、相机、环境光和背景组成的几何摘要；
- 子采样配置摘要；
- 本块帧区间及这些帧的相机外参和 seed。

任一相关输入变化都会改变相应块指纹。检查点格式：

```json
{
  "checkpoint_version": 1,
  "format_version": 1,
  "schema_version": 1,
  "renderer_version": "ray-sphere-srgb-v1",
  "chunk_id": "chunk_000000_000019",
  "frame_start": 0,
  "frame_end": 19,
  "fingerprint": "...",
  "geometry_digest": "...",
  "sampling_digest": "...",
  "artifact_manifest": "chunks/chunk_000000_000019/manifest.json",
  "artifact_checksum": "sha256:..."
}
```

工作器先把帧逐个写入临时目录，完成后再原子改名为正式帧块目录，最后原子
写入检查点。帧块清单记录每个 PNG 的相对路径、字节数和 SHA-256。恢复时
逐项验证版本、指纹、帧区间、清单校验和、文件大小及文件 SHA-256。记录
损坏、字段不匹配或版本不匹配都会使检查点失效，该块被强制重渲染，不会
静默沿用旧产物。

工作目录结构：

```text
work/
  checkpoints/chunk_*.json
  chunks/chunk_*/manifest.json
  chunks/chunk_*/frames/frame_*.png
  state/scheduler_state.json
  tasks/*.json
  results/*.json
  reports/*.json
  reports/last_run_summary.json
  logs/*.log
```

## 子进程与失败语义

调度器使用 `subprocess.Popen` 启动独立 Python 操作系统进程，每个任务和
结果都通过 JSON 文件传递。同时存活的子进程数不会超过 `--jobs`。已完成
产物和累计尝试/重试次数持久化在工作目录中。

- 捕获到普通可恢复异常时，工作器在退出前原子写入失败报告，包含失败原因、
  当前块已完成帧号、启动任务时已完成的其他块快照、`chunk_completed=false`
  和 traceback；调度器随后按 `--max-retries` 重试。
- 工作器被 `os._exit` 或外部强制终止时不保证报告存在。只要已有检查点有效
  且指纹一致，调度器就视为该块完成；否则记录 `forced_termination`，继续
  处理其他块并在后续运行中重算该块。
- 调度器自身收到 `Ctrl+C` 时终止当前子进程、持久化状态并以 130 退出；再次
  运行会从有效检查点继续。

`--max-retries` 是单次调度器调用的额外重试预算；工作目录中的累计尝试次数
和累计重试次数会保留并出现在摘要中。
每次调度结束时还会原子写入 `reports/last_run_summary.json`，其中包含
已完成块 `completed_chunks`、未完成块 `unfinished_chunks`、累计尝试次数和
`retry_counts`。

## 拼装与完整性

拼装前重新验证所有预期帧块及指纹，按帧号升序建立唯一映射。任何缺块、
缺帧、重复帧、文件损坏、大小不匹配或指纹不一致都会阻止输出。验证通过
后，先写入同级临时目录，逐帧流式复制并重新计算 SHA-256；再验证最终组合
摘要，最后替换输出目录。输出包括：

```text
output/
  frame_000000.png ... frame_000239.png
  sequence_manifest.json
```

`sequence_manifest.json` 记录每个帧的源帧块、帧块指纹、文件大小和
SHA-256，以及整个序列的 `sequence_digest`。

缺帧示例 stderr 摘要包含要求字段：

```json
{
  "status": "error",
  "error": "assembly_integrity",
  "successful_frames": 220,
  "missing_frames": [100, 101, "..."],
  "corrupted_chunks": [
    {
      "chunk_id": "chunk_000100_000119",
      "frame_start": 100,
      "frame_end": 119,
      "reason": "frame_file_missing:100"
    }
  ],
  "duplicate_frames": []
}
```

退出码：

- `0`：成功；
- `2`：输入、参数或一般 I/O 错误；
- `3`：渲染未完成（包含重试耗尽或强制终止未恢复）；
- `4`：拼装/序列完整性失败；
- `130`：调度器被中断。

## 内存与并发约束

工作器一次只保留当前帧的射线、着色和图像数组，PNG 在帧完成后立即落盘；
块内只保留帧元数据清单。拼装阶段逐帧读取和复制，不做整段序列解码。
因此内存约为 `并发进程数 x 单帧峰值内存`，而不是 `帧数 x 单帧内存`。
并发上限只影响吞吐，不参与任何像素计算或输出排序。

## 测试与实测结果

运行自动化测试：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

运行完整 240 帧验收脚本：

```bash
.venv/bin/python scripts/validate_acceptance.py
```

验收脚本会重建 `samples/scene_240.json`，在 `var/acceptance/` 下执行实际
渲染、恢复和破坏性测试，并写入 `artifacts/acceptance_report.json`。

本次实际验证环境为 Python 3.11.18、numpy 1.26.4、Pillow 10.2.0。样本为
240 帧、128x72、每帧 4x4 超采样、块大小 20。结果：

- 自动化测试 6 项全部通过；
- 1 进程与 4 进程最终像素摘要一致：
  `8066db6e562894862eefb7adb41349467c5ef5d33be56c74d44cfa853c3d1739`；
- 两种并发度的 PNG 字节摘要也一致：
  `92aeb235b217875c602f167e5dcf520e0919f77046fe34706fa630e8e8ecaccc`；
- 正常退出后再次运行，12/12 帧块复用、0 块重渲染；
- 破坏 `chunk_000100_000119` 检查点版本后，只重渲染该块，最终像素仍一致；
- `os._exit(23)` 测试后，11 块复用、1 块重算，输出与串行结果一致；
- 删除一帧后拼装退出码为 4，成功帧 220、缺失帧 100..119，并指出损坏块；
- 1 进程完整耗时 40.340 秒，4 进程完整耗时 13.233 秒（本机实测）。

`OFR_TEST_FAULT` 环境变量只用于测试脚本确定性地注入可恢复失败或
`os._exit`；正常渲染不会设置或依赖它。
