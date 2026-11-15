# matrender

确定性、纯 CPU 的离线材质球阵列渲染器，用于游戏引擎材质的色彩校准与回归测试。

管线从自描述 JSON 场景生成：

- 线性 Rec.709 HDR 浮点图像 `linear_hdr.npy`（`float32`）
- 经 Narkowicz ACES filmic 拟合与 sRGB 伽马编码的 8 位 PNG
- 整图及逐球体区域的 HDR 统计日志
- 逐球体材质报告与可见球体掩码
- 资源降级、线程限制及渲染参数的完整决策日志

渲染核心包括 GGX/Schlick PBR、方向光阴影、法线扰动、确定性分层抖动抗锯齿、深度/法线 SSAO、固定分块多线程和内存预检降级。整个过程不使用 GPU、显示设备、随机种子或系统时钟。

本次实际验证结果见 `verification/results.json` 和 `VERIFICATION.md`；设计与算法说明见 `TECHNICAL_REPORT.md`。

## 环境

当前验证环境：

- Python 3.11.18（64 位）
- NumPy 2.4.6
- Pillow 13.0.0
- 24 个逻辑 CPU 核心

在全新 Ubuntu 环境安装：

```bash
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

也可以使用标准 `venv`：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 快速开始

校验场景：

```bash
.venv/bin/python -m matrender validate scenes/material_grid.json
```

渲染 4x3 金属度/粗糙度阶梯：

```bash
.venv/bin/python -m matrender render scenes/material_grid.json \
  -o out/material_grid \
  --samples 8 \
  --threads 4
```

也可以直接使用便捷入口：

```bash
.venv/bin/python render.py scenes/material_grid.json -o out/material_grid
```

SSAO 对比：

```bash
.venv/bin/python -m matrender render scenes/ssao_contrast.json \
  -o out/ssao_on --samples 2 --threads 4 --ssao on

.venv/bin/python -m matrender render scenes/ssao_contrast.json \
  -o out/ssao_off --samples 2 --threads 4 --ssao off
```

## 输出

每次成功渲染在输出目录写入固定文件名：

| 文件 | 内容 |
|---|---|
| `linear_hdr.npy` | 形状 `(H, W, 3)`、C 顺序、线性 Rec.709 RGB、`float32` |
| `preview_aces_srgb.png` | ACES filmic + sRGB 编码后的 8 位 RGB PNG |
| `render_log.json` | 参数、线程、分块、资源决策、整图/区域 HDR min/max、警告 |
| `material_report.json` | 每个球体的平均亮度、最大亮度、主导 HSV 色相 |
| `sphere_masks.npz` | `sphere_000`、`sphere_001` ... 可见球体布尔掩码 |

`render_log.json` 的 `regions` 条目和 `material_report.json` 的 `regions` 条目按场景中的球体顺序一一对应。区域统计直接从合并后的 HDR 图像与球体掩码计算。

## 场景格式

完整样例见 `scenes/material_grid.json`。顶层字段：

| 字段 | 必填 | 说明 |
|---|---|---|
| `scene_name` | 否 | 非空字符串，默认 `untitled` |
| `resolution.width/height` | 是 | 输出分辨率，范围 1..8192 |
| `camera` | 是 | `position`、`target`、`fov_y_deg` 必填；`up` 默认 `[0,1,0]` |
| `light` | 是 | `direction` 必填；`color` 默认白；`intensity` 默认 1 |
| `background` | 否 | `color`、`ambient`、`ambient_color` |
| `ssao` | 否 | `enabled`、`radius`、`intensity`、`samples` |
| `spheres` | 是 | 非空球体数组；兼容别名 `materials` |

球体字段：

| 字段 | 必填 | 约束 |
|---|---|---|
| `name` | 否 | 非空且唯一，默认 `sphere_000` |
| `center` | 是 | 三个有限数 |
| `radius` | 是 | `> 0` |
| `base_color` | 是 | 线性 RGB，每通道 `[0,1]` |
| `metallic` | 是 | `[0,1]` |
| `roughness` | 是 | `(0,1]` |
| `normal_perturbation` | 是 | `[0,1]` |

所有数值必须是有限数。相机 `position` 与 `target` 必须不同，`up` 不得与视线平行。颜色除明确写为编码空间外，均为线性 RGB。

## 错误与降级

场景缺失字段、类型错误、JSON 语法错误或数值越界时：

- 向 stderr 输出包含 `E_INVALID_SCENE` 的业务错误；
- 退出码为 `0`；
- 不创建输出目录，不渲染 `npy`/PNG。

示例：

```text
E_INVALID_SCENE: scenes/invalid_roughness.json.spheres[0].roughness: must be greater than 0
```

内存预检超过 `--max-memory-mb` 时：

- 先逐级减半采样数，再按 0.75 比例降低分辨率；
- 向 stderr 输出 `E_RESOURCE_LIMIT`；
- 继续完成渲染并写出所有输出；
- 在 `render_log.json.resource.degradation_events` 中记录每一步的旧值、新值和估算内存。

线程数大于逻辑核心数时限制到逻辑核心数，并在日志和 stderr 记录 `W_THREAD_LIMIT`。

## 命令行

```text
matrender render SCENE [-o DIR]
  --samples N          1..64，默认 8
  --threads N          默认 min(4, 逻辑核心数)
  --tile-size N        row-major 方块边长，默认 64
  --max-memory-mb N    预检上限，默认 2048
  --ssao auto|on|off   默认 auto（采用场景设置）
  --width N            覆盖场景宽度
  --height N           覆盖场景高度
  --quiet              不输出成功摘要
```

安装项目后也可以使用 `matrender` 命令：

```bash
.venv/bin/python -m pip install -e .
matrender render scenes/material_grid.json -o out/material_grid
```

## 测试与验证

运行确定性测试：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

重新生成全部验收工件：

```bash
.venv/bin/python scripts/verify.py
```

验证脚本会执行 18 个测试，并生成：

- `verification/results.json`：机器可读验收结果
- `verification/material_grid/`：640x480、8 spp 官方材质阵列输出
- `verification/ssao_on/`、`verification/ssao_off/`：384x256、2 spp SSAO 对比
- `verification/resource_fallback/`：强制低内存上限后的降级输出

## 目录结构

```text
matrender/
  scene.py       JSON 解析、校验、不可变场景模型
  renderer.py    光线求交、PBR、SSAO、AA、分块线程、资源计划
  color.py       ACES filmic、sRGB 编解码、亮度与 HSV 色相
  reporting.py   HDR/区域/材质统计
  cli.py         CLI、输出文件与日志
scenes/          确定性正常、SSAO 对比、非法场景
tests/           18 个 unittest 验收测试
scripts/verify.py 可复现验证与工件生成
verification/    实际运行结果、日志、HDR、PNG、掩码
```

