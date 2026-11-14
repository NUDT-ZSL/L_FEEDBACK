# matrelight

离线分层材质合成与方向光反射渲染工具。读取 YAML 材质描述，按层顺序合成
OpenGL(+Y) 切线空间法线、基础色与粗糙度，生成可重光照材质缓存，并对一组
方向光用 GGX 微表面模型渲染反射图。纯 CPU、无外部素材、无 GPU、无在线服务。

## 环境与安装

- Windows 10/11 64 位或 Linux；Python 3.10+（开发环境为 Python 3.14）

```bash
python -m pip install -r requirements.txt
# 若无 pip（如部分 Linux 发行版），可用 uv：
uv venv .venv
uv pip install --python .venv/bin/python numpy pyyaml pillow
```

## 快速开始

```bash
# 1. 生成确定性本地样本（贴图 + YAML + 光照列表）
python scripts/make_samples.py -o samples

# 2. 单材质渲染
python -m matrelight render samples/material_mixed.yaml -o out --seed 20260101

# 3. 批量并行渲染（并发数由 -j 限制）
python -m matrelight batch samples/material_mixed.yaml \
    samples/material_weighted.yaml samples/material_seam.yaml \
    -o batch_out --seed 20260101 -j 2

# 4. 运行测试
python -m unittest discover -s tests -v
```

Windows 下把 `python` 换成对应解释器，模块调用方式 `python -m matrelight ...` 不变。

## YAML 材质描述

```yaml
name: example
resolution: [64, 64]              # 输出与所有贴图的统一分辨率
tangent_threshold_deg: 30.0       # 切线连续性修正阈值（可配置）
uv_wrap: error                    # error | clamp | repeat
view_dir: [0.0, 0.0, 1.0]         # 观察方向，计算前单位化
lights:                           # 方向光列表，非单位向量按单位化处理
  - [0.5, 0.6, 0.8]
  - [-0.7, 0.2, 0.6]
uv_checks:                        # 可选：显式 UV 查询，触发越界校验并写入缓存
  - [0.25, 0.25]
layers:
  - name: base
    normal: textures/normal_base.png
    base_color: textures/color_base.png
    roughness: textures/rough_base.png
    weight: 1.0
    mode: overlay                 # overlay | mix | weighted（支持中文别名）
    uv_scale: [1.0, 1.0]          # 可选
    uv_offset: [0.0, 0.0]         # 可选
```

混合模式：

- `overlay` 叠加：法线按平面细节提升，颜色/粗糙度用 Photoshop 风格通道叠加。
- `mix` 混合：按归一化权重在累积结果与当前层之间线性插值。
- `weighted` 加权：全部为加权层时走全局加权快速路径，仅一次重归一化。

## 输出结构

每个材质输出到 `<output>/<material_name>/`：

- `normal_composite.png`：合成并重归一化的 OpenGL(+Y) 法线贴图，与输入同分辨率。
- `reflection_light_XX.npy`：每盏灯的线性 HDR 反射图（float32，无 tonemap 损失）。
- `reflection_light_XX.png`：Reinhard tonemap 后的 8 位预览图。
- `material_cache.mzc`：确定性 ZIP 缓存，含 JSON 元数据与 float32 数组。
- `result.json`：单项结果清单。

批量模式额外在输出根写 `batch_summary.json`，按输入顺序记录每项状态码。

## 材质缓存内容

`material_cache.mzc` 内 `metadata.json` 包含：种子、分辨率、法线编码、BRDF 模型、
`normalized_weights`、每层 `layer_contributions`（原始权重与归一化权重）、
切线基修正元数据（阈值、修正计数、修正比例）、最终 BRDF 参数、灯光与 UV 采样记录。
数组区含 `normal/tangent/bitangent/base_color/roughness/correction_marker`。

归一化权重只计算一次（`matrelight/blend.py::normalized_weights`），法线合成、
颜色粗糙度合并、反射渲染与缓存写入共用同一数组，保证三处一致。

## 错误码

错误以单行 JSON 写到 stderr，并以对应非零退出码结束。单文件与批量入口共用同一套定义
（`matrelight/errors.py`），批量失败项写入汇总且不影响其他项。

| 错误码 | 退出码 | 触发条件 |
| --- | --- | --- |
| `E_INPUT_MISSING` | 2 | YAML 或引用贴图不存在/不可读 |
| `E_LAYER_INVALID` | 3 | 层缺字段、权重非法、分辨率不一致、总权重非正、YAML 结构错 |
| `E_MIX_MODE` | 4 | 混合模式不是 overlay/mix/weighted |
| `E_UV_OUT_OF_BOUNDS` | 5 | `uv_wrap: error` 下 UV 落出 `[0,1)` |
| `E_RUNTIME` | 1 | 防御性兜底（正常校验路径不产生） |

失败项通过“临时目录构建 + 成功后原子替换”保证不会留下任何部分输出文件。

## 确定性

- `--seed` 驱动 `numpy.random.SeedSequence`，按阶段（法线合成、标量场、切线基、GGX）
  派生独立 RNG。
- 法线重归一化、半角向量归一化与加权求和的浮点归约顺序由种子置换控制
  （`matrelight/deterministic.py`），不依赖底层并行归约顺序。
- PNG 固定压缩参数且不写时间戳；`.npy` 与自定义 `.mzc`（固定 1980-01-01 ZIP 时间戳）
  均无时间戳；JSON 排序键、固定分隔符。
- 批量子种子由基础种子经 `SeedSequence.spawn` 按输入位置派生，与调度顺序、并发数无关。

同输入同种子重复运行，缓存、法线/反射图与汇总 JSON 逐字节一致（见 `docs/TECHNICAL.md`
记录的实际校验结果）。

## 目录结构

```
matrelight/
  errors.py         错误码与结构化输出
  deterministic.py  种子上下文与受控浮点归约
  image_io.py       确定性贴图读写
  material.py       YAML 解析与校验
  sampler.py        UV 采样与越界处理
  blend.py          权重归一化与分层合成
  basis.py          切线基重建与连续性修正
  shading.py        GGX 方向光反射
  cache.py          确定性材质缓存
  pipeline.py       单材质编排与原子输出
  cli.py            render/batch 命令行
scripts/make_samples.py
tests/test_pipeline.py
```
