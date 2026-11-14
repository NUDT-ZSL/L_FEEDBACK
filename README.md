# spatmix — 确定性离线空间音频混音管线

读取 JSON 场景描述（听者轨迹、声源、材质、障碍物、反射面），程序化合成单声道干音，
按固定帧推进完成空间化与动态混音，输出多声道 WAV 与版本化 JSON 事件日志。
支持快照/恢复：从中断处继续处理的输出与不间断处理**逐样本、逐事件一致**。

## 环境

- Python ≥ 3.10（本仓库使用 `.venv`，Python 3.14）
- 依赖：`numpy`、`scipy`、`soundfile`（测试另需 `pytest`）
- 不需要外部音频素材与网络访问；所有样本由代码确定性生成

```bash
uv venv .venv && uv pip install --python .venv/bin/python numpy scipy soundfile pytest
```

## 用法

```bash
# 生成示例场景
python -m spatmix make-example --out examples

# 渲染（输出 mix.wav / events.json / events.jsonl / mix.rawf32）
python -m spatmix render examples/example_scene.json --out out/

# 部分渲染（在指定帧停止并写快照）
python -m spatmix render examples/example_scene.json --out out/ --max-frames 200

# 从快照恢复（输出与一次性渲染逐样本一致）
python -m spatmix resume --snapshot out/snapshot.json
```

错误时 stdout 输出 `{"error_code": ..., "message": ...}` 并以非零码退出：

| error_code | 退出码 | 含义 |
|---|---|---|
| `E_SCHEMA` | 2 | JSON 结构/类型非法 |
| `E_TRACK_ORDER` | 3 | 轨迹时间戳非严格单调 |
| `E_RANGE` | 4 | 数值非有限或越界 |
| `E_SNAPSHOT_VERSION` | 5 | 快照版本不匹配 |
| `E_SNAPSHOT_MISMATCH` | 6 | 场景哈希/输出目录与快照不符 |

## 处理模型（每帧依次执行）

1. **轨迹采样**：听者位置/朝向按关键帧线性插值，端点外钳制。
2. **距离衰减**：`g = (ref/max(d,ref))^2`（反平方，参考距离可配）。
3. **空气吸收**：一阶低通，截止 `fc = 0.45·sr·exp(-α·d)`（下限 100 Hz）；
   材质标签覆盖默认 α，未知材质回退默认值并记录 `unknown_material` 事件（不静默）。
4. **多普勒**：由帧间距离差分径向速度，`factor = c/(c+v)`；系数在帧内线性斜坡、
   跨帧一阶平滑（`doppler_smoothing` 时间常数），超出 `[doppler_min, doppler_max]`
   限幅并记录 `doppler_clamp_start/end`；线性插值重采样，相位为 float64 绝对位置。
5. **早期反射**：每个轴对齐反射面生成一阶镜像声源，延迟按 `round(d_img/c·sr)`
   采样点对齐，增益 `(1-absorption)·(ref/d_img)^2`；延迟 <1 采样
   （`degenerate_image`）或增益低于阈值（`too_weak`）时跳过并记录原因。
   反射参数在处理开始时（t=0 听者位置）确定，帧间保持恒定延迟线。
6. **遮挡**：听者-声源线段与 AABB 做 slab 相交测试；遮挡施加低通
   （`fc·(1-0.95·occ)`，下限 200 Hz）与增益 `(1-0.9·occ)`；端点位于盒内的退化
   情形按固定部分遮挡 0.5 处理并记录 `occlusion_degenerate_*` 事件。
7. **声像**：VBAP（2.0: L/R ±30°；5.1: L/R/C/LS/RS，LFE 恒为 0），等功率归一化，
   帧间增益线性交叉淡化；2.0 后方等 VBAP 无解区域采用镜像回退（确定性规则）。
8. **写入**：每帧完成空间化与状态更新后，帧数据追加到 `mix.rawf32`，
   结束时转换为 `mix.wav`（float32）。

## 确定性设计

- 噪声干音使用 `PCG64`，种子由声源 ID 的 SHA-256 派生，跨进程逐样本一致。
- 全部中间状态为 float64；快照 JSON 以最短往返精度序列化浮点，无精度损失。
- 快照内容：版本号、已完成帧数、混音缓冲位置、每声源的 RNG 状态、干音缓冲、
  重采样相位与平滑系数、两个低通的滤波器状态、反射延迟线内容与读写位置、
  上一帧声像增益、遮挡/多普勒/可听性状态机标志。
- 事件日志增量写入 `events.jsonl`（首行为含 `log_version` 的 header），
  恢复时仅追加新事件，保证与不间断运行完全一致。

## 验证结果（本机实测）

`python -m pytest tests/ -q` → **14 passed**，覆盖：

- 5.1 与 2.0 渲染：声道数、采样数、LFE 静音、全部业务事件类型齐备且按帧有序
- 跨进程两次渲染：WAV 样本与 events.json 字节级一致
- 快照恢复：`--max-frames 200` 中断后 `resume`，输出与一次性渲染逐样本、逐事件一致
- 非法输入：`E_TRACK_ORDER`(3) / `E_RANGE`(4，NaN 与超奈奎斯特频率) / `E_SCHEMA`(2)
- 快照版本不匹配 `E_SNAPSHOT_VERSION`(5)、场景哈希不匹配 `E_SNAPSHOT_MISMATCH`(6)
- 单元级：噪声种子确定性、AABB 相交/包含、VBAP 关键方位增益

示例场景（2 s / 48 kHz / 5.1）实测事件：3×`reflection_added`、3×`reflection_skipped`
(far_wall too_weak)、`occlusion_start/end`、`occlusion_degenerate_start`（声源位于盒内）、
6×`doppler_clamp_start`、`unknown_material`(unobtanium→default)、`source_audible/inaudible`。

## 目录结构

```
spatmix/
  scene.py     场景解析与校验（E_SCHEMA/E_TRACK_ORDER/E_RANGE）
  synth.py     确定性干音合成（sine/noise/impulse，懒生成+RNG 状态）
  dsp.py       一阶低通（系数斜坡去拉链）、延迟线、变速重采样器
  geometry.py  轨迹插值、AABB slab 测试、镜像声源
  panning.py   VBAP（2.0 / 5.1）
  engine.py    帧推进引擎：空间化、事件、状态快照
  snapshot.py  快照存取与版本/场景哈希校验
  cli.py       render / resume / make-example
tests/test_pipeline.py   14 项端到端与单元测试
examples/example_scene.json
```
