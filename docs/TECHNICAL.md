# 技术说明

## 模块边界

| 模块 | 职责 |
|---|---|
| `game_audio_mixer/timeline.py` | JSON 解析、字段别名、素材路径解析、非法条目隔离 |
| `game_audio_mixer/dsp.py` | 重采样、K 加权响度、短时响度、真峰值、压缩、限幅、量化 |
| `game_audio_mixer/engine.py` | 素材缓存、事件渲染、轨道状态、混音、归一化、报告 |
| `game_audio_mixer/cli.py` | 参数验证、诊断输出、退出码、文件写入调度 |

时间轴解析先把输入归一到内部模型，再由引擎进行两阶段处理：先确定所有成功事件的
结束帧并缓存渲染片段，再分配统一混音缓冲。这样输出长度只依赖有效事件，单个
素材/事件失败不会影响其他事件。

## 采样与对齐

素材的原始采样率通过 `soundfile.info` 读取。检查通过后使用
`scipy.signal.resample_poly` 转换到 48 kHz，上下采样比例由最大公约数化简，
例如 44.1 kHz 为 `up=160, down=147`。报告保存原始采样率、原始声道数、
转换公式、原始帧数、引用 ID 和实际加载次数。

缓存键为解析后的绝对文件路径，失败结果也缓存。多个事件引用同一路径时只尝试
一次 `soundfile.read`，后续仅按片段偏移切片。事件时间换算：

```text
start_frame = round(time_ms * 48000 / 1000)
offset_frame = round(offset_ms * 48000 / 1000)
clip_frames = round(duration_ms * 48000 / 1000)
```

整数毫秒在 48 kHz 下对应整数个 48 样本块。同一时间索引的片段直接求和：

```text
mix[start:end, channel] += source[offset:offset+clip_frames, channel]
```

单声道素材复制到两个声道。素材片段超过源末尾时截断并写入 warning；偏移位于
源末尾之后时该事件跳过。

## 轨道状态

JSON 中的 `mute`/`solo` 作为初始状态。CLI 给出对应参数时，全部轨道的该状态被
替换为命令行集合。最终可听性：

```text
if any_effective_solo:
    audible = effective_solo
else:
    audible = not effective_mute
```

因此独奏轨即使静音也可听。报告为每个轨道记录初始值、最终值、状态来源
(`timeline`/`cli`) 与 `audible`。有效但不可听的事件状态为 `skipped`，其时长
仍参与输出长度计算。

## 动态范围控制

压缩器为立体声联动前馈峰值压缩，默认值：

- 阈值 -24 dBFS
- 压缩比 2.5:1
- 软拐点 6 dB
- attack 10 ms
- release 180 ms

level 取左右声道绝对峰值的较大者。软拐点增益计算后，用 attack/release 单极
平滑器得到平滑增益曲线，再作用于两个声道。为让阈值与目标响度相关，混音先按
目标做预归一化，再进入压缩器；该预增益、实际最大衰减量和活跃比例均写入报告。

## 响度与归一化

响度实现 ITU-R BS.1770-4：

1. 两级 48 kHz K 加权（高架滤波 + RLB 高通）。
2. 400 ms 块，100 ms 步进；块能量为各声道均方能量之和。
3. 绝对门限 -70 LUFS。
4. 相对门限为绝对门限后响度减 10 LU。
5. 最终 `-0.691 + 10 log10(mean(block_power))`。

1 kHz 单声道峰值 -20 dBFS 正弦的测试值为 -23.00 LUFS，可作为校准检查。
短时响度使用 3 s 窗口、1 s 步进；短素材不足 3 s 时使用完整素材长度。静音或
无定义响度在 JSON 中写 `null`，不输出非标准 `Infinity`/`NaN`。
若整合响度素材不足一个 400 ms 标准块，使用现有全部样本计算能量估计，并在
报告输出中保持实际样本长度。

归一化是闭环过程。每轮用当前增益生成候选信号，经过限幅、真峰值保护和 24-bit
量化后再测量。若与目标偏差超过 0.03 LU，就按偏差更新增益，最多 16 轮，并保留
离目标最近的候选。由于测的是实际量化输出，报告值与最终 WAV 对齐。

## 真峰值、软限幅与触发记录

真峰值使用 4 倍多相过采样后的绝对峰值，报告单位 dBTP。若限幅前真峰值高于
-1.0 dBTP，则启用限幅。基础级使用阈值连续、斜率连续的 tanh 软膝：

```text
y = sign(x) * (T + (C-T) * tanh((abs(x)-T)/(C-T)))
```

其中 T 为 -1 dBFS，C 为 -0.2 dBFS。之后计算局部前瞻增益包络，对过采样域的
残留 intersample peak 在 5 ms 前瞻、50 ms release 范围内局部衰减，避免一个
孤立瞬态导致整段节目被压小。最后仍有残差时，以小范围/全局保护增益严格保证
0 dBTP 上限。

限幅活跃 mask 按 50 ms 间隔合并为触发区。每个触发区记录：

- `track`、`event_id`、`event_type`：区内贡献最大的可听事件；
- `time_s`、`start_s`、`end_s`：峰值时间与区段；
- `peak_dbfs`、`gain_reduction_db`；
- `contributors`：所有重叠且有贡献的事件。

## 量化与输出

内部混音为 float64。最终样本按 24-bit 有符号整数范围量化：

```text
q = round(x * 8388607)
q_float = clip(q, -8388608, 8388607) / 8388607
```

再以 `subtype="PCM_24", format="WAV"` 写出。输出固定双声道、48 kHz。
JSON 通过临时文件加 `os.replace` 原子替换，`allow_nan=False` 保证严格可解析。

## 报告结构

顶层固定字段：

```text
report_schema, status, project, input, output, target, loudness, tracks,
assets, asset_cache, events, processing, limiter_triggers, warnings, errors
```

- `loudness`：`integrated_lufs`、`true_peak_dbtp`、`sample_peak_dbfs`、
  `short_term` 数组及测量方法。
- `tracks`：初始/最终 mute/solo、来源、可听性、轨道增益。
- `assets`：转换方式、重采样比例、原始采样率、引用和加载次数。
- `asset_cache`：唯一文件数、加载尝试、成功、缓存命中、失败。
- `events`：每个成功解析事件的 `mixed`/`skipped`、时间和原因。
- `processing`：预归一化增益、压缩参数/实际衰减、归一化迭代、限幅参数。
- `warnings`/`errors`：可继续处理的诊断；事件错误不改变成功退出码。

## 失败隔离

以下情况只跳过单个条目并继续：

- 事件不是对象，或缺少类型/轨道/素材/时间；
- 时间戳为负、非有限数、超出声明时间窗口或最大支持时间；
- 素材缺失、文件损坏、采样率不在白名单、声道数不支持；
- 片段偏移超过源长度、片段时长为零；

事件 JSON 根结构错误、`events` 不是数组、JSON 语法错误或外置清单不可解析是
致命错误，返回 3。响度或真峰值交付失败时保留输出并返回 5。
