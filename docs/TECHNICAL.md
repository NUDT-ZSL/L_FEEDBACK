# 技术设计

## 1. 处理流水线

单变体渲染顺序固定为：

1. 读取配方并完成 schema/数值校验。
2. 预读取全部源 WAV，校验 WAV 结构与 SHA-256。
3. 为每个事件解析源声道到目标声道的增益矩阵。
4. 用按变体索引隔离的随机流抽取增益抖动、时移抖动和尾音泛音参数。
5. 源重采样到目标采样率。
6. 事件源补零或裁剪到事件段长度，再按时间偏移混入目标音轨。
7. 总线直流去除。
8. 软压缩。
9. 尾音淡出。
10. RMS/LUFS 归一化。
11. 真峰值限制器。
12. 编码 WAV、回读头字段、检查削波与真峰值。
13. 写入清单；全部变体完成后原子发布整批目录。

生成和清单重放共用 `render_timeline()`，避免出现“生成一套算法、重放另一套算法”的不一致。

## 2. 时间线与裁剪

毫秒到帧的换算统一使用：

```text
frames = floor(milliseconds * sample_rate / 1000 + 0.5)
```

事件时间偏移为 `offset_frames`。负偏移、等于或超过输出长度的偏移直接报 `SFX-E2003`，上下文包含 `event_index`、原始 `source_path` 和解析后路径。

事件窗口长度：

- 配方给出 `duration_ms`：使用指定长度；
- 未给出：使用重采样后的完整源长度。

映射后的事件块：

- 短于窗口：尾部补零；
- 长于窗口：裁剪；
- 窗口超过输出末端：仅混合能落入输出音轨的部分。

## 3. 重采样与声道矩阵

重采样使用 FFT 频谱截断/补零。源长度 `N`、目标采样率 `dst`、源采样率 `src` 的输出长度为：

```text
round(N * dst / src)
```

降采样时高频频谱被截断，避免直接抽取造成混叠。同采样率直接复制，不经过 FFT。

声道映射最终都归一为 `目标声道数 × 源声道数` 矩阵，输出为：

```text
mapped[frame, target] = sum(source[frame, src] * matrix[target, src])
```

因此字符串映射、规则数组和显式矩阵共享同一条重放路径。清单保存最终矩阵。

## 4. 随机流隔离

变体摘要种子：

```text
SeedSequence([base_seed, variant_index]).generate_state(1, uint64)
```

每个事件、每种随机用途使用独立 SeedSequence：

```text
SeedSequence([base_seed, variant_index, event_index, stream_id])
```

`stream_id` 分别对应增益、时移和尾音泛音。这样：

- 一个变体的抽样次数不影响其他变体；
- 一个事件的抽样不影响同一变体的其他事件；
- 并发完成顺序不影响任何值；
- 清单保存实际抽样值，重放不依赖随机数生成器状态。

不同变体至少需要一种随机源处于启用状态，否则多变量配方会被拒绝。

## 5. 总线处理

### 5.1 直流去除

逐声道减去整条输出的均值，清单记录被移除的直流值。

### 5.2 软压缩

探测器为双声道联动样本峰值。软拐点曲线：

```text
delta = level_db - threshold_db

if delta < -knee/2:
    reduction = 0
elif delta > knee/2:
    reduction = (1/ratio - 1) * delta
else:
    reduction = (1/ratio - 1) * (delta + knee/2)^2 / (2*knee)
```

增益衰减（dB）再经过 attack/release 一极点平滑，最后转换为线性增益乘到所有声道。平滑发生在 dB 域，避免瞬时阶跃。

### 5.3 尾音淡出

尾音余量：

```text
tail_margin = output_frames - latest_event_end_frame
```

请求淡出长度取 `min(requested_frames, tail_margin)`，使用升余弦曲线。若被截断，`processing.tail_fade.warning` 与 `analysis.warnings` 都记录 `fade_truncated`。

### 5.4 响度归一化

配方可选：

- `target_lufs`：K-weighting 后按 400 ms 块/100 ms 步长、绝对与相对门限计算综合响度；
- `target_rms_dbfs`：全样本均方根。

K-weighting 在频域应用两级 biquad 传递函数，避免外部依赖。归一化增益受 `min_gain_db` / `max_gain_db` 限制，并记录归一化前/后测量值。

### 5.5 真峰值限制器

真峰值估计先做 4 倍 FFT 过采样，再按基础采样点取局部最大包络。限制器：

1. 从目标 ceiling 计算每点所需增益；
2. 在 lookahead 窗口内取未来最小值；
3. 对增益做一极点 attack/release 平滑；
4. 应用后重新估计真峰值；
5. 若仍有残差，应用常量安全修正；
6. 预留 `0.02 dB` 内部量化余量，保证 16-bit 量化后仍不越过配置 ceiling。

清单记录动态限制器触发区间、最大增益衰减、安全修正量、动态后真峰值与最终真峰值。

## 6. WAV 封装与回读

写入器生成确定的 RIFF/WAVE 字节流：

- PCM 格式码 `1`，16/24/32-bit；
- IEEE float 格式码 `3`，32-bit；
- 24-bit 为 3 字节小端打包；
- 奇数数据长度自动补一个 RIFF pad 字节。

写入后立即重新解析：

- `RIFF` / `WAVE` 标识；
- `fmt ` 与 `data` 块边界；
- 格式码、采样率、声道、位深、block align、byte rate；
- RIFF 长度与文件长度；
- `data` 字节数、block align 和帧数一致性；
- 解码后的削波计数和真峰值。

任何失败都在 stderr 输出 `SFX-E4001` 或 `SFX-E4002`，并附带路径和字段上下文。

## 7. 清单重放

清单是生成时的完整渲染描述，不依赖原配方。`verify` 对每个变体：

1. 校验清单版本和派生种子；
2. 读取清单引用的源 WAV 并验证 SHA-256；
3. 用清单中的时间线、映射矩阵、抖动实际值、泛音实际值、总线参数重建音频；
4. 用同一编码器生成预期 WAV 字节；
5. 验证实际文件 SHA-256；
6. 逐字节比较实际文件与重建字节；
7. 回读实际 WAV 头并检查削波；
8. 重算归一化增益和限制器触发次数，与清单记录比较。

因此修改输出、源文件、时间线、随机实际值、总线参数或种子都会导致 `SFX-E5001`。

## 8. 并发与原子发布

- `ThreadPoolExecutor` 工作线程数限制为 `1..64`；
- 每个变体只写自己的暂存 WAV，互不共享可变状态；
- 源音频以只读数组共享；
- 任一 Future 失败后设置取消事件、取消尚未启动的任务；
- 等待已运行任务退出后删除整个暂存目录；
- 输出目录必须不存在或为空；
- 全部 WAV 和 `manifest.json` 完成并通过回读后，使用 `os.replace` 将整批暂存目录改名为最终目录。

因此失败批次不会留下最终目录，也不会影响已有非空输出目录。相同种子的音频结果只由变体索引和配方决定，与 worker 数、完成顺序无关。

## 9. 可复现性边界

本工具在相同 Python/NumPy/平台运行时内验证为逐字节一致。FFT、libm 和 SIMD 实现可能在其他版本/架构上产生末位浮点差异；若要求跨平台分发位级一致，应固定运行时并将 WAV 哈希作为版本资产契约。

`numpy` 的 PCG64/SeedSequence 流本身是确定算法；本项目没有使用 Python 全局 `random` 状态、时间戳随机数或线程调度相关随机源。
