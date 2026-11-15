# 技术说明与验证记录

## 1. 模块结构

| 模块 | 职责 |
| --- | --- |
| `audiopack/timeutils.py` | 固定换算规则 `floor(round(ms*sr/1000))` |
| `audiopack/resample.py` | Kaiser 加窗 sinc 重采样（64 taps，β=9.5，分块向量化） |
| `audiopack/audioio.py` | WAV 读取（float32 内部格式）、声道归一、24-bit 写出 |
| `audiopack/timeline.py` | timeline.json 加载与严格字段/范围校验 |
| `audiopack/renderer.py` | 切片、循环、补静音、淡化、事件增益、优先级合成、重叠记录 |
| `audiopack/loudness.py` | BS.1770-4 测量、增益曲线（≥100 ms 平滑）、归一化 |
| `audiopack/memoryutil.py` | 大缓冲分配前的 512 MiB 高水位估算 |
| `audiopack/pipeline.py` | 端到端编排、部分输出清理 |
| `audiopack/cli.py` / `__main__.py` | 命令行入口与 stderr JSON 错误契约 |

## 2. 关键设计决策

### 重采样

- 窗函数 sinc 插值，截止频率 `0.5 * min(1, ratio)`：上采样时为重建滤波，
  下采样时同时充当抗混叠滤波。
- 每个输出样本对核重新归一到单位 DC 增益；边界采用源索引钳位，输出长度严格为
  `round(in_len * out_sr / in_sr)`，脉冲峰值位置与理想位置误差不超过 ±1 采样
  （见 `tests/test_resample.py`）。
- 8192 帧分块 + `einsum` 向量化，长信号内存占用有界。

### 渲染与重叠

- 每条事件渲染到独立段后与主缓冲逐样本合成；`owner`（事件索引）与
  `priority` 两张映射决定覆盖：高 priority 胜；同 priority 时大索引（文件中后
  出现者）胜。
- 重叠报告按事件对输出，含起止采样点、时长、胜/败事件索引与判定规则。
- 循环使用 `np.tile` 平铺后截断，循环接缝不做淡化；仅事件头/尾施加线性淡化。
- 非循环素材不足时尾部保留零缓冲并记录 `W_SHORT_CLIP`（含补静音采样数）。
- 损坏片段在事件粒度捕获 `E_CORRUPT_CLIP`，该事件标记 error 后继续渲染其余事件。

### 响度

- `pyloudnorm` 按 BS.1770-4 测积分响度；静音返回 `-inf`，报告中以 `null` 表示。
- 目标增益 = `-23 - measured`；当 measured ≤ -70 LUFS 或所需增益 > +20 dB 时
  封顶 +20 dB 并记录 `W_LOW_LOUDNESS`。
- `gain_envelope` 提供自动化增益曲线，任何增益变化的过渡长度至少 100 ms
  （48 kHz 下 4800 采样），dB 域线性插值；整体归一化采用恒定增益（等价于
  零变化的自动化曲线），因此输出响度可稳定命中 -23 LUFS。

### 内存与异常

- 仅统计长度相关的大缓冲（主 mix、ownership 映射、最长事件段、缓存源数据），
  分配前按 float32 字节数做高水位校验，超限抛 `E_MEMORY_LIMIT` 且不执行分配；
  写出阶段失败会删除已生成的 WAV/报告。

### 确定性

- 无随机并发；噪声样本固定 `np.random.default_rng(20261116)`。
- 目录扫描 `sorted`、事件按文件顺序处理、浮点按固定顺序累乘。
- 报告不含时间戳，`json.dump(..., sort_keys=True)`；WAV 由 libsndfile 确定性写 24-bit PCM。

## 3. 验证结果

验证命令与实测输出见仓库根目录 `VERIFICATION.md`（由实际运行生成）。
