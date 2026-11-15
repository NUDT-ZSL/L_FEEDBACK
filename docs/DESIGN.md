# 技术设计说明

## 1. 总体架构

离线处理器以 48 kHz / 立体声 / 32 位浮点为内部统一表示，处理链路为：

```
事件清单 JSON
   │  manifest.py   字段校验 + 全局毫秒时间轴
   ▼
逐事件（单事件失败只记录错误码并跳过，不影响其它事件）
   ├─ decoder.py   FFmpeg + soxr(precision=28) 解码重采样到 48k 立体声 f32
   ├─ loudness.py  BS.1770-4 K 加权积分响度 + 4x 过采样真峰值
   ├─ builder.py   偏移 / 循环 / 裁剪 / 静音填充 / 余弦淡化
   └─ 分类目标响度增益（dialog/sfx/music）
   ▼
mixer.py      时间轴放置 + 重叠优先级仲裁（replace / mix）
   ▼
pipeline.py   母带综合归一化到 -16 LUFS + 真峰值限制 -1 dBTP
   ▼
writer.py     确定性 24-bit PCM WAV + JSON 报告
```

## 2. 关键算法

### 2.1 解码与重采样（decoder.py）

仅接受 `.wav` / `.flac`，其它后缀在清单校验阶段直接给 `E_FORMAT_UNSUPPORTED`。
解码命令通过 `aresample=resampler=soxr:precision=28` 使用 SoX 高精度重采样器，
`osf=fltp` 输出浮点、`out_chlayout=stereo` 统一立体声，单声道会被上混为双声道。
FFmpeg 进程返回非零或无输出时映射为 `E_DECODE_FAILED`（损坏文件同样落入此码）。
同一素材在一次运行内只解码一次（按绝对路径缓存）。

### 2.2 响度测量（loudness.py）

K 加权滤波严格采用 ITU-R BS.1770-4 的两级双二阶级联（高频架 + 高通）。
为避免纯 Python 递归滤波过慢，冲激响应截断为 4096 抽头，通过 8192 点 FFT
overlap-save 分块卷积实现，长素材也是 O(N log N)。

积分响度使用 400 ms 测量块、75% 重叠（hop=100 ms），每块计算双声道加权均方功率，
再施加双门限：绝对门限 -70 LUFS，相对门限为未门限响度 -10 LU。
**短事件回退**：素材短于一个 400 ms 测量块时，退化为对整段做单次无门限测量，
报告中以 `short_event_fallback=true` 标注；纯静音（无任何有效功率）返回 `None`，
事件记录 `E_LOUDNESS_FAILED` 并跳过。

### 2.3 真峰值（loudness.py）

采用 4 倍零填充 FFT 频谱插值（BS.1770 推荐的过采样思路）。为避免分块边界
引入人为不连续，每块两侧加零填充、块间 50% 重叠，且只取每块中央无边界区域
的过采样结果取最大值。dBTP = 20·log10(max|upsampled|)。与 FFmpeg `ebur128`
在常规频段读数差 < 0.2 dB，高频（17 kHz）差约 0.35 dB。

### 2.4 循环与淡化（fades.py / builder.py）

* 余弦窗采用半样本相位 `g(k)=0.5-0.5·cos(π(k+0.5)/N)`，首尾不产生精确 0/1
  的边界突变；交叉淡化两端增益严格互补，恒定信号过循环点保持恒定。
* 循环素材按 `事件时长 - 素材内偏移` 计算所需采样数，循环点用 10 ms
  等增益余弦交叉淡化拼接，循环长度不足时自动重复。
* 非循环素材超出事件时长则裁剪，并在尾部做余弦淡化消除爆音；素材不足则
  静音填充并在报告 `padded_silence=true` 标注。
* 淡化时长全部可配置并带合法范围校验（见 config.py 的 `RANGES`）。

### 2.5 时间轴与重叠（mixer.py）

所有时间以毫秒声明、以采样落位（`ms_to_samples` 四舍五入，误差 ≤ 1/48 ms）。
每个 replace 事件对重叠采样点按 `priority` 仲裁归属：高优先级覆盖低优先级；
优先级相同则保留先到事件（first-come-first-served）并写
`E_WARN_EQUAL_PRIORITY` 警告。`overlap="mix"` 的事件始终以加法混入。
最终时间轴长度严格等于清单 `total_duration_ms` 对应的采样数，不足部分保持静音。

### 2.6 母带归一化与真峰值限制（pipeline.py）

先对完整混音测积分响度并施加增益到 -16 LUFS（增益受 `max_gain_db` 钳制），
再测真峰值；若超过 -1 dBTP，则按 `10^((-1-TP)/20)` 整体衰减。注意限制器
基于**真峰值**而非采样峰值，因此与 FFmpeg 读数一致，不会出现自报达标却削波。

### 2.7 确定性

* 所有随机源仅存在于样本生成器（固定种子），处理链路无随机数、无抖动。
* 24-bit 量化使用 `np.rint`（就近偶数）后小端三字节打包，混音按固定事件顺序累加。
* JSON 报告以 `sort_keys=True` 固定键序，不含运行时间戳，故同输入多次运行
  WAV 与报告均字节一致（已用 `cmp` 验证）。

## 3. 错误码

全部以 `E_` 前缀定义在 errors.py，并在每份报告的 `error_catalog` 中统一列出。
致命级（清单/配置）导致非零退出码；事件级只跳过该事件。

| 代码 | 级别 | 触发场景 |
| --- | --- | --- |
| E_MANIFEST_PARSE | fatal | 清单不是合法 JSON / 顶层非对象 / 文件不存在 |
| E_MANIFEST_FIELD | fatal | 缺 events 或 total_duration_ms |
| E_DURATION_INVALID | fatal | 总时长缺失、非正数或非数值 |
| E_CONFIG_INVALID | fatal | 配置参数越界、未知键、类型错误 |
| E_FIELD_MISSING | event | 缺 id/path/start_ms |
| E_TIME_INVALID | event | 负时间、非数值、偏移超出素材 |
| E_FILE_NOT_FOUND | event | 引用文件不存在 |
| E_FORMAT_UNSUPPORTED | event | 非 WAV/FLAC 后缀 |
| E_DECODE_FAILED | event | FFmpeg 解码失败 / 文件损坏 |
| E_LOUDNESS_FAILED | event | 静音或测量无有效功率 |

## 4. 性能

10 分钟、499 个事件（含循环/重叠/失败事件）在本机（24 核 Linux，
FFmpeg 8.0.1）实测约 37–50 秒、峰值驻留约 2.4 GB，满足 4 分钟预算。
主要开销为逐素材 FFmpeg 解码与 FFT 卷积/过采样，均为线性或 N log N。
