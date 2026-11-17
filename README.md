# audiotl — 确定性音频时间轴构建工具

将游戏语音对白、环境音效、过场音乐等 WAV PCM 片段按 JSON 时间轴清单拼接混音，
输出统一采样率/声道数的混音 WAV、单帧精度的双向时间戳映射、逐样本一致的波形摘要
以及可追溯的 JSON 渲染报告。渲染分为 analyze → resample → mix → writeout 四个可重入
阶段，基于内容哈希增量转码，支持多进程并发共享缓存。

## 环境

- Windows 10+ / Linux，Python 3.11+，NumPy 1.26+
- 不需要 FFmpeg（测试素材由 `audiotl/samplegen.py` 确定性生成）

## 快速开始

```bash
# 生成确定性示例素材与清单（多采样率/多声道）
python tools/make_samples.py example

# 构建时间轴（输出 mix.wav / mapping.json / summary.json / report.json）
python -m audiotl example/manifest.json example/out --workers 4

# 运行测试
python -m pytest tests/
```

退出码：`0` 成功；`1` 内部错误；`2` 清单非法；`3` 输入缺失；`4` 渲染业务错误
（淡化/重叠/格式）；`5` 阶段版本冲突。所有业务错误同时写入 `report.json` 的
`errors[].code`。

## 清单格式

```json
{
  "version": 1,
  "output": {"sample_rate": 48000, "channels": 2, "window_ms": 100},
  "segments": [
    {"id": "voice", "path": "samples/voice.wav",
     "fade_in":  {"length_ms": 0,   "curve": "linear"},
     "fade_out": {"length_ms": 250, "curve": "equal_power"}}
  ]
}
```

- `path` 相对清单文件解析；支持 44.1kHz/48kHz、单声道/立体声 PCM 输入。
- 片段 i 与 i+1 的交叉淡化重叠长度 = 片段 i 的 `fade_out`（换算为输出帧）；
  下一片段的 `fade_in` 独立作用于其头部，对称交叉淡化时两者设为相同长度。
- 淡化曲线边界在采样层明确定义：`t = i/(n-1)`，`gains[0]=0.0`、`gains[-1]=1.0`
  （`n==1` 时增益为 1.0）；`linear` 为 `t`，`equal_power` 为 `sin(t·π/2)`。

## 确定性规则

- 重采样：线性插值，输出帧 `j` 映射到输入位置 `j·in_rate/out_rate`（float64），
  输出长度 `(n·out_rate + in_rate//2) // in_rate`，无库默认/随机行为。
- 混音：按清单顺序将各片段 float32 数据累加进 float64 缓冲，操作顺序固定。
- 量化：`rint(clip(x)·32767)`（round-half-even），输出 16-bit PCM。
- 映射位置全部由整数帧计数推导，重采样与声道转换后仍精确到单帧。

## 增量与并发

- 片段缓存键 = SHA-256(源文件字节 + 渲染参数 + 引擎/阶段版本)，缓存文件按内容寻址。
- 所有缓存/状态/产物文件均以“临时文件 + `os.replace`”原子发布，读者不会看到半写文件。
- 共享状态索引 `state.json` 在跨进程锁文件（`O_EXCL` 原子创建）下合并更新。
- 状态文件损坏或版本不符时安全回退为全量渲染；缓存片段损坏时仅重渲染该片段。
- 阶段产物（`_stages/analysis.json`、`resample.json`、`mix.meta.json`）携带阶段版本，
  版本或输入哈希变化时下游自动重算，不沿用旧偏移或旧统计。
- 波形摘要始终从最终量化后的 int16 样本重算，增量复用下也不会沿用旧窗口。
- `--workers N` 受本机逻辑核数约束（`min(N, os.cpu_count())`）。

## 输出物

| 文件 | 说明 |
| --- | --- |
| `mix.wav` | 混音输出（清单指定的采样率/声道，16-bit PCM） |
| `mapping.json` | 每片段输出起始帧/输入起始帧/有效帧数/重叠帧数，支持按输出毫秒查询 |
| `summary.json` | 按 `window_ms` 窗口统计每声道峰值与 RMS，末尾不足窗口按实际长度 |
| `report.json` | 状态、业务错误码、缓存命中、阶段版本与各阶段耗时 |

## 实测验证结果（本机，Python 3.14 + NumPy 2.5.5）

- `python -m pytest tests/`：28 项全部通过（DSP、流水线、增量、阶段版本失效、
  并发、非法输入错误码）。
- 示例清单（44.1k 单声道 + 48k 立体声 + 44.1k 立体声 → 48kHz 立体声）：
  全量渲染与增量重渲染 `mix.wav` SHA-256 完全一致
  （`aff0ef7b…f6ad`），二次运行 3/3 缓存命中、mix 阶段复用。
- 修改单个源片段后仅该片段重渲染（misses=1），输出与全新状态目录全量渲染字节一致。
- 同一清单、同一状态目录并发启动两个进程：均退出码 0，输出与顺序渲染字节一致。
- 错误路径实测：负淡化 → 退出码 4 / `E_FADE_NEGATIVE`；清单缺失 → 退出码 3 /
  `E_INPUT_MISSING`；淡化超长 → `E_FADE_TOO_LONG`；重叠过多 → `E_OVERLAP_EXCESSIVE`；
  清单 JSON 非法 → 退出码 2 / `E_MANIFEST_INVALID`。
