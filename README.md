# 技术播客离线后期工具

这是一个可离线、确定性运行的多轨播客后期工具。输入为 JSON 剪辑清单和
48000 Hz、24-bit PCM WAV；输出为 16-bit PCM 单集、章节 JSON 与处理报告
JSON。所有时间点均为 48 kHz 采样点整数，浮点混音内部使用 `float64`。

## 快速运行

Windows 10/11（PowerShell）：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python generate_demo.py demo
.\.venv\Scripts\python postproduce.py --manifest demo\manifest.json --output out
```

Linux/macOS：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python generate_demo.py demo
.venv/bin/python postproduce.py --manifest demo/manifest.json --output out
```

若发行版 Python 缺少 `ensurepip`，可先用 `uv venv --python 3.11 .venv` 创建
环境，再用 `uv pip install --python .venv/bin/python -r requirements.txt` 安装依赖。

成功时生成：

- `out/episode.wav`：48 kHz、双声道、16-bit PCM。
- `out/chapters.json`：章节起止采样点、采样点长度与秒数。
- `out/report.json`：轨道峰值、淡化诊断、闪避触发、章节调整、响度与真峰值。

`generate_demo.py` 不读取外部附件，使用固定随机种子和固定合成参数生成三个
24-bit WAV 及清单。重复运行得到相同素材。

## 清单结构

完整示例由 `generate_demo.py` 写出。关键结构如下：

```json
{
  "version": 1,
  "sample_rate": 48000,
  "channels": 2,
  "bit_depth": 24,
  "total_samples": 864000,
  "tracks": [
    {"name": "host", "role": "host", "asset": "assets/host.wav", "gain_db": 0.0},
    {"name": "guest", "role": "guest", "asset": "assets/guest.wav", "gain_db": -1.5},
    {"name": "music", "role": "music", "asset": "assets/music.wav", "gain_db": -9.0}
  ],
  "clips": [
    {
      "id": "h0",
      "track": "host",
      "asset": "assets/host.wav",
      "source_in": 0,
      "source_out": 144000,
      "timeline_in": 0,
      "timeline_out": 144000,
      "gain_db": 0.0
    }
  ],
  "edit_points": [
    {
      "id": "h_cut_1",
      "track": "host",
      "outgoing_clip": "h0",
      "incoming_clip": "h1",
      "at_sample": 144000,
      "fade_samples": 4800
    }
  ],
  "chapters": [
    {"title": "开场", "start_sample": 0}
  ],
  "processing": {
    "fade_samples": 4800,
    "ducking": {
      "enabled": true,
      "trigger_track": "host",
      "target_track": "music",
      "threshold_dbfs": -28.0,
      "energy_window_ms": 20.0,
      "depth_db": 10.0,
      "attack_ms": 15.0,
      "release_ms": 220.0
    },
    "loudness": {
      "target_lufs": -16.0,
      "max_true_peak_dbtp": -1.0
    }
  }
}
```

约束与语义：

- 素材路径相对清单目录解析；必须是 48 kHz、24-bit、单声道或双声道的整数
  PCM WAV。每个片段的 `asset` 必须等于其轨道的 `asset`。
- `source_out - source_in` 必须等于 `timeline_out - timeline_in`，且不得越过
  素材末尾或清单声明总长度。
- 普通同轨重叠按 `clips` 顺序后写覆盖；显式 `edit_points` 在同一区域合并指定
  的前后两个片段。交叉淡化也不会覆盖其后写入的同轨片段。
- 淡化区间为 `[at_sample - effective_fade, at_sample)`。请求时长超过两片段
  实际可用重叠时截断，并累计 `crossfades.truncated_count`。
- `chapters[0].start_sample` 必须为 0。章节若落在淡化区间内，起点向后移动到
  淡化终点；输出章节按采样点首尾严格衔接，末章结束于 `total_samples`。

## 处理算法

1. **时间轴**：每个轨道先建立 `total_samples × channels` 的 `float64` 缓冲；
   片段增益为轨道增益与片段增益之和。不同轨道逐采样相加。
2. **等功率淡化**：长度为 `N` 的重叠使用
   `cos(θ)`/`sin(θ)`，`θ` 在 `[0, π/2]` 上按采样点等间隔。
   两侧平方和恒为 1；`N=1` 时两侧均为 `sqrt(0.5)`。
3. **侧链闪避**：在主持人轨上计算逐采样因果短时均方能量；超过阈值时目标为
   `depth_db`，否则为 0。衰减量用一阶平滑：
   `α = 1 - exp(-1 / tau_samples)`，`tau_samples = ms × 48000 / 1000`。
   攻击、释放使用各自时间常数；报告中的触发次数为越过阈值的上升沿数。
4. **章节**：先做淡化后移，再以调整后的章节起点构造连续半开区间
   `[start, next_start)`，末章截止到总采样点。
5. **BS.1770 响度**：48 kHz K 加权预滤波与 RLB 高通，400 ms 块、100 ms
   步进，绝对门限 -70 LUFS，相对门限比绝对门限后的均值低 10 LU；积分按
   BS.1770 块能量计算。
6. **真峰值**：对每个声道局部绝对峰值附近做 4 倍窗函数 sinc 插值，报告
   原始样本与插值结果中的最大绝对幅度。
7. **归一化与量化**：计算达到 `target_lufs` 所需增益和达到
   `max_true_peak_dbtp` 允许的最大增益，取较小值。量化成 16-bit 后再次
   测量真峰值；若量化造成越限则回退增益。峰值约束优先，报告中写入实际积分
   响度、实际真峰值和是否 `peak_limited`。

报告同时提供便于验收的顶层字段：
`track_peaks`、`track_peaks_after_normalization`、`ducking_trigger_count`、
`chapter_alignment_adjustment_count`、`final_integrated_lufs`、
`final_true_peak_dbtp`。

## 校验错误

错误只写 stderr，格式为 `错误码: 说明`，CLI 返回非零退出码。清单、素材、
淡化和章节全部在创建输出目录之前完成校验，因此业务校验失败不会产生部分输出。

| 错误码 | 退出码 | 含义 |
| --- | ---: | --- |
| `E_SCHEMA` | 2 | JSON 结构、类型、引用或参数语义错误 |
| `E_ASSET_MISSING` | 3 | 清单或素材不存在/不可读 |
| `E_FORMAT_UNSUPPORTED` | 4 | 非 PCM WAV、采样率/位深/声道不符 |
| `E_CLIP_RANGE` | 5 | 入出点、时间轴、淡化或总长度越界 |
| `E_CHAPTER_RANGE` | 6 | 章节时间点非法、乱序或首章不为 0 |
| `E_SILENCE` | 7 | 音频过短或低于 BS.1770 门限 |
| `E_PROCESSING` | 8 | 输出写入或其他运行时错误 |

## 验证

```bash
.venv/bin/python -m unittest discover -v
```

本环境实际验证结果（Python 3.14.4、NumPy 2.5.4、FFmpeg 8.0.1）：

- 17 个测试全部通过，覆盖等功率包络、同轨覆盖、跨轨求和、淡化截断、闪避
  上升沿与一阶平滑、章节后移、响度目标、峰值优先、真实样本生成、CLI
  错误码、无部分输出和字节级确定性。
- 演示集输出：864000 采样点（18.000 s）、双声道、16-bit，数据区
  3456000 字节；RIFF 头采样率 48000、byte rate 192000、block align 4。
- 10 条淡化，1 条按要求截断；闪避触发 13 次；章节对齐调整 3 次。
- 工具测得最终积分响度 `-15.999999 LUFS`，真峰值 `-6.675556 dBTP`。
  FFmpeg `ebur128=peak=true` 独立测得 `-16.0 LUFS`、`-6.7 dBFS`。
- 两次独立运行输出的 SHA-256：

```text
77d828b988c62e9cebad79a57bde8294de00c734f0fb0e87abaae8a40d29f16a  episode.wav
edd76b91146205112b3b0c42fce4229a88186fb56b2f4d9eb79b9bb80e28f139  chapters.json
2f72fed0d5c71da688d77d466023f0feb46a637deb89dfd8759c51234ed41f89  report.json
```

运行期不需要网络、账号或 Docker；FFmpeg 只用于测试中的独立响度复核，核心
渲染不依赖 FFmpeg。
