# 验证记录（本地确定性样本实测）

验证环境：Linux 容器（代码仅使用 Python 3.10+ 通用语法，目标 Windows 10/11
+ Python 3.10/3.11 同样适用）、Python 3.14、numpy 2.5.4、soundfile 0.14.0、
pyloudnorm、libsndfile、FFmpeg 8.0.1（在 PATH）。样本由 `tests/samples.py`
以确定性函数生成（正弦/静音/固定种子噪声），无外部附件。

复现命令：

```bash
python -m pip install -r requirements.txt
python -m pytest                 # 单元 + 集成测试
python scripts/acceptance_demo.py # 端到端验收演示（打印下文证据）
```

## 1. 单元/集成测试

`python -m pytest` 实测：**35 passed**。覆盖：

- sinc 重采样长度规则、脉冲对齐 ±1 采样、上/下采样往返相关性 >0.99
- 毫秒换算 `floor(round(ms*sr/1000))` 多组参数
- timeline 三类错误码（`E_TIMELINE_FORMAT`/`E_RANGE`/`E_MISSING_CLIP`）、
  bool 不得充当 priority、缺省值
- 优先级覆盖与同优先级后出现者覆盖（含实际音频内容替换断言）
- 循环平铺、非循环补静音（`W_SHORT_CLIP` 与补静音采样数）
- 响度命中 -23 LUFS（±0.5）、低响度封顶 +20 dB 与 `W_LOW_LOUDNESS`
- 损坏文件跳过续跑、退出码 1、其余 6 个事件正常渲染
- 两次运行 WAV/报告 sha256 完全一致（字节级确定性）
- 512 MiB 内存限额触发 `E_MEMORY_LIMIT` 且无部分输出残留
- CLI stderr 单行 JSON 错误契约与非零退出码
- 增益自动化曲线最小 100 ms 过渡、单调、有限

## 2. 端到端验收（acceptance_demo 实测摘要）

场景含 7 个事件：44.1 kHz/16-bit、32 kHz/24-bit、48 kHz/32-bit 浮点立体声、
短素材、循环素材、损坏 WAV、极安静素材；3 处区间重叠。

- 退出码：两次运行均为 `1`（存在一个 `E_CORRUPT_CLIP`），stderr 为单行 JSON。
- 输出格式：`48000 Hz, 2 channels, PCM_24, WAV`。
- 确定性：两次运行 WAV sha256 相同
  （`818f6df6f7cb4784a011a361c79422d0e58748ef69fcb539af9ac9505f75e388`），
  报告 sha256 相同
  （`058a0f7126ba0423cdb789db3fa3b7b980d8d3073cc871dac5847572de9ff69f`）。
- 输出帧数 `172800 == round(3600 ms * 48000 / 1000)`，与报告 `output_frames` 一致。
- 响度：measured `-10.070868` LUFS，applied gain `-12.929132` dB，
  output `-23.0` LUFS（误差 0.0 LU）。
- 重叠记录（采样点）：
  - 事件 1 覆盖事件 0，区间 `[24000, 43200)`，规则 `higher_priority`；
  - 事件 2 覆盖事件 0，区间 `[33600, 48000)`，规则 `higher_priority`；
  - 事件 2 覆盖事件 1（同优先级），区间 `[33600, 43200)`，规则 `later_index_on_tie`。
- 补静音：事件 3 源 14400 帧、需求 33600 帧、补静音 19200 帧，代码实测该区域全零。
- 循环：事件 4（600 ms 事件 / 250 ms 素材）整段有声，循环接缝无额外淡化。
- 报告中每个事件起止采样点用固定换算规则复核一致，且均落在输出索引范围内。

## 3. 错误/警告探针实测

| 探针 | stderr / 报告 | 退出码 |
| --- | --- | --- |
| `end_ms < start_ms` | `{"error":{"code":"E_RANGE",...}}` | 1 |
| `priority: true` | `{"error":{"code":"E_TIMELINE_FORMAT",...}}` | 1 |
| 缺失片段文件 | `E_MISSING_CLIP`（pytest 覆盖） | 1 |
| 损坏 WAV | 事件级 `E_CORRUPT_CLIP`，其他事件继续 | 1 |
| 1 MiB 内存上限 | `E_MEMORY_LIMIT`，WAV/报告均未残留 | 1 |
| 极安静素材 | `W_LOW_LOUDNESS`，增益封顶 `+20.0 dB`，`gain_boost_capped=true` | 0 |
| 素材不足且不循环 | `W_SHORT_CLIP`（含 padding 帧数） | 0 |

## 4. 已知边界说明

- 整体响度归一化采用单一恒定增益（自动化曲线的零变化特例），保证输出稳定命中
  -23 LUFS；`gain_envelope` 的 ≥100 ms 平滑能力由独立单测覆盖，可用于多增益点场景。
- 静音/不可测响度在报告中以 `null` 表示（JSON 不支持 Infinity）。
- 核心重采样为内置窗口 sinc，实现无外部进程依赖；FFmpeg 仅作为环境侧配套工具。
