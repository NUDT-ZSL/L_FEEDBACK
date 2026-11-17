# 节奏动作游戏离线音频处理引擎

读取本地 WAV 分层素材与技能音效，依据技能触发时间轴与逐帧连击状态，
生成 BPM 对齐、背景音乐自动闪避、多轨渐变混合的立体声输出，并产出
可复核的 JSON 质量报告。纯离线、确定性：同一输入重复运行产物字节一致。

## 环境

- Python 3.10+（本机验证：Python 3.14.4）
- `pip install numpy soundfile`（本机验证：numpy 2.5.5 / soundfile 0.14.0）

## 快速开始

```bash
python3 tools/generate_samples.py assets          # 生成确定性 WAV 素材 + config.json
python3 main.py --config assets/config.json \
    --output out/mix.wav --report out/report.json # 运行引擎
python3 tests/run_acceptance.py                   # 27 项验收检查
```

## 模块结构

| 模块 | 职责 |
| --- | --- |
| `audio_engine/loader.py` | 读取 16-bit PCM WAV（单/立体声），Kaiser 窗 sinc 带限插值重采样至 48 kHz，保留原声道数 |
| `audio_engine/quantize.py` | 按首拍偏移 + BPM 将触发时间吸附到 1/4、1/8、1/16 网格最近点，输出量化误差 |
| `audio_engine/ducking.py` | 音效区间内 BGM 20 ms 线性降至 -6 dB、结束后 100 ms 恢复；多区间取最小增益；包络无单样本跳变 |
| `audio_engine/looping.py` | BGM 不足时按整小节（4 拍）循环延展，接缝 10 ms 等长线性交叉淡化 |
| `audio_engine/layers.py` | 连击 0-4 / 5-9 / 10+ 映射基础层 / +打击层 / +氛围层，增删 50 ms 内完成（因果线性渐变） |
| `audio_engine/sfx.py` | 音效按量化时间叠加（重叠相加而非覆盖）、独立增益 + 等功率声像；超限时 tanh 软限幅 |
| `audio_engine/engine.py` | 流水线编排、峰值上限 -1 dBFS、JSON 报告、原子写文件 |
| `tools/generate_samples.py` | 固定种子生成全部测试素材与时间轴 |
| `main.py` | CLI 入口，业务错误码写 stderr 并以非零码退出 |

## 配置格式（assets/config.json）

- `bpm` / `first_beat_offset`：节拍网格参数（BPM ≤ 0 报 `ERR_INVALID_BPM`）
- `duck_gain_db`：闪避目标增益（默认 -6 dB）
- `layers`：`base` / `percussion` / `ambience` 三层 WAV 路径
- `sfx`：音效 id → WAV 路径
- `combo_fps` / `combo_states`：逐帧连击状态
- `events`：`{time, sfx, gain_db, pan, grid}`，grid 为 `1/4`、`1/8`、`1/16`

## 质量报告字段（report.json）

- `quantization.events[]`：每个音效的量化误差（ms）与量化后时间
- `ducking.max_slope_db_per_s`：闪避曲线最大斜率
- `mix.peak_dbfs`：混音峰值电平（保证 ≤ -1 dBFS）
- `loop.count` / `loop.per_layer`：循环延展次数
- `output.duration_s`：总输出时长（覆盖最后触发点 + 2 s）

所有数值保留两位小数；同一输入重复运行报告与 WAV 字节一致。

## 错误码与失败原子性

| 错误码 | 触发条件 |
| --- | --- |
| `ERR_MISSING_FILE` | 引用的 WAV / 配置文件缺失 |
| `ERR_BAD_SAMPLE_RATE` | 采样率非 44100 / 48000 |
| `ERR_INVALID_BPM` | BPM ≤ 0 或时间轴为空 |

错误写 stderr、退出码 2；所有输出先写临时文件再 `os.replace`，
失败时不留下任何部分输出。

## 实测验证结果（tests/run_acceptance.py，27/27 通过）

- 样本生成确定性：两次生成 7 个文件 SHA-256 全等
- 重采样：5 kHz 带内幅度误差 0.01%；镜像抑制比 0.00006；23 kHz 混叠残余 0.003
- 量化：12 个事件最大误差 3.88 ms（≤ 5 ms）
- 闪避：深度 -6 dB、20 ms 攻 / 100 ms 放、包络最大单样本步进 0.00052（无跳变）
- 循环：整小节延展、接缝最大相邻样本差 0.00052、循环次数正确上报
- 分层：50 ms 内完成增删；混音峰值 -2.08 dBFS（≤ -1 dBFS）
- 音效：5 ms 间隔重叠正确叠加；软限幅峰值 0.871 < 0.891（-1 dBFS）
- 端到端：输出 48 kHz 立体声 10.05 s；报告与 WAV 两次运行字节一致
- 异常：三类错误码均 stderr 输出、退出码 2、无残留输出文件
