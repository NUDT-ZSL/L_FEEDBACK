# 游戏本地化多语言口型对齐工具（lipsync_tool）

离线工具：读取动画元数据（关键帧时间戳 + 音素口型事件）与多语言配音 WAV，
完成重采样/单声道归一化、WSOLA 时间伸缩、能量 + 动态规划口型对齐，并把无音轨
MP4 与伸缩后音频重新封装为带 AAC 音轨的 MP4，同时输出每语言对齐口型 JSON 与
汇总处理报告。所有样本均由代码确定性构造，不访问任何外部网络目标。

## 1. 环境与安装（Windows）

1. 安装 Python 3.9+（开发验证使用 3.14，代码语法兼容 3.9+）。
2. 安装 FFmpeg 5.0+，将 `ffmpeg.exe` / `ffprobe.exe` 所在目录加入系统 `PATH`
   （`where ffmpeg` 与 `where ffprobe` 能找到即可；验证环境为 FFmpeg 8.0.1）。
3. 安装依赖：

   ```bat
   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install -r requirements.txt
   ```

   NumPy 为必需依赖；SciPy 可选，缺失时自动切换到内置 NumPy 线性插值重采样器。
4. 不使用 Docker 或任何容器环境。

Linux/macOS 下把激活命令换成 `source .venv/bin/activate`，其余流程一致。

## 2. 快速开始

```bash
# 1) 确定性构造样本（元数据 JSON、无声 MP4、en/ja/zh 三个 WAV）
python -m lipsync_tool.cli make-samples --output-dir sample_data

# 2) 处理全部语言
python -m lipsync_tool.cli process \
  --metadata sample_data/animation_metadata.json \
  --audio-dir sample_data/audio \
  --output-dir output \
  --languages en,ja,zh \
  --video sample_data/video/cutscene_silent.mp4 \
  --config config.example.json

# 或一条命令完成两步
python run_demo.py --workdir demo_workdir
```

CLI 参数：

| 参数 | 说明 |
| --- | --- |
| `--metadata` | 动画元数据 JSON（帧率、总时长、目标采样率、口型事件） |
| `--audio-dir` | 配音 WAV 目录，按 `dub_<lang>.wav` / `<lang>.wav` 等模式查找 |
| `--output-dir` | 输出目录（MP4、伸缩 WAV、口型 JSON、报告） |
| `--languages` | 逗号分隔语言代码，如 `en,ja,zh` |
| `--video` | 无音轨源 MP4，缺省尝试 `../video/cutscene_silent.mp4` |
| `--config` | 可选 JSON 配置；文件缺失或字段非法时回退默认值并写入报告 |

## 3. 模块结构

| 模块 | 职责 |
| --- | --- |
| `lipsync_tool/config.py` | 配置模型、字段校验与默认值回退（含回退原因记录） |
| `lipsync_tool/wavio.py` | PCM WAV 读写（8/16/24/32-bit）、重采样、单声道混合 |
| `lipsync_tool/wsola.py` | WSOLA 时间伸缩与伸缩因子边界截断 |
| `lipsync_tool/align.py` | 能量包络、静音检测、帧→时间映射、DP 对齐、覆盖率 |
| `lipsync_tool/mux.py` | FFmpeg 封装（视频 copy、音频 AAC、PTS 重建）与探测 |
| `lipsync_tool/samples.py` | 确定性样本构造（元数据、testsrc 无声视频、三语 WAV） |
| `lipsync_tool/pipeline.py` | 单语言编排、异常恢复、报告聚合 |
| `lipsync_tool/cli.py` | `make-samples` / `process` 命令行接口 |

## 4. 算法说明

### 4.1 音频归一化

- 多声道按声道均值混合为单声道；采样率与元数据 `target_sample_rate` 不一致时
  重采样（SciPy 多相滤波优先，否则线性插值），并在报告 `notes` 中记录实际动作。
- 输出 WAV 保持声明采样率与原始位深（样本为 32 kHz / 16-bit 单声道）。

### 4.2 WSOLA 时间伸缩

- 伸缩因子 = 动画时长 / 配音时长，限制在 `[stretch_min, stretch_max]`
  （默认 0.75–1.25），越界截断并产生警告。
- 32 ms 汉宁窗、50% 重叠；参考帧锚定自然分析时间线，在 ±8 ms 邻域做归一化
  互相关相似度搜索，避免参考帧跟随“接受位置”导致的相位漂移与变调。
- 因子为 1.0 时走精确拷贝快路径；输出长度严格等于 `输入帧数 × 伸缩因子`。

### 4.3 口型事件对齐

1. 事件先按帧率从帧号映射到秒级音频时间轴。
2. 对每个事件在 ±`align_tolerance_ms`（默认 50 ms）范围内按 5 ms 网格生成候选
   位置，计算汉宁窗 RMS 能量失配代价。
3. 动态规划联合最小化：能量失配 + 绝对偏移正则 + 相邻事件重叠惩罚
   （权重 `overlap_penalty_weight`），回溯得到全局最优对齐时间戳。
4. 输出每个事件的 `start_time/end_time/offset_ms`，并保证偏移不超过容差。

### 4.4 静音与覆盖诊断

- 在伸缩后音频上检测能量低于 `silence_threshold_db`（默认 -45 dB）且持续超过
  `silence_min_ms`（默认 500 ms）的静音段，写入 `silence_markers` 与报告警告，
  不中断流程。
- 事件合并覆盖率低于 `coverage_min_ratio`（默认 0.5）时输出警告并补充
  `uncovered_gap_markers`。

### 4.5 MP4 封装与失败恢复

- `ffmpeg -fflags +genpts` 重建时间戳，`-c:v copy` 直拷视频流，`-c:a aac`
  编码音频，`-async 1` 保证起始对齐；封装后用 ffprobe 校验双流与时长。
- 单语言封装失败（含配置 `simulate_mux_failure` 测试钩子、ffmpeg 缺失、输入
  丢失等）时记录错误，状态置为 `mux_fallback`，保留伸缩 WAV + 对齐 JSON 继续
  处理后续语言，并在报告 `mux_fallback_segments` / `failed_segments` 汇总。

## 5. 输出文件

每语言（以 `zh` 为例）：

- `cutscene_zh.mp4`：视频流直拷 + AAC 音轨的成片。
- `dub_zh_stretched.wav`：伸缩后单声道 WAV（采样率/位深与声明一致）。
- `lipsync_zh.json`：对齐口型事件（含 `offset_ms`）、静音标记、未覆盖间隙标记。

全局：

- `processing_report.json`：配置与回退说明、每语言伸缩因子（请求值/实际值）、
  平均/最大偏移、事件数量、覆盖率、警告与错误、失败片段汇总。
  报告中的 `event_count` 与对应 `lipsync_<lang>.json` 的 `event_count` 及
  `lip_events` 长度严格一致（有自动化测试守护）。

## 6. 确定性样本设计

| 语言 | 输入 WAV 特征 | 配音时长 | 请求/实际伸缩因子 | 覆盖的验收点 |
| --- | --- | --- | --- | --- |
| en | 22050 Hz、立体声、16-bit | 4.4 s | 1.3636 → 截断 1.25 | 重采样 + 下混、上界截断、700 ms 静音段 |
| ja | 48000 Hz、单声道、16-bit | 8.4 s | 0.7143 → 截断 0.75 | 重采样、下界截断 |
| zh | 44100 Hz、单声道、16-bit | 6.0 s | 1.0 | 常规重采样、无截断基线 |

所有 WAV 由固定种子的 NumPy 生成器与音素事件时间上的确定性抖动合成；视频由
FFmpeg `testsrc` 生成 6.0 s / 24 fps / 144 帧的 H.264 无声 MP4。

## 7. 实测验证结果

验证环境：Python 3.14.4、NumPy 2.5.5、SciPy 1.18.1、FFmpeg 8.0.1（Linux 容器
外本机执行；Windows 上命令与路径规则见第 1 节）。

`python -m unittest discover -s tests`：**22 个测试全部通过**，覆盖

- WAV 8/16/24/32-bit 读写往返、立体声下混、两种重采样后端时长保持；
- WSOLA 单位因子精确拷贝、输出长度精度、音高保持、边界截断；
- 帧→时间映射、能量峰值对齐、50 ms 容差约束、静音检测、覆盖率与间隙；
- 配置缺失/非法/上下界倒置的默认值回退与说明；
- 端到端三语言处理、报告与口型 JSON 事件数一致性、MP4 双流探测、
  伸缩上下界截断、英文静音标记、封装失败回退、缺失语言不中断、位深保持、
  低覆盖间隙标记。

一次真实端到端运行的报告摘要：

| 语言 | 状态 | 实际伸缩 | 平均偏移 | 最大偏移 | 事件数 | 警告 |
| --- | --- | --- | --- | --- | --- | --- |
| en | ok | 1.25（请求 1.3636，截断） | 4.55 ms | 40 ms | 11 | 截断 + 2 段静音标记 |
| ja | ok | 0.75（请求 0.7143，截断） | 1.36 ms | 5 ms | 11 | 截断 |
| zh | ok | 1.0 | 3.64 ms | 40 ms | 11 | 无 |

MP4 校验：视频流 `h264` 直拷、时长 6.000 s、144 帧、PTS 从 0 连续；音频流
`aac`、起始时间 0（AAC 首包负 PTS 为编码器原生延迟，由 edit list 补偿）。
截断场景下音频为 5.5 s / 6.3 s 属边界保护的预期结果，视频仍保持 6.0 s 且
A/V 起点对齐。

## 8. 需求条目映射

| 业务要求 | 实现位置 |
| --- | --- |
| 1. 加载/校验/重采样/单声道 + 帧→音频时间映射 | `wavio.normalize_to_mono`、`align.map_events_to_audio_time` |
| 2. WSOLA 伸缩、0.75–1.25 截断警告、采样率位深不变 | `wsola.wsola_time_stretch`、`wsola.clamp_stretch_factor`、`pipeline` |
| 3. ±50 ms 能量搜索 + DP 偏移/重叠最小化 | `align.align_events` |
| 4. 视频 copy + AAC 封装、PTS 连续、失败记录并继续 | `mux.remux_mp4`、`pipeline.run_pipeline` |
| 5. 多语言独立 MP4/JSON、按语言代码命名 | `pipeline.process_language` |
| 6. >500 ms 静音/覆盖不足警告与标记、封装失败回退 WAV+JSON | `align.detect_silence`、`align.uncovered_segments`、`pipeline` |
| 7. CLI + 配置文件（伸缩界/容差/重叠权重），非法回退并写入报告 | `cli.py`、`config.load_config` |
| 8. 报告含对齐统计/伸缩因子/警告，事件数与口型 JSON 一致 | `pipeline.run_pipeline` + `tests/test_pipeline_e2e.py` |

## 9. 测试

```bash
python -m unittest discover -s tests -v
```

端到端测试依赖本机 FFmpeg；检测不到 `ffmpeg/ffprobe` 时相关用例自动跳过，
纯算法测试仍可独立运行。
