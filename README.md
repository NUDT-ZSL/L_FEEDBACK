# Game Audio Mixer

离线游戏音频混音器：读取多轨 WAV 素材和事件时间轴 JSON，将事件按毫秒时间戳放到
48 kHz 统一时间基准，完成轨道静音/独奏、动态范围压缩、EBU R128 响度归一化、
软限幅与真峰值保护，输出 24-bit PCM 混音 WAV 和固定结构的 JSON 响度报告。

## 安装

要求 Python 3.9+、`numpy`、`soundfile`、`scipy`。`soundfile` 需要可用的
libsndfile；常见平台由 wheel 自带。

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# 可选：安装命令行入口
pip install -e .
```

未安装包时也可直接运行：

```bash
python mixer.py --help
```

## 快速使用

```bash
python mixer.py \
  --input-dir ./project \
  --timeline ./project/events.json \
  --output-dir ./out \
  --target-lufs -23 \
  --mute ambience,music \
  --solo dialogue
```

输出：

- `out/mix.wav`：48 kHz、双声道、24-bit PCM WAV。
- `out/loudness_report.json`：固定报告，包含响度、真峰值、短时响度序列、
  各文件采样率/转换方式、轨道状态、事件结果、限幅触发、警告和参数。

安装后入口为 `game-audio-mixer`，参数与 `python mixer.py` 一致。

## 时间轴格式

`timeline_duration_ms` 是可选的合法时间窗口；输出长度仍严格由所有成功解析事件的
最晚结束时间决定，不会用该字段延长输出。时间戳必须非负；提供时间窗口时，
`time_ms >= timeline_duration_ms` 的事件会被跳过。

```json
{
  "title": "level_01",
  "duration_ms": 5000,
  "tracks": [
    {"id": "sfx", "name": "SFX", "mute": false, "solo": false, "gain_db": 0},
    {"id": "amb", "name": "Ambience", "mute": false, "solo": false, "gain_db": -3},
    {"id": "vo", "name": "Dialogue", "mute": false, "solo": false, "gain_db": 0}
  ],
  "assets": [
    {"id": "footstep", "path": "audio/footstep_44k.wav"},
    {
      "id": "forest_bed",
      "path": "audio/forest_22k.wav",
      "offset_ms": 250,
      "duration_ms": 3000
    }
  ],
  "events": [
    {
      "id": "step_001",
      "type": "footstep",
      "time_ms": 1200,
      "track": "sfx",
      "asset": "footstep",
      "gain_db": -2
    },
    {
      "id": "line_001",
      "type": "dialogue",
      "time_ms": 1800,
      "track": "vo",
      "asset": "footstep",
      "offset_ms": 100,
      "duration_ms": 800
    }
  ]
}
```

要点：

- `time_ms` 是事件在统一时间轴上的起点；`offset_ms`/`duration_ms` 选取素材片段。
- 素材路径相对 `--input-dir` 解析；若不存在，再尝试相对时间轴文件目录解析。
- 同一物理文件被多个 ID 或事件引用时只解码和重采样一次。
- `assets` 也可用 `clips`，事件可用 `event_type`、`timestamp_ms`、`track_id`、
  `clip_id` 等常见别名。支持 `asset_manifest` 指向外置 JSON 清单。
- 支持采样率：8k、11.025k、12k、16k、22.05k、24k、32k、44.1k、48k、
  88.2k、96k、176.4k、192k。单声道会复制为双声道；其他声道数按错误隔离。

## 混音规则

- 时间换算为 `round(time_ms * 48000 / 1000)`，整数毫秒在 48 kHz 下是精确的
  48 样本倍数。
- 同轨和多轨重叠事件均逐样本相加；不会因事件重叠而互相覆盖。
- 有效事件即使因静音/独奏不参与混音，也会参与最晚结束时间计算。
- 独奏优先于静音：任一轨道独奏时，仅独奏轨可听；此时“静音+独奏”轨仍可听。
- `--mute`/`--solo` 是完整覆盖而非追加：列出的轨道为真，未列出轨道为假。
  传空字符串可清除 JSON 初始状态；参数可重复并支持逗号分隔。

## 处理链与报告

1. 素材按文件缓存，统一重采样到 48 kHz。
2. 按轨道增益和事件增益叠加原始混音。
3. 预归一化到目标附近，再执行立体声联动、软拐点前馈压缩。
4. 迭代归一化。每次候选结果均经过软限幅、4 倍过采样真峰值保护和 24-bit
   量化，再重新测量响度。
5. 最终整合响度必须位于目标值 ±0.5 LU。预限幅真峰值高于 -1.0 dBTP 时启用
   tanh 软限幅；局部前瞻增益包络控制瞬态，残差再做全局保护，保证最终
   true peak 不超过 0 dBTP。
6. 每个限幅活跃区记录主导轨道、事件、绝对时间、区间、峰值、衰减量及贡献者。

响度采用 ITU-R BS.1770-4/EBU R128：48 kHz K 加权、400 ms 块/100 ms
步进、绝对门限 -70 LUFS、相对门限 -10 LU。短时响度窗口为 3 s、步进 1 s。
真峰值通过 4 倍多相过采样估计。

## 退出码

| 退出码 | 含义 |
|---:|---|
| 0 | 成功；事件级错误只写入报告和 stderr，不中断其余事件 |
| 2 | 命令行参数非法 |
| 3 | 事件 JSON 或外置素材清单不可读/不可解析 |
| 4 | 输出目录或输出文件写入失败 |
| 5 | 最终响度/真峰值交付约束未满足；仍尽量写出 WAV 和报告 |

事件缺失素材、解码失败、采样率不支持、时间戳非法时，会在报告的 `errors`
中给出原因并继续处理有效事件。

## 测试

```bash
python -m pytest -q
```

测试使用临时目录中确定性生成的 44.1/48/22.05 kHz WAV，覆盖事件对齐、重叠叠加、
缓存、重采样报告、响度达标、短时响度、初始/覆盖静音独奏、独奏优先、限幅记录、
异常隔离和致命 JSON 错误。详细设计和实测结果见
[`docs/TECHNICAL.md`](docs/TECHNICAL.md) 与 [`docs/VERIFICATION.md`](docs/VERIFICATION.md)。

## 已知边界

- 输出固定为双声道 24-bit PCM；不提供多声道声道映射。
- 输出最长 2 小时，避免离线容器出现不可控内存占用。
- 若素材的峰值因数导致在 0 dBTP 上限内物理上无法达到目标响度，程序会写出
  约束失败报告并返回 5，而不会伪造响度。
