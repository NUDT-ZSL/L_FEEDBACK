# locdub — 离线游戏本地化配音适配与混流工具

读取对白时间轴 JSON、各语言 WAV 配音与源视频，使用变速不变调（WSOLA）
将每条配音适配到原始时间轴，混入视频背景音轨后按语言封装为独立 MP4，
并生成按语言分组的 JSON 处理报告。

## 环境要求

- 64 位 Python 3.9+（开发与验证使用 Python 3.14）
- NumPy 1.21+、soundfile、ffmpeg-python
- FFmpeg 5.0+（含 `ffprobe`），其 `bin` 目录在 `PATH` 中（验证版本 8.0.1）

安装依赖：

```bash
python -m pip install -r requirements.txt
```

## 命令行用法

Linux/macOS：

```bash
python -m locdub \
  --timeline samples/fixture/timeline.json \
  --video samples/fixture/source.mp4 \
  --lang en=samples/fixture/dubs/en \
  --lang ja=samples/fixture/dubs/ja \
  --out output --workers 2 --gain-db -2
```

Windows 命令提示符（`^` 为 cmd 换行符，也可写成单行）：

```cmd
python -m locdub --timeline timeline.json --video source.mp4 ^
  --lang en=dubs\en --lang ja=dubs\ja ^
  --out output --workers 2 --gain-db -2
```

或直接使用封装脚本，参数原样转发并打印退出码：

```cmd
run_locdub.bat --timeline timeline.json --video source.mp4 --lang en=dubs\en --out output
```

### 参数

| 参数 | 说明 |
| --- | --- |
| `--timeline` | 时间轴 JSON 文件 |
| `--video` | 源视频文件 |
| `--lang NAME=DIR` | 语言名与配音目录，可重复指定多语言 |
| `--out` | 输出目录（MP4、adapted WAV、报告、错误日志） |
| `--workers` | 语言级并发上限（默认 1） |
| `--gain-db` | 配音混音增益，单位 dB（默认 0） |
| `--sample-rate` / `--channels` | 混流音频采样率/声道（默认 48000 / 2） |
| `--max-segment-seconds` | 单片段内存保护上限（默认 600 秒） |
| `--block-seconds` | 流式混音分块大小（默认 60 秒） |
| `--quiet` | 关闭进度输出，错误仍写入日志 |

### 退出码

| 码值 | 含义 |
| --- | --- |
| 0 | 全部语言成功 |
| 2 | 部分成功（存在被跳过的片段，已保留静音占位） |
| 3 | 至少一种语言整体失败 |
| 4 | 不可恢复错误（JSON 解析失败、视频缺失/无视频流等） |

## 时间轴 JSON 格式

时间基准统一为毫秒整数；片段组可任意深度嵌套，解析时递归展平。

```json
{
  "video_duration_ms": 33000,
  "default_ratio_range": [0.5, 2.0],
  "segments": [
    {
      "id": "root_group",
      "segments": [
        {
          "id": "line_001",
          "start_ms": 1000,
          "end_ms": 3500,
          "audio": "line_001.wav",
          "ratio_range": [0.6, 1.8]
        }
      ]
    }
  ]
}
```

- `ratio_range` 缺省时使用 `default_ratio_range`，再缺省为 `[0.5, 2.0]`。
- 校验：`start_ms < end_ms`、片段不重叠、不超出视频时长。

## 输出结构

```
output/
  error.log                # 结构化错误/警告（每行一个 JSON）
  report.json              # 按语言分组的处理报告
  en/
    en.mp4
    adapted/line_001.wav   # 适配后、精确对齐时间轴的片段
  ja/
    ja.mp4
    adapted/...
```

## 生成确定性样本与运行测试

```bash
python -m tools.generate_samples --out samples/fixture
python -m pytest tests/ -q
```

长视频/多片段样本（5.5 分钟、110 段、双语言）：

```bash
python -m tools.generate_samples --out samples/fixture_large --scale 10
```

更多设计与验证细节见 `docs/TECHNICAL.md`。
