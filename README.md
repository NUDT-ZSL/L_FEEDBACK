# 游戏过场动画多语言配音工具

本项目实现一个 Windows 可直接运行、结果确定性的命令行工具。它读取本地化
JSON 清单，校验时间轴，归一化 WAV，按语言混音，生成 WebVTT，并用 FFmpeg
封装为带多音轨的 faststart MP4。

工具只把 FFmpeg/FFprobe 作为外部依赖；WAV 读写使用 `numpy 1.26.4` 与
`soundfile 0.12.1`。所有样本由 `tools/make_samples.py` 确定性构造。

## Windows 环境

1. 安装 Python 3.11，并确保 `py -3.11 -m pip --version` 可执行。
2. 下载 FFmpeg 6.0 Windows 静态构建并解压。
3. 将包含 `ffmpeg.exe` 和 `ffprobe.exe` 的 `bin` 目录加入 `PATH`。
4. 安装依赖：

```powershell
cd <项目根目录>
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

FFmpeg 路径也可显式传入，避免修改系统环境：

```powershell
.\.venv\Scripts\python.exe -m localizer .\manifest.json `
  --ffmpeg D:\ffmpeg\bin\ffmpeg.exe `
  --ffprobe D:\ffmpeg\bin\ffprobe.exe
```

## 运行

```powershell
python -m localizer manifest.json
python -m localizer manifest.json --jobs 4 --output-dir out
python -m localizer manifest.json --no-overwrite
```

安装 `pyproject.toml` 后也可使用 `game-dub-localizer` 入口。

默认输出文件：

```text
<output-dir>/<name>.mp4
<output-dir>/<name>.<iso639-2b>.stereo.wav
<output-dir>/<name>.<iso639-2b>.mono.wav
<output-dir>/<name>.<iso639-2b>.vtt
```

`--jobs` 默认 2，最终实际值不超过 `os.cpu_count()`。输出同名文件默认覆盖；
`--no-overwrite` 会在开始处理前预检全部目标路径并以 `E_EXISTS` 退出。

## 清单格式

时间与时长单位均为毫秒。音频路径相对清单文件所在目录解析；清单内
`output.directory` 也相对清单文件目录解析，命令行 `--output-dir` 则相对当前目录。

```json
{
  "version": 1,
  "timeline": { "duration_ms": 2600 },
  "target": {
    "sample_rate": 48000,
    "channels": 2,
    "bit_depth": 16
  },
  "output": {
    "directory": "out",
    "name": "cutscene"
  },
  "languages": [
    {
      "code": "en",
      "name": "English",
      "clips": [
        {
          "id": "line-001",
          "start_ms": 0,
          "duration_ms": 800,
          "path": "audio/en/line-001.wav",
          "text": "Engine ready.\nPress & continue."
        },
        {
          "id": "line-002",
          "start_ms": 795,
          "duration_ms": 905,
          "path": "audio/en/line-002.wav",
          "text": "A five-millisecond fade overlap."
        }
      ]
    },
    {
      "code": "fr",
      "name": "Français",
      "clips": [
        {
          "id": "line-001",
          "start_ms": 200,
          "duration_ms": 700,
          "path": "audio/fr/line-001.wav",
          "text": "Bonjour,\n\n<salut> & bienvenue."
        }
      ]
    }
  ]
}
```

也支持顶层扁平 `clips` 数组，每个片段自行声明 `language`。常用别名字段：

- 语言：`code` / `language` / `lang`
- 起点：`start_ms` / `start`
- 时长：`duration_ms` / `duration`
- 音频：`path` / `file` / `audio`
- 字幕：`text` / `subtitle` / `subtitle_text`

语言可写 ISO 639-1（如 `en`、`zh-CN`）或 ISO 639-2。工具统一转换为
ISO 639-2/B，例如 `en -> eng`、`fr -> fre`、`zh -> chi`、`de -> ger`。

### 时间轴规则

- `start_ms >= 0` 且 `duration_ms > 0`。
- 每种语言独立校验，跨语言可同步重叠。
- 同语言相邻片段允许 `0 < overlap <= 10ms`，混音时应用互补淡出/淡入。
- 超过 10ms 的同语言重叠为 `E_TIMELINE`。
- 相邻片段间隔 `<= 5ms` 视为连续剪切；更大间隔保留为静音填充，不报错。
- 未声明总时长时使用所有片段的最大结束时间；显式总时长不得短于该值。
- 源 WAV 若比清单声明时长短则补静音，若更长则裁剪，并在进度事件中记录。

## 处理行为

### 音频归一化与混音

逐个读取 WAV，检查真实容器格式、采样率、声道数和 PCM/float 位深。缺失、
非 WAV、损坏、空帧或不可解码文件归为 `E_MEDIA`，该语言被跳过，其余语言继续。
检测到差异时：

- 任意常见采样率线性重采样到目标采样率。
- 单声道复制为立体声；多声道先下混为单声道再构造双声道。
- 混音在 float64 中完成，最终按目标位深写成 PCM WAV。

每个语言先生成总长等于时间轴的静音基底，再按毫秒对应的采样点叠加片段。
立体声和单声道版本分别归一化到 `-1 dBFS`，最终原子写入两个 WAV。

### WebVTT

每个语言输出一个 UTF-8 无 BOM 的 WebVTT。时间直接使用清单整数毫秒，因此
字幕边界与片段边界为 0ms 误差。`&`、`<`、`>` 会实体转义，CRLF/CR 统一为 LF；
字幕内部空行使用 NBSP 保留，避免提前终止 cue。

### MP4

FFmpeg 生成 1280x720、100fps 的黑场占位视频，帧数为
`ceil(total_ms / 10)`，保证视频与音频时长差不超过 9ms。最终 MP4：

- 一条 MPEG-4 Part 2 视频流。
- 每个成功语言一条 48kHz 双声道 AAC 音轨。
- 音轨 `language` 标签使用 ISO 639-2/B。
- `+faststart`、固定 `creation_time`、`-fflags +bitexact`、关闭 timecode track、
  固定线程数并移除输入元数据。
- 封装后用 FFprobe 检查音频数量、标签、声道、采样率、各音轨与视频时长差
  （上限 10ms）以及容器时长；失败输出 `E_MUX`。

`E_MUX` 时保留 `WAV`、`VTT` 和 `<name>.intermediate.<pid>.<id>/black.mp4`
等中间产物，仅清理未提交的 `.tmp`。

## JSON 输出约定

stdout 只写 JSON Lines。每条进度事件包含 `language`、`stage`、`elapsed_ms` 和
`monotonic_ns`；时间戳在进程内串行化且单调不减。

```json
{"event":"progress","language":"eng","stage":"mix","elapsed_ms":18,"monotonic_ns":123456789}
{"event":"summary","ok":true,"exit_code":0,"outputs":{},"elapsed_ms":420,"monotonic_ns":123459999}
```

stderr 只写 JSON 错误对象：

```json
{"ok":false,"code":"E_TIMELINE","message":"...","exit_code":2,"details":{"field":"..."}}
```

| 退出码 | 业务码 | 含义 |
|---:|---|---|
| 0 | - | 全部成功 |
| 2 | `E_TIMELINE` | 清单、时间轴、时长或参数校验失败 |
| 3 | `E_EXISTS` | `--no-overwrite` 且目标已存在 |
| 4 | `E_MEDIA` / `E_PROCESS` | 单语言媒体或处理失败；汇总后非零 |
| 5 | `E_MUX` | FFmpeg 封装或封装后校验失败 |
| 130 | `E_INTERRUPTED` | Ctrl+C / SIGBREAK 中断 |

单语言失败不会阻止其他语言继续处理，最终 summary 的 `failed_languages` 列出全部
失败项。只要至少一个语言成功，仍会用成功语言生成 MP4，但退出码保持非零。

## 原子性、并发与中断

- 每个目标路径有独立互斥锁；临时文件与目标文件位于同一目录。
- 写入完成后用 `os.replace` 原子提交。
- Ctrl+C/SIGBREAK 设置协作取消标志。已完成当前写入的语言会提交已完成的
  WAV/VTT；排队或其他未完成语言不再启动。
- 主线程最多等待 1.5 秒；仍在进行的 FFmpeg 子进程会被终止，未提交 `.tmp`
  被删除，随后以 130 退出。

## 确定性

相同输入、相同 Python/依赖版本、相同 FFmpeg 构建和相同环境下：

- WAV/VTT 字节一致。
- MP4 的轨道顺序、编码参数、时间基、元数据和 timecode 设置固定。
- 与 `--jobs` 数量无关。

不同操作系统或不同 FFmpeg 构建之间，WAV/VTT 仍一致，但 MP4 容器字节可能因
编码器/复用器构建差异而不同；确定性承诺针对相同环境重复执行。

## 自测

```powershell
py -3.11 tools\make_samples.py --root build\acceptance
py -3.11 tests\acceptance.py `
  --ffmpeg D:\ffmpeg\bin\ffmpeg.exe `
  --ffprobe D:\ffmpeg\bin\ffprobe.exe
```

`tests/acceptance.py` 覆盖正常流程、源格式差异、峰值、静音填充、VTT 对齐与转义、
MP4 多音轨/时长/黑场/faststart、默认覆盖、`--no-overwrite`、确定性、错误码
以及中断后的 130/原子提交/临时文件清理。实际执行结果见 `VALIDATION.md`。
