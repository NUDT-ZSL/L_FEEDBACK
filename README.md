# mediamixer

Windows 本地运行的命令行多轨音频/视频混流工具。读取自定义工程描述 JSON，
使用 FFmpeg 解码/编码、NumPy 做采样与混音，输出单一 MP4 文件，并支持仅
修改音频偏移或音量包络时的增量复用。

## 环境要求

- Python 3.10+，并安装依赖：`pip install -r requirements.txt`（NumPy >= 1.24）
- FFmpeg 6.0+，包含 `ffmpeg` 与 `ffprobe`
  - 将其 bin 目录加入 `PATH`，或
  - 通过环境变量 `FFMPEG_PATH` 指定 `ffmpeg.exe` 所在目录（也可直接指向可执行文件）

## 快速开始（Windows PowerShell / cmd）

```bat
pip install -r requirements.txt
generate_samples.bat
run_mixer.bat samples\project.json
run_tests.bat
```

非 Windows 或直接使用模块：

```bash
python tools/generate_samples.py --out samples
python -m mediamixer samples/project.json
python -m unittest discover -s tests -v
```

## 工程描述 JSON

```json
{
  "video_path": "video.mp4",
  "output_path": "output.mp4",
  "audio_tracks": [
    {
      "path": "track_dialog.wav",
      "offset_ms": 0,
      "volume_envelope": [1.0, 0.8, 0.5, 0.9]
    }
  ],
  "frame_markers": [
    {"frame": 15, "drift_ms": -5}
  ]
}
```

路径相对于工程 JSON 所在目录解析。校验规则：

- `video_path`：必须存在，封装后缀为 `.mp4` 或 `.mkv`
- `audio_tracks`：1 至 8 条；每条含 `path`、整数 `offset_ms`（-5000..5000）、
  `volume_envelope`（长度 >= 2，每个浮点在 0.0..1.0，按等间隔时间点应用）
- `frame_markers`：0 至 3 个；`frame` 为非负整数，`drift_ms` 在 -10..10
- `output_path`：必须以 `.mp4` 结尾，父目录不存在时自动创建

校验失败向 stderr 输出 `{"code": ..., "message": ...}` 并以退出码 2 结束；
运行期错误（子进程失败、IO 异常、解码数据异常）同样输出 JSON 错误对象，
退出码 1。正常运行的进度信息以 JSON Lines 写到 stdout。

## 处理管线

1. **校验**：解析并校验工程描述。
2. **探测**：`ffprobe` 读取宽高、帧率（有理数）与总帧数。
3. **视频**：`ffmpeg` 解码为 RGB24 原始帧，通过管道送入 H.264 编码器；
   每帧显示时间戳（微秒）用整数运算 `idx * 1_000_000 * den / num` 计算，
   避免逐帧浮点累积误差。
4. **帧标记**：标记颜色由 `crc32(frame:drift_ms)` 决定；矩形以画面中心为
   基准，按每毫秒 2 像素水平偏移，超出边界则夹取；仅在对应帧序号上绘制。
5. **音频**：每条轨道解码为 48 kHz 立体声 float32 PCM；负偏移裁掉视频起点
   之前的数据，正偏移在开头补静音，超过视频总时长的尾部截断。
6. **包络与混音**：包络点按轨道自身时长等间隔分布、线性插值后逐样本相乘，
   再累加到统一立体声缓冲区。峰值超过 1.0 时整体缩放到峰值 0.99，并在
   stdout 的 `peak_normalization` 事件中记录缩放因子。
7. **编码封装**：视频 H.264（CRF 23，preset medium，无音频中间文件），
   最终与 AAC 192k 音频混流为 MP4；原始帧数据全程走管道。

## 增量更新与持久化状态

首次全量运行在输出文件同目录写入：

- `<output>.intermediate.mp4`：独立持久化的无音频 H.264 中间视频
- `<output>.state.json`：视频/中间视频/各音频文件的 SHA-256、帧率、总帧数、
  偏移与包络哈希、帧标记哈希
- `<output>.audiocache/`：按音频源文件内容哈希命名的解码 PCM 缓存

再次运行时：

- 仅音频偏移或包络变化 → 跳过视频解码与标记处理，直接复用中间视频，
  重新混音并封装，stdout 输出 `incremental_reuse` 事件及受影响轨道
- 视频内容、音频文件内容、中间视频摘要、帧标记或几何参数变化 → 全量重建

状态文件采用“写临时文件 + 原子重命名”。启动时清理 `*.state.json.tmp`
与 `*.intermediate.mp4.tmp` 残留，因此被强制终止后无需退出回调即可恢复。

## 目录结构

```
mediamixer/          主程序包
  cli.py             命令行入口
  project.py         工程解析与校验
  ffmpeg_utils.py    ffmpeg/ffprobe 封装
  video.py           帧时间戳与标记渲染
  audio.py           解码、偏移、包络、混音
  state.py           状态持久化与增量判定
  pipeline.py        全量/增量编排
tools/generate_samples.py  确定性样本生成（testsrc + sine）
tests/               30 个 unittest 用例
```

## 已验证项

完整验证记录见 `VERIFICATION.md`。
