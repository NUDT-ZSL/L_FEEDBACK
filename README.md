# 多语言配音离线唇形处理工具（lipsync pipeline）

离线批量处理多语言游戏配音：解码 WAV/OGG → 重采样与声道转换 → 解析音素时间戳 →
音素映射为 viseme → 按视频帧率生成归一化口型权重曲线 → 汇总 JSON 处理报告。
仅依赖 C++17 标准库、线程库与 FFmpeg 共享库（avformat/avcodec/avutil/swresample）。

## 目录结构

```
app/main.cpp                 命令行入口
src/lipsync/                 可链接静态库 lipsync_core
  error_codes.h              可恢复业务错误码
  json.{h,cpp}               零依赖 JSON 解析/确定性序列化
  config.{h,cpp}             JSON 配置加载与校验
  viseme_mapper.{h,cpp}      数据驱动的音素->viseme 映射（内置 en/ja/es）
  phoneme_parser.{h,cpp}     音素侧车 JSON 解析与时长越界校验
  audio_processor.{h,cpp}    FFmpeg 解码/重采样/声道转换 + 手写 PCM16 WAV
  lip_sync.{h,cpp}           帧率对齐与归一化 viseme 权重曲线
  pipeline.{h,cpp}           有界并行管线、单调时钟计时、报告生成
config/default.json          默认配置
tools/generate_samples.py    确定性样本生成器（6 文件、3 语言、2 类异常）
tests/                       单元/集成/并发测试（无第三方测试框架）
scripts/run_demo.bat         Windows 10/11 + VS2022 端到端脚本
scripts/run_demo.sh          Linux/macOS 端到端脚本
```

## 构建

依赖：CMake 3.20+、支持 C++17 的编译器（VS2022 / GCC / Clang）、
FFmpeg 5.1+ 共享库及开发头文件。

```bash
# 方式一：系统/包管理器安装了 FFmpeg 开发包（vcpkg、apt 等）
cmake -S . -B build
cmake --build build --parallel

# 方式二：手动指定 FFmpeg 根目录（Windows 下设置 FFMPEG_ROOT 环境变量）
cmake -S . -B build -DFFmpeg_ROOT="C:/ffmpeg-5.1-full-build-shared"
cmake --build build --config Release
```

运行测试：

```bash
ctest --test-dir build --output-on-failure
# 若环境无 ctest，可直接执行 build/tests/lipsync_*_test
```

## 使用

```bash
lipsync --input <输入目录> --output <输出目录> --config <配置 JSON>
```

输入目录中每个音频 `name.wav`/`name.ogg` 需要同名侧车 `name.json`：

```json
{
  "language": "en",
  "phonemes": [
    {"phoneme": "HH", "start": 0.05, "end": 0.18},
    {"phoneme": "AA", "start": 0.18, "end": 0.40}
  ]
}
```

配置字段：

| 字段 | 含义 | 默认 |
| --- | --- | --- |
| `target_sample_rate` | 输出 WAV 采样率（Hz） | 48000 |
| `target_channels` | 输出声道数，仅允许 1/2 | 1 |
| `fps` | 口型曲线帧率 | 30.0 |
| `languages` | 允许的语言标识列表 | en/ja/es |
| `max_workers` | 同时处理文件数上限 | 硬件并发数 |
| `overflow_tolerance_ms` | 音素越界容差（毫秒） | 100.0 |
| `viseme_map_path` | 可选外部映射表，扩展新语言 | 内置映射 |

输出布局（报告内路径均相对输出目录）：

```
<output>/audio/<name>.wav              严格匹配目标采样率/声道的 PCM16 WAV
<output>/lipsync/<name>_lipsync.json   每帧全量 viseme 权重，和为 1
<output>/report.json                    逐文件状态、错误码、单调时钟耗时
```

## 关键设计

- 语言解耦的映射：映射器只认“语言键 + 音素键 -> 单一 viseme”，
  内置英语 ARPAbet、日语片假名罗马音、西班牙语音素三套表；
  通过 `viseme_map_path` 可无代码扩展任意语言。每个音素仅映射一个 viseme，
  未知音素记录警告并回退静默 viseme `SIL`，处理不中断。
- 帧对齐：帧数 `ceil(音频时长 * fps)`，曲线从 0 时刻开始并覆盖到音频结束；
  帧权重按与音素区间的重叠时长分配，未覆盖部分计入 `SIL`，最后归一化保证和为 1。
- 时长精度：原始时长按源采样率总样本数计算并四舍五入到毫秒，用于越界校验。
- 有界并行：固定大小工作线程池（上限 `max_workers`）+ 原子索引派发；
  单文件失败被隔离在该任务内，其余文件继续完成。
- 可恢复错误码：`E_IO`、`E_PARSE_JSON`、`E_AUDIO_DECODE`、`E_RESAMPLE`、
  `E_PHONEME_OVERFLOW`（跳过唇形但保留音频产物）、`E_CONFIG`、`E_INTERNAL`。
  越界与 JSON 损坏属于业务结果，进程整体仍返回 0；仅管线级致命错误返回非 0。

## 确定性样本与验收

`tools/generate_samples.py` 使用固定公式合成 2 秒 24kHz 单声道 WAV，
生成 6 个样本：英语 2 个、日语 2 个（含 1 个未知音素）、西班牙语 2 个，
其中 1 个音素区间结束于 2.6s（超出 2.0s 音频 600ms），1 个侧车 JSON 语法损坏。
安装了 ffmpeg CLI 时日语音频自动编码为 OGG，否则回退 WAV（解码路径另有 OGG 测试覆盖）。

```bash
python3 tools/generate_samples.py --output build/samples
./build/lipsync --input build/samples --output build/output --config config/default.json
```

本地实测结果（GCC 15.2 + FFmpeg 8.0 共享库）：

- 汇总：6 文件 -> success=4、skipped=1、failed=1，处理继续完成。
- 报告同时包含 `E_PHONEME_OVERFLOW`（skipped，音频产物保留）与
  `E_PARSE_JSON`（failed），其余 4 文件 `OK`。
- 全部输出 WAV 为 48000Hz/单声道/PCM16，2.000s、96000 帧。
- 口型曲线均为 60 帧（2s × 30fps），每帧含 15 个 viseme 权重且和为 1，
  无音素覆盖的帧静默权重为 1。
- 未知音素 `XX` 产生警告并回退 `SIL`。
- 并发探针实测峰值并发等于 `max_workers=2`；`max_workers=1` 串行同样完整处理。
- 测试总计 644 项断言全部通过（单元 214、集成 416、并发 14）。

Windows 环境执行 `scripts/run_demo.bat`（先设置 `FFMPEG_ROOT`）即可复现
“构建 -> 生成样本 -> 运行管线 -> ctest”完整流程。
