# 离线空间音频母带渲染器

一个无外部素材、无 NuGet 依赖的 .NET 8 控制台项目。它把 JSON 场景中的三维环境音效事件离线渲染为双声道 48 kHz、16-bit PCM WAV，并支持 48,000 帧分块、原子落盘、摘要校验、断点恢复和多进程独占写入。

## 快速开始

要求：.NET 8 SDK。项目不访问网络，`NuGet.Config` 已清空包源。

```bash
dotnet restore SpatialAudioPipeline.sln --configfile NuGet.Config
dotnet build SpatialAudioPipeline.sln -c Release --no-restore

# 生成固定种子的内置音源和场景
dotnet run --project src/AudioPipeline.Cli -c Release --no-build -- \
  fixture --output artifacts/demo --seed 1592598566

# 校验场景与音源
dotnet run --project src/AudioPipeline.Cli -c Release --no-build -- \
  validate --scene artifacts/demo/scene.json

# 完整渲染
dotnet run --project src/AudioPipeline.Cli -c Release --no-build -- \
  render --scene artifacts/demo/scene.json --output artifacts/demo/master.wav

# 检查 WAV 头
dotnet run --project src/AudioPipeline.Cli -c Release --no-build -- \
  inspect --wav artifacts/demo/master.wav
```

Windows PowerShell 可运行：

```powershell
.\scripts\verify.ps1
```

Linux/macOS：

```bash
chmod +x scripts/verify.sh
./scripts/verify.sh
```

测试是可独立执行的控制台验收套件，不依赖 xUnit 或网络包源。

## 项目结构

```text
src/AudioPipeline/            核心库：WAV、场景、DSP、分块恢复、锁、母带拼接
src/AudioPipeline.Cli/        命令行入口
tests/AudioPipeline.Tests/    无 NuGet 依赖的确定性验收套件
scripts/                      一键构建与验证
artifacts/                    实际运行产生的夹具、块、清单和测试结果
```

本次实现的关键文件：

- `src/AudioPipeline/SceneLoader.cs`：场景解析、上限检查、音源校验、事件摘要哈希
- `src/AudioPipeline/SpatialDsp.cs`：距离衰减、声像、多普勒、遮挡和一阶低通
- `src/AudioPipeline/Renderer.cs`：逐帧混音、分块状态、恢复与前缀复用
- `src/AudioPipeline/ManifestStore.cs`：块/清单原子写入、摘要与尺寸校验
- `src/AudioPipeline/WavIO.cs`：纯手写 RIFF/WAVE 读写
- `src/AudioPipeline/WavAssembler.cs`：按序拼接母带并裁剪最后不足一块的部分
- `src/AudioPipeline/FixtureGenerator.cs`：固定种子音源与 JSON 场景生成

## 场景格式

```json
{
  "version": 1,
  "name": "example",
  "seed": 1592598566,
  "listener": { "x": 0, "y": 1.6, "z": 0 },
  "events": [
    {
      "id": "doppler-pass",
      "start": { "x": -10, "y": 1, "z": 8 },
      "end": { "x": 10, "y": 1, "z": 8 },
      "durationSeconds": 2.2,
      "delaySeconds": 0,
      "gain": 0.65,
      "sourcePath": "sources/ambient_a.wav",
      "minDistance": 1,
      "maxDistance": 24,
      "doppler": true,
      "occluders": [
        {
          "min": { "x": -1.2, "y": -2, "z": 5 },
          "max": { "x": 1.2, "y": 2, "z": 6 }
        }
      ]
    }
  ]
}
```

音源必须是单声道、48,000 Hz、16-bit PCM WAV。相对路径以场景文件所在目录为基准。场景版本固定为 1。

显式限制：

- 最多 128 个事件。
- 单个事件时长、延迟和声源最长 300 秒；总时间线最多 14,400,000 帧。
- 单块固定 48,000 帧，Q16 双声道累加器为 384,000 字节；默认中间缓冲上限 1,048,576 字节。
- 每个轴对齐遮蔽盒必须满足 `min <= max`；`0 <= minDistance < maxDistance`。

## 渲染定义

### 帧与位置

- 事件起始帧：`round(delaySeconds * 48000)`。
- 事件帧数：`ceil(durationSeconds * 48000)`，先消除浮点尾差。
- 场景总帧数：全部事件 `[起始帧, 结束帧)` 的最大结束帧。
- 事件内位置：

```text
u = localFrame / max(1, durationFrames - 1)
position = start + (end - start) * u
```

每帧只计算一次 `position`，随后复用于距离、方位、遮挡和多普勒。

### 距离与声像

监听者前方为 `+Z`，右方为 `+X`。

```text
if distance <= minDistance: gain = 1
else if distance >= maxDistance: gain = 0
else: gain = (maxDistance - distance) / (maxDistance - minDistance)

pan = clamp(atan2(dx, dz) / (pi / 2), -1, 1)
left = (1 - pan) / 2
right = (1 + pan) / 2
```

其中 `(dx, dz)` 是监听者指向声源的水平向量。

### 多普勒

源速度由线性轨迹和实际事件帧数得到：

```text
velocity = (end - start) / playbackSeconds
radialSpeed = dot(velocity, normalize(listener - source))
factor = 343 / (343 - radialSpeed)
factor = clamp(factor, 0.5, 2.0)
```

`radialSpeed > 0` 表示接近监听者。启用多普勒时，维护连续的分数读取位置：

```text
sample = linearInterpolate(source, readPosition)
readPosition += factor
```

禁用多普勒时直接读取整数帧 `localFrame`。两种模式的读取位置都在事件状态中跨块保存；启用多普勒时，块边界前保存完整读取位置和其小数相位，因此恢复不会产生相位跳变。

### 遮挡与低通

每帧使用解析 slab 法判断监听者到声源的闭线段是否与任一 AABB 相交。滤波状态按事件维护并逐帧更新：

```text
y[n] = 0.6 * x[n] + 0.4 * y[n - 1]
output = occluded ? y[n] : x[n]
```

即使当前未遮挡也持续更新 `y[n]`，这样遮挡开始、解除和块边界都不会重置滤波器状态。清单中的 `lowPassState` 在块出口原子记录。

### 定点叠加与裁剪

- 每个事件先按声像和增益计算左右贡献。
- 贡献转成 Q16 写入单块 `int[frame * 2]` 累加器。
- 多事件逐个累加，加法在 32 位边界饱和，但不做逐事件 16 位裁剪。
- 全块完成后统一转回 16-bit，并裁剪到 `[-32768, 32767]`。
- 事件区间按半开区间处理，最后活动帧之后立即停止写入，不产生额外尾音。

## 分块、清单与恢复

输出目录布局：

```text
master.wav
master.wav.state/
  manifest.json
  blocks/
    block-000000.pcm
    block-000001.pcm
    ...
.audio-pipeline.lock
```

每个块固定 48,000 帧，即 192,000 字节（双声道 16-bit）。最后不足一块的内部块文件仍补满静音，拼接母带时只复制到精确的总帧数。

清单包含：

- 清单版本、算法版本、采样率、声道数、位深、块帧数。
- 总帧数、总块数、规范化事件摘要 SHA-256。
- 每块状态、帧数、字节数、块 SHA-256。
- 每块出口的事件 DSP 状态：`nextLocalFrame`、`lowPassState`、`dopplerReadPosition`、`dopplerDelayPhase`。

写入顺序：

1. 在块目录创建唯一临时文件。
2. 写入完整块并 `Flush(true)`。
3. 同卷原子替换 `block-NNNNNN.pcm`。
4. 原子替换 `manifest.json`。

恢复流程：

1. 先取得输出目录的独占 `.audio-pipeline.lock`。
2. 比较清单版本、算法版本、格式、总帧数、总块数和事件摘要。
3. 任意关键字段不匹配时清空块并整体重建，不沿用旧块偏移。
4. 匹配时逐块校验存在性、精确字节数、SHA-256，以及后续块所必需的事件出口状态。
5. 找到首个失效块后，仅保留此前连续有效前缀；失效块及其后所有块标记为 `pending`。
6. 从有效前缀最后一块的出口状态恢复 DSP，顺序重算待处理后缀。
7. 全部块完成后按序拼接母带；下次运行全块有效时只校验和拼接，不重算。

清单不可解析时会安全地整体重建。强制终止可能留下临时文件；下次加锁后会清理临时文件，且未提交的块不会出现在清单中，因此仍可恢复。

## 并发与中断

- 同一输出目录始终由一个 `FileShare.None` 锁文件的持有者写入；其他进程返回 `OUTPUT_LOCKED`。
- 锁覆盖块渲染和母带拼接全过程。
- CLI 捕获 `Ctrl+C`，只设置取消令牌，不中断当前块；当前块完成并写入清单后退出码为 130。
- 下次正常执行同一 `render` 命令会从清单断点继续。
- 测试可用 `--stop-after-blocks N` 无信号地模拟同一恢复路径。

## 稳定业务错误码

CLI 在错误时输出 JSON 错误包，并同时写 stderr：

```json
{
  "code": 2001,
  "name": "OUTPUT_LOCKED",
  "message": "Output directory is already owned by another writer: '...'."
}
```

| 数值 | 名称 | 含义 |
|---:|---|---|
| 1001 | `SCENE_IO` | 场景文件无法读取 |
| 1002 | `SCENE_JSON` | JSON 语法或结构错误 |
| 1003 | `SCENE_VERSION` | 不支持的场景版本 |
| 1004 | `SCENE_VALIDATION` | 字段值、路径或 AABB 非法 |
| 1005 | `SOURCE_MISSING` | 音源不存在 |
| 1006 | `SOURCE_FORMAT` | 非单声道 48 kHz 16-bit PCM WAV 或 WAV 损坏 |
| 1007 | `DURATION_LIMIT` | 时长、延迟或总时间线超限 |
| 1008 | `EVENT_LIMIT` | 事件总数超限 |
| 1009 | `BLOCK_BUFFER_LIMIT` | 单块中间缓冲超限 |
| 2001 | `OUTPUT_LOCKED` | 输出目录被其他写入者占用 |
| 2002 | `MANIFEST_INVALID` | 已请求读取清单但清单不可用或状态不一致 |
| 3001 | `RENDER_IO` | 输出、原子替换或母带拼接 I/O 失败 |
| 3002 | `CANCELLED` | 渲染被取消 |

库 API 抛出的 `PipelineException` 同时提供 `Code`、`NumericCode`、`StableCode` 和 `ToEnvelope()`。

## 验证

验收套件覆盖：

- 固定种子夹具与音源可重复生成。
- 场景版本、JSON、缺失音源、格式、事件上限、时长上限和字段校验的稳定错误码。
- 母带 RIFF 头、48 kHz/双声道/16-bit 参数、精确帧数、末尾块补静音。
- 分块暂停后复用前缀；续渲结果与一次渲染字节一致。
- 缺失块、尺寸不符、摘要不符时从首个坏块重算后缀。
- 清单版本改变时整体重建。
- 多普勒分数读取相位和遮挡低通状态跨块继承。
- 真实子进程争用同一目录锁时返回 `OUTPUT_LOCKED`。
- Q16 定点叠加只在最终统一裁剪，事件结束后无尾音。
- 线段-AABB 相交的解析判定。

最近一次实际结果见 `VERIFICATION.md`。
