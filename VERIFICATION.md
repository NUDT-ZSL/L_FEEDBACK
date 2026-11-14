# 实际验证记录

## 环境

- .NET SDK：8.0.428
- 运行时：Microsoft.NETCore.App 8.0.34
- 实际执行环境：Ubuntu 26.04 x64（任务目标 Windows 10/11 使用同一跨平台 .NET 8 代码；未在 Windows 机器上实跑）
- 构建：Release，0 warning，0 error
- 网络/NuGet：`NuGet.Config` 清空包源；测试项目无第三方包

## 构建命令

```bash
dotnet restore SpatialAudioPipeline.sln --configfile NuGet.Config
dotnet build SpatialAudioPipeline.sln -c Release --no-restore
```

## 验收命令

```bash
dotnet tests/AudioPipeline.Tests/bin/Release/net8.0/AudioPipeline.Tests.dll
```

## 实际结果

```text
PASS deterministic fixture generation
PASS scene and source validation with stable error codes
PASS WAV frame count, RIFF sizes, and final-block padding
PASS resume skips completed blocks and is byte-identical
PASS missing, size-mismatched, and hash-mismatched blocks recalculate the suffix
PASS manifest version change rebuilds every block
PASS Doppler read phase and occlusion low-pass survive block boundaries
PASS exclusive output-directory lock rejects a second process
PASS fixed-point mixing clips once and leaves no event tail
PASS analytical segment/AABB intersection

10 passed, 0 failed
```

最近一次完整测试运行目录：

`artifacts/test-runs/20261114-093512-6609b68f`

测试同时验证了两种入口：

```bash
dotnet run --project tests/AudioPipeline.Tests -c Release --no-build
dotnet tests/AudioPipeline.Tests/bin/Release/net8.0/AudioPipeline.Tests.dll
```

两种入口均为 10 passed、0 failed。

## 关键断言

- 内置夹具场景总帧数严格为 `105,600`，母带数据区 `422,400` 字节，文件总长 `422,444` 字节。
- 三个内部块均固定为 `192,000` 字节；最后一块在 9,600 个有效帧之后全部补静音。
- 暂停/续渲、缺失块、尺寸篡改、摘要篡改、出口状态篡改和清单版本变化后的最终母带 SHA-256 与完整一次渲染一致。
- 第一个块出口的清单状态包含非整数 `dopplerReadPosition`，其 `dopplerDelayPhase` 等于小数部分；`lowPassState` 非零并在下一块继续变化。
- 独立 CLI 子进程在父进程持锁时失败，stdout 包含 `OUTPUT_LOCKED`；释放锁后完整渲染成功。
- Q16 累加器对 `1.9 + (-1.4)` 得到 `0.5`，证明未在逐事件阶段裁剪到 16-bit；最终超范围样本裁剪到 `32767/-32768`。
- 事件结束到后续事件开始之间所有双声道样本均为 0。

## 手动端到端冒烟记录

```text
fixture: expectedFrames = 105600
render: totalFrames = 105600, totalBlocks = 3, renderedBlocks = 3
resume: renderedBlocks = 0, reusedBlocks = 3
inspect: sampleRate = 48000, channels = 2, bitsPerSample = 16,
         frameCount = 105600, fileLength = 422444
sha256: 68767772377b07cd70331f6169177a537dede956f18321d9ee3f6afd76e9e3b6
```

端到端产物目录：`artifacts/final-validation-20261114`

## Ctrl+C 实际信号验证

使用 300 秒、300 块的内置源场景，通过 `timeout --preserve-status -s INT` 向前台 CLI 发送真实 `SIGINT`：

```text
Ctrl+C received; the active block will finish before stopping.
block 000046 rendered EA0787F65F73
exit = 130
renderedBlocks = 47, reusedBlocks = 0, completed = false
manifest = 47 complete / 253 pending
master.wav exists = false
```

随后执行 `--stop-after-blocks 1`，结果从第 47 块继续：

```text
blocks 000000..000046 reused
block 000047 rendered
renderedBlocks = 1, reusedBlocks = 47
```

信号测试产物：`artifacts/signal-check-20261114c`
