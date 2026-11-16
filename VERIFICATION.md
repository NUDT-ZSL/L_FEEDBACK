# 实际验证记录

验证日期：2026-11-17

## 环境

- Windows 11：10.0.26200
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU
- 编译器：MSVC 19.29.30157.0（Visual Studio 2019 Build Tools）
- CMake：Visual Studio 2019 自带 CMake
- Windows SDK：10.0.19041.0
- 特性级别：Direct3D 11_0
- 运行方式：64 位 Release，纯离屏，无可见窗口

## 构建结果

使用的命令：

```bat
scripts\build_windows.bat
```

结果：构建成功，生成 `build\Release\deterministic_csm.exe`，运行时 HLSL
`MainVS`、`MainPS`、`ShadowVS`、`ShadowPS` 均成功编译。

## 60 帧与设备丢失验收

执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_acceptance.ps1 `
  -Exe build\Release\deterministic_csm.exe
```

两次运行参数等价于：

```text
--frames 60
--simulate-device-loss-at 30
```

结果：

- 两次运行退出码均为 0。
- 每次运行生成 60 张 PNG、60 份每帧 JSON 和 1 份 `stats.json`，共 121 个文件。
- 每次运行的 60 张 PNG SHA-256 全部不同，说明帧内容随动画时间变化。
- 两次运行的 121 个文件逐字节一致。
- 额外执行一次无设备丢失注入的 60 帧运行，第 0..59 帧 PNG 与注入运行
  逐帧 SHA-256 完全相同，确认重建未改变像素内容或帧序。
- 帧号严格为 `0000..0059`，无重复、无缺失。
- 第 0 帧时间 `0.000000` 秒，第 30 帧 `0.500000` 秒，第 31 帧
  `0.516667` 秒，第 59 帧 `0.983333` 秒。
- 每帧绘制调用数为 4；三级阴影贴图尺寸均为 1024x1024。
- 第 0..30 帧累计恢复次数为 0；第 31..59 帧累计恢复次数为 1。
- 最终 `stats.json` 记录 `deviceLossInjectionAfterFrame=30`、
  `deviceLossRecoveries=1`。

实际哈希：

```text
run_a/stats.json SHA-256:
28519388f8acafff4d0fc1351d239dba64604b4da7a010eb2c34c94b77db91af

run_b/stats.json SHA-256:
28519388f8acafff4d0fc1351d239dba64604b4da7a010eb2c34c94b77db91af

两次运行全部 121 个文件的组合摘要（文件名 + 文件 SHA-256）：
9aad7a3c305e610d25219045ddc670e3c48bea559ea521f20064b589d4cb5f6c
```

独立 Python 校验：

```text
OK: 60 frames per run, byte-identical outputs
OK: 60 unique PNG frames per run
Combined output digest: 9aad7a3c305e610d25219045ddc670e3c48bea559ea521f20064b589d4cb5f6c
```

预览帧位于 `artifacts/preview_frame_0030.png`。

## 参数错误验收

以下输入均向 stderr 输出错误并以退出码 2 结束：

| 输入 | 结果 |
| --- | --- |
| `--frames -1` | 拒绝，退出码 2 |
| `--cascade-splits 0.1,0.3` | 拒绝，退出码 2 |
| `--cascade-splits 0.4,0.3,0.8` | 拒绝，退出码 2 |
| `--cascade-splits 0,0.3,0.8` | 拒绝，退出码 2 |
| `--frames 2 --simulate-device-loss-at 2` | 拒绝，退出码 2 |
| `--frames 1 --const-offset -0.1` | 拒绝，退出码 2 |
| `--frames 1 --not-an-option 1` | 拒绝，退出码 2 |

## 自定义参数抽查

执行：

```bat
deterministic_csm.exe --frames 3 --output-dir out_loss ^
  --simulate-device-loss-at 1 ^
  --cascade-splits 0.2,0.45,0.8 ^
  --slope-offset 1.25 --const-offset 0.002
```

输出统计正确记录了：

```text
cascadeSplitRatios: [0.200000, 0.450000, 0.800000]
cascadeSplitDistances: [0.500000, 28.400000, 63.275000, 140.000000]
slopeOffset: 1.250000
constOffset: 0.002000
deviceLossRecoveries: 1
```

