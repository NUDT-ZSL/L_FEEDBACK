# 实际验证记录

验证日期：2026-11-17  
环境：WSL2 Linux + Windows 主机，MinGW-w64 GCC 13 交叉编译，Linux GCC 15 本机构建。

## 1. 构建与核心测试

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
cmake --build build --target test
```

结果：

```text
100% tests passed, 0 tests failed out of 1
checks=1447 failures=0
```

Windows 后端交叉编译：

```text
[100%] Built target uid166
```

可执行文件仅依赖系统 DLL：

```text
d3d11.dll
D3DCOMPILER_47.dll
KERNEL32.dll
msvcrt.dll
USER32.dll
```

## 2. HLSL 入口

用 HLSL 前端编译全部 7 个入口：

```text
SceneVS
SkinnedVS
ScenePS
StaticShadowVS
SkinnedShadowVS
FullscreenVS
FullscreenPS
```

结果：

```text
all 7 HLSL entry points compiled
```

此外，真实 Windows D3D11 运行时已通过 `D3DCompile` 创建全部着色器并渲染 120 帧。

## 3. 5000 物体、120 帧基准

命令：

```powershell
uid166.exe --benchmark --frames 120 --objects 5000 --seed 166 ^
  --data-dir generated --output benchmark_final.json
```

Windows 结果：

```json
{
  "frameStatsHash": -5703625736082534301,
  "validationPassed": true,
  "minVisible": 1056,
  "maxVisible": 1511,
  "averageVisible": 1141.825,
  "totalOccluded": 50845,
  "totalCsmUpdates": 360,
  "totalLocalShadowUpdates": 240,
  "totalShadowUpdates": 600,
  "averageRecordingUs": 71.7333333,
  "maxRecordingUs": 249,
  "budgetHitFrames": 0,
  "longestStableMatrixRun": 45,
  "stableWindowStart": 46,
  "stableWindowEnd": 90,
  "qualityLevelsSeen": 3,
  "levelCounts": [83, 20, 17, 0, 0, 0, 0, 0],
  "finalQualityLevel": 0,
  "finalRenderScale": 1,
  "finalShadowScale": 1,
  "defaultQualityRestored": true,
  "adaptiveHistoryEntries": 120
}
```

Linux 运行得到同一个 `frameStatsHash`，平均可见数、稳定窗口和自适应结果也一致。

关键检查：

- 每帧 `visible + occluded == frustumVisible`；
- 可见/遮挡 ID 集合长度与计数一致；
- 每帧输出 3 个 CSM 矩阵；
- 每个点光源输出 6 个立方体面矩阵，每个聚光灯输出 1 个矩阵；
- 每帧局光源实际更新数 <= 2；
- 第 46–90 帧相机冻结，连续 45 帧三个 CSM 矩阵逐元素相等；
- `adaptiveHistory` 固定 120 条；
- 质量等级历史为 83/20/17 帧，最终回到等级 0。

## 4. 跨平台确定性

Linux 与 Windows 独立生成的数据：

```text
scene SHA-256      98f186b3a2bef61a19f1e6f00504d51dd5f564009703bea8853fa6c1f1ce2341
animation SHA-256  837537cde427bd7f2c6de856a1bb4d04c48ad7893073221ca2110be387b8d94b
frameStatsHash     -5703625736082534301
```

Windows 同机重复运行 120 帧两次，`frameStatsHash` 完全一致。

## 5. 真实 D3D11 窗口运行

命令：

```powershell
uid166.exe --live-frames 120 --width 1280 --height 720 ^
  --reuse-data --data-dir generated
```

结果：

- 退出码 `0`；
- 120 行逐帧统计；
- CSM 每帧更新 3 级；
- 局部光源每帧更新 2 个；
- 实际硬件遮挡查询每帧产生 0–766 个查询与对应结果（首帧无历史遮挡体时为 0）；
- 稳定帧 GPU timestamp 约 0.8–4.3 ms，首帧回退值为 16.67 ms；
- 线程池命令记录约 37–302 微秒；
- 无设备移除、着色器、交换链或资源错误。

完整输出：`artifacts/windows/live_120frames.log`。
窗口抓帧：`artifacts/windows/frame_capture.png`（PrintWindow 在 DPI 缩放下只覆盖客户区左上部分，但可见城市、角色和光照均正常）。

## 6. 错误码

Linux 验证脚本：

```bash
./tools/verify_errors.sh ./build/uid166
```

Windows 直接执行并读取 `%ERRORLEVEL%`：

```text
missing scene      -> 2001
missing animation  -> 2101
truncated animation-> 2103
invalid JSON       -> 2003
```

所有路径均只输出 stderr 诊断并以非零码退出，没有崩溃。

## 7. 复验命令

```bash
./build/uid166_core_tests
./build/uid166 --benchmark --frames 120 --objects 5000 \
  --data-dir artifacts/generated --output /tmp/benchmark.json
python3 tools/verify_benchmark.py /tmp/benchmark.json
./tools/verify_errors.sh ./build/uid166
```

