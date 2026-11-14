# Dynamic Light Cluster Deferred Renderer

Windows OpenGL 4.5 Core 延迟着色渲染器。项目从空工作区实现，包含：

- G-Buffer 几何通道（世界坐标、法线、反照率）和全屏四边形光照通道。
- 64 个初始点光源 + 8 个初始方向光源，运行时按固定种子增删改。
- 前端场景双缓冲快照和单调递增全局版本号。
- 立即同步与最大 N 帧延迟同步；版本不变跳过上传，容量内复用 SSBO。
- 固定容量动画任务池，按帧插值位置/光强，池满返回 `ANIM_POOL_FULL`。
- 完整场景快照保存/恢复，恢复版本严格大于恢复时刻的版本。
- 调试模式 GPU SSBO 回读，逐字节比对 CPU 期望，不一致退出码 `75`。
- 每帧输出确定性 JSON 行日志。

## 目录

```text
include/lightcluster/   公共接口
src/core/               场景、版本、快照、动画池、同步器、确定性序列
src/app/                Windows/OpenGL 渲染器
src/sim_main.cpp        可移植逻辑模拟器（用于 CI 和本地验证）
tests/core_tests.cpp    核心单元/并发/延迟收敛测试
scripts/validate_cli.py 端到端 JSON 和退出码断言
docs/                   架构与实测记录
```

## Windows 构建

前置条件：Windows 10/11、支持 OpenGL 4.5 的驱动、Visual Studio 2022、CMake 3.20+。

```powershell
git clone https://github.com/microsoft/vcpkg.git C:\tools\vcpkg
C:\tools\vcpkg\bootstrap-vcpkg.bat

cmake -S . -B build `
  -DCMAKE_TOOLCHAIN_FILE=C:/tools/vcpkg/scripts/buildsystems/vcpkg.cmake `
  -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build build --config Release
```

`vcpkg.json` 以 manifest 模式声明 `glfw3`、`glm`、`glew`。首次配置时 vcpkg 会按仓库当前版本安装依赖。若使用 MinGW-w64，请使用对应 triplet，例如 `x64-mingw-dynamic`。

主要产物：

```text
build\Release\lightcluster.exe       OpenGL 渲染器
build\Release\lightcluster_sim.exe   无 OpenGL 依赖的逻辑模拟器
build\Release\lightcluster_core_tests.exe
```

关闭渲染目标、只构建可移植验证程序：

```powershell
cmake -S . -B build-core -DLIGHTCLUSTER_BUILD_RENDERER=OFF
cmake --build build-core --config Release
ctest --test-dir build-core -C Release --output-on-failure
```

## 运行

立即同步并开启逐帧 GPU 回读，使用隐藏窗口进行本地验证：

```powershell
.\build\Release\lightcluster.exe `
  --frames 120 --sync immediate --pool-capacity 8 --seed 20261115 `
  --debug --headless
```

延迟同步，最多滞后 5 帧：

```powershell
.\build\Release\lightcluster.exe `
  --frames 120 --sync deferred --delay 5 `
  --pool-capacity 8 --seed 20261115 --debug --headless
```

快照保存/恢复验证：

```powershell
.\build\Release\lightcluster.exe `
  --frames 120 --sync immediate --debug --headless `
  --save-snapshot-frame 20 --restore-snapshot-frame 60
```

主动破坏 GPU 缓冲计数，验证调试失败路径：

```powershell
.\build\Release\lightcluster.exe `
  --frames 3 --sync immediate --debug --headless --inject-debug-fault
# 预期退出码 75，最后一行 business_code=GPU_COUNT_MISMATCH
```

命令行参数：

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `--frames N` | 运行帧数 | `120` |
| `--sync immediate\|deferred` | 同步模式 | `immediate` |
| `--delay N` | 延迟模式最大上传滞后帧数 | `0` |
| `--pool-capacity N` | 固定动画槽位数 | `8` |
| `--seed N` | 固定随机种子，支持 `0x` 十六进制 | `20261115` |
| `--save-snapshot-frame N` | 第 N 帧后保存完整快照 | 未设置 |
| `--restore-snapshot-frame N` | 第 N 帧开始时恢复，必须大于保存帧 | 未设置 |
| `--debug[=0\|1]` | 每帧回读并校验 GPU 光源缓冲 | 关闭 |
| `--inject-debug-fault[=0\|1]` | 仅用于验证错误退出路径 | 关闭 |
| `--headless[=0\|1]` | 创建隐藏 GLFW 窗口，仍使用真实 GPU | 关闭 |

## 同步语义

初始场景先完成一次同步，避免延迟模式下前 N 帧没有可渲染光源。之后：

- `immediate`：当前快照版本与 resident 版本不同才上传；相同时跳过上传。
- `deferred N`：每个新版本记录 `ready_frame = source_frame + N`。每帧选择已就绪的最新快照上传，旧快照被淘汰。GPU 始终持有某个完整历史快照，渲染只读取 resident 版本，不使用半更新 CPU 数据。
- 容量按 64 起步、按 2 的幂增长；只有所需数量超过容量才调用 `glBufferData` 扩容，否则只调用 `glBufferSubData` 复用现有 SSBO。
- 固定种子序列最后保留静默窗口，足以让动画结束且延迟 N 帧的上传完成，因此运行末尾 `gpu_version == scene_version`。
- 版本相等时无需上传；有效快照来源帧会前移到当前帧，使 `snapshot_age_frames` 归零。

调试模式比较的是“调度器认定的 GPU resident 快照”与真实 GPU 回读，而不是延迟模式下的最新前端快照。这样既能逐帧发现真实缓冲错位，也不会把设计上允许的 N 帧滞后误判为错误。最新前端版本仍在 `scene_version` 字段输出。

## JSON 日志

每个输出行都是独立 JSON。`event=frame` 行包含：

```json
{"event":"frame","frame":20,"sync_mode":"deferred","delay_frames":5,"scene_version":83,"gpu_version":72,"active_lights":74,"point_lights":64,"directional_lights":10,"uploaded":true,"upload_reason":"deferred_due","buffer_capacity":128,"buffer_grew":false,"gpu_light_count":70,"gpu_point_count":61,"gpu_directional_count":9,"gpu_source_frame":15,"snapshot_age_frames":5,"pending_snapshots":5,"animation_active":8,"animation_rejected_total":12,"debug_checked":true,"debug_ok":true}
```

其他业务事件：

- `animation_queued`：任务已进入固定容量池。
- `animation_rejected`：`business_code=ANIM_POOL_FULL`，不会阻塞或静默丢弃。
- `light_added` / `light_removed`：场景版本已递增。
- `snapshot_saved` / `snapshot_restored`：恢复事件包含 `version_strictly_increased=true`。
- `debug_error`：GPU 回读不一致，渲染循环立即终止。
- `fatal`：参数、初始化或运行期致命错误。

退出码：

| 退出码 | 含义 |
| --- | --- |
| `0` | 成功 |
| `2` | 参数或运行期逻辑错误 |
| `3` | OpenGL/GLFW/GLEW 初始化失败 |
| `4` | 上传/渲染/回读调用失败 |
| `75` | GPU 缓冲回读校验不一致 |

## 本地验证

当前 Linux 环境没有可用的 OpenGL 4.5 上下文，因此实际执行的是与 Windows 渲染器共享同一套核心逻辑的模拟器、单元测试、ASan/UBSan、CLI JSON 断言和 GLSL 编译器校验。OpenGL 目标也已用 GLFW 3.4、GLM、GLEW 完成编译。

```bash
cmake -S . -B build -DLIGHTCLUSTER_BUILD_RENDERER=OFF
cmake --build build -j
./build/lightcluster_core_tests
python3 scripts/validate_cli.py build/lightcluster_sim
```

详细实测数据见 `docs/VERIFICATION.md`，设计说明见 `docs/ARCHITECTURE.md`。
