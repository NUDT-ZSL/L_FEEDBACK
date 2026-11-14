# Verification Record

验证日期：2026-11-15

## 当前环境

- Linux x86_64，GCC 15.2，CMake 3.30.5
- GLFW 3.4 / GLM / GLEW 临时解包用于编译检查
- glslang 16.2.0
- 当前 WSL 环境无可用 OpenGL 4.5 上下文；`--headless` 仍需要真实 OpenGL 上下文，GPU 运行验证应在 Windows 目标机执行

## 核心测试

命令：

```bash
cmake -S . -B build -DLIGHTCLUSTER_BUILD_RENDERER=OFF
cmake --build build -j
./build/lightcluster_core_tests
```

结果：

```text
All 675 checks passed
```

覆盖：

- 每次增删改版本号递增。
- 双缓冲快照并发读写一致性（2000 次写入）。
- 动画位置/光强插值、完成后回收、池满业务码。
- 版本不变跳过上传。
- 容量复用与超限扩容。
- 延迟 3 帧上传及多快照淘汰。
- 快照恢复后版本严格增加。
- GPU 回读计数、类型计数、版本、光源负载四类错误。
- 固定种子即时/延迟序列不变量和最终收敛。

## CLI 端到端测试

命令：

```bash
python3 scripts/validate_cli.py build/lightcluster_sim
```

结果：

```text
CLI validation passed: immediate, deferred, snapshot, debug, determinism
```

断言包括：

- 48 帧内每帧至少 64 个点光源、至少 4 个方向光源，类型计数之和等于总数。
- 立即模式帧 0 版本不变、`uploaded=false`；同容量版本变化时上传但 `buffer_grew=false`。
- 固定种子在帧 20-29 产生扩容尖峰，帧 26 的光源数从 128 增至 129，SSBO 容量由 128 扩至 256，后续版本复用该容量。
- 延迟模式 `N=4`：游标滞后不超过 4，实际出现 `snapshot_age_frames=4` 的上传，静默尾部最终 `gpu_version == scene_version` 且有效快照年龄归零。
- 池容量 4 时产生 `ANIM_POOL_FULL` JSON 业务事件。
- 第 8 帧保存、第 20 帧恢复时，恢复后版本 83 严格大于恢复前版本 82 和快照版本 33。
- 注入 GPU 计数错误时只执行 1 帧、输出 `GPU_COUNT_MISMATCH`、退出码 75。
- 相同种子两次运行的 stdout 完全相同。

## Sanitizer

命令：

```bash
cmake -S . -B build-asan -DLIGHTCLUSTER_BUILD_RENDERER=OFF \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build-asan -j
ASAN_OPTIONS=detect_leaks=1 ./build-asan/lightcluster_core_tests
python3 scripts/validate_cli.py build-asan/lightcluster_sim
```

结果：

```text
All 675 checks passed
CLI validation passed: immediate, deferred, snapshot, debug, determinism
```

无 ASan、UBSan 或泄漏报告。

## 120 帧长序列

```bash
./build/lightcluster_sim --frames 120 --sync deferred --delay 5 \
  --pool-capacity 8 --seed 20261115 --debug
```

结果：

```text
frames=120
final_scene_version=913
final_gpu_version=913
final_debug_ok=true
max_snapshot_age=5
max_active_lights=156
final_buffer_capacity=256
```

同一构建执行“第 20 帧保存、第 60 帧恢复”的长序列：保存版本 149，恢复前版本 545，恢复后版本 546，严格递增；120 帧全部通过 GPU 镜像回读校验。

## OpenGL 目标与 GLSL

使用 GLFW 3.4、GLM、GLEW 对 `src/app/main.cpp` 和 `src/app/renderer.cpp` 完成 C++17 编译链接：

```text
/tmp/lightcluster_renderer  (ELF executable)
```

GLSL 450 五个阶段使用 `glslang -S ...` 校验：

```text
all shaders valid
```

WSL 中运行 GPU 可执行文件时，GLFW 能创建 EGL 路径窗口，但 GLEW 因无可用 GLX/4.5 驱动返回 `Unknown error`，程序正确输出：

```json
{"event":"fatal","business_code":"GPU_INIT_FAILED","message":"glewInit failed: Unknown error"}
```

退出码为 3。该结果仅说明当前容器无 GPU 上下文，不代表 Windows 目标行为。

## Windows 验收建议

在 Windows 机器上执行：

```powershell
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure

.\build\Release\lightcluster_sim.exe `
  --frames 120 --sync deferred --delay 5 --pool-capacity 8 --debug > sim.jsonl

.\build\Release\lightcluster.exe `
  --frames 120 --sync deferred --delay 5 --pool-capacity 8 `
  --debug --headless > gpu.jsonl

.\build\Release\lightcluster.exe `
  --frames 3 --sync immediate --debug --headless --inject-debug-fault
# 预期 $LASTEXITCODE = 75
```

检查 `gpu.jsonl` 最后一个 `event=frame` 行：`gpu_version` 应等于 `scene_version`，`debug_ok=true`。
