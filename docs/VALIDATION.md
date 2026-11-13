# 实际验证记录

验证日期：2026-11-14  
运行环境：WSL2 Linux x86_64，GCC 15.2.0，CMake 3.30.5，Release 构建  
说明：当前执行环境不是原生 Windows；代码使用 C++17 标准库和 Windows 条件编译，
并附 PowerShell 集成脚本，但本记录中的实际进程测试均在 Linux 完成。

## 构建与单元测试

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
build/rigskin_unit_tests
```

结果：

```text
[100%] Built target rigskin
[100%] Built target rigskin_unit_tests
rigskin unit tests: PASS
```

构建在 `-Wall -Wextra -Wpedantic` 下无警告。单元测试覆盖 SHA-256 已知向量、
ASCII 解析、单根/无环层级、父先于子、权重归一化、零权重检测、非均匀关键帧
平移插值、全局父子矩阵关系、面法线与环绕一致、退化三角形计数、网格二进制头、
固定相机渲染与 PPM 头。

额外使用 `-fsanitize=address,undefined -fno-omit-frame-pointer` 构建并运行
单元测试、8 帧多线程渲染与干净续跑，未报告内存错误或未定义行为。

## 完整集成脚本

```bash
bash tests/run_integration.sh build/rigskin
```

实际输出：

```text
PASS: unit tests
PASS: serial/parallel byte-identical output trees
PASS: threads greater than frames produces no empty or duplicate frames
PASS: clean resume reuses all frames
PASS: corrupt frame and subsequent frames recomputed
PASS: unsupported version quarantined and rebuilt
PASS: input hash change rebuilds the snapshot
PASS: SIGKILL recovery matches clean execution
PASS: SIGINT summary and resume
ALL INTEGRATION TESTS PASSED
```

## 串行/并行逐字节比较

参数：16 帧，640x480，串行 `--threads 1`，并行 `--threads 8`。

```text
diff -rq serial parallel  -> exit 0
serial tree hash listing SHA-256:
770743964c82833bdf6e90f384437ce2b0363c29674ec6924ce99c3e29ff00d6
file count: 37
frame payload files: 32 (16 .mesh + 16 .ppm)
input hash:
84590627f60533f2af19f722345bc34044875136088c5aaca50e970739d6bcdd
manifest SHA-256:
b4858fb2f5b8a86f3232df2a2e7a7c86bc3f07c9c267745e3203fe7f5c324056
```

并行快照生成相同的 tree hash listing 和相同的 manifest SHA-256。两棵目录树
（包括 `summary.txt`、`input_model.txt`、`config.txt` 和所有帧文件）逐字节一致。

额外的“线程数大于帧数”测试使用 3 帧和 `--threads 64`。串行与并行
`diff -rq` 退出码为 0，摘要仍为 `frames_success=3`，没有空帧或重复条目。

## 干净续跑

对已完整快照再次运行 3 线程：

```text
status=complete
frames_total=16
frames_success=16
frames_reused=16
frames_recomputed=0
error_codes=none
```

## 损坏帧恢复

集成脚本向 `frame_0005.mesh` 追加 1 字节，触发大小错误：

```text
status=complete
frames_success=16
frames_reused=5
frames_recomputed=11
error_codes=E2002
```

`errors/frame_recovery.txt`：

```text
E2002 frame=5 detail=frame payload size does not match manifest
policy=recompute_corrupt_frame_and_all_subsequent_frames
```

另外手工做了同大小翻转字节测试：`frame_0005.mesh` 中间字节异或 `0x5a`，得到
`E2003`，摘要仍为 `frames_reused=5`、`frames_recomputed=11`。修复后的全部
`.mesh`/`.ppm` 与 `manifest.txt` 同干净串行基线 `cmp`/`diff` 一致。

## 未知清单版本

把清单头与字段版本改为 `999` 后运行：

```text
status=complete
frames_reused=0
frames_recomputed=16
error_codes=E1003
```

独立报告 `errors/invalid_manifest_version.txt` 记录
`detected_version=999`、`expected_version=2`。旧快照共 37 个文件完整保留在
`quarantine/invalid_manifest_version/`，新帧文件与干净基线一致。

## 输入哈希变化

同一目录把宽度从 640 改为 641：

```text
old input hash:
84590627f60533f2af19f722345bc34044875136088c5aaca50e970739d6bcdd
new input hash:
5d366e5c4dfd1e9284fe838f41a533d63bf05909583ca65e2501eb08868894f6
status=complete
frames_reused=0
frames_recomputed=16
error_codes=E1004
```

旧产物 37 个文件保存在 `quarantine/input_hash_changed/`，没有沿用旧帧校验或偏移。

## 强制终止与恢复

命令使用 4 线程、24 帧、640x480、每帧提交后延时 200ms。脚本轮询清单，在已有
4 个 `entry` 时发送 SIGKILL：

```text
process exit code: 137
summary.txt exists: no
manifest complete: 0
committed entries: 4
```

随后用 5 线程恢复：

```text
status=complete
frames_total=24
frames_success=24
frames_reused=4
frames_recomputed=20
error_codes=E2101
```

恢复后的 24 对帧文件与 `manifest.txt` 同干净 24 帧基线逐字节一致。缺失条目报告为
`E2101 manifest_entry_missing frame=4`，符合“从首个缺失帧继续”的边界。

## SIGINT 摘要

同样在下发若干帧后发送 SIGINT：

```text
process exit code: 130
summary status: interrupted
frames_total=24
frames_success=4
frames_reused=0
frames_recomputed=4
error_codes=E9001
manifest complete: 0
```

摘要已持久化。再次运行后完成剩余帧，最终帧文件与清单同 24 帧干净基线一致。

## 渲染检查

把一帧 PPM 转成临时 PNG 后目视检查，图像为固定相机下的平直着色角色，包含头部、
躯干、双臂、双腿和脚部，背景与角色轮廓正常。PPM 头为固定
`P6\n<width> <height>\n255\n`，无时间戳或动态字段。

