# 验证记录

验证日期：2026-11-14

## 环境

| 项目 | 值 |
|---|---|
| 操作系统 | Ubuntu 26.04.1 LTS |
| 编译器 | GCC 15.2.0 |
| CMake | 3.30.5 |
| Python（端到端测试） | 3.14.4 |
| 构建类型 | Release；另做 ASan/UBSan Debug |

当前执行环境没有 MSVC/Windows SDK，因此未在本机实际运行 Windows `.exe`。Windows 专用路径已实现为 `wmain`、UTF-8 参数转换和 `MoveFileExW(..., MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)`，CMake 提供 VS 2022 构建入口；Windows 实机可按 README 的命令验收。下列业务与字节一致性结论来自同一 C++ 实现的本机 Linux 实测。

## 构建与测试命令

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
/home/liang/tools/cmake-3.30.5-linux-x86_64/bin/ctest --test-dir build --output-on-failure
```

结果：

```text
Start 1: lf_baker_unit
1/2 Test #1: lf_baker_unit ....................   Passed    0.00 sec
Start 2: lf_baker_integration
2/2 Test #2: lf_baker_integration .............   Passed    7.26 sec

100% tests passed, 0 tests failed out of 2
```

命令行用法错误、解析、校验和 I/O 退出码均在集成测试中断言；目录不可写和缓存不可写路径会在任何 `FRAME` 输出或帧文件产生前失败。

## 样本烘焙

样本：`samples/scene.json`，96×96，8 个方向，3 层材质，2 个光源。

```text
首轮：  SUMMARY 8 0 8
全命中：SUMMARY 0 8 8
```

首轮输出与全缓存命中输出的全部 9 个文件（8 PNG + manifest）逐字节相同：

| 文件 | SHA-256 |
|---|---|
| `frame_000000.png` | `30eae107013678e602ef80de977df45613a3e5f143f25cb5896e5caef2f1fe17` |
| `frame_000001.png` | `27d5ea0d73d54b928fa89228a78a42ceec6777374ec44bde530897c223dcc7b9` |
| `frame_000002.png` | `6803de71db871fe920e0ed5da80750f441c804fc43a217fc98a66f51ea494d65` |
| `frame_000003.png` | `6fe202858a5e6452177bea93db89bd7e63c84345ed8fc98871cf7b1a43af899a` |
| `frame_000004.png` | `c3cbcaf8f3585fe1f143862bb2b3305396514180dbe8d6c99a92cc73e906035b` |
| `frame_000005.png` | `90ce1aaa363247dff6ff09f785d46f89b1f10f8ebbb3b92383a49a0f368989f4` |
| `frame_000006.png` | `0725c2cca3dd84c9d2b551751a9ae8ab5324d665a3bdfd7083e8b2baba79fe2d` |
| `frame_000007.png` | `229bb8146c293be542a7281a18500b9e20b5acc67630a35246d521e7492e4a74` |
| `manifest.json` | `8dfe134e791c90fd0b9a9d02bbd4d2a7faeb1a91e6cdba75063afcd3a8c0d0c0` |

## 增量失效

只把第 4 帧（索引 3）的采样方向从 `(0.3826834, 0, 0.9238795)` 改为 `(0.35, 0.12, 0.93)`：

```text
SUMMARY 1 7 8
```

逐文件哈希比较只有 `frame_000003.png` 改变，其余 7 帧及未变化的子键缓存直接复用。集成测试另外验证：

- 材质粗糙度修改：`SUMMARY 12 0 12`，所有帧子键失效，但全局键不变。
- 分辨率修改：所有帧失效并生成新的 64 位十六进制全局键。
- 帧列表缩短：新的全局键下烘焙，并删除输出目录中不再属于序列的旧 `frame_*.png`。

## 中断恢复

300 帧压力样本实测：

```text
SIGINT exit=0
SUMMARY 3 0 300
manifest exists=False, PNG count=3
SIGINT resume exit=0
SUMMARY 297 3 300

SIGKILL exit=-9
PNG count=3, manifest exists=False
SIGKILL resume exit=0
SUMMARY 297 3 300
```

`SIGINT` 在被捕获后完成当前帧提交并以 0 退出；强制 `SIGKILL` 无法给出退出报告。两条恢复路径完成后分别与正序基线比较，PNG 序列与 `manifest.json` 全部逐字节相同。中断运行不会发布不完整清单。

## 缓存半成品

集成测试实际删除了一帧的元数据、写入了损坏的 RGBA 缓存，并留下 `.tmp-leftover`：

```text
缺失元数据：  SUMMARY 1 11 12
损坏像素校验：SUMMARY 1 11 12
```

两次均未失败，只重渲受影响帧；残留临时文件被启动清理，修复后的输出与基线逐字节一致。

## 错误码

| 场景 | 退出码 | stderr 实测 |
|---|---:|---|
| 缺失 `roughness` | 2 | `E_SCENE_PARSE: missing required field 'roughness' in layers[0]` |
| `roughness = 1.5` | 3 | `E_SCENE_VALID: layers[0].roughness must be within [0, 1]` |
| `--output` 指向普通文件 | 4 | `E_OUTPUT_IO: cannot create directory ...: Not a directory` |
| `--cache` 指向普通文件 | 5 | `E_CACHE_IO: cannot create directory ...: Not a directory` |

此外集成测试覆盖了错误 JSON、字段类型错误、负法线扰动强度、33 层材质和零方向向量，并确认失败路径没有 `FRAME` 输出或部分 PNG。
场景颜色同时接受归一化 `[0,1]` 与 0-255 美术色，集成测试使用 `[194,51,26]`、`[255,242,219]` 完成了一轮烘焙。

## 内存与未定义行为检查

使用以下配置重新构建并运行全部测试：

```text
-fsanitize=address,undefined -fno-omit-frame-pointer
ASAN_OPTIONS=detect_leaks=1:abort_on_error=1
UBSAN_OPTIONS=halt_on_error=1
```

结果：

```text
1/2 Test #1: lf_baker_unit ....................   Passed    0.01 sec
2/2 Test #2: lf_baker_integration .............   Passed   25.78 sec

100% tests passed, 0 tests failed out of 2
```
