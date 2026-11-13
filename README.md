# skinbake

`skinbake` 是一个只依赖 C++17 标准库与线程库的离线骨骼动画预计算工具。它把
关键帧曲线、骨骼层级和蒙皮网格求值成逐帧快照，并提供版本/校验和恢复、
逐帧原子进度、中断续算、片段混合与从零重算验证。全部输入样本都由
`src/scenarios.cpp` 确定性构造，不需要图形库、Docker 或网络。

## 构建

Windows 10/11 + Visual Studio 2019/2022（C++ 桌面开发工作负载）：

```powershell
cmake -S . -B build -G "Visual Studio 16 2019" -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

CMake 3.20 及以上、任意支持 C++17 的 MSVC 2019+/GCC/Clang 也可使用普通
`cmake -S . -B build && cmake --build build` 流程。工程在 MSVC 下启用 `/fp:strict`，
在 GCC/Clang 下启用 `-ffp-contract=off -fno-fast-math`，避免浮点收缩改变逐字节结果。

## 快速使用

```bash
# 60 fps、180 帧、4 个工作单元；.progress 每帧原子提交
./build/skinbake bake --scenario idle --output out/demo.snapshot \
  --frames 180 --fps 60 --version 3 --workers 4

# 从零重算并逐帧比较；容差 1e-5
./build/skinbake verify --scenario idle --file out/demo.snapshot \
  --expect-version 3

# 片段切换：第 12 帧切到 action，200 ms 顶点线性混合
./build/skinbake bake --scenario transition --output out/blend.snapshot \
  --frames 48 --fps 60 --version 1 --workers 16

# 10000 帧循环时间轴漂移检查
./build/skinbake timeline-check --frames 10000 --fps 60

# 稳定错误码演示
./build/skinbake error --scenario bezier-denominator
./build/skinbake load --file out/demo.snapshot --expect-version 9
```

成功结果写 stdout JSON；业务失败写 stderr JSON 并以非零码退出（业务错误 `2`，
验证不一致 `3`，快照拒绝 `4`，内部错误 `5`）。

## 求值与蒙皮语义

- 每个关键帧包含整数毫秒时间戳、值和入/出切线。切线是到控制点的二维偏移
  `(timeMs, value)`：段控制点为
  `P0=(t0,v0)`、`P1=P0+out0`、`P2=P3-in1`、`P3=(t1,v1)`。
- 每段要求 `P1.x-P0.x > 0`、`P2.x-P1.x > 0`、`P3.x-P2.x > 0`，否则报告
  `BEZIER_DENOMINATOR_ZERO`（1002）。参数 `u` 用固定 80 次二分求解
  `x(u)=time`，中间量全为 `double`，写入快照前才转 `float`。首尾之外常数外延。
- 骨骼局部矩阵按平移、Z 旋转、缩放生成；按父骨索引做拓扑排序，自顶向下
  `world = parentWorld * local` 累积。顶点先经各绑定骨的 world 矩阵变换，
  再按输入权重加权；权重不做归一化。
- 权重和恰为 0 报 `WEIGHT_SUM_ZERO`（2005）；与 1 的偏差
  `abs(sum-1) > 1e-4` 报 `WEIGHT_SUM_NOT_ONE`（2006）。每个顶点最多 4 个影响。

## 片段状态机与混合

`StateMachine` 维护当前片段名、片段内播放时间、单调传播时间，以及切换瞬间的
`PoseSnapshot`（全部世界矩阵和顶点位置）。切换时新片段从 0 ms 开始，旧姿态
冻结；窗口内按 `elapsedMs / blendWindowMs` 对骨骼矩阵及顶点位置做线性插值。
历史 `PoseSnapshot` 使切换首帧严格连续。

并行烘焙使用与状态机相同的纯时间轴规则：切换帧、当时旧姿态和窗口期限都在
准备阶段固化为只读 `TimelineEvaluator`，因此每帧可独立、可并行求值。

所有帧时间由 `llround(frame * 1000 / fps)` 直接算出，不从上一帧累加；循环片段
使用整数毫秒取模。单线程 10000 帧检查的最大播放时间偏差为 0 ms。

## 增量烘焙与原子性

- 帧块大小固定为 `--block-frames`（默认 8），与 worker 数无关；结果按全局帧号
  合并。共享曲线段缓存、骨架拓扑和片段查找表在工作线程中只读。
- 每提交一帧，完整进度快照先写入同目录 `<output>.progress.tmp`，`fsync` 后用
  平台原子替换改名为 `<output>.progress`。因此硬终止最多丢失未提交帧。
- 重启时先完整校验进度文件，再从 `frameCount` 继续；已提交帧不重复、不回退。
- 版本或场景指纹变化时，旧进度整体作废并写入空进度后从头开始。事件日志
  记录 `discard`、`resume`、`commit`、`complete`。
- `--crash-after-commit-frame N` 是验收用故障注入点：完成该帧原子提交并刷新
  事件日志后调用 `_Exit(137)`，不执行析构或清理。

## 快照格式

文件以 `SKNBKE01` 开头，带版本、fps、场景指纹、总帧数、已完成帧数、骨骼/顶点
数、完成标志和 CRC32。每个帧记录包含帧号、取整时间以及 3x4 骨骼矩阵和顶点
坐标。详细偏移见 [docs/SNAPSHOT_FORMAT.md](docs/SNAPSHOT_FORMAT.md)。

恢复顺序固定为：魔数 → 版本（拒绝时输出实际和目标版本）→ CRC32 → 维度和帧
数据。所有检查完成前不返回任何快照对象，因此失败不会部分应用数据。

## 错误码

| 数值 | 代码 | 含义 |
| ---: | --- | --- |
| 1001 | `CURVE_TIME_NOT_INCREASING` | 曲线时间戳非严格递增 |
| 1002 | `BEZIER_DENOMINATOR_ZERO` | 贝塞尔 x 参数化控制增量非正 |
| 1003 | `CURVE_EMPTY` | 曲线无关键帧 |
| 1004 | `CURVE_NON_FINITE` | 关键帧或切线非有限数 |
| 2001 | `SKELETON_PARENT_INVALID` | 父骨索引越界 |
| 2002 | `SKELETON_CYCLE` | 骨架存在环 |
| 2003 | `VERTEX_BONE_INDEX_INVALID` | 顶点骨骼索引非法/重复 |
| 2004 | `VERTEX_TOO_MANY_INFLUENCES` | 顶点影响数超过 4 |
| 2005 | `WEIGHT_SUM_ZERO` | 权重和为零 |
| 2006 | `WEIGHT_SUM_NOT_ONE` | 权重和偏差超过 1e-4 |
| 2007 | `WEIGHT_INVALID` | 负权重或非有限权重 |
| 3001 | `CLIP_NOT_FOUND` | 片段不存在/重名/骨曲线数不符 |
| 3002 | `CLIP_DURATION_INVALID` | 片段时长非正 |
| 3003 | `TRANSITION_INVALID` | 切换时间或窗口非法 |
| 4001 | `SNAPSHOT_MAGIC_MISMATCH` | 文件魔数错误 |
| 4002 | `SNAPSHOT_TRUNCATED` | 文件头或负载被截断 |
| 4003 | `SNAPSHOT_VERSION_MISMATCH` | 版本不匹配 |
| 4004 | `SNAPSHOT_CHECKSUM_MISMATCH` | CRC32 不匹配 |
| 4005 | `SNAPSHOT_IO_ERROR` | 读写/原子替换失败 |
| 4006 | `SNAPSHOT_DIMENSION_MISMATCH` | 帧内矩阵/顶点尺寸错误 |
| 4007 | `SNAPSHOT_DATA_INVALID` | 帧号、标志或数值非法 |
| 4008 | `SNAPSHOT_FINGERPRINT_MISMATCH` | 场景指纹不匹配 |
| 5001 | `INVALID_ARGUMENT` | CLI 或公共 API 参数非法 |
| 5002 | `INVALID_SCENARIO` | 场景名非法 |

同一输入重复运行产生相同的错误 JSON（字段顺序和数值格式固定）。

## 验证

```bash
cmake --build build
ctest --test-dir build --output-on-failure
```

单元测试覆盖曲线外延/重复位级求值、层级蒙皮、权重错误、骨架环、快照往返与
篡改、混合连续性、并行逐字节一致和前缀续算；集成测试覆盖真实子进程硬退出、
恢复、版本/指纹作废、CRC/版本拒绝、错误码重复性和 10000 帧时间轴。

本轮实测记录见 [docs/verification.md](docs/verification.md)。
