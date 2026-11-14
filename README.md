# terrain_baker — 离线地形烘焙管线

面向游戏地形的离线烘焙管线：输入为程序化生成的高程场与视点参数，输出多层级
LOD 索引网格与配套的世界空间法线图（RGB8，Z 轴向上）。纯 CPU、纯 Python +
numpy + Pillow，无 GPU、无外部数据集，全部样本由内置种子确定性生成，可在
Windows 10/11（Python 3.10+）上复现运行。

## 安装与运行

```bat
pip install numpy pillow
python -m terrain_baker --config config.example.json --output out --jobs 4
python -m terrain_baker --config config.example.json --output out --resume --jobs 4
```

退出码：`0` 成功；`1` 配置/用法错误；`2` 存在法线一致性超标块；`3` 存在失败任务。

运行测试：

```bat
python -m unittest discover -s tests -v
```

## 目录结构

- `terrain_baker/heightfield.py` — 确定性高程场生成器（splitmix64 哈希梯度噪声
  fBm，仅用 uint64 整数运算，跨平台、跨 numpy 版本逐位一致），支持按块查询与
  钳制采样（`sample_grid` / `sample_world` / `chunk_grid`）。
- `terrain_baker/normalmap.py` — 世界空间法线图烘焙器。求导模板与网格三角面片
  法线严格一致（见下），边界使用钳制（edge-replicate）采样。
- `terrain_baker/lod_mesh.py` — LOD 索引网格生成器、边缘顶点索引、跨 LOD 缝合
  条带（`build_stitch_strip`）、基于视点距离/屏幕误差的 LOD 选择。
- `terrain_baker/consistency.py` — 一致性校验：网格面法线面积加权累加（边界补
  钳制幽灵面），逐顶点对比法线图，超阈值记为错误块。
- `terrain_baker/jobs.py` — 多线程作业队列（有序提交、无序完成）+ 原子状态文件
  （`<out>/_state/bake_state.json`）+ 崩溃/失败恢复。
- `terrain_baker/report.py` — JSON 报告（每块的 LOD 层级、法线图路径、一致性
  误差统计、失败原因，错误码为字符串业务码）。
- `terrain_baker/cli.py` — 命令行入口（`--config` / `--output` / `--resume` /
  `--jobs`）。

## 关键设计

### 法线图与网格的边界一致性

网格采用固定对角线（SW→NE）三角剖分。对该剖分做面积加权面法线累加，内部顶点
等价于一个确定的 3×3 求导模板；边界顶点若补上"钳制幽灵四边形"（幽灵顶点位置
外推、高度钳制等于边缘值），则整块等价于对边缘复制填充后的高度场施加同一模
板。法线图烘焙器直接使用该模板（`normalmap._KX/_KY`，Nz = 6·s²），因此法线图
与网格面法线在**包括边界的所有顶点**上一致到浮点精度（实测 < 2e-6°，唯一残差
来自 RGB8 量化，约 ≤0.35°）。同一块内网格与法线图共享同一套高程采样
（`chunk_grid` 的 step）与钳制规则。

### LOD 无裂缝切换

LOD L 的顶点网格是 LOD 0 每隔 2^L 个顶点的子集，世界坐标完全重合；相邻同
级 LOD 块共享边界顶点位置。相邻不同级 LOD 之间用 `build_stitch_strip` 生成缝
合三角形条带：细边每个顶点都被引用且落在粗边对应线段上，消除 T 型裂缝。

### 多线程与失败恢复

任务按 (cy, cx, lod) 确定顺序提交到 `ThreadPoolExecutor`，以 `as_completed`
无序回收。每次状态迁移都原子写入状态文件（临时文件 + `os.replace`）。进程意
外终止后：

- `done`：跳过，绝不重烘焙；
- `running`：视为被中断，重新排队；
- `failed`：记录字符串业务码（如 `E_TASK_EXECUTION`），`--resume` 时重试。

配置指纹（SHA-256）写入状态文件；`--resume` 时配置变更会提示并从头开始。

### 故障注入（仅用于测试）

- `TERRAIN_BAKER_FAIL_TASKS="cx,cy,lod;..."`：指定任务抛出异常；
- `TERRAIN_BAKER_CRASH_AT="cx,cy,lod"`：指定任务内 `os._exit(3)` 硬崩溃。

## 输出布局

```
out/
  chunk_000_000/mesh_lod0.npz  normal_lod0.png  ... (每块每级)
  _state/bake_state.json       # 任务状态（崩溃恢复用）
  report.json                  # 汇总报告
```

网格以 `.npz` 持久化（vertices/indices/edges/spacing，可用
`ChunkMesh.load` 读回）；法线图为 RGB8 PNG，图像行 0 对应网格 y=0（南缘）。

## 实测验证结果（本仓库环境，Python 3.14 + numpy 2.5 + Pillow 13）

- `python -m unittest discover -s tests`：19 个测试全部通过（确定性、钳制采
  样、边界求导、LOD 对齐、缝合覆盖、一致性、崩溃恢复、失败重试、退出码）。
- 示例配置（seed=20261115，257²，4×4 块 × 3 级 LOD = 48 任务，`--jobs 4`）：
  全部成功，退出码 0；法线图/网格最大角度偏差 0.350°（RGB8 量化极限内，
  阈值 2.0°）；报告 `consistency_errors = 0`。
- 崩溃恢复：`TERRAIN_BAKER_CRASH_AT=2,1,1` 时进程以退出码 3 终止，状态文件
  保留 15 done / 33 running；`--resume` 后 33 个中断任务重新排队完成，已完成
  任务未重烘焙（文件 mtime 不变，测试断言），最终 48/48 完成，退出码 0。
- 失败重试：`TERRAIN_BAKER_FAIL_TASKS=0,1,1` 时退出码 3，状态记录
  `E_TASK_EXECUTION`；`--resume` 后任务成功，报告 `failed = 0`。
