# rigskin

`rigskin` 是一个从零实现的离线骨骼蒙皮与网格变形预览管线，使用 C++17 标准库，
不依赖图形库、第三方运行时、网络或外部资源。内置确定性 ASCII 样本包含 13 根
骨骼、355 个顶点、649 个三角形、非均匀关键帧、逐顶点混合权重，以及一个用于
异常路径验证的零权重顶点和一个退化三角形。

## 构建

Linux 或 MinGW-w64：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
build/rigskin_unit_tests
build/rigskin --threads 4 --snapshot-dir build/snapshot --frames 24
```

MSVC 2019：

```powershell
cmake -S . -B build-msvc -G "Visual Studio 16 2019" -A x64
cmake --build build-msvc --config Release
.\build-msvc\Release\rigskin.exe --threads 4 --snapshot-dir .\snapshot --frames 24
```

## 命令行

```text
rigskin --threads N --snapshot-dir DIR [options]

--threads N            工作线程数，1..4096
--snapshot-dir DIR     快照与输出目录
--frames N             时间轴帧数，默认 24
--width N              PPM 宽度，默认 640
--height N             PPM 高度，默认 480
--sample NAME          内置样本名，目前仅 preview
--commit-delay-ms N    可选的提交后延时，用于强制终止/故障注入测试
```

线程数只影响计算调度，不参与输入哈希；因此相同样本、帧数、分辨率下，串行与
并行输出文件逐字节一致。线程数大于帧数时，实际工作线程数取
`min(threads, frames_to_compute)`，每个帧任务只被领取一次。

退出码：`0` 表示完整成功；`130` 表示捕获到 SIGINT/SIGTERM 或 Windows 控制台
中断并已写最终摘要；`2` 表示参数或不可恢复错误；Linux 下 SIGKILL 通常表现为
`137`，不要求写摘要，下次启动可恢复。

## 输出布局

```text
snapshot/
  manifest.txt                版本化快照清单
  summary.txt                 本次运行最终摘要
  input_model.txt             实际解析的 ASCII 输入
  config.txt                  参与输入哈希的管线配置
  zero_weight_report.txt      零权重顶点异常报告
  frames/
    frame_0000.mesh           变形网格、顶点法线、面法线、统计头
    frame_0000.ppm            固定相机/方向光的 P6 渲染
    ...
  errors/                     仅在错误/恢复/中断场景出现
    invalid_manifest_version.txt
    input_hash_changed.txt
    frame_recovery.txt
    fatal.txt
  quarantine/                 旧快照被整体重建前的保留副本
  .tmp/                       原子写临时目录
```

`manifest.txt` 是权威文件。每个 `entry` 同时记录帧号、清单版本、输入哈希、
网格/PPM 的 SHA-256、合并输出校验和、文件字节数，以及顶点数、三角形数和
退化三角形数。合并校验和定义为
`SHA256(mesh_file_bytes || ppm_file_bytes)`。

网格二进制采用小端固定格式：

```text
8 bytes   "FMSH0001"
u32       frame
u32       vertex_count
u32       triangle_count
u32       degenerate_triangle_count
u32       zero_weight_vertex_count
u32       reserved = 0
per vertex:   3x f32 position, 3x f32 vertex normal
per triangle: 3x u32 indices, 3x f32 winding-derived face normal
```

## 动画与蒙皮

- 骨骼层级先做拓扑校验、环检测和单根约束，再以 DFS 先序求值，保证父骨骼先于
  子骨骼。
- 每个关键帧存储局部平移和四元数旋转；采样按相邻非均匀关键帧的时间区间做
  平移线性插值、四元数最短路径 slerp，不假设固定关键帧频率。
- 局部姿态按 `T * R` 组合，全局姿态为 `parent_global * local`。
- 蒙皮矩阵为 `pose_global * inverse(bind_global)`。位置和法线均按归一化权重
  累加；零权重顶点保留绑定姿态。权重在加载阶段合并同骨骼项并归一化。
- 面法线始终由变形后三角形的索引环绕叉积计算；退化三角形写入零法线并计数，
  不影响后续帧或光栅化。

## 渲染确定性

相机由绑定姿态包围盒推导一次后固定，正交投影；方向光固定。每帧单线程执行
三角形顺序扫描和像素中心采样。Z 缓冲使用严格小于比较，深度相等时索引靠前的
三角形保留。背面剔除由面法线与相机前向的固定点积决定。PPM 头和像素缓冲区
均为无时间戳、无未初始化数据的确定性字节流。

## 恢复与损坏策略

启动时校验清单版本、输入哈希、帧数、清单字段、文件存在性、文件大小、
网格/PPM SHA-256、合并校验和、网格头计数及 PPM 尺寸。

- 清单版本无法识别：写独立的 `errors/invalid_manifest_version.txt`，把旧清单和
  帧产物移入 `quarantine/invalid_manifest_version/`，再整体重建，不静默覆盖。
- 输入哈希或帧数变化：同路径整体重建，旧产物放入
  `quarantine/input_hash_changed/`。
- 清单损坏或缺失但目录已有产物：报 `E1001/E1002`，保留旧产物后重建。
- 某帧缺失、尺寸不符或校验失败：从该帧开始重算到最后一帧，之前的有效帧复用。
  损坏事件写入 `errors/frame_recovery.txt`。
- 清单每提交一帧都通过临时文件加替换写回；帧文件也原子替换。SIGKILL 最多留下
  未入清单的孤儿文件或临时文件，下次启动会忽略或清理并重算。
- 正常结束和可捕获中断都会持久化 `summary.txt`。SIGKILL 不写摘要。

业务错误码：

```text
E1001 清单解析失败
E1002 清单缺失但存在旧产物
E1003 不支持的清单版本
E1004 输入哈希或帧数变化
E2001 帧载荷文件缺失
E2002 帧载荷文件大小不符
E2003 帧载荷 SHA-256/合并校验和不符
E2004 PPM 尺寸/头不符
E2005 网格头/长度不符
E2101 清单中缺少某帧条目
E3001 清单条目字段、版本或统计不一致
E8001 帧计算/写出异常
E9000 启动/不可恢复异常
E9001 可捕获中断且尚未完成全部帧
```

## 测试

Linux：

```bash
bash tests/run_integration.sh build/rigskin
```

Windows PowerShell：

```powershell
.\tests\run_integration.ps1 `
  -Rigskin .\build-msvc\Release\rigskin.exe `
  -UnitTests .\build-msvc\Release\rigskin_unit_tests.exe
```

集成测试覆盖单元测试、串行/并行全树哈希比较、干净续跑、损坏帧重算、未知清单
版本重建、输入哈希变化重建、SIGKILL 续跑和 SIGINT 摘要。详细设计见
`docs/DESIGN.md`，本机实测记录见 `docs/VALIDATION.md`。

