# 可回放赛局解说母带导出器 — 技术说明

仅依赖 Python 3.11 标准库（`wave`、`array`、`struct`、`hashlib`、`json`、
`threading`、`unittest` 等），无外部服务、无第三方包、无容器。所有样本由
`replay_exporter/samplegen.py` 用种子化 LFSR 与正弦波确定性生成。

## 1. 模块结构

| 模块 | 职责 |
| --- | --- |
| `errors.py` | 稳定业务错误码 `ErrorCode` 与结构化异常 `ExporterError`（含段号、毫秒偏移、上下文） |
| `audioio.py` | RIFF/WAVE 解析（检测非整帧）、48kHz 单声道读取、可随机寻址的立体声母带写入器 |
| `planner.py` | 输入校验、毫秒→帧换算、不相交时间轴布局、交叉淡化区间、长度恒等式 |
| `model.py` | 段、交叉淡化、计划、区块、快照、增益报告的数据类 |
| `dsp.py` | 纯函数 DSP：淡化曲线、闪避包络、增益插值、响度/峰值增益、硬限幅、分块合成 |
| `snapshot.py` | 版本化确定性快照、规范化 JSON、区块输入 SHA-256、版本/哈希对账与废弃清单 |
| `checkpoint.py` | 原子检查点（临时文件 + `os.replace`）、半写/截断块识别 |
| `locks.py` | 应用层导出锁（进程内注册表 + 锁文件 + 心跳/死进程接管）与带锁只读计划缓存 |
| `engine.py` | 规划→快照→分块渲染→检查点→增益报告的编排与断点恢复 |
| `validator.py` | 母带回放校验：起点对齐、交叉淡化能量连续、闪避包络、逐样本参考一致性 |
| `samplegen.py` | 确定性本地样本工程生成 |
| `exporter.py` / `cli.py` | 门面 API 与命令行入口 |

## 2. 关键设计决策

### 2.1 不相交时间轴与“样本级不重叠”

每段占据严格不相交的母带区间，相邻段之间的 `gap` 渲染为数字静音，因此
任意两段在样本层面永不重叠。长度恒等式为：

```
总帧数 = 各段帧数之和 + 各边界 gap 帧数之和
```

该恒等式由 `planner.verify_length_invariant` 强制校验。每个边界生成一条
`CrossfadeRegion`，把左段淡出尾、（可能为空的）gap、右段淡入头合并描述为一个
交叉淡化区间。淡化采用线性曲线，端点增益精确为 0，保证拼接处无点击声。

### 2.2 确定性与分块独立性

所有包络都由“绝对母带帧号 + 规范化计划”解析计算，`render_block` 是无状态纯
函数：不从历史块累积任何状态。因此从任意块起点续渲与一次性整渲逐字节一致
（见 `tests/test_block_determinism.py` 与 `test_render.py`）。

### 2.3 两级哈希

* 快照中的 `input_hash`：渲染前即可计算，对影响该块的规范化计划与源文件字节
  求 SHA-256，跨机器稳定，用于判断块是否可复用。
* 检查点中的块输出哈希：实际渲染 PCM 的 SHA-256，恢复时回读母带对应字节比对，
  识别半写块与截断文件。

### 2.4 版本不匹配的处理

* 旧快照版本号与当前 `SNAPSHOT_VERSION` 不一致：所有旧块进入
  `deprecated_blocks`（原因 `version_mismatch`），整份替换快照，旧偏移一律不沿用。
* 版本相同但输入哈希变化：原因 `input_hash_mismatch`，仅相关块作废重算。
* 检查点版本与快照版本冲突：返回 `E4001_SNAPSHOT_VERSION_MISMATCH`，拒绝混合
  不兼容偏移，且不破坏既有检查点。

### 2.5 断点恢复与半写块

母带 WAV 头按计划总时长预先写好，续渲时按块偏移 seek 覆盖。恢复流程先以
“不自动重建”模式打开母带，对检查点中的每个块回读并校验 PCM 哈希；文件截断、
撕裂写、零填充都会导致哈希不符而被标记 `half_written_or_hash_mismatch` 重算，
随后才初始化/补齐容器。检查点始终经临时文件 + `fsync` + `os.replace` 原子落盘。

### 2.6 响度归一化与限幅

每段测量人声/环境混合的峰值与 RMS，分别计算响度增益（目标 RMS）与峰值增益
（目标峰值），取两者较小值，并在段边界做线性平滑插值。限幅器把归一化浮点
四舍五入到 int16 后硬限到 `-1 dBFS`（上限 29203），保证不溢出。每段实际增益、
测量峰值/RMS、是否平滑、限幅是否激活都写入 `gain_report.json`。

### 2.7 并发协调

* 不同输出目录可并发导出；同一目录的重复启动由进程内注册表 + 原子锁文件确定性
  拒绝，返回 `E5001_OUTPUT_DIR_BUSY`，不依赖操作系统咨询锁的跨平台偶然行为。
* 锁文件带 PID 与 UTC 心跳；死进程锁或超时锁可被安全接管。
* 共享只读计划解析由 `PlanCache` 加锁保护，缓存键含全部输入文件哈希，输入变更
  时键随之变化，杜绝竞态下的脏缓存。

## 3. 业务错误码

| 代码 | 触发场景 |
| --- | --- |
| `E1000_INVALID_JSON` | 时间轴/元数据 JSON 非法 |
| `E1001_FILE_NOT_FOUND` / `E1002_FILE_NOT_WRITABLE` | 文件缺失/目录不可写 |
| `E1003_INVALID_WAV` | 非 RIFF/WAVE、缺 fmt/data、非 PCM |
| `E2001_NEGATIVE_DURATION` | 负时长/负淡化时长 |
| `E2002_SEGMENT_ORDER_CONFLICT` | 段区间重叠/段序冲突 |
| `E2003_FADE_LONGER_THAN_SEGMENT` | 淡入+淡出超过段长 |
| `E2004_MISSING_DIALOGUE_RANGE` | 缺对白起止字段 |
| `E2005_INVALID_DIALOGUE_RANGE` | 对白区间非法（负、倒置、越界），带毫秒偏移 |
| `E2006_MISSING_METADATA_FIELD` / `E2007_NEGATIVE_GAP` / `E2008_INVALID_TIMELINE` | 字段缺失/负间隔/时间轴结构错 |
| `E3001_SAMPLE_RATE_MISMATCH` | 人声/环境采样率不一致或非 48kHz |
| `E3002_NON_INTEGER_FRAME` | data 块字节数非整帧，带段号与毫秒偏移 |
| `E3003_UNEXPECTED_CHANNELS` / `E3004_UNEXPECTED_SAMPLE_WIDTH` | 非单声道/非 16 位 |
| `E4001_SNAPSHOT_VERSION_MISMATCH` / `E4002_BLOCK_HASH_MISMATCH` / `E4003_CHECKPOINT_CORRUPT` | 快照/块/检查点恢复异常 |
| `E5001_OUTPUT_DIR_BUSY` | 同目录重复导出 |

失败时除抛出异常外，还会向输出目录原子写入 `error_report.json`，且不修改既有
检查点。

## 4. 运行方式

```bash
# 生成确定性样本工程
python3 -m replay_exporter.cli generate-samples demo_work/project --segments 4

# 渲染（可重复调用以断点续渲）；--stop-after-blocks 用于模拟中断
python3 -m replay_exporter.cli export demo_work/project/timeline.json demo_work/out

# 回放校验报告
python3 -m replay_exporter.cli validate demo_work/out

# 单元测试
python3 -m unittest discover -s tests -t .

# 端到端验收（含真实子进程强制 kill 后续渲）
python3 scripts/acceptance_demo.py
```

## 5. 实际验证结果

在本机（Python 3 标准库环境）执行：

* `python3 -m unittest discover -s tests -t .` → **34 个测试全部通过**。
* `python3 scripts/acceptance_demo.py` → `overall_passed = true`，退出码 0。
  * 干净整渲与“强制终止后续渲”的 `master.wav` SHA-256 完全相同（逐字节一致）。
  * 校验报告 4 段全部通过、3 个交叉淡化边界能量连续（增益归一化 RMS 跳变均
    小于 0.01 容差 0.08），闪避包络零违规。
  * 结构化错误路径返回 `E2004_MISSING_DIALOGUE_RANGE` 并定位到段 0。
  * 3 个并发导出到不同目录全部成功，结束后无残留 `export.lock`。

母带规格经测试确认为 48kHz、16 位、立体声 PCM，最大样本不超过 -1 dBFS 上限
29203；限幅专项测试用满量程声源验证硬限幅生效且无 int16 溢出。
