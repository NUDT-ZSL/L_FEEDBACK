# 赛事回放多语种音频离线打包器 — 技术说明

## 1. 目标与范围

把一份事件清单 JSON 与本地合成的 `{语音键}_{语种}.wav` 配音片段，离线打包为：

- 每语种一条 48kHz / 16bit / 单声道 WAV；
- CUE 式时间戳索引（`.cue` 文本 + `index_<lang>.json` 机读索引）；
- 一份聚合 `report.json`（丢弃、移位、响度告警、清单哈希、对账结果）；
- 可中断续跑的内容寻址快照（`work/blocks/<lang>/`）。

所有时间计算基于 48kHz 整数采样点，禁止浮点累计漂移。

## 2. 模块结构

| 模块 | 职责 |
| --- | --- |
| `audiopack/config.py` | 配置项（并发、响度目标、增益上下限、交叉淡化、快照版本等）与配置哈希 |
| `audiopack/errors.py` | 业务码（`E_DUP_ID`/`E_BAD_TIME`/`E_MISSING_ASSET`/…）与不短路的问题收集器 |
| `audiopack/hashing.py` | 规范化 JSON 哈希、文件 SHA-256、原子写（临时文件 + rename + fsync） |
| `audiopack/manifest.py` | 清单解析、全量字段校验、时间单调性/范围、清单哈希、整数采样起点 |
| `audiopack/assets.py` | 线程池受限并发 WAV 加载、缓存、可注入失败、重试与终端业务码 |
| `audiopack/scheduler.py` | 同信道重叠消解：高优先级固定、低优先级整体后移、超最大延迟丢弃 |
| `audiopack/normalize.py` | 100ms 窗 RMS、整轨等功率响度对齐、增益钳制与超标区间告警 |
| `audiopack/renderer.py` | 静音填充、5ms 等功率交叉淡化、PCM16 写盘与帧头/帧数校验 |
| `audiopack/cue_index.py` | CUE/JSON 索引生成与音频—索引逐项对账 |
| `audiopack/snapshots.py` | 块快照、复用判定、受影响块识别、复用偏移重新校验 |
| `audiopack/pipeline.py` | 端到端编排与聚合报告 |
| `audiopack/samples.py` | 确定性本地样本与演示清单生成 |
| `audiopack/cli.py` | `generate` / `pack` 命令行入口 |

## 3. 关键设计决策

### 3.1 无漂移整数时间轴

- 每事件起点 `start_sample = time_ms * (sample_rate // 1000)`，48kHz 下即
  `time_ms * 48`，全程为 `int`。
- 片段自然时长直接取源 WAV 的采样帧数（`soundfile` 读出的 `frames`），不做秒级
  浮点换算后再回取整。
- 总时长以最后保留片段结束采样点向上取整到整毫秒（48 的倍数）。

### 3.2 同信道重叠消解

候选顺序按 `(-优先级, 自然起点, 事件ID)` 排序，保证确定性：

1. 高优先级先占位，锚定其绝对时间，永不被后到事件推动；
2. 低优先级片段若与任何已放置片段重叠，整体平移到阻挡片段 `end` 之后；
3. 平移只作用于当前低优先级事件，不会回写高优先级事件，因此不存在连锁改动；
4. 若平移量 `shift > max_delay_ms` 对应采样数，标记 `exceeds_max_delay` 丢弃并写入报告。

### 3.3 等功率响度归一化

- 以 100ms（4800 采样）非重叠窗计算 RMS（dBFS）；
- 低于静默门限（默认 -80 dBFS）的窗不参与能量平均；
- 整轨目标响度由“窗能量均值再转 dB”的等功率方式得到，请求增益
  `gain = target - measured`；
- 增益钳制在 `[-20, +12] dB`；触发钳制时记录告警，并给出归一化后仍偏离目标
  （容差默认 1.5dB）的连续窗合并区间（起止采样点与毫秒）。

### 3.4 交叉淡化与静音

- 调度保证片段区间不发生全量重叠；仅当相邻片段“首尾相接（无静音间隔）”时，
  在接合点应用默认 5ms 等功率曲线 `gain_in² + gain_out² = 1`，淡化长度同时受
  两侧片段时长一半约束；
- 存在静音间隔的片段直接复制，间隔保持零填充；
- 渲染结果在写盘前 clip 到 [-1,1] 并量化为 16bit little-endian PCM。

### 3.5 可中断续跑与块失效

每个语种一个块目录 `work/blocks/<lang>/`：

- `track.f32.npy`：归一化后的 float32 整轨（先落盘并 fsync）；
- `snapshot.json`：快照版本、配置哈希、清单哈希、源片段内容哈希、调度指纹、
  轨文件 SHA-256、总帧数、索引项、响度结果、丢弃事件。

快照始终在轨文件持久化之后才原子发布，崩溃只会留下没有快照引用的临时轨，
不会出现“半写快照”。恢复时按以下顺序判定：

1. 快照版本不一致 → `version_mismatch` 全量重算；
2. 配置哈希变化 → `config_changed`；
3. 清单哈希变化 → `manifest_changed`；
4. 该块引用的源片段内容哈希变化 → `asset_changed`（只重算受影响语种）；
5. 调度指纹变化 → `schedule_changed`；
6. 轨文件缺失或 SHA-256 不匹配 → 重算。

复用块不会沿用旧字节偏移：`revalidate_offsets` 用当前调度结果逐项重算
`sample_offset/end_sample/frames`，并校验结束点不超过新总时长；任何不一致返回
`E_OFFSET_REVALIDATION` 并降级为重算该块。

### 3.6 并发受限加载与失败隔离

- `AssetStore` 用 `ThreadPoolExecutor(max_workers=配置并发数)` 并行读取，按路径去重
  并缓存成功结果；
- `FailureInjector` 支持缺失（`E_MISSING_ASSET`，不可重试）、损坏
  （`E_CORRUPT_ASSET`，不可重试）、瞬态（`E_LOAD_FAILED`，按 `max_retries` 重试）；
- 物理损坏/错误声道/错误采样率/空音频统一归类为损坏码；
- 加载在任何块写入之前完成，失败不写入快照、不污染已完成块，报告一次性收集全部
  失败路径。

## 4. 业务码

| 码 | 含义 |
| --- | --- |
| `E_DUP_ID` | 事件 ID 重复 |
| `E_BAD_TIME` | 时间戳非单调或越界 |
| `E_BAD_FIELD` | 必填字段缺失/类型错误（含优先级、时间戳类型） |
| `E_BAD_LANG` | 语种列表非法或含未授权语种 |
| `E_MISSING_ASSET` | 配音源文件缺失（终端，不重试） |
| `E_CORRUPT_ASSET` | WAV 损坏、声道/采样率不符或为空 |
| `E_LOAD_FAILED` | 可重试的瞬态读取失败，重试耗尽后终端失败 |
| `E_SNAPSHOT` | 快照/对账/偏移复校异常 |

## 5. 运行方式

```bash
# 依赖（本机使用 venv + 国内镜像引导，见 docs/VERIFICATION.md）
python -m pip install -r requirements.txt

# 生成确定性样本与清单
python -m audiopack.cli generate --assets-dir demo/assets --manifest demo/manifest.json

# 打包（支持中断模拟）
python -m audiopack.cli pack \
  --manifest demo/manifest.json --assets-dir demo/assets \
  --work-dir demo/work --output-dir demo/out \
  --max-workers 3 --target-loudness-db -23

# 一键演示（中断→续跑→全复用）
./tools/run_demo.sh
```

## 6. 一致性与对账

- 写盘后用 `soundfile.info` 复核采样率/声道/子类型/帧数，帧数必须等于编码数组长度；
- `reconcile` 逐条比对索引项与调度结果的事件 ID、采样偏移、帧数、结束点、来源文件、
  CUE 时间，并确认所有片段落在最终 WAV 范围内；
- 任一对账失败都会进入 `reconciliation_errors` 并以非零 CLI 退出码（2）返回。
