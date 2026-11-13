# 验证记录

## 1. 环境准备（无 root / 无预装 pip）

任务机仅有 Python 3.14 且无 pip、无 sudo，无法直接安装任务推荐的 Python 3.11 +
numpy 1.26。采用等价本地方案（语言/框架允许替换等价本地库）：

- `python3 -m venv --without-pip .venv` 建虚拟环境；
- 从可达的阿里云镜像下载 `pip` wheel 引导安装（PyPI 官方源在本机代理下 TLS 异常）；
- 经镜像安装二进制 wheel，避免在 3.14 上从源码编译 numpy。

实测版本：

```text
Python 3.14.4
numpy 2.5.4
soundfile 0.14.0
libsndfile 1.2.2
```

WAV 读写冒烟：48kHz/mono/PCM_16 写后读回，采样率、声道、子类型、帧数全部一致。

> 说明：任务推荐的 numpy 1.26 没有 cp314 wheel；所选 numpy 2.5 / soundfile 0.14
> 为同一 API 族的等价本地库，代码仅使用稳定的 ndarray 与 soundfile 读写接口。

## 2. 自动化测试

命令：

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m pyflakes audiopack tests
```

实测结果（41 个用例全部通过，静态检查零告警）：

| 测试模块 | 用例数 | 覆盖点 |
| --- | --- | --- |
| `test_manifest.py` | 6 | 多错误一次性收集、`E_DUP_ID`/`E_BAD_TIME`/`E_BAD_FIELD`/`E_BAD_LANG`、整数采样无漂移 |
| `test_scheduler.py` | 6 | 高优先级锚定、低优先级整体后移、不回写高优先级、超延迟丢弃、确定性、无重叠 |
| `test_normalize.py` | 6 | 100ms 窗、目标对齐、+12/-20dB 钳制、静默处理、超标区间帧范围 |
| `test_renderer.py` | 6 | 整毫秒取整、静音零填充、等功率交叉淡化、确定性、PCM16 帧头/帧数 |
| `test_assets.py` | 7 | 并发加载与缓存、瞬态重试成功/耗尽、缺失/损坏终端码、物理坏文件、批量失败收集 |
| `test_snapshots.py` | 5 | 快照往返复用、块级资产失效、版本强制重算、偏移复校、轨篡改哈希失效 |
| `test_pipeline.py` | 5 | 端到端对账、CUE/WAV、中断续跑复用、清单变更重算、失败零快照、坏清单业务码 |

```text
Ran 41 tests in 0.96s
OK
PYFLAKES CLEAN
```

## 3. 端到端演示（tools/run_demo.sh）

确定性生成 7 个语音键 × 3 语种 = 21 个 WAV，与含 7 个事件的清单，执行
“中断（停在 en）→ 续跑 → 第三次全复用”。

第三次运行报告关键字段（实测）：

```text
status: ok
reused: ['en', 'ja', 'zh']   recomputed: []
reconciliation_errors: []
outputs:
  en 393600 frames 8200.0ms retained=5 dropped=1
  ja 393600 frames 8200.0ms retained=5 dropped=0
  zh 393600 frames 8200.0ms retained=5 dropped=1
shifts: EVT-001 在 en/ja/zh 均整体后移 850.0ms（40800 采样）
drops : EVT-003 在 en/zh 因 required 570ms > max_delay 80ms 丢弃
```

中断/续跑分阶段实测：

```text
第 1 次（--stop-after-lang en）: status=interrupted, 仅 en 块落盘
第 2 次（续跑）                : reused=['en'], recomputed=['ja','zh']
第 3 次（相同输入）            : 全部 reused, recomputed=[]
```

## 4. 物理 WAV 与 CUE/索引对账

```text
zh/en/ja: 48000 Hz, 1 channel, PCM_16, 393600 frames, 8200.0ms, frames % 48 == 0
index_<lang>.json: total_frames 与 WAV 帧数相等，每条 end_sample <= total_frames
reconcile: 0 错误
```

CUE 片段（`demo_run/out/track_zh.cue`），含采样偏移对应的 75fps CUE 时间、
来源文件、优先级与移位采样数：

```text
TRACK 02 ... TITLE "EVT-001" REM SOURCE kill_zh.wav
           REM SHIFT_SAMPLES 40800  INDEX 01 00:01:64
```

## 5. 响度归一化与告警

演示轨（目标 -23 dBFS）实测 `zh`：measured `-13.529 dB`，applied gain `-9.471 dB`，
未触发钳制；归一化后仍偏离目标容差的窗被合并为区间报告，例如：

```text
start_ms=1000.0 end_ms=1100.0 measured_db=-26.309
start_ms=7500.0 end_ms=8100.0 measured_db=-21.451
```

增益上下限钳制由专门用例验证（`test_normalize.py`）：

```text
低电平恒定信号(-50.46dB): requested +27.46dB -> applied +12.0dB, clamped=True, 产生告警
高电平恒定信号(-0.09dB) : requested -22.91dB -> applied -20.0dB, clamped=True
```

## 6. 失败注入与失败隔离

通过 `FailureInjector` 注入，实测：

```text
missing            -> status=failed, codes=['E_MISSING_ASSET']
corrupt            -> status=failed, codes=['E_CORRUPT_ASSET']
transient(2 次)    -> status=ok, 第 3 次尝试成功（attempts=3）
transient(耗尽)    -> status=failed, codes=['E_LOAD_FAILED'], attempts=3 (max_retries=2)
```

失败发生在任何块写入之前，对应工作目录 `work/blocks` 不存在或为空，证明失败不产生
半写快照、不污染已完成块。

## 7. 快照失效矩阵

| 变更 | 实测 reason | 影响范围 |
| --- | --- | --- |
| 清单内容变化 | `manifest_changed` | 引用该清单的语种块重算 |
| 单个源 WAV 内容变化 | `asset_changed` | 仅引用它的语种块重算，其余复用 |
| `snapshot_version` +1 | `version_mismatch` | 全部块强制重算 |
| 轨文件被篡改 | `track_hash_mismatch` | 该块重算 |
| 索引偏移被篡改 | `E_OFFSET_REVALIDATION` | 降级为该块重算 |
