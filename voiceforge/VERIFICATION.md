# 实测验证记录

执行环境：WSL2（Ubuntu，Linux 6.18 内核）、Python 3.14 运行纯标准库代码；
代码仅使用 Python 3.11 已具备的标准库 API，目标机为 Windows 10/11 +
Python 3.11。中断/锁的 Windows 原生分支（`SIGBREAK`、`OpenProcess`）已在
`interrupt.py`、`lock.py` 中按平台条件实现，POSIX 侧用等价机制实测。

## 1. 单元与集成测试

命令：`python -m unittest discover -s tests -v`

结果：`Ran 15 tests ... OK`，全部通过。

| 用例 | 覆盖需求 |
| --- | --- |
| `test_fixed_seed_is_byte_reproducible` | 同种子 WAV/JSON 逐字节复现 |
| `test_player_and_segment_counts_and_rates` | 3–5 玩家、每人 6–10 段、三种采样率、16bit |
| `test_drift_and_missing_first_packets_are_present` | >20ms 漂移、首包缺失、缺失玩家 |
| `test_drift_threshold_and_linear_resample` | 20ms 门限判定与线性插值重采样 |
| `test_gate_gain_and_saturation` | 硬门控、增益、逐采样 16bit 饱和与削波计数 |
| `test_missing_source_does_not_crash` | 缺失来源不崩溃 |
| `test_overlapping_intervals_become_disjoint_events` | 同时发言稳定合并、事件无重叠 |
| `test_missing_player_is_flagged` | 时间线 `missing_source` 标记 |
| `test_interrupt_then_resume_is_byte_identical` | 安全点中断（退出码 0）后恢复逐字节一致 |
| `test_real_sigint_flushes_checkpoint_and_resumes` | 真实 SIGINT 信号处理路径 |
| `test_hard_kill_recovers_from_durable_checkpoint` | SIGKILL 后仅靠落盘检查点恢复 |
| `test_changed_shard_invalidates_player_and_rebuilds_identical_output` | SHA-256 变化失效与下游重建 |
| `test_repeat_run_summary_is_identical` | 重复运行摘要完全一致 |
| `test_live_owner_is_rejected_and_released_on_exit` | 存活锁所有者拒绝（退出码 3） |
| `test_stale_lock_of_dead_process_is_taken_over` | 残留锁安全接管 |

## 2. 确定性样本（固定种子 20261114）

- 玩家 5 个，分片 38 个，事件区间 39 个，缺失来源玩家 `P5`。
- 采样率分布：16000Hz × 12、44100Hz × 18、48000Hz × 8。
- 触发重采样（漂移 >20ms）分片 11 个，首包缺失分片 11 个，最大漂移 81ms。

## 3. 中断恢复逐字节比对

`run_demo.sh` 在第三个区间后制造中断再恢复，与一次性运行的 SHA-256：

```text
6beeb4af948ea6b3bc740d81091526d37f937df031884ae6cb3cfb59e9224edb  full/voice_track.wav
6beeb4af948ea6b3bc740d81091526d37f937df031884ae6cb3cfb59e9224edb  resume/voice_track.wav
f498eb91922fc1cff3f923d5662d30e0ed900ca17cc60be95563ecfcdc0c9e99  full/timeline.json
f498eb91922fc1cff3f923d5662d30e0ed900ca17cc60be95563ecfcdc0c9e99  resume/timeline.json
064bca1398bdd8b663ab3957034f4d5015acaec25d1cfe1ece08cbc35d079cec  full/summary.json
064bca1398bdd8b663ab3957034f4d5015acaec25d1cfe1ece08cbc35d079cec  resume/summary.json
35e1d0e8877fb1e8da166ad2b75f392f91d2e7e691105b46e7061ed77e246fa0  full/mix_float.f32
35e1d0e8877fb1e8da166ad2b75f392f91d2e7e691105b46e7061ed77e246fa0  resume/mix_float.f32
```

重复运行摘要同样一致：

```text
064bca1398bdd8b663ab3957034f4d5015acaec25d1cfe1ece08cbc35d079cec  full/summary.json
064bca1398bdd8b663ab3957034f4d5015acaec25d1cfe1ece08cbc35d079cec  full.repeat/summary.json
```

## 4. 混音与统计结果

```json
{
  "output_sample_rate": 48000,
  "total_samples": 1241473,
  "interval_count": 39,
  "event_count": 69,
  "clipped_samples": 230326,
  "missing_source_players": ["P5"],
  "missing_source_player_count": 1,
  "missing_source_interval_count": 1
}
```

- 时间线 69 个事件两两不重叠，其中 63 个为多玩家同时事件。
- `loudness_report.json` 对每个区间给出 `peak`、`rms`、
  `new_clipped_samples`；削波采样位图保证同一采样只计入一次。
- 锁冲突实测：存活所有者时退出码 `3`，stderr 输出
  “输出目录已被进程 PID=... 锁定...”；进程退出后再次运行自动接管并成功。

## 5. Windows 运行说明

在 Windows 10/11 安装 Python 3.11 后，于项目目录执行：

```bat
demo_windows.bat
:: 或手动
python -m voiceforge generate sample\in
python -m voiceforge synthesize sample\in sample\out
```

Ctrl+C 由 `SIGINT`/`SIGBREAK` 处理器在区间安全点落盘检查点后以退出码 0 退出；
若被任务管理器结束进程，下次 `synthesize` 仅依据 `.checkpoint/state.json`
恢复，输出与未中断运行一致。

