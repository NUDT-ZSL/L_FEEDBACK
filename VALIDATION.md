# 实际验证记录

验证日期：2026-11-16

环境：

- Python 3.14.4
- NumPy 2.5.4
- SoundFile 0.14.0
- 24 个逻辑 CPU
- Linux 验证容器；代码只使用 Windows/Linux 均支持的 Python、NumPy、
  SoundFile 和 `concurrent.futures` API。

## 自动化测试

执行：

```text
python -m unittest discover -s tests -v
```

结果：

```text
test_gain_envelope_has_linear_50ms_crossfade ... ok
test_k_filter_matches_direct_iir ... ok
test_alignment_loudness_gain_warnings_and_output ... ok
test_batch_isolation_and_jobs_cap ... ok
test_strong_kill_recovery_accumulates_retry_count ... ok
test_synth_is_sample_deterministic_and_has_required_faults ... ok

Ran 6 tests in 5.805s
OK
```

## 端到端样本

使用 6 路、静音第 2 路、削波第 3 路的 seed 101 样本：

- 所有可听轨（含削波轨）偏移估计误差：`0` 个样本。
- 静音轨：`status=skipped`，`business_code=SILENT_TRACK`，不参与混音。
- 削波轨：`clipped_input=true`，仍给出正确偏移。
- 4 个场景均输出 BS.1770 响度；需求增益超出 ±12 dB 的场景被钳制。
- 全局警告包含：
  `SILENT_TRACK`, `CLIPPED_INPUT`, `GAIN_CLAMPED`, `LIMITER_ACTIVE`。
- 归一化前峰值大于 0 dBFS，软限幅路径实际执行。
- 输出为 48 kHz、双声道、PCM_16，报告最终峰值为 -1.0 dBFS。
- `report.json` 与 `timeline.json` 均为 UTF-8 无 BOM。

同 seed 重复生成时，`timeline.json` 和全部 `cam_XX.wav` 均逐字节一致。

另按 README 快速开始命令处理 seeds `101,202,303`（每组 6 路、2 号静音、
3 号削波、`--jobs 2`）：

```text
success            retries=0 seed_101
success            retries=0 seed_202
success            retries=0 seed_303
summary success=3 failed=0 jobs=2/24
```

三组的可听轨偏移误差均为 `[0, 0, 0, 0, 0]`；均触发四个预期警告；
场景增益均为 `-12.0 dB`（钳制边界）；归一化前峰值分别约为
`+1.88 / +1.89 / +1.85 dBFS`；最终输出均为 48 kHz、双声道、PCM_16，
峰值 `-1.0 dBFS`。

## 并发与强杀恢复

- 一批包含 1 个正常组和 1 个损坏时间轴组，使用并发执行：
  正常组成功输出；损坏组失败并保留 `.progress`；批处理继续完成并汇总，
  证明单组失败不会中断其他组。
- `--jobs 999` 被限制到逻辑 CPU 核数。
- 对单组批处理发送 `SIGKILL`：`.progress` 保留。
- 原命令重新运行后：该组重试成功，报告 `retry_count >= 1`，成功后
  `.progress` 删除；再次运行则该组标记为 `skipped_completed`。
