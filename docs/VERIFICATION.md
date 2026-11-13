# 实际验证记录

验证日期：2026-11-14

执行环境：

- Ubuntu 26.04.1 LTS, x86_64
- Python 3.14.4
- FFmpeg 6.0-static / ffprobe 6.0-static
- 本地确定性 WAV 样本，固定种子 `0xA0D10F1C`
- 未使用网络、容器或外部素材

## 内置自检

命令：

```bash
python3 audio_pipeline.py self-test \
  --dir .audio-pipeline-selftest \
  --ffmpeg /home/liang/.feedback/runtime/ffmpeg6/ffmpeg \
  --ffprobe /home/liang/.feedback/runtime/ffmpeg6/ffprobe \
  --jobs 4
```

结果：`SELF-TEST PASS`，`80/80` 项检查通过。

首次运行关键数据：

```text
first_pipeline_exit_code=1
counts={'total': 8, 'completed': 4, 'skipped': 0, 'failed': 4, 'interrupted': 0}

01_stereo48_normal      LUFS=-17.15  requested=1.15    applied=1.15   clamped=False  completed  0
02_fade_overlap_bed     LUFS=-19.85  requested=4.85    applied=4.85   clamped=False  completed  0
03_clamp_boost          LUFS=-19.85  requested=14.85   applied=6.00   clamped=True   completed  0
04_clamp_attenuation    LUFS=-17.15  requested=-32.85  applied=-6.00  clamped=True   completed  0
05_missing_source       -            -               -              -              failed     1  SOURCE_MISSING
06_invalid_manifest     -            -               -              -              failed     1  MANIFEST_INVALID
07_unsupported_rate     -            -               -              -              failed     1  UNSUPPORTED_AUDIO_FORMAT
08_unsupported_channels -            -               -              -              failed     1  UNSUPPORTED_AUDIO_FORMAT
```

钳制结果：

- `03_clamp_boost`: 请求 `+14.85 dB`，`max_boost_db=6`，实际 `+6.00 dB`，
  `clamp=max_boost`。
- `04_clamp_attenuation`: 请求 `-32.85 dB`，`max_attenuation_db=6`，实际
  `-6.00 dB`，`clamp=max_attenuation`。

音乐淡出实测（最终输出，660 Hz 测试音）：

```text
before_amplitude=8952.352
mid_fade_amplitude=4177.736    # 淡出时间点约 53.3% 处
after_fade_amplitude=0.032
mid_ratio=0.4667
after_ratio=0.000004
```

淡出中段比例与理论值 `1 - (1.80-1.00)/1.50 = 0.4667` 一致。

断点/重跑检查：

```text
second_run_counts={'total': 8, 'completed': 0, 'skipped': 4, 'failed': 4, 'interrupted': 0}
changed_job=completed
source_changed_job=completed
deleted_output=completed
stale_run_counts={'total': 8, 'completed': 0, 'skipped': 4, 'failed': 4, 'interrupted': 0}
```

清单始终为 8 个唯一作业标识；manifest 变化、源音频内容变化和删除产物都只触发
对应作业重做。

## 强制终止后恢复

使用一个只增加 `sleep` 后 `exec` 真实 FFmpeg 的本地包装器，在第二个作业处理中
执行 `SIGKILL`：

```text
completed_before_kill=1
journal_after_kill=[('01_stereo48_normal', 'completed')]
resume_exit=0
resume_counts={'total': 4, 'completed': 3, 'skipped': 1, 'failed': 0, 'interrupted': 0}
journal_unique=True entries=4
journal_statuses={
  '01_stereo48_normal': 'skipped',
  '02_fade_overlap_bed': 'completed',
  '03_clamp_boost': 'completed',
  '04_clamp_attenuation': 'completed'
}
```

恢复只依赖持久清单与临时目录命名，不存在退出报告依赖，也没有重复作业条目。

## Ctrl+C 可控收敛

在第二个作业处理中向前台管线发送 `SIGINT`：

```text
sigint_exit=130
counts={'total': 4, 'completed': 2, 'skipped': 0, 'failed': 0, 'interrupted': 2}
stderr_line_count=1

01_stereo48_normal      completed    0
02_fade_overlap_bed     completed    0
03_clamp_boost          interrupted  130
04_clamp_attenuation    interrupted  130
```

当前作业完成落盘后才退出；未启动作业未写入清单。随后重跑：

```text
resume_exit=0
resume_counts={'total': 4, 'completed': 2, 'skipped': 2, 'failed': 0, 'interrupted': 0}
journal unique=True entries=4
```

## stderr/JSON 一致性

CLI 运行后 stderr 只有一行，且计数与结果文件一致：

```text
exit=1 stderr_lines=1
audio-pipeline exit=1 total=8 completed=0 skipped=4 failed=4 interrupted=0 result=...
result counts={'total': 8, 'completed': 0, 'skipped': 4, 'failed': 4, 'interrupted': 0}
```

缺失 FFmpeg 的致命运行：

```text
exit=2 stderr_lines=1
audio-pipeline exit=2 total=8 completed=0 skipped=0 failed=8 interrupted=0 result=...
fatal_error.code=FFMPEG_UNAVAILABLE
counts={'total': 8, 'completed': 0, 'skipped': 0, 'failed': 8, 'interrupted': 0}
```

## 自动化回归测试

命令：

```bash
FFMPEG=/home/liang/.feedback/runtime/ffmpeg6/ffmpeg \
FFPROBE=/home/liang/.feedback/runtime/ffmpeg6/ffprobe \
python3 -m unittest -v tests.test_pipeline
```

结果：

```text
Ran 5 tests in 2.618s
OK
```

测试包括 WAV RIFF/data 长度、截断 WAV 拒绝、manifest 等价字段/默认钳制上限、
路径逃逸拒绝，以及完整自检和 CLI 单行摘要一致性。
