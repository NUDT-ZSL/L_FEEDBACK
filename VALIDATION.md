# 验证记录

验证日期：2026-11-17

执行环境：

- Python 3.14.4
- NumPy 2.5.4
- Pytest 9.2.0
- FFmpeg / FFprobe 8.0.1
- 当前工作区提供的 Linux 执行环境

工具仅使用 Python 3.11+ 标准库、NumPy 及跨平台 FFmpeg 子进程接口；Windows 分支包含 `CREATE_NO_WINDOW`，原子发布使用 `os.replace`，不依赖 POSIX 专属能力。Windows 目标机可直接按 README 的命令运行。

## 自动化测试

```text
$ python -m pytest -q
....................                                                     [100%]
20 passed
```

重点覆盖：

- 字幕越出镜头、对白越出字幕、对白重叠时返回结构化 `stderr` JSON，并包含 `code`、`entry_id`；
- 不可适配时 CLI 退出码为 `4`，报告 `status=failed`，不发布最终 WAV；
- 同一时间轴与种子在独立目录生成的 5 个 WAV 逐字节一致；
- 相同对白在不同目标语种下会生成不同实际时长与不同波形；
- 静默裁剪后无需变速、以及 atempo 伸缩到完整窗口；
- 邻近对白闪避区间合并，间隔内保持衰减；
- 闪避曲线逐采样增益变化低于配置上限，过短渐变返回 `E_DUCK_RAMP_UNSAFE`；
- 输出为 48 kHz/16-bit/2 声道，时间轴帧误差为 0；
- 高增益混音触发整体衰减，量化峰值不高于 -1 dBFS，同时实测闪避深度仍为 -16 dB；
- 完整性哈希复用、输出损坏后重建；
- `KeyboardInterrupt` 中断样本生成/混流后的恢复；
- 子进程被强制 `kill()` 后，通过持久化 `writing` 状态和 `.part` 文件恢复并重建。

中断测试单独执行：

```text
$ python -m pytest -q tests/test_recovery.py
...                                                                      [100%]
3 passed
```

其中第三个测试启动真实 CLI 子进程，在写出部分 WAV 后执行 `process.kill()`，确认最终 WAV 不存在、`.part` 存在、状态为 `writing`；再次运行后产物完整且状态转为 `complete`。

另外执行了真实 `SIGINT` 手工复验：子进程在写 WAV 时收到信号，退出码为 `130`，`stderr` 为 `E_INTERRUPTED` JSON；最终 WAV 与报告均不存在，`.part` 存在，状态为 `interrupted`。再次运行退出码为 `0`，状态转为 `complete`，`.part` 被清除。

## 固定种子端到端复验

```text
$ python -m cutscene_dub generate-samples \
    -t examples/timeline.json -o <tmp>/a --seed 189
$ python -m cutscene_dub generate-samples \
    -t examples/timeline.json -o <tmp>/b --seed 189
```

两目录 WAV 的 `sha256sum` 完全一致：

```text
80ba928a303fb4d185bdd065e3308ff28f3b5d2a98c4e93a7d868a38d033fd5d  ./bgm.wav
4e0384ae9b37868a2a76bc7d2ccd6769fbf1184ea4363ec5a067e364c8cc9ab7  ./en-US/dlg_001.wav
1eb1615a7f70cfe8c6c7d7f6c54c2ed46f480e87f2267ba103c5f0929612d266  ./en-US/dlg_002.wav
cec68b9fc00566436556b57c4173fca8474b1291fb3e8b66c2be0b98dfbecd40  ./en-US/dlg_003.wav
21eb9292c198bfcf06704ec7817ef0fd228bfcb3d4d7df4cf046e8087af1c5fe  ./en-US/dlg_004.wav
```

混流：

```text
$ python -m cutscene_dub render \
    -t examples/timeline.json -a <tmp>/a \
    -o <tmp>/build/cutscene.wav \
    -r <tmp>/build/alignment_report.json
```

结果摘要：

- 输出 SHA-256：`69bbdef41f0c68bcb77a911895a019e9e1e985cc501bf0f75cb884bdae37e7fb`
- 帧数：`480000`，即 10.000000 秒
- 时间轴时长误差：`0` 帧
- 峰值：`-8.441265499166946 dBFS`
- 闪避配置深度：`-16.0 dB`
- 对白活跃区实测音乐衰减：`-16.0 dB`
- 最大逐采样增益变化：`0.005006927656124785 dB`（上限 `0.5 dB`）
- 对白相对闪避后 BGM 的 RMS 差：`22.268219230451304 dB`
- `dlg_001` 使用 atempo：`1.0689666666666666`
- 4 个对白的 `final.within_window` 均为 `true`

四个对白最终落点：

| ID | 起点 s | 终点 s | 窗口 s | 裁剪量 s | 伸缩比例 |
|---|---:|---:|---:|---:|---:|
| dlg_001 | 0.500000 | 3.000000 | 0.5-3.0 | 0.577583 | 1.068967 |
| dlg_002 | 3.400000 | 4.444521 | 3.4-4.7 | 0.655479 | 1.0 |
| dlg_003 | 5.200000 | 7.555333 | 5.2-7.8 | 0.544667 | 1.0 |
| dlg_004 | 8.300000 | 9.493167 | 8.3-9.6 | 0.356833 | 1.0 |

## FFprobe 输出

```json
{
  "streams": [
    {
      "codec_name": "pcm_s16le",
      "sample_rate": "48000",
      "channels": 2,
      "duration": "10.000000"
    }
  ],
  "format": {
    "duration": "10.000000"
  }
}
```

输出 WAV 为交错立体声：帧数 `480000`，左右声道各 `480000` 个 16-bit 样本，文件大小 `1920044` 字节。

## 峰值限制复验

同一素材使用 `--dialogue-gain-db 20` 混流：

```text
peak_before_limit_dbfs: 11.325321666710696
global_attenuation_db: -12.325321666710698
peak_after_limit_dbfs: -0.9999999999999997
quantized_peak_dbfs: -1.0001519038193567
duck_depth_db: -16.0
measured_ducking_db: -16.0
```

整体衰减与闪避分别记录，闪避深度未因峰值控制而改变。

