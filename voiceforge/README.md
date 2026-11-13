# VoiceForge

离线多玩家语音事件对齐、门控混音与可恢复合成的 Windows 命令行程序。仅依赖
Python 3.11 标准库，无需网络、第三方包或容器；所有样本由固定种子的内置
生成器确定性构造。

## 目录结构

```text
voiceforge/
├─ voiceforge/
│  ├─ prng.py          自带的确定性 xorshift64* 随机流（跨版本字节可复现）
│  ├─ audio.py         16-bit PCM WAV 读写、归一化、逐采样饱和
│  ├─ hashing.py       SHA-256 文件/文本摘要
│  ├─ resampling.py    20ms 漂移判定与线性重采样/时间轴映射
│  ├─ generator.py     确定性样本与 events.json 生成器
│  ├─ engine.py        门控混音、削波计数、响度、断点恢复主引擎
│  ├─ checkpoint.py    内容寻址检查点与失效/重建
│  ├─ timeline.py      扫描线事件时间线（同时发言合并、无重叠）
│  ├─ outputs.py       WAV/浮点中间结果/时间线/响度/摘要产物
│  ├─ lock.py          原子锁文件与所有者存活探测
│  ├─ interrupt.py     Ctrl+C/SIGTERM/SIGBREAK 安全点中断
│  └─ cli.py           generate / synthesize 子命令
├─ tests/              15 个 unittest 用例（含子进程中断与强杀）
├─ run_demo.sh         跨平台演示（Linux/macOS/WSL 实测）
└─ demo_windows.bat    Windows 一键演示
```

## 快速开始（Windows）

```bat
python -m voiceforge generate sample\in
python -m voiceforge synthesize sample\in sample\out
```

`synthesize` 可安全重复执行：完成后再次运行会复用检查点并产出相同产物；
中断后重新运行只补做未完成区间。

## 关键设计与需求对应

1. **确定性生成器**：`generator.py` 使用固定种子（默认 `20261114`）的
   xorshift64* 流，为 3–5 个玩家各生成 6–10 个 16-bit 单声道分片，采样率取自
   16000/44100/48000，并显式制造 20ms 内漂移、>20ms 漂移、首包缺失、跨玩家
   同时发言和高增益削波；同种子输出 WAV 与 JSON 逐字节一致。
2. **时间轴对齐**：每个分片记录 `record_offset_ms` 与采样率。
   `resampling.render_segment` 始终把分片线性插值到统一 48kHz 输出网格；当
   `|实际时长-声明时长| > 20ms` 时按“起录偏移 + 声明时长”重采样校正，而非截断。
3. **门控混音**：按区间门限做硬性门控（低于门限置零），再施加每区间线性增益
   并累加到 float64 主轨；最后用“四舍五入 + 钳位”逐采样饱和到有符号 16 位。
   float32 浮点中间结果写入 `mix_float.f32`，`loudness_report.json` 给出每区间
   峰值、均方根与新增削波采样数（用位图保证一个采样只被计数一次）。
4. **事件时间线**：区间先栅格化为采样区间，再用扫描线在每个激活边界切分，
   保证事件互不重叠；同一采样上同时激活的玩家按 `(player_id, interval_id)`
   稳定排序合并为一条 `simultaneous` 事件并保留参与者列表。
5. **可恢复增量合成**：检查点位于 `.checkpoint/`，包含每区间贡献 `.bin`、
   侧车 `.json`（含分片 SHA-256、偏移、长度）、float64 累加器快照与原子替换的
   `state.json`。分片摘要变化时，从该玩家最早受影响区间起丢弃下游累计状态，
   利用其余区间贡献确定性重建累加器，禁止沿用旧摘要的偏移/长度。
6. **中断处理**：信号处理器仅置位，引擎在区间边界安全点落盘检查点后以退出码
   `0` 结束；被 `SIGKILL`/任务管理器强杀时无清理代码，恢复只信任最后已提交的
   `state.json`。浮点求和顺序固定（区间按 `(start_ms, interval_id)` 排序），
   因此恢复与一次成型逐字节相同。
7. **并发保护**：以 `O_CREAT|O_EXCL` 原子创建 `.voiceforge.lock`，内含 PID/主机/
   平台所有者信息。Windows 用 `OpenProcess`/`GetExitCodeProcess`、POSIX 用
   `kill(pid,0)` 探测存活；所有者存活返回退出码 `3` 并打印可读错误，所有者
   已死则删除残留锁并接管。
8. **一致性校验**：结束产出 `voice_track.wav`、`timeline.json`、
   `loudness_report.json`、`mix_float.f32` 与 `summary.json`（各输入分片摘要、
   各输出摘要、采样总数、缺失来源玩家/区间计数）。清单引用但无分片的玩家在
   时间线中标记 `missing_source` 并计入统计，不导致崩溃。

## 确定性与字节一致性说明

- JSON 统一 `indent=2`、`\n` 换行并经临时文件 + `os.replace` 原子落盘；
  不写入时间戳或主机相关字段。
- WAV 由标准库 `wave` 写出固定头，PCM 来自确定性整数饱和。
- 浮点路径使用固定顺序的 IEEE-754 运算，同平台/同 Python 小版本结果稳定；
  测试通过实际哈希比对验证一次运行、中断恢复与重复运行完全一致。

## 测试与验证

```bash
python -m unittest discover -s tests -v
./run_demo.sh          # 生成、中断恢复、哈希比对
```

实测结果见 [VERIFICATION.md](VERIFICATION.md)。

