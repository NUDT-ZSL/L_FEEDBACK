# avswitch — 多码率视频动态合流器（本地离线可复现）

面向直播/实时互动场景的离线仿真合流器：读取 JSON 清单声明的多路 H.264
视频流（不同分辨率/码率/GOP 长度）与一条独立 AAC 音频流，按清单中的
带宽模拟曲线自适应切换视频源，仅在关键帧处切换，输出单一 MPEG-TS 文件，
并生成切换记录与音画同步质量报告（JSON）。

## 环境要求

- Windows 10/11 64 位（开发验证于 Linux，代码为纯跨平台 Python）
- Python 3.9+，NumPy
- FFmpeg 4.4+（`ffmpeg` / `ffprobe` 在 PATH 中）
- 无需网络、Docker 或外部账号

## 快速开始

```bash
python create_samples.py          # 确定性生成样本流（约 3 秒）
python -m avswitch manifest.json out/switched.ts
python run_validation.py          # 一键完整验证（样本 + 正常/回退两组运行）
python -m unittest discover -s tests -v   # 单元/验收测试（10 项）
```

运行结束后生成 `<输出>.report.json`，包含每次切换的时间点/源流/目标流/
原因、回退记录、时间戳调整日志、逐切换点同步误差与整体同步统计。

## 样本（create_samples.py）

| 流 | 分辨率 | 码率 | GOP |
|---|---|---|---|
| video_low_800k.mp4 | 640x360 | 800 kbps | 2s (48 帧 @24fps) |
| video_mid_2500k.mp4 | 1280x720 | 2500 kbps | 2s |
| video_high_5000k.mp4 | 1920x1080 | 5000 kbps | 4s |
| audio_main.aac | 48 kHz 立体声 | 128 kbps | — |

全部使用 `testsrc2`/`sine` 确定性生成，无 B 帧（`-bf 0`，低延迟场景），
固定关键帧间隔（`-sc_threshold 0`）。另生成 `video_mid_corrupt.mp4`
（faststart 后截断至 45%）用于回退测试。

## 设计要点

- **GOP 解析与切换点选择**（`avswitch/gop.py`, `avswitch/muxer.py`）：
  ffprobe 提取全部包的 PTS/DTS/关键帧标记；切换时间向下吸附到目标流
  最近关键帧，吸附偏差记入 `timestamp_adjustments` 日志。
- **ABR 决策**（`avswitch/planner.py`）：按 2s 块模拟播放，有效带宽 =
  带宽 × (1 − buffer_margin) − 音频码率，并要求缓冲区能吸收下载耗时 +
  切换开销；降档立即生效，升档需满足 min_dwell 驻留时间以防抖动；
  缓冲区欠载时强制降到最低档。
- **音画同步封装**：分段无损抽取（`-ss/-t -c copy`）→ concat 拼接 →
  与 AAC 音频以 `-c copy` 复用为 MPEG-TS（`-muxdelay 0`），时间基统一
  重映射到 90 kHz；音频来自单一连续文件，天然无中断。
- **帧连续性**：样本无 B 帧故 DTS==PTS；复用后校验视频 DTS 严格单调、
  帧间隔等于 1/24s（无重复帧/时间空洞）；ffmpeg 若修正非单调 DTS 会
  被计数并写入调整日志。
- **失败回退**：每段先做解码级检查（`ffmpeg -v error ... -f null -`），
  失败则回退到最近成功使用的流（起点吸附到其关键帧），记录回退时间、
  目标流、回退流与原因。
- **TS 结构校验**（`avswitch/tsvalidate.py`）：解析 188 字节包、PAT/PMT，
  确认视频（H.264, PID 256）与音频（AAC, PID 257）各占一个 PID，音频
  PES 长度非零且与实际负载一致，视频 PES 允许长度 0（符合
  ISO/IEC 13818-1）。

## 实测验证结果（本机，FFmpeg 8.0.1）

正常清单（5 次切换：4s low→high、6s high→low、12s low→mid、
18s mid→low、24s low→high）：

- 输出 720 帧，DTS 单调，无丢帧/重复帧/时间空洞
- 音画 PTS 最大偏差 10.667 ms，均值 5.349 ms，漂移 6.333 ms（均 < ±20 ms）
- 逐切换点同步误差：0 ~ 10.667 ms
- PES 长度违规 0；`ffmpeg -v error -i out.ts -f null -` 无任何错误

回退清单（mid 流损坏）：t=12s 目标 mid 解码失败，自动回退 low，
日志记录回退时间点/原因；输出仍 720 帧、同步与连续性全部达标。

## 目录结构

```
create_samples.py      确定性样本生成
manifest.json          正常清单（含带宽曲线与切换策略）
manifest_fallback.json 回退测试清单（mid 指向损坏文件）
avswitch/              合流器包（probe/gop/planner/muxer/verify/tsvalidate/cli）
tests/test_switcher.py 验收测试（10 项）
run_validation.py      一键验证脚本
```
