# Dialogue Master

离线对白响度标准化与采样级时间轴合成工具。工具读取一组 PCM/浮点 WAV
对白片段和动画时间轴标记，按 ITU-R BS.1770-4 计算积分响度，执行增益、
峰值约束、余弦平方淡入淡出和贪心放置，最终输出游戏引擎可直接读取的
多声道 WAV 主轨及确定性 JSON 处理日志。

## 环境

- Python 3.9+
- NumPy
- SoundFile
- FFmpeg 4.4+（仅用于独立验收/探测；处理流程本身不调用 FFmpeg）

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 使用方法

支持选项或位置参数两种形式：

```bash
python3 -m dialogue_master --config config.json --output-dir output
python3 -m dialogue_master config.json output
```

成功时退出码为 `0`。输出目录包含：

- `master.wav`：与配置时间轴等长的 WAVE_FORMAT_EXTENSIBLE PCM WAV，
  含标准声道掩码。
- `processing_log.json`：UTF-8、键排序、固定两空格缩进、无时间戳/随机
  来源的确定性日志。

若所有片段都被跳过，退出码为 `2`，只写日志，不保留 `master.wav`。

## 配置格式

相对 `input_path` 以配置文件所在目录为基准。

```json
{
  "timeline_ms": 5000,
  "sample_rate": 48000,
  "bit_depth": 24,
  "channels": 2,
  "channel_layout": "stereo",
  "clips": [
    {
      "id": "line_01",
      "input_path": "audio/line_01.wav",
      "start_ms": 300,
      "target_lufs": -23.0,
      "fade_in_ms": 30,
      "fade_out_ms": 50
    }
  ]
}
```

基准字段：

- `timeline_ms`：总时间轴长度，非负有限数字，必须产生至少一个采样。
- `sample_rate`：整数，1000 到 768000。
- `bit_depth`：`8`、`16`、`24` 或 `32`。输出为对应 PCM 位深；输入
  位深相同即可，32-bit IEEE float WAV 也可接受。
- `channels`：1 到 16。
- `channel_layout`：可选；如果省略，则使用 1/2/3/4/5/6/8 声道的标准
  默认布局。支持 `mono`、`stereo`、`2.1`、`3.0`、`quad`/`4.0`、
  `5.0`、`5.1`、`7.1`。
- `clips`：数组。片段按数组原始顺序处理。

片段字段均必填（`id` 可选）：

- `start_ms`：目标起点，毫秒。
- `target_lufs`：目标积分响度。
- `fade_in_ms` / `fade_out_ms`：淡变时长，毫秒，可为 `0`。

字段缺失、类型错误、数值非法、布局与声道数不符、重复 ID 等均输出
`E_CFG` 到 stderr，以退出码 `1` 结束；配置校验完成前不会创建输出目录
或任何半成品。

## 处理规则

1. 输入必须是 WAV/WAVEX。采样率、位深或声道数与基准不一致时，片段
   记录 `W_FORMAT` 并跳过。损坏/不可解码文件记录 `E_AUDIO` 并跳过。
2. 每次导出的和每个中间结果都保留 `float64` 精度。BS.1770-4 实现包含
   K 加权、400 ms 门限块、100 ms 步进、绝对门限 `-70 LUFS` 和相对
   门限 `-10 LU`；LFE 不参与响度能量。
3. 当 `abs(target_lufs - integrated_lufs) > 10 LU` 时施以
   `target_lufs - integrated_lufs` 的线性增益，初始增益限制在
   `[-20, +20] dB`。若增益后峰值超过 `0 dBFS`，则改为峰值可达增益，
   记录 `W_PEAK` 和 `peak_constrained`。10 LU 以内视为容差，不改变
   增益。
4. 起点采样使用十进制 `ROUND_HALF_UP` 从毫秒换算。发生冲突或越界时，
   按配置顺序从请求点向后找最近的空闲区，记录 `W_SHIFTED`；找不到则
   记录 `W_LATE` 并跳过。
5. 淡变采样数为毫秒时长的 half-up 换算值。淡入增益为
   `cos²(π/2 · (1-t))`，淡出为 `cos²(π/2 · t)`，`t` 在淡变窗口内从
   `0` 到 `1` 含两端；首/末采样端点精确为 `0`。顺延不会改变淡变窗口。
6. 所有已放置片段在最终采样起点直接叠加。主轨峰值超过 `0 dBFS` 时，
   stderr 和 JSON 均记录 `W_PEAK`，并记录峰值及超满幅采样数，不会
   “静默”削波。

## 日志状态与业务码

片段 `status`：

- `success`：成功放置且无警告。
- `warning`：成功放置，但存在 `W_SHIFTED`、`W_PEAK`、`W_GAIN_LIMIT`、
  `W_SILENT` 或 `W_FADE` 等警告。
- `skipped`：未放置；`warnings`/`errors` 中记录具体原因，包括
  `W_FORMAT`、`W_LATE`、`E_AUDIO`。

日志还包含每个片段的 `integrated_lufs`、`gain_requested_db`、最终
`gain_db`、`final_start_ms`、`final_start_sample`、`sample_shift`、
峰值和淡变采样信息。`summary` 包含成功/跳过数量、业务码计数、
`master_peak_dbfs`、超满幅采样数、时间轴长度和退出码。

退出码和 stderr 业务码：

- `1` / `E_CFG`：配置参数、JSON 或命令行参数错误。
- `2` / `E_AUDIO`：不可恢复音频解码错误；或所有片段均被跳过。
  单个损坏片段仍按 `E_AUDIO` 跳过，只要其他片段成功放置，整体退出码
  可为 `0`。
- `0`：至少一个片段成功放置。

`W_FORMAT`、`W_LATE` 等是警告码，不是配置或音频错误码，不会与其他
退出码混用。

## 测试

```bash
python3 -m pytest -q
```

测试覆盖响度与 FFmpeg `ebur128` 交叉校验、采样四舍五入、淡变端点、
重叠顺延、越界跳过、格式/损坏片段、增益与峰值约束、5.1 声道布局、
8/16/24/32-bit 输出、日志和 WAV 的逐字节确定性，以及 `0/1/2` 三类
退出码。

生成确定性本地演示数据：

```bash
python3 scripts/generate_demo.py --output-dir demo_data
python3 -m dialogue_master --config demo_data/config.json \
  --output-dir demo_output
```
