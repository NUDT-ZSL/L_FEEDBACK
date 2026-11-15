# audiopipe — 广播级音频转码 / 响度归一化 / 拼接管线

将多段不同采样率、位深、声道布局的 WAV 素材统一转码为播出中间格式
（48 kHz / 24 bit / 立体声），完成 ITU-R BS.1770 响度归一化、交叉淡化拼接，
并输出 JSON 报告。管线为纯本地实现（仅依赖 numpy），输出具有字节级确定性。

## 环境

- Python 3.10/3.11（64 位），numpy >= 1.24
- ffmpeg >= 5.0 在 PATH 中（仅用于生成测试样本与作为响度对照，不参与管线处理）

```bash
python3.11 -m venv .venv && .venv/bin/pip install "numpy>=1.24"
```

## 使用

```bash
python -m audiopipe MANIFEST.json --output-dir OUT_DIR [--jobs N] [--crossfade-ms MS]
```

- 退出码：`0` 全部成功；`1` 部分失败但输出有效；`2` 致命错误无输出。
- 结束后向标准输出打印汇总（成功/失败/复用/耗时/输出路径）。
- 输出：`OUT_DIR/output.wav`、`OUT_DIR/report.json`（均为临时文件 + 原子重命名），
  中间检查点在 `OUT_DIR/.state/`。

### 清单格式

```json
{
  "crossfade_ms": 20,
  "jobs": 4,
  "segments": [
    {"source": "a.wav", "start_sec": 0.5, "end_sec": 5.5,
     "target_loudness_lkfs": -23.0, "fade_in_ms": 10, "fade_out_ms": 10}
  ]
}
```

- `source` 相对路径按清单所在目录解析；`fade_in_ms/fade_out_ms` 可省略（默认 0）。
- `crossfade_ms`（默认 20）与 `jobs`（默认 4）可在清单或命令行指定，命令行优先。

## 处理链（每个片段）

1. **解码**：自研 RIFF/WAVE 解析，支持 PCM 8/16/24/32 bit 整型及
   WAVE_FORMAT_EXTENSIBLE；IEEE float 等编码以 `E_UNSUPPORTED_FORMAT` 拒绝。
2. **有效区间切片**：越界区间记 `E_INVALID_RANGE`。
3. **重采样**：有理比多相（polyphase）重采样，Kaiser 窗 sinc 抗混叠滤波器
   （固定参数：half-width 32、beta 8.6、截止 0.95 × 较低奈奎斯特频率），
   群延迟补偿保证时间对齐；矩阵化实现，结果确定。
4. **响度测量（ITU-R BS.1770-4）**：固定 48 kHz K 加权系数
   （预滤波 + RLB 高通，以 8192 点截断冲激响应实现，残差 < 1e-9）；
   固定门限参数：400 ms 块 / 100 ms 步进（75% 重叠）、绝对门限 -70 LKFS、
   相对门限 -10 LU；声道权重：前置 1.0、环绕 1.41、LFE 不计
   （支持单声道 / 立体声 / 5.1）。
5. **增益与限幅**：`gain = target - loudness`；增益后峰值超过 -1 dBFS 时
   自动回退增益并记 `PEAK_LIMITED` 警告；静音段记 `SILENT_SEGMENT`。
6. **声道映射**：5.1→立体声按 ITU-R BS.775（C、Ls、Rs 以 -3 dB 混入，
   LFE 丢弃）；单声道以 -3 dB 复制为双声道，保持响度一致。
7. **淡入淡出**：段首/段尾升余弦斜坡，超长时保守截断并记 `FADE_TRUNCATED`。

### 拼接

- 严格按清单顺序拼接成功片段；失败片段跳过并记入报告，不影响其他片段。
- 拼接点等功率（sin/cos）交叉淡化，默认 20 ms；若超过相邻片段长度则
  保守截断（保证同一短片段两侧的淡化区不重叠），记 `XFADE_TRUNCATED`。

### 并发与断点恢复

- `ThreadPoolExecutor` 最多 N 路并发处理片段；每片段独立解码/分析/渲染，
  中间结果写入 `.state/seg_XXXX.npy`，互不干扰；拼接在主线程按序完成。
- 每完成一个片段即以 `in_progress` 状态原子写报告；再次运行时若报告存在
  且清单 SHA-256 匹配，已成功片段直接复用检查点，仅处理缺失片段。

### 报告字段（report.json）

每片段：`source`、`start_sec/end_sec`、`target_loudness_lkfs`、
`integrated_loudness_lkfs`、`applied_gain_db`、`fade_in_ms/fade_out_ms`、
`xfade_in_ms/xfade_out_ms`（实际值）、`offset_sec`（时间轴偏移）、
`duration_sec`、`source_sample_rate/source_channels`、`status`、
`error_code`、`warnings[]`。顶层含 `status`、`params`、`summary`。

### 业务错误码

| 代码 | 含义 |
|---|---|
| `E_MISSING_FIELD` | 清单缺字段或类型非法 |
| `E_FILE_NOT_FOUND` | 源文件不存在 |
| `E_INVALID_RANGE` | 有效区间非法或超出文件时长 |
| `E_UNSUPPORTED_FORMAT` | 不支持的编码（如 float32） |
| `E_UNSUPPORTED_LAYOUT` | 声道数非 1/2/6 |
| `E_DECODE_ERROR` | 文件损坏/无法解码 |
| `E_INTERNAL` | 未预期异常（隔离到单片段） |

## 测试与验收

```bash
.venv/bin/python -m unittest discover -s tests   # 15 项单元测试
.venv/bin/python tools/gen_samples.py SAMPLES    # 确定性样本生成
.venv/bin/python tools/acceptance.py WORK_DIR    # 18 项端到端验收
```

实测结果（本仓库交付时）：单元测试 15/15 通过；验收 18/18 通过，包括：

- 自研 BS.1770 测量与 `ffmpeg ebur128` 对照误差 < 0.1 LU（6 种格式样本）；
- 输出各片段响度与目标偏差 <= 0.5 LU（5.1 下混段 <= 2 LU，系 BS.775 下混的
  固有电平偏移，报告中的测量基准为源布局）；
- 拼接点样本差分无突刺（等功率交叉淡化）；
- 重复执行、jobs=1/3、断点恢复后输出 WAV 均逐字节一致；
- 错误隔离（缺失文件 / float32 / 损坏文件 / 非法区间 / 缺字段）互不影响，
  退出码 0/1/2 行为符合规范。

## 已知限制

- 响度测量基准为源声道布局；5.1 下混为立体声后输出响度会有约 1.5 LU 的
  固有偏移（BS.775 系数所致），如需以输出响度为准可将测量移到下混后。
- 短于 400 ms 的片段按整段单块测量（BS.1770 块长不足时的保守处理）。
- 仅支持整型 PCM WAV 输入；float 素材需先离线转换。
