# scene-mix

Windows 命令行工具：从内置确定性合成器生成多路摄像机 WAV 和时间轴，
完成整数样本级对齐、BS.1770 分段响度/自适应增益、平滑包络、软限幅与
48 kHz / 16-bit PCM 立体声交付混音，并输出 UTF-8（无 BOM）JSON 报告。

## 环境

- Windows 10/11 64 位（代码同时可在 Linux/macOS 运行）
- Python 3.9+
- NumPy、SoundFile

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

无需 Docker、FFmpeg 或外部样本。

## 快速开始

生成两个输入组，每组 6 路；第 2 路静音，第 3 路削波：

```powershell
python main.py generate --seeds 101,202 --channels 6 --out inputs --silent-track 2 --clip-track 3
python main.py batch --input-root inputs --output-root delivery --jobs 4
```

单组处理：

```powershell
python main.py mix --input-dir inputs\seed_101 --output-dir delivery\seed_101
```

也可以使用 `python -m scene_mix`、Windows 包装脚本 `scene-mix.cmd`，
或安装后使用 `scene-mix` 命令。

## 输入与输出

`generate` 为每个 seed 创建 `inputs/seed_<N>/`：

- `cam_XX.wav`：48 kHz、16-bit PCM、立体声。
- `timeline.json`：参考轨、场景边界、目标响度、每路偏移和合成元数据。

`batch` 默认同时处理 4 组，实际并发数不会超过逻辑 CPU 核数。每组有独立输出目录：

- `mix.wav`：最终 48 kHz / 16-bit PCM 立体声。
- `report.json`：处理报告。
- `.progress`：仅存在于未完成或失败组；成功后删除。
- `delivery/batch_summary.json`：逐组成功、跳过或失败汇总。

### 时间轴

关键字段如下（实际文件还包含合成参数和测量值）：

```json
{
  "schema_version": "1.0",
  "seed": 101,
  "sample_rate": 48000,
  "reference_track": "cam_01.wav",
  "target_lufs": -23.0,
  "fade_ms": 50,
  "tracks": [
    {
      "id": "cam_01",
      "file": "cam_01.wav",
      "offset_samples": 0,
      "target_lufs": -18.0
    }
  ],
  "scenes": [
    {
      "id": "scene_01",
      "start_sample": 38400,
      "end_sample": 144000,
      "target_lufs": -23.0
    }
  ]
}
```

`offset_samples` 是仅供验收比对的真值，处理逻辑不会读取它来对齐。约定：

- 正值：该路内容相对参考路延迟，对齐时向左裁剪对应样本。
- 负值：该路内容相对参考路提前，对齐时向前补零对应样本。
- 合成器保证每路相对参考路偏移在 ±300 ms 内，且场景坐标位于参考路时间轴。

### 报告

报告包含：

- `tracks[]`：`offset_samples`、`offset_error_samples`、削波/静音状态、
  每路警告和 `business_code`。
- `scenes[]`：BS.1770 响度、需求增益、实际应用增益、是否钳制。
- `warnings[]` / `business_codes[]`：全局警告码。
- `pre_normalization_peak_dbfs`、`final_output_peak_dbfs`、
  `normalization_gain_db`、`limiter_active`。
- `retry_count`：批处理恢复时累计的该组重试次数。

警告码固定为：

| 代码 | 含义 |
| --- | --- |
| `SILENT_TRACK` | 静音路被跳过，不参与偏移估计和混音 |
| `CLIPPED_INPUT` | 输入存在饱和样本；仍估计偏移并参与混音 |
| `GAIN_CLAMPED` | 某场景需求增益超出 ±12 dB，已钳制到边界 |
| `LIMITER_ACTIVE` | 增益后、归一化前峰值超过 0 dBFS，已启用软限幅 |

## 算法说明

1. **确定性合成**：使用显式 `SeedSequence` 和 NumPy PCG64；共享节目底噪、
   不规则同步 chirp、每路独立噪声/标记音、场景响度纹理均由 seed 决定。
   同一 seed、通道和故障轨参数生成逐样本一致的 WAV 与字节一致的 JSON。
2. **偏移估计**：下混为单声道后做去直流、滑动能量归一化 FFT 互相关，
   在 ±1 秒范围搜索整数样本峰值；合成样本的实测误差为 0 个样本。
3. **分段响度**：48 kHz BS.1770-4 两级 K 加权；400 ms 块、100 ms 跳步；
   先做 -70 LUFS 绝对门限，再做相对 -10 dB 门限，计算场景积分响度。
4. **增益包络**：目标为每场景 `-23 LUFS`，需求增益限制在 ±12 dB；
   场景边界前后各 50 ms 线性交叉淡化。
5. **对齐混音**：按估计偏移裁剪/补零，乘同一场景增益包络后求和；
   预归一化峰值超过 0 dBFS 时使用平滑拐点软限幅，最后峰值归一化到
   -1 dBFS，写 PCM16 立体声。
6. **并发与恢复**：每组工作进程在开始处理前原子写 `.progress`，包含
   `attempts`/`retry_count`。成功删除；失败或强杀保留。下次运行跳过
   “有报告和 WAV 且无 `.progress`”的组，其余组重试并把累计次数写进新报告。

## 测试

```powershell
python -m unittest discover -s tests -v
```

测试覆盖：

- 合成器同 seed 的 JSON/WAV 字节级确定性、±300 ms 偏移、静音和削波。
- K 加权与直接 IIR 递推的一致性、50 ms 线性交叉淡化。
- 静音跳过、削波偏移、偏移误差 ≤5 样本、场景响度/增益钳制和全部警告码。
- 48 kHz / 16-bit / 立体声输出、-1 dBFS 峰值、UTF-8 无 BOM。
- 并发失败隔离、`--jobs` CPU 上限、已完成组跳过。
- POSIX 下真实 `SIGKILL` 强杀后 `.progress` 保留、重跑成功且报告累计重试次数。

本次实际验证结果见 `VALIDATION.md`。

## 约束

- 输入 WAV 必须是 48 kHz；工具会明确报错，不做隐式重采样。
- 批处理组目录必须包含固定文件名 `timeline.json`。
- 输出时长和场景坐标以参考 WAV 为准；参考轨必须可听。
