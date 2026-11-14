# 自适应游戏配乐离线混音引擎

纯 Python/NumPy 实现的离线自适应配乐混音引擎。根据游戏状态（探索、战斗、
过场）对弦乐、打击乐、氛围等分轨做场景调度、ITU-R BS.1770-4 响度标准化、
C0/C1 连续的样条交叉淡化、真峰值限制与帧精确时间轴对齐，输出
48 kHz/24-bit 单声道 PCM WAV（含 RIFF 头与 `fact` 块）及 JSON 报告。

不依赖任何外部音频工具或容器；WAV 读取优先使用 SoundFile（并内置纯
Python PCM 回退读取器），WAV 写入由本引擎直接构造 RIFF 字节，保证
`fact` 块与位深完全可控。

## 环境要求

- Python 3.9+（开发验证使用 3.14）
- NumPy 1.21+
- SoundFile 0.11+
- Windows / Linux / macOS 均可运行；路径全部使用 `pathlib`，无 fork、
  无 POSIX 专属调用，线程池基于标准库 `concurrent.futures`

```powershell
py -m pip install -r requirements.txt
```

## 目录结构

```
adaptive_mixer/
  errors.py      统一 E_ 前缀结构化错误码与 MixerError
  config.py      场景配置解析、校验、SHA-256 版本指纹
  loudness.py    BS.1770-4 K 加权、400ms 短期/积分响度、门限
  envelopes.py   线性/等功率/S 曲线交叉淡化与淡入淡出包络
  limiter.py     4x 过采样真峰值检测与前瞻限制器（-1 dBTP）
  wavio.py       WAV 读取（SoundFile + 回退）与 24bit/fact 写入
  sample_gen.py  固定种子确定性分轨合成
  state.py       断点状态原子持久化与版本兼容校验
  engine.py      场景调度、增益、混音、总线修正、限制、报告
  batch.py       最多 4 线程并发、任务隔离、失败汇总
  cli.py         命令行入口
tests/           34 个 unittest 用例
scripts/run_acceptance.py  端到端验收驱动
```

## 场景配置结构

```json
{
  "sample_rate": 48000,
  "frame_rate": 100,
  "total_frames": 900,
  "target_lufs": -23.0,
  "crossfade_ms": 150.0,
  "crossfade_type": "s_curve",
  "tracks": [
    {"name": "strings", "path": "tracks/strings.wav"}
  ],
  "scenes": [
    {"scene_id": "explore", "start_frame": 0, "end_frame": 300,
     "target_lufs": -26.0, "crossfade_type": "s_curve",
     "active_tracks": ["strings", "ambience"]}
  ],
  "output_format": {"sample_rate": 48000, "channels": 1,
                    "subtype": "PCM_24"}
}
```

校验规则：采样率必须是帧率的整数倍（保证帧到样本零误差映射）；场景必须
首尾相接且结束帧等于 `total_frames`；交叉淡化样本数不得超过场景长度；
所有分轨文件必须存在、采样率一致且样本数等于 `total_frames *
sample_rate / frame_rate`。

## 关键算法

- **K 加权响度**：48 kHz 使用 BS.1770-4 规范双二阶系数（高频架 + RLB
  高通），级联后截断为 FIR，通过重叠相加 FFT 卷积应用；400ms 窗口、
  100ms 跳点，执行 -70 LUFS 绝对门限与相对门限，输出短期与积分 LUFS。
- **分段增益**：每个分轨按场景目标 LUFS 计算线性增益，叠加 10ms 余弦
  边缘淡入淡出防止爆音；场景总线再做一次响度修正，使实际响度收敛到目标。
- **交叉淡化**：以场景边界为中心两侧对称；线性曲线中段恒等、两端三次
  Hermite 平滑；等功率使用 `cos/sin` 恒功率对（`g_out^2+g_in^2=1`）
  并对角度参数做 smoothstep；S 曲线为五次 smootherstep。三条曲线端点
  导数均为 0，与恒定增益区域拼接时 C0、C1 连续且单调。
- **真峰值限制器**：4x 零填充 + 窗函数 sinc 抗混叠上采样测量样本间峰值；
  瞬时攻击、50ms 释放的前瞻包络计算增益衰减，再抗混叠降采样；输出前再次
  测量并保留安全网，确保不超过 -1 dBTP。全局响度通过“限制-补偿”迭代
  收敛到时长加权目标，偏差控制在 ±0.5 LU 内。
- **时间轴**：所有区间由帧号乘 `samples_per_frame` 得到整数样本位置，
  报告中的 `alignment_error_samples` 恒为 0，输出样本数严格等于总帧数
  对应样本数。

## 断点续混

每完成一个场景即原子写入状态文件（临时文件 + `os.replace`），记录状态
格式版本、配置 SHA-256 指纹、输出路径、采样率、总长度与已完成场景；
场景中间产物以 `.npy` 分片保存。恢复时若配置指纹、格式版本、输出路径或
长度任一不匹配，抛出 `E_STATE_VERSION` 拒绝旧状态；状态文件损坏抛出
`E_STATE_CORRUPT`。续混将分片拼接为连续 WAV，输出与一次渲染样本一致。

## 错误码

`E_CONFIG_PARSE`、`E_CONFIG_MISSING_FIELD`、`E_CONFIG_VALUE`、
`E_SAMPLE_RATE_MISMATCH`、`E_TRACK_NOT_FOUND`、`E_TRACK_FORMAT`、
`E_TRACK_LENGTH`、`E_SCENE_OVERLAP`、`E_CROSSFADE_RANGE`、
`E_LUFS_TARGET`、`E_STATE_VERSION`、`E_STATE_CORRUPT`、
`E_OUTPUT_IO`、`E_LIMITER`、`E_INTERNAL`。CLI 失败时以 JSON 打印
`{error, code, message, details}` 并返回非零退出码。

## 使用方式

```powershell
# 生成确定性分轨
py -m adaptive_mixer generate .\work\tracks --tracks strings,percussion,ambience --seconds 9

# 渲染单个配置（同时输出 WAV 与 JSON 报告）
py -m adaptive_mixer render .\work\scene.json .\work\mix.wav `
    --report .\work\report.json --state .\work\state.json

# 并发批处理（任务清单见 artifacts/acceptance/batch/batch_summary.json 示例）
py -m adaptive_mixer batch .\work\jobs.json .\work\batch_report.json --workers 4
```

Python API：

```python
from adaptive_mixer.config import load_config
from adaptive_mixer.engine import MixEngine

cfg = load_config("scene.json")
result = MixEngine(cfg, base_dir=".", output_path="mix.wav").render()
print(result.report["loudness"])
```

## 验证

运行单元测试与端到端验收：

```powershell
py -m unittest discover -s tests
py scripts\run_acceptance.py
```

最近一次本地验证结果（详见 `artifacts/acceptance/acceptance_report.json`）：

- 34/34 单元测试通过，耗时约 2 秒
- 16/16 验收检查通过：RIFF 块顺序 `fmt/fact/data`、432000 样本
  （9.000 秒 @48kHz）、时间轴对齐误差 0、真峰值 -1.004 dBTP、
  整体响度 -23.162 LUFS（目标 -22.667，偏差 0.495 LU）、三场景实际
  响度与目标误差 0.0 LU
- 重复渲染位级一致；中断后续混输出与一次渲染一致；旧版本状态被
  `E_STATE_VERSION` 拒绝；4 并发任务中 3 成功、1 个损坏配置返回
  `E_CONFIG_PARSE` 且不影响其他任务
