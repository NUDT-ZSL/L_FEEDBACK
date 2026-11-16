# BPM 对齐粒子合成工具

从本地确定性生成的 44.1 kHz PCM WAV 构造技能音效：分析 BPM，生成与节拍对齐的
4 拍粒子循环，叠加原始音频和可选打击乐，经总线软限幅后输出 16 位立体声 WAV。
运行时不访问网络，也不依赖外部音频素材。

## 环境

- Python 3.9+
- NumPy（`pip install numpy`，或使用 `requirements.txt`）

Windows PowerShell 示例：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python bpm_grain.py --help
```

也可直接使用 `bpm_grain.cmd` 启动；`verify_windows.cmd` 会在本地构建虚拟环境、
生成确定性样本、运行测试并执行主场景。脚本使用相对模块目录、`pathlib` 与
`os.replace`，不依赖 Unix 命令。

Linux/macOS 示例：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python bpm_grain.py --help
```

## 基本用法

```bash
python bpm_grain.py \
  --input samples/skill_120_mono.wav \
  --percussion samples/percussion_120_stereo.wav \
  --output demo/mix.wav \
  --seed 4242 \
  --log-level INFO
```

也可以使用模块入口：

```bash
python -m bpm_grains --input samples/skill_120_mono.wav --output demo/mix.wav --seed 4242
```

参数：

- `--input`：必填，44.1 kHz、16 位、单声道或立体声整数 PCM WAV。
- `--percussion`：可选，格式要求与主输入相同。
- `--output`：必填，最终 44.1 kHz、16 位、立体声 WAV 路径。
- `--seed`：粒子源位置、变速变调比例与声像的确定性种子，默认 `0`。
- `--log-level`：`DEBUG`、`INFO`、`WARNING`、`ERROR` 或 `CRITICAL`。

所有日志写入 stderr，工具不会向 stdout 写入业务数据。

## 输出文件

输出 WAV 所在目录还会生成：

- `mixing_plan.json`：必含 `bpm`、`confidence`、`beat_samples`，并记录 4 拍粒子
  循环长度、输出长度及三条轨道的起点。
- `grains.json`：确定性粒子调度表，每个粒子记录 `start_sample`、`end_sample`、
  `source_position`、播放速率、移调半音、声像与循环边界信息。
- `report.json`：总样本数、峰值（线性与 dBFS）、BPM、粒子数量、软限幅状态及
  峰值增益调整量。

WAV 与 JSON 都先写入目标目录中的临时文件，再原子替换正式文件。格式校验或分析
失败时，不会留下半写入的 WAV。

## 错误码

- `ERR_LOW_CONFIDENCE`：BPM 置信度低于 `0.7`，stderr 输出该错误码，退出码 `1`。
- `ERR_BAD_FORMAT`：采样率、位深、声道、PCM 编码、文件头或数据截断不合法，
  stderr 输出该错误码，退出码 `2`。

示例：

```bash
python bpm_grain.py --input bad.wav --output demo/mix.wav
echo $?
```

## 确定性样本与验证

生成内置样本与错误夹具：

```bash
python tools/make_samples.py --directory samples
```

运行自动化验收：

```bash
python -m unittest discover -v
```

测试覆盖 BPM/置信度、端到端 WAV 格式、`mixing_plan.json`、粒子数与起止采样、
同种子逐字节复现、软限幅、立体声输入、低置信度退出码、坏格式退出码以及已有输出
不被破坏。实际执行结果见 [VALIDATION.md](VALIDATION.md)，算法细节见
[TECHNICAL.md](TECHNICAL.md)。
