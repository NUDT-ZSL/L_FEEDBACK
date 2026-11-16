# RhythmForge 技术说明

面向节奏游戏的离线、确定性音轨构建工具。输入一份谱面 JSON 与若干本地
PCM WAV 分层素材，输出成品混音、逐分层分轨、采样帧精确的同步标记、
响度归一化报告以及支持位级增量重建的清单。

## 1. 目录与模块

| 模块 | 职责 |
| --- | --- |
| `generator.py` | 确定性样本与谱面生成（固定种子 `RandomState`，逐字节可复现） |
| `waveio.py` | 16/24/32-bit PCM WAV 读写，含手工 24-bit 有符号编解码 |
| `timing.py` | BPM/拍号解析，拍→秒→采样帧分段线性单调映射与非法映射拒绝 |
| `dsp.py` | 多相 FIR 重采样、单/双声道归一化、相位声码器变速不变调、等功率交叉淡化 |
| `renderer.py` | 素材加载、按拍调度、段落级变速与分层渲染 |
| `assembly.py` | 段落拼接与边界等功率交叉淡化，总帧数严格等于段落帧数之和 |
| `loudness.py` | BS.1770 风格 K 加权 + 分块门控响度、可复现增益、峰值保护 |
| `markers.py` | 段落/BPM/拍号同步标记，帧坐标与成品音频一一对应 |
| `hashing.py` | 素材、谱面语义、数组的稳定 SHA-256 |
| `locking.py` | 输出目录跨进程互斥锁（POSIX `fcntl` / Windows `msvcrt`） |
| `pipeline.py` | 主流水线、内容寻址缓存、增量重建、原子写与清单 |
| `cli.py` | `generate` / `build` / `verify` 命令与 JSON 错误契约 |
| `ffmpeg_check.py` | 可选的 FFmpeg 容器封装校验，不参与核心构建 |

内部统一基准：48 kHz、双声道、`float32`。

## 2. 时间映射

`TimeMap` 以 `(beat, bpm)` 锚点构造分段线性函数。相邻锚点间每拍时长为
`60 / bpm` 秒，锚点处秒坐标连续。`beat_to_frame` 在 float64 参考值上
四舍五入到整数帧，单点误差不超过 0.5 帧；`validate_switch_accuracy`
在每个 BPM 切换点前后各一拍的稠密网格上验证取整误差 ≤ 1 帧。

非法输入（零/负/非有限 BPM、锚点非严格递增、段落存在间隙或重叠、
`end_beat` 不合法）一律抛出 `MAPPING_CONFLICT`（退出码 5）。

## 3. 素材归一化与变速

* 重采样使用 `scipy.signal.resample_poly`（多相 FIR），输出长度按
  `frames * target/source` 精确对齐，前置静音与起拍偏移随速率比换算保留。
* 单声道复制为双声道；双声道保持不变；时间语义（长度、起拍位置）不改变。
* 变速采用 STFT 相位声码器（2048 窗 / 512 跳），`factor == 1` 时直通，
  输出按段落声明帧数强制裁剪或补零，保证帧级时长一致。

## 4. 段落、混音与叠加关系

每个段落先对各层素材按 `native_bpm / section_bpm` 预伸缩，再在最终帧
网格上按音符拍位置放置（起拍对齐）。段落输出帧数由 `TimeMap` 唯一确定。
边界使用等功率余弦交叉淡化，段落槽位长度不变，因此
`总帧数 = Σ 段落帧数`，标记位置不发生偏移。

成品混音仅叠加谱面 `enabled=true` 的分层；所有分轨与成品使用**同一个**
响度增益，因此在浮点容差（测试阈值 `2e-4`，实测约 `1e-7`）内
`master == Σ enabled stems`。

## 5. 响度归一化与峰值保护

测量遵循 BS.1770/EBU R128 结构：高架 + 高通两级 K 加权、声道均方能量、
400 ms/75% 重叠分块、−70 LUFS 绝对门与 −10 LU 相对门。先计算达到目标
响度所需增益，再按 `true_peak_dbfs` 计算峰值保护系数并取二者乘积。
默认谱面目标 `-33 LUFS` 可在 `-1 dBFS` 峰值限制内直接达成（峰值保护
增益为 0）；激进目标由测试 `test_peak_protection_engages_for_aggressive_target`
覆盖，验证峰值保护强制介入且任何采样点不超过满刻度限制。

## 6. 同步标记与增量重建

`markers.json` 含 `section`（含时间轴结束点）、`bpm`、`timesig` 三类，
均带 `beat` 与整数 `frame`，按帧排序。段落缓存键只纳入影响该段落采样的
输入（段落自身参数、覆盖其拍区间的 BPM、相关音符、分层描述与素材哈希），
因此修改某一段落仅重建该段落，其余 `.npy` 缓存位级不变。所有写盘经
临时文件 + `os.replace` 原子完成；无操作重建的全部产物（含
`manifest.json`）逐字节一致（易变的 `rebuilt/reused` 动作摘要仅随返回值
给出，不写入清单）。

## 7. 退出码与错误契约

| 码 | 常量 | 触发场景 |
| --- | --- | --- |
| 0 | `OK` | 成功 |
| 1 | `INTERNAL_ERROR` | 未预期内部错误 |
| 2 | `CHART_SYNTAX` | JSON 语法/结构非法、未知分层、字段缺失 |
| 3 | `MISSING_SOURCE` | 谱面或 WAV 素材缺失 |
| 4 | `UNSUPPORTED_FORMAT` | 非 PCM、不支持的位深/声道、容器损坏 |
| 5 | `MAPPING_CONFLICT` | 非单调/零负 BPM、段落间隙重叠、非法伸缩因子 |
| 6 | `DURATION_MISMATCH` | 段落帧数与声明时长不符、叠加/峰值校验失败 |
| 7 | `LOCKED_OUTPUT` | 输出目录已被另一构建持有互斥锁 |
| 8 | `ARGUMENT_ERROR` | 命令行参数错误 |

失败时向 stderr 输出单行 JSON：
`{"ok": false, "error": {"code": <int>, "message": <str>, "details": {...}}}`。

## 8. Windows 命令行流程

```powershell
py -3.11 -m pip install numpy==1.26.4 scipy==1.11.4 pytest==8.2.2
ffmpeg -version
# 解压项目后，在项目根目录：
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
python -m rhythmforge generate --samples-dir samples --chart samples/chart.json
python -m rhythmforge build    --chart samples/chart.json --samples-dir samples --output output
python -m rhythmforge verify   --chart samples/chart.json --samples-dir samples --output output
python -m pytest tests
```

也可 `pip install .` 后直接使用 `rhythmforge` 入口脚本。FFmpeg 仅在
`verify` 阶段做封装校验，缺失时自动跳过且不影响构建。

## 9. 实际验证结果（Linux/WSL2，Python 3.14 + NumPy 2.5 + SciPy 1.18）

* `pytest tests`：48 项全部通过。
* 生成器重复运行：6 个 WAV 逐字节一致。
* 格式覆盖：44.1k/48k × 单/双声道 × 16/24-bit 组合齐全。
* 无操作重建：全部产物 SHA-256 不变；修改单一段落仅重建该段，其余缓存不变。
* 分轨叠加最大绝对误差约 `3.6e-7`（阈值 `2e-4`）。
* 默认构建输出响度 −33 LUFS（误差 < 2e-10），采样峰值 0.838 < 0.891 限制。
* 并发构建：持锁时第二个进程立即返回退出码 7。
* FFmpeg 对 `master.wav` 与分轨解码校验返回码 0。

说明：任务环境给定 Python 3.11 与 NumPy 1.26/SciPy 1.11 版本号；当前
容器实际为 Python 3.14，pip 安装了对应最新稳定轮子（NumPy 2.5 /
SciPy 1.18 / pytest 9）。代码仅使用两版均稳定的公共 API，
`requirements.txt` 与 `pyproject.toml` 同时声明了任务要求的版本区间，
在 Windows + Python 3.11 上按第 8 节安装即可复现。
