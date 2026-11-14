# 离线多轨混流处理器 — 技术说明

## 1. 目标与范围

`audiomixer` 是一个完全离线、确定性的多轨音频混流处理器。所有测试素材由程序以相位累加
正弦波确定性生成，不读取外部媒体、不访问网络。处理链覆盖：

1. 确定性 WAV 素材生成（16/24/32-bit PCM，单/双声道）；
2. 多轨时间轴对齐、每轨增益、等功率声像、静音；
3. 采样率转换（FFmpeg SoX 28-bit 优先，SWResampler 次之，纯 NumPy 带限插值兜底）；
4. 单声道按声像映射到立体声；
5. 线性 / 等功率淡入淡出与同轨交叉淡化；
6. ITU-R BS.1770-4 K 加权 + 双门限响度测量与标准化、峰值约束；
7. 自校验 JSON 混流报告（文件/样本双 SHA-256、可复算）；
8. 业务错误码、非零退出码、失败无残留 WAV；
9. 批量处理与汇总报告。

## 2. 目录结构

```
audiomixer/
  config.py      场景/批量 JSON 配置模型与时间轴语义校验
  errors.py      业务错误码与 MixerError
  generator.py   确定性正弦波 WAV 生成器
  wavio.py       PCM WAV 读写（原子写、24-bit 打包、量化）
  resampler.py   FFmpeg soxr/swr + NumPy 带限插值，长度残差统计
  dsp.py         等功率声像、线性/等功率淡化曲线、dB 换算
  loudness.py    BS.1770-4 K 加权、门限、峰值、标准化
  mixer.py       时间轴对齐、声道映射、淡化、增益、混音总线
  report.py      报告组装、哈希、verify 复算、批量汇总
  pipeline.py    单场景/批量编排与失败隔离
  cli.py         run / batch / verify 命令行
tests/           44 个 unittest 用例（无需 pytest）
examples/        单场景、峰值失败、混合批量示例配置
```

## 3. 处理流水线

```
场景 JSON
  └─ parse_scene（结构校验 + 时间轴/淡化越界校验）
       └─ generate_assets（每事件一个确定性 WAV，可通过 generate:false 关闭）
            └─ 逐事件 read_wav
                 └─ resample_mono（每声道；同采样率走 passthrough）
                      └─ 等功率声像映射为立体声
                           └─ 淡入/淡出/交叉淡化包络
                                └─ 轨增益 -> 累加进立体声总线
                                     └─ normalise_loudness（目标响度 + 峰值上限）
                                          └─ 量化 -> 原子写 WAV -> 组装并原子写报告
```

WAV 先于报告写入，因为报告需要对最终 WAV 求 SHA-256；两者都经过临时文件 +
`os.replace` 原子替换。任何步骤抛出 `MixerError` 都会先删除目标 WAV（含历史残留），
再写失败报告，因此不会留下不完整输出。

## 4. 关键算法

### 4.1 确定性生成

`render_sine` 使用 `sin(2π f n / fs mod 2π)`，无随机数、无时间戳，相同配置逐样本一致。
跨机器、跨次运行的 WAV 字节级 SHA-256 相同（由 `test_deterministic_outputs_are_bit_identical`
锁定）。

### 4.2 等功率声像

`L = cos((p+1)π/4)`，`R = sin((p+1)π/4)`，满足 `L²+R²=1`。`p=-1` 极左、`0` 中央
（各 -3.01 dB）、`+1` 极右。单声道素材按该增益复制为立体声；立体声素材对两个声道
分别施加 L/R 增益。

### 4.3 重采样与长度残差

优先 `aresample=resampler=soxr:precision=28`；soxr 不可用时退回 swr 窗 sinc；
ffmpeg 缺失时使用内置 Kaiser(β≈14.77) 窗 sinc（截止 0.9 倍奈奎斯特）。
理论输出长度 `round(N_in * f_out / f_in)`，FFmpeg 原始长度与理论值的差记为
`length_residual_samples`，超出部分 `trimmed_samples`、不足部分 `padded_samples`
零填充，最终长度严格等于理论值，保证时间轴对齐。

### 4.4 淡化与交叉淡化

- 线性：`linspace(0,1)`；等功率：淡入 `sin(0→π/2)`、淡出 `cos(0→π/2)`，
  配对曲线满足 `g_in² + g_out² = 1`，交叉区功率恒定。
- 同轨后一事件声明 `crossfade`（秒），其淡入段被交叉淡化曲线覆盖；
  淡化帧数被钳制到事件持续时间内，配置阶段即拒绝 `fade_in+fade_out > duration`
  与超过事件长度的交叉淡化（`E_OUT_OF_RANGE`）。
- 报告记录每段淡化的事件相对帧范围、工程绝对帧范围、帧数与曲线类型。

### 4.5 BS.1770-4 响度

- K 加权两级双二阶：前置高架（+4 dB）+ RLB 高通。48/44.1/32 kHz 使用 ITU 官方
  参考系数；其他采样率用 RBJ cookbook 逼近。转置直接 II 型、float64 逐样本滤波。
- 400 ms 块、100 ms 跳幅（75% 重叠），块均方能量按声道权重求和（立体声权重均为 1.0）。
- 绝对门限 -70 LKFS，再以非门限参考 -10 LU 得相对门限，对通过两级门限的块求积分响度。
- 峰值默认采样峰值；可选 4× 过采样 sinc 插值真峰值。
- 标准化先求目标增益 `target - measured`（dB），再与峰值允许增益
  `peak_limit - peak_before` 取较小值。若峰值钳制使响度缺口超过
  `peak_shortfall_tolerance_lu`，抛 `E_PEAK_LIMIT` 且不写 WAV。

与 FFmpeg `ebur128` 的交叉验证（48 kHz、满幅正弦）：

| 频率 | 本实现 LUFS | FFmpeg LUFS | 偏差 |
| --- | --- | --- | --- |
| 100 Hz | -4.84 | -4.80 | -0.04 |
| 500 Hz | -3.66 | -3.70 | +0.04 |
| 1 kHz | -3.00 | -3.00 | -0.00 |
| 2 kHz | -0.63 | -0.60 | -0.03 |
| 5 kHz | +0.31 | +0.30 | +0.01 |
| 10 kHz | +0.34 | +0.30 | +0.04 |

全频段偏差 ≤ 0.05 LU。

## 5. 报告与自校验

成功报告含：工程采样率、总帧数/时长、输出 WAV 的 `file_sha256`、量化后样本的
`sample_sha256` 与首尾样本、轨/事件清单（事件工程起止帧、对齐帧数、声道映射、
重采样残差、淡化区间）、标准化前后响度与峰值、门限统计、环境（ffmpeg 版本/重采样引擎）。
报告对除 `report_sha256` 外的全部内容做规范化 JSON 哈希，`verify` 重新打开 WAV：

- 重算 `report_sha256`、WAV 文件哈希、样本哈希；
- 复算采样率、位宽、帧数、声道；
- 重新测量积分响度（与报告差 < 0.05 LU）并核对峰值上限。

任一项失败则 `verify` 以退出码 2 结束。

## 6. 错误码与退出码

| 错误码 | 触发条件 |
| --- | --- |
| `E_MISSING_ASSET` | 引用素材/配置/报告文件不存在 |
| `E_UNSUPPORTED_FORMAT` | 非 WAV、损坏 WAV、非 16/24/32-bit PCM、非单/双声道 |
| `E_OUT_OF_RANGE` | 事件越出工程时长、淡化/交叉淡化超出事件、声像越界、事件非法重叠 |
| `E_PEAK_LIMIT` | 峰值约束无法在容差内满足目标响度，或对静音做标准化 |
| `E_INVALID_CONFIG` | JSON 非法、缺字段、类型错误 |

CLI 退出码：成功 `0`，业务错误 `2`，用法错误 `64`。失败场景的输出 WAV 一定不存在
（含对历史残留的清理），失败报告与批量汇总仍落盘。

## 7. 配置要点

- 顶层 `generate`（默认 `true`）：`false` 时不生成素材，严格依赖既有 WAV，
  用于“缺素材”类异常验证。
- 事件 `asset` 可显式指定素材路径；缺省按 `asset_dir/<scene>_<event>.wav` 约定。
- 事件 `sample_rate` 与工程不同即触发重采样；`channels:2` 生成/读取立体声素材。
- `loudness.peak_shortfall_tolerance_lu` 控制峰值钳制与目标响度之间的可接受缺口。

## 8. 运行方式

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 确保 ffmpeg 在 PATH（可选；缺失时自动使用 NumPy 重采样）
python -m audiomixer run examples/scene_dialogue.json
python -m audiomixer verify examples/out/dialogue_scene.report.json
python -m audiomixer batch examples/batch_scenes.json
python -m unittest discover -s tests
```
