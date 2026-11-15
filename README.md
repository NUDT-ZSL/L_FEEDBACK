# Seamless Multi-Track Loop Processor

离线多轨音频无缝循环处理器。读取 JSON 项目描述，把多个分轨按给定循环区间切片，
在循环点生成等功率（equal-power）正弦/余弦交叉淡化，所有轨道严格采样点对齐，
重复到目标时长后混音为一个立体声 WAV，并输出 JSON 处理报告。

## 环境与依赖

- Windows 10+（同样适用于 macOS / Linux）
- Python 3.9+
- 依赖：`pip install numpy soundfile`
- 不使用 Docker，不访问外部音频或网络；测试样本由固定公式与固定随机种子合成，
  完全可复现。

## 快速开始

```bash
pip install numpy soundfile
python make_samples.py                 # 生成确定性 WAV 样本与示例 JSON
python -m seamless_loop samples/project_demo.json
```

命令行只接受一个参数——项目 JSON 路径：

```bash
python -m seamless_loop path/to/project.json
```

输出写入 JSON 指定的 `output_dir`：

- `<项目名>_loop.wav`：32-bit float 立体声混音
- `<项目名>_loop_report.json`：处理报告

## 项目 JSON 字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `project_name` | string | 项目名，决定输出文件命名 |
| `input_files` | string[] | 输入 WAV 路径列表，相对路径基于 JSON 所在目录 |
| `loop_start` / `loop_end` | number | 循环起止时间（秒） |
| `crossfade_duration` | number | 交叉淡化时长（秒），必须 > 0 |
| `target_duration` | number | 可选，输出总时长（秒），可含多个完整循环 |
| `output_dir` | string | 输出目录，不存在会自动创建，执行前校验可写 |
| `overwrite` | bool | 可选，默认 `true`；为 `false` 且输出已存在时退出码 5 |

输入约束：PCM WAV；采样率 44100 或 48000 Hz；位深 `PCM_16` / `PCM_24` /
32 位 `FLOAT`；单声道或立体声。所有轨道必须同一采样率，否则拒绝处理。

## 采样点与对齐

- 秒到采样点统一使用“四舍五入（half-up）”：`floor(t * sr + 0.5)`，与平台
  默认的银行家舍入无关，结果确定。
- 循环区间为半开区间 `[start, end)`，循环长度 `N = end - start`（采样点）。
- 所有轨道使用同一对起止采样点切片，因此逐轨循环长度完全一致、严格对齐。

## 等功率交叉淡化原理

将长度为 `N` 的循环区域划分为头部 `H`、尾部 `T`（各 `F` 个采样点）与中间段
`M`（`N-2F`）。原始拼接的爆音来自尾→头的包裹跳变 `T[-1] -> H[0]`。淡化区
`Z` 长度为 `F`，相位 `p = k/(F-1)`：

```
Z[k] = T[k] * cos(pi/2 * p) + H[k] * sin(pi/2 * p)
loop = [ M ][ Z ]          # 输出循环长度 N - F
```

- `cos(p)^2 + sin(p)^2 = 1`：淡化区总功率恒定，是**等功率**而非等幅度淡化；
- 端点被钳制为精确的 0.0/1.0，`M->Z` 连接点与周期接缝 `Z[-1]->M[0]` 都只
  保留源信号内部相邻采样点的自然步进，原始包裹跳变被完全吸收进淡化区，无爆音；
- 单声道输入会复制为左右相同的双声道；多轨各自独立交叉淡化后再求和混音。

## 逐轨淡化长度动态缩短

每个轨道单独构造循环。若某轨循环区间样本不足以容纳一个完整淡化区
（可用样本 `< 2F`），该轨淡化长度自动缩短为 `可用样本 // 2`，并通过 logging
输出 WARNING，同时在报告的 `crossfade_adjusted` / `crossfade_samples_used`
字段中记录。

## 重复与混音

- 按 `target_duration` 换算目标总采样点，循环向上取整重复后截断到精确点数；
  当总点数不是循环长度整数倍时记录 WARNING。
- 各轨转立体声后逐点求和；若混音峰值超过满刻度，按 `1/peak` 整体衰减避免削波，
  衰减系数写入报告 `output_gain`（未衰减时为 1.0）。
- 输出 WAV 固定为 32-bit float 立体声，最大限度保留处理精度。

## 处理报告

`<项目名>_loop_report.json` 包含：

- 每轨：配置淡化长度、实际使用淡化长度（采样点）、是否调整、循环长度（采样点）、
  源声道与位深；
- 循环区间起止采样点、循环时长、重复次数、输出增益；
- `output_summary`：采样率、声道数、总采样点数、时长、子类型。

## 退出码

| 退出码 | 含义 |
| --- | --- |
| 0 | 成功 |
| 2 | 配置错误（JSON 缺失/非法、字段缺失、输出目录不可写等） |
| 3 | 文件读取错误（文件缺失、格式/位深/采样率/声道不支持、采样率不一致） |
| 4 | 参数范围错误（循环区间超出文件时长、淡化时长 ≤ 0、目标时长过短等） |
| 5 | `overwrite=false` 且输出文件已存在 |

所有错误均以非零退出码终止，并在 stderr 输出 `ERROR: ...` 明确信息。

## 项目结构

```
seamless_loop/
  __init__.py        # 包入口与公开 API
  errors.py          # 异常类型与退出码 2/3/4/5
  crossfade.py       # 等功率窗口与无缝循环构造
  audioio.py         # 时间->采样点换算、WAV 校验与读取
  config.py          # JSON 项目加载、字段与可写性校验
  processor.py       # 端到端管线：切片/淡化/重复/混音/报告
  cli.py / __main__.py
make_samples.py      # 确定性样本与示例 JSON 生成器
tests/               # unittest 测试（51 个用例）
```

## 运行测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖：等功率窗口恒功率与精确端点、循环接缝无爆音、逐轨淡化缩短与日志、
采样点四舍五入、配置/范围校验、采样率不一致与不支持格式、单声道转立体声、
精确采样点输出、确定性，以及全部退出码（2/3/4/5）。
