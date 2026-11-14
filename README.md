# 离线游戏音频渲染工具

一个纯 Python 的 Windows 命令行离线渲染器：读取 YAML 场景与多个 WAV 素材，按时间轴、监听者轨迹、确定性随机变体、参数曲线、总线效果链进行流式混音，输出多轨 32 位浮点 WAV 和 JSON 报告。

## 环境与安装

- Windows 10/11 64 位，Python 3.10+
- 无 GPU、外部服务或音频设备依赖

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

依赖：`numpy`、`soundfile`、`PyYAML`。`soundfile` 的 wheel 会自带 libsndfile。

## 命令行

```powershell
python -m offline_audio_render path\to\scene.yaml -o render_out
python -m offline_audio_render path\to\scene.yaml -o render_out --block-size 1024
```

- 成功：stdout 输出摘要 JSON，退出码 `0`
- 致命场景错误：stderr 输出 `{"error": true, "code": "...", "message": "..."}`，退出码 `2`
- 单个素材损坏/采样率不匹配：跳过事件、保留静音、继续渲染，并在报告中记录警告

## 场景 YAML

```yaml
sample_rate: 48000          # 必填，正整数
layout: stereo              # 必填：mono | stereo | 5.1
global_seed: 20261115       # 可选，默认 0
block_size: 4096            # 可选，源读取块大小；不影响输出样本
duration: 5.0               # 可选；缺省取最后事件结束点
min_distance: 1.0           # 可选，反平方衰减最小距离

trajectory:
  - time: 0.0
    position: [0.0, 0.0, 0.0]
    yaw: 0.0                # 弧度，0 朝 +X，正向朝 +Y 旋转
  - time: 2.0
    position: [2.0, 0.0, 0.0]
    yaw: 0.5

buses:
  - name: fx
    gain_db: -1.0
    limiter: true
    delay_seconds: 0.1
    delay_feedback: 0.25    # [0, 1)
    route_to: master
  - name: master
    gain_db: 0.0
    limiter: true

events:
  - id: ev_left
    time: 0.2
    source: sources/tone.wav
    position: [-3.0, 2.0, 0.0]
    voice: voice_a           # 可选；同 voice 不允许同时触发
    seed_offset: 11
    bus: master
    gain_curve:
      - {t: 0.0, value: -6.0}
      - {t: 1.0, value: 0.0}
    lowpass_curve:
      - {t: 0.0, value: 500.0}
      - {t: 1.0, value: 12000.0}
```

曲线时间 `t` 使用事件内部归一化时间 `[0, 1]`，必须从 `0.0` 单调递增到 `1.0`，渲染时按采样率逐样本线性插值。未定义总线时自动创建带限幅器的 `master` 总线。

## 关键实现

- **确定性随机**：每事件由 `SeedSequence([global_seed, seed_offset])` 派生独立 RNG，固定抽取音高 `±5%`、起始偏移 `±20ms`、增益抖动 `±1.5dB`；不使用系统时间或全局 RNG。
- **空间化**：监听者轨迹线性插值，超出范围钳到端点；立体声等功率声像，5.1 输出顺序为 `L, R, C, LFE, Ls, Rs`，LFE 默认不路由；距离增益为 `(min_distance / max(distance, min_distance))^2`。
- **滤波**：每事件、每源通道独立的一阶时变低通 `y[n] = alpha*x[n] + (1-alpha)*y[n-1]`，截止频率逐样本更新，状态跨块连续。
- **流式处理**：源以固定 512 帧内部量子读取/重采样，时间轴以 512 帧量子混音；外部 `block_size` 只改变 I/O 请求，不改变任何浮点运算顺序。
- **总线链**：按 `gain -> soft-knee limiter -> feedback delay` 顺序处理；限幅阈值 -1 dBFS，延迟线和限幅包络在总线内跨块保持。
- **通道布局**：mono 直接空间化；多通道源按输出布局上/下混，6 通道源下混记录 `W2004`；最终写盘前钳制到 `[-1, 1]`，输出 FLOAT WAV。
- **字节级确定性**：关闭 libsndfile 浮点 WAV 默认 PEAK chunk（其中含运行时间戳），重复运行输出完全相同的 WAV 字节。

## 错误码

| 码 | 含义 |
| --- | --- |
| `E1001` | 缺失必填字段 |
| `E1002` | 非法输出通道布局 |
| `E1003` | 同一 voice 的事件时间冲突 |
| `E1004` | 非法数值 |
| `E1005` | 监听者轨迹非法 |
| `E1006` | 参数曲线非法 |
| `E1007` | 总线配置/路由非法 |
| `E1008` | 非法采样率 |
| `W2001` | 素材无法打开/无法识别 |
| `W2002` | 素材采样率与场景不匹配 |
| `W2003` | 素材损坏 |
| `W2004` | 发生通道下混 |
| `W2005` | 不支持的源通道数 |

## 本地验证

生成确定性夹具：

```powershell
python scripts\generate_fixtures.py fixtures
```

运行测试：

```powershell
python -m unittest discover -s tests -v
```

测试覆盖：YAML 错误码、RNG 稳定性、空间化增益、反平方衰减、曲线/滤波、总线限幅与延迟、错误恢复、通道布局、跨进程字节确定性、块大小无关性，以及超过 10 分钟素材的流式内存比例。

## 报告

`report.json` 包含：

- 输出布局、采样率、总帧数、全局种子与实际块大小
- 每个事件的 `start_sample`、`end_sample`、总增益抖动（dB）、事件种子值、跳过标记和事件级警告
- 跳过事件列表与顶层警告列表
- 最大源素材 float32 等效大小、流式工作区峰值字节及比例
- 每个顶层总线对应的 WAV 路径
