# 实际验证记录

验证环境：Linux 容器中的 Python 3.14.4（代码语法兼容 Python 3.10+），依赖 `numpy 2.5.4`、`soundfile 0.14.0`、`PyYAML 6.0.3`。所有素材均由 `scripts/generate_fixtures.py` 通过固定种子本地生成，未访问网络或音频设备。

## 自动化测试

命令：

```bash
.venv/bin/python -m unittest discover -s tests
```

最近一次结果：

```text
Ran 44 tests in 15.645s

OK
```

覆盖点：

- 场景缺失字段、非法布局、voice 时间冲突、轨迹、曲线、总线环的错误码
- RNG 种子隔离、范围约束、重复运行稳定性
- 立体声等功率声像、5.1 单位功率、LFE 静默、反平方衰减与最小距离
- 监听者轨迹线性插值和端点钳制
- 分段线性增益/低通曲线、时变一阶低通与直接递推参考一致
- 软拐点限幅阈值、跨块状态、反馈延迟与参考环形缓冲一致
- 素材损坏、采样率不匹配时跳过事件、保留静音、继续输出有效 WAV
- mono/stereo/5.1 通道数一致、下混警告、32 位浮点输出
- 同场景重复渲染 WAV 字节一致
- 不同 `block_size` 输出 WAV 字节一致
- 超过 10 分钟的 8 kHz mono 与 48 kHz stereo 素材流式读取，工作区峰值低于源素材 float32 等效大小的 20%

## 端到端确定性

命令：

```bash
.venv/bin/python scripts/generate_fixtures.py fixtures
for i in 1 2 3 4 5 6 7 8; do
  .venv/bin/python -m offline_audio_render fixtures/scene_stereo.yaml -o /tmp/det-$i
done
sha256sum /tmp/det-*/master.wav
```

实测 8 次独立进程渲染得到唯一哈希：

```text
2deffb6aea1999d5dd2ee6bae29ccd916a7c78169e5bc0f351b37b4d21981576
```

不同块大小（257、1024、4096、8192）输出同一哈希，证明结果与块大小无关。

## 错误恢复

夹具包含 `broken.wav`（libsndfile 无法识别）和 24 kHz 素材（场景为 48 kHz）。渲染后：

- `report.json` 的 `skipped_events` 为 `["ev_broken", "ev_rate_mismatch"]`
- 顶层警告包含 `W2001` 与 `W2002`
- `master.wav` 仍为有效 stereo FLOAT WAV，跳过事件对应区间保持静音

## 5.1 布局

`fixtures/scene_surround.yaml` 渲染输出：

- WAV 通道数：6
- 通道顺序：`L, R, C, LFE, Ls, Rs`
- LFE 通道保持全零（场景不生成独立 LFE）
- 6 通道源上/下混处理产生 `W2004` 警告

## 内存控制

流式测试直接插桩 `LinearResampler` 的源缓冲、就绪缓冲与每次读取块：

- 8 kHz、660 秒 mono 源：约 21.1 MB float32 等效源大小，工作区峰值远小于 20% 阈值
- 48 kHz、601 秒 stereo 源：约 220.1 MB float32 等效源大小，工作区峰值远小于 20% 阈值
- 渲染全程不一次性加载完整源文件
