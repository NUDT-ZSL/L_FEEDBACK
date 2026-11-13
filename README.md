# audiopipe — 确定性多语言 WAV 混音流水线

纯 Python 标准库实现（无 NumPy / 无第三方库 / 无网络），在 Python 3.11 与
3.14 上均可运行。满足“同输入必出同字节”的确定性要求。

## 运行环境

- Windows 10/11，Python 3.11+（本机验证解释器为 Python 3.14，仅使用稳定标准库）。
- 无外部音频编解码库、无在线资源、无 GPU；WAV 全部为 PCM16 单声道，手工解析/写入。

## 目录结构

- `audiopipe/generate.py` 确定性素材生成器（固定种子，`random.Random` + 整数/三角函数）。
- `audiopipe/wavio.py` 严格 RIFF/WAVE/fmt/data 解析、校验与确定性写入。
- `audiopipe/resample.py` 线性相位多相（polyphase）加窗 sinc 重采样。
- `audiopipe/mixer.py` 时间轴对齐、同轨淡入覆盖、跨轨求和、增益与峰值限幅、降级报告。
- `audiopipe/checkpoint.py` 带版本号与输入/参数双指纹的原子检查点。
- `audiopipe/pipeline.py` 并发编排、资源边界、Ctrl+C、报告聚合与清单打包。
- `audiopipe/cli.py` 命令行入口。
- `tests/test_pipeline.py` 19 个验收测试（含强杀恢复子进程测试）。

## 命令行用法

```
python -m audiopipe generate <material_dir> [--seed N] [--clips N] [--languages en ja]
python -m audiopipe mix <material_dir> <out_dir> --target-rate 48000 \
       --headroom-db -1.0 --workers 4 --max-open-files 8 [--languages en]
python -m audiopipe verify-wav <file.wav>
python -m audiopipe resample <file.wav> --target-rate 48000
```

## 关键设计约定

- 毫秒→样本：全局统一 round-half-up（`floor(x+0.5)`），负偏移照常换算并在窗口边界裁剪。
- 重采样：有理比率 L/M 多相滤波器，逐分支单位 DC 增益；输出长度与理论值偏差 ≤ 1 采样；
  元数据记录源采样率、源采样数、转换后采样数与长度偏差。
- 同轨重叠：后到事件以 5ms 线性淡入覆盖前段（旧段同步淡出）。
- 跨轨：按轨道序号升序确定性求和，增益按 dB→线性应用；最后按可配置峰值余量硬限幅，
  并统计被限幅采样点数。
- 降级分类：
  - 可恢复（写静音占位并记录区间）：声明缺失、文件打不开、零时长、时长超窗截断、时长不足补静音、完全落在窗外。
  - 致命（终止该片段，不影响其他片段/语言）：WAV 结构/格式错误（非 PCM、非单声道、非 16bit、非法块长、data 字节不齐等）。
- 检查点：JSON 文件内含 `version`（当前 v2），发现旧版本文件直接报 `E_CHECKPOINT_VERSION`，
  不按旧行偏移读取；记录输入内容哈希 + 参数指纹，任一变化即重算；原子 `os.replace` 落盘。
- 打包：每语言输出 WAV + 对齐元数据（每事件最终样本起止、实际增益、是否占位/静音），
  清单 `manifest.json` 记录每个产物的 sha256、字节数（不含检查点目录）。
- 并发：`ThreadPoolExecutor` 限 worker 数，`BoundedSemaphore` 限同时打开文件句柄数；
  单片段失败隔离。Ctrl+C 落盘检查点并以业务码 130 + `E_INTERRUPTED` JSON 退出；
  强杀不承诺退出报告，仅靠已有检查点恢复。

## 业务/退出码

| 码 | 含义 |
|----|------|
| 0 | 成功 |
| 2 | WAV/检查点校验失败（JSON 写 stderr） |
| 3 | 至少一个片段致命失败，其余片段已完成 |
| 130 | 收到 Ctrl/Cmd 中断，检查点已落盘 |
| 64 | 命令行用法错误 |
| 70 | 未预期内部错误 |

## 实测验证结果

`python -m unittest tests.test_pipeline` → `Ran 25 tests ... OK`，覆盖：

1. 同种子重新生成的同名文件 sha256 完全一致；不同种子内容不同；语言过滤生效。
2. 严格 WAV 校验：坏签名、非 PCM、非 16bit、非法 RIFF/块长度、块越界、奇数 data 字节、fact 采样数不符均拒绝；零长度 data 合法；CLI 非零退出且 stderr 为 JSON 业务码。
3. 重采样长度偏差 ≤ 1 采样；DC 增益误差 ~1e-12；1kHz 音调 44.1k→48k 频率保持；重复调用结果逐点相同。
4. 毫秒舍入全局一致；dB 增益与峰值余量限幅正确，限幅采样被计数。
5. 降级报告区分可恢复与致命（含强制静音、同轨占位覆盖、完全出窗）；损坏单个 WAV 仅该片段失败，其他语言产物正常。
6. 检查点版本不匹配报错；指纹不一致触发重算，一致则复用。
7. 清单哈希可校验完整性；重复打包及不同 worker/句柄上限下全目录产物字节一致。
8. 子进程 SIGTERM 强杀后检查点存在，重启复用并补全全部片段；SIGINT 返回 130 且输出 `E_INTERRUPTED`。
