# 赛事回放多语种音频离线打包器

读取事件清单 JSON 与本地合成的 `{语音键}_{语种}.wav` 配音片段，按语种把配音放到
48kHz 整数采样时间轴，做同信道冲突消解、等功率响度归一化、静音填充与确定性交叉
淡化，输出 48kHz/16bit 单声道 WAV、CUE 时间戳索引与 JSON 报告，并通过内容寻址快照
支持中断续跑与受影响块重算。

## 快速开始

```bash
python -m pip install -r requirements.txt

# 生成确定性样本与清单
python -m audiopack.cli generate --assets-dir demo/assets --manifest demo/manifest.json

# 打包
python -m audiopack.cli pack \
  --manifest demo/manifest.json --assets-dir demo/assets \
  --work-dir demo/work --output-dir demo/out \
  --max-workers 3 --target-loudness-db -23

# 一键演示：中断 -> 续跑 -> 全复用
./tools/run_demo.sh

# 测试
python -m unittest discover -s tests
```

## 文档

- 设计与模块说明：`docs/TECHNICAL.md`
- 实测验证记录：`docs/VERIFICATION.md`

## 主要能力

- 清单全量校验，业务码一次性收集（`E_DUP_ID`、`E_BAD_TIME`、`E_MISSING_ASSET`、
  `E_BAD_LANG` 等），不遇错即崩；
- 整数采样时间轴，片段时长取源文件帧数，无浮点累计漂移；
- 高优先级锚定、低优先级整体后移、超最大延迟丢弃，且不连锁改动高优先级；
- 100ms 窗 RMS 等功率归一化，增益钳制 `[-20, +12] dB`，超标区间入报告；
- 5ms 等功率交叉淡化、整毫秒时长取整、WAV 帧头/帧数与索引逐项对账；
- 每语种快照带配置/清单/片段/轨内容哈希，恢复时按版本、清单、片段、调度、轨哈希
  识别受影响块，复用块偏移按新总时长重新校验；
- 线程池并发受限加载，支持缺失/损坏/瞬态失败注入、重试与终端业务码，失败不产生
  半写快照。
