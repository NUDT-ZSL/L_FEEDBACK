# RhythmForge

离线、确定性的节奏游戏音轨构建工具：把谱面 JSON 与本地 PCM WAV 分层
素材转换为成品混音、对齐分轨、采样帧精确同步标记与响度报告，支持位级
增量重建与输出目录并发互斥。

## 快速开始

```bash
pip install -r requirements.txt
export PYTHONPATH=src            # Windows PowerShell: $env:PYTHONPATH="src"
python -m rhythmforge generate --samples-dir samples --chart samples/chart.json
python -m rhythmforge build    --chart samples/chart.json --samples-dir samples --output output
python -m rhythmforge verify   --chart samples/chart.json --samples-dir samples --output output
python -m pytest tests
```

## 产物

- `output/master.wav`：启用分层的成品混音（24-bit PCM，48 kHz 立体声）
- `output/stems/<layer>.wav`：每个分层的对齐分轨，与成品同增益
- `output/markers.json`：段落/BPM/拍号同步标记（含精确采样帧）
- `output/normalization.json`：响度与峰值保护归一化报告
- `output/manifest.json`：内容寻址清单，支撑增量重建
- `output/cache/section_*.npy`：按段落缓存的分层渲染结果

## 文档

架构、算法、退出码契约与实测结果见 `docs/TECHNICAL.md`。
