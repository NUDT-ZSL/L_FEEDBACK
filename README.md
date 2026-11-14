# 离线多轨混流处理器（audiomixer）

确定性、离线的多轨音频混流工具：素材生成、时间轴对齐、重采样、等功率声像、
淡入淡出/交叉淡化、BS.1770-4 响度标准化、自校验混流报告与批量处理。

## 环境准备

- Python ≥ 3.10（实测 3.14）
- NumPy（`pip install -r requirements.txt`）
- FFmpeg（可选但建议；提供 SoX 高质量重采样与 ebur128 交叉验证；缺失时自动使用
  内置 NumPy 带限重采样）

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 快速开始

```bash
# 单场景：生成素材 -> 混流 -> 响度标准化 -> WAV + 报告
python -m audiomixer run examples/scene_dialogue.json

# 用报告重新校验输出 WAV（哈希/帧数/响度/峰值）
python -m audiomixer verify examples/out/dialogue_scene.report.json

# 批量处理（含成功与失败场景，输出汇总报告，失败返回退出码 2）
python -m audiomixer batch examples/batch_scenes.json

# 全量单元测试（44 个，仅依赖标准库 + NumPy）
python -m unittest discover -s tests
```

## 文档

- 架构、算法、报告字段、错误码与 FFmpeg 响度交叉验证：`docs/TECHNICAL_NOTES.md`
- 场景配置示例：`examples/scene_dialogue.json`、`examples/scene_peak_fail.json`
- 批量配置示例（成功 + `E_PEAK_LIMIT` + `E_MISSING_ASSET`）：`examples/batch_scenes.json`

## 业务错误码

`E_MISSING_ASSET`、`E_UNSUPPORTED_FORMAT`、`E_OUT_OF_RANGE`、`E_PEAK_LIMIT`、
`E_INVALID_CONFIG`。失败时以退出码 2 结束、写 JSON 失败报告，且不保留任何输出 WAV。
