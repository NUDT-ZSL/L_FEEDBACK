# L_FEEDBACK — Replay Master Exporter

确定性、可断点续渲的竞技游戏赛局解说母带导出器。纯 Python 标准库实现，
输出 48kHz / 16-bit / 立体声 PCM 母带，支持交叉淡化、解说闪避、响度归一化、
硬限幅、版本化计划快照、原子检查点与回放校验报告。

## 快速开始

```bash
# 生成确定性样本工程（短音频，每段约 1 秒）
python3 -m replay_exporter.cli generate-samples demo_work/project --segments 4

# 渲染母带（重复执行即从检查点续渲）
python3 -m replay_exporter.cli export \
    demo_work/project/timeline.json demo_work/out

# 生成回放校验 JSON 报告
python3 -m replay_exporter.cli validate demo_work/out

# 单元测试（34 个）
python3 -m unittest discover -s tests -t .

# 端到端验收：真实子进程强制终止 → 重启续渲 → 逐字节比对 → 校验
python3 scripts/acceptance_demo.py
```

## 输出目录产物

| 文件 | 含义 |
| --- | --- |
| `plan_snapshot.json` | 版本号、规范化计划、每块输入 SHA-256、废弃块清单 |
| `render_checkpoint.json` | 已完成块、母带字节长度、每块输出 PCM 哈希（原子写入） |
| `master.wav` | 48kHz/16-bit 立体声母带，头部按总时长预分配以便 seek 续写 |
| `gain_report.json` | 每段测量峰值/RMS、响度与峰值增益、实际增益、限幅状态 |
| `validation_report.json` | 段级起点/闪避/参考一致性与交叉淡化能量连续校验 |
| `error_report.json` | 失败时的结构化业务错误码（含段号、毫秒偏移） |
| `export.lock` | 运行期导出锁，正常结束自动删除 |

详细架构、错误码表与验证结果见 `docs/TECHNICAL_NOTES.md`。
