# 实际验收记录

验证日期：2026-11-17（Asia/Shanghai）

环境：Linux x86_64，CPython 3.14.4，NumPy 2.5.5，Pillow 13.0.0，纯 CPU。

兼容性复检：CPython 3.8.20、NumPy 1.24.4、Pillow 10.4.0 下同样 `Ran 15 tests ... OK`。

## 执行命令

```bash
python -m unittest discover -v
python verify_acceptance.py \
  --workdir artifacts/acceptance \
  --report docs/acceptance_report.json
```

单元/集成测试结果：`Ran 15 tests ... OK`。

验收脚本结果：`passed: true`，`failures: []`，`temporary_files: []`。

## 确定性样本

- 验收样本由 `soft_renderer.sample_scene` 生成，尺寸 96×72。
- 两次生成 SHA-256 均为：
  `e277d6c845ee8ceca7f8da281bab5c68056f1d5fc31a9454cd41cbe88629e4c1`
- 项目内样例 `samples/indoor.json`（320×240）文件 SHA-256：
  `db742a1cf358cbdd617c298413a28c54f5c4b339efaf84e617acb4867d870ffd`

## 块尺寸一致性

low 预设、96×72，块尺寸 1/2/4/8 全部得到：

- PFM 文件 SHA-256：
  `4e13c4765fe1f8aef699d1ae8392c2f7ab35657f266e21ee9df9cdf80f91f4c4`
- 报告像素 SHA-256：
  `dd8a4587c121f5c67737ddb25407de19aa9dd5052df8c1756e3f4f8c32208328`

单次耗时（仅作本机记录，允许波动）：

| 块尺寸 | 总耗时 |
| ---: | ---: |
| 1 | 4277.8 ms |
| 2 | 1486.5 ms |
| 4 | 784.8 ms |
| 8 | 546.0 ms |

`--no-tiling` 单块渲染也得到同一 PFM SHA-256：
`4e13c4765fe1f8aef699d1ae8392c2f7ab35657f266e21ee9df9cdf80f91f4c4`。

## 分层合成

medium 预设（阴影 256、PCF 4、SSAA 2）、块尺寸 4：

- 整体帧与 `composite` 子命令输出 PFM 字节完全一致。
- 两边像素哈希均为：
  `bc0bab980e7a7d60aeac6743dfecd600810e773e867d0f8b30d6468deb5ff1eb`

## 阴影分类

medium 报告中的片元样本统计：

| 光源 | 全受光 | 半影 | 全遮挡 |
| --- | ---: | ---: | ---: |
| 点光源 0 | 28729 | 492 | 1902 |
| 聚光灯 1 | 18317 | 541 | 2848 |

平坦地面自阴影回归检查通过；无阴影纹理条带或自阴影痤疮。

## 质量预设与哈希

32×24 验收场景，block size 4：

| 预设 | 阴影分辨率 | PCF | SSAA | 像素哈希 | 本机耗时 |
| --- | ---: | ---: | ---: | --- | ---: |
| low #1 | 128 | 1 | 1 | `ee65cd20...5a1121` | 415.9 ms |
| low #2 | 128 | 1 | 1 | `ee65cd20...5a1121` | 405.9 ms |
| medium | 256 | 4 | 2 | `46cfa591...b40e369` | 666.3 ms |
| high | 512 | 9 | 3 | `cdb725a0...dde814` | 1205.8 ms |

两次 low 预设像素哈希完全相同，耗时不同，符合“像素哈希稳定、耗时允许波动”的验收要求。完整哈希见 `docs/acceptance_report.json`。

## HDR/PNG 与异常

- PNG 像素与“读取 PFM 后执行固定 ACES + 2.2 gamma”的逐像素结果完全一致。
- 损坏 JSON 返回非零，stderr 含 `invalid JSON`。
- 块尺寸 0 返回非零，stderr 含 `unsupported block size`。
- 输出目录不可写时返回非零，未生成功 PFM，未留下 `.tmp-*` 或 `.backup` 文件。

## 样例预览

`samples/indoor_preview.png` 使用以下命令生成：

```bash
python -m soft_renderer render samples/indoor.json \
  artifacts/sample_medium.pfm \
  --preset medium \
  --block-size 8 \
  --png samples/indoor_preview.png \
  --exposure 0.55
```

HDR 范围为 `[8.515576e-06, 138.4014]`，未做 HDR 截断。
