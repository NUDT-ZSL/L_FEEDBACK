# 实测产物

本目录来自 `sample/scene.json` + `sample/noise.raw` 的最终本地运行。

- `render.png`：主输出，SHA-256
  `4b4a5e6082ddb6b4061aed6e98d1913d3b913e356cab5559ec5ad88705927c62`。
- `reference.png`：固定参考帧，SHA-256
  `138e6f97c3d49d47c63c9dcd5ec77b74139ea5093a30bd57bd8b97e21533d1cf`。
- `stats.json`：采样数、阴影/帧耗时、双帧哈希、参考亮度误差。
- `sample-run.log`：样本运行输出，同时显示上次恢复帧的启动提示。
- `acceptance.log`：Python 验收脚本通过记录。
- `ctest.log`：CTest 通过记录。
- `recovery/`：最近完整主帧的固定命名 raw/metadata/ready 文件。

时间字段会随机器负载变化；PNG 和像素哈希不参与时间变化。
