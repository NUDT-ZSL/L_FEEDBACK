# Actual Verification

验证日期：2026-11-15。环境：WSL2 Linux、Python 3.11.18、NumPy 1.24.4、
opencv-python 4.8.1。没有使用 Docker、外部图像或网络资产。

## 单元与集成测试

命令：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

结果：`Ran 22 tests in 0.618s`，`OK`。

覆盖内容：

- PCHIP 端点精确、区间无过冲、短轨道端点复用和警告。
- 乱序/越界关键帧的帧索引异常、序列结束后延迟校验。
- 深度线性映射、无效/近零/超范围深度兜底、CoC 对焦零点与半径截断。
- 圆核归一化与对称性、多边形核归一化、零半径核。
- 背景不可渗入聚焦前景、前景散焦覆盖背景、轮数不反转遮挡。
- 单切片与多切片结果、报告字段、失败部分统计、重复渲染确定性。

## 完整 CLI 验收

命令：

```bash
.venv/bin/python scripts/acceptance.py --root validation/acceptance \
  --width 240 --height 135 --frames 8 --fps 12 --seed 12345
```

结果：`status: passed`，全部 10 项检查为 `true`：

```text
png_sequence_count                   true
png_hashes_byte_identical            true
report_equal_excluding_timing        true
slice_consistency_passed             true
circle_kernel_normalized             true
polygon_kernel_normalized            true
failure_exit_nonzero                 true
partial_report_preserved             true
partial_and_conventional_match       true
rendered_frames_preserved_on_failure true
```

关键测量：

- 8 帧样本，240 x 135，fps=12，seed=12345。
- 多切片（8 层）与单次全图散射：最大 8 位绝对误差 `1.0`，阈值 `2.0`；
  平均绝对误差 `3.858024691358025e-06`。
- 圆形与 6 叶多边形核均为 `21 x 21`，权重和均为 `1.0`。
- 两次运行生成的 8 个 PNG 文件 SHA-256 全部相同。
- 两次 `report.json` 在将所有 `render_time_ms` 归一化后完全相同。
- 故意把焦点轨道改为乱序后，CLI 退出码为 `2`，异常帧索引为 `2`；
  已生成 2 个 PNG，`report.partial.json` 与失败的 `report.json` 均保留且内容相同。

详细机器可读结果：

- `validation/acceptance/acceptance_summary.json`
- `validation/acceptance/slice_verification.json`
- `validation/acceptance/failure_output/report.partial.json`
- `validation/acceptance/render_a/report.json`

## 样本渲染观察

示例序列包含远墙、中景椭圆和移动的近景角色形状。焦点从近处移动到中远景时：

- 近景从明显散焦逐渐变清晰，远墙/中景出现随焦点变化的大半径虚化。
- 近景边缘保持其颜色，没有被背景颜色渗入。
- 近景散焦时，其颜色可以跨深度边界覆盖背景。

## 平台说明

本次任务提供的运行环境为 WSL2/Linux，未在原生 Windows 机器上重复执行。实现避开了
shell 专用命令和符号链接，依赖均提供 Windows wheel，路径和文件操作使用
`pathlib`/`os.replace`/OpenCV 等跨平台接口；验证脚本本身也可在 Windows 上以
`python scripts/acceptance.py` 运行。
