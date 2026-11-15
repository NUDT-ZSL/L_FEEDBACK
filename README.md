# skeletal_baker

离线骨骼动画烘焙工具：将程序确定性生成的双骨骼胶囊体网格烘焙为逐帧
顶点/法线缓冲与正交预览 PNG，业务错误以 JSON Lines 记录且不中断批处理。

## 环境

- Python 3.10 / 3.11，依赖 `numpy`（1.24–1.26）与 `pillow`（10.x）
- 本仓库使用 uv 创建 `.venv`（CPython 3.11.18, numpy 1.26.4, pillow 10.4.0）

## 用法

```bash
python -m skeletal_baker --output-dir out --frames 48 --seed 86 --fps 24
```

- 参数非法时退出码为 2，并在 stderr 打印用法说明（argparse 行为）。
- 正常完成退出码为 0，即使过程中产生 `E_*` 业务错误。

## 输出

输出目录包含：

- `frame_XXXX.vert.bin` / `frame_XXXX.norm.bin`：裁剪后三角形汤的 NDC
  位置与法线，float32 小端，无文件头/时间戳；全裁剪帧保留空文件。
- `frame_XXXX.png`：64px 宽视口（高度由投影半宽半高比确定）的平面着色
  预览，单方向光 + 逐像素线性深度测试。
- `overview.png`：全部帧缩略图总览，每行 10 帧，行数按总帧数自动计算。
- `errors.log`：JSON Lines，每行含 `frame` / `code` / `message`。
- `stats.jsonl`：每帧裁剪统计（输入/通过/裁剪/剔除/输出三角形数）。

所有文件先写同目录临时文件再原子重命名，避免中断产生半截文件。

## 业务错误码

- `E_NON_FINITE_MATRIX`：骨骼世界矩阵含非有限元素，跳过该帧。
- `E_NON_INVERTIBLE_MATRIX`：骨骼矩阵 3x3 部分不可逆，跳过该帧。
- `E_DEGENERATE_TRIANGLE`：面积 < 1e-8 的退化三角形，跳过该三角形，
  单帧最多记录 10 条。
- `E_FRAME_FULLY_CULLED`：该帧所有三角形被完全裁剪，仍写出空缓冲与
  空白 PNG。

## 实现要点

- `skeletal_baker/mesh.py`：Y 轴胶囊体网格；蒙皮权重按到双骨骼轴线
  （线段）距离反比并归一化。
- `skeletal_baker/animator.py`：`SkeletalAnimator`，固定帧率采样根骨骼
  平移与子骨骼绕本地 Z 轴旋转，正向运动学输出 4x4 世界矩阵并做有限性
  校验；种子驱动动画相位，保证确定性。
- `skeletal_baker/skinning.py`：线性混合蒙皮；法线用各骨骼 3x3 逆转置
  按同权重混合并归一化。
- `skeletal_baker/normals.py`：蒙皮后位置缓冲上按面法线面积加权累加
  重算逐顶点法线。
- `skeletal_baker/clip.py`：固定正交视锥，NDC 空间 Sutherland-Hodgman
  逐边裁剪 + 扇形重三角化 + 每帧统计。
- `skeletal_baker/raster.py`：平面着色光栅化与总览图拼接。
- `skeletal_baker/pipeline.py`：逐帧调度、原子写出。

## 测试与验证

```bash
python tests/test_pipeline.py   # 8 个用例全部通过
```

实际验证结果（本机，CPython 3.11.18 / numpy 1.26.4 / pillow 10.4.0）：

- `--frames 48 --seed 86 --fps 24` 连续运行两次，`diff -r` 确认全部
  147 个输出文件（含 .vert.bin / .norm.bin / errors.log / PNG）逐字节
  一致；总览 PNG 逐像素一致。
- 第 4–7 帧因根骨骼平移出视口触发 `E_FRAME_FULLY_CULLED`，对应
  `frame_0004..0007.vert.bin` 为 0 字节空文件，批处理未中断，退出码 0。
- 非法参数（`--frames 0`、`--fps -1`、负数 seed、缺少 --output-dir 等）
  退出码均为 2，stderr 打印 usage。
