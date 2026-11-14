# 离线 PBR 纹理烘焙工具

这是一个从零实现的确定性离线烘焙工具。它读取代码内部生成的低模、高模和逐顶点材质样本，完成纹理空间光栅化、切线空间最近命中光线投射、PBR 通道输出、接缝膨胀、法线约定校验，并用纯 NumPy 在离屏内存中渲染一张可重光照预览图。

工具不需要 Docker、图形驱动、GPU 或离屏图形上下文。纹理 PNG 使用内置的固定编码器写出，不依赖 Pillow；Pillow 仅用于测试中读取 PNG 做独立核验。

## Windows 安装与运行

目标环境为 Windows 10/11、Python 3.10 或 3.11。

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m bake_tool --out out --resolution 128
```

DirectX 约定、自定义膨胀半径和错误场景示例：

```powershell
python -m bake_tool --out out_dx --resolution 128 --normal-convention directx --dilation-radius 3
python -m bake_tool --out out_error --scenario uv_out_of_range
```

也可以安装为控制台命令：

```powershell
python -m pip install .
offline-pbr-bake --out out --resolution 128
```

## 输出约定

所有文件位于 `--out` 指定目录。成功运行提交以下文件：

| 文件 | 内容 |
| --- | --- |
| `normal.png` | 低模 UV 空间中的切线空间法线，RGB8 |
| `albedo.png` | 最近高模命中点的反照率，RGB8 |
| `roughness.png` | 粗糙度，灰度8位 |
| `metallic.png` | 金属度，灰度8位 |
| `tangent.png` / `bitangent.png` | 逐像素世界空间 T/B 向量，RGB8 编码 `[-1,1]` |
| `coverage_mask.png` | 光栅化原始覆盖掩码，0/255 |
| `coverage_mask_dilated.png` | 膨胀后的覆盖掩码 |
| `triangle_index.png` | 每个覆盖像素唯一归属的低模三角形 ID + 1，灰度16位，0 为背景 |
| `dilation_difference.png` | 255=原始有效，128=膨胀新增，0=仍无覆盖 |
| `normal_consistency_mask.png` | 目标 Y 约定下的法线不一致像素，255=不一致 |
| `normal_consistency_alternate_mask.png` | 交换 Y 约定后检测到的不一致像素 |
| `preview.png` | 纯 NumPy 软件光栅化 + PBR 着色预览，尺寸与烘焙分辨率一致 |
| `bake_report.json` | 确定性统计、参数快照、TBN/光栅/射线/膨胀/预览统计和纹理 SHA-256 |
| `runtime_profile.json` | 非确定性物理墙钟耗时侧车文件 |

失败运行不提交任何纹理，只写 `error_report.json`。

## 关键约定

- UV 原点为左下角。像素中心规则为 `u=(x+0.5)/W`、`v=1-(y+0.5)/H`，PNG 行号自上而下。
- 光栅化按像素中心采样；三角形包围盒先裁剪到纹理边界；几何面法线与插值顶点法线相反时执行背面剔除。
- UV 重叠采用确定性“首个三角形拥有像素”策略，保证每个像素只属于一个三角形。
- TBN 使用归一化几何法线、UV 导数切线及 `B = cross(N, T)`。该构造在任何情况下都保持右手系；镜像 UV 通过切线方向反转表达，UV 退化顶点使用基于法线的稳定回退轴。
- OpenGL 法线输出保存 `Y`，DirectX 输出自动翻转绿色通道。预览使用同一 TBN 和同一约定解码。
- `normal_dot_threshold=0.98` 时，所选约定的 8 位量化法线应几乎全部通过；报告同时计算交换 Y 约定后的不一致比例，用于检测方向约定不一致。
- 膨胀使用多源逐环邻域填充，只写入原始无效像素，绝不覆盖任何岛屿的原始有效像素。
- 光线从低模表面沿插值 TBN 的 `(0,0,1)` 方向发出，即沿低模几何法线，偏移 `1e-5`；在 BVH 中取最近高模交点。未命中时回退低模材质并计数。

## 确定性说明

`bake_report.json` 与所有 PNG 在相同输入和参数下字节级一致。物理墙钟时间天然不确定，因此：

- `bake_report.json` 的 `timing.total_elapsed_ms` 是固定工作量成本模型得到的确定性时间账目；
- 真实运行耗时写入 `runtime_profile.json`，不参与确定性验收。

该拆分同时满足“可复现字节输出”和“记录实际总耗时”，但实际墙钟不会破坏统计报告的确定性。

## 错误码

| 业务码 | 退出码 | 场景 |
| --- | --- | --- |
| `E_UV_OUT_OF_RANGE` | 2 | 低模 UV 超出 `[0,1]` |
| `E_HIGH_MESH_MISSING` | 2 | 高模为空或无三角形 |
| `E_RESOLUTION_LIMIT` | 2 | 分辨率超过 `max_resolution` |
| `E_MISS_RATE_EXCEEDED` | 2 | 射线未命中率超过阈值 |
| `E_COVERAGE_EMPTY` | 2 | 光栅化没有覆盖像素 |
| `E_INVALID_ARGUMENT` | 2 | 参数非法 |
| `E_TRIANGLE_INDEX_LIMIT` | 2 | 三角形 ID 超出 16 位诊断图范围 |
| `E_INTERNAL_ERROR` | 1 | 未预期的内部异常 |

错误报告包含 `business_code`、整数 `exit_code`、上下文、参数快照和恢复建议。所有成功纹理先写入内存，完成后再通过临时目录事务提交；业务错误路径不会留下部分纹理。

## 实现结构

```text
bake_tool/
  samples.py     确定性双 UV 岛屿、低/高模与材质样本
  mesh.py        数据校验与 UV 越界错误
  tbn.py         逐顶点 TBN、镜像/退化处理、逐像素正交化
  rasterizer.py  纹理空间像素中心光栅化、裁剪、背面剔除、所有权
  raycast.py     BVH 与 Moller-Trumbore 最近交点
  dilation.py    不覆盖有效像素的多源邻域膨胀
  pipeline.py    通道合成、法线校验、统计、事务输出
  preview.py     NumPy 正交软件光栅化、PBR 着色和色调映射
  pngio.py       固定压缩参数的确定性 PNG 8/16 位编码器
  cli.py         JSON 结果、业务码和进程退出码
tests/
  test_core.py      裁剪、背面剔除、TBN 回退、膨胀边界
  test_pipeline.py  通道/预览、Y 约定、确定性、错误链路
```

## 验证命令与实际结果

```powershell
python -m unittest discover -s tests -v
python -m bake_tool --out out --resolution 128
```

在当前验证环境中，11 项测试全部通过。128² 默认样本的实际结果记录在 `out/bake_report.json`；样例包含 384 个低模三角形、2,560 个高模三角形和两个 UV 岛屿，其中一个岛屿使用镜像 U 展开。运行两次相同参数时，除 `runtime_profile.json` 外，所有文件和报告 SHA-256 完全一致。

参考实测值（128²，OpenGL，膨胀半径 2）：

- 覆盖像素：11,424；高模射线命中：11,424；回退像素：0。
- 选定约定法线不一致为 0；交换 DirectX 约定后产生非零不一致掩码。
- 接缝像素：644；膨胀新增像素：1,312；原始有效像素覆盖写入：0。
- 预览尺寸与烘焙分辨率一致，渲染完全由 NumPy 软件光栅化和着色完成。
