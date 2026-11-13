# 赛道闸门程序化渲染组件 — 技术说明

纯 CPU 确定性软件光栅化管线，输入为程序化场景描述与相机视锥参数，输出
RGBA/深度缓冲快照与按视锥水平分块（tile）的绘制覆盖清单。无 GPU、无外部
资产、无网络依赖。

## 运行环境

- Python 3.11+（开发验证机为 Python 3.14）
- NumPy（任务书要求 1.26；本机 Python 3.14 无法构建 NumPy 1.26，实际使用
  NumPy 2.5.4，代码仅使用 1.26/2.x 兼容 API，Windows + NumPy 1.26 可直接运行）
- Windows 命令行：`python main.py run --seed 7 --segments 6 --out artifacts\run`

## 模块结构

| 文件 | 职责 |
| --- | --- |
| `trackgate/scene.py` | 固定种子确定性场景生成器；冻结 dataclass，共享材质表不可变 |
| `trackgate/camera.py` | 透视相机、视锥 AABB 判定、相机版本标识 |
| `trackgate/arch.py` | 拱形框架三角形细分，顶点携带归一化弧长参数 s |
| `trackgate/raster.py` | 软件光栅化：RGBA + float32 深度 + 段 id 缓冲 |
| `trackgate/tiles.py` | 水平 tile 划分、覆盖清单、清单↔像素缓冲交叉核对 |
| `trackgate/incremental.py` | 版本驱动的增量 tile 重绘与磁盘缓存作废 |
| `trackgate/pipeline.py` | 变更序列编排、全量重算一致性校验、快照输出 |
| `main.py` | CLI：`run` / `edges`，整数退出码 |

## 关键设计

**确定性场景生成（需求 1）**：`random.Random(seed)` 依次生成每段的平移、
绕竖直轴（Y）旋转、拱高、宽度、厚度、材质编号与条纹参数（条纹数、占空
比、颜色）。规范 JSON（排序键、定精度舍入）的 SHA-1 前 16 位作为场景版本
号。材质表为共享只读元组，段之间只共享材质编号；所有参数对象均为冻结
dataclass，不存在共享可变状态。

**相机与视锥（需求 2）**：位置 + 偏航/俯仰 + 竖直 FOV + 宽高比构造透视相
机；`aabb_visible` 对包围盒 8 角点做 6 平面保守测试——仅当所有角点在同一
平面外侧才剔除，部分落入视锥的段必然保留。相机版本号为参数散列。

**软件光栅化（需求 3）**：拱架按 28 段椭圆半拱细分为前/后面、内/外圈与端
盖三角形；近平面/远平面 Sutherland–Hodgman 裁剪（同步插值弧长属性）；屏
幕空间重心坐标 + 透视校正插值；逐像素“最近 z 胜出”，可见性与绘制顺序无
关（有专门测试：交换段顺序渲染结果逐字节一致）。

**弧长条纹遮罩（需求 4）**：条纹参数化在拱中线累计弧长 s∈[0,1] 上，同一
角度站位上内/外/前/后四个顶点共享同一 s，条纹沿径向对齐，细长三角形与
轮廓边缘不会撕裂；像素中心经透视校正重心插值得到 s 后按
`frac(s*count) < duty` 判定条纹归属，相位与占空比对任意拱高/宽度稳定。

**分块清单（需求 5）**：图像按固定像素宽度水平分块（末块可更窄）。清单记
录每 tile 的 `{段编号: 像素数}` 与总像素；只登记最终真正留在缓冲里的像
素（由段 id 缓冲逐 tile 重计数得到），零像素段不出现、被写入段不漏记；
`cross_check` 用独立重计数与清单交叉核对。

**增量重绘（需求 6）**：场景/相机各自维护内容散列版本号。版本均未变 →
全部 tile 复用；任一版本变化 → 重算“上一帧触及 tile ∪ 新投影触及 tile”，
其余 tile 沿用上一版本结果。tile 内重绘与全量渲染逐字节一致，因为所有边
函数都在整图像素坐标系求值，裁剪只限制写出范围。磁盘缓存
（`cache/tile_cache.json`）以 (generation, scene_version, camera_version,
宽高, tile 宽) 为头部，任一字段不匹配即整体作废，绝不沿用旧文件行偏移伪
装命中。

**一致性校验（需求 7）**：每个变更步后，将增量帧与从头全量重算逐 tile 比
对（颜色/深度位级、id 缓冲、清单），不一致时返回结构化差异报告（tile 编
号、段编号、像素计数差），不静默覆盖。

**异常与边界（需求 8）**：

| 情形 | 结果 |
| --- | --- |
| 空场景 | `E_EMPTY_SCENE`，退出码 2 |
| 退化视场角（<1° 或 >179°） | `E_DEGENERATE_FOV`，退出码 2 |
| 非法 tile 宽度（≤0） | `E_BAD_TILE_WIDTH`，退出码 2 |
| tile 宽度不整除图像宽 | 确定的末块窄 tile，正常渲染 |
| 相机位于闸门段内部 | 确定的业务结果：近平面裁剪正常渲染，并在摘要中报告 `camera_inside_segments` |
| 成功路径 | 退出码 0，写出 color/depth `.npy`、`.ppm`、清单与 `summary.json` |

## 实际验证结果（本机，2026-11-14）

- `python3 main.py run --seed 7 --segments 6 --width 320 --height 180 --tile-width 64 --out artifacts/run`
  → 退出码 0；6 个变更步（含无变化全复用、纯相机移动、段条纹变更、相机进
  入闸门、共享材质换色 + 宽 FOV）增量产物与全量重算 **0 差异**；清单↔像
  素缓冲交叉核对 0 差异；缓存自命中成功、过期 generation 被拒绝。
- `python3 main.py run --segments 0` → 输出 `E_EMPTY_SCENE` JSON，退出码 2。
- `python3 main.py edges` → 6 项边界用例全部符合预期，退出码 0。
- `python3 -m unittest discover -s tests` → **20 个测试全部通过**（场景逐字
  节确定性、版本号、视锥部分包含、深度最近面与绘制顺序无关、tile 限定渲
  染与全量一致、弧长参数归一且径向对齐、条纹相位尺寸不变、清单篡改检
  测、增量逐步一致性、缓存作废、退出码）。

## 复现命令

```bat
python main.py run --seed 7 --segments 6 --width 320 --height 180 --tile-width 64 --out artifacts\run
python main.py edges
python -m unittest discover -s tests -v
```
