# 裂变晶体洞穴 CPU 光栅化渲染器

纯 Python 3.11+ / NumPy 实现的确定性 CPU 光栅化器，用于游戏内「裂变晶体洞穴」
场景的离屏预览。无 GPU、无外部素材、无网络依赖；场景、关键帧与异常注入样本均由
`crystal_renderer/scene_gen.py` 按种子确定性生成。

## 运行方式

```bash
pip install numpy          # 唯一第三方依赖
python -m crystal_renderer.cli gen --out scenes/demo.json --seed 7 --frames 40
python -m crystal_renderer.cli gen --out scenes/anomaly.json --seed 11 --frames 12 --anomaly
python -m crystal_renderer.cli run --config scenes/demo.json --out runs/incr --mode incremental
python -m crystal_renderer.cli run --config scenes/demo.json --out runs/full --mode full
python -m crystal_renderer.cli verify --config scenes/demo.json --out runs/verify
python -m unittest discover -s tests        # 8 项验收测试
```

`run` 输出：逐帧 `frames/frame_XXXX.npy`（RGBA uint8）与 `.ppm` 预览、
`render_report.json`（着色计数、脏区、逐帧绘制序列、错误码）、
`snapshots/`（版本化快照）、`recovery_report.json`（中断/恢复事件）。

## 架构

- `config.py` — JSON 装载与逐晶体校验；非法晶体以业务错误码（E101–E106）写入
  stderr 与报告 JSON 后被剔除，其余晶体继续渲染，整帧不崩溃。
- `scene.py` — 整数步进的关键帧求值（`set` 语义，按 step 排序累积应用），
  相机变换后得到屏幕空间多边形；折射半径 `R = strength * (ior - 1)`。
- `raster.py` — 核心光栅化。画家算法基底序（layer, id）+ 逐像素深度修正：
  每块晶体带深度平面 `z = layer*2 + gx*x + gy*y`，每像素按 (z, id) 计算 rank，
  支持相互穿插；按 rank 从远到近做预乘 Alpha 合成。背面法线折射采样：片元
  相对多边形形心的归一化透镜法向乘以折射半径得到采样偏移，从「该晶体背后
  已合成内容」中采样，越界按边缘钳制。
- `renderer.py` — 全量/增量两条管线共用同一 `render_region`，所有逐像素
  计算均为 elementwise，因此限制写出区域不会改变区域内像素值。
- `snapshot.py` — 快照保存 float64 累加缓冲（保证恢复后逐位一致）+
  版本号 + SHA-256 校验；版本或校验不匹配则整体丢弃并从第 0 帧重算。
- `cli.py` — gen/run/verify 子命令；SIGINT/SIGTERM 可处理中断（写恢复报告
  并保存进度快照，退出码 3），SIGKILL 强杀后依赖最近一次完整快照恢复。

## 脏区最小化与一致性论证

单块晶体在帧 t 发生变化时，受影响像素集合为：

1. 变化晶体在 t-1 与 t 两态的轮廓包围盒，各按自身折射半径外扩（采样扩散区，
   覆盖其片元的全部折射采样坐标）；
2. 任何未变化晶体，若其外扩包围盒与当前脏区相交，则其片元可能采样到已变化
   像素，必须整体纳入脏区（同样按外扩包围盒并入）——迭代至不动点，处理
   折射链式传播。

增量重绘时，脏区内的累加缓冲先重置为背景再按同一 (rank, crystal) 顺序重算；
由于任何覆盖脏区像素的晶体其外扩包围盒必完整包含于脏区，所有折射采样坐标
均落在脏区内、读取的是新帧状态，脏区外像素逐字回复用上一帧。因此增量结果
与全量重绘逐像素一致（`verify` 子命令逐帧断言）。

## 错误码

| 代码 | 含义 |
| ---- | ---- |
| E101 | 退化多边形（顶点数 < 3） |
| E102 | 零面积晶体 |
| E103 | 非法折射率（需 1.0 < ior ≤ 3.0） |
| E104 | 非法颜色分量 |
| E105 | 关键帧越界或非法（剔除该关键帧，晶体保留） |
| E106 | 非法变换参数 |
| E107 | 顶层配置非法（致命） |
| E108 | 快照版本不匹配，丢弃重算 |
| E109 | 快照校验和不匹配，丢弃重算 |

## 实际验证结果（本机，Python 3.14 + NumPy 2.5.4）

- `verify`（seed=7，40 帧，160x120，6 晶体）：增量与全量 **40/40 帧逐像素一致**；
  全量着色 21965 片元，增量着色 1531 片元（约 7%）；13 个单晶体变化帧全部满足
  增量着色次数严格低于全量。
- 确定性：同配置两次运行全部帧字节一致；晶体输入列表乱序后帧序列与
  绘制序列完全一致。
- 快照：篡改缓冲字节触发 E109、篡改版本号触发 E108，均从第 0 帧重算且
  最终帧序列与未中断参考运行逐像素一致。
- 中断恢复：`kill -9` 强杀后由最近完整快照恢复、SIGTERM 优雅中断（退出码 3、
  恢复报告记录 interrupted 事件与进度快照）后恢复，两者恢复后的帧序列均与
  未中断运行逐像素一致，恢复报告标明来源（snapshot/config）。
- 异常注入：E101/E102/E103/E105 同时出现在 stderr 与 render_report.json，
  非法晶体被剔除、越界关键帧被剔除而晶体保留，12 帧全部正常输出。
- `python -m unittest discover -s tests`：8 项测试全部通过。
