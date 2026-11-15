# 离线地形 LOD 与法线烘焙管线

这是一个从零实现的、固定种子驱动的离线地形资产生成器。它生成多级 LOD 网格、LOD 过渡缝合网格、切线空间法线贴图、材质混合预览图，以及带 SHA-256 校验和的结构化清单。

实现不读取任何外部美术资产。默认样本全部由高度场尺寸和整数种子确定生成。

实际执行记录见 `VERIFICATION.md`，可复现样例位于 `artifacts/terrain`。

## 快速运行

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt

python bake.py \
  --size 65 \
  --seed 20261116 \
  --lods 3 \
  --normal-resolution 512 \
  --preview-width 800 \
  --preview-height 600 \
  --output artifacts/terrain \
  --preview-backend auto
```

也可以使用模块入口：

```bash
python -m terrain_bake --size 65 --seed 20261116 --output artifacts/terrain
```

`--preview-backend auto` 会先尝试 OpenGL 3.3 离屏上下文；创建或使用失败时自动切换到 CPU 光栅器，并把尝试列表、最终后端和回退原因写入清单。`--preview-backend cpu` 用于强制验证无 GPU 路径，`opengl` 则表示 OpenGL 不可用时返回 `E_RENDER_FAILED`。

本次工作区实际使用 Python 3.14.4，并通过 `uv pip` 将 `numpy`、`Pillow`、`PyOpenGL`、`glfw` 安装到 `.venv`。

## 输出

成功运行后，输出目录包含：

| 文件 | 内容 |
| --- | --- |
| `lod_0.obj` | 基础网格 |
| `lod_1.obj` | 一次边缘折叠后的 LOD |
| `lod_2.obj` | 最低 LOD，预览渲染使用该级 |
| `transition_lod_0_1.obj` | LOD0/LOD1 邻接块及缝合三角形 |
| `transition_lod_1_2.obj` | LOD1/LOD2 邻接块及缝合三角形 |
| `normal_map.png` | 切线空间法线贴图 |
| `normal_map.json` | 法线坐标、切线基、有限差分步长元数据 |
| `material_preview.png` | 草地/岩石/雪地权重混合俯视预览 |
| `preview.png` | 最低 LOD + 法线贴图 + 材质颜色的 CPU/OpenGL 预览 |
| `heightfield.npy` | 实际使用的确定性高度场样本 |
| `material_weights.npy` | 逐顶点草地/岩石/雪地混合权重 |
| `bake_manifest.json` | 状态、种子、LOD 统计、路径、校验和与错误信息 |

清单 schema 见 `schema/bake_manifest.schema.json`。成功和失败路径都保留同一组顶层字段；失败时产物列表和校验和为空，`error` 为业务错误码。

## 统一坐标约定

所有模块共享 `terrain_bake/coordinates.py` 中的定义：

- 高度场数组按 `[row, col]` 存储。
- 世界坐标 `X = col / (cols - 1)`，`Z = row / (rows - 1)`，范围均为 `[0, 1]`。
- 世界高度 `Y = heightfield * 0.32`。
- 地面基础法线为 `+Y`。
- 切线基为 `T=+X`、`N=+Y`、`B=cross(N,T)=-Z`，因此 `T x B = N`。
- 法线图 RGB 解码后为 `normalize(T*r + B*g + N*b)`。

法线烘焙按输出像素采样高度场。步长固定为 `1 / (normal_resolution - 1)` 的世界 XZ 单位，中心差分使用相隔一个输出像素的左右/上下采样点，图像边缘自动夹取采样位置。网格顶点、材质斜率、法线梯度和预览 UV 都使用这一坐标系，避免跨模块错位。

## LOD 与缝合

1. 从规则三角网格出发，为每个顶点累积三角形平面二次误差矩阵。
2. 使用确定性的端点边缘折叠和优先队列选择候选边，折叠代价按网格包围盒对角线归一化。
3. 每次折叠前检查 link condition、边界类型、重复面和非退化条件；四个角点固定，边界可沿边折叠。
4. 每级阈值逐级放宽，且单级实际相对 QEM 误差不得超过该级阈值，最终也不得超过 `--max-error`。
5. 每级输出都检查单边界盘拓扑、每条边至多两个面、环绕方向一致、无非流形边和无零面积三角形。

相邻 LOD 过渡块沿 Z 端到端排列，并用一条窄过渡带连接。缝合三角网只引用已有顶点，不计算新的高度采样点。高细节侧顶点与原始 LOD 坐标逐位相等；验证会检查过渡网格流形性、单边界环和 T 型接缝计数。`transition_lod_*.obj` 是该验证组合网格。

## 错误码与原子输出

| 业务码 | CLI 退出码 | 触发条件 |
| --- | ---: | --- |
| `E_INVALID_ARGUMENT` | 2 | 参数、输出路径或后端选项非法 |
| `E_INVALID_HEIGHTFIELD` | 3 | 高度场尺寸小于 2，含 NaN/Inf，或无法生成有效权重 |
| `E_WEIGHT_MISMATCH` | 4 | 权重形状错误、负值、非有限值，或权重和偏离 1 超过容差 |
| `E_GEOMETRY_INVALID` | 5 | 无法在误差上限内折叠、非流形、退化三角形或缝合失败 |
| `E_RENDER_FAILED` | 6 | 显式 OpenGL 渲染失败或 CPU 渲染失败 |
| `E_IO_ERROR` | 7 | 原子提交或清单写入失败 |
| `E_INTERNAL_ERROR` | 8 | 未预期的内部异常（CLI 仍不输出 traceback） |

所有成功产物先写入输出目录同级的隐藏 staging 目录，完成文件写入和校验和计算后再整体 rename 为 `--output`。任何业务失败都会删除 staging，只原子写入一个失败 `bake_manifest.json`。因此不会把半写的 OBJ/PNG 留在输出目录里。

调试和测试时可用 `--heightfield file.npy` 与 `--weights file.npy` 注入输入数组；正常烘焙不需要外部数据文件。

## 测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```

测试覆盖：

- 相同尺寸/种子下高度场和权重逐位一致。
- 三级 LOD 顶点/面数递减、误差不超过逐级阈值。
- 过渡网格复用高细节顶点、T 型接缝为 0、组合网格仍为单边界流形。
- 平坦/斜面高度场的法线 RGB 方向与分辨率缩放步长。
- OpenGL 不可用时强制触发 CPU 回退并记录原因。
- 非法尺寸、NaN 高度场、权重和不匹配均返回业务码且输出目录只有失败清单。
- 成功清单中的每个文件校验和与实际文件一致；成功与失败顶层 schema 一致。

## 实测结果

默认命令成功生成 `artifacts/terrain`：

- `LOD0`: 4225 顶点 / 8192 面，误差 0。
- `LOD1`: 2324 顶点 / 4500 面，相对误差 `0.005887529 <= 0.0105`。
- `LOD2`: 634 顶点 / 1204 面，相对误差 `0.024790484 <= 0.03`。
- 两级过渡分别生成 102 和 51 个缝合三角形，T 型接缝计数为 0，组合网格均为流形。
- 法线图 512x512，像素步长 `0.0019569471624266144`。
- 当前环境无可用 OpenGL 上下文，自动回退到 CPU，清单记录了 `attempted=["opengl","cpu"]`、`used="cpu"` 和具体原因。
- 11 个产物的清单 SHA-256 全部与磁盘文件复核一致。
