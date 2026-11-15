# 地形矢量图层离线光栅化管线

本项目从零实现一条确定性、无网络的批处理管线：读取分层矢量 JSON，按 LOD
光栅化多边形与折线，输出 RGBA albedo、8 位高度图、切线空间法线图及元数据。
运行期只依赖 Python 3.10+、NumPy、Pillow；输入样本由项目代码确定性生成。

## Windows 快速开始

```powershell
py -3.10 -m pip install -r requirements.txt
py -3.10 -m terrain_pbr.sample --output samples\sample_scene.json
py -3.10 -m terrain_pbr `
  --input samples\sample_scene.json `
  --output-dir artifacts\sample_high `
  --width 128 --height 96 --lod high --grid-size 16
```

也可使用入口脚本：

```powershell
py -3.10 terrain_pipeline.py --input samples\sample_scene.json `
  --output-dir artifacts\sample_low --width 128 --height 96 `
  --lod low --grid-size 8
```

成功时 stdout 只有一行 JSON，包含四个输出文件的绝对路径、耗时及峰值栅格像素数，
进程退出码为 `0`。输出目录中生成：

- `albedo.png`：RGBA，逐层 straight-alpha `over` 混合。
- `height.png`：8 位灰度，来自几何覆盖率。
- `normal.png`：RGB，3x3 Sobel 烘焙的切线空间法线。
- `metadata.json`：图层顺序、包围盒、非透明像素数、自相交标志、LOD 顶点数、
  网格与峰值内存相关统计。

## 输入格式

完整可运行样例见 `samples/sample_scene.json`。每个逻辑图层包含若干
`polygon`/`polyline` 形状，每个形状必须提供 `high`、`medium`、`low` 三档点集。

```json
{
  "version": 1,
  "name": "terrain-scene",
  "normal_strength": 1.25,
  "layers": [
    {
      "id": "river",
      "color": [42, 117, 181],
      "alpha": 0.72,
      "shapes": [
        {
          "id": "river-band",
          "kind": "polygon",
          "geometry": {
            "high": [[0, 10], [40, 12], [50, 30]],
            "medium": [[0, 10], [40, 12], [50, 30]],
            "low": [[0, 10], [50, 30]]
          },
          "simplify": {"merge_distance": 1.25, "tolerance": 1.0}
        }
      ]
    }
  ]
}
```

折线可额外设置 `"width": 4.0` 与 `"closed": false`。多边形自动闭合；首尾重复点
会被去除。自相交多边形不会被拒绝，而会以奇偶规则渲染，并在元数据中置位。
格式错误、缺字段、非有限/非数值坐标、非法 LOD/画布/网格参数会向 stderr 输出
单行结构化 JSON，业务码为 `E_INVALID_INPUT`，退出码 `2`。

详细算法、元数据定义和失败回滚机制见 `TECHNICAL.md`。

## 测试与实际验证

```powershell
py -3.10 -m unittest discover -v
```

本次在提供的本地环境执行 `python -m unittest discover -v`，9 项测试全部通过
（约 3.5 秒），覆盖：

- 多图层 over 混合与 4x4 抗锯齿固定覆盖率（已知像素 alpha 为 `143/255`）。
- 高度图、Sobel 法线与零梯度默认法线 `[128, 128, 255]`。
- high/medium/low 切换、折线/多边形低档简化及顶点数元数据。
- 五角星自相交标志、奇偶规则中心空洞、重复运行 PNG 字节一致。
- 完全包围画布的多边形裁剪后无空洞。
- 非法字段、坐标、LOD、画布和网格的 `E_INVALID_INPUT`、退出码 `2` 及无部分输出。

网格统计实测（128x96 高层样本）：

| grid-size | peak_rasterized_pixels | total_rasterized_pixels | tiles_processed |
| ---: | ---: | ---: | ---: |
| 8 | 64 | 25920 | 405 |
| 16 | 256 | 27904 | 109 |
| 32 | 1024 | 34816 | 34 |

三种网格输出的 `albedo.png`、`height.png`、`normal.png` 均逐字节一致，说明空间
分块不改变像素结果。验证使用本地进程与本地文件，不访问网络；代码未使用容器、
平台专用命令或外部素材。本次命令执行环境为工作区提供的 WSL Ubuntu，路径与文件
替换逻辑使用 `pathlib`/`os.replace`，可直接在 Windows Python 3.10+ 运行。
