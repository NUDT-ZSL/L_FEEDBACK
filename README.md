# Skin Bake

离线烘焙工具，把蒙皮网格在动画期间的绑定姿势切线空间位移编码为
RG16F 曲线图集，并附带可直接用于实时渲染的映射表、报告和 OpenGL 3.3
查看器。

工作区中的样本完全由 `samples/generate_samples.py` 确定性生成，不下载任何
模型或纹理。

## 快速开始

要求 Python 3.10+、NumPy 1.24+；查看器额外要求 PyOpenGL 3.1.6+ 和支持
OpenGL 3.3 Core 的显卡驱动。

```bash
python -m pip install -r requirements.txt
python -m pip install -e .

python samples/generate_samples.py
python -m skin_bake bake samples/generated/scene_loop.json -o bakes/loop
python -m skin_bake check samples/generated/scene_loop.json bakes/loop \
  --frame 12 --output previews/software_frame12.png
```

Windows 本地交互查看器：

```powershell
python -m skin_bake view samples/generated/scene_loop.json bakes/loop
```

无桌面环境或 CI 中可走本地 EGL pbuffer，不访问网络：

```bash
python -m skin_bake view samples/generated/scene_loop.json bakes/loop \
  --frame 12 --headless-check previews/egl_frame12.png
```

查看器按键：

- `Space`：播放/暂停
- `Left` / `Right`：暂停并逐帧后退/前进
- `R`：回到首帧并播放
- `Esc`：退出

Windows 查看器使用纯 `ctypes` 创建 Win32 窗口和 WGL 3.3 Core 上下文，
不依赖 GLFW、GLUT、浏览器或在线资源。

## 场景格式

JSON 顶层字段：

```json
{
  "format": "skin-bake-scene",
  "version": "1.0.0",
  "hashAlgorithm": "sha256",
  "contentHash": "64-character-lowercase-hex",
  "mesh": {
    "positions": [[0, 0, 0]],
    "normals": [[0, 0, 1]],
    "tangents": [[1, 0, 0, 1]],
    "uvs": [[0, 0]],
    "boneWeights": [[0.5, 0.5, 0, 0]],
    "boneIndices": [[0, 1, 0, 0]],
    "triangles": [[0, 1, 2]],
    "skeleton": [
      {
        "name": "root",
        "parent": -1,
        "bindLocalMatrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "inverseBindMatrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
      }
    ]
  },
  "animation": {
    "fps": 24,
    "duration": 2.0,
    "loopMode": "loop",
    "boneTracks": [
      {
        "bone": 0,
        "keyframes": [
          {
            "time": 0.0,
            "translation": [0, 0, 0],
            "rotation": [0, 0, 0, 1],
            "scale": [1, 1, 1]
          }
        ]
      }
    ]
  },
  "displacement": {
    "space": "bind_normal_frame",
    "integration": "delta",
    "vertexCurves": [
      {
        "vertex": 0,
        "keys": [
          {"time": 0.0, "offset": [0.01, 0.0, 0.0]}
        ]
      }
    ]
  },
  "bake": {
    "textureSampleRate": 60.0,
    "atlasWidth": 128,
    "atlasHeight": 128,
    "minBlockSize": 1,
    "padding": 1,
    "errorTolerance": 0.03
  }
}
```

约束与语义：

- 矩阵均为 4x4 行主序；四元数为 `[x, y, z, w]`。
- 每顶点允许 1 到 4 组权重/索引，读取时补零到 4 组；权重必须非负且和为
  正，解析时归一化。
- `rotation` 插值使用短路径 slerp；相邻关键帧四元数点积为负时先取反。
- `displacement.space` 当前为 `bind_normal_frame`，偏移向量
  `[x, y, z]` 分别沿绑定姿势的切线、副切线、法线。
- `integration = "absolute"` 时关键帧值就是位移；`"delta"` 时按动画帧
  作为增量并用 Kahan 补偿求和累积。样本使用 `delta`。
- 对于循环动画，位移曲线末关键帧强制与首帧一致；缺失端关键帧时，最后的
  环绕区间插值回首帧。非循环动画在 `[0, duration]` 外保持首/末值。
- `contentHash` 是删除顶层 `contentHash` 后，对 JSON 按 key 排序、紧凑
  分隔符、UTF-8 编码得到的 SHA-256。文件被篡改时返回 `E_SCENE_HASH`。
- `fps * duration` 必须是整数，保证动画帧与声明时间轴对齐。

## 烘焙流程

1. 校验格式、版本、哈希、数组形状、骨骼层级、三角形索引和时间轴。
2. 根据绑定姿势法线/切线/手性重建正交 TBN。切线长度小于 `1e-6`、
   切线退化、法线退化或重建后基不正交时，使用固定的 64 点球面采样选择
   稳定切线，并记录顶点索引。
3. 每帧计算 `world_current * inverseBind`，按四权重线性混合矩阵；位置、
   法线、切向量均通过混合后的蒙皮矩阵变换。
4. 在绑定姿势 TBN 上投影输入位移，记录
   `|d - (uT + vB)| / |d|`。超过 `bake.errorTolerance` 时不写任何输出文件。
5. 循环动画烘焙 `round(duration * fps)` 帧；非循环动画额外包含终点帧。
6. 对每顶点位移曲线求二维包围盒。块内部像素矩形按包围盒长宽比和临时采样
   数确定，块四周扩张 `padding` 像素（默认 1）。
7. 用确定性的 MaxRects/BSSF 算法打包。存在重叠、越界或任一 padded block
   无法放入时，对所有块应用同一个时间采样缩放因子并重排；达到最小分辨率
   仍失败则返回 `E_ATLAS_CAPACITY`。
8. 按最终实际采样率重采样曲线，按行优先写入块内 texel，编码为 RG16F。
   边界像素复制边缘 texel 完成扩张，所有未被 padded block 覆盖的区域为零。
9. 所有产物先写入同级 staging 目录，成功后才原子替换目标目录。

## 图集与纹理格式

`atlas_map.json` 中每个 block 包含：

- `pixelOffset`：RG16F payload 左下角像素坐标
- `blockSize`：payload 宽高
- `paddedSize` / `blockOrigin`：包含边缘扩张的外框
- `sampleCount` / `sampleRate`
- `curveBounds`：该顶点投影位移曲线的二维包围盒

解码一个时间样本 `i`：

```text
row = i // blockSize.x
col = i % blockSize.x
texel = pixelOffset + (col, row)
displacement = texture[texel.y, texel.x].rg
```

循环动画在 `sampleCount` 个样本间环绕；非循环动画在
`sampleCount - 1` 个时间区间上线性映射。所有块的 `sampleRate` 相同，因为
下采样因子统一。

`displacement.rg16f` 是自描述二进制容器：

```text
offset 0   8 bytes   magic = "SKBAKETX"
offset 8   4 bytes   uint32 little-endian container version = 1
offset 12  4 bytes   uint32 little-endian JSON header byte length
offset 16  N bytes   UTF-8 compact JSON header
offset ... H*W*2*2   little-endian float16 RG payload, row-major, bottom-to-top
```

Header 包含尺寸、通道顺序、端序、原点、payload 字节数和 payload SHA-256。
读取时会校验 magic、长度和 payload 哈希。

## 输出与报告

一次成功烘焙输出：

```text
bakes/loop/
  displacement.rg16f
  atlas_map.json
  report.json
  scene.json
```

`report.json` 包含输入文件 SHA-256、场景内容哈希、实际/基准采样率、图集
尺寸、烘焙帧数、下采样因子、回退顶点及原因、每顶点最大投影误差、
float16 编码误差、循环闭合误差和烘焙耗时。

稳定业务错误写到 stderr，格式为：

```text
skin-bake: E_SCHEMA: detail
```

主要错误码：

- `E_SCENE_IO`、`E_SCENE_JSON`、`E_SCENE_HASH`
- `E_SCHEMA`、`E_VERSION`、`E_NUMERIC`、`E_TIMELINE`
- `E_TOLERANCE_EXCEEDED`
- `E_ATLAS_CAPACITY`
- `E_OUTPUT_IO`、`E_TEXTURE_CORRUPT`
- `E_GL_UNAVAILABLE`、`E_GL_RUNTIME`

## 查看器实现

- 运行时从映射表解码最近时间样本；CPU 计算当前帧蒙皮位置、法线、切线和
  副切线。
- OpenGL 顶点着色器从 RG16F 图集纹理取 `(u, v)`，沿当前帧切线空间基
  位移顶点；片元着色器做简单方向光照。
- Windows 使用 Win32 + WGL 创建 3.3 Core 上下文。
- Linux 无窗口验证使用 EGL pbuffer；本仓库已在 Mesa llvmpipe 上实际编译
  GLSL 330 shader 并读取帧缓冲。
- `check` 子命令使用独立的 NumPy 软件光栅器生成对照图，不要求 OpenGL。

## 测试

```bash
python -m pytest -q
```

测试覆盖哈希/版本/时间轴错误、短路径四元数、多骨骼权重、退化切线球面
回退、MaxRects 确定性与无重叠、统一下采样、容量失败、RG16F 哈希校验、
边缘扩张、循环/钳制边界、误差超限零输出、运行时解码、软件逐帧预览和
EGL OpenGL 实际渲染。

实际运行结果见 `VERIFICATION.md`。

