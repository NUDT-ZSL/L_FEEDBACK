# 实际验证记录

验证日期：2026-11-16。

## 环境

- Linux/WSL 提供的验收环境
- Python `3.14.4`
- NumPy `2.5.4`
- Pillow `13.0.0`
- 无 GPU、网络请求、容器或外部素材

本轮自动化命令在提供的 Linux/WSL 环境执行。实现只使用 `pathlib`、
`tempfile`、`os.replace` 等跨平台接口，目标 Windows Python 3.10+ 环境不需要
平台专属改动。

## 自动化测试

```bash
python -m unittest discover -s tests -v
```

结果：

```text
Ran 18 tests
OK
```

覆盖了采样边界、解析光栅化命中缓冲、四通道合成、行/列及正反分块一致性、
缺失/零法线回退、极端法线有限性、最近偶数量化、成功/失败 CLI、报告统计、
覆盖写出和提交失败回滚。

## 确定性样本

样本由以下命令生成并渲染：

```bash
python make_demo.py --out examples/demo --size 128 --texture-size 64
for scene in examples/demo/scene_*.json; do
  python render.py --scene "$scene"
done
```

五个场景均返回 `0`，渲染结果均为 `128x128`、`6480` 个命中像素。共有
`16384 - 6480 = 9904` 个背景像素。所有配置的 `nan_count=0`、
`inf_count=0`、`normal_fallback_count=0`，合成检查最大绝对误差为 `0.0`。

### 双线性采样：分块一致性

| 场景 | 分块配置 | 命中 | 合成最大误差 | 量化合成缓冲 SHA-256 |
| --- | --- | ---: | ---: | --- |
| `bilinear_16_column_reverse` | 16，列，逆序 | 6480 | 0.0 | `a5ef78a0cfb377deb94dcfac57078c4d6254430b7eb6d2c9b07ac1e7732d68ae` |
| `bilinear_32_row_forward` | 32，行，正序 | 6480 | 0.0 | `a5ef78a0cfb377deb94dcfac57078c4d6254430b7eb6d2c9b07ac1e7732d68ae` |
| `bilinear_64_row_forward` | 64，行，正序 | 6480 | 0.0 | `a5ef78a0cfb377deb94dcfac57078c4d6254430b7eb6d2c9b07ac1e7732d68ae` |
| `bilinear_128_whole` | 128，整幅 | 6480 | 0.0 | `a5ef78a0cfb377deb94dcfac57078c4d6254430b7eb6d2c9b07ac1e7732d68ae` |

上表哈希来自报告中的量化像素缓冲，不受 PNG 压缩器版本影响。实际
`composite.png` 文件在四种双线性配置下也逐字节一致，文件 SHA-256 为：

```text
9c494a8ddbcd2fa2daa6bc29fbf4e7727dbe92e294ea184eeaae219aa875a05a
```

### 最近邻对照

| 场景 | 过滤 | 命中 | 合成最大误差 | 量化缓冲 SHA-256 |
| --- | --- | ---: | ---: | --- |
| `nearest_32_row_forward` | nearest | 6480 | 0.0 | `464a8539c8f74f29bace2aba6726ae6cce677fc7a88e44f869e73149d69a88db` |

最近邻输出与双线性输出不同，符合两种采样的数学定义。其 `composite.png`
文件 SHA-256 为：

```text
e6e795aedf872f6fa8de8fc0c06dd598d96ee82109786f4724789e99d735b520
```

### 128x128 全图通道均值

双线性/最近邻场景的背景相同；以下为完整图像（含背景）的浮点均值：

| 通道 | 双线性 | 最近邻 |
| --- | ---: | ---: |
| base color | 0.211097888 | 0.211102000 |
| direct light | 0.064900988 | 0.064912349 |
| ambient light | 0.063386733 | 0.063381021 |
| composite | 0.101147362 | 0.101153496 |

## 错误路径

通过真实子进程调用 `render.py` 验证：

| 输入异常 | stderr 业务错误码 | 退出码 | 通道文件 |
| --- | --- | ---: | --- |
| 删除 `camera.target` | `E_SCENE_FIELD` | 2 | 未写出 |
| 基础色贴图 `1x1` | `E_TEXTURE_SIZE` | 3 | 未写出 |
| 相机 `[0,0,0.5]` 位于半径 1 球体内 | `E_CAMERA_INSIDE_SPHERE` | 3 | 未写出 |
| 64x64 输出使用 `block_size=24` | `E_BLOCK_NOT_DIVISIBLE` | 3 | 未写出 |
| 输出父路径是普通文件 | `E_OUTPUT_NOT_WRITABLE` | 3 | 未写出 |

实际 stderr 示例：

```text
E_SCENE_FIELD: $.camera.target: missing required field
E_TEXTURE_SIZE: base color texture size 1x1 is outside 2..4096
E_CAMERA_INSIDE_SPHERE: camera position is inside the sphere
E_BLOCK_NOT_DIVISIBLE: block_size 24 must divide output width 64 and height 64
E_OUTPUT_NOT_WRITABLE: cannot create output directory ...: [Errno 20] Not a directory
```

单元测试还通过注入 `os.replace` 失败，确认在已有成功输出的情况下提交中断会恢复
上一轮五个文件的原始哈希，并清理 staging 目录。
