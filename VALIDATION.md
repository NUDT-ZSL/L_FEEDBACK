# 实际验证记录

验证日期：2026-11-17。

当前执行容器没有 Python 3.11，实际解释器为 Python 3.14.4；NumPy 2.5.4，Pillow 13.0.0。实现只使用 Python 3.11 支持的语法与 NumPy 2.0 API，目标 Windows 环境按 `requirements.txt` 安装。以下结果来自当前环境。

## 自动化测试

命令：

```text
python -m pytest -q
```

结果：

```text
16 passed in 26.81s
```

覆盖点包括齐次裁剪、近平面后方三角形、共享边无空洞、级联重叠、投影阴影、单格细长脊线阴影、材质逐层加权等价、未归一化权重告警、增量/全量逐数组一致、脏区不完整回退、相机与几何尺度缓存失效、CLI 重复输出及四类非法参数。

## 重复渲染字节一致性

命令参数（两次仅输出目录不同）：

```text
python render_terrain.py --height-size 17 --resolution 128 96 \
  --cascades 3 --shadow-size 128 --tile-size 16 \
  --unit-scale 0.4 --height-scale 2.0 --output-dir <run>
```

SHA-256：

```text
color.png             a08c96d76136ec7ee7953d2ec003032c17a84be32fdf61a570be33181557cc4e
color.png             a08c96d76136ec7ee7953d2ec003032c17a84be32fdf61a570be33181557cc4e
depth.png             338937a6bd75f9a837b6deca41738d865505e726d849a4c27d8ee787d4900008
depth.png             338937a6bd75f9a837b6deca41738d865505e726d849a4c27d8ee787d4900008
shadow_visibility.png bfc50c717bbbe4a06f3344cd3bcc1d86d317084f34750fc581e1d22d99e79a33
shadow_visibility.png bfc50c717bbbe4a06f3344cd3bcc1d86d317084f34750fc581e1d22d99e79a33
```

清单中的 `summary_sha256` 两次一致；耗时字段不参与摘要。

## 脏区增量正确性

场景：17x17 高度场，中心脊线，160x120 输出，3 级 128x128 阴影。第二次将 `[4:6,4:6]` 高度增加 0.2，并声明脏区 `(3,3,7,7)`。

增量清单：

```text
mode                 incremental
full_rebuild         false
reason               valid_dirty_rect
rendered_tiles       16 / 80
shadow_tiles_updated 769
```

增量输出与同一修改后的全量重渲染比较：

```text
color.png             c7168b4a071884319b513868fe1627aa1a67fcf75790901f6163f3b2fb1d6cb6
color.png (full)      c7168b4a071884319b513868fe1627aa1a67fcf75790901f6163f3b2fb1d6cb6
depth.png             fae4346911d5f19b70d90524f538eb4c36c0900cf24360dc73cb4d850d38918e
depth.png (full)      fae4346911d5f19b70d90524f538eb4c36c0900cf24360dc73cb4d850d38918e
shadow_visibility.png d4e437213a9cfcdc63e62986f46abb9175557518abaa8f72e7521bb796f77a71
shadow_visibility.png d4e437213a9cfcdc63e62986f46abb9175557518abaa8f72e7521bb796f77a71
```

## 告警码

用每层权重均设为 `0.37` 的 4 层权重图运行：

```text
render_terrain.py --height-size 17 --resolution 64 48 \
  --cascades 2 --shadow-size 64 --weights-file unnormalized.npy \
  --output-dir warn
```

清单结果：

```text
warning_codes ["MATERIAL_WEIGHTS_NOT_NORMALIZED"]
```

渲染继续，权重按像素和归一化。

## 非法参数退出码

```text
负数分辨率          exit 2  ERROR: 参数校验失败: width 必须为正整数
级联数为零          exit 2  ERROR: 参数校验失败: 级联数量必须为正整数
脏区越界            exit 2  ERROR: 参数校验失败: 脏区越界: (0,0,99,4)，高度场尺寸为 9x9...
权重图尺寸不匹配    exit 2  ERROR: 参数校验失败: 权重图尺寸不匹配: 高度场 (9, 9), 权重图 (4, 5, 5)
```

四类错误均未创建对应输出目录，满足“校验先于写文件、无半成品输出”。

## PNG 读取检查

使用 Pillow 读取：

```text
color.png              mode RGB,   uint8  (H,W,3)
depth.png              mode I;16,  uint16 (H,W)
shadow_visibility.png  mode L,     uint8  (H,W)
```
