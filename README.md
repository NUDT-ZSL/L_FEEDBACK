# 确定性离线材质外观渲染色卡

该项目从零实现一条无 GPU、无第三方渲染引擎的离线预览管线。几何、UV
纹素与材质均由内置程序化函数生成；核心计算全部使用 NumPy 向量化，运行
时仅依赖 Python 标准库与 NumPy。测试使用 Pillow 作为独立 PNG 解码器。

## 快速运行

Windows 10/11、Python 3.10+：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python generate_sample.py --output samples/character_card_sample.json
python render_cards.py --scene samples/character_card_sample.json --out outputs/cards
```

离线安装时将 wheel 放入本地目录后执行：

```powershell
python -m pip install --no-index --find-links .\wheels -r requirements.txt
```

入口退出码：

- `0`：全部预设成功且一致性检查通过。
- `2`：存在可隔离的业务错误，已产出诊断 JSON；成功预设仍保留。
- `1`：输入 JSON 无法解析或其他致命错误。

测试：

```powershell
python -m unittest discover -s tests -v
```

## 项目结构

```text
render_cards.py                  批量运行入口
generate_sample.py               内置确定性样本生成器
render_pipeline/scene.py         JSON 加载、结构校验、逐光照预设校验
render_pipeline/geometry.py      程序化角色壳网格、正交相机、z-buffer 栅格化
render_pipeline/patterns.py      棋盘、条纹、整数哈希值噪声
render_pipeline/material.py      图层混合、UV 平铺/旋转、粗糙度与金属度遮罩
render_pipeline/shading.py       Lambert + GGX/Smith/Schlick 能量守恒近似
render_pipeline/tonemap.py       Reinhard、ACES、线性到 sRGB 编码
render_pipeline/renderer.py      视角渲染、逐像素元数据、关键像素记录
render_pipeline/consistency.py   UV/法线、金属边界、能量守恒断言
render_pipeline/pngio.py         固定块顺序的 8 位 RGB PNG 编码器
render_pipeline/batch.py         缓存、清单哈希、整目录发布、错误隔离
tests/test_pipeline.py           端到端验收测试
samples/character_card_sample.json
```

## 输入模型

`samples/character_card_sample.json` 是完整示例。关键字段：

- `uv_layout`：纹素图尺寸和程序化网格分段、半径、高度、轮廓参数。
- `material.base_color`：sRGB 底色；运行中先转为线性空间。
- `material.layers`：按数组顺序合成。支持 `albedo`、`roughness`、
  `metallic` 通道。
- `pattern.type`：`checker`、`stripes`、`noise`。
- `blend`：`normal`、`multiply`、`screen`、`overlay`、`add`。
- `opacity`：图层不透明度。
- `tiling`：UV 平铺倍率。
- `rotation_deg`：在每个平铺单元内旋转 UV。
- `mask_size`：粗糙度/金属度图层必需，必须严格等于
  `[uv_layout.width, uv_layout.height]`。
- `lighting_presets[]`：主光方向、主光 RGB 辐照度、环境 RGB 辐照度，以及
  `tone_mapping.operator/exposure/white_point`。
- `cameras[]`：默认是 `front`（yaw 0°）与 `side`（yaw 90°），也可在 JSON
  中指定其他离线视角。

噪声使用整数坐标哈希和值噪声插值，不读取系统时间、进程随机种子、环境随机
源或任何外部图片。

## 渲染与着色

网格是以 `u/v` 为参数的高分段闭合接缝角色壳。正面与侧面相机均通过同一个
UV 缓冲取材质，因此视角只改变可见表面，不改变 UV 语义。

着色在覆盖像素上执行：

1. 双线性采样合成后的线性反照率、粗糙度和金属度。
2. 使用 GGX 法线分布、Schlick-GGX 几何项和 Schlick Fresnel。
3. Lambert 漫反射与 GGX 高光按材质权重分配：

   ```text
   F_avg   = mix(0.04, mean(albedo), metallic)
   w_diff  = (1 - metallic) * (1 - F_avg)
   w_spec  = 1 - w_diff
   ```

   原始 GGX 响应经 `x / (1 + x)` 有界化，反射率再做硬上限归一化，保证
   `w_diff + w_spec <= 1` 且 `reflectance <= 1`。
4. 先在线性空间累加环境辐照度与主光辐照度，再乘曝光并执行 Reinhard 或
   ACES，最后进行 sRGB 传递函数编码并量化到 8 位。

任意预设在运行时都会检查：

- 两个视角的共同 UV 地标：UV L2 误差 <= `0.02`，法线 L2 误差 <= `0.08`。
- 金属度遮罩阈值边界两侧的线性着色差 <= `0.06`。
- 覆盖金属度 `[0, 1]` 的采样中，反射率与双瓣权重大小 <= `1.0`。

## 输出与像素元数据

```text
outputs/cards/
  cache/
    mesh_*.npy
    material_*.npy
    cache_manifest.json
  presets/<preset>/<camera>.png
  presets/<preset>/<camera>.metadata.json
  presets/<preset>/consistency.json
  diagnostics.json
  manifest.json
  manifest.sha256
```

每张 PNG 为 8 位 RGB、sRGB 语义。PNG 使用固定 IHDR/sRGB/IDAT/IEND 块顺序，
固定 zlib 压缩等级，不写入时间或其他可变辅助块。

每个 `*.metadata.json` 的 `pixels` 是行优先列式记录。所有列长度均为
`width * height`；背景像素在浮点/颜色列中为 `null`，在 `coverage` 中为
`0`。列包括：

- `x`、`y`、`coverage`
- `uv_u`、`uv_v`
- `normal_x`、`normal_y`、`normal_z`
- `linear_irradiance_r/g/b`
- `linear_color_before_tone_map_r/g/b`
- `tone_mapped_linear_r/g/b`
- `srgb8_r/g/b`
- `metallic`、`roughness`

`key_pixels` 另行记录三个跨视角 UV 地标及一个金属边界探测像素，字段包含
对应 UV、法线、辐照度、映射前后颜色和最终 8 位颜色。

`manifest.json` 记录原始输入 SHA-256、规范化 JSON SHA-256、每个预设结果，
以及除 `manifest.json` 自身外全部输出和缓存文件的 SHA-256、字节数与相对
路径。`manifest.sha256` 单独记录清单本身哈希，避免自引用。

## 诊断与发布

稳定业务错误码包括：

- `MAT_MASK_SIZE_MISSING`
- `MAT_MASK_SIZE_MISMATCH`
- `MAT_BLEND_UNSUPPORTED`
- `LIGHT_FIELD_MISSING`
- `TONE_MAP_OPERATOR_UNSUPPORTED`
- `CONSISTENCY_UV_MISMATCH`
- `CONSISTENCY_METALLIC_JUMP`
- `CONSISTENCY_ENERGY_VIOLATION`
- `SCENE_JSON_INVALID`
- `SCENE_FIELD_MISSING` / `SCENE_FIELD_INVALID`

材质错误会阻止依赖该材质的预设，但不会中断整个运行；单个光照预设缺少字段
或一致性检查失败也不会删除其他预设已生成的文件。所有错误写入
`diagnostics.json`，并在 `manifest.json` 的预设结果中汇总。

批量运行先写同级 staging 目录，完成后将旧目录改名为备份、将 staging 改名
为正式目录，再删除备份。因而正式目录不会混入上一版本的陈旧文件；若发布前
发生异常，旧目录保持不变。

## 实际验证结果

本工作区实际验证环境：Python 3.14.4、NumPy 2.5.4、Pillow 13.0.0
（满足 `numpy>=1.24`、`Pillow>=10`）。

```text
python -m unittest discover -s tests -v
Ran 10 tests in 26.734s
OK
```

内置样本通过两个独立 CLI 进程各运行一次，均返回 `0`，各产出 6 张 PNG、
6 份逐像素元数据、3 份一致性报告及确定性缓存。两次输出的全部 26 个文件
逐文件 SHA-256 一致，缓存文件也包含在比较中。

```text
scene input SHA-256: df3edea27ca79d81a0113f322d2db9b902bad74693500d615331dfffd5b3d54a
manifest SHA-256:    0a1908f79fbd4149eddfc2eca6a54d6b352bb9dd0d20362fb4dc216a7d981bf4
tree comparison:     EQUAL (26 files)
diagnostics:         0
```

三个预设的一致性实测：

```text
preset          max UV L2   max normal L2   max metal color delta   max reflectance   max lobe sum
studio_aces     0.003271    0.019532        0.048845                0.201553          1.0
dusk_reinhard   0.003271    0.019532        0.039788                0.201831          1.0
cool_rim_aces   0.003271    0.019532        0.030518                0.201534          1.0
```

测试还覆盖了：

- PNG 解码像素与元数据 `srgb8_*` 列逐像素一致。
- Reinhard/ACES 公式与线性空间转 sRGB 编码。
- 遮罩尺寸不匹配返回 `MAT_MASK_SIZE_MISMATCH`。
- 未知混合模式返回 `MAT_BLEND_UNSUPPORTED`。
- 单个光照预设缺字段返回 `LIGHT_FIELD_MISSING`，另外两个预设继续产出。
- 预先存在的目录被整体替换，陈旧文件消失。
- 清单中所有输出哈希、字节数和输入哈希可复核。
