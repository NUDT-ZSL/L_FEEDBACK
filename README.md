# material-baker

离线 PBR 材质烘焙命令行工具。读取固定尺寸的 `albedo.png`、`roughness.png`、
`metallic.png`、`normal.png`，按 JSON 中的方向光和半球环境光参数计算：

- `irradiance.png`：仅环境漫反射贡献
- `specular.png`：方向光镜面贡献 + 环境镜面贡献
- `final_color.png`：方向光与环境光的总和
- `metadata.json`：光照、尺寸、位深、伽马和文件名等实际输出信息

支持 Windows、Python 3.10+、8/16 位 PNG、128/256/512 输出尺寸、8/16 位输出、
`gamma=1.0/2.2`、绿色通道翻转、零长度法线回退和确定性自检。

## 安装与运行

```bash
python -m pip install -r requirements.txt
python -m material_baker selftest
python -m material_baker bake --config examples/config.json --input <贴图目录> --output <输出目录>
```

也可通过兼容入口运行：`python main.py bake ...` 或 `python bake.py bake ...`。

也可使用位置参数或 Windows 启动脚本：

```bash
python -m material_baker bake examples/config.json --input <贴图目录> --output <输出目录>
material_baker.cmd bake examples/config.json --input <贴图目录> --output <输出目录>
```

安装为 Python 包后可使用 `material-baker` 命令：

```bash
python -m pip install .
material-baker bake --config examples/config.json --input <贴图目录> --output <输出目录>
```

`--input`/`--input-dir` 优先于配置中的 `input_dir`；`--output`/`--output-dir`
同理。命令行和配置都未提供目录时返回 `E_INVALID_CONFIG`。

## 配置

最小配置：

```json
{
  "resolution": 256,
  "output_bits": 8,
  "output_gamma": 2.2,
  "flip_green": false,
  "view_direction": [0.0, 0.0, 1.0],
  "directional_light": {
    "direction": [0.4, 0.5, 0.8],
    "color": [1.0, 0.9, 0.8],
    "intensity": 2.0
  },
  "hemisphere_light": {
    "sky_color": [0.3, 0.55, 0.9],
    "ground_color": [0.12, 0.08, 0.06],
    "intensity": 0.7
  }
}
```

字段说明：

- `resolution`：仅允许 `128`、`256`、`512`，缺省 `256`。
- `output_bits`：仅允许 `8`、`16`，缺省 `8`。
- `output_gamma`：仅允许 `1.0`、`2.2`，缺省 `2.2`。
- `flip_green`：解码法线前翻转绿通道，缺省 `false`。
- `view_direction`：从表面指向观察者的向量，缺省 `[0,0,1]`。
- `directional_light.direction`：从表面指向光源的向量，载入时归一化。
- `directional_light.color/intensity`：线性光颜色和非负强度。
- `hemisphere_light.sky_color/ground_color`：半球天空/地面辐照度颜色；
  只提供 `color` 时，天空和地面使用相同颜色。
- `alpha_epsilon`：可选的 GGX 粗糙度下限，缺省 `1e-3`。
- `input_dir`/`output_dir`：可选路径，命令行参数优先。
- `output_names`：可选，可分别覆盖 `irradiance`、`specular`、
  `final_color` 的 PNG 文件名。

兼容别名包括 `gamma`、`bit_depth` 或 `output_bit_depth`、`directional`/`light`、
`ambient`/`environment`、`input_directory`/`output_directory`、`sky`/`ground` 等。

四张输入图必须同时存在、尺寸一致，且每张均为 8 或 16 位 PNG。输入贴图按线性值
处理，不做隐式 sRGB 解码；PNG 输出强度统一按
`round(clamp(linear,0,1)^(1/output_gamma))` 编码。

## 光照模型

所有方向均在像素处归一化，`L` 为表面指向光源，`V` 为表面指向观察者。
切线空间法线按 Y-up 解码：

```text
n = normalize((2R-1, 2G-1, 2B-1))
```

`flip_green=true` 时先令解码后的 `ny = -ny`。解码或缩放后向量长度不大于
`1e-2` 时回退为 `(0,0,1)`，向 stderr 输出 `WARNING`，但退出码保持 0。该阈值
同时覆盖 8 位编码无法精确表示 `0.5` 时产生的近零法线。

### 方向光

令 `alpha=max(roughness^2, alpha_epsilon)`，`H=normalize(L+V)`：

```text
D_GGX = alpha^2 / (pi * ((NoH^2*(alpha^2-1)+1)^2 + 1e-12))
G1(x) = 2x / max(x + sqrt(alpha^2 + (1-alpha^2)x^2), 1e-12)
G_Smith = G1(NoV) * G1(NoL)
F0 = 0.04*(1-metallic) + albedo*metallic
F = F0 + (1-F0)*(1-VoH)^5
f_diffuse = (1-F)*(1-metallic)*albedo/pi
f_specular = D_GGX*G_Smith*F/max(4*NoV*NoL, 1e-12)
incident = light_color*intensity*NoL
direct_diffuse = f_diffuse*incident
direct_specular = f_specular*incident
```

如果某个 RGB 通道的 `direct_diffuse+direct_specular` 超过 `incident`，两项按同一
比例缩放，确保方向光直接贡献不超过入射能量。`NoL<=0` 时直接光为零。

### 半球环境光

半球梯度 `E(direction) = ground + (sky-ground)*clip(0.5*dz+0.5,0,1)`。

```text
E_hemi = intensity * E(normal)
F_ambient = F0 + (1-F0)*(1-NoV)^5
env_diffuse = (1-F_ambient)*(1-metallic)*albedo*E_hemi
R = normalize(2*NoV*normal - V)
prefiltered = intensity * (E(R)*(1-roughness^2) + mean(sky,ground)*roughness^2)
env_specular = F_ambient*prefiltered
```

三张浮点结果：

```text
irradiance = env_diffuse
specular = direct_specular + env_specular
final_color = direct_diffuse + direct_specular + env_diffuse + env_specular
```

金属材质使用 `F0=albedo`，其环境镜面亮度随反射方向和朝向变化；纯电介质使用
`F0=0.04`。

### 缩放

法线先解码为向量再缩放，缩放后重新归一化。反照率、粗糙度和金属度在线性空间
处理。下采样使用可分离高斯核，边界采用 edge clamp，核标准差为
`0.5*sqrt(1/scale^2-1)`；滤波后按像素中心双线性采样。上采样使用双线性插值。
最后一律写入前钳制到 `[0,1]`。

## 输出元数据

`metadata.json` 同时包含 `lighting`、`output` 和扁平兼容字段，记录：

- 归一化方向光方向、颜色和强度
- 半球天空/地面颜色和强度
- 输出宽度、高度、位深、伽马
- 三张贴图的实际文件名及每张图的尺寸/位深/伽马
- 输入贴图位深、`flip_green`、模型参数和零长度法线计数

输出 PNG 由内置编码器直接支持 8/16 位 RGB，不依赖 Pillow 的 16 位 RGB 写入路径。
输入解码支持灰度、RGB、灰度+Alpha、RGBA、8 位调色板，以及普通或 Adam7 隔行 PNG。

## 错误处理

业务错误写入 stderr，格式为 `ERROR <code>: <message>`，进程返回非零整数：

| 错误码 | 退出码 | 含义 |
| --- | ---: | --- |
| `E_MISSING_FILE` | 2 | 配置文件、输入目录或四张必需 PNG 缺失 |
| `E_INVALID_FORMAT` | 3 | 非 PNG、位深不是 8/16、损坏 PNG、尺寸不一致等 |
| `E_INVALID_CONFIG` | 4 | JSON、字段类型、分辨率/位深/伽马取值非法 |
| `E_IO` | 5 | 输入读取或输出写入的 I/O 失败 |
| `E_INTERNAL` | 1 | 非预期内部错误或非有限数值 |

零长度法线属于数值警告，不改变退出码。

## 自检

```bash
python -m material_baker selftest
python -m material_baker selftest --work-dir ./selftest-output --keep
```

自检使用固定种子 `1592598566`（`0x5EED2026`），生成渐变/噪声样本，覆盖：

- 法线 Y-up 解码、`flip_green`、零向量回退和 stderr 警告
- 独立标量参考模型的抽样像素对比，绝对容差 `3e-4`
- 方向光漫反射+镜面反射能量上限、零粗糙度、纯金属/纯电介质
- 四角/边界和随机像素
- 16 位灰度输入、16 位 RGB 输出、Pillow PNG 完整性检查
- 128 下采样、512 上采样、8/16 位输出、`gamma=1.0/2.2`
- PNG/JSON 尺寸、位深、伽马、文件名一致性
- 缺文件与 4 位 PNG 的 CLI 非零退出码及 stderr 错误码

实际运行结果：`SELFTEST PASSED: 15 checks`，抽样理论值最大绝对偏差约
`7.6e-6`（容差 `3e-4`）。

单元测试：

```bash
python -m unittest discover -s tests -v
```

