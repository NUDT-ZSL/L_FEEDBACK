# 确定性离线棱镜路径追踪器

这是一个仅依赖 Python 3.11/3.12 与 NumPy 的 CPU 离线渲染器。场景完全由
`prism_tracer/scene.py` 内置构造，不读取模型、纹理或 HDR 等外部资产。

## 运行

```powershell
python render.py --scene prism --res 320x240 --spp 64 --seed 7 --tile 64
```

其他常用参数：

```powershell
# 关闭间接漫反射，只保留直接光与焦散光子图
python render.py --res 320x240 --spp 64 --seed 7 --no-indirect

# 完全禁用焦散光子图，便于做对照实验
python render.py --res 160x120 --spp 16 --seed 7 --caustic-photons 0

# 显式指定输出目录
python render.py --res 160x120 --spp 16 --seed 7 --out out
```

默认输出到 `out/`：

- `prism_<W>x<H>_spp<N>_seed<S>_d<D>[_noindirect].ppm`：8-bit P6 静帧，
  使用固定曝光和 Reinhard + sRGB 传递函数显示。
- `..._aux.ppm`：路径类型辅助图。
- `..._report.json`：确定性 JSON 统计报告。

## 场景与积分器

- 封闭三角网格棱镜由 8 个三角形组成，位于有限平面水层上方；水面以下是漫反射地面。
- 相机为透视投影，从侧上方拍摄。面光源是带方向矩形，向下发射白色光谱辐射。
- 玻璃、水、空气的折射率与吸收系数均按波长表插值。界面使用无偏振菲涅耳方程：
  反射按概率 `F` 采样，透射按 `1-F` 采样，分支概率本身已经完成能量分配，因此
  不再额外乘以选择概率。透射路径对辐射亮度乘以 `eta^2`，以处理折射界面的
  辐射亮度缩放。
- 当高折射率介质射向低折射率介质且超过临界角时，菲涅耳反射率置为 1，路径
  保持在同一介质中且吞吐量不变，因而玻璃内部多次全反射不会丢能量。
- 漫反射点使用独立随机流做矩形面光源 NEE。光源采样与 BSDF/材质采样使用完全
  不同的 RNG stream。每个漫反射点都继续做余弦加权间接反弹；`--no-indirect`
  只关闭这个反弹，不关闭直接光或焦散。
- 为避免直接光被 NEE 与随机 BSDF 方向重复计算，漫反射后的直接光由 NEE 和焦散
  光子图负责。只有 NEE/光子图未覆盖的水面镜面反射链才保留 BSDF 命中光源贡献。
- 玻璃焦散使用确定性的前向光子图：矩形面光采样位置和余弦方向，光子经玻璃/水面
  的菲涅耳分支与介质吸收后，落到地面。NEE 把玻璃视为遮挡，因此 NEE 直射项与
  光子图玻璃路径项互补，不重复计数。光子能量按双线性离散到地面网格，再做三次
  可分离五点高斯平滑；查询时除以网格面积得到辐照度。

## 光谱与白点

每条相机路径和每个光子使用分层 hero wavelength。可见光谱为 400–700 nm，
波长到 RGB 的映射采用三条固定、平滑、严格正值的 RGB 响应曲线，中心/宽度分别为
`620/90`、`560/100`、`470/90` nm。每条曲线独立归一化，使等能白光谱积分得到
`(1,1,1)`；报告中的 `fixed_weight_white_error_rgb` 给出残差（当前约 `1e-4`）。

介质中的体积吸收按光学长度积分：

```text
throughput *= exp(-sigma(lambda) * segment_length)
```

因此吸收是波长相关的，而不是简单的后处理调色。玻璃和水的吸收表不同，但系数
保持温和，灰白光源穿过场景后不会产生明显色偏。

## 确定性、续渲与检查

- 随机数使用 counter-based SplitMix64，随机值是
  `(seed, pixel/photon, sample, stream, event, draw)` 的纯函数。改变 tile 边界不会
  改变样本；波长、光源、材质和相机分层还使用互相独立的每像素随机置换，避免
  不同积分维度的分层相关性。
- 每个 tile 完成后立即把全图累加数组、样本平方和、直接/间接分量、辅助图分量
  原子写入 `.partial-<config-hash>.npz`。如果按 `Ctrl+C`，退出码为 130，不写最终
  PPM；再次用相同参数运行会读取检查点并跳过已完成 tile。
- 最终 PPM 和 JSON 均通过临时文件 + `os.replace` 原子替换。成功完成后检查点自动
  删除。检查点指纹包含场景哈希、分辨率、spp、seed、tile、深度、间接光开关、
  光子数和曝光，参数不一致时会报告 `E_CHECKPOINT`。
- JSON 不包含时间戳、运行耗时或“是否续渲”等非确定字段，因此一次跑完与断点续渲
  的图像、辅助图和报告逐字节一致。

## JSON 报告

报告包含：

- `seed`、`samples_per_pixel`、`max_path_depth`、`resolution`、tile、间接光开关。
- `channel_stats.R/G/B`：线性辐射亮度均值、逐像素蒙特卡洛方差、标准差、像素均值
  方差、最小/最大像素均值、相对噪声，以及直接/间接光均值与能量闭合误差。
- 顶层 `noise` 的每通道均值/方差/估计标准差/相对噪声摘要。
- `hashes.image_sha256` 与 `hashes.auxiliary_sha256`。
- `path_type_aux.energy_share` 和 `sample_class_hits`。
- 焦散光子发射数、折射/全反射沉积数、玻璃透射事件数、网格分辨率和归一化说明。
- 光谱范围、固定 RGB 响应曲线、白点残差、介质 IOR/吸收表。

辅助图三个通道分别独立按自身最大值缩放：

- 红：直接漫反射输运。
- 绿：经玻璃折射透射的路径/焦散。
- 蓝：包含全反射的透射链路。

## 错误码

错误只写 stderr，格式为 `ERROR [CODE] message`，不写最终 PPM。

| 代码 | 退出码 | 含义 |
|---|---:|---|
| `E_USAGE` | 2 | 命令行/数值格式错误 |
| `E_SCENE` | 10 | 未知内置场景 |
| `E_SCENE_FIELDS` | 11 | 场景必填字段缺失 |
| `E_SCENE_VALUE` | 12 | 场景数值/表格/网格非法 |
| `E_RESOLUTION` | 13 | 非法分辨率 |
| `E_SPP` | 14 | 非正或过大的采样数 |
| `E_SEED` | 15 | 非法随机种子 |
| `E_TILE` | 16 | 非法 tile 尺寸 |
| `E_DEPTH` | 17 | 非法最大路径深度 |
| `E_CAUSTIC_PHOTONS` | 18 | 非法光子数 |
| `E_CAMERA_MEDIUM` | 20 | 相机位于水/玻璃介质内部 |
| `E_IO` | 21 | 输出 I/O 失败 |
| `E_CHECKPOINT` | 22 | 检查点损坏或参数不匹配 |
| `E_INTERRUPTED` | 130 | 用户中断；已完成块已检查点化 |

## 验证结果

环境：Python 3.14.4、NumPy 2.5.4（代码只使用 Python 3.11+ 语法与 NumPy 1.26+
API；Windows 上无需 GPU 或外部资产）。

说明：本次自动化工作区实际运行在 Linux 容器中；代码未使用 `fork`、Unix 信号处理、
`fcntl` 或其他仅 POSIX 可用的接口，文件替换统一使用 `pathlib` 与 `os.replace`，
因此可直接在 Windows 的 `python` 命令下运行。真实 Windows 实机验收仍需按上文命令
执行一次。

单元/集成测试：

```powershell
python -m unittest discover -s tests -v
```

结果：10 项全部通过，覆盖白点归一、菲涅耳/TIR、非法参数、缺字段、相机介质校验、
续渲字节一致、tile 无关性和 `--no-indirect` 的零间接能量。

验收命令实际运行：

```powershell
python render.py --scene prism --res 320x240 --spp 64 --seed 7 --tile 64
```

本机耗时约 68 秒，输出：

```text
image  c594b2d158d980df9039cf1a34e5c91a7d1d6c8f5cce130db1bf2394e907c144
aux    d2d3838088f6b6c61087f18865fafc147df68e8dfcee3659be2a77c49e25c479
JSON   7826ed54a5f1db965c7bb557f1a98c00e337db2f84d64ed35df181be298266e4
```

同一命令再次完整运行时，上述三个文件的 `sha256sum -c` 全部为 `OK`。

真实 `SIGINT` 中断测试（160×120、32 spp）返回 130，中断后没有 PPM，只留下
`.partial-*.npz`；继续运行后与一次跑完的三份文件哈希完全一致：

```text
PPM   2b0c1fe084621af11fcae631a119cbca9bf0e5e3ca227287f4f9954f77f8338f
AUX   7493d4566f7b454bacef4c5b16d0252f40b0ac77cf3cab27e629be31d29c8353
JSON  6321a98e4735c619d5f07d95753fe5534812d24bb157379ecac360819a4dc9cf
```

非法参数实测退出码：分辨率 13、负 spp 14、未知场景 10、负 tile 16、深度 0 为 17、
非法光子数 18、负 seed 15；均未产生 PPM。场景 API 校验中缺字段为 11，相机在玻璃
内部为 20。

## 已知边界

- 水面 NEE 使用平面水层的直线可见性近似，不做目标点与光源之间的折射方向求解；
  玻璃引起的直接焦散由前向光子图承担。
- 光子图提供平滑的焦散辐照度估计，不使用 kd-tree 密度估计；网格分辨率与平滑
  次数在报告中明确记录。
- PPM 为 8-bit 显示结果；JSON 方差统计基于线性辐射亮度，而不是 tone-map 后
  的 8-bit 像素。
