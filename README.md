# terrain-baker

离线地形贴图烘焙工具：从 JSON 场景描述出发，以确定性程序化噪声合成多层
高度/材质数据，输出分层漫反射图、切空间法线图、材质权重图与烘焙清单。
不读取外部图片，不访问网络；相同输入多次烘焙结果逐字节一致。

## 环境

- Python 3.10+，NumPy，Pillow（测试另需 pytest）

## 用法

```bash
python3 -m terrain_baker samples/scene.json --output out
python3 -m terrain_baker samples/scene.json --output out --tile-size 128
python3 -m pytest tests/ -q
```

输出目录包含：

| 文件 | 说明 |
| --- | --- |
| `diffuse.png` | 分层混合后的漫反射 RGB（8 位） |
| `normal.png` | 切空间法线 RGB（8 位，z 分量非负，即 >= 128） |
| `weights.png` | 材质权重 RGBA，每通道对应 `materials` 中同序材质 id |
| `manifest.json` | 烘焙清单（无时间戳，可复现） |
| `.tbcache/` | 自定义二进制缓存（高度场 float32 + 法线 uint8） |

## 场景格式

```json
{
  "seed": 20261116,
  "resolution": {"width": 512, "height": 512},
  "tile_size": 128,
  "normal_strength": 6.0,
  "materials": ["grass", "rock", "sand", "snow"],
  "layers": [
    {"material_id": "grass", "tile_scale": 6.0, "rotation_deg": 0.0,
     "blend_mode": "over", "height_scale": 1.0, "opacity": 1.0}
  ]
}
```

- `tile_scale` 取整为噪声平铺周期（>= 1，保证无缝平铺）。
- `blend_mode` 支持 `over` / `add` / `multiply`，按图层顺序累积。
- `max_pixels`（默认 4096²）限制分辨率内存预算，超限报 `E_RESOLUTION_LIMIT`。
- 材质最多 4 个（权重图 RGBA 四通道）。

## 混合公式（同时记录在清单 `blend_formulas` 中）

- 每层高度 `h = base + height_scale * (fbm - 0.5) * 2`（base 为低频基础高度，
  fbm 为旋转平铺后的分形值噪声扰动）
- 每层反照率 `a = material_color * (0.7 + 0.6 * fbm)`，材质基色由材质名
  SHA-256 确定性导出
- 覆盖率 `alpha = clip(opacity * mask_noise, 0, 1)`
- `over`: `dst = dst*(1-alpha) + src*alpha`，权重贡献 `alpha * remaining`
- `add`: `dst = dst + src*alpha`，权重贡献 `alpha`
- `multiply`: `dst = dst*(1 - alpha + alpha*src)`，权重贡献 `alpha * 0.5`
- 权重按材质累积后逐像素归一化（和为 1）

## 法线烘焙

- 梯度使用 Sobel（双线性权重）中心差分：
  `n = normalize(-gx*strength, -gy*strength, 8)`，z 恒为正。
- 边缘采用 1 像素环绕（wrap）padding，保证接缝连续；分块烘焙时每块生成
  halo 区域后裁剪，块间法线无差异。
- 8 位量化后对解码法线求平均长度，整体归一化校正因子
  `1 / mean_length` 记录于清单 `normal_correction_factor`，供引擎采样时修正。

## 确定性设计

所有噪声都是 `(seed, 整数像素坐标)` 的纯函数（uint64 环绕整数混合的
splitmix 风格哈希 + 值噪声/fbm），不使用任何全局随机状态，因此：

- 多次烘焙输出逐字节一致；
- 分块（任意 tile_size）与全图一次性烘焙逐像素一致（测试覆盖）。

## 二进制缓存格式（`.tbcache/*.tbc`）

```
0   8   魔数 "TBCACHE1"
8   4   格式版本 (uint32 LE)
12  4   负载类型 (1=高度 f32, 2=法线 u8x3)
16  4   宽度 / 20 4 高度
24  64  参数哈希 (ascii, 零填充)
88  ..  分块: [u32 长度][u32 CRC32][负载] 直至 EOF
```

启动时校验魔数、版本、参数哈希与逐块 CRC；版本不匹配或 CRC 失败时
清除全部旧缓存并重建，绝不沿用旧偏移读取。

## 错误码（stderr 输出 JSON，非零退出码）

| 退出码 | 错误码 | 含义 |
| --- | --- | --- |
| 2 | `E_SCENE_MISSING` | 场景文件不存在 |
| 3 | `E_SCENE_INVALID` | JSON 结构/字段校验失败 |
| 4 | `E_UNKNOWN_MATERIAL` | 图层引用未注册材质 id |
| 5 | `E_RESOLUTION_LIMIT` | 分辨率超出内存预算 |
| 6 | `E_TARGET_EXISTS` | 输出目录已存在且非本工具创建 |
| 7 | `E_PATH_UNSAFE` | 路径含 `..` 穿越或符号链接 |
| 8 | `E_CACHE_INVALID` | 缓存校验失败（保留码） |
| 1 | `E_INTERNAL` | 未预期内部错误 |

失败时清理本次运行写入的所有文件，不留下部分输出。

## 安全边界

- 输出路径拒绝 `..` 穿越与任一符号链接组件（`E_PATH_UNSAFE`）。
- 输出目录已存在且不含本工具清单标记（`manifest.json` 中
  `generator: terrain-baker/...`）时拒绝覆盖（`E_TARGET_EXISTS`）。
- 所有输出先写临时文件再原子重命名。

## 实际验证结果（本机，Python 3.14 / NumPy 2.5.4 / Pillow 13.0.0）

- `python3 -m pytest tests/ -q` → 16 passed。
- 示例场景（512×512，4 层，seed 20261116）两次独立运行输出 SHA-256
  完全一致；`params_hash = 43ac499607c1839c46eef9bb6a9b7c38e69ece08f2fefb42259275c13713a860`。
- 分块（tile_size=128）与全图（tile_size=4096）烘焙：diffuse/normal/weights
  逐像素一致，清单图层统计与校正因子一致。
- 法线图 z 通道最小值 >= 128；权重图每像素通道和 ∈ [254, 256]（8 位舍入）。
- 六类异常路径（场景缺失/JSON 错误/未知材质/分辨率超限/目标已存在/
  路径穿越与符号链接）均返回对应错误码与非零退出码，且无残留输出。
- 缓存版本号篡改与 CRC 损坏均能触发自动清除重建，重建结果与原始一致。
