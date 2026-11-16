# tilegen：离线矢量瓦片生成管线

这是一个自包含的 C++17/CMake 实现，用于把分层矢量场景 JSON 光栅化为 RGBA 瓦片金字塔、瓦片索引清单和等距柱状投影预览图。工程不依赖 Docker、libpng、Eigen 或系统 zlib；内置严格 JSON 解析器、PNG 编解码器、CRC32、矢量光栅器和预乘 Alpha 合成器。

## 构建与运行

Windows（VS 2022 / MSVC v143）：

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
.\build\Release\tilegen.exe --config samples\pipeline.json
```

Linux：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/tilegen_tests
./build/tilegen --config samples/pipeline.json
```

CMake 要求 3.20 或更高版本。只有 C++17 标准库要求，没有 vcpkg 包依赖。

```bash
./build/tilegen \
  --scene samples/scene.json \
  --output samples/out \
  --tile-size 128 \
  --levels 3 \
  --preview-level 1 \
  --preview-width 512 \
  --preview-height 384 \
  --samples 4
```

## 配置优先级

优先级从低到高：

1. 内置默认值：瓦片 256、3 层、预览中间层、预览 512x256、4x4 超采样。
2. `--config` 指定的 JSON。
3. 命令行显式参数。

配置示例见 `samples/pipeline.json`。配置项支持 `scene`、`output`、`tile_size`、`levels`、`preview_level`、`preview_width`、`preview_height`、`samples`、`resume`、`simulate_disk_full_level`。路径按当前工作目录解析；纹理路径相对于场景文件解析。

## 场景格式

完整样本见 `samples/scene.json`。

```json
{
  "canvas": {"width": 512, "height": 384},
  "background": "#101827",
  "layers": [
    {
      "id": "terrain",
      "visible": true,
      "opacity": 0.9,
      "blend": "normal",
      "fill_rule": "nonzero",
      "transform": {
        "origin": [256, 192],
        "translate": [0, 0],
        "rotate": 0,
        "scale": [1, 1]
      },
      "primitives": [
        {
          "type": "polygon",
          "rings": [
            [[0, 0], [512, 0], [512, 384], [0, 384]],
            [[120, 80], [220, 80], [220, 180], [120, 180]]
          ],
          "fill_rule": "evenodd",
          "fill": {
            "color": [188, 220, 188, 255],
            "texture": "terrain_texture.png",
            "uv_rect": [0, 0, 1, 1],
            "repeat": false
          }
        },
        {
          "type": "polyline",
          "points": [[12, 328], [110, 290], [184, 314]],
          "stroke": {
            "width": 10,
            "color": [242, 178, 72, 210],
            "cap": "round",
            "join": "miter",
            "miter_limit": 4
          }
        }
      ]
    }
  ]
}
```

支持的图元：

- `polygon`：`points` 或 `rings`，可选 `holes`；支持 `nonzero` 和 `evenodd` 填充。
- `polyline`：由 `stroke.width` 控制线宽；端点支持 `butt`、`round`、`square`，连接支持 `miter`、`round`、`bevel`。
- `circle`：`center`、正数 `radius`，可同时填充和描边。

坐标约定为 x 向右、y 向下，旋转单位为角度。层变换为：

```text
p' = origin + translate + R(rotate) * S(scale) * (p - origin)
```

颜色接受 `#RGB`、`#RGBA`、`#RRGGBB`、`#RRGGBBAA`、0..1 归一化数组或 0..255 数组，也接受颜色名和 `{r,g,b,a}` 对象。`opacity` 为 0..1。混合模式为：

- `normal`：预乘 source-over。
- `add`：颜色混合函数 `min(1, Cb + Cs)`，再按 Alpha 合成。
- `multiply`：颜色混合函数 `Cb * Cs`，再按 Alpha 合成。

合成使用 W3C 预乘公式：

```text
Ao = As + Ab - As*Ab
Co = (1-As)*Cb*Ab + (1-Ab)*Cs*As + As*Ab*B(Cb,Cs)
```

所有通道在写入前截断到 `[0,1]`，因此 `add` 溢出、负输入和误差不会泄漏到输出。纹理使用双线性采样、UV 矩形、可选重复，并按填充颜色/透明度调制。当前解码器支持非隔行 8 位灰度、RGB、调色板、灰度 Alpha 和 RGBA PNG，并验证每个 chunk 的 CRC 和 zlib Adler-32。

## 金字塔与瓦片

- 层级 `L=0` 是场景全分辨率；每增加一层，世界到像素比例乘 `0.5`。
- 每块瓦片固定为 `tile_size x tile_size`，边缘瓦片在场景外的部分透明。
- 同一瓦片内，图元按输入顺序合成；每层先独立光栅化，再整体应用层 `opacity` 和 `blend`。
- 线宽在世界空间定义，进入层级时乘该层缩放因子，因此不同分辨率下视觉宽度一致。
- 每个层级只渲染可见层包围盒覆盖的瓦片；没有像素的瓦片仍写入清单，标记 `"empty": true, "present": false`。

输出结构：

```text
out/
  manifest.json
  index.json                  # manifest.json 的同等索引副本
  preview.png
  tilegen.log
  tiles/0/0/0.rgba            # 直通 RGBA8，行主序
  tiles/0/0/0.png             # 同内容 PNG 便捷副本
```

`manifest.json` 中的关键字段：

- `completed_levels`：已完整生成并持久化的层级。
- `tiles[].bbox`：该瓦片在世界空间中的包围盒。
- `tiles[].bbox_covered`：可见层包围盒是否覆盖该瓦片。
- `tiles[].layer_ids`：实际对像素有贡献的层 ID。
- `tiles[].layer_bboxes`：贡献层的世界包围盒。
- `tiles[].crc32`：对 `.rgba` 文件全部字节计算的 CRC32；与 `path` 指向的二进制严格一致。
- `tiles[].png_crc32`：对 PNG 文件字节计算的 CRC32。
- `preview.level`、`preview.scale`、`preview.projection_parameters`：预览源层级和投影参数。

预览先把指定层级完整渲染到该层分辨率，再用双线性重采样到请求尺寸。等距柱状映射使用经度 `[-180,180]`、纬度 `[-90,90]`，与所选层级记录在清单中。

## 错误与续跑

退出码：

| 退出码 | 含义 |
|---|---|
| 0 | 成功 |
| 1 | 命令行或配置错误 |
| 2 | 输出目录不可写或清单写入失败 |
| 3 | 磁盘空间不足（包括模拟失败） |
| 4 | 纹理缺失或损坏 |
| 5 | 场景 JSON/字段校验失败 |
| 6 | 渲染阶段错误 |

`--simulate-disk-full-level N` 会在层级 N 开始前返回 ENOSPC 退出码 3，用于确定性恢复测试；也可使用环境变量 `TILEGEN_SIMULATE_DISK_FULL_LEVEL`。

每完成一个层级、以及层级内发生异常时，清单都会原子更新，已写瓦片和此前完成层级保留。`--resume` 行为：

1. 校验清单 schema、场景 CRC32、瓦片尺寸和层级数。
2. 从最早未完整层级继续；已完成层级必须存在全部清单记录且文件 CRC 匹配。
3. 对已有且清单校验通过的瓦片直接复用，不重写 `.rgba`；缺失或损坏的 PNG 旁路文件会从已验证 RGBA 重建。
4. 损坏/缺失的 RGBA 瓦片会重新生成。

## 实际验证记录

环境：Linux x86-64，GCC 15.2，CMake 3.30.5，Release 构建。

```text
$ ./build/tilegen_tests
123 checks, 0 failures
```

测试覆盖：JSON 路径错误、负半径、缺失字段、动态 Huffman PNG 解码、损坏 PNG、双线性采样、预乘混合顺序、add/multiply 溢出与负值截断、nonzero/evenodd 填充、线宽随层级缩放、完整管线、空白瓦片、CRC 对应、预览尺寸/投影、确定性重跑、ENOSPC 后清单保留及无覆盖续跑。

确定性样本 `samples/pipeline.json` 实测：

```text
层级格网: 4x3, 2x2, 1x1
瓦片记录: 17，其中 present=17，empty=0
预览: 512x384，level=1，equirectangular，crc32=14890f3a
首个瓦片 tiles/0/0/0.rgba:
  65536 bytes, manifest crc32=14511472,
  Python zlib.crc32=14511472
总耗时: 约 2.6 秒（4x4 超采样）
```

恢复路径实测：

```text
--simulate-disk-full-level 2 -> exit 3
manifest.completed_levels = [0, 1]
--resume -> exit 0
manifest.completed_levels = [0, 1, 2]
已完成瓦片 mtime 未变化，未覆盖有效瓦片
```

其他错误路径实测：

```text
不可写输出目录 -> exit 2
缺失纹理 -> exit 4，日志含资源路径和 JSON 字段路径
配置设为 tile_size=128/levels=3/preview_level=1，
命令显式覆盖为 32/1/0/64x48 -> 清单采用命令行值
```
