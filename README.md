# 地牢离线延迟渲染验证器

这是一个无窗口、无外部资源依赖的 C++17 CPU 延迟渲染模拟器，用于在地图编辑器保存关卡前验证
G-Buffer、16x16 光源分块、点/聚光灯、阴影图集和增量重建结果。程序不创建图形窗口，也不访问网络。

## 构建

Windows 10/11 + Visual Studio 2019/2022：

```bat
cmake -S . -B build
cmake --build build --config Release
build\Release\dungeon_deferred.exe --help
```

Linux 本地验证：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j4
./build/dungeon_deferred --help
```

MSVC 使用 `/fp:strict`，GCC/Clang 使用 `-ffp-contract=off`。程序没有并行浮点归约，
所有光源、tile 和 G-Buffer 遍历顺序固定。

## 快速运行

基准样本为 160x112，默认使用 2x2、每 tile 64x64 的阴影图集。第一次运行会重建
3 个可投影光源的 tile：

```bash
./build/dungeon_deferred \
  --out out/summary_v0.json \
  --cache out/shadow_cache.bin \
  --width 160 --height 112 --geometry-version 0
```

使用同一个缓存文件再次运行，3 个 tile 应全部命中，渲染校验和不变：

```bash
./build/dungeon_deferred \
  --out out/summary_reuse.json \
  --cache out/shadow_cache.bin \
  --width 160 --height 112 --geometry-version 0
```

只移动光源 ID 2，验证选择性失效；其余两个 tile 命中，ID 2 重建：

```bash
./build/dungeon_deferred \
  --out out/summary_light_shift.json \
  --cache out/shadow_cache.bin \
  --light-shift 0.25
```

注入几何版本 1。场景实际增加一个木箱并移动柱子，几何哈希改变，所有阴影 tile 重建：

```bash
./build/dungeon_deferred \
  --out out/summary_v1.json \
  --cache out/shadow_cache.bin \
  --geometry-version 1
```

图集容量不足时不会崩溃，而是按光源 ID 升序保留前 N 个 caster，其余无阴影降级：

```bash
./build/dungeon_deferred \
  --out out/summary_atlas_full.json \
  --atlas-cols 1 --atlas-rows 1
```

阴影偏移越界返回业务错误码 `E_BIAS_RANGE`，进程退出码为 3：

```bash
./build/dungeon_deferred --out out/summary_bias_error.json --bias 0.5
```

## 管线实现

1. **场景构造**：`buildDungeonScene()` 确定性生成地面、墙、天花板、两根柱子和可选木箱。
   材质 ID 为地面 1、墙 2、柱子 3、木箱 4。`--geometry-version` 为奇数时启用第二套几何；
   `--geometry-scale` 控制平面尺寸。
2. **几何通道**：CPU 三角形光栅化。三角形先做近平面裁剪，再依据面向相机的几何法线剔除背面。
   深度测试使用相机空间 Z；位置和法线做透视正确插值，法线写入 G-Buffer 前再次归一化。
   G-Buffer 包含世界位置、归一化法线、材质 ID 和深度。
3. **16x16 分块剔除**：网格尺寸为 `ceil(width/16) x ceil(height/16)`。每块计算四角射线，
   由四角射线构造经过相机原点的四个侧平面，并用光源包围球做保守相交测试。光源 ID 升序输出。
   非 16 倍数分辨率由最后一个块裁剪为实际剩余像素。
4. **光照**：支持点光源和聚光灯。点光源使用距离衰减与平滑范围窗；聚光灯额外使用内外锥角
   smoothstep 衰减。每像素先写入非零环境光常量，因此空光源块和背景仍有非零亮度。
5. **阴影图集**：所有 caster 按 ID 升序分配固定 slot，槽位行优先。默认图集 2x2，每 tile
   64x64。slot 分配和输出顺序均确定。缓存文件保存 tile 深度、slot、光源变换哈希、几何版本
   和几何哈希。
6. **增量重建**：slot 缓存同时匹配 `lightTransformHash`、`geometryVersion` 和 `sceneHash`
   时复用；任一不匹配仅重建对应 tile。几何版本变化会改变场景并失效全部 caster tile。
7. **阴影采样**：聚光灯 tile 使用透视投影，点光源 tile 使用经纬等距柱面投影。深度 tile 通过
   离线射线求交生成。光照阶段使用 3x3 comparison sampling（PCF），按世界距离与
   `--bias` 比较；偏移参数允许范围为 `[0, 0.25]`。
8. **确定性摘要**：像素亮度乘 `1e6` 后取整，再用 FNV-1a 64 位哈希。逐块输出光源列表和块校验和，
   逐光源输出 tile 索引、命中/重建动作和阴影采样成功率。JSON 通过同目录临时文件写完后原子替换，
   不会出现半截文件。

## 摘要字段

主要字段如下：

- `config`：分辨率、16px tile 网格、相机参数、几何版本/哈希、bias、图集和缓存状态。
- `gbuffer`：G-Buffer 覆盖像素数及各材质像素数，用于确认几何通道实际写入了像素。
- `errorCodes`：去重后的业务错误码，包含 `E_ATLAS_FULL`、`E_BIAS_RANGE`。
- `businessErrors`：错误码、说明和可选的 `lightId`。
- `summary.tileHits` / `summary.tileRebuilds`：本次运行阴影 tile 复用/重建总数。
- `summary.ambientOnlyTiles`：光源索引列表为空的块数。
- `summary.pixelLuminanceChecksum`：全屏亮度校验和。
- `tiles[]`：块坐标、裁剪后尺寸、`lightCount`、升序 `lightIds`、块亮度校验和、平均亮度。
- `lights[]`：光源类型、固定 `tileIndex`/`atlasX`/`atlasY`、`action`（`hit`、`rebuild`、
  `degraded`、`none`）、`reused`/`rebuilt` 标记、变换哈希和阴影采样统计。

退出码：

- `0`：成功；`E_ATLAS_FULL` 是已降级的非致命业务错误。
- `2`：参数、相机、几何或图集配置非法。
- `3`：`E_BIAS_RANGE`。
- `4`：摘要或缓存文件写入失败。

即使发生错误，程序也会尽量原子写出完整错误摘要，而不是覆盖成半截 JSON。

## 自动验收

```bash
python3 tests/acceptance.py build/dungeon_deferred
# 或
cmake --build build --target run_acceptance
```

验收脚本覆盖：两次全新运行字节级一致、缓存全命中、光源移动选择性重建、几何注入全重建、
空光源块环境光、点/聚光灯、图集不足降级、bias 越界、非法视锥/分辨率、161x113 边缘 tile。

## 本地实测结果

验证环境：Ubuntu 26.04、G++ 15.2.0、CMake 3.30.5、`Release`。浮点结果只承诺在同一平台
和同一编译器下一致，不承诺跨编译器/跨 CPU 架构逐位一致。

| 场景 | 命中/重建 | 像素亮度校验和 | 几何哈希 | 结果 |
| --- | ---: | --- | --- | --- |
| 基准 v0 首次运行 | 0 / 3 | `0xee0b8e85368d4656` | `0xd0b42e69ebd7e8f1` | 3 个 caster 重建，3 个空光源块 |
| 基准 v0 同参数再运行 | 3 / 0 | `0xee0b8e85368d4656` | 同上 | 全命中，亮度完全一致 |
| 仅移动光源 ID 2 | 2 / 1 | `0x009842ae52ab8b0c` | 同上 | ID 1、3 命中，ID 2 重建 |
| 几何版本 v1 | 0 / 3 | `0xc576a97a7ccf9b82` | `0x3b432082eee6e8c5` | 几何改变，全部重建 |
| 1x1 图集 | 0 / 1 | `0xe69753d34fa91114` | `0xd0b42e69ebd7e8f1` | `E_ATLAS_FULL`，光源 2、3 降级 |
| 161x113 边缘裁剪 | 0 / 3 | `0x8afd1225d10b79a9` | `0xd0b42e69ebd7e8f1` | 11x8 块，最后一格 1x1 |

基准 v0 的 G-Buffer 覆盖 13,726 / 17,920 像素；材质像素数为地面 7,841、墙 5,176、
柱子 709、木箱 0。v1 中木箱为 178 像素。3 个 caster 的阴影采样成功率分别为
`0.998259`、`0.999836`、`0.997375`。

自动验收实际输出：

```text
PASS deterministic repeated run: byte-identical summaries
PASS cache reuse: initial 3 rebuild / warm 3 hit, identical shading
PASS selective invalidation: light shift 2 hit / 1 rebuild
PASS geometry injection: scene hash changed, 3 rebuild
PASS ambient-only tiles: non-zero luma and checksum
PASS point + spot lights and shadow comparison sampling
PASS E_ATLAS_FULL: deterministic 1x1 degradation for lights 2 and 3
PASS E_BIAS_RANGE: exit 3 and complete atomic error summary
PASS invalid frustum/resolution: exit 2 and error JSON
PASS 161x113 clipped tiles: 11x8 grid with 1x1 corner
```

## 设计边界

- 这是离线模拟器，不提交 GPU 命令，也不显示窗口。
- 点光源阴影使用单个固定 tile 的经纬深度图，而不是六个 cube-map 面，以满足“每光源固定 tile”。
- 光源分块剔除按业务要求使用包围球，可能保守地保留少量实际锥内能量为零的光源；最终能量
  由聚光锥衰减再次判定。
- 阴影图集缓存状态会影响 `tileHits`/`tileRebuilds` 和逐光源 `action`，但相同几何、光源和
  bias 下的像素亮度校验和保持一致。
