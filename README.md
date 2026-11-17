# 离线软光栅渲染器

这是一个不依赖 GPU 或图形 API 的 C++17 软光栅原型。程序读取自定义 JSON
场景，完成视锥剔除、近/远平面裁剪、透视正确光栅化、深度测试、双纹理层混合、
方向光着色、Reinhard 色调映射和 sRGB 伽马校正，最后写出 24 位 BGR BMP。

渲染数学全部使用 `double`。Windows 构建会包含 Windows SDK 的
`DirectXMath.h` 并执行平台头文件校验；由于 DirectXMath 的 SIMD 向量运算本身是
单精度，管线中的投影、插值、深度和颜色计算使用项目内的双精度数学类型，避免
精度和可复现性冲突。非 Windows 环境使用同一套双精度后备实现，方便在 WSL/Linux
执行确定性验收。

## 构建

### Windows + Visual Studio 2022

前置条件：

- Visual Studio 2022 或更高版本，安装“使用 C++ 的桌面开发”工作负载。
- Windows 10/11 SDK，确保包含 `DirectXMath.h`。
- CMake 3.20 或更高版本。

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
.\build\Release\soft_raster.exe --input .\build\samples\acceptance.json `
  --output .\build\acceptance.bmp
```

Visual Studio 的 Windows SDK include 路径由工具链自动提供。项目不链接任何
图形库。

### Linux/WSL 验证环境

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/soft_raster --input build/samples/acceptance.json --output build/acceptance.bmp
```

构建目标 `sample_assets` 会确定性生成
`build/samples/textures/base.bmp` 与 `detail.bmp`。渲染器不使用网络、Docker、
GPU 或外部素材。

## 命令行

```text
soft_raster --input <scene.json> --output <image.bmp>
            [--width <pixels>] [--height <pixels>]
```

也支持 `-i`、`-o` 和 `--flag=value` 形式。宽度、高度可单独覆盖；未覆盖的维度
使用场景中的 `resolution`，场景未写该字段时默认 `640x480`。

成功时 stdout 只输出一行统计 JSON：

```json
{"totalTriangles":6,"culledMeshes":1,"rasterizedTriangles":4,"totalFragments":97210,"depthTestFailures":43211}
```

退出码：

- `0`：渲染和 BMP 写入成功。纹理缺失/损坏只写 stderr 警告。
- `1`：参数、JSON、场景校验或渲染资源错误。
- `2`：输出 BMP 无法打开或写入失败。

## 场景格式

完整字段说明和公式见 [`TECHNICAL.md`](TECHNICAL.md)。最小结构如下：

```json
{
  "resolution": [320, 240],
  "camera": {
    "position": [0, 0, -4],
    "direction": [0, 0, 1],
    "up": [0, 1, 0],
    "fovY": 60,
    "near": 0.5,
    "far": 20
  },
  "lights": [
    {"direction": [0, 0, 1], "color": [1, 1, 1], "intensity": 1.0}
  ],
  "materials": [
    {
      "name": "layered",
      "diffuse": [1, 1, 1],
      "baseTexture": "textures/base.bmp",
      "detailTexture": "textures/detail.bmp",
      "baseColor": [1, 0, 0],
      "detailColor": [0, 0, 1],
      "baseWeight": 1.0,
      "detailWeight": 1.0
    }
  ],
  "meshes": [
    {
      "positions": [-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0],
      "normals": [0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1],
      "uvs": [0, 0, 1, 0, 1, 1, 0, 1],
      "colors": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      "indices": [0, 1, 2, 0, 2, 3],
      "material": 0
    }
  ]
}
```

数组既可以是扁平数值列表，也可以是嵌套数组。模型矩阵为 16 个数的行主序
矩阵，也可以使用 `{"translation": [...], "rotation": [...], "scale": [...]}`
对象。相机可用 `direction` 或 `target` 表示朝向。纹理路径相对场景 JSON
所在目录解析。

## 测试

Python 3 验收脚本不依赖第三方包：

```bash
python3 tests/acceptance.py \
  --renderer build/soft_raster \
  --texture-gen build/generate_textures \
  --sample samples/acceptance.json
```

若环境提供 `ctest`，也可运行：

```bash
ctest --test-dir build --output-on-failure
```

没有 CTest 前端时可直接使用项目目标：

```bash
cmake --build build --target check
```

验收覆盖：

- BMP 尺寸、BGR 字节序、中心关键像素与背景像素。
- 统计 JSON 中三角形、剔除、片元、深度失败计数。
- 重复运行输出字节一致。
- 删除屏外网格后，可见像素逐字节不变。
- 近平面跨越三角形的裁剪与重新三角化。
- 超出 `[0,1]` 的 UV 通过重复寻址得到预期关键像素。
- 纹理缺失仅告警且退出码为 0。
- 缺失字段、材质越界、顶点属性数量不匹配、NaN、Infinity 返回 1。
- 输出路径不可写返回 2。
- 分辨率覆盖生效。

当前 WSL2/Linux 环境（GCC 15.2、CMake 3.30）实测结果：

```text
PASS: stats={"culledMeshes": 1, "depthTestFailures": 43211,
"rasterizedTriangles": 4, "totalFragments": 97210, "totalTriangles": 6}
center_bgr=(127, 0, 159) background_bgr=(0, 0, 0)
BMP SHA-256=f538990fe0a89812347fe7ad4155ef760ddabc3584b60969d561bdc9fd077c99
```

该环境没有 CTest 前端，因此测试由同一 CMake 注册的 Python 脚本直接执行；渲染器
本身无 Python 运行时依赖。Windows 构建路径需要在目标 Windows 机器上执行上述
命令完成工具链实测。

同一环境还通过了从空目录生成的 Release `-Werror` 构建，以及
AddressSanitizer + UndefinedBehaviorSanitizer 下的完整 `check` 目标。

## 目录

```text
include/raster/       公开头文件（数学、JSON、场景、BMP、渲染器）
src/                  实现与 CLI
tools/                确定性纹理生成器
samples/              内置验收场景
tests/                端到端验收脚本
TECHNICAL.md          管线、格式、公式与统计定义
```
