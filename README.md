# Deferred Lighting Snapshot Renderer

一个可实际运行的 OpenGL 3.3 Core 延迟渲染器。它使用 800x600 固定分辨率，
支持 64 个点光源和 8 个方向光，并把当前 G-Buffer、光源状态和最终光照结果写入
带版本号与校验和的二进制快照。有效快照可在不重绘几何的情况下恢复光照；动态增删
光源后会生成不覆盖旧文件的新版本快照。

## 功能

- 三张颜色附件 G-Buffer：
  - `GL_RGBA32F` 世界坐标
  - `GL_RGBA16F` 世界法线
  - `GL_RGBA8` 反照率 + 高光参数
  - 独立 `GL_DEPTH_COMPONENT32F` 深度纹理
- 全屏延迟光照：最多 96 个点光、16 个方向光，样本场景使用 64 + 8。
- 样本几何由本地 JSON 确定性生成，不依赖外部模型或网络资源。
- PNG 输出、实时/快照恢复逐像素比较。
- 二进制快照：格式版本、光源版本、文件长度、payload FNV-1a 校验和、G-Buffer、
  光源、相机位置、环境光和最终 RGBA8 结果。
- 损坏、截断、版本不匹配都会拒绝加载并自动回退到实时路径，错误以 JSON 写到
  stderr，包含业务错误码和版本字段。
- 恢复后增删光源直接使用已恢复的 G-Buffer 重算光照，不重绘几何。每次修改增加
  `light_version`，并写入 `.vN.dgsnap` 新文件；旧版本原样保留。
- `GlTexture`、`GlFramebuffer`、`GlBuffer`、`GlVertexArray`、`GlQuery`、
  `GlProgram` 和临时文件清理均为 RAII。
- 每帧向 stdout 输出 JSON 计时，并输出汇总中位数；区分 `realtime`、
  `realtime_fallback` 和 `snapshot_restore`。

## 目录

```text
include/                 头文件
src/                     C++ 实现
third_party/glad/        本地生成的 OpenGL 3.3 Core loader
third_party/nlohmann/    JSON 单头文件
tools/generate_scene.py  确定性场景生成器
tools/validate.py        端到端验收脚本
tools/lsan.suppressions  第三方图形栈的 LSan 抑制
sample_scene.json        已生成的确定性场景
validation/report.json   最近一次实际验收结果
```

## 构建

`vcpkg.json` 声明 `glfw3`、`glad`、`libpng`、`nlohmann-json`。Windows 上可用
Visual Studio 2022 的 x64 工具链：

```powershell
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
C:\vcpkg\vcpkg.exe install --triplet x64-windows

cmake -S . -B build `
  -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake `
  -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build build --config Release
```

Linux 上可使用同一 vcpkg 流程，或安装系统开发包：

```bash
sudo apt install cmake g++ libglfw3-dev libgl1-mesa-dev libpng-dev zlib1g-dev
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

仓库已固定 GLAD 生成代码，因此正常构建不需要在配置阶段联网。CMake 若检测到
项目根目录下存在 `.local-deps/usr`，会自动使用其中的头文件和库；该目录仅用于本次
无 sudo 环境的本地实测，不属于交付源码。

### AddressSanitizer

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON
cmake --build build-asan -j

ASAN_OPTIONS=detect_leaks=1 \
LSAN_OPTIONS=suppressions="$PWD/tools/lsan.suppressions" \
./build-asan/deferred_renderer --snapshot-load \
  --snapshot snapshots/lighting.v1.dgsnap \
  --output artifacts/asan_restore.png
```

当前 Mesa/llvmpipe + X11 环境会在进程退出时保留 fontconfig 和 LLVM JIT 的全局缓存。
`tools/lsan.suppressions` 只抑制这些第三方栈帧；快照解析、IO 和全部 GL RAII 对象不在
抑制范围内。未抑制运行仍会显示这些第三方缓存，可用作对照。

## 运行

```bash
# 实时渲染
./build/deferred_renderer --render \
  --scene sample_scene.json \
  --output artifacts/live.png \
  --frames 7 --warmup 2

# 实时渲染并保存 v1 快照
./build/deferred_renderer --snapshot-save \
  --scene sample_scene.json \
  --snapshot snapshots/demo \
  --output artifacts/live.png

# 恢复快照并和实时图像比较
./build/deferred_renderer --snapshot-load \
  --scene sample_scene.json \
  --snapshot snapshots/demo.v1.dgsnap \
  --output artifacts/restored.png \
  --compare artifacts/live.png

# 在快照上新增点光源，写出 v2
./build/deferred_renderer --snapshot-load \
  --snapshot snapshots/demo.v1.dgsnap \
  --add-light 'p64:point:0,3.2,0:1.0,0.35,0.15:4.0:6.0' \
  --output artifacts/add_light.png

# 在快照上删除光源，写出下一个版本
./build/deferred_renderer --snapshot-load \
  --snapshot snapshots/demo.v2.dgsnap \
  --remove-light p64 \
  --output artifacts/remove_light.png
```

`--add-light` 格式：

```text
点光：id:point:x,y,z:r,g,b:intensity[:radius]
方向光：id:directional:dx,dy,dz:r,g,b:intensity
```

退出码为 `0` 成功、`1` 运行时/校验失败、`2` 参数错误。`--help` 打印完整用法。

## 快照格式

所有整数和浮点位模式均为小端。固定头长为 160 字节：

| 偏移 | 大小 | 字段 |
|---:|---:|---|
| 0 | 8 | magic `DGSNAP01` |
| 8 | 4 | `format_version`，当前为 1 |
| 12 | 4 | `header_size`，当前为 160 |
| 16 | 8 | 完整文件字节数 |
| 24 | 8 | `light_version` |
| 32 | 4 | width |
| 36 | 4 | height |
| 40 | 4 | 点光源数 |
| 44 | 4 | 方向光数 |
| 48 | 4 | 颜色 G-Buffer 附件数，固定 3 |
| 52 | 4 | flags |
| 56 | 8 | payload 字节数 |
| 64 | 8 | payload FNV-1a 64 位校验和 |
| 72 | 32 | scene id，NUL 结尾 |
| 104 | 12 | 相机位置 |
| 116 | 4 | 曝光 |
| 128 | 12 | 环境光 |

payload 顺序为：可变长光源记录、world `RGBA32F`、normal `RGBA16F`（以 float 序列化）、
albedo/specular `RGBA8`、depth float、最终结果 `RGBA8`（自上而下）。加载器在创建任何
GPU 资源前完成长度、类型、维度、版本和校验和验证。

错误码：

- `ERR_SNAPSHOT_NOT_FOUND`
- `ERR_SNAPSHOT_READ`
- `ERR_SNAPSHOT_CORRUPT`
- `ERR_SNAPSHOT_VERSION`
- `ERR_SNAPSHOT_INVALID_GBUFFER`
- `ERR_SNAPSHOT_EXISTS`
- `ERR_SNAPSHOT_WRITE`

## 验证

```bash
python3 tools/validate.py
```

脚本会重新生成隔离的实时图、v1 快照、动态光源 v2/v3、损坏样本，执行性能对比和
ASan 案例，并把结果写到 `validation/report.json` 与 `validation/report.md`。

本次在 `llvmpipe (LLVM 21.1.8), Mesa 26.0.8`、800x600、72 光源、7 个测量帧和
2 个预热帧下的实际结果：

| 指标 | 实时路径 | 快照恢复 | 比例 |
|---|---:|---:|---:|
| CPU 中位数 | 110.209 ms | 0.891 ms | 0.81% |
| GPU 查询中位数 | 117.796 ms | 8.923 ms | 7.58% |
| 图像最大通道差 | - | 0/255 | 通过 |

其他实际检查：

- 快照恢复与实时图像：`max_channel_difference = 0`，不同像素 0。
- `v1 -> v2` 新增 p64 后为 65 个点光，`v1` SHA-256 不变。
- `v2 -> v3` 删除 p64 后回到 64 个点光，图像与原始实时图差 0，`v2` SHA-256 不变。
- payload 翻转：`ERR_SNAPSHOT_CORRUPT`，回退图像差 0，JSON 含格式/光源版本。
- 文件截断：`ERR_SNAPSHOT_CORRUPT`，回退图像差 0。
- 格式版本改为 2：`ERR_SNAPSHOT_VERSION`，`format_version=2`，回退图像差 0。
- 不存在的快照：`ERR_SNAPSHOT_NOT_FOUND`，回退图像差 0。
- `--help` 返回 0；非法参数组合返回 2 并打印 `ERR_ARGUMENTS` 与使用说明。
- ASan：保存快照、有效恢复、损坏回退三个案例退出码均为 0，无 AddressSanitizer 或
  LeakSanitizer 错误。

软件光栅器的绝对耗时会随机器、Mesa/LLVM 版本和负载变化；比例阈值的验证逻辑已固化在
`tools/validate.py`，不依赖本次记录的具体数值。
