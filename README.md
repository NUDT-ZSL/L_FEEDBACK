# LF Layered Character Engine

Windows 下可直接构建运行的 OpenGL 4.3 角色渲染示例。工程从零生成确定性分层材质、光源配置和带蒙皮权重的胶囊体角色，不依赖外部模型或贴图；使用五附件 G-Buffer 做延迟光照，并将漫反射、GGX 镜面、布料 Charlie sheen 三层 BRDF 按逐像素权重混合。

## 已实现

- `lighting.json` 导入与严格校验：固定三层材质顺序、权重、四类纹理引用、点光/方向光/聚光、颜色/强度/范围/锥角/衰减、唯一 ID 和纹理存在性。
- OpenGL 延迟管线：世界坐标、世界法线、基础色、三层权重、粗糙度/金属度写入 5 个 MRT，再切回默认帧缓冲执行全屏光照。
- 三层 BRDF：`diffuse`、`ggx_specular`、`cloth_sheen`，CPU 参考实现与 GLSL 公式一致。
- 程序化角色：18 骨骼、15 段胶囊体、3135 顶点/5400 三角形；每个顶点带 2 个骨骼影响和归一化权重。
- 两套内置姿态关键帧 `rest`/`walk`；顶点着色器用线性混合蒙皮计算位置，并用同一蒙皮矩阵的逆转置变换法线。
- 版本化快照：当前格式 v2，保存 UTC 时间戳、灯光集、材质权重、动画时刻；v1 自动迁移并记录逐字段日志；新版本拒绝加载。
- 快捷键快照恢复：整体替换状态、暂停动画并立即重绘；自动化验证保存/恢复帧的 307200 个像素字节完全一致。
- 配置修改时间热重载：先解析校验到临时对象，成功才原子交换；失败保留旧状态并记录 `E_CONFIG_RELOAD_INVALID`。
- 固定种子样本：12 张 256x256 TGA（三层的 base color/normal/roughness/metallic）与 JSON 配置重复生成逐字节一致。
- 统一业务错误码和文件日志；纹理损坏时记录 `E_TEXTURE_LOAD` 并使用确定性后备纹理继续运行。

## Windows 构建

前置条件：

- Visual Studio 2022，安装“使用 C++ 的桌面开发”和 MSVC v143。
- CMake 3.25 或更高版本。
- vcpkg；`vcpkg.json` 将 GLFW/GLM/nlohmann-json 锁定到题目要求的版本。
- 支持 OpenGL 4.3 Core 的显卡驱动。

```powershell
$env:VCPKG_ROOT = "C:\src\vcpkg"

# 一次性安装 vcpkg；已有可跳过
git clone https://github.com/microsoft/vcpkg.git $env:VCPKG_ROOT
& "$env:VCPKG_ROOT\bootstrap-vcpkg.bat"

cmake --preset windows-x64-release
cmake --build --preset windows-x64-release
ctest --preset windows-x64-release
```

manifest 固定基线为 `2023.08.09`，并显式覆盖：

- `glfw3 3.3.8`
- `glm 0.9.9.8`
- `nlohmann-json 3.11.2`

也可直接运行：

```powershell
.\scripts\build_windows.ps1 -Verify
```

程序不引入 GLAD/GLEW；所需 OpenGL 4.3 入口由 `glfwGetProcAddress` 解析，Windows 链接系统 `opengl32.lib`。

## 运行

```powershell
.\build\windows-x64\bin\lf_character_engine.exe --data-dir .\lf_runtime
```

首次运行在数据目录生成：

```text
lf_runtime/
  lighting.json
  textures/{diffuse,ggx,cloth}_{basecolor,normal,roughness,metallic}.tga
  snapshots/light_snapshot.json   # 手动保存后出现
  logs/lf_engine.log
```

快捷键：

- `F5`：保存当前灯光集、材质权重和动画时刻。
- `F9`：加载并整体替换状态；恢复后自动暂停在当前保存时刻。
- `F6`：暂停/继续固定步长动画。
- `R`：强制立即重载配置。
- `Esc`：正常关闭窗口。

其他参数：

```text
--data-dir PATH
--shader-dir PATH
--width N --height N
--frames N
--run-seconds S
--verify
```

`--verify` 创建隐藏窗口，真实渲染并读取默认帧缓冲，执行快照逐像素一致性、三层 BRDF 图像差异和三类光源检查。结果写入 `<data-dir>/artifacts/verification.json`，同时生成 PNG/BMP 帧。

## 配置格式

完整示例由首次运行生成。核心结构如下：

```json
{
  "formatVersion": 1,
  "material": {
    "name": "deterministic_layered_character",
    "layers": [
      {
        "type": "diffuse",
        "weight": 0.55,
        "baseColorTexture": "textures/diffuse_basecolor.tga",
        "normalTexture": "textures/diffuse_normal.tga",
        "roughnessTexture": "textures/diffuse_roughness.tga",
        "metallicTexture": "textures/diffuse_metallic.tga"
      },
      { "type": "ggx_specular", "weight": 0.28, "...": "..." },
      { "type": "cloth_sheen", "weight": 0.17, "...": "..." }
    ]
  },
  "lights": [
    {
      "id": "warm_key",
      "type": "point",
      "enabled": true,
      "position": [2.6, 3.7, 2.9],
      "color": [1.0, 0.73, 0.46],
      "intensity": 44.0,
      "range": 12.0,
      "constantAttenuation": 1.0,
      "linearAttenuation": 0.09,
      "quadraticAttenuation": 0.032
    }
  ]
}
```

校验失败时初载返回 `E_CONFIG_INVALID`，此时保留错误文件并使用内存中的确定性回退配置；运行中热重载失败返回 `E_CONFIG_RELOAD_INVALID`，当前有效配置、灯光集和材质不会被覆盖。

## 快照格式

当前 `formatVersion = 2`：

```json
{
  "formatVersion": 2,
  "savedAt": "2026-11-14T21:07:53Z",
  "materialWeights": [0.55, 0.28, 0.17],
  "renderState": { "animationTime": 0.42 },
  "lights": [{ "id": "warm_key", "type": "point", "...": "..." }]
}
```

迁移规则：

- v1 -> v2：缺少的 `materialWeights` 使用 `[0.55, 0.28, 0.17]`。
- v1 -> v2：缺少的 `renderState.animationTime` 使用 `0`。
- v1 -> v2：灯光缺少 `enabled` 时使用 `true`。
- v1 -> v2：点光/聚光缺少 `range`、`constantAttenuation`、`linearAttenuation`、`quadraticAttenuation` 时分别使用 `10`、`1`、`0`、`0`。
- 每次迁移逐条写入日志；迁移后的内存结构统一为 v2。
- `formatVersion > 2` 直接返回 `E_SNAPSHOT_VERSION_UNSUPPORTED`；版本小于 1、JSON 损坏或语义字段非法返回 `E_SNAPSHOT_CORRUPT`。

保存采用临时文件加原子替换，避免进程中断留下半写文件。

## 业务错误码

| 错误码 | 数值 | 处理 |
|---|---:|---|
| `E_RENDER_WINDOW_CREATE` | 1001 | 窗口/GLFW 创建失败，捕获后正常退出 |
| `E_RENDER_CONTEXT_CREATE` | 1002 | OpenGL 4.3 上下文不可用 |
| `E_RENDER_GL_LOAD` | 1003 | 必需 OpenGL 入口缺失 |
| `E_SHADER_COMPILE` | 1004 | GLSL 编译失败，输出编译日志 |
| `E_SHADER_LINK` | 1005 | 着色器链接失败 |
| `E_TEXTURE_LOAD` | 1006 | TGA 读取失败，使用后备纹理继续 |
| `E_FRAMEBUFFER_INCOMPLETE` | 1007 | G-Buffer 不完整 |
| `E_CONFIG_INVALID` | 1101 | 初载配置非法或回退配置非法 |
| `E_CONFIG_RELOAD_INVALID` | 1102 | 热重载非法，保留旧状态 |
| `E_SNAPSHOT_CORRUPT` | 1201 | 快照 JSON 或语义损坏 |
| `E_SNAPSHOT_VERSION_UNSUPPORTED` | 1202 | 快照版本高于当前支持版本 |
| `E_SNAPSHOT_IO` | 1203 | 快照写入/原子替换失败 |
| `E_RESOURCE_MISSING` | 1301 | 配置、快照或 shader 资源缺失 |
| `E_MESH_GENERATION` | 1302 | 角色参数或蒙皮生成失败 |
| `E_ARGUMENT` | 1401 | 命令行参数错误 |
| `E_INTERNAL` | 9000 | 未分类异常 |

所有错误同时写控制台与 `logs/lf_engine.log`。

## 渲染管线

1. 几何阶段写入 `GL_RGBA32F` 世界位置、`GL_RGBA16F` 世界法线、`GL_RGBA8` 基础色、`GL_RGBA16F` 三层权重、`GL_RG16F` 粗糙度/金属度。
2. 延迟阶段对每个像素归一化材质权重，按 `diffuse + GGX + Charlie sheen` 合成，再遍历点光、方向光、聚光。
3. 点光/聚光使用常数-线性-二次衰减和范围窗口；聚光额外使用内外锥 smoothstep。
4. 光照后执行 Reinhard 映射与 gamma 编码。
5. 默认帧缓冲绘制完成后，显式恢复 `glBindFramebuffer(0)`、关闭深度/裁剪/混合、重置 depth mask、纹理和 program 绑定。

## 验证

核心逻辑：

```powershell
ctest --preset windows-x64-release
```

真实 OpenGL 像素验收：

```powershell
.\build\windows-x64\bin\lf_character_engine.exe --verify `
  --data-dir .\artifacts\windows-verify --width 640 --height 480
```

本工作区的实际执行记录、环境、哈希和故障注入结果见 `docs/VALIDATION.md`。当前环境是 WSL2 + Mesa llvmpipe，已完成 GCC 构建和 OpenGL 4.5 Core 运行验证；本机没有 MSVC，因此 Windows/MSVC 构建路径通过固定 vcpkg manifest、Visual Studio 2022 preset 和平台条件代码交付，未声称已在原生 Windows 上执行。
