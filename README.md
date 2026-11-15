# Vulkan Volume Shadow Renderer

Windows 本地离线渲染器。读取固定 schema 的场景 JSON 与 32x32x32 float32
噪声 `.raw`，用 Vulkan 1.3 图形/计算管线完成方向光阴影、点光六面阴影、
3D 噪声扰动的体积积分、G-buffer 深度截止、PCF 软阴影、Reinhard 色调映射和
sRGB PNG 输出。

仓库包含确定性样本生成器、完整实现、CTest 验收脚本和本机实际验证记录。

## 已验证结果

在 Ubuntu 26.04 + Mesa lavapipe（Vulkan API 1.4.335）上实际运行，启用
`VK_LAYER_KHRONOS_validation`：

- 两次独立执行输出 PNG SHA-256 均为
  `4b4a5e6082ddb6b4061aed6e98d1913d3b913e356cab5559ec5ad88705927c62`。
- 两次主帧的 PNG/像素哈希一致。
- 主输出相对固定参考帧的平均绝对 Rec.709 线性亮度误差为 `0.000278`
  （阈值 `0.02`）。
- 三组步长 `0.18/0.12/0.08 m` 的亮度误差为
  `0.00046063/0.00027783/0.00015525`，随步长减小单调下降。
- `E_SCHEMA/E_SHADER/E_ALLOC/E_DESC/E_DEVICE_LOST` 路径均实际触发；注入后
  `cleanup: resources_live=0 allocator_live=0`。
- 设备丢失恢复文件和下次启动提示已实际验证。
- `ctest` 验收测试通过。

详细记录见 `docs/VALIDATION.md`，样本结果见 `validation/`。

## Windows 构建

前置条件：

- Vulkan SDK 1.3 或更高版本，包含 `glslc`/`glslangValidator` 和
  `VK_LAYER_KHRONOS_validation`。
- CMake 3.20 或更高版本。
- Visual Studio 2022，安装“使用 C++ 的桌面开发”。
- 支持 Vulkan 1.3、图形和计算着色器的 GPU/驱动。
- 运行自动验收测试时需要 Python 3（渲染器本身不依赖 Python）。

PowerShell 示例：

```powershell
$env:VULKAN_SDK = "C:\VulkanSDK\1.4.341.0"
$env:Path = "$env:VULKAN_SDK\Bin;$env:Path"
$env:VK_LAYER_PATH = "$env:VULKAN_SDK\Bin"

cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config RelWithDebInfo
ctest --test-dir build -C RelWithDebInfo --output-on-failure
```

生成并运行样本：

```powershell
.\build\RelWithDebInfo\scene_gen.exe .\run\sample
.\build\RelWithDebInfo\volume_renderer.exe `
  --scene .\run\sample\scene.json `
  --out .\run\render.png `
  --stats .\run\stats.json `
  --reference-out .\run\reference.png `
  --recovery-dir .\run\recovery `
  --shader-dir .\build\shaders `
  --validation on
```

也可以运行 `scripts/run_sample.ps1`。

## Linux 参考验证

```bash
export VULKAN_SDK=/path/to/vulkan-sdk
export PATH="$VULKAN_SDK/bin:$PATH"
export VK_LAYER_PATH="$VULKAN_SDK/share/vulkan/explicit_layer.d"

cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  ctest --test-dir build --output-on-failure
```

## 文档

- `docs/SPEC.md`：单位、数值范围、业务错误码、恢复语义、参考亮度来源和
  管线细节。
- `docs/VALIDATION.md`：实际测试环境、命令、哈希和通过数据。
- `sample/scene.json`、`sample/noise.raw`：由内置生成器产出的确定性样本。
- `validation/render.png`、`validation/reference.png`、
  `validation/stats.json`、`validation/acceptance.log`：本次实测产物。

## 说明

图像哈希可复现性保证适用于固定可执行文件、同一 Vulkan 设备/驱动、相同
场景 JSON、相同 `.raw` 字节和相同命令行参数。不同 GPU 驱动可能因浮点实现
差异产生末位差异，这是 Vulkan 浮点渲染的固有边界；同机双帧和独立进程复现
由验收脚本强制检查。
