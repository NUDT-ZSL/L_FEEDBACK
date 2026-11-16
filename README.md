# Skin Material Batch Renderer

离线 OpenGL 4.3+ 皮肤材质批量渲染器。程序为每个 JSON 材质组执行一次独立渲染过程，绘制两个具有不同粗糙度/各向异性参数的 UV 球体，并输出确定性的 RGB8 PPM 图像。

## 已实现功能

- GPU 启动预计算 RGB16F 皮肤反射剖面 LUT，参数来自 JSON；曲面着色器按 `N·L` 与曲率采样同一 LUT。
- 各向异性 GGX 高光；切线由片元着色器中的世界位置和 UV 导数计算。
- 矩形面光源确定性采样、半球环境光、固定 ACES/Reinhard/无色调映射模式。
- 多组材质批量渲染，组间复用同尺寸/格式的 FBO、纹理和 shader program；材质切换只更新 uniform。
- 非法材质与 OpenGL 错误按组隔离，错误组不写图像，后续组继续执行。
- 原子 PPM 写入与同名序号去重：`name.ppm`、`name_001.ppm`、`name_002.ppm`，不覆盖已有文件。
- 可选 PPM 基准比较，记录 MSE、平均绝对差、最大绝对差、图像指纹和统计 schema 版本。
- 所有警告、错误和统计写入运行目录的 `render.log`；控制台只输出成功数、失败数和退出码。

详细设计见 `docs/TECHNICAL.md`，本机验证记录见 `docs/VALIDATION.md`。

## 构建

### Windows / Visual Studio 2022

安装 Visual Studio 2022 的 “使用 C++ 的桌面开发” 工作负载，并安装 vcpkg。项目提供 `vcpkg.json` manifest，依赖为 `glfw3`、`glm`、`stb`、`nlohmann-json`。

```powershell
git clone https://github.com/microsoft/vcpkg C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat

cmake -S . -B build `
  -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake `
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

vcpkg manifest 模式会自动安装 `vcpkg.json` 中的依赖。Windows 上使用隐藏 GLFW 窗口创建 OpenGL 4.3 core context，渲染仍全部发生在线程绑定的离屏 FBO 中。

### Linux / WSL 无窗口

Mesa EGL 路径不依赖 X11/Wayland。无 GPU 的机器可使用 llvmpipe：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSKINRENDER_USE_GLFW=OFF \
  -DSKINRENDER_BUILD_TESTS=ON
cmake --build build --parallel

LIBGL_ALWAYS_SOFTWARE=1 ./build/skin_renderer --config configs/demo.json
```

CMake 优先使用 vcpkg/系统提供的依赖；仓库内的 `third_party/` 提供离线回退头和 GLAD loader。Linux 构建需要 `libEGL` 运行库，或启用 GLFW 后端。

运行测试：

```bash
python3 scripts/validate.py --exe build/skin_renderer
# 或
scripts/build_and_test.sh
```

## 使用

```text
skin_renderer --config <config.json>
              [--output-dir <dir>]
              [--log <path>]
              [--backend auto|egl|glfw]
```

- `--output-dir` 覆盖 JSON 中的 `output_dir`。
- 相对路径规则：JSON 中的 `output_dir` 和组 `baseline` 相对于配置文件目录解析；`render.log` 和 `diff_stats.json` 写入进程当前工作目录。
- 正常运行时控制台示例：`success=2 failed=0 exit=0`。

退出码：

- `0`：全部材质组成功。
- `1`：至少一个材质组失败，但批量流程完成。
- `2`：配置、输出目录、OpenGL context 或渲染管线初始化失败。

## 配置

最小材质组只要求 `name`、`albedo`、`roughness`、`anisotropy`。完整示例见 `configs/demo.json`。

```json
{
  "width": 512,
  "height": 512,
  "seed": 12345,
  "output_dir": "../output",
  "sss_lut": {
    "profile_samples": 128,
    "curvature_samples": 64,
    "scattering_samples": 24,
    "curvature_min": 0.0,
    "curvature_max": 0.12,
    "scatter_color": [1.0, 0.70, 0.58],
    "scatter_radius": [0.16, 0.09, 0.055],
    "wrap": 0.35,
    "profile_power": 1.35
  },
  "tone_mapping": {
    "mode": "aces",
    "exposure": 1.08,
    "gamma": 2.2
  },
  "materials": [
    {
      "name": "skin_satin",
      "albedo": [0.78, 0.38, 0.30],
      "roughness": 0.30,
      "anisotropy": 0.65,
      "anisotropy_rotation": 0.2,
      "sss": {
        "strength": 0.95,
        "curvature": 0.035,
        "color": [1.0, 0.72, 0.64]
      },
      "specular_f0": 0.028,
      "baseline": "../baselines/skin_satin.ppm",
      "baseline_version": "v1"
    }
  ]
}
```

关键约束：

- `width`/`height` 缺省为 `512x512`，范围为 `1..8192`。
- `roughness` 必须在 `(0, 1]`；`anisotropy` 必须在 `[0, 1]`。
- `profile_samples`、`curvature_samples` 范围为 `8..4096`，`scattering_samples` 范围为 `1..256`，且 `curvature_max > curvature_min`。
- 材质 `sss` 可写为嵌套对象，也可使用扁平的 `sss_strength`、`sss_curvature`、`sss_color`。
- 每个材质组可写 `reference_material` 覆盖第二颗球；也可用 `spheres: [materialA, materialB]` 显式提供两颗球的材质。
- 基准文件不存在、无法解码或尺寸不一致时只记录警告，不影响该组成功状态。

## 输出

PPM 使用最小二进制头：`P6\n<width> <height>\n255\n`，随后是自上而下的 RGB8 像素，无额外字符。写文件采用同目录临时文件 + rename，只有完整成功后才出现最终文件。

`diff_stats.json` 是版本化统计输出，当前 `stats_schema_version = 1`，包含：

- 渲染器版本、配置指纹、OpenGL 后端、运行维度、seed、成功/失败/退出码。
- 每组输出路径、渲染图像 FNV-1a 指纹、基准状态和可选的 `comparison` 统计。
- 失败组错误码和组名。

## 实际验证

验证环境：WSL2 Ubuntu 26.04，Mesa llvmpipe，OpenGL 4.5 Core，GCC 15.2，CMake 3.30.5。

```text
PASS deterministic PPM output and resource reuse
PASS versioned baseline comparison statistics
PASS collision-safe PPM naming
PASS invalid-group isolation and recovery
PASS initialization failure exit code 2
PASS default 512x512 dimensions
validation: ALL PASSED
```

样例结果：

- `configs/demo.json`：`success=2 failed=0 exit=0`；两次运行 PPM SHA-256 分别为
  `b67478f48d0bf6b063c0ec04426271e1f915bc0818ca602b6a5dd088511cd252`（`skin_satin`）
  和 `fed5ca2e95ec97058c2a3272537eb30e9eaf42b14416c11d048f24d6a859d411`（`skin_rough`）。
- 基准自比较：`mse=0`、`mean_abs=0`、`max_abs=0`。
- 资源释放计数：FBO `2/2`、纹理 `2/2`、program `2/2`、VAO `2/2`、buffer `2/2`，全部 `balanced=yes`。
- `configs/failure_demo.json`：`success=2 failed=2 exit=1`；负粗糙度组码 `111`、越界各向异性组码 `112`；后续组仍输出，且同名文件生成 `valid_duplicate.ppm` 与 `valid_duplicate_001.ppm`。
- `configs/init_failure.json`：`success=0 failed=0 exit=2`，不生成 PPM。
