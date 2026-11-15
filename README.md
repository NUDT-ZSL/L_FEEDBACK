# 可重组离屏渲染器

一个无交互窗口的 OpenGL 3.3 命令行渲染器。场景由 64 位种子确定性生成，流水线由外部 JSON 驱动，依次执行 G-Buffer 几何、阴影、视图空间延迟 PBR 着色与可重排后处理链，最终写入 PNG。

## 已实现能力

- 固定版本依赖：GLFW 3.3.8、GLEW 2.2.0、stb_image_write 与 nlohmann/json 均在仓库内提供；也可切换为 vcpkg/预编译包。
- JSON 阶段表严格校验：`geometry`、`shadow`、`deferred`、`postprocess`，严格按数组顺序执行，未知阶段名直接报错。
- 延迟着色：G-Buffer 保存视图空间位置/法线、粗糙度、反照率与金属度；支持平行光、点光、聚光；无光源时仍输出环境光照。
- 独立后处理模块：Bloom、ACES/Reinhard 色调映射、FXAA、颜色分级；通过全分辨率 ping-pong 纹理传递，另含 Bloom 半分辨率临时资源。
- 事务式管线重组：先完整构建新 FBO/纹理/程序，成功后替换旧资源；失败时保留旧管线并统一走 RAII 清理。
- 确定性场景：SplitMix64 PRNG，固定网格生成；每帧包含 1 个平面、至少 1 个立方体和 1 个球体、8 组调色板材质，以及平行光/点光/聚光三类光源；固定相机与采样，不依赖时间、随机设备或 GPU 计时扩展。
- 状态保存/恢复：JSON 顶层的 `version` 字段会参与格式校验；阶段顺序、启用状态、后处理顺序与参数完整序列化。
- 单一 JSON 错误输出：上下文、配置、阶段名、着色器、FBO、I/O、版本与状态恢复错误均有稳定错误类型和非零退出码。

## 目录

```text
CMakeLists.txt
src/                 renderer, pipeline, scene, post effects, GL RAII, CLI
shaders/             runtime-loadable GLSL 330 shaders
configs/             基础、重组后与最小管线配置
include/             stb_image_write.h, nlohmann/json.hpp
third_party/         pinned GLFW 3.3.8, GLEW 2.2.0, GL headers
tests/               acceptance.sh 与 PNG 环境亮度检查
docs/                设计与实际验证记录
```

## 构建

### Linux / WSL2 / 无显示环境（OSMesa）

本工作区使用的 OSMesa 预编译运行库位于 `/home/liang/deps/osmesa-prefix/usr`：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DOSMESA_ROOT=/home/liang/deps/osmesa-prefix/usr
cmake --build build -j 4
```

如系统已安装 OSMesa 开发包，可省略 `OSMESA_ROOT`。GLFW 的 OSMesa backend 不创建可见窗口，最终图像从自建 FBO 读取。

本机若 `ctest` 没有加入 `PATH`，可使用：

```bash
/home/liang/tools/cmake-3.30.5-linux-x86_64/bin/ctest \
  --test-dir build --output-on-failure
```

### Windows / Visual Studio 2022

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

Windows 下使用 GLFW 隐藏窗口创建 OpenGL 3.3 Core 上下文；所有渲染目标仍是离屏 FBO。

### 使用外部预编译依赖

仓库默认 `SCENE_USE_VENDORED_DEPS=ON`。如需 vcpkg 或既有 GLFW/GLEW：

```bash
cmake -S . -B build -DSCENE_USE_VENDORED_DEPS=OFF \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg.cmake
```

CMake 会查找 `glfw3` 与 `GLEW 2.2`；也可通过 `CMAKE_PREFIX_PATH` 指向预编译安装目录。

## 运行

```bash
./build/offscreen_renderer \
  --seed 1337 \
  --pipeline configs/pipeline.json \
  --output out.png \
  --width 960 --height 540 \
  --warmup 2 \
  --shaders shaders
```

主要参数：

| 参数 | 说明 |
|---|---|
| `--seed N` | 必填，无符号 64 位场景种子 |
| `--pipeline FILE` | 必填，管线 JSON |
| `--output FILE` | 必填，最终 PNG |
| `--width/--height` | 分辨率，默认 960x540 |
| `--warmup N` | 预热帧数，默认 2；预热后才计时 |
| `--shaders DIR` | GLSL 文件目录 |
| `--save-state FILE` | 保存当前管线状态 |
| `--state-roundtrip FILE` | 同进程保存、重载、重绘并逐字节比较像素 |
| `--reconfigure FILE --reconfigure-output FILE` | 同进程切换管线并输出第二张 PNG |
| `--no-lights` | 移除种子生成的光源，用于环境项验证 |

标准输出是单个 JSON 对象，包含 `total_render_ms`、顺序化的 `stages_ms` 与 `post_effects_ms`、准备/呈现/回读耗时和上下文信息。每个阶段边界与回读后均调用 `glFinish()`，计时不依赖 GPU timer query。

## 管线 JSON

```json
{
  "version": 1,
  "clear_color": [0.012, 0.018, 0.030],
  "stages": [
    {"name": "geometry", "enabled": true},
    {"name": "shadow", "enabled": true, "resolution": 1024,
     "bias": 0.0012, "strength": 0.88},
    {"name": "deferred", "enabled": true,
     "ambient": [0.075, 0.085, 0.110]},
    {"name": "postprocess", "enabled": true, "effects": [
      {"type": "bloom", "enabled": true, "threshold": 0.9,
       "intensity": 0.65, "radius": 1.0, "knee": 0.35},
      {"type": "tonemap", "enabled": true, "mode": "aces", "exposure": 1.0},
      {"type": "fxaa", "enabled": true, "edge_threshold": 0.08},
      {"type": "color_grade", "enabled": true, "contrast": 1.04,
       "saturation": 1.06, "tint": [1.02, 1.0, 0.98],
       "lift": [0.0, 0.0, 0.0]}
    ]}
  ]
}
```

规则：

- 顶层只接受 `version`、`clear_color`、`stages`；当前版本仅支持 `1`。
- 主阶段名必须精确匹配且不能重复；未知名称抛出 `unknown_stage`。
- 未知阶段字段、后处理类型或参数都会报 `config_error` / `unknown_effect`，避免拼写错误被静默忽略。
- `enabled: false` 的主阶段保留在状态中但不执行，不创建该阶段独占资源。
- 后处理数组顺序就是执行顺序；禁用的效果仍保留参数并可被状态序列化。

## 状态恢复与动态重组

保存文件与管线配置使用同一版本化 schema。`--state-roundtrip` 会在同一进程内：

1. 保存当前管线；
2. 从磁盘重新解析；
3. 事务式替换全部依赖资源；
4. 预热后重绘；
5. 对原始 RGBA 像素逐字节比较，不一致则返回 `state_restore_mismatch`。

`--reconfigure` 可先渲染基础配置，再在同一进程应用另一阶段集合/效果顺序，验证资源释放、重建和参数更新。

## 错误协议

所有失败路径仅向 `stderr` 写一行 JSON：

```json
{"exit_code":6,"message":"...geometry.frag (fragment) compilation failed...","type":"shader_compile"}
```

退出码：CLI=2、配置=3、未知阶段=4、未知效果=5、着色器=6、FBO 不完整=7、上下文=8、I/O=9、版本=10、状态不一致=11、GL 运行时=12、内部=13。

## 验证

```bash
./tests/acceptance.sh ./build/offscreen_renderer "$PWD" "$PWD/artifacts/acceptance"
# 或
ctest --test-dir build --output-on-failure
```

测试覆盖跨进程确定性、直接加载状态文件、同进程状态往返、同进程重组、未知阶段、版本拒绝、无光源环境项和非法着色器 JSON/退出码。实际哈希、耗时与错误样本见 `docs/VALIDATION.md`。

## 说明

- 当前阴影阶段使用首盏平行光的单张 2D 阴影图与 3x3 PCF；点光和聚光无独立阴影图。
- 未启用 MSAA；FXAA 是配置链中的独立效果，而非固定末尾阶段。
- 如果 `deferred` 被禁用但 `geometry` 和 `postprocess` 启用，几何阶段会额外解析 G-Buffer 反照率作为稳定后处理输入；该行为不改变配置阶段顺序。
