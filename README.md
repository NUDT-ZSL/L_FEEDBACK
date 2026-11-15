# Catmull-Clark Offline Subdivision Preview

一个面向图形工程师的离线细分预览工具。程序内置确定性低多边形三角网格构造器，
在 CPU 上执行 Catmull-Clark 细分，把结果上传到 OpenGL 4.1 Core GPU 管线，并在
片元着色器中完成视线步进表面着色和法线可视化。运行时不读取模型、贴图或网络资源，
唯一的外部输入是 JSON 场景参数文件。

## 功能

- 确定性构造 7 x 5 控制点、48 个三角形的半封闭曲面，包含开放边界和逐边折痕。
- 完整 Catmull-Clark 迭代：三角形拆成三个四边形，后续四边形拆成四个。
- 支持软折痕、锐折痕、边界维持，以及细分顶点到原始顶点的溯源映射。
- 容错 JSON 场景加载；缺字段、类型错误、越界数值均回退默认值并打印字段路径。
- 细分级别运行时切换；每次切换释放旧 VAO/VBO/EBO 后重建。
- 位置着色、独立法线模式、面/顶点法线切换、线框切换。
- 每次级别变化输出面数、高精度耗时、着色器状态、GPU 规模和 RSS。
- 上下文、着色器、资源上传和运行时 GL 错误均以明确 stderr 信息和非零码退出。

## Windows 构建

前提：

- Windows 10/11
- Visual Studio 2019 或更高版本，安装“使用 C++ 的桌面开发”
- CMake 3.20+
- GLFW 3.3.8+，推荐 vcpkg
- 支持 OpenGL 4.1+ 的显卡驱动

使用 vcpkg 的示例：

```powershell
vcpkg install glfw3:x64-windows

cmake -S . -B build -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake `
  -DSUBDIV_FETCH_GLFW=OFF

cmake --build build --config Release
.\build\Release\subdivision_preview.exe --scene scene.json
```

如果没有预装 GLFW，可保留默认的 `SUBDIV_FETCH_GLFW=ON`，CMake 会通过
`FetchContent` 获取并构建 GLFW 3.4。该网络访问只发生在构建阶段；程序运行时完全离线。

运行测试：

```powershell
ctest --test-dir build -C Release --output-on-failure
```

Linux 验证构建：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
./build/subdiv_tests
./build/subdivision_preview --scene scene.json
```

## 操作

| 按键 | 功能 |
| --- | --- |
| `Left` / `Down` | 降低一级细分 |
| `Right` / `Up` | 提高一级细分 |
| `0` - `6` | 直接切换级别，并受 `maxLevel` 限制 |
| `N` | 开关法线可视化 |
| `F` | 在顶点法线与面法线之间切换 |
| `W` | 开关线框 |
| `R` | 恢复场景初始相机 |
| `Esc` | 正常关闭窗口 |

窗口标题显示当前细分级别、法线模式和线框状态。每次级别变化都会在控制台输出：

```text
[stats] level=3 original_triangles=48 subdivided_quads=2304 subdivision_ms=2.768 \
shader=compiled+linked vertices=2385 gpu_triangles=4608 boundary_edges=160 \
crease_edges=264 traceable_vertices=2385 rss_mib=139.4
```

## 场景 JSON

`scene.json` 是完整示例，包含当前构造器的全部 82 条边折痕权重。

```json
{
  "seed": 20251029,
  "material": {
    "diffuse": [0.20, 0.52, 0.82],
    "roughness": 0.36
  },
  "subdivision": {
    "maxLevel": 4,
    "initialLevel": 1
  },
  "creaseWeights": {},
  "camera": {
    "position": [4.8, 3.6, 5.4],
    "target": [0.0, 0.55, 0.0],
    "up": [0.0, 1.0, 0.0],
    "fovDegrees": 42.0
  }
}
```

| 字段 | 范围 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `seed` | 非负整数 | `20251029` | 确定性构造器种子 |
| `material.diffuse` | 三个 `[0,1]` 数 | `[0.20,0.52,0.82]` | 漫反射颜色 |
| `material.roughness` | `[0,1]` | `0.36` | GGX 粗糙度 |
| `subdivision.maxLevel` | 整数 `[0,6]` | `4` | 细分级别上限 |
| `subdivision.initialLevel` | 整数 `[0,maxLevel]` | `1` | 初始级别 |
| `creaseWeights` | 数组或对象 | 构造器内置权重 | 逐边折痕权重 |
| `camera.position` | 分量 `[-1000,1000]` | `[4.8,3.6,5.4]` | 初始相机位置 |
| `camera.target` | 分量 `[-1000,1000]` | `[0,0.55,0]` | 观察目标 |
| `camera.up` | 非零有限向量 | `[0,1,0]` | 相机上方向 |
| `camera.fovDegrees` | `[20,100]` | `42` | 垂直视场角 |

兼容别名：`material.diffuseColor`、`maxSubdivision`、`initialSubdivision`、
`edgeCreaseWeights`、`camera.fov`。

`creaseWeights` 可使用以下两种写法：

```json
"creaseWeights": [0.0, 1.0, 0.42]
```

数组下标就是边编号。也可写成按边编号索引的对象：

```json
"creaseWeights": { "0": 0.0, "23": 0.42, "40": 1.0 }
```

边编号顺序固定为端点对 `(min(vertexA, vertexB), max(vertexA, vertexB))` 的
字典序。运行 `--list-edges` 可打印边编号、端点和内置权重。折痕权重语义：

- `1.0`：完全软边，使用普通 Catmull-Clark 平滑规则。
- `0.0`：完全锐边，边点取中点，相关顶点使用折痕/角点规则。
- `0 < w < 1`：在平滑位置与锐位置之间按 `w` 混合。
- 开放边界始终使用边界中点与 `(6P + b0 + b1) / 8` 规则。

缺失、类型错误、非有限值和越界值不会导致崩溃。控制台会打印类似：

```text
[scene] warning: field 'material.roughness' must be a finite number in [0, 1]; using default 0.360000
[scene] warning: field 'creaseWeights[4]' must be a finite number in [0, 1]; using built-in weight for this edge
```

## 实现说明

### 确定性构造

`src/mesh.cpp` 使用固定 7 x 5 网格、固定三角化顺序和基于 `splitmix64` 的整数散列
扰动。没有随机数引擎、时间输入、静态计数或全局可变几何状态。相同种子产生完全相同
的位置、索引、法线和折痕权重；控制台打印几何指纹便于跨进程比对。

### Catmull-Clark

一次迭代输出 `V + F + E` 个新顶点：

- 面点：多边形顶点平均。
- 软边点：`(v0 + v1 + f0 + f1) / 4`。
- 折痕边点：软边点与中点按权重混合；边界边固定为中点。
- 内部顶点：`(F + 2R + (n - 3)P) / n`。
- 边界顶点：`(6P + b0 + b1) / 8`。
- 两条折痕相关顶点：`(6P + c0 + c1) / 8`，再按锐度与平滑规则混合。
- 三条及以上锐折痕：向角点 `P` 混合。

三角形 `(v0,v1,v2)` 的每个角点生成一个四边形
`(vi, edgePoint(next), facePoint, edgePoint(previous))`，因此第一级严格产生
`3 * triangleCount` 个四边形；后续每级四倍。

每个新顶点保存：

- `representativeOriginalVertex`：用于快速追踪的代表原始顶点。
- `sourceOriginalVertices`：经排序去重的全部原始顶点来源。

`--trace-map FILE` 会把完整映射写为 TSV。

### GPU 与着色

CPU 四边形按扇形三角化后上传。顶点缓冲包含位置、面积加权顶点法线、面法线及原始
顶点编号；索引缓冲保存三角形。切换级别时先删除旧 EBO、VBO、VAO，再创建新资源。

片元着色器先在切空间内沿相机视线步进一个确定性的微观表面壳层，再做 GGX 直接光、
补光、环境项和色调映射。法线模式完全不进入位置着色分支，只将选定法线映射到
`normal * 0.5 + 0.5`；面法线模式还使用稳定的面编号做极轻的明暗编码，使高密度
四边形分面在 3 级以上仍可辨认。面法线来自控制网格 quadrilateral，顶点法线用于
观察细分后的平滑法线变化。

## 命令行与诊断

```text
--scene PATH          场景 JSON
--level N             覆盖初始级别
--normal-mode         启动时开启法线模式
--face-normal         法线模式使用面法线
--wireframe           启动线框
--frames N            渲染 N 帧后退出，便于自动化
--stress-switches N   自动来回切换 N 次，便于资源/GL 错误测试
--screenshot PATH     当前帧保存为二进制 PPM
--trace-map PATH      输出细分顶点溯源 TSV
--list-edges          输出确定性边编号与权重
```

退出码：

| 码 | 含义 |
| --- | --- |
| `0` | 正常关闭 |
| `1` | 命令行参数错误 |
| `2` | 图形显示预检或 GLFW 初始化失败 |
| `3` | 窗口或 OpenGL 上下文创建失败 |
| `4` | 必需 OpenGL 函数加载失败，或上下文低于 4.1 |
| `5` | 着色器编译或链接失败 |
| `6` | CPU 网格转换或 GPU 资源上传失败 |
| `7` | 运行时 OpenGL 错误或截图写入失败 |

所有致命错误写入 stderr，不会以空白窗口继续运行。

## 测试与实测

- `tests/core_tests.cpp` 覆盖构造确定性、拓扑数、边界/折痕规则、溯源映射、JSON
  容错和默认值。
- 本次实际构建、渲染、1000 次资源切换、内存采样的结果见
  [`VALIDATION.md`](VALIDATION.md)。
- 验收截图位于 `artifacts/`：未细分、细分后位置着色、顶点法线、面法线。
