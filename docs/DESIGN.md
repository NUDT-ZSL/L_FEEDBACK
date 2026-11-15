# 设计说明

## 执行模型

`PipelineConfig` 保存主阶段数组、阴影/延迟参数和后处理数组。`Renderer::renderFrame()` 只遍历主阶段数组一次，不按硬编码顺序重排。每次 `applyPipeline()` 完成后再由 CLI 执行预热帧，因此资源切换后不会读取新分配但未初始化的纹理。

主阶段语义：

| 阶段 | 输出/依赖 |
|---|---|
| `geometry` | 写入位置、法线+粗糙度、反照率+金属度 MRT 与深度；若最终无延迟阶段，则解析一个稳定反照率预览到 scene color |
| `shadow` | 渲染首盏平行光深度图；未执行时延迟着色使用已初始化的 1x1 深度纹理并关闭阴影采样 |
| `deferred` | 从 G-Buffer 与阴影图生成 RGBA16F HDR scene color |
| `postprocess` | 按数组顺序执行启用的效果；ping-pong RGBA16F 中间纹理，结束后回写 scene color |

最终 `present` pass 是固定的像素输出步骤，不属于可配置渲染阶段：它把线性 HDR 结果钳制并编码到 sRGB RGBA8 最终纹理，再由 `glReadPixels` 交给 stb。

## G-Buffer

| 附件 | 格式 | 内容 |
|---|---|---|
| 0 | RGBA16F | 视图空间位置；`w=1` 表示有几何 |
| 1 | RGBA16F | 视图空间法线；`w=粗糙度` |
| 2 | RGBA8 | 线性反照率；`a=金属度` |
| 深度 | DEPTH_COMPONENT24 | 深度测试 |

`prepareFrame()` 每帧在绑定 G-Buffer FBO 后清空三个颜色附件与深度。即使配置中禁用几何阶段但启用延迟阶段，延迟着色读取到的也是已定义背景状态。

## 延迟 PBR

- CPU 将光源位置和方向变换到视图空间；着色器也在视图空间计算 `N`、`V`、`L`、`H`。
- 平行光使用常量辐照度，点光使用平方衰减近似，聚光再乘内外锥角 `smoothstep`。
- 材质从 G-Buffer 读取，Cook-Torrance 近似包含 GGX 法线分布、Smith 可见性、Schlick Fresnel。
- 环境项由反照率漫反射项与 `F0` 环境高光项组成，因此 `--no-lights` 仍能看到材质。
- 阴影坐标由 `uInvView` 恢复世界位置，再乘以平行光 view-projection；3x3 PCF 与配置 bias/strength 参与计算。

## 后处理资源

- 任意启用效果存在时分配两张全分辨率 RGBA16F ping/pong 纹理及 FBO。
- Bloom 独立持有半分辨率 A/B 纹理：阈值降采样、水平高斯、垂直高斯、原场景加泛光合成。
- Tonemap、FXAA、Color Grade 各自持有独立 shader program；没有共享可变全局效果状态。
- 所有效果都消费“上一效果输出纹理”并写入“下一 FBO”，不会形成读写同一纹理的反馈环。

## 资源生命周期

`buildResources()` 先构造临时 `PipelineResources`，只按启用阶段创建资源：

- scene color、final RGBA8、present program 是固定输出依赖。
- G-Buffer 仅在几何或延迟阶段启用时创建。
- 阴影深度 FBO 仅在 `shadow.enabled=true` 时创建。
- 后处理 ping/pong 仅在 `postprocess.enabled=true` 且至少一个效果启用时创建。
- Bloom 半分辨率资源仅在该效果启用时创建。

构建任一步（着色器编译、FBO 完整性、GL 分配）失败都会销毁临时对象，旧管线仍保持有效。构建全部成功后一次替换 `std::unique_ptr<PipelineResources>`，旧资源在析构中释放。所有 GL 对象使用 RAII，Renderer 在上下文销毁前析构。

## 确定性

1. 场景只由 SplitMix64 生成器与种子派生；每个随机值通过高位整数缩放，不使用实现相关的 `std::uniform_*`。
2. 网格拓扑、相机、曝光、采样核、阴影 bias 和效果卷积权重固定。
3. 渲染不使用时间、帧计数或动态分辨率。
4. 预热帧只用于稳定驱动状态，不参与最终图像内容。
5. stb 写出前统一垂直翻转，PNG 字节在相同二进制/驱动/分辨率下可重复。

## 状态 schema

状态文件就是完整的 `PipelineConfig` JSON：

- `version`: 格式版本，当前必须为 1。
- `clear_color`: 背景色。
- `stages`: 顺序化主阶段及启用标志；`shadow`、`deferred`、`postprocess` 携带参数。
- `postprocess.effects`: 顺序化效果数组，保存类型、启用标志和完整参数默认值合并结果。

恢复路径不信任文件中的省略参数：解析时重新填充默认值并做范围校验，然后重新构建所有资源。

## 错误与清理

`AppError` 携带错误类别与退出码。`main()` 将图形对象放在内层作用域；异常离开该作用域时 Renderer、Scene、Context 依次析构，之后 catch 分支才写单行 JSON。GLFW 错误回调只记录文本，不直接打印，保证 stderr 只有一个 JSON 对象。

## 平台策略

- Linux/Unix：若找到 OSMesa，则 GLFW 3.3.8 使用 Null + OSMesa backend，无需 X11 或可见窗口。
- 未找到 OSMesa：使用本机 GLFW 平台后端（通常 X11），创建隐藏窗口上下文。
- Windows：隐藏 Win32 窗口 + WGL OpenGL 3.3 Core。
- 所有实际颜色/深度渲染都在项目自建 FBO，上下文窗口尺寸固定为 1x1。

## 已知边界

- 单张平行光阴影图；点光/聚光目前只参与直接光照。
- 没有 HDR 输出、MSAA、TAA 或 GPU timestamp。
- 后处理顺序完全由用户配置决定，不自动移动 FXAA 到色调映射之后。

