# 技术说明

## 1. 执行环境

`deferred_validation.gl_context.EGLContext` 使用 EGL surfaceless 平台创建
OpenGL 4.5 Core Profile 上下文，不创建窗口或 EGL surface。上下文创建后读取
`GL_VERSION`、`GL_RENDERER`、`GL_SHADING_LANGUAGE_VERSION` 和
`GL_MAX_TEXTURE_SIZE`；版本低于 4.5 或没有可用 framebuffer config 时直接报错。

PyOpenGL 负责 GL 调用，PyGLM 负责相机视图/投影和层级变换。Mesa llvmpipe 是
本任务实际使用的软件后备实现，仍报告完整 OpenGL 4.5 Core 能力。

## 2. 场景模型

加载顺序：

1. 解析 JSON 并逐字段校验类型、范围、引用和路径。
2. 生成立方体/平面/球体程序化网格，或读取显式三角网格。
3. 校验索引范围；缺失法线时按面法线累积计算；按 UV 三角形计算平均切线与
   bitangent 方向符号。
4. 解析材质和可选法线贴图。内联贴图按 UNORM `[0,1]` 编码，文件贴图相对
   JSON 文件目录解析。
5. 递归求值节点世界矩阵，检测未知 parent 和环。
6. 变换所有网格顶点得到场景 AABB，供方向光正交阴影拟合和位置通道可视化。
7. 计算几何、材质、光照、渲染设置和总场景 SHA-256 校验和。

所有错误通过 `ConfigError` 输出到 stderr，CLI 退出码为 2。

## 3. 渲染通道

内部渲染尺寸为 `(width * ssaa, height * ssaa)`。

1. **阴影通道**：每个投影光源独立生成深度贴图。
   - 方向光：按场景包围球拟合正交投影。
   - 聚光灯：视锥 FOV 为 `2 * outer_deg + 1`，远平面为 `range`。
   - 点光：GL 立方体深度贴图六个面，90° 投影，远平面为 `range`。
   - 深度 pass 开启 polygon offset，且不写颜色。
2. **G-buffer 通道**：MRT 同时写 4 个附件。
3. **直接光照通道**：每个光源用一个全屏三角形 pass，使用 `ONE, ONE` 混合叠加。
   每个 pass 单独绑定 2D 或 cube 阴影采样器，避免动态 sampler 数组。
4. **间接光照通道**：半球环境探针对法线方向插值天空/地面辐照度，并加入近似
   环境镜面项。该通道可选，关闭时输出全零。
5. **合成通道**：`direct_linear + indirect_linear` 写入 RGBA32F HDR 目标。
6. **SSAA 降采样**：使用 `texelFetch` 对每个输出像素精确平均 `ssaa x ssaa`
   个内部像素，避免线性过滤只采 4 个样本。
7. **色调映射**：在同一降采样 pass 中应用曝光、ACES 或 Reinhard、gamma 2.2，
   写入 RGBA8 最终目标。

所有颜色中间目标为 RGBA32F；G-buffer 位置为 RGBA32F，法线和材质为 RGBA16F，
反照率为 RGBA8。

## 4. G-buffer 合约

唯一定义位于 `deferred_validation/gbuffer.py`：

| location | 附件 | 格式 | 语义 |
|---:|---|---|---|
| 0 | position | RGBA32F | world xyz + camera distance |
| 1 | normal | RGBA16F | world normal xyz + coverage |
| 2 | albedo | RGBA8 | linear RGB + material mask |
| 3 | material | RGBA16F | metallic + roughness + reserved + mask |

报告中的 `gbuffer_layout` 直接由该模块生成。渲染器使用同一列表创建纹理和
`glDrawBuffers` 顺序，PNG 导出也按同一附件名字选择数据。

## 5. BRDF 与阴影

直接光使用 Cook-Torrance：

- GGX/Trowbridge-Reitz 法线分布；
- Smith-Schlick 几何遮蔽；
- Schlick Fresnel；
- 漫反射项乘以 `(1 - F) * (1 - metallic)`，避免金属材质产生非物理漫反射。

方向光辐射度不衰减；点光使用平方反比和 range 窗口；聚光灯再乘以内/外锥角
平滑衰减。阴影采样使用硬件深度比较，并支持 3x3/5x5 PCF、深度 bias 和法线
相关 bias。方向光和聚光灯纹理使用 border depth 1，锥外或投影外默认不遮挡。

法线贴图在 G-buffer 顶点/片元阶段通过切线空间 TBN 变换到世界空间后再存入
法线附件，因此后续光照 pass 不需要重新采样材质纹理。

## 6. 资源生命周期

`DeferredRenderer` 的资源签名包含：

- 场景总校验和；
- 内部宽高和 SSAA；
- 阴影开关和三处阴影分辨率。

签名变化时 `_destroy_scene_resources()` 删除 G-buffer、HDR 目标、阴影 FBO/
纹理、网格 VAO/VBO/EBO 和法线纹理，再完整重建。程序退出时 `close()` 删除
程序对象、全屏 VAO 和 dummy 纹理。测试会在同一上下文内从 SSAA 1 切换到
SSAA 2，并断言内部尺寸与资源代数变化。

## 7. 报告与验证

报告中的统计在 HDR 线性空间计算；最终图像统计使用归一化 sRGB。主要一致性项：

- `direct_plus_indirect_vs_composite`：验证直接/间接分通道与合成通道像素一致。
- `reconstructed_tone_map_vs_final`：用与 GLSL 对应的 CPU 色调映射重建最终图像，
  验证后处理/SSAA 链路。
- `direct_vs_indirect`：提供两路光照差异。
- `nonlinear_preview_sum_vs_final`：诊断项，说明色调映射后图像不可线性相加。

`verify` 同时比较校验和、布局和实际 PNG。实质差异包括参数校验和变化、布局变化、
尺寸变化、缺失产物和超过阈值的显著像素差异。仅图像哈希变化但像素差异低于阈值，
且参数校验和一致时，归类为 `floating_point_noise`。

## 8. 已知边界

- 间接光是解析半球探针近似，不包含屏幕空间或路径追踪全局光照。
- 点光阴影每帧为六个面各执行一次场景深度 pass。
- 阴影贴图为单级深度纹理，PCF 通过多次硬件比较采样完成。
- 当前主要面向 Linux EGL/Mesa；其他平台需提供等价的 EGL surfaceless 或
  可替换的上下文后端。
