# 技术说明

## 1. 确定性输入

场景完全由编译进程序的常量构造：

- 相机轨迹：以解析正弦/余弦函数生成眼点和目标点，输入只有 `frame / fps`。
- 角色网格：由程序化盒体、椭圆截面管和低纬球体组成。
- 骨骼：`hips -> chest -> head` 三级链。
- 灯光：一个定向光、两个点光源。
- 遮挡体：三个静态盒体；地面为静态大平面。

不使用系统时间、随机数、GPU 查询结果或任何外部文件来决定场景内容。

## 2. 网格与权重

角色顶点格式为 56 字节：

```text
float position[3]
float normal[3]
uint32 boneIndex[4]
float boneWeight[4]
```

构建器为每个顶点生成最多 3 个有效影响（格式支持 4 个），随后：

1. 将负值/非有限权重清零。
2. 对权重和做归一化。
3. 若和小于 `1e-6`，回退为骨骼 0 的权重 1。
4. 校验索引 `< 3` 且权重和与 1 的误差小于 `1e-4`。

角色、遮挡体和地面都走权重校验。角色网格可在进程内通过
`DeterministicScene::rebuildCharacter()` 重建，随后 `Renderer::uploadCharacter()` 更新
VAO 绑定的 VBO/EBO 并继续渲染。`MeshParams` 提供 `bodyWidth`、`heightScale`、
`headScale`、`armSpan` 等参数；CLI 的 `--regenerate-variant` 只是这些参数的固定、可
复现实例。热重建路径已用 `--regenerate-frame` 实际跑通。

## 3. 关键帧和姿势

每根骨骼有关键帧 `{time, translation, rotation}`。时间不要求等间隔，且严格递增。

采样流程：

1. `time <= first.time` 使用首帧；`time >= last.time` 使用末帧。
2. 否则用 `upper_bound` 找到相邻两帧，按时间做平移线性插值。
3. 旋转先归一化；若两点积为负，翻转第二个四元数，保证最短弧。
4. 点积接近 1 时用归一化线性插值，否则用球面插值，最后再次归一化。

CPU 姿势矩阵使用 `double`：

```text
local_i = T(key.translation_i) * R(key.rotation_i)
world_i = world_parent * local_i
skin_i  = world_i * inverseBind_i
inverseBind_i = inverse(bindWorld_i)
```

上传给 GPU 时才转换为 `float` 矩阵。CPU 与 GPU 使用同一组姿势矩阵，CPU 另外计算变形后
顶点哈希写入统计，便于发现姿势/插值回归。

## 4. 蒙皮和法线

设归一化权重为 `w_i`，每个顶点的位置：

```text
p' = sum_i w_i * (skin_i * p)
```

法线不能直接乘位置蒙皮矩阵。实现使用每根骨骼蒙皮矩阵的逆转置加权：

```text
n' = normalize(sum_i w_i * inverseTranspose(mat3(skin_i)) * n)
```

这等价于对每个蒙皮后局部坐标系做逆转置，并正确处理非均匀缩放。顶点着色器对 4 个槽位都
计算逆转置，权重为零的槽位不贡献。CPU 测试用 `scale(2,1,0.5)` 验证：

- 变换后法线与变换后切向的点积约为 0。
- 直接乘缩放矩阵的错误法线与切向有明显非零点积。

权重在 CPU 构建阶段归一化，GPU 顶点阶段再次防御性归一化并处理全零权重。

## 5. 渲染通道

所有对象使用同一顶点布局，但静态对象使用不读取骨骼的顶点着色器：

1. **定向阴影通道**：每级独立 FBO + `D32F` 深度纹理。角色用蒙皮阴影着色器，遮挡体用静态
   阴影着色器；地面不投射阴影。
2. **主视图深度预通道**：向主 FBO 的 `D32F` 深度纹理写入场景深度。该通道和颜色通道共用
   同一 UBO 绑定点、同一相机矩阵、同一网格和同一蒙皮块。
3. **主视图颜色通道**：深度测试 `LEQUAL`、关闭深度写入，采样级联阴影纹理并计算定向光、
   两个点光源、环境光和材质。颜色做确定性 Reinhard + gamma 映射后写入 `RGBA8`。
4. **回读**：从主 FBO 读取 `RGBA8` 颜色和 `float` 深度，翻转行序后写 PNG。

主视图颜色附件的深度附件就是最终保存的深度图；阴影通道另外保存各级深度纹理。

## 6. 光源/材质数据一致性

三个 UBO 绑定点为：

```text
binding 0: BoneBlock    mat4 boneMatrices[4]
binding 1: ShadowBlock  mat4 lightViewProjection[4], splits, shadow params
binding 2: LightBlock   定向光、两个点光源、环境光
```

`LightBlock` 在颜色通道和深度通道都绑定；深度通道不读取颜色项，但上传统一参数使通道
共享同一份灯光状态。阴影矩阵由同一个 `DirectionalLight` 生成，阴影通道写图和颜色通道
采样使用完全相同的 `lightViewProjection` 矩阵。

材质参数只影响颜色通道，不改变几何深度；阴影通道与颜色通道共享同一网格、模型矩阵、骨骼
姿势和变换链。这是深度通道与材质之间的正确一致性边界：材质不能改变几何深度，但也不允许
两个通道使用不同几何或不同光源方向。

## 7. 级联阴影

### 7.1 划分

先由可见投射体包围盒计算场景远深度：

```text
sceneFar = min(camera.far, maxDepthOfVisibleCasterBounds + margin)
```

对第 `i` 段使用 practical split：

```text
u = near + (sceneFar - near) * i / N
l = near * (sceneFar / near)^(i / N)
split_i = lambda * l + (1 - lambda) * u
```

默认 `lambda = 0.25`。在本次 640x360 验证场景中，两级视深范围为：

```text
[0.450, 5.272]
[5.272, 12.056]
```

两段都有非零阴影图占用率，避免近级联只覆盖空白。

### 7.2 光源空间拟合

每级执行：

1. 由相机投影逆矩阵生成该视深切片的 8 个视锥角点。
2. 如果可见投射体包围盒与该切片（加 caster padding）重叠，则把包围盒 8 个角点加入候选集。
3. 用定向光方向建立光源视图。
4. 在光源观察空间中计算候选点 AABB。
5. 对 X/Y 使用正方形范围，增加 6% 和 2 texel 余量。
6. 将 X/Y 中心吸附到 texel 网格，减少相机移动时的阴影抖动。
7. 对 Z 做深度平移，把全部候选点移到相机前方，再建立正交投影。

每级获得独立 `lightView`、`lightProjection`、`lightViewProjection` 和深度纹理。

### 7.3 边界和偏移

- 阴影纹理过滤为 `NEAREST`。
- `WRAP_S/T = GL_CLAMP_TO_BORDER`，边界颜色为白色。
- 片元在投影坐标超出 `[0,1]` 时直接返回“未遮挡”。
- 使用 3x3 PCF；每个 tap 比较 `projected.z - bias` 与存储深度。
- 恒定偏移与斜率偏移同时存在，阴影通道另使用 polygon offset。
- 不使用阴影图集，因此不会发生相邻级联纹理串采。

片元按相机视深选择级联；每个级联选择分支都使用对应矩阵和纹理。所有矩阵来自同一
`ShadowBlock`，颜色通道不会自行重建光源矩阵。

### 7.4 精度和分辨率取舍

`D32F` 比 `D16` 更不容易产生大范围 acne，但每 texel 占 4 字节。分辨率提高一倍，线性
texel 尺寸减半，但显存和填充成本约变为四倍。本次 512x512、两级验证中统计的世界单位
texel 约为：

- 近级联：约 `0.0199 m/texel`。
- 远级联：约 `0.0385 m/texel`。

默认使用 1024x1024 两级；只有确实需要时再改成 2048/4096。每级独立纹理让近级联获得更高
有效分辨率，同时远级联用更大覆盖范围换取可见范围。

## 8. 确定性和错误处理

- 无随机数、无墙钟输入、无外部资源。
- 关闭混合、MSAA、抖动、裁剪面等不必要的实现相关状态。
- `stbi_write_png_compression_level` 固定为 8。
- 深度 PNG 由内置无压缩 deflate 块编码器写出，不含文本/时间 chunk。
- 相机和关键帧只依赖整数帧号和固定 fps。
- 读取像素后按固定行序翻转，颜色与深度使用不同编码器但相同行序。

资源全部使用 RAII 风格析构或显式 `shutdown()` 释放。错误通过带类别和符号的异常传播到
`main`，映射到稳定退出码并打印到 `stderr`。测试参数可以强制着色器编译失败或不完整 FBO，
用于实际验收错误路径。

## 9. 上下文后端

- Windows：隐藏 GLFW 窗口 + OpenGL 4.3 Core；GLAD 通过 `glfwGetProcAddress` 加载。
- Linux 验证：OSMesa 4.3 Core 离屏上下文；GLAD 通过 `OSMesaGetProcAddress` 加载。
- 两者共享同一渲染目标、GLSL、CPU 数学、CSM 和输出代码。

Windows 代码路径不依赖 OSMesa；Linux 路径不依赖 GLFW。当前工作区没有 Windows 图形会话，
因此实际像素验证在 OSMesa/llvmpipe 上完成。
