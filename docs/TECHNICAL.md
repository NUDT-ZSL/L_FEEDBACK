# 技术说明

## 1. 执行模型

程序是单进程离线渲染器。Windows 上通过 GLFW 创建隐藏的 OpenGL 4.5 core 上下文；
Linux 上为了无头验收，使用 EGL surfaceless/pbuffer 上下文。两条路径共享相同的
OpenGL 状态、着色器、帧缓冲和 CPU 后处理代码。

每个子帧执行一次全屏三角形绘制。片元着色器对每个像素做球体阵列射线求交，所有
直接光照、环境球谐间接光和材质 BRDF 都在同一个片元着色器中完成。结果写入
`GL_RGBA32F` 纹理；所有子帧通过固定顺序的加法混合累积，最后用
`glReadPixels` 从帧缓冲读取。另一次辅助通道输出每个像素的主材质 ID 与镜面反射
方向，供 CPU 统计使用。

## 2. 场景解析与材质 ID

`scene.cpp` 使用严格类型检查的 nlohmann/json 解析器。材质数组的顺序就是材质 ID。
球体可以使用数字 `material_id` 或名称 `material`；两者同时存在时解析器验证一致性。
渲染器上传到 shader 的 `uSphereMaterial[]`、辅助通道返回的 ID 和 CPU 统计使用的 ID
都来自同一个 `Scene::spheres` 结果，不存在两套独立映射。

## 3. BRDF

法线分布使用各向异性 GGX：

```text
alpha  = roughness^2
aspect = sqrt(max(1 - 0.9 * anisotropy, 0.05))
alphaT = alpha / aspect
alphaB = alpha * aspect
D = 1 / (pi * alphaT * alphaB *
         (Ht^2/alphaT^2 + Hb^2/alphaB^2 + Hn^2)^2)
```

其中 `Ht/Hb/Hn` 是半程向量在材质切空间中的分量。切空间由球面法线构造并按材质的
`anisotropy_rotation_degrees` 旋转。几何项使用 Smith-GGX，Fresnel 使用 Schlick 近似；
介电质 F0 由 IOR 计算，金属 F0 取基础色。漫反射项为
`baseColor * (1 - metallic) * (1 - F) / pi`。

## 4. 直接光

- 4 个点光源按 delta 光源直接求和，使用平方反比衰减。
- 矩形光每次采样按 `luminance(radiance) * area` 的 CDF 选择一个光源，再在矩形上
  均匀取点。选择概率、面积 PDF、距离平方和光源/着色点余弦共同组成蒙特卡洛估计。
- 所有阴影射线按当前子帧的球体位置求交。

点光源本身不需要方向采样；矩形光源的概率选择与面积采样构成直接光重要性采样。

## 5. 环境球谐

启动时对内建环境描述（天顶色、地平线色、地面色、太阳方向/角半径/辐射度）用固定
8192 个 Fibonacci 球面方向积分 9 个 L2 SH 系数。另按各阶余弦卷积因子计算辐照度
系数。

- 漫反射间接光：`albedo * E(n) / pi`。
- 镜面间接光：沿反射方向求值辐射 SH，再乘粗糙 Fresnel。
- 背景：直接求值解析环境函数。

整个过程只依赖场景 JSON 中的环境参数和内置积分序列，不读取环境贴图。

## 6. 物理相机、景深与运动模糊

相机像素方向由焦距和传感器尺寸决定：

```text
dir = normalize(forward * focalLength
              + right * ndc.x * sensorWidth/2
              + up    * ndc.y * sensorHeight/2)
```

光圈半径为 `focalLength / (2 * fNumber)`。每个样本在镜头圆盘上抖动，并把光线从
抖动后的镜头点射向对焦平面，从而自然产生景深。像素内也有亚像素抖动。

运动模糊使用 `--subframes` 个等距时间子帧，时间点为 `(k + 0.5) / N`。球体阵列的
位移由 JSON 的 `motion` 速度乘以相机 `exposure_time`，所以改变曝光时间会改变固定
快门窗口内的移动距离。子帧结果按 `1/N` 累加。

景深/低采样降噪使用确定性 CPU 双边滤波：默认 5x5，`samples < 64` 时 7x7。空间
权重为高斯，颜色停止权重由亮度差决定，并且只混合相同材质 ID 的像素，避免跨轮廓
渗色。默认参数下没有随机线程或时间依赖。

## 7. 色彩管线

- 帧缓冲保持线性场景辐射值。
- HDR 用 Radiance RGBE/RLE 写入；文件头包含曝光 EV、色调映射名、曝光倍率、采样数、
  子帧数和场景路径。HDR 像素本身不应用曝光或曲线。
- PNG 流程：`linear * 2^EV` -> Reinhard/ACES/Uncharted2 -> sRGB -> 8-bit RGBA。

## 8. 确定性

确定性来自以下固定处理：

1. 512x512 固定分辨率和固定种子 `0x105`。
2. 片元内 PCG 哈希随机流；种子只由像素坐标、子帧编号和固定种子组成。
3. 固定采样顺序、固定子帧中心时刻、固定矩形光选择顺序。
4. CPU SH 使用固定 8192 个 Fibonacci 方向，不使用系统随机数。
5. 后处理滤波按行/列固定顺序执行，不使用多线程归约。
6. 不写入时间戳或运行期随机数据。

相同 GPU/驱动和相同输入下，HDR 输出应逐字节一致。不同 GPU 驱动的浮点指令细节可能
造成极小的统计差异，因此验证阈值不要求跨厂商像素级完全一致。

## 9. 统计与验证

CPU 从辅助帧缓冲读取材质 ID 和主反射方向，并结合线性颜色计算：

- `mean_color`：材质像素的线性 RGB 均值。
- `highlight_histogram`：每个材质像素亮度 `log2` 后映射到 `[-8, 8]` 的 16 个桶，
  再按像素数归一化。
- `specular_direction`：以 `luminance^2` 为权重平均反射方向，强调高光区域。

验证时按材质 ID 匹配基准条目，计算颜色欧氏距离、直方图 L1 和方向夹角。任一指标
超限只标记报告失败，不抛出异常、不改变进程退出码。

## 10. 错误处理

- 参数解析错误 -> 1。
- 场景读取、JSON 语法/类型/字段/范围错误 -> 2。
- GLFW/EGL/OpenGL、着色器、帧缓冲、HDR/PNG/报告/参考文件 IO 错误 -> 3。

所有错误写 stderr；场景解析错误包含具体 JSON 路径，例如
`'scene.spheres[0].material_id': material ID 99 is out of range [0, 4]`。
