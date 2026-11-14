# 技术说明与实际验证记录

## 1. 离屏上下文

- `deferred_renderer/gl_context.py` 用 `glfw.window_hint(VISIBLE, FALSE)` 创建
  隐藏窗口，并请求 3.3 Core + forward-compat 上下文。
- 初始化时记录 `GL_VENDOR`、`GL_RENDERER`、`GL_VERSION`、GLSL 版本与 GLFW 版本。
- 本环境 GLFW 3.4 在 X11 平台下经 EGL 建立上下文，PyOpenGL 自带的
  `GetCurrentContext()`（基于 `glXGetCurrentContext`）返回空，导致其按上下文
  跟踪数组的 `glVertexAttribPointer` 抛 “no valid context”。
  解决方式是在上下文存活期间把 `OpenGL.platform.GetCurrentContext` 替换为稳定
  非零哨兵，退出时恢复，改动内聚在上下文类中，跨平台且不影响渲染语义。
- 全项目关闭 PyOpenGL 自动错误检查（`OpenGL.ERROR_CHECKING=False`），
  统一用 `glGetError()` 手动收集错误码，满足“错误码来自 glGetError”的要求。

## 2. MRT 几何阶段

- 单个 FBO 挂 3 个颜色纹理 + 1 个 `DEPTH_COMPONENT24` 渲染缓冲。
- `glDrawBuffers` 按槽位 0/1/2 绑定三个颜色附件，与几何 fragment shader 的
  `layout(location=...)` 输出严格一致（见 README 附件契约表）。
- 几何阶段启用深度测试与背面剔除；光照阶段禁用二者并切回默认帧缓冲。

## 3. Cook-Torrance 光照阶段

- 用 `gl_VertexID` 程序化生成全屏三角形，无需顶点缓冲。
- 逐像素从 G-Buffer 取位置、法线、反照率、材质 id，查材质数组得到
  粗糙度、金属度、自发光。
- 镜面项为 GGX NDF + Schlick-GGX 几何项 + Schlick 菲涅耳；
  支持 1 个平行光与最多 16 个点光源（含距离衰减与范围裁切）。
- 末端 Reinhard 色调映射 + gamma 2.2，写入 8 位默认帧缓冲。
- 光源与材质全部来自 `scene.json`，着色器不硬编码业务参数。

## 4. 半精度模式（--fast-approx）

GLSL 3.30 没有原生 fp16。为保证在任意驱动（含软件光栅）上结果确定，
`lighting.frag` 用整数位运算实现 IEEE-754 half 的压栈/弹栈（`f32ToF16Bits`、
`f16BitsToF32`），以 `H1/H3` 宏包裹 BRDF 关键中间量。`--fast-approx` 时定义
`FAST_APPROX` 启用，否则宏退化为恒等（float32）。两种模式各自确定，
但彼此帧哈希允许且确实不同。

## 5. 渲染目标保存与恢复

- 第 10 帧渲染完成后调用 `rebuild_gbuffer(11)`：`glDelete*` 销毁旧 FBO/纹理/
  深度缓冲，再以**完全相同规格**重建并重新绑定，随后继续输出 11–20 帧。
- 事件写入 `manifest.json` 的 `events`：含 `resume_frame`、销毁/重建的 FBO id、
  实际附件格式，以及重建前后的绑定状态快照（FBO 绑定、视口、深度/剔除开关、
  当前 program、纹理单元等）。
- 由于场景仅由 `(seed, frame_index)` 决定且重建不改变任何规格，
  重建运行与 `--no-rebuild` 对照运行的同帧 PNG 哈希逐像素一致。

## 6. 格式降级链路

- `--force-format 附件=UNSUPPORTED` 把该附件 internalformat 设为非法枚举
  （`0xFFFFFFFF`）。Mesa 对 `glTexImage2D` 返回 `GL_INVALID_VALUE(0x501/1281)`。
- PyOpenGL 可能在参数转换期就抛 `GLError`，代码同时捕获异常并调用
  `glGetError()` 取权威错误码，记录结构化 warning：
  业务码 `RENDER_FBO_INCOMPLETE`、附件名、请求/回退格式、GL 错误码。
- 随后用设备支持的默认格式重建该附件并继续渲染，降级成功退出码为 0；
  若默认格式仍失败则抛出致命错误、退出码 1，不崩溃不挂起。

## 7. 帧间状态一致性校验

- 几何通道结束后，对位置附件做一次**真实 GPU 回读**（屏幕中心连续 10 像素，
  `glReadPixels(... GL_FLOAT)`），SHA-256 作为 `gbuffer_write_sha256`。
- 光照前再独立回读同样 10 像素得到 `gbuffer_verify_sha256`，两者不一致即
  `RENDER_STATE_MISMATCH`，记录期望/实际哈希并以退出码 1 终止。
- 回读点选在屏幕中心条带而非角落，确保能命中随帧运动的球体，
  因此 G-Buffer 哈希会随动画变化（实测 21 帧中 16 个不同哈希）。

## 8. 踩坑记录（关键驱动/绑定问题）

- **矩阵上传转置**：NumPy 为行主序（C-order），GLSL uniform 为列主序。
  `glUniformMatrix*fv` 必须传 `GL_TRUE` 转置，否则视图矩阵等效被转置，
  场景被投到视锥外表现为黑屏。`set_mat4/set_mat3` 已统一处理。
- **地面绕序**：地面顶点需按俯视逆时针排列，才能在 `GL_BACK` 剔除 + CCW
  正面下保留朝上表面。
- **勿用 `glGetBufferParameteriv` 判成败**：该驱动上此查询返回不可靠，
  `glBufferData` 是否成功应以 `glGetError`/`glMapBuffer` 回读为准。

## 9. 实际验证结果（Mesa llvmpipe, GL 4.5, Python 3.14）

- 同种子两次运行：21 帧 PNG 哈希完全一致。
- 重建运行 vs `--no-rebuild`：全部 21 帧（含 11–20）逐像素一致。
- `--fast-approx`：同模式两次运行一致；与 float32 的 21 帧哈希全部不同；
  该模式下重建前后仍逐像素一致。
- 异种子（42 vs 7）：所有帧哈希不同，`scene.json` 内容可复现且随种子变化。
- 强制 `UNSUPPORTED`：产生 `RENDER_FBO_INCOMPLETE` warning（含 1281 错误码），
  回退到默认格式，退出码 0。
- 注入默认格式也失败：退出码 1；注入 G-Buffer 篡改：`RENDER_STATE_MISMATCH`
  且退出码 1。
- `python -m unittest discover -s tests`：11 个测试全部通过。
