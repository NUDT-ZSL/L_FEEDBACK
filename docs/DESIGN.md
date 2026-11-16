# Technical Design

## Platform Backends

- **Windows:** GLFW 3.4 creates a 64x64 `GLFW_VISIBLE=FALSE` window and an OpenGL 3.3
  core context. `glfwGetProcAddress` loads the project's self-contained GL entry points.
  Destruction order is renderer resources, hidden window, `glfwTerminate`.
- **Linux verification:** libEGL is loaded dynamically. A surfaceless Mesa display plus
  a 1x1 pbuffer supplies an OpenGL 3.3 core context. No visible window or X/Wayland
  surface is created. `eglGetProcAddress` is used with `libGL` symbol fallback.

No GL, EGL, GLEW, or GLAD headers are needed to build the project. GLM is also not needed.

## Offscreen G-Buffer

Five `RGBA16F` color textures are attached to one FBO:

| Attachment | RGB | Alpha |
| --- | --- | --- |
| 0 | albedo | metallic |
| 1 | world normal | roughness |
| 2 | world position | positive view-space depth |
| 3 | tangent | anisotropy strength |
| 4 | anisotropy rotation | UV (diagnostic only) |

A `D24S8` texture is attached as `GL_DEPTH_STENCIL_ATTACHMENT`. This exceeds the required
minimum of three float color attachments and keeps all anisotropic material inputs
available in the deferred pass.

The lighting output is a separate `RGBA16F` target: RGB is HDR linear radiance and alpha
is the aggregate shadow-factor diagnostic.

## Shadow Maps

### Directional Light

The world-space bounds of all transformed mesh vertices define a scene center and radius.
The light camera looks along the directional-light ray direction and uses an orthographic
projection covering the bounding sphere. The shadow target is a `DEPTH_COMPONENT24`
texture with the requested square resolution.

The deferred shader projects the world position into light space and compares nine
neighbor taps:

```text
visibility = count(current - bias <= sampled_depth) / 9
```

The material normal is offset slightly before projection to reduce self-shadowing.

### Point Lights

Each point light owns a `DEPTH_COMPONENT24` cube map. Six 90-degree perspective faces are
rendered through one reusable depth-only FBO. A depth-writing fragment shader stores
normalized radial distance, not ordinary perspective depth:

```text
gl_FragDepth = clamp(distance(fragment, light) / far, 0, 1)
```

This makes the cube samples comparable using `length(fragment - light) / far`. The
deferred point-shadow lookup perturbs the direction by approximately one texel in the
current cube-face tangent plane and performs nine compare samples. Every point cube uses
the same `--shadow-size` as the directional map.

`shadow_method` is built from actual configured resources:

- no lights: `none`
- directional only: `directional_2d_pcf3x3`
- points only: `point_cubemap_pcf3x3`
- both: `directional_2d_pcf3x3+point_cubemap_pcf3x3`

## Deferred Lighting

The fullscreen deferred pass reconstructs material values and world position directly
from the G-Buffer. It evaluates:

- Directional radiance with directional-shadow visibility.
- Every point light with inverse-square attenuation, a range window, and its own cube
  shadow visibility.
- Metal/dielectric Fresnel `F0 = mix(0.04, albedo, metallic)`.
- Energy-reduced diffuse for dielectrics.
- Tangent-space anisotropic GGX distribution.

Let `roughness` be `r` and anisotropy strength be `a`:

```text
alpha    = max(r^2, 0.001)
aspect   = sqrt(clamp(1 - 0.9a, 0.05, 1))
alphaT   = max(alpha / aspect, 0.0005)
alphaB   = max(alpha * aspect, 0.0005)
D         = 1 / (pi * alphaT * alphaB * H^2)
H^2       = (h.T/alphaT)^2 + (h.B/alphaB)^2 + (h.N)^2
```

Binding the stored tangent to the interpolated normal, rotating it by the material offset,
and using its cross product with the normal creates the anisotropic highlight ellipse.
Schlick Fresnel and Smith-Schlick visibility complete the direct-light BRDF.

## Image and Statistics

The HDR final buffer is read back as floats after `glFinish`. Full-color output uses an
ACES fit followed by gamma 2.2 encoding. Geometry output encodes normals into RGB.
Lighting output displays the aggregate shadow factor in grayscale.

`render_time_ms` starts immediately before shadow rendering and stops after
`glFinish` for the one complete pipeline. Readback and file writing are excluded.

Statistics luminance uses tone-mapped linear RGB. Visibility comes from the G-Buffer
position alpha. Values are clamped to `[0,1]`; if clamping is needed, the corresponding
boolean flag is emitted.

## Transactionality and Recovery

Before the GL context is created, `<prefix>.render-in-progress.json` is written. Output
files use same-directory temporary names:

```text
<target>.tmp.<pid>.<counter>
```

Each temp is flushed, then atomically renamed over the target. The marker is removed only
after both PPM and stats have been committed.

On startup, the marker path is checked before scene loading. If present, matching
`<prefix>_*.tmp.*` files are removed and the marker is removed. `RECOVERY_CLEANED` is
reported on stderr. This same path handles an actual forced kill in the integration test.

Handled signals set only `volatile sig_atomic_t` state. Worker stages poll it. On
interruption, RAII releases VAOs, buffers, textures, FBOs, programs, context, and hidden
window before the final interrupted-state JSON is emitted.
