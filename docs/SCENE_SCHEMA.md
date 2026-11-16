# Scene JSON Contract

The scene is a single JSON object. All fields listed below are required unless explicitly
marked optional. Unknown fields are ignored. Duplicate keys, non-finite numbers, and JSON
syntax errors are rejected.

## Top Level

```json
{
  "camera": { "...": "..." },
  "meshes": [ { "...": "..." } ],
  "lights": { "...": "..." }
}
```

## Camera

```json
{
  "position": [5.2, 4.4, 7.2],
  "target": [0.0, 0.25, 0.0],
  "up": [0.0, 1.0, 0.0],
  "projection": "perspective",
  "fov_y_deg": 43.0,
  "near": 0.1,
  "far": 100.0
}
```

- `position`, `target`, and `up` are finite 3-component arrays.
- `target` must differ from `position`; `up` must be non-zero.
- `projection` must be the enum string `"perspective"`. Any other value produces
  `SCENE_ERR_INVALID_ENUM`.
- `fov_y_deg` is in `[1,179]`.
- `near` and `far` are positive, with `far > near`.

## Mesh

```json
{
  "name": "torus",
  "model": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
  "vertices": {
    "positions": [[0.0, 0.0, 0.0]],
    "normals": [[0.0, 1.0, 0.0]],
    "tangents": [[1.0, 0.0, 0.0]],
    "uvs": [[0.0, 0.0]]
  },
  "indices": [0, 0, 0],
  "material": { "...": "..." }
}
```

- `name` is a string.
- `model` is optional and defaults to identity. It is a 16-number column-major matrix, as
  consumed directly by GLSL.
- `positions`, `normals`, `tangents`, and `uvs` are parallel arrays and must have the same
  non-zero length.
- Each position, normal, and tangent has 3 finite floats; each UV has 2 finite floats.
- A normal must have non-zero length.
- Every tangent must exist, be finite, have non-zero length, and not be parallel to its
  vertex normal. A missing tangent array, short tangent array, zero tangent, or parallel
  tangent produces `SCENE_ERR_INVALID_TANGENT`; there is no fallback tangent.
- Indices are non-negative integers, divisible into triangles, and in
  `[0, vertex_count)`. Invalid values produce `SCENE_ERR_INVALID_INDEX`.
- Meshes use counter-clockwise front faces. Back faces are culled in geometry and shadow
  passes.

## Material

```json
{
  "base_color": [0.82, 0.55, 0.23],
  "roughness": 0.24,
  "metallic": 0.92,
  "anisotropy_strength": 0.9,
  "anisotropy_rotation_rad": 0.35
}
```

- `base_color` components are in `[0,1]`.
- `roughness` and `metallic` are in `[0,1]`.
- `anisotropy_strength` is in `[0,1]`; zero reduces the GGX lobe to isotropic.
- `anisotropy_rotation_rad` rotates the anisotropy direction around the interpolated
  normal in the tangent plane. It is a finite radian offset.

The final tangent is orthonormalized against the normal, rotated by the offset, and used
to build the anisotropic GGX tangent frame in deferred shading.

## Lights

Both keys must exist. The directional light may be `null`; the point-light array may be
empty. This is how a zero-light scene is expressed.

```json
{
  "directional": {
    "type": "directional",
    "direction": [-0.45, -1.0, -0.55],
    "color": [1.0, 0.95, 0.82],
    "intensity": 3.4
  },
  "point_lights": [
    {
      "type": "point",
      "position": [3.0, 2.1, 2.1],
      "color": [1.0, 0.32, 0.12],
      "intensity": 32.0,
      "range": 11.0
    }
  ]
}
```

- At most four point lights are allowed. A fifth produces
  `SCENE_ERR_TOO_MANY_LIGHTS` with exit code 2.
- Light `type` values are enums: `"directional"` and `"point"`. Invalid values produce
  `SCENE_ERR_INVALID_ENUM`.
- Directional `direction` is the direction in which rays travel; non-zero and finite.
- Colors are finite and each component is in `[0,1]`.
- Intensities are finite and non-negative.
- Point light `range` is finite and positive.
- Point radiance uses inverse-square attenuation multiplied by a finite-range window.

## Zero-Light Scene

```json
{
  "lights": {
    "directional": null,
    "point_lights": []
  }
}
```

The main camera still renders the geometry G-Buffer, and the final lighting image is
black. `shadow_method` is `"none"` and `light_count` is `0`.

## Error JSON

Example:

```json
{"ok":false,"code":"SCENE_ERR_INVALID_INDEX",
 "message":"index is outside the vertex array",
 "path":"scene.meshes[0].indices[17]"}
```

All scene validation completes before the render marker or any output file is created.
