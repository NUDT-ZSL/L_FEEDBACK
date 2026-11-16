from __future__ import annotations

import argparse
import hashlib
import json

from .diagnostics_cli import configure_logging
from .scene_generator import DEFAULT_SEED, generate_scene, write_generated_scene


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the deterministic scene JSON.")
    parser.add_argument("--output", default="scene.json")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--summary", default="")
    args = parser.parse_args()
    configure_logging()
    config = write_generated_scene(args.output, args.seed)
    _, mesh = generate_scene(args.seed)
    summary = {
        "config": args.output,
        "seed": config.seed,
        "materials": len(config.materials),
        "lights": len(config.lights),
        "triangles": mesh.triangle_count,
        "vertices": len(mesh.vertices),
        "material_groups": [
            {
                "material": group.material_name,
                "index_offset": group.index_offset,
                "index_count": group.index_count,
                "vertex_offset": group.vertex_offset,
            }
            for group in mesh.groups
        ],
        "mesh_sha256": hashlib.sha256(
            mesh.vertices.tobytes() + mesh.indices.tobytes()
        ).hexdigest(),
    }
    print(json.dumps(summary, indent=2))
    if args.summary:
        with open(args.summary, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

