"""Mesh format conversion. Thin wrapper around trimesh's load+export.

Used by `gen3dhub run --format <fmt>` to convert each adapter's native GLB
output to whatever format the user actually wants (.obj, .ply, .stl). Lives
here so adapters don't need to know anything about format conversion — they
keep producing GLB and the CLI handles the rest.

Caveats per target format (documented for users in `--format` help):
  - glb  full PBR materials, embedded textures, UV coords. Native target.
  - obj  writes .obj + .mtl + texture PNG sidecars (multiple files).
         Materials preserved when the source has them. Standard for
         Blender / 3ds Max / legacy DCC pipelines.
  - ply  preserves geometry + vertex colors. Drops UV-mapped textures
         (PLY can't carry them in a portable way). Good for point-cloud
         workflows or DCC tools that prefer it.
  - stl  geometry only. Discards all color, material, and UV info.
         Right format for 3D printing slicers.
"""

from __future__ import annotations

from pathlib import Path

#: Tuple of formats `gen3dhub run --format` accepts. Order = preference for
#: extension-based inference (.glb wins on ties when nothing's been specified).
SUPPORTED_FORMATS: tuple[str, ...] = ("glb", "obj", "ply", "stl")


def convert_mesh(src: Path, dst: Path) -> None:
    """Read a mesh from `src` and write it to `dst`. Format is inferred from
    `dst`'s suffix. Raises on failure.

    Best-effort with respect to materials:
      - GLB → OBJ: trimesh writes the .mtl + textures alongside dst.
      - GLB → PLY: vertex colors preserved if present; UV textures dropped.
      - GLB → STL: geometry only.
    """
    import trimesh

    loaded = trimesh.load(str(src), force=None, process=False)
    if loaded is None:
        raise ValueError(f"Could not load source mesh: {src}")
    # trimesh.export auto-detects the format from the extension.
    loaded.export(str(dst))


def format_for_path(path: Path | None, default: str = "glb") -> str:
    """Return the format implied by `path`'s extension, or `default` when the
    extension isn't one we support."""
    if path is None:
        return default
    suffix = path.suffix.lstrip(".").lower()
    return suffix if suffix in SUPPORTED_FORMATS else default
