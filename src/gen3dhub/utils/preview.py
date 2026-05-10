"""Render a quick visual preview of a generated 3D mesh as a PNG.

Used by `gen3dhub run` after a successful inference so the user can scan
results visually without opening Blender. Intentionally low-fidelity — this
is a "did the asset come out roughly right?" thumbnail, not a final render.

Why matplotlib instead of pyrender / pyglet / Blender:
  matplotlib's Agg backend is fully headless and works on any system without
  an OpenGL / EGL setup. The output is a wireframe-style trisurf, recognizable
  enough to identify the asset at a glance. For final-quality renders the user
  opens the GLB directly.

The function is best-effort: if anything goes wrong (corrupt mesh, missing
matplotlib backend, etc.) the caller catches the exception and continues —
preview generation must never break the actual run.
"""

from __future__ import annotations

from pathlib import Path


def render_thumbnail(glb_path: Path, out_path: Path, *, size: int = 1024) -> None:
    """Write a 2x2-grid PNG preview of the mesh at `glb_path` to `out_path`.

    The grid shows the mesh from four camera angles (front-ish / side / back /
    other side) so the user can verify topology and silhouette quickly.
    """
    # Heavy imports kept inside the function so importing this module is cheap
    # — the CLI's startup path doesn't pay matplotlib's ~1s init cost unless
    # a preview is actually being generated.
    import matplotlib

    matplotlib.use("Agg")  # headless backend; no display required
    import matplotlib.pyplot as plt
    import numpy as np
    import trimesh

    loaded = trimesh.load(str(glb_path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)
        ]
        if not meshes:
            raise ValueError(f"No triangle meshes found in {glb_path}")
        merged = trimesh.util.concatenate(meshes)
    else:
        merged = loaded

    verts = np.asarray(merged.vertices, dtype=float)
    faces = np.asarray(merged.faces, dtype=int)
    if verts.size == 0 or faces.size == 0:
        raise ValueError(f"Empty mesh at {glb_path}")

    # Center on origin and scale to a unit cube so the camera framing is
    # consistent regardless of the source units.
    center = verts.mean(axis=0)
    verts = verts - center
    extent = max(float(np.abs(verts).max()), 1e-6)
    verts = verts / extent

    angles = [(20, 30), (20, 120), (20, 210), (20, 300)]
    titles = ["front-ish", "side", "back-ish", "other side"]

    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100, facecolor="white")
    for idx, ((elev, azim), title) in enumerate(zip(angles, titles, strict=True), start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        ax.plot_trisurf(
            verts[:, 0], verts[:, 1], faces, verts[:, 2],
            cmap="gray", linewidth=0.2, edgecolor="#444", antialiased=True,
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.set_title(title, fontsize=9, color="#333")

    fig.suptitle(glb_path.name, fontsize=11, color="#222")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
