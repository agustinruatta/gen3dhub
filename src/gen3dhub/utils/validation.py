"""Inspect a generated mesh and produce a structured report.

Used after every `gen3dhub run` (auto-summary) and exposed standalone via
`gen3dhub validate <glb>`. The metrics are picked to answer questions a game
dev typically asks before committing to an asset:

  - "Is this too heavy?"            → vertex / triangle counts
  - "Will the engine show it right?" → albedo / roughness / metallic / normal
  - "Will physics behave?"           → watertight + winding consistency
  - "What units is it in?"           → bounding-box extents
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

# Soft thresholds for the warnings list. Low-poly mobile assets aim for
# < ~5k tris; mid-poly desktop targets < 30k; cinematic above that. The
# 50k threshold is conservative — it's a "pay attention" line, not "broken".
_HIGH_TRI_COUNT_WARN = 50_000
# 5 MB GLB is already heavy for a single asset; mostly a "did you bake a 4K
# texture into a tiny prop?" check.
_LARGE_FILE_WARN_BYTES = 5 * 1024 * 1024


@dataclass
class MeshReport:
    """Structured metrics about a mesh file. JSON-serializable."""

    path: str
    vertex_count: int
    triangle_count: int
    file_size_bytes: int
    bounding_box: tuple[float, float, float]
    has_albedo: bool       # baseColorTexture present
    has_pbr: bool          # metallicRoughness texture or non-default factors
    has_normal_map: bool
    has_vertex_colors: bool
    is_watertight: bool
    is_winding_consistent: bool
    component_count: int   # number of disconnected components — 1 = single object
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def validate_mesh(path: Path) -> MeshReport:
    """Run the metrics on `path`. Raises ValueError if the file isn't a parseable mesh."""
    # Lazy imports — both deps are heavy, only worth paying for when we actually
    # validate something.
    import trimesh
    from trimesh.visual import ColorVisuals, TextureVisuals

    if not path.exists():
        raise FileNotFoundError(path)

    loaded = trimesh.load(str(path), force=None, process=False)

    # Collect every Trimesh in the file (Scenes wrap one or more meshes).
    meshes: list[trimesh.Trimesh] = []
    if isinstance(loaded, trimesh.Scene):
        for geom in loaded.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                meshes.append(geom)
    elif isinstance(loaded, trimesh.Trimesh):
        meshes.append(loaded)

    if not meshes:
        raise ValueError(f"No triangle meshes found in {path}")

    merged = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    file_size = path.stat().st_size

    # --- Material detection ---------------------------------------------------
    has_albedo = False
    has_pbr = False
    has_normal = False
    has_vertex_colors = False

    for mesh in meshes:
        visual = getattr(mesh, "visual", None)
        if visual is None:
            continue
        if isinstance(visual, ColorVisuals) and visual.kind == "vertex":
            has_vertex_colors = True
        material = getattr(visual, "material", None)
        if material is None:
            continue
        # PBRMaterial fields: baseColorTexture, metallicRoughnessTexture,
        # normalTexture (and friends). They're all None when absent.
        if getattr(material, "baseColorTexture", None) is not None:
            has_albedo = True
        if (
            getattr(material, "metallicRoughnessTexture", None) is not None
            or (getattr(material, "metallicFactor", None) not in (None, 1.0)
                and getattr(material, "roughnessFactor", None) is not None)
        ):
            has_pbr = True
        if getattr(material, "normalTexture", None) is not None:
            has_normal = True
        if isinstance(visual, TextureVisuals) and getattr(visual, "uv", None) is not None:
            # Texture coordinates present — counts as having an albedo channel
            # even if we didn't see an explicit baseColorTexture.
            has_albedo = has_albedo or True

    # --- Geometry checks ------------------------------------------------------
    extents = merged.bounding_box.extents
    bbox = (float(extents[0]), float(extents[1]), float(extents[2]))

    is_watertight = bool(merged.is_watertight)
    is_winding_consistent = bool(merged.is_winding_consistent)

    # Component count via connected_components on faces — 1 means a single
    # connected object; >1 means floating bits, sometimes intentional but
    # often a sign of a bad reconstruction.
    try:
        components = trimesh.graph.connected_components(merged.face_adjacency)
        component_count = len(components) if len(components) > 0 else 1
    except Exception:
        component_count = 1

    # --- Warnings -------------------------------------------------------------
    warnings: list[str] = []
    tri_count = int(merged.faces.shape[0])
    if tri_count > _HIGH_TRI_COUNT_WARN:
        warnings.append(
            f"High polycount ({tri_count:,} triangles) — heavy for mobile/web targets."
        )
    if file_size > _LARGE_FILE_WARN_BYTES:
        warnings.append(
            f"Large file ({file_size / 1024 / 1024:.1f} MB) — check texture resolution."
        )
    if not is_watertight:
        warnings.append(
            "Not watertight — has holes. Avoid for 3D printing / closed-volume physics."
        )
    if not is_winding_consistent:
        warnings.append(
            "Winding inconsistent — flipped faces. Some renderers will show black patches."
        )
    if component_count > 1:
        warnings.append(
            f"{component_count} disconnected components — check for floating geometry."
        )
    if not (has_albedo or has_vertex_colors):
        warnings.append(
            "No albedo / vertex colors detected — asset will render gray-flat in engine."
        )

    return MeshReport(
        path=str(path),
        vertex_count=int(merged.vertices.shape[0]),
        triangle_count=tri_count,
        file_size_bytes=file_size,
        bounding_box=bbox,
        has_albedo=has_albedo,
        has_pbr=has_pbr,
        has_normal_map=has_normal,
        has_vertex_colors=has_vertex_colors,
        is_watertight=is_watertight,
        is_winding_consistent=is_winding_consistent,
        component_count=component_count,
        warnings=warnings,
    )


def format_report_human(report: MeshReport) -> str:
    """Multi-line human-readable summary of a MeshReport."""

    def yn(flag: bool) -> str:
        return "[green]✓[/green]" if flag else "[red]✗[/red]"

    size_mb = report.file_size_bytes / (1024 * 1024)
    bbox_str = " x ".join(f"{x:.2f}" for x in report.bounding_box)
    lines = [
        f"[b]{Path(report.path).name}[/b]",
        f"  {report.vertex_count:,} verts  ·  {report.triangle_count:,} tris  ·  "
        f"{size_mb:.1f} MB  ·  bbox {bbox_str}",
        f"  Materials: albedo {yn(report.has_albedo)}  "
        f"PBR {yn(report.has_pbr)}  "
        f"normal {yn(report.has_normal_map)}  "
        f"vcolors {yn(report.has_vertex_colors)}",
        f"  Topology:  watertight {yn(report.is_watertight)}  "
        f"winding-OK {yn(report.is_winding_consistent)}  "
        f"components: {report.component_count}",
    ]
    if report.warnings:
        lines.append("  [bold yellow]Warnings:[/bold yellow]")
        lines.extend(f"    [yellow]⚠ {w}[/yellow]" for w in report.warnings)
    return "\n".join(lines)
