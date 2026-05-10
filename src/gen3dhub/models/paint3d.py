"""Adapter for Paint3D (OpenTexture / Tencent), a mesh-to-texture model.

Reference: https://github.com/OpenTexture/Paint3D
Paper:     https://arxiv.org/abs/2312.13913

Inputs:  a 3D mesh (.obj / .glb / .ply / .off) + a reference image.
Output:  a textured GLB mesh (Paint3D natively writes OBJ + MTL + PNG; we
         convert to GLB with trimesh for uniformity with the other adapters).

Two-stage pipeline (both run by our runner):
  1. Coarse texturing: depth-conditioned multi-view inpainting projected
     into UV space. Produces a 1024x1024 albedo PNG.
  2. UV refinement: UV-space inpainting + tile-based super-resolution to
     clean seams and occluded regions.

Important caveats vs SF3D's PBR pipeline:
  - Paint3D outputs ALBEDO ONLY. No normal map, no roughness, no metallic.
    For game pipelines that expect PBR, you'll need to author the other
    channels separately (or use SF3D's pipeline instead, which generates
    them from a single image).
  - Upstream is effectively unmaintained (last code commit 2024-06-26),
    pinned `diffusers==0.25.0` is broken with modern huggingface_hub, and
    YAML configs reference the deleted `runwayml/stable-diffusion-v1-5`
    repo. We work around these by bumping diffusers/huggingface_hub to
    versions that follow HF's HTTP redirects transparently.
  - No CPU mode upstream (.to("cuda") is hard-coded). 8 GB VRAM is
    marginal but feasible thanks to upstream's `enable_model_cpu_offload`
    inside the diffusers pipelines.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from gen3dhub.console import info, success, warn
from gen3dhub.models.base import (
    HardwareNeeds,
    InputKind,
    InputSpec,
    ModelAdapter,
    ModelInfo,
    ParamKind,
    ParamSpec,
    RunRequest,
)
from gen3dhub.utils.process import run_streaming
from gen3dhub.utils.system import check_build_toolchain

_REPO_URL = "https://github.com/OpenTexture/Paint3D.git"
# Pinned for reproducibility. Latest commit on main as of 2026-05-10.
# The repo has no tags; this is the only stable handle.
_REPO_COMMIT = "d5545d38db6aaf78efc563269f70177688a4218e"

# License URL pointing at the Apache 2.0 LICENSE file in the upstream repo.
_LICENSE_URL = "https://github.com/OpenTexture/Paint3D/blob/main/LICENSE"

# Modernized stack — Python 3.11 + torch 2.4 + kaolin 0.17 from NVIDIA's
# wheel index. Keeps us off the EOL Python 3.8 / torch 1.12 combo the
# upstream environment.yaml pins.
_TORCH_PACKAGES: tuple[str, ...] = ("torch==2.4.0", "torchvision==0.19.0")
_KAOLIN_INDEX_URL = "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html"
_KAOLIN_PACKAGE = "kaolin==0.17.0"

# Minimal runtime deps. Skips the dead weight in environment.yaml (streamlit,
# pytorch-lightning, webdataset, kornia, albumentations, etc.) which the
# inference pipelines never import.
_PIP_PACKAGES: tuple[str, ...] = (
    "numpy<2",
    "opencv-python",
    "Pillow",
    "einops",
    "omegaconf",
    "loguru",
    "trimesh",
    "xatlas",
    "transformers>=4.36,<4.46",  # IP-Adapter image encoder needs CLIPVisionModel
    "accelerate>=0.29",
    "diffusers>=0.27,<0.31",     # bumped from upstream's broken 0.25.0
    "huggingface_hub>=0.30",     # follows the SD 1.5 HF redirect transparently
    "tqdm",
    "safetensors",
)

# Runner script: orchestrates the two upstream pipeline scripts and converts
# the resulting OBJ+MTL+PNG triple to a single GLB with trimesh. Written
# verbatim into the model dir during setup so a `setup --force` always
# rewrites the latest version, and the upstream `repo/` stays clean.
_RUNNER_SOURCE = """\
'''gen3dhub runner for Paint3D. Runs both stages then converts to GLB.

Usage: python runner.py <mesh_path> <reference_image_path> <output_glb>

Tunables (read from environment so the adapter can pass them per-run):
  PAINT3D_PROMPT   text prompt to combine with the reference image
                   (empty -> upstream's whitespace fallback for IP-Adapter only)
'''
import os
import sys
import subprocess
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit("usage: runner.py <mesh_path> <ref_image_path> <output_glb>")
    mesh_path = Path(sys.argv[1]).expanduser().resolve()
    image_path = Path(sys.argv[2]).expanduser().resolve()
    output_glb = Path(sys.argv[3]).expanduser().resolve()

    repo_dir = Path(__file__).parent / "repo"
    work_root = Path(__file__).parent / "work"
    work_root.mkdir(exist_ok=True)

    # Use a fresh temp subdir per invocation to keep artifacts separate.
    import tempfile
    work_dir = Path(tempfile.mkdtemp(prefix="paint3d-", dir=work_root))
    stage1_dir = work_dir / "stage1"
    stage2_dir = work_dir / "stage2"

    # Empty prompt isn't allowed by upstream's argparse. Pass " " when the
    # user supplied nothing — IP-Adapter handles image-only conditioning fine.
    prompt = os.environ.get("PAINT3D_PROMPT", "").strip() or " "

    print("=== Paint3D stage 1: coarse texture (depth-conditioned multi-view) ===", flush=True)
    subprocess.run(
        [
            sys.executable, "pipeline_paint3d_stage1.py",
            "--sd_config", "controlnet/config/depth_based_inpaint_template.yaml",
            "--render_config", "paint3d/config/train_config_paint3d.py",
            "--mesh_path", str(mesh_path),
            "--prompt", prompt,
            "--ip_adapter_image_path", str(image_path),
            "--outdir", str(stage1_dir),
        ],
        cwd=repo_dir,
        check=True,
    )

    # Stage 1 writes res-0/albedo.png (1024x1024).
    stage1_albedo = stage1_dir / "res-0" / "albedo.png"
    if not stage1_albedo.exists():
        sys.exit(f"Stage 1 produced no albedo at {stage1_albedo}")

    print("=== Paint3D stage 2: UV refinement + tile super-resolution ===", flush=True)
    subprocess.run(
        [
            sys.executable, "pipeline_paint3d_stage2.py",
            "--sd_config", "controlnet/config/UV_based_inpaint_template.yaml",
            "--render_config", "paint3d/config/train_config_paint3d.py",
            "--mesh_path", str(mesh_path),
            "--texture_path", str(stage1_albedo),
            "--prompt", prompt,
            "--ip_adapter_image_path", str(image_path),
            "--outdir", str(stage2_dir),
        ],
        cwd=repo_dir,
        check=True,
    )

    # Stage 2 writes per-iteration result subdirs; pick the most-refined one.
    candidates = sorted(stage2_dir.rglob("mesh.obj"))
    if not candidates:
        # Fallback: stage 1 also writes a textured mesh.obj.
        candidates = sorted(stage1_dir.rglob("mesh.obj"))
    if not candidates:
        sys.exit("No mesh.obj produced by either stage")
    final_obj = candidates[-1]

    print(f"Converting {final_obj.name} -> {output_glb}", flush=True)
    import trimesh
    loaded = trimesh.load(str(final_obj), force='mesh', process=False)
    loaded.export(str(output_glb))
    print(f"Wrote: {output_glb}", flush=True)


if __name__ == "__main__":
    main()
"""


class Paint3DAdapter(ModelAdapter):
    info = ModelInfo(
        id="paint3d",
        display_name="Paint3D",
        description=(
            "Mesh-to-texture: takes an existing 3D mesh and a reference image, "
            "produces a textured GLB. Stable Diffusion 1.5 + ControlNet + "
            "IP-Adapter, two-stage (coarse + UV refinement)."
        ),
        best_for=(
            "Adding textures to an existing mesh — handcrafted, scanned, or "
            "shape-only output from `hunyuan3d-2`. Use when you need to texture "
            "a mesh from a reference image and don't need PBR channels."
        ),
        strengths=(
            "Apache 2.0 license — fully permissive, no gating",
            "Works on any mesh topology (.obj / .glb / .ply)",
            "Pairs with hunyuan3d-2 for high-quality geometry + textures",
        ),
        weaknesses=(
            "Albedo only — no PBR (no normal / roughness / metallic)",
            "Slow: ~5-10 min/asset, no CPU fallback",
            "Upstream unmaintained since mid-2024; we patch its dep pins",
        ),
        hardware=HardwareNeeds(
            min_gpu_vram_gb=6.0,
            recommended_gpu_vram_gb=10.0,
            cpu_fallback=False,
            cpu_speed_hint="not supported",
        ),
        homepage="https://github.com/OpenTexture/Paint3D",
        license_url=_LICENSE_URL,
        requires_hf_auth=False,
        inputs=(
            InputSpec(
                kind=InputKind.MESH,
                name="mesh",
                description=(
                    "Path to an existing 3D mesh (.obj, .glb, .ply, .off). "
                    "Non-OBJ inputs are auto-converted to OBJ internally by "
                    "Paint3D via trimesh."
                ),
                required=True,
            ),
            InputSpec(
                kind=InputKind.IMAGE,
                name="image",
                description=(
                    "Reference image describing the desired look. Used as "
                    "conditioning for IP-Adapter."
                ),
                required=True,
            ),
        ),
        output_extension=".glb",
        params=(
            ParamSpec(
                name="prompt",
                label="Text prompt",
                description=(
                    "Optional text guidance combined with the reference image. "
                    "Empty by default; the IP-Adapter handles image-only conditioning."
                ),
                kind=ParamKind.TEXT,
                default="",
            ),
        ),
    )

    # ---------- setup ----------

    def setup(self, *, force: bool = False) -> None:
        if self.is_installed and not force:
            info(f"'{self.model_id}' is already installed. Use --force to reinstall.")
            return

        toolchain_problems = check_build_toolchain()
        if toolchain_problems:
            raise RuntimeError(
                "Cannot install Paint3D — system requirements not met:\n  - "
                + "\n  - ".join(toolchain_problems)
            )

        model_dir = self.paths.model_dir(self.model_id)
        if force and model_dir.exists():
            warn(f"Removing existing installation at {model_dir}")
            shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        self._clone_repo()
        self._create_venv()
        self._install_dependencies()
        self._write_runner()
        self.mark_installed()
        success(f"'{self.model_id}' installed successfully.")

    def _clone_repo(self) -> None:
        repo_dir = self.paths.model_repo_dir(self.model_id)
        if repo_dir.exists():
            info(f"Repository already cloned at {repo_dir}")
            return
        info(f"Cloning {_REPO_URL} -> {repo_dir}")
        run_streaming(
            ["git", "clone", _REPO_URL, str(repo_dir)],
            description="Cloning Paint3D repository",
        )
        info(f"Checking out pinned commit {_REPO_COMMIT[:12]}")
        run_streaming(
            ["git", "-C", str(repo_dir), "checkout", "--quiet", _REPO_COMMIT],
            description="Pinning Paint3D to known-good commit",
        )

    def _create_venv(self) -> None:
        venv_dir = self.paths.model_venv_dir(self.model_id)
        if venv_dir.exists():
            info(f"Virtualenv already exists at {venv_dir}")
            return
        info(f"Creating virtualenv at {venv_dir}")
        run_streaming(
            ["uv", "venv", str(venv_dir), "--python", "3.11"],
            description="Creating isolated virtualenv via uv",
        )

    def _install_dependencies(self) -> None:
        venv_python = self._venv_python()

        info("Installing build prerequisites (setuptools, wheel)")
        run_streaming(
            [
                "uv", "pip", "install",
                "--python", str(venv_python),
                "-U", "setuptools", "wheel",
            ],
            description="Installing build prerequisites",
        )

        info(f"Installing PyTorch ({', '.join(_TORCH_PACKAGES)}) — large download")
        run_streaming(
            [
                "uv", "pip", "install",
                "--python", str(venv_python),
                *_TORCH_PACKAGES,
            ],
            description="Pre-installing torch/torchvision into the venv",
        )

        info("Installing kaolin from NVIDIA's wheel index (matches torch 2.4 / CUDA 12.1)")
        run_streaming(
            [
                "uv", "pip", "install",
                "--python", str(venv_python),
                _KAOLIN_PACKAGE,
                "--find-links", _KAOLIN_INDEX_URL,
            ],
            description="Installing kaolin",
        )

        # Modernized minimal dep set. We deliberately do NOT install upstream's
        # environment.yaml verbatim — half of it is dead weight inherited from
        # an SD fork (streamlit, pytorch-lightning, webdataset, kornia, etc.)
        # that the inference pipelines never import. Plus, the pinned
        # `diffusers==0.25.0` is broken with current huggingface_hub.
        info("Installing model dependencies (modernized stack)")
        run_streaming(
            [
                "uv", "pip", "install",
                "--python", str(venv_python),
                *_PIP_PACKAGES,
            ],
            description="Installing minimal Paint3D runtime deps",
        )

    def _write_runner(self) -> None:
        runner_path = self.paths.model_dir(self.model_id) / "runner.py"
        info(f"Writing runner script -> {runner_path}")
        runner_path.write_text(_RUNNER_SOURCE)

    # ---------- post-setup ----------

    # Paint3D's HF assets aren't gated; the diffusers pipelines download
    # them lazily on first run. No credentials to configure. Default no-op
    # post_setup from the base class is fine.

    # ---------- verify ----------

    def verify(self) -> list[str]:
        problems: list[str] = []
        problems.extend(check_build_toolchain())

        if not self.is_installed:
            problems.append("Not installed. Run: gen3dhub setup --model paint3d")
            return problems

        repo_dir = self.paths.model_repo_dir(self.model_id)
        if not (repo_dir / "pipeline_paint3d_stage1.py").exists():
            problems.append(
                f"Missing pipeline_paint3d_stage1.py in {repo_dir}. Reinstall with --force."
            )

        venv_python = self._venv_python()
        if not venv_python.exists():
            problems.append(f"Missing virtualenv interpreter at {venv_python}. Reinstall.")

        runner = self.paths.model_dir(self.model_id) / "runner.py"
        if not runner.exists():
            problems.append(f"Missing runner script at {runner}. Reinstall with --force.")

        return problems

    # ---------- run ----------

    def run(self, request: RunRequest) -> Path:
        problems = self.verify()
        if problems:
            raise RuntimeError("Pre-run checks failed:\n  - " + "\n  - ".join(problems))

        mesh = request.inputs.get("mesh")
        image = request.inputs.get("image")
        if mesh is None:
            raise ValueError("Paint3D requires a 'mesh' input (path to an existing 3D mesh).")
        if image is None:
            raise ValueError("Paint3D requires an 'image' input (reference image).")

        mesh_path = Path(mesh).expanduser().resolve()
        image_path = Path(image).expanduser().resolve()
        if not mesh_path.exists():
            raise FileNotFoundError(f"Input mesh not found: {mesh_path}")
        if not image_path.exists():
            raise FileNotFoundError(f"Reference image not found: {image_path}")

        default_name = f"{mesh_path.stem}_painted.glb"
        output_path = (request.output_path or Path.cwd() / default_name).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        venv_python = self._venv_python()
        runner = self.paths.model_dir(self.model_id) / "runner.py"

        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        if request.extra.get("force_cpu"):
            warn(
                "Paint3D does not support CPU mode (upstream pipelines hard-code "
                ".to('cuda')). Ignoring --cpu and attempting GPU inference."
            )

        # Forward user-supplied params to the runner via env vars.
        prompt_value = request.params.get("prompt")
        if isinstance(prompt_value, str) and prompt_value.strip():
            env["PAINT3D_PROMPT"] = prompt_value

        with tempfile.TemporaryDirectory(prefix="paint3d-") as _:
            info(f"Running Paint3D on {mesh_path.name} with reference {image_path.name}")
            try:
                run_streaming(
                    [
                        str(venv_python), str(runner),
                        str(mesh_path), str(image_path), str(output_path),
                    ],
                    env=env,
                    description="Running Paint3D (stage 1 + stage 2 + GLB convert)",
                )
            except Exception:
                warn(
                    "If the failure above mentions CUDA OOM, Paint3D is at the "
                    "edge for 8 GB GPUs even with cpu_offload. Close other GPU "
                    "apps and retry, or run on a larger GPU."
                )
                raise

            if not output_path.exists():
                raise RuntimeError(
                    f"Runner finished but no .glb was produced at {output_path}"
                )

        success(f"Wrote textured mesh -> {output_path}")
        return output_path

    # ---------- helpers ----------

    def _venv_python(self) -> Path:
        venv = self.paths.model_venv_dir(self.model_id)
        if sys.platform == "win32":
            return venv / "Scripts" / "python.exe"
        return venv / "bin" / "python"
