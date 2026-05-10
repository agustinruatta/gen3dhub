"""Adapter for Tencent's Hunyuan3D-2 (image-to-3D, mini variant).

Reference: https://github.com/Tencent-Hunyuan/Hunyuan3D-2
Model:     https://huggingface.co/tencent/Hunyuan3D-2mini

Scope (v1): SHAPE GENERATION ONLY.
  - Uses the 0.6B "mini" DiT variant, comfortable on 8 GB-class GPUs.
  - The matching `Hunyuan3D-Paint` texture pipeline (1.3B, pushes total to
    ~16 GB) is intentionally *not* installed. Adding texture support later
    would also require building two custom C++/CUDA extensions
    (`custom_rasterizer`, `differentiable_renderer`).
  - Output is a textureless `.glb` mesh (geometry + vertex colors only).

Why ship this on top of stable-fast-3d:
  - Higher shape fidelity (DiT-based generative model vs SF3D's transformer
    backbone) at the cost of ~30s/asset vs SF3D's ~1s.
  - Different model class — useful for cross-checking and for cases where
    SF3D's particular bias toward objet-centric inputs hurts.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from gen3dhub.console import info, success, warn
from gen3dhub.models.base import (
    InputKind,
    InputSpec,
    ModelAdapter,
    ModelInfo,
    RunRequest,
)
from gen3dhub.utils.process import run_streaming
from gen3dhub.utils.system import check_build_toolchain

_REPO_URL = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git"
# Pinned for reproducibility. Bump deliberately when upgrading. Upstream has
# no tags or releases — main is the only branch — so a commit SHA is the
# only stable handle.
_REPO_COMMIT = "f8db63096c8282cb27354314d896feba5ba6ff8a"

_HF_REPO_ID = "tencent/Hunyuan3D-2mini"
_HF_SUBFOLDER = "hunyuan3d-dit-v2-mini"
_HF_VARIANT = "fp16"
_LICENSE_URL = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/LICENSE"

# Hunyuan3D-2's requirements.txt deliberately omits torch/torchvision (the
# upstream README says "install Pytorch via the official site first"). Pin the
# same versions as SF3D for consistency — both work on CUDA 12.1 wheels.
_TORCH_PACKAGES: tuple[str, ...] = ("torch==2.4.1", "torchvision==0.19.1")


# Runner script written into the model dir during setup. Imports hy3dgen
# (installed editable in the venv) and invokes the shape pipeline.
#
# Kept as a string constant so a `setup --force` always rewrites the latest
# version, and so the upstream `repo/` stays clean of our own files.
_RUNNER_SOURCE = """\
'''gen3dhub runner for Hunyuan3D-2 (shape-only, mini variant).

Usage: python runner.py <image_path> <output_path>
Reads device from HUNYUAN3D_DEVICE env (default "cuda"; set to "cpu" to force).
Reads seed   from HUNYUAN3D_SEED   env (default 12345).
'''
import os
import sys

import torch
from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: runner.py <image_path> <output_path>")
    image_path, output_path = sys.argv[1], sys.argv[2]

    device = os.environ.get("HUNYUAN3D_DEVICE", "cuda")
    seed = int(os.environ.get("HUNYUAN3D_SEED", "12345"))

    img = Image.open(image_path)
    # If the user provided an alpha-having image (cutout), trust it. Otherwise
    # run rembg to isolate the foreground — the shape model needs a clean
    # subject on a transparent background.
    if img.mode == "RGBA":
        image = img
    else:
        image = BackgroundRemover()(img.convert("RGB"))

    print(f"Loading Hunyuan3D-2mini on device={device}…", flush=True)
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mini",
        subfolder="hunyuan3d-dit-v2-mini",
        variant="fp16",
        use_safetensors=True,
        device=device,
    )

    print("Running shape generation…", flush=True)
    mesh = pipe(
        image=image,
        num_inference_steps=30,
        octree_resolution=380,
        num_chunks=20000,
        generator=torch.manual_seed(seed),
        output_type="trimesh",
    )[0]

    mesh.export(output_path)
    print(f"Wrote: {output_path}", flush=True)


if __name__ == "__main__":
    main()
"""


class Hunyuan3D2Adapter(ModelAdapter):
    info = ModelInfo(
        id="hunyuan3d-2",
        display_name="Hunyuan3D-2 (mini)",
        description=(
            "Image-to-3D using Tencent's DiT-based shape model (mini 0.6B variant). "
            "Generates higher-fidelity geometry than SF3D."
        ),
        strengths=(
            "Best geometric fidelity at this VRAM tier",
            "Public on Hugging Face (no gating, no token required)",
            "Generative DiT — handles harder shapes than feed-forward models",
        ),
        weaknesses=(
            "Slower: ~30s per asset on GPU, 10+ min on CPU",
            "Shape-only in this adapter — no textures yet",
            "Tencent community license restricts EU/UK/KR",
        ),
        homepage="https://huggingface.co/tencent/Hunyuan3D-2mini",
        license_url=_LICENSE_URL,
        requires_hf_auth=False,  # Repo is NOT gated — no token required.
        inputs=(
            InputSpec(
                kind=InputKind.IMAGE,
                name="image",
                description=(
                    "Path to input image. RGBA cutouts are used as-is; RGB photos "
                    "go through rembg for foreground extraction."
                ),
                required=True,
            ),
        ),
        output_extension=".glb",
    )

    # ---------- setup ----------

    def setup(self, *, force: bool = False) -> None:
        if self.is_installed and not force:
            info(f"'{self.model_id}' is already installed. Use --force to reinstall.")
            return

        toolchain_problems = check_build_toolchain()
        if toolchain_problems:
            raise RuntimeError(
                "Cannot install Hunyuan3D-2 — system requirements not met:\n  - "
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
        self._show_license_notice()
        self.mark_installed()
        success(f"'{self.model_id}' installed successfully.")

    def _clone_repo(self) -> None:
        repo_dir = self.paths.model_repo_dir(self.model_id)
        if repo_dir.exists():
            info(f"Repository already cloned at {repo_dir}")
            return
        info(f"Cloning {_REPO_URL} → {repo_dir}")
        run_streaming(
            ["git", "clone", _REPO_URL, str(repo_dir)],
            description="Cloning Hunyuan3D-2 repository",
        )
        info(f"Checking out pinned commit {_REPO_COMMIT[:12]}")
        run_streaming(
            ["git", "-C", str(repo_dir), "checkout", "--quiet", _REPO_COMMIT],
            description="Pinning Hunyuan3D-2 to known-good commit",
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
        repo_dir = self.paths.model_repo_dir(self.model_id)
        requirements = repo_dir / "requirements.txt"
        if not requirements.exists():
            raise FileNotFoundError(f"Expected {requirements} to exist after cloning")
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

        # Step 1: PyTorch first. Same rationale as SF3D — Hunyuan3D-2's
        # requirements.txt and setup.py both reference torch but don't
        # install it; the README directs the user to install it from the
        # PyTorch wheel index manually.
        info(f"Installing PyTorch ({', '.join(_TORCH_PACKAGES)}) — large download")
        run_streaming(
            [
                "uv", "pip", "install",
                "--python", str(venv_python),
                *_TORCH_PACKAGES,
            ],
            description="Pre-installing torch/torchvision into the venv",
        )

        # Step 2: requirements.txt. Use --no-build-isolation defensively —
        # any future build extension that needs torch at compile time can
        # then find it. Doesn't hurt for the current pure-Python deps.
        info("Installing model dependencies (this may take several minutes)")
        run_streaming(
            [
                "uv", "pip", "install",
                "--python", str(venv_python),
                "--no-build-isolation",
                "-r", str(requirements),
            ],
            cwd=repo_dir,
            description="Installing requirements.txt without build isolation",
        )

        # Step 3: editable install of hy3dgen so the runner can `import
        # hy3dgen.shapegen` from anywhere (without depending on cwd).
        info("Installing hy3dgen package (editable)")
        run_streaming(
            [
                "uv", "pip", "install",
                "--python", str(venv_python),
                "--no-build-isolation",
                "-e", ".",
            ],
            cwd=repo_dir,
            description="Installing hy3dgen as editable package",
        )

        # We deliberately skip the texgen C++/CUDA extensions
        # (`hy3dgen/texgen/custom_rasterizer/`, `differentiable_renderer/`).
        # Those are only needed for texture generation, which is out of scope
        # for the 8 GB VRAM target.

    def _write_runner(self) -> None:
        runner_path = self.paths.model_dir(self.model_id) / "runner.py"
        info(f"Writing runner script → {runner_path}")
        runner_path.write_text(_RUNNER_SOURCE)

    def _show_license_notice(self) -> None:
        # Not enforced — the HF repo isn't gated, the user has already chosen
        # to install. But surface the restrictive terms so they know what
        # they accepted by running setup. SF3D doesn't need this because HF's
        # gating UI handles it; here, we're the only place the license is
        # mentioned.
        warn(
            "Hunyuan3D-2 is released under the Tencent Hunyuan 3D 2.0 Community "
            "License. Notable terms:\n"
            "  - Use is restricted in EU, UK, and South Korea (geographic carve-out).\n"
            "  - Commercial use up to 1M MAU; beyond that, contact Tencent.\n"
            "  - Outputs may NOT be used to train competing models.\n"
            f"  - Full text: {_LICENSE_URL}\n"
            "By using this adapter you accept these terms."
        )

    # ---------- post-setup ----------

    # No credentials to configure — the HF repo is public. Default no-op
    # inherited from ModelAdapter is fine.

    # ---------- verify ----------

    def verify(self) -> list[str]:
        problems: list[str] = []
        problems.extend(check_build_toolchain())

        if not self.is_installed:
            problems.append("Not installed. Run: gen3dhub setup --model hunyuan3d-2")
            return problems

        repo_dir = self.paths.model_repo_dir(self.model_id)
        if not (repo_dir / "hy3dgen").exists():
            problems.append(f"Missing hy3dgen package in {repo_dir}. Reinstall with --force.")

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

        image = request.inputs.get("image")
        if image is None:
            raise ValueError("Hunyuan3D-2 requires an 'image' input.")
        image_path = Path(image).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        output_path = (request.output_path or Path.cwd() / f"{image_path.stem}.glb").expanduser()
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        venv_python = self._venv_python()
        runner = self.paths.model_dir(self.model_id) / "runner.py"

        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        force_cpu = bool(request.extra.get("force_cpu"))
        if force_cpu:
            # Hunyuan3D-2's pipeline takes a `device` kwarg directly. Use the
            # runner's HUNYUAN3D_DEVICE env var rather than CUDA_VISIBLE_DEVICES,
            # so any auxiliary CUDA-using tools (none currently, but future-
            # proofing) keep their GPU visibility.
            env["HUNYUAN3D_DEVICE"] = "cpu"
            info(
                "Forcing CPU inference. Hunyuan3D-2 on CPU is much slower than "
                "SF3D's CPU mode (expect 10+ minutes); use only if VRAM is "
                "the bottleneck."
            )

        with tempfile.TemporaryDirectory(prefix="hunyuan3d-") as staging:
            staging_dir = Path(staging)
            staging_output = staging_dir / "result.glb"
            info(f"Running inference on {image_path.name}")
            try:
                run_streaming(
                    [str(venv_python), str(runner), str(image_path), str(staging_output)],
                    env=env,
                    description="Running Hunyuan3D-2 shape generation",
                )
            except Exception:
                if not force_cpu:
                    warn(
                        "If the failure above is `torch.OutOfMemoryError: CUDA out "
                        "of memory`, your GPU doesn't have enough free VRAM. Either "
                        "close other GPU-using apps (`nvidia-smi`) or re-run with "
                        "--cpu. Note Hunyuan3D-2 on CPU is slow (~10+ min)."
                    )
                raise
            if not staging_output.exists():
                raise RuntimeError(
                    f"Inference completed but no .glb file was produced at {staging_output}"
                )
            shutil.copy2(staging_output, output_path)

        success(f"Wrote 3D mesh → {output_path}")
        return output_path

    # ---------- helpers ----------

    def _venv_python(self) -> Path:
        venv = self.paths.model_venv_dir(self.model_id)
        if sys.platform == "win32":
            return venv / "Scripts" / "python.exe"
        return venv / "bin" / "python"
