"""Adapter for TencentARC's InstantMesh (sparse-view image-to-3D).

Reference: https://github.com/TencentARC/InstantMesh
Model:     https://huggingface.co/TencentARC/InstantMesh

InstantMesh produces a textured mesh from a single image by:
  1. Generating six novel views with a Zero123++ finetune (diffusion UNet),
  2. Reconstructing geometry + appearance with a large transformer
     ("instant-mesh-large" by default; LRM-style sparse-view reconstruction),
  3. Marching cubes over the implicit field and (optionally) baking a UV
     texture map.

Native output is `.obj`. The adapter requests `.obj` from upstream's run.py
and lets gen3dhub's `--format` machinery transcode to `.glb` (or any other
target) when the user asks for it.

Versus the other adapters in this hub:
  - Higher VRAM than Hunyuan3D-2 (defaults around 16 GB; ~10 GB possible with
    smaller configs and `--view 4`), and noticeably slower than SF3D.
  - Trades the geometric ceiling of Hunyuan3D-2 for an end-to-end pipeline
    that ships textures in the same pass (no separate paint3d step needed).
  - Apache 2.0 license — fewer commercial caveats than Hunyuan3D-2.
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

_REPO_URL = "https://github.com/TencentARC/InstantMesh.git"
# Pinned for reproducibility. Bump deliberately when upgrading. Upstream has
# no tagged releases, so a commit SHA is the only stable handle.
_REPO_COMMIT = "08822c52fdc399b93ea00e4fa9e596344ed52ccc"

_HF_REPO_ID = "TencentARC/InstantMesh"
_LICENSE_URL = "https://github.com/TencentARC/InstantMesh/blob/main/LICENSE"

# Upstream README pins: "Python>=3.10, PyTorch>=2.1.0, and CUDA>=12.1".
# We match those exactly — `xformers==0.0.22.post7` requires `torch==2.1.0`,
# and the prebuilt nvdiffrast wheels likewise expect a matching CUDA runtime.
# Each model has its own venv so this never collides with another adapter.
_TORCH_PACKAGES: tuple[str, ...] = (
    "torch==2.1.0",
    "torchvision==0.16.0",
    "torchaudio==2.1.0",
)
_XFORMERS_PACKAGE = "xformers==0.0.22.post7"

# nvdiffrast isn't on PyPI — InstantMesh's requirements.txt installs it from
# the NVlabs git repo. We list it explicitly here so it is installed *after*
# torch is present (it imports torch at install time).
_NVDIFFRAST_PACKAGE = "git+https://github.com/NVlabs/nvdiffrast/"


_CONFIG_CHOICES: tuple[str, ...] = (
    "instant-mesh-large",
    "instant-mesh-base",
    "instant-nerf-large",
    "instant-nerf-base",
)


class InstantMeshAdapter(ModelAdapter):
    info = ModelInfo(
        id="instant-mesh",
        display_name="InstantMesh",
        description=(
            "Image-to-3D using TencentARC's sparse-view reconstruction model. "
            "Generates six novel views with a Zero123++ finetune, then "
            "reconstructs a textured mesh in one pass."
        ),
        best_for=(
            "End-to-end image-to-textured-mesh when you'd rather not chain a "
            "separate texturing step. Apache 2.0, friendlier license than "
            "Hunyuan3D-2."
        ),
        strengths=(
            "Textured mesh in a single pass (no paint3d follow-up needed)",
            "Apache 2.0 license — permissive, OSI-approved",
            "Multiple model variants exposed (large / base, mesh / NeRF)",
        ),
        weaknesses=(
            "Higher VRAM than Hunyuan3D-2 (~16 GB for the large config)",
            "GPU-only — no CPU fallback (upstream hardcodes CUDA)",
            "Slower than SF3D: ~30-60s/asset on a comfortable GPU",
        ),
        hardware=HardwareNeeds(
            # 4-view + `instant-mesh-base` fits in ~10 GB; the default 6-view
            # `instant-mesh-large` wants ~16 GB to stay comfortable.
            min_gpu_vram_gb=10.0,
            recommended_gpu_vram_gb=16.0,
            cpu_fallback=False,
            cpu_speed_hint="not supported",
        ),
        homepage="https://github.com/TencentARC/InstantMesh",
        license_url=_LICENSE_URL,
        requires_hf_auth=False,  # HF repo is public, no token required.
        inputs=(
            InputSpec(
                kind=InputKind.IMAGE,
                name="image",
                description=(
                    "Path to input image. RGBA cutouts are used as-is; RGB "
                    "photos are background-removed via rembg unless "
                    "'skip_rembg' is set."
                ),
                required=True,
            ),
        ),
        output_extension=".obj",  # upstream's native — gen3dhub converts on request
        params=(
            ParamSpec(
                name="config",
                label="Model variant",
                description=(
                    "Which reconstruction config to use. 'large' = better "
                    "quality, more VRAM. NeRF variants trade mesh quality for "
                    "smoother radiance fields."
                ),
                kind=ParamKind.SELECT,
                default="instant-mesh-large",
                choices=_CONFIG_CHOICES,
            ),
            ParamSpec(
                name="diffusion_steps",
                label="Diffusion steps",
                description=(
                    "Denoising iterations for the multi-view generator. "
                    "More = sharper views, more time."
                ),
                kind=ParamKind.INT,
                default=75,
            ),
            ParamSpec(
                name="seed",
                label="Random seed",
                description=(
                    "Same seed + same input = same output. Useful for "
                    "iterating without random variation."
                ),
                kind=ParamKind.INT,
                default=42,
            ),
            ParamSpec(
                name="view",
                label="Input views",
                description=(
                    "Number of views fed to the reconstruction model. 4 = "
                    "lower VRAM, slightly weaker geometry; 6 = default."
                ),
                kind=ParamKind.SELECT,
                default="6",
                choices=("4", "6"),
            ),
            ParamSpec(
                name="export_texmap",
                label="Export UV texture map",
                description=(
                    "Bake a UV texture instead of vertex colors. Recommended "
                    "for downstream DCC pipelines; slightly slower."
                ),
                kind=ParamKind.BOOL,
                default=False,
            ),
            ParamSpec(
                name="skip_rembg",
                label="Skip background removal",
                description=(
                    "Set when the input already has a clean alpha mask. "
                    "Maps to upstream's --no_rembg flag."
                ),
                kind=ParamKind.BOOL,
                default=False,
            ),
        ),
    )

    # ---------- setup ----------

    def setup(self, *, force: bool = False) -> None:
        if self.is_installed and not force:
            info(f"'{self.model_id}' is already installed. Use --force to reinstall.")
            return

        # nvdiffrast builds a C++ extension at install time, so the host needs
        # a working toolchain. Fail fast with a clear message instead of a
        # cryptic gcc-not-found error halfway through pip install.
        toolchain_problems = check_build_toolchain()
        if toolchain_problems:
            raise RuntimeError(
                "Cannot install InstantMesh — system requirements not met:\n  - "
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
            description="Cloning InstantMesh repository",
        )
        info(f"Checking out pinned commit {_REPO_COMMIT[:12]}")
        run_streaming(
            ["git", "-C", str(repo_dir), "checkout", "--quiet", _REPO_COMMIT],
            description="Pinning InstantMesh to known-good commit",
        )

    def _create_venv(self) -> None:
        venv_dir = self.paths.model_venv_dir(self.model_id)
        if venv_dir.exists():
            info(f"Virtualenv already exists at {venv_dir}")
            return
        info(f"Creating virtualenv at {venv_dir}")
        # InstantMesh requires Python >=3.10. We pin 3.10 because xformers
        # 0.0.22.post7 doesn't ship wheels for newer Python versions paired
        # with torch 2.1.0 — building from source there is finicky.
        run_streaming(
            ["uv", "venv", str(venv_dir), "--python", "3.10"],
            description="Creating isolated virtualenv via uv",
        )

    def _install_dependencies(self) -> None:
        repo_dir = self.paths.model_repo_dir(self.model_id)
        requirements = repo_dir / "requirements.txt"
        if not requirements.exists():
            raise FileNotFoundError(f"Expected {requirements} to exist after cloning")
        venv_python = self._venv_python()

        info("Installing build prerequisites (setuptools, wheel, ninja)")
        run_streaming(
            [
                "uv", "pip", "install",
                "--python", str(venv_python),
                "-U", "setuptools", "wheel", "ninja",
            ],
            description="Installing build prerequisites",
        )

        # Step 1: PyTorch first. The upstream README spells it out: "install
        # PyTorch first, then run pip install -r requirements.txt." nvdiffrast
        # and xformers both `import torch` at install time and will fail
        # without it.
        info(f"Installing PyTorch ({', '.join(_TORCH_PACKAGES)}) — large download")
        run_streaming(
            [
                "uv", "pip", "install",
                "--python", str(venv_python),
                *_TORCH_PACKAGES,
            ],
            description="Pre-installing torch/torchvision/torchaudio into the venv",
        )

        # Step 2: xformers. Upstream pins 0.0.22.post7, which requires
        # torch==2.1.0 — installing it separately keeps any later resolver
        # work from second-guessing the pin.
        info(f"Installing {_XFORMERS_PACKAGE}")
        run_streaming(
            [
                "uv", "pip", "install",
                "--python", str(venv_python),
                _XFORMERS_PACKAGE,
            ],
            description="Installing xformers",
        )

        # Step 3: requirements.txt. --no-build-isolation lets nvdiffrast and
        # any other torch-aware source builds find the already-installed
        # torch instead of failing in an isolated sandbox.
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

        # Step 4: ensure nvdiffrast is present even if upstream's
        # requirements.txt evolves. Idempotent — uv pip skips when already
        # installed at the requested ref.
        info("Ensuring nvdiffrast is installed")
        run_streaming(
            [
                "uv", "pip", "install",
                "--python", str(venv_python),
                "--no-build-isolation",
                _NVDIFFRAST_PACKAGE,
            ],
            description="Installing nvdiffrast from NVlabs",
        )

    # ---------- post-setup ----------

    # No credentials to configure — the HF repo is public. Default no-op
    # inherited from ModelAdapter is fine.

    # ---------- verify ----------

    def verify(self) -> list[str]:
        problems: list[str] = []
        problems.extend(check_build_toolchain())

        if not self.is_installed:
            problems.append("Not installed. Run: gen3dhub setup --model instant-mesh")
            return problems

        repo_dir = self.paths.model_repo_dir(self.model_id)
        if not (repo_dir / "run.py").exists():
            problems.append(f"Missing run.py in {repo_dir}. Reinstall with --force.")

        if not (repo_dir / "configs" / "instant-mesh-large.yaml").exists():
            problems.append(
                f"Missing configs/ directory in {repo_dir}. Reinstall with --force."
            )

        venv_python = self._venv_python()
        if not venv_python.exists():
            problems.append(f"Missing virtualenv interpreter at {venv_python}. Reinstall.")

        return problems

    # ---------- run ----------

    def run(self, request: RunRequest) -> Path:
        problems = self.verify()
        if problems:
            raise RuntimeError("Pre-run checks failed:\n  - " + "\n  - ".join(problems))

        image = request.inputs.get("image")
        if image is None:
            raise ValueError("InstantMesh requires an 'image' input.")
        image_path = Path(image).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        output_path = (request.output_path or Path.cwd() / f"{image_path.stem}.obj").expanduser()
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        repo_dir = self.paths.model_repo_dir(self.model_id)
        venv_python = self._venv_python()

        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        if request.extra.get("force_cpu"):
            # Upstream's run.py hardcodes `device = torch.device('cuda')` —
            # there's no clean route to CPU without patching the script. Be
            # upfront rather than letting torch raise a less helpful error.
            raise RuntimeError(
                "InstantMesh does not support CPU inference (upstream hardcodes "
                "CUDA). Re-run without --cpu, or pick a different model (SF3D and "
                "Hunyuan3D-2 both have CPU fallbacks)."
            )

        # Resolve params, falling back to ParamSpec defaults so the run is
        # deterministic even when the caller omits everything.
        params = request.params
        config_name = str(params.get("config", "instant-mesh-large"))
        if config_name not in _CONFIG_CHOICES:
            raise ValueError(
                f"Unknown InstantMesh config '{config_name}'. "
                f"Choose one of: {', '.join(_CONFIG_CHOICES)}."
            )
        diffusion_steps = int(params.get("diffusion_steps", 75))
        seed = int(params.get("seed", 42))
        view = int(params.get("view", 6))
        export_texmap = bool(params.get("export_texmap", False))
        skip_rembg = bool(params.get("skip_rembg", False))

        cli_extra: list[str] = []
        if skip_rembg:
            cli_extra.append("--no_rembg")
        if export_texmap:
            cli_extra.append("--export_texmap")

        with tempfile.TemporaryDirectory(prefix="instant-mesh-") as staging:
            staging_dir = Path(staging)
            info(f"Running inference on {image_path.name} ({config_name})")
            try:
                run_streaming(
                    [
                        str(venv_python), "run.py",
                        f"configs/{config_name}.yaml",
                        str(image_path),
                        "--output_path", str(staging_dir),
                        "--diffusion_steps", str(diffusion_steps),
                        "--seed", str(seed),
                        "--view", str(view),
                        *cli_extra,
                    ],
                    cwd=repo_dir,
                    env=env,
                    description="Running InstantMesh inference",
                )
            except Exception:
                warn(
                    "If the failure above is `torch.OutOfMemoryError: CUDA out "
                    "of memory`, your GPU doesn't have enough free VRAM. Try "
                    "`--param config=instant-mesh-base --param view=4` for the "
                    "smaller config, or close other GPU-using apps "
                    "(`nvidia-smi`). InstantMesh has no CPU mode."
                )
                raise

            # Upstream writes to `<output_path>/<config_name>/meshes/<stem>.obj`.
            produced = staging_dir / config_name / "meshes" / f"{image_path.stem}.obj"
            if not produced.exists():
                # Fall back to a broader scan in case the layout changes.
                matches = sorted(staging_dir.rglob("*.obj"))
                if not matches:
                    raise RuntimeError(
                        f"Inference completed but no .obj file was produced under {staging_dir}"
                    )
                produced = matches[0]

            # `.obj` carries sidecars (.mtl + texture PNGs) when --export_texmap
            # is on. Copy them all next to the output file so downstream tools
            # find them by relative path.
            self._copy_obj_with_sidecars(produced, output_path)

        success(f"Wrote 3D mesh → {output_path}")
        return output_path

    # ---------- helpers ----------

    def _venv_python(self) -> Path:
        venv = self.paths.model_venv_dir(self.model_id)
        if sys.platform == "win32":
            return venv / "Scripts" / "python.exe"
        return venv / "bin" / "python"

    @staticmethod
    def _copy_obj_with_sidecars(src: Path, dst: Path) -> None:
        """Copy `src.obj` to `dst`, bringing along any `.mtl` and texture
        sidecars that share the basename. InstantMesh's --export_texmap path
        writes `<name>.obj` + `<name>.mtl` + `<name>_*.png` next to each
        other; preserving that layout keeps relative refs in the .mtl valid.
        """
        shutil.copy2(src, dst)
        src_dir = src.parent
        stem = src.stem
        for sidecar in src_dir.iterdir():
            if sidecar == src or not sidecar.is_file():
                continue
            if sidecar.stem == stem or sidecar.name.startswith(f"{stem}_"):
                shutil.copy2(sidecar, dst.parent / sidecar.name)
