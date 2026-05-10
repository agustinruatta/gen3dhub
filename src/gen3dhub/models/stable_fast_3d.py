"""Adapter for Stability AI's Stable Fast 3D (image-to-3D mesh) model.

Reference: https://github.com/Stability-AI/stable-fast-3d
Model: https://huggingface.co/stabilityai/stable-fast-3d
"""

from __future__ import annotations

import getpass
import os
import shutil
import sys
import tempfile
from pathlib import Path

from gen3dhub.console import error, info, success, warn
from gen3dhub.models.base import (
    InputKind,
    InputSpec,
    ModelAdapter,
    ModelInfo,
    RunRequest,
)
from gen3dhub.utils.process import run_streaming
from gen3dhub.utils.system import check_build_toolchain

_REPO_URL = "https://github.com/Stability-AI/stable-fast-3d.git"
# Pinned for reproducibility. Bump deliberately when upgrading.
_REPO_COMMIT = "ff21fc491b4dc5314bf6734c7c0dabd86b5f5bb2"
_HF_REPO_ID = "stabilityai/stable-fast-3d"

# SF3D's requirements.txt deliberately omits torch/torchvision (the user is
# expected to install them per-platform). We pin known-working versions here so
# every install is reproducible. The README recommends PyTorch >= 2.4.0.
_TORCH_PACKAGES: tuple[str, ...] = ("torch==2.4.1", "torchvision==0.19.1")


class StableFast3DAdapter(ModelAdapter):
    info = ModelInfo(
        id="stable-fast-3d",
        display_name="Stable Fast 3D",
        summary="Image-to-3D: generates a textured GLB mesh from a single image (~1s on GPU).",
        homepage="https://huggingface.co/stabilityai/stable-fast-3d",
        license_url="https://huggingface.co/stabilityai/stable-fast-3d",
        requires_hf_auth=True,
        inputs=(
            InputSpec(
                kind=InputKind.IMAGE,
                name="image",
                description="Path to the input image (recommended 512x512).",
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

        # Fail fast (and clearly) if the host is missing the toolchain SF3D needs
        # to build texture_baker / uv_unwrapper. Otherwise the user gets a
        # cryptic gcc-not-found error mid-install.
        toolchain_problems = check_build_toolchain()
        if toolchain_problems:
            raise RuntimeError(
                "Cannot install Stable Fast 3D — system requirements not met:\n  - "
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
        # Full clone so we can check out the pinned commit afterwards. The repo
        # is small (~25 MiB), so we don't bother with partial-clone tricks.
        run_streaming(
            ["git", "clone", _REPO_URL, str(repo_dir)],
            description="Cloning Stable Fast 3D repository",
        )
        info(f"Checking out pinned commit {_REPO_COMMIT[:12]}")
        run_streaming(
            ["git", "-C", str(repo_dir), "checkout", "--quiet", _REPO_COMMIT],
            description="Pinning SF3D to known-good commit",
        )

    def _create_venv(self) -> None:
        venv_dir = self.paths.model_venv_dir(self.model_id)
        if venv_dir.exists():
            info(f"Virtualenv already exists at {venv_dir}")
            return
        info(f"Creating virtualenv at {venv_dir}")
        # SF3D supports Python >=3.8; we pin 3.11 for broader wheel availability.
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
                "-U", "setuptools==69.5.1", "wheel",
            ],
            description="Installing build prerequisites",
        )

        # Step 1: install PyTorch + torchvision. SF3D's requirements.txt omits
        # them (the project expects the user to install per-platform). We pin
        # versions for reproducibility. This also seeds the venv with torch so
        # that step 2 can build the local C++/CUDA extensions against it.
        info(f"Installing PyTorch ({', '.join(_TORCH_PACKAGES)}) — large download")
        run_streaming(
            [
                "uv", "pip", "install",
                "--python", str(venv_python),
                *_TORCH_PACKAGES,
            ],
            description="Pre-installing torch/torchvision into the venv",
        )

        # Step 2: install the rest. We pass --no-build-isolation because the
        # local packages `./texture_baker/` and `./uv_unwrapper/` `import torch`
        # in their setup.py without declaring it in build-system.requires. With
        # build isolation (uv's default), the build sandbox can't see torch and
        # the wheels fail to compile. Without isolation, builds run inside the
        # already-prepared venv where torch is available.
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

    # ---------- post-setup (credentials) ----------

    def post_setup(self, *, interactive: bool) -> None:
        """Ensure a Hugging Face token is configured and grants repo access.

        SF3D weights live behind a gated repo on Hugging Face. The user must
        (a) accept the license in a browser, (b) issue a Read token, and
        (c) make that token available to local HF tooling.

        This hook handles (c) only — (a) and (b) require human action in a
        browser. We:
          - skip silently if a token is already set AND it grants repo access,
          - in interactive mode, prompt for the token (input is hidden) and
            persist it via `huggingface_hub.login()`, which writes
            ~/.cache/huggingface/token with mode 0600 — the canonical place
            every HF library reads from automatically,
          - in non-interactive mode, print a concrete next-step message and
            return cleanly (no prompt, no crash).
        """
        if _hf_token_available() and _hf_can_access_repo(_HF_REPO_ID):
            return

        if not interactive:
            warn(
                "Stable Fast 3D needs a Hugging Face access token, but none is "
                "configured (or it doesn't grant repo access). Set HF_TOKEN, run "
                "`huggingface-cli login`, or run `gen3dhub setup -m stable-fast-3d` "
                "again from an interactive terminal to be prompted."
            )
            return

        info("")
        info("[bold]Stable Fast 3D needs a Hugging Face token.[/bold]")
        info(
            f"  1. Open  https://huggingface.co/{_HF_REPO_ID}  "
            "and click 'Agree and access repository'."
        )
        info(
            "  2. Open  https://huggingface.co/settings/tokens  "
            "and create a token with the 'Read' role."
        )
        info(
            "  3. Paste the token here. It will be saved to "
            "~/.cache/huggingface/token (mode 0600)."
        )
        info("")

        try:
            token = getpass.getpass("HF token (input hidden, blank to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            warn("\nSkipped. Configure later with `huggingface-cli login`.")
            return

        if not token:
            warn(
                "Empty input — token not saved. You can still configure it later:\n"
                "  - export HF_TOKEN=hf_xxx in your shell, or\n"
                "  - run `huggingface-cli login`,\n"
                "then verify with `gen3dhub doctor -m stable-fast-3d`."
            )
            return

        try:
            from huggingface_hub import login

            login(token=token, add_to_git_credential=False)
        except Exception as exc:
            error(f"Failed to save token: {exc}")
            warn("You can still set HF_TOKEN in your environment as a fallback.")
            return

        # huggingface_hub 1.x writes the token file with the umask-default
        # mode (often 0644). Tighten to 0600 so the secret isn't readable by
        # other users on the system.
        _harden_token_file_mode()

        if _hf_can_access_repo(_HF_REPO_ID):
            success("Token saved and verified — access to the gated repository confirmed.")
        else:
            warn(
                "Token saved, but it does NOT grant access to "
                f"https://huggingface.co/{_HF_REPO_ID} yet.\n"
                "Make sure step (1) is done — accept the license on that page — "
                "and re-run `gen3dhub doctor -m stable-fast-3d`."
            )

    # ---------- verify ----------

    def verify(self) -> list[str]:
        problems: list[str] = []

        # System-level checks first — relevant even before installation, so the
        # user can fix toolchain issues before attempting setup.
        problems.extend(check_build_toolchain())

        if not self.is_installed:
            problems.append("Not installed. Run: gen3dhub setup --model stable-fast-3d")
            return problems

        repo_dir = self.paths.model_repo_dir(self.model_id)
        if not (repo_dir / "run.py").exists():
            problems.append(f"Missing run.py in {repo_dir}. Reinstall with --force.")

        venv_python = self._venv_python()
        if not venv_python.exists():
            problems.append(f"Missing virtualenv interpreter at {venv_python}. Reinstall.")

        if not _hf_token_available():
            problems.append(
                "No Hugging Face token detected. Run `huggingface-cli login` "
                "or set HF_TOKEN. Required because the model is gated."
            )
        elif not _hf_can_access_repo(_HF_REPO_ID):
            problems.append(
                f"Hugging Face token is set but cannot access '{_HF_REPO_ID}'. "
                f"Accept the license at {self.info.license_url}."
            )

        return problems

    # ---------- run ----------

    def run(self, request: RunRequest) -> Path:
        problems = self.verify()
        if problems:
            raise RuntimeError("Pre-run checks failed:\n  - " + "\n  - ".join(problems))

        image = request.inputs.get("image")
        if image is None:
            raise ValueError("Stable Fast 3D requires an 'image' input.")
        image_path = Path(image).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        output_path = (request.output_path or Path.cwd() / f"{image_path.stem}.glb").expanduser()
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        repo_dir = self.paths.model_repo_dir(self.model_id)
        venv_python = self._venv_python()

        env = os.environ.copy()
        # Reduce CUDA memory fragmentation. Important for 8 GB-class GPUs
        # because SF3D's working set (~6 GB) lives close to the limit, and
        # without this PyTorch leaves enough scattered free pages to fail a
        # single contiguous allocation. Harmless on bigger GPUs.
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        force_cpu = bool(request.extra.get("force_cpu"))
        if force_cpu:
            # Hide every CUDA device from the subprocess. SF3D autodetects this
            # and falls back to its CPU code path. ~10-60x slower than GPU but
            # always works, regardless of VRAM pressure.
            env["CUDA_VISIBLE_DEVICES"] = ""
            info("Forcing CPU inference (--cpu / Force CPU enabled).")

        with tempfile.TemporaryDirectory(prefix="sf3d-") as staging:
            staging_dir = Path(staging)
            info(f"Running inference on {image_path.name}")
            try:
                run_streaming(
                    [
                        str(venv_python), "run.py",
                        str(image_path),
                        "--output-dir", str(staging_dir),
                    ],
                    cwd=repo_dir,
                    env=env,
                    description="Running Stable Fast 3D inference",
                )
            except Exception:
                if not force_cpu:
                    warn(
                        "If the failure above is `torch.OutOfMemoryError: CUDA out of "
                        "memory`, your GPU doesn't have enough free VRAM right now. "
                        "Either close other GPU-using apps (check with `nvidia-smi`) "
                        "or re-run with --cpu (CLI) / 'Force CPU' (TUI)."
                    )
                raise
            produced = _find_first_glb(staging_dir)
            if produced is None:
                raise RuntimeError(
                    f"Inference completed but no .glb file was produced under {staging_dir}"
                )
            shutil.copy2(produced, output_path)

        success(f"Wrote 3D mesh → {output_path}")
        return output_path

    # ---------- helpers ----------

    def _venv_python(self) -> Path:
        venv = self.paths.model_venv_dir(self.model_id)
        if sys.platform == "win32":
            return venv / "Scripts" / "python.exe"
        return venv / "bin" / "python"


def _hf_token_available() -> bool:
    if any(os.environ.get(name) for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")):
        return True
    try:
        # huggingface_hub.get_token() is the canonical, version-agnostic API.
        # The legacy HfFolder shim was removed in huggingface_hub 1.x.
        from huggingface_hub import get_token

        return bool(get_token())
    except Exception:
        return False


def _hf_can_access_repo(repo_id: str) -> bool:
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

        HfApi().model_info(repo_id)
        return True
    except (GatedRepoError, RepositoryNotFoundError):
        return False
    except Exception:
        # Network errors and other failures shouldn't false-fail the check.
        return True


def _find_first_glb(root: Path) -> Path | None:
    matches = sorted(root.rglob("*.glb"))
    return matches[0] if matches else None


def _harden_token_file_mode() -> None:
    """Set ~/.cache/huggingface/token (or $HF_HOME/token) to mode 0600.

    `huggingface_hub.login()` in 1.x leaves the file with the umask-default
    mode (e.g. 0644), which means other local users can read the secret.
    Best-effort: silently skip on platforms / file systems where chmod is
    a no-op (e.g. Windows).
    """
    from contextlib import suppress

    hf_home_env = os.environ.get("HF_HOME")
    hf_home = (
        Path(hf_home_env).expanduser()
        if hf_home_env
        else Path.home() / ".cache" / "huggingface"
    )
    token_file = hf_home / "token"
    if not token_file.exists():
        return
    with suppress(OSError):
        token_file.chmod(0o600)
