# gen3dhub

A console hub for AI models that generate 3D assets — image-to-3D, text-to-3D,
and similar. Usable both as an interactive TUI and as a flag-driven CLI, with
a uniform interface across models.

Each model is installed into its own isolated virtual environment so models
with incompatible dependency trees (different PyTorch versions, custom CUDA
extensions, etc.) coexist on the same machine without conflict.

## Supported models

| ID                | Type            | Input | Output |
|-------------------|-----------------|-------|--------|
| `stable-fast-3d`  | image → 3D mesh | image | `.glb` |

Adding new models is a matter of writing one adapter file (see *Adding a model*
below). The CLI surface does not need to change.

## Requirements

### Universal (any Linux distro / macOS)

- **Python** 3.11+ for the host process. (Per-model venvs use whatever Python
  the adapter pins — `uv` will download the right one if it's not installed.)
- **uv** (`https://docs.astral.sh/uv/`) — used both to manage this project and
  the per-model isolated virtualenvs.
- **git** — used to clone model source repositories on first install.
- A **Hugging Face** account for gated models (e.g. Stable Fast 3D). The
  easiest path is running `gen3dhub setup` in an interactive terminal — its
  post-setup hook prompts for the token (input hidden) and stores it via
  `huggingface_hub.login()` at `~/.cache/huggingface/token` (mode 0600), the
  canonical location every HF library reads from automatically. Alternatives:
  `huggingface-cli login`, or `export HF_TOKEN=hf_xxx` in the shell.
- A **GPU** with CUDA is strongly recommended for inference; CPU is supported
  as a fallback by some models (slower).

### Per-distro system dependencies

Some adapters (currently Stable Fast 3D) build C/C++ extensions during
installation, so the host needs a working toolchain. The tool detects this
automatically and refuses to start the install with a clear error if anything
is missing — `gen3dhub doctor` shows the exact problem and a
copy-pasteable command for your distro.

| Distro family            | One-time install                                                |
|--------------------------|-----------------------------------------------------------------|
| Ubuntu / Debian / Mint   | `sudo apt install build-essential python3-dev git`              |
| Arch / Manjaro / EndeavourOS | `sudo pacman -S --needed base-devel git`                    |
| Fedora / RHEL / Rocky    | `sudo dnf groupinstall 'Development Tools' && sudo dnf install python3-devel git` |
| openSUSE                 | `sudo zypper install -t pattern devel_C_C++ && sudo zypper install git` |
| macOS                    | `xcode-select --install` (CLT includes git)                     |

`uv` itself is one curl away on every platform:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For GPU acceleration (NVIDIA), the **driver** must be installed via the
distro's package manager (e.g. `nvidia-driver-XXX` on Ubuntu, `nvidia` on
Arch). The CUDA *runtime* libraries are bundled with the PyTorch wheels —
you do **not** need to install CUDA system-wide.

If you skip the GPU setup, SF3D and similar models fall back to CPU
automatically — slower, but works the same on any distro.

## Installation

```bash
git clone <this-repo> gen3dhub
cd gen3dhub
```

There are three ways to launch the tool:

**1. Launcher script (easiest — auto-bootstraps on first run):**

```bash
./gen3dhub --help
./gen3dhub            # interactive TUI menu
```

The `./gen3dhub` shell script in the project root runs `uv sync` the
first time (or whenever `pyproject.toml` changes) and then forwards every
argument to the CLI. Nothing else to set up.

**2. Through `uv run` (explicit):**

```bash
uv sync
uv run gen3dhub --help
```

**3. Installed globally so you can run `gen3dhub` from anywhere:**

```bash
./gen3dhub install         # wraps `uv tool install .`
gen3dhub --help            # works from any directory
```

The `install` subcommand of the launcher uses `uv tool install` under the
hood: it builds the project into a frozen, isolated environment and drops a
launcher at `~/.local/bin/gen3dhub`. After running it once, you can call
`gen3dhub …` from any shell.

If `gen3dhub` is not found after install, ensure the uv tool bin dir is on
your PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"   # add to ~/.bashrc, ~/.zshrc, etc.
# or run:
uv tool update-shell
```

To remove the global install:

```bash
./gen3dhub uninstall       # wraps `uv tool uninstall gen3dhub`
```

To upgrade after a code change in this repo, run `./gen3dhub install` again
— it passes `--reinstall` to uv tool, so the global copy is replaced with
the current source.

## Usage

There are two distinct ways to drive the tool:

- **Persistent TUI** — for human use. A full-screen interface that stays open
  while you navigate; you only exit when you say so.
- **Flag-driven CLI** — for scripts and AI agents. Every action is reachable via
  subcommand + flags, never asking for input on stdin.

### TUI (interactive)

```bash
./gen3dhub              # opens the TUI
./gen3dhub tui          # explicit equivalent
```

Keybindings:

| Key            | Action                                    |
|----------------|-------------------------------------------|
| `↑` / `↓`      | Move between options / rows               |
| `Tab` / `Shift+Tab` | Cycle focus between widgets in a form |
| `Enter`        | Activate / submit                         |
| `Escape`       | Go back to the previous screen            |
| `Q`            | Back (sub-screen) / Quit (main menu)      |
| `Ctrl+C`       | Quit immediately from anywhere            |

The TUI screens are: main menu → models list, setup, run inference, doctor.
Long-running operations (downloads, dependency installs, inference) suspend the
TUI temporarily so the underlying tool's output (Rich progress bars, `pip`
logs, etc.) renders normally in the terminal. When the operation finishes,
press `Enter` to return to the menu — the app does not exit on its own.

### CLI subcommands (non-interactive)

The CLI exposes five subcommands. Every option can be passed via flags —
convenient for scripting and AI agents — or omitted to trigger interactive
`questionary` prompts as a fallback.

### `list` — show available models

```bash
gen3dhub list
```

### `setup` — install a model

Clones the source repository (when applicable), creates a per-model virtualenv,
installs pinned dependencies into it, and verifies the install.

```bash
# Non-interactive
gen3dhub setup --model stable-fast-3d

# Reinstall from scratch
gen3dhub setup --model stable-fast-3d --force

# Interactive: prompts you to pick a model
gen3dhub setup
```

### `run` — execute inference

```bash
# Non-interactive (preferred for AI agents and scripts)
gen3dhub run \
    --model stable-fast-3d \
    --image path/to/photo.png \
    --output path/to/result.glb \
    --yes

# Interactive: omit any flag and you'll be prompted
gen3dhub run
```

Flags:
- `--model, -m` — model ID (see `list`).
- `--image, -i` — input image path. Used by image-input models (e.g. SF3D).
- `--text, -t` — input text prompt. Used by text-input models (none yet).
- `--output, -o` — destination path. Defaults to `./<image-stem>.glb` in the
  current directory.
- `--auto-setup / --no-auto-setup` — when on (default), the tool offers to
  install the model if it isn't installed yet.
- `--yes, -y` — skip all confirmation prompts. Use this when calling from a
  non-interactive context.

### `doctor` — diagnose the environment

Runs every adapter's verification routine: checks the marker file, the cloned
repo, the per-model virtualenv, and Hugging Face authentication / license
acceptance.

```bash
# Check everything
gen3dhub doctor

# Check a single model
gen3dhub doctor --model stable-fast-3d
```

Exits non-zero if any check fails — useful in CI and as a precondition in agent
workflows.

### `tui` — explicit TUI launcher

Same as running `gen3dhub` with no subcommand. Documented above.

## How it works

```
~/.cache/gen3dhub/                  (overridable via GEN3DHUB_CACHE_DIR)
└── models/
    └── <model-id>/
        ├── repo/        # cloned source repository, checked out at a pinned commit
        ├── .venv/       # uv-managed isolated virtualenv with pinned deps
        └── installed    # marker written after a successful setup
```

Model weights are downloaded by each model's own machinery into Hugging Face's
standard cache (`~/.cache/huggingface`).

The host process running `gen3dhub` itself stays light: it only depends on
`typer`, `rich`, `textual`, `questionary`, `huggingface-hub`, and `pillow`. Heavy
dependencies (PyTorch, custom CUDA kernels, model code) live in the per-model
virtualenvs and are invoked through `subprocess`.

### Reproducibility

Each adapter pins:

- The **upstream source commit** (`git checkout <hash>` after clone), not just
  the branch tip — so the install is repeatable across days/months even if the
  upstream repo evolves.
- The **runtime dependency versions** (e.g. `torch==2.4.1`,
  `torchvision==0.19.1` for SF3D), defined as constants in the adapter file.
- The **Python version** of the per-model venv (3.11), independent of whatever
  Python is on the host.

To upgrade a model deliberately, edit the constants at the top of the adapter
(`_REPO_COMMIT`, `_TORCH_PACKAGES`, etc.) and run
`gen3dhub setup --model <id> --force`.

### Build isolation note (SF3D and similar models)

`uv pip install` runs each package's build in an isolated sandbox by default.
Some ML-adjacent packages — e.g. SF3D's local `texture_baker` and
`uv_unwrapper` extensions — `import torch` from their `setup.py` without
declaring it in `build-system.requires`. Under build isolation, the sandbox
doesn't have torch and the wheel fails to compile (`ModuleNotFoundError: No
module named 'torch'`).

The SF3D adapter handles this with a deliberate two-pass install:

1. Install `torch` + `torchvision` (pinned versions) into the venv.
2. Install everything in `requirements.txt` with `--no-build-isolation`, so
   that local C++/CUDA extensions can find the just-installed torch in the
   same environment they're being built into.

If you write a new adapter for a model with similar build quirks, follow the
same pattern.

## For AI agents calling this tool

The fastest way to onboard an agent: have it run `gen3dhub agent`. That
prints a complete, plain-text usage guide (purpose, exit codes, env vars,
gated-model handling, troubleshooting, end-to-end example) intended to be
piped straight into the agent's context window.

The same guide is also surfaced in `gen3dhub --help` as a short quickstart
at the top, so an agent that calls `--help` first sees the recommended
non-interactive workflow without any extra effort.

### Recommended pattern

```bash
# 1. Confirm the host is healthy. Exits non-zero if not — read stderr for
#    distro-specific install commands the agent can surface to the user.
gen3dhub doctor || exit 1

# 2. Install on demand if needed (idempotent).
gen3dhub setup --model stable-fast-3d

# 3. Run inference. --yes suppresses ALL confirmation prompts.
gen3dhub run --model stable-fast-3d \
    --image /tmp/input.png \
    --output /tmp/output.glb \
    --yes
```

### Don'ts

- Don't call bare `gen3dhub` or `gen3dhub tui` from an agent — both open
  the interactive TUI, which requires a TTY and will hang under an agent
  runner.
- Don't omit `--yes` on `run` — it may stop on a confirmation prompt.
- Don't try to invoke the model's underlying script directly. Pinned
  per-model venvs live under `~/.cache/gen3dhub/models/<id>/.venv` and
  aren't on PATH; only `gen3dhub run` knows the right invocation.

### Exit code contract

| Code | Meaning                                                      |
|------|--------------------------------------------------------------|
| 0    | Success.                                                     |
| 1    | Precondition failure (toolchain, HF auth, license, etc.).    |
| 2    | CLI usage error (Typer/Click).                               |
| >2   | Subprocess error (uv, pip, git, model inference).            |

The `run` subcommand exits 0 on success and prints the output path on the
last `✓` line.

## Adding a model

Each adapter is a single file in `src/gen3dhub/models/`.

1. Subclass `ModelAdapter` from `gen3dhub.models.base`.
2. Define the `info` class attribute (`ModelInfo`): id, display name, summary,
   homepage, license URL, declared `InputSpec`s, and the produced file
   extension.
3. Implement `setup(force: bool)`, `verify() -> list[str]`, and
   `run(request: RunRequest) -> Path`.
4. Register the adapter in `src/gen3dhub/registry.py`.

The `stable-fast-3d` adapter
(`src/gen3dhub/models/stable_fast_3d.py`) is the canonical example — it
covers cloning a source repo, creating a `uv` venv, installing dependencies,
verifying Hugging Face authentication, and shelling out for inference.

## Environment variables

- `GEN3DHUB_CACHE_DIR` — override the location of the cache root.
- `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` — Hugging Face auth token. Required
  for gated models. Persistently storing the token via `gen3dhub setup`'s
  prompt or `huggingface-cli login` removes the need to set this env var on
  every shell. Setting it explicitly takes precedence over the on-disk file.

## License

MIT.
