"""The text printed by `gen3dhub agent` — a comprehensive usage guide aimed at
AI agents (Claude Code, Cursor, custom LLM tools) and shell scripts.

Kept as a single module-level constant so the `agent` subcommand can dump it
verbatim without any markdown rendering — agents and shells handle plain text
better than ANSI-decorated output.
"""

from __future__ import annotations

AGENT_GUIDE = """\
gen3dhub — usage guide for AI agents and scripts
=================================================

PURPOSE
  gen3dhub is a hub for AI models that generate 3D assets (image-to-3D,
  text-to-3D, etc.). It handles upstream source download, isolated per-model
  install, and inference behind a uniform CLI. Currently ships with one
  adapter: stable-fast-3d (image -> textured GLB mesh).

NON-INTERACTIVE USAGE — REQUIRED FOR AGENTS
  - Pass --yes (-y) on `run` to skip ALL confirmation prompts.
  - Always pass --model.
  - DO NOT call `gen3dhub` with no subcommand or `gen3dhub tui`. Both open a
    persistent TUI that requires a TTY and will hang under an agent runner.

EXIT CODES
  0    success
  1    precondition failure (toolchain missing, HF auth missing, model not
       installed, license not accepted, etc.). Details printed to stderr.
  2    Typer/Click usage error (invalid flags, unknown options).
  >2   subprocess error from a downstream tool (uv, pip, git, the model's
       inference script). Surface the original error to the user.

THREE-STEP WORKFLOW (idempotent)
  1) Health check — exits 0 if the host AND the requested model are ready.
       gen3dhub doctor                            # check all models
       gen3dhub doctor --model <id>               # check one
     If exit != 0, parse stderr for actionable messages — they include
     copy-pasteable install commands for the detected Linux distro.

  2) One-time install of a model. No-op if already installed; --force to
     wipe and reinstall.
       gen3dhub setup --model <id>
       gen3dhub setup --model <id> --force
     SF3D first install: ~5-15 minutes (clones repo, creates per-model venv,
     downloads ~1.5 GB of pinned PyTorch wheels, builds two C++ extensions).
     After install, `setup` runs a post-setup hook that re-checks
     credentials. Under a TTY it prompts for any missing secrets (e.g. the
     HF token); under an agent runner (no TTY, or invoked with `--yes`) it
     just prints a warning with the next step. So `setup` never hangs.

  3) Run inference.
       gen3dhub run --model <id> --image <path> --output <path> --yes
     On success prints `Wrote 3D mesh -> <path>` and exits 0.

DISCOVERY
  gen3dhub list                  show supported models, inputs, output ext
  gen3dhub <subcommand> --help   per-subcommand flags
  gen3dhub agent                 this guide

ENVIRONMENT VARIABLES
  HF_TOKEN, HUGGING_FACE_HUB_TOKEN
      Hugging Face access token. Required for gated models (e.g. SF3D). The
      token only grants download access AFTER the user has accepted the
      model's license on its HF page (this step is browser-only and cannot
      be automated). `gen3dhub doctor` distinguishes "no token" from "token
      present but no license access" and reports the right fix.

      Persistent storage: when the user runs `gen3dhub setup` in an
      interactive terminal, post-setup will prompt for the token (input
      hidden) and save it via `huggingface_hub.login()` to
      ~/.cache/huggingface/token (mode 0600). All HF tools — including
      this one — read from there automatically; no env var needed
      afterwards. Setting HF_TOKEN at the env level overrides the file.

  GEN3DHUB_CACHE_DIR
      Overrides the default ~/.cache/gen3dhub root. Useful when the home
      partition is small or for sandboxed CI runs.

  CUDA_VISIBLE_DEVICES
      Standard PyTorch GPU selection. Set to "" to force CPU. Inference
      works on CPU but is much slower.

GATED MODELS — ONE-TIME HUMAN STEP
  Some models (Stable Fast 3D among them) are GATED on Hugging Face. Before
  download, the user must:
    1. Visit the model page (shown by `gen3dhub list` as the homepage).
    2. Click "Agree and access repository".
    3. Provide a Hugging Face access token. Three ways:
       a) Run `gen3dhub setup -m <id>` interactively — its post-setup hook
          prompts for the token and saves it under ~/.cache/huggingface/.
       b) Run `huggingface-cli login`.
       c) Set HF_TOKEN in the environment.
  Agents cannot perform step 1-2 (browser action). If `gen3dhub doctor`
  reports a gated-repo error, surface the model homepage URL to the user
  and ask them to accept the license, then retry.

CACHE LAYOUT
  $GEN3DHUB_CACHE_DIR (default ~/.cache/gen3dhub)
    models/
      <model-id>/
        repo/        cloned upstream source, checked out at a pinned commit
        .venv/       isolated uv-managed virtualenv with pinned deps
        installed    sentinel file written after a successful setup

REPRODUCIBILITY GUARANTEES
  Each adapter pins:
    - upstream commit (so `setup` produces the same code today and in 6 mo)
    - PyTorch / framework versions
    - Python interpreter version of the per-model venv
  Re-running `setup` is a no-op. To deliberately upgrade, edit the adapter
  constants in src/gen3dhub/models/<id>.py and run `setup --force`.

EXAMPLE — Stable Fast 3D end-to-end
  $ export HF_TOKEN=hf_xxx
  $ gen3dhub doctor --model stable-fast-3d
    # exits 1 if license not accepted; 0 if all good
  $ gen3dhub setup --model stable-fast-3d
    # ~5-15 min on first run
  $ gen3dhub run --model stable-fast-3d \\
      --image /tmp/cat.png \\
      --output /tmp/cat.glb \\
      --yes
    # exits 0; produces /tmp/cat.glb (textured GLB mesh)

DON'T
  - Don't call bare `gen3dhub` or `gen3dhub tui` from an agent — both open
    the interactive TUI and require a TTY.
  - Don't omit --yes on `run`. Without it the command may stop on a
    confirmation prompt waiting for keyboard input.
  - Don't invoke the model's underlying script directly. The pinned venv
    lives at ~/.cache/gen3dhub/models/<id>/.venv and is not on PATH; only
    `gen3dhub run` knows how to set up the env, args, and output handling.
  - Don't run `setup` twice in parallel for the same model — the install
    isn't fully concurrency-safe.

TROUBLESHOOTING
  - "No C compiler found": install build-essential / base-devel /
    Development Tools per the distro hint that doctor prints.
  - "Hugging Face token is set but cannot access <repo>": user must accept
    the model's license on its HF page (browser action).
  - "Cannot install Stable Fast 3D — system requirements not met": check
    `gen3dhub doctor` for specific tools missing.
  - Inference fails with CUDA OOM: set CUDA_VISIBLE_DEVICES="" to force CPU,
    or reduce input image size.
"""
