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
  text-to-3D, mesh-to-texture, etc.). It handles upstream source download,
  isolated per-model install, and inference behind a uniform CLI.
  Currently ships with three adapters:
    - stable-fast-3d   image -> textured GLB mesh, ~1s/asset, GATED on HF
    - hunyuan3d-2      image -> shape GLB mesh,    ~30s/asset, public on HF
                       (Tencent community license — restrictive; not OSI)
    - paint3d          mesh + reference image -> textured GLB,
                       ~5-10 min/asset, Apache 2.0, no CPU mode

  Each adapter declares a `best_for` line surfaced in `gen3dhub list` to
  help pick the right one. Common pairing: `hunyuan3d-2` for geometry,
  then `paint3d` to texture the resulting mesh.

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

  3) Run inference. Inputs differ per model — pass only the flags the
     model declares. `gen3dhub list` shows each model's `Inputs` line.
     A 4-angle PNG preview is rendered next to the output (best-effort);
     pass `--no-preview` to skip when you don't need it (CI, batch loops).

     Image-input models (stable-fast-3d, hunyuan3d-2):
       gen3dhub run --model <id> --image <path> --output <path> --yes

     Mesh-input model (paint3d) — needs BOTH a mesh and a reference image:
       gen3dhub run --model paint3d \\
           --mesh <path-to-existing-mesh.glb> \\
           --image <path-to-reference.png> \\
           --output <path-to-textured.glb> --yes

     Pass `--cpu` to force CPU inference. Useful when:
       - the host has no NVIDIA GPU,
       - the GPU runs out of VRAM (CUDA OutOfMemoryError),
       - or running in CI / headless servers.
     ~10-60x slower than GPU. Note: paint3d does NOT support CPU mode
     (upstream pipelines hard-code .to('cuda')); --cpu is silently ignored
     for that model.

     gen3dhub auto-sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True in
     the inference subprocess to reduce CUDA fragmentation on near-the-edge
     8 GB GPUs. Override by setting the env var explicitly before invoking.

TUNABLE PARAMETERS
  Each adapter declares model-specific parameters (texture resolution,
  remesh options, inference steps, seed, etc.). Pass them with the
  repeatable `--param NAME=VALUE` flag:

    gen3dhub run --model stable-fast-3d --image cat.png \\
        --output cat.glb --yes \\
        --param remesh_option=quad \\
        --param texture_resolution=2048

  Discover the parameters for a model:
    - `gen3dhub list` prints a "⚙ Parameters" block per model (name,
      default, choices/type).
    - Or pass an unknown parameter to provoke a helpful error: the
      message lists every available parameter for that model.

  Validation is strict and happens before the subprocess starts:
    - SELECT params reject values outside their `choices`.
    - INT/FLOAT params reject unparseable strings.
    - Unknown parameter names are rejected with the model's allowed list.
  So a typo never reaches the underlying model — the run aborts with an
  exit-2 (Typer usage error) and a clear message on stderr.

DISCOVERY
  gen3dhub list                  show supported models, inputs, output ext
  gen3dhub <subcommand> --help   per-subcommand flags
  gen3dhub history --json        machine-readable log of past runs
  gen3dhub agent                 this guide

HISTORY
  Every successful or failed `gen3dhub run` appends an entry to
  ~/.cache/gen3dhub/history.jsonl (one JSON object per line). Useful for
  agents that want to:
    - check recent state without re-running anything
    - find a past output path by id
    - reproduce a prior call

  Read the log:
    gen3dhub history --json              # newline-delimited JSON for parsing
    gen3dhub history --rerun <id>        # prints the equivalent CLI command
                                         # (does NOT auto-execute — copy & run)
  Each entry includes: id, timestamp (UTC), model, inputs, params, output
  path, preview path, duration_s, exit_code.

UNINSTALL
  Models are big (3-10 GB each, in the per-model venv). When you're done
  with one, free the disk:
    gen3dhub uninstall --model <id>      # confirmation prompt
    gen3dhub uninstall --model <id> --yes  # no prompt
    gen3dhub uninstall --all             # remove every model
  Hugging Face weights in ~/.cache/huggingface/ are NOT touched (shared
  across HF tools); see `huggingface-cli scan-cache` to clean those.

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

EXAMPLE — Stable Fast 3D end-to-end (single-shot textured asset)
  $ export HF_TOKEN=hf_xxx
  $ gen3dhub doctor --model stable-fast-3d
    # exits 1 if license not accepted; 0 if all good
  $ gen3dhub setup --model stable-fast-3d
    # ~5-15 min on first run
  $ gen3dhub run --model stable-fast-3d \\
      --image /tmp/cat.png \\
      --output /tmp/cat.glb \\
      --yes \\
      --param remesh_option=quad \\
      --param texture_resolution=2048
    # exits 0; produces /tmp/cat.glb (textured GLB mesh, quad topology)

EXAMPLE — Hunyuan3D-2 + Paint3D pipeline (high-fidelity geometry + textures)
  $ gen3dhub setup --model hunyuan3d-2          # one-time
  $ gen3dhub setup --model paint3d              # one-time, slow first run
  $ gen3dhub run --model hunyuan3d-2 \\
      --image /tmp/cat.png --output /tmp/cat_shape.glb --yes
  $ gen3dhub run --model paint3d \\
      --mesh /tmp/cat_shape.glb \\
      --image /tmp/cat.png \\
      --output /tmp/cat_textured.glb --yes \\
      --param prompt="hyperrealistic photo of a cat, fur detail"
    # final exits 0; produces /tmp/cat_textured.glb

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
  - Inference fails with CUDA OOM: pass `--cpu` to gen3dhub run, or close
    other GPU-using apps (`nvidia-smi` shows them). The tool already sets
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for you to reduce
    fragmentation; --cpu is the next escape hatch.
"""
