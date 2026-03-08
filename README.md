# LlamaHerder

A PyQt5 GUI for managing llama.cpp RPC cluster inference. Launch a
llama-server backed by multiple GPU workers across your network from a
single interface.

## Features

### Cluster Management
- **Worker table** with live status, latency, VRAM usage, battery, and
  power source for every RPC node.
- **Auto-discovery** of the local llama-server binary with manual path
  override in Options.
- **Model selector** scanning your configured model directory for GGUF files.
- **Context size selector** with common presets (2K-128K).

### Sampler Settings (File > Sampler)
- Full control over all llama.cpp sampling parameters across five tabs:
  - **Sampling** -- temperature, top_k, top_p, min_p, typical_p,
    dynamic temperature, top_n_sigma, sampler ordering.
  - **Penalties** -- repeat, frequency, presence penalties. DRY and XTC
    parameters. Penalize-newlines toggle.
  - **Generation** -- max tokens, seed, mirostat (0/1/2), tail-free
    sampling, BOS/EOS token toggles, prompt caching, stop strings.
  - **Grammar** -- GBNF grammar and JSON schema constraints.
  - **Model Associations** -- map individual models to sampler profiles
    so settings switch automatically on launch.
- **Profile manager** -- create, save, rename, and delete sampler
  profiles. The built-in "Default" profile is read-only and cannot be
  deleted or overwritten.

### Worker Deployment
- **Generate Worker** -- create a ready-to-run worker script (Python)
  for Linux, Windows, or macOS with pre-filled connection details.
- **Deploy via SSH** -- push and start a worker agent on a remote host
  directly from the GUI with live log output.

### Server Control
- **Launch / Switch Model** -- start llama-server with the selected
  model, context size, and sampler profile. All configured RPC workers
  are connected automatically.
- **Stop / Unload** -- gracefully stop the server or unload the model
  from VRAM.
- **Server monitor** -- live slot utilization, prompt/generation token
  counts, and tokens-per-second for each connected worker.
- **Log viewer** -- scrollable, searchable server stdout/stderr output
  with Prev/Next search navigation.
- **Open in Browser** -- quick-launch to the llama-server web UI.

### Configuration (File > Options)
- **Workers tab** -- add, edit, and remove RPC worker entries (name, IP,
  port).
- **Paths & Ports tab** -- model directory, llama-server binary path,
  agent port, server listen port, local machine name, parallel slots,
  and theme (dark/light).

### System Tray
- Minimizes to the system tray on close. Show/hide and exit from the
  tray icon context menu. Notification on first minimize.

## Requirements

- Python 3.10+
- PyQt5
- llama.cpp built with RPC support (`llama-server` and `llama-rpc-server`)

## Quick Start

```bash
pip install PyQt5
python3 llamaherder.py
```

On first launch, go to **File > Options** to set:
1. Your model directory (where your `.gguf` files live).
2. The path to your `llama-server` binary.
3. Your RPC worker IPs and ports.

Select a model, pick a context size, and hit **Launch**.

## Configuration

Settings are stored in `~/.config/llamaherder/config.json` and include
worker definitions, paths, ports, theme, sampler profiles, and
model-to-profile associations. The config is auto-created on first run
and migrates from older formats automatically.

## License

MIT
