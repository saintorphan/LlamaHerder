#!/usr/bin/env python3
# ── LlamaHerder Worker ──────────────────────────────────────────────────────
#
# Cross-platform RPC worker + health agent.
# Works on Windows, Linux, and macOS.
#
# Usage:
#   python3 llama-worker.py
#   python3 llama-worker.py --rpc-port 50052 --agent-port 50053
#   python3 llama-worker.py --no-kill              (skip killing bloat — Windows only)
#   python3 llama-worker.py --kill-explorer        (also kill explorer — Windows only)
#   python3 llama-worker.py --no-firewall          (skip firewall setup — Windows only)
#   python3 llama-worker.py --rpc-server /path/to/rpc-server
#   python3 llama-worker.py --agent-only           (run health agent only)
#
# Requires:
#   - Python 3.6+
#   - rpc-server from llama.cpp (place alongside this file or in PATH)
#   - Administrator/root privileges for firewall rules (Windows only, optional)
#
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import atexit
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

# ── Substitution targets (replaced during generation) ────────────────────────
DEFAULT_RPC_PORT = 50052
DEFAULT_AGENT_PORT = 50053

SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
OS_TYPE = platform.system()  # "Windows", "Linux", "Darwin"

# ── Process kill list (Windows only) ─────────────────────────────────────────
# These silently skip anything not running.

BLOAT_PROCESSES = [
    # Browsers / Electron
    "msedge.exe", "chrome.exe", "firefox.exe", "brave.exe",
    # Microsoft Office / Teams
    "Teams.exe", "ms-teams.exe", "OneDrive.exe", "Outlook.exe",
    "EXCEL.EXE", "WINWORD.EXE", "POWERPNT.EXE",
    # Widgets / Search
    "Widgets.exe", "WidgetService.exe", "SearchHost.exe",
    "SearchApp.exe", "Cortana.exe",
    # Xbox / Game Bar
    "GameBar.exe", "GameBarPresenceWriter.exe", "XboxGameBarWidgets.exe",
    # Misc background apps
    "PhoneExperienceHost.exe", "YourPhone.exe", "SkypeApp.exe",
    "Spotify.exe", "Discord.exe", "Slack.exe",
    "Steam.exe", "EpicGamesLauncher.exe",
    "ShareX.exe", "snippingtool.exe",
]

# ── Utility functions ────────────────────────────────────────────────────────


def is_admin():
    """Check if running with elevated privileges."""
    if OS_TYPE == "Windows":
        try:
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except Exception:
            return False
    else:
        return os.getuid() == 0


def kill_processes(proc_list):
    """Kill a list of processes by name (Windows only, silent on failure)."""
    if OS_TYPE != "Windows":
        return
    for name in proc_list:
        subprocess.run(
            ["taskkill", "/F", "/IM", name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


def setup_firewall(rpc_port, agent_port):
    """Configure firewall rules for RPC and agent ports (Windows only)."""
    if OS_TYPE != "Windows":
        return

    if not is_admin():
        print("  WARNING: Not running as admin — skipping firewall config.")
        print("  Run as Administrator or manually open ports", rpc_port, "and", agent_port)
        return

    # Set Ethernet to Private profile
    subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-NetAdapter | Where-Object {"
         "$_.InterfaceDescription -like '*Ethernet*' -and $_.Status -eq 'Up'"
         "} | ForEach-Object {"
         "Set-NetConnectionProfile -InterfaceIndex $_.ifIndex"
         " -NetworkCategory Private }"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    rules = [
        ("llama RPC", rpc_port),
        ("llama Agent", agent_port),
    ]
    for name, port in rules:
        subprocess.run(
            ["netsh", "advfirewall", "firewall", "delete", "rule",
             f"name={name}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["netsh", "advfirewall", "firewall", "add", "rule",
             f"name={name}", "dir=in", "action=allow", "protocol=TCP",
             f"localport={port}", "profile=any"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    print(f"  Firewall configured (ports {rpc_port}, {agent_port} open).")


def find_rpc_server(configured_path):
    """Locate rpc-server binary. Returns path or None."""
    if configured_path and os.path.isfile(configured_path):
        return configured_path

    # Check same directory as this script
    if OS_TYPE == "Windows":
        candidates = ("rpc-server.exe", "rpc-server")
    else:
        candidates = ("rpc-server",)

    for name in candidates:
        p = os.path.join(SCRIPT_DIR, name)
        if os.path.isfile(p):
            return p

    # Check PATH
    search_names = ["rpc-server", "llama-rpc-server"]
    if OS_TYPE == "Windows":
        search_names += ["rpc-server.exe", "llama-rpc-server.exe"]
    for name in search_names:
        w = shutil.which(name)
        if w:
            return w

    return None


# ── VRAM / Battery queries ───────────────────────────────────────────────────


def get_vram():
    """Query nvidia-smi for free/total VRAM in MiB (summed across all GPUs)."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            total_free = 0
            total_total = 0
            for line in result.stdout.strip().splitlines():
                parts = line.split(",")
                if len(parts) >= 2:
                    total_free += int(parts[0].strip())
                    total_total += int(parts[1].strip())
            if total_total > 0:
                return total_free, total_total
    except Exception:
        pass
    return None, None


def get_battery():
    """Query battery percentage and charge status (OS-specific).

    Desktops with no battery return (None, "AC").
    """
    if OS_TYPE == "Windows":
        pct, source = _get_battery_windows()
    elif OS_TYPE == "Linux":
        pct, source = _get_battery_linux()
    elif OS_TYPE == "Darwin":
        pct, source = _get_battery_mac()
    else:
        pct, source = None, None

    # No battery detected — it's a desktop, always on AC
    if pct is None and source is None:
        return None, "AC"
    return pct, source


def _get_battery_windows():
    """Query Windows WMI for battery info."""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Battery | "
             "Select-Object EstimatedChargeRemaining, BatteryStatus | "
             "ConvertTo-Json)"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            if isinstance(data, list):
                data = data[0]
            pct = data.get("EstimatedChargeRemaining")
            code = data.get("BatteryStatus", 0)
            source_map = {
                1: "Discharging", 2: "AC", 3: "Full",
                4: "Low", 5: "Critical", 6: "Charging",
                7: "Charging", 8: "Charging", 9: "Charging",
            }
            return pct, source_map.get(code, "Unknown")
    except Exception:
        pass
    return None, None


def _get_battery_linux():
    """Read battery info from /sys/class/power_supply/."""
    try:
        import glob
        bat_dirs = glob.glob("/sys/class/power_supply/BAT*")
        if not bat_dirs:
            return None, None
        bat_dir = bat_dirs[0]

        pct = None
        cap_path = os.path.join(bat_dir, "capacity")
        if os.path.isfile(cap_path):
            with open(cap_path) as f:
                pct = int(f.read().strip())

        status = None
        status_path = os.path.join(bat_dir, "status")
        if os.path.isfile(status_path):
            with open(status_path) as f:
                status = f.read().strip()

        return pct, status
    except Exception:
        pass
    return None, None


def _get_battery_mac():
    """Parse pmset -g batt output on macOS."""
    try:
        result = subprocess.run(
            ["pmset", "-g", "batt"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            output = result.stdout
            # Example: " -InternalBattery-0 (id=...)	85%; charging; 1:23 remaining"
            for line in output.splitlines():
                if "InternalBattery" in line:
                    # Extract percentage
                    pct = None
                    source = None
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        detail = parts[1]
                        tokens = [t.strip().rstrip(";") for t in detail.split(";")]
                        for tok in tokens:
                            if tok.endswith("%"):
                                pct = int(tok.rstrip("%"))
                            elif tok in ("charging", "discharging", "charged",
                                         "finishing charge", "AC attached"):
                                source = tok.capitalize()
                    return pct, source
    except Exception:
        pass
    return None, None


# ── Health agent HTTP server ─────────────────────────────────────────────────


class AgentHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            body = json.dumps({"status": "ok"})
        else:
            vram_free, vram_total = get_vram()
            battery_pct, power_source = get_battery()
            body = json.dumps({
                "vram_free_mb": vram_free,
                "vram_total_mb": vram_total,
                "battery_pct": battery_pct,
                "power_source": power_source,
            })
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, fmt, *args):
        pass  # silence request logs


def run_agent(port):
    """Start the health agent HTTP server (runs in a daemon thread)."""
    server = HTTPServer(("0.0.0.0", port), AgentHandler)
    server.serve_forever()


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    hostname = os.environ.get("COMPUTERNAME",
                              os.environ.get("HOSTNAME",
                                             platform.node() or "worker"))

    parser = argparse.ArgumentParser(
        description="LlamaHerder Worker — RPC server + health agent",
    )
    parser.add_argument("--rpc-port", type=int, default=DEFAULT_RPC_PORT,
                        help=f"RPC server listen port (default: {DEFAULT_RPC_PORT})")
    parser.add_argument("--agent-port", type=int, default=DEFAULT_AGENT_PORT,
                        help=f"Health agent listen port (default: {DEFAULT_AGENT_PORT})")
    parser.add_argument("--rpc-server", type=str, default="",
                        help="Path to rpc-server binary (auto-detected if omitted)")
    parser.add_argument("--no-kill", action="store_true",
                        help="Skip killing bloat processes (Windows only)")
    parser.add_argument("--kill-explorer", action="store_true",
                        help="Also kill explorer.exe for maximum VRAM (Windows only)")
    parser.add_argument("--no-firewall", action="store_true",
                        help="Skip firewall configuration (Windows only)")
    parser.add_argument("--agent-only", action="store_true",
                        help="Run health agent only (no RPC server)")
    args = parser.parse_args()

    print("=" * 50)
    print(f"  LlamaHerder Worker — {hostname}")
    print(f"  OS: {OS_TYPE} ({platform.machine()})")
    print("=" * 50)
    print()

    explorer_killed = False

    # ── Phase 1: Kill bloat (Windows only) ──
    if OS_TYPE == "Windows" and not args.no_kill:
        print("[1] Killing bloat for workhorse mode...")
        kill_processes(BLOAT_PROCESSES)
        if args.kill_explorer:
            kill_processes(["explorer.exe"])
            explorer_killed = True
            # Keep SMB alive without explorer
            subprocess.run(
                ["net", "start", "LanmanServer"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        print("  Done — workhorse mode active.")
    elif OS_TYPE == "Windows":
        print("[1] Skipping bloat kill (--no-kill).")
    else:
        print("[1] Bloat kill skipped (non-Windows).")
    print()

    # ── Phase 2: Firewall (Windows only) ──
    if OS_TYPE == "Windows" and not args.no_firewall:
        print("[2] Configuring firewall rules...")
        setup_firewall(args.rpc_port, args.agent_port)
    elif OS_TYPE == "Windows":
        print("[2] Skipping firewall config (--no-firewall).")
    else:
        print("[2] Firewall config skipped (non-Windows).")
    print()

    # ── Cleanup handler ──
    def cleanup():
        if explorer_killed:
            print("\nRestoring explorer...")
            subprocess.Popen("explorer.exe")

    atexit.register(cleanup)

    # Handle SIGTERM for graceful shutdown on Linux/Mac
    if OS_TYPE != "Windows":
        def _sigterm(signum, frame):
            print("\nSIGTERM received, shutting down...")
            sys.exit(0)
        signal.signal(signal.SIGTERM, _sigterm)

    # ── Phase 3: Start health agent ──
    print(f"[3] Starting health agent on port {args.agent_port}...")
    agent_thread = threading.Thread(target=run_agent, args=(args.agent_port,), daemon=True)
    agent_thread.start()
    print("  Health agent running.")
    print()

    # ── Phase 4: Start RPC server ──
    if args.agent_only:
        print("[4] Agent-only mode — no RPC server.")
        print("  Press Ctrl+C to stop.")
        print()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopped.")
        return

    rpc_bin = find_rpc_server(args.rpc_server)
    if not rpc_bin:
        print("[4] ERROR: rpc-server not found.")
        print("  Place rpc-server alongside this script, add to PATH,")
        print("  or use --rpc-server <path>.")
        print()
        print("  Build it from llama.cpp:")
        print("    cmake -B build && cmake --build build --target rpc-server")
        print()
        print("  Running in agent-only mode. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopped.")
        return

    print(f"[4] Starting RPC server on port {args.rpc_port}...")
    print(f"  Binary: {rpc_bin}")
    print("  Press Ctrl+C to stop everything.")
    print()

    try:
        proc = subprocess.run(
            [rpc_bin, "-H", "0.0.0.0", "-p", str(args.rpc_port)],
        )
    except KeyboardInterrupt:
        pass
    except FileNotFoundError:
        print(f"ERROR: Could not execute {rpc_bin}")

    print("\nRPC server stopped.")


if __name__ == "__main__":
    main()
