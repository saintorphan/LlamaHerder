0<0# : ^
"""
@echo off
where python >nul 2>&1 && (python "%~f0" %* & exit /b)
where python3 >nul 2>&1 && (python3 "%~f0" %* & exit /b)
where py >nul 2>&1 && (py -3 "%~f0" %* & exit /b)
echo Python not found. Install from https://python.org
pause
exit /b 1
"""
# ── LlamaHerder Worker ──────────────────────────────────────────────────────
#
# Combined RPC worker + health agent for Windows machines.
# Double-click the .cmd or run from a terminal:
#
#   llama-worker.cmd
#   llama-worker.cmd --rpc-port 50052 --agent-port 50053
#   llama-worker.cmd --no-kill              (skip killing bloat)
#   llama-worker.cmd --kill-explorer        (also kill explorer for max VRAM)
#   llama-worker.cmd --no-firewall          (skip firewall setup)
#   llama-worker.cmd --rpc-server C:\path\to\rpc-server.exe
#
# Requires:
#   - Python 3.6+
#   - rpc-server.exe from llama.cpp (place alongside this file or in PATH)
#   - Administrator privileges for firewall rules (optional)
#
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import atexit
import ctypes
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))

# ── Process kill list ────────────────────────────────────────────────────────
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
    """Check if running with administrator privileges."""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        return False


def kill_processes(proc_list):
    """Kill a list of processes by name (silent on failure)."""
    for name in proc_list:
        subprocess.run(
            ["taskkill", "/F", "/IM", name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


def setup_firewall(rpc_port, agent_port):
    """Configure Windows firewall rules for RPC and agent ports."""
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
    for name in ("rpc-server.exe", "rpc-server"):
        p = os.path.join(SCRIPT_DIR, name)
        if os.path.isfile(p):
            return p

    # Check PATH
    for name in ("rpc-server", "rpc-server.exe",
                 "llama-rpc-server", "llama-rpc-server.exe"):
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
    """Query Windows WMI for battery percentage and charge status."""
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
    hostname = os.environ.get("COMPUTERNAME", os.environ.get("HOSTNAME", "worker"))

    parser = argparse.ArgumentParser(
        description="LlamaHerder Worker — RPC server + health agent",
    )
    parser.add_argument("--rpc-port", type=int, default=50052,
                        help="RPC server listen port (default: 50052)")
    parser.add_argument("--agent-port", type=int, default=50053,
                        help="Health agent listen port (default: 50053)")
    parser.add_argument("--rpc-server", type=str, default="",
                        help="Path to rpc-server binary (auto-detected if omitted)")
    parser.add_argument("--no-kill", action="store_true",
                        help="Skip killing bloat processes")
    parser.add_argument("--kill-explorer", action="store_true",
                        help="Also kill explorer.exe for maximum VRAM")
    parser.add_argument("--no-firewall", action="store_true",
                        help="Skip firewall configuration")
    parser.add_argument("--agent-only", action="store_true",
                        help="Run health agent only (no RPC server)")
    args = parser.parse_args()

    print("=" * 50)
    print(f"  LlamaHerder Worker — {hostname}")
    print("=" * 50)
    print()

    explorer_killed = False

    # ── Phase 1: Kill bloat ──
    if not args.no_kill:
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
    else:
        print("[1] Skipping bloat kill (--no-kill).")
    print()

    # ── Phase 2: Firewall ──
    if not args.no_firewall:
        print("[2] Configuring firewall rules...")
        setup_firewall(args.rpc_port, args.agent_port)
    else:
        print("[2] Skipping firewall config (--no-firewall).")
    print()

    # ── Cleanup handler ──
    def cleanup():
        if explorer_killed:
            print("\nRestoring explorer...")
            subprocess.Popen("explorer.exe")

    atexit.register(cleanup)

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
        print("  Place rpc-server.exe alongside this script, add to PATH,")
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
