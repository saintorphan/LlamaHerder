#!/usr/bin/env python3
"""Llama Cluster Launcher — PyQt5 GUI for llama.cpp RPC cluster inference."""

import json
import os
import platform
import shutil
import socket
import subprocess
import signal
import sys
import tempfile
import time
import urllib.request

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QColor, QFont, QIcon, QPalette, QPixmap, QPainter, QTextDocument
from PyQt5.QtWidgets import (
    QAction, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QHeaderView,
    QInputDialog, QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox,
    QPushButton, QScrollArea, QSpinBox, QSplitter, QSystemTrayIcon,
    QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout,
    QWidget,
)

# ── Configuration ────────────────────────────────────────────────────────────

CONFIG_DIR = os.path.expanduser("~/.config/llamaherder")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
_OLD_WORKERS_FILE = os.path.join(CONFIG_DIR, "workers.json")

_DEFAULT_SAMPLER_PROFILE = {
    # Sampling
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.95,
    "min_p": 0.05,
    "typical_p": 1.0,
    "dynatemp_range": 0.0,
    "dynatemp_exponent": 1.0,
    "top_n_sigma": -1.0,
    "samplers": "top_k,tfs_z,typical_p,top_p,min_p,temperature",
    # Penalties
    "repeat_penalty": 1.1,
    "repeat_last_n": 64,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "penalize_nl": True,
    "dry_multiplier": 0.0,
    "dry_base": 1.75,
    "dry_allowed_length": 2,
    "dry_penalty_last_n": -1,
    "xtc_probability": 0.0,
    "xtc_threshold": 0.1,
    # Generation
    "n_predict": -1,
    "seed": -1,
    "mirostat": 0,
    "mirostat_tau": 5.0,
    "mirostat_eta": 0.1,
    "tfs_z": 1.0,
    "use_bos": True,
    "ignore_eos": False,
    "cache_prompt": True,
    "stop": "",
    # Grammar
    "grammar": "",
    "json_schema": "",
}

_DEFAULT_CONFIG = {
    "workers": [
        {"ip": "10.0.0.2", "port": 50052, "name": "gorgoroth"},
        {"ip": "10.0.0.3", "port": 50052, "name": "asus-tuf"},
    ],
    "model_dir": "~/models",
    "llama_server": "~/llama.cpp/build/bin/llama-server",
    "agent_port": 50053,
    "server_port": 8080,
    "local_name": "ag1 (local)",
    "parallel_slots": 1,
    "theme": "dark",
    "sampler_profiles": {"Default": dict(_DEFAULT_SAMPLER_PROFILE)},
    "active_sampler_profile": "Default",
    "model_profile_map": {},
}

PING_INTERVAL_MS = 2000
SERVER_POLL_INTERVAL_MS = 1000

# Approximate VRAM needed for model weights (GB). Used for cluster suggestions.
MODEL_SIZES = [
    ("1B Q4",    0.8),
    ("3B Q4",    2.0),
    ("7B Q4",    4.5),
    ("7B Q8",    8.0),
    ("13B Q4",   8.0),
    ("13B Q8",  14.0),
    ("27B Q4",  16.0),
    ("27B Q6",  21.0),
    ("27B Q8",  28.0),
    ("32B Q4",  19.0),
    ("32B Q6",  25.0),
    ("32B Q8",  33.0),
    ("34B Q4",  20.0),
    ("34B Q8",  36.0),
    ("70B Q2",  27.0),
    ("70B Q4",  42.0),
    ("70B Q6",  55.0),
    ("70B Q8",  72.0),
    ("120B Q2", 45.0),
    ("120B Q4", 72.0),
    ("405B Q2", 150.0),
    ("405B Q4", 240.0),
]


# ── Sampler tab layout & CLI flag mapping ─────────────────────────────────────

# Human-readable labels for sampler fields, grouped by tab
_SAMPLER_TABS = {
    "Sampling": [
        ("temperature", "Temperature"),
        ("top_k", "Top-K"),
        ("top_p", "Top-P"),
        ("min_p", "Min-P"),
        ("typical_p", "Typical-P"),
        ("dynatemp_range", "Dynamic Temp Range"),
        ("dynatemp_exponent", "Dynamic Temp Exponent"),
        ("top_n_sigma", "Top-N Sigma"),
        ("samplers", "Sampler Order"),
    ],
    "Penalties": [
        ("repeat_penalty", "Repeat Penalty"),
        ("repeat_last_n", "Repeat Last N"),
        ("frequency_penalty", "Frequency Penalty"),
        ("presence_penalty", "Presence Penalty"),
        ("penalize_nl", "Penalize Newlines"),
        ("dry_multiplier", "DRY Multiplier"),
        ("dry_base", "DRY Base"),
        ("dry_allowed_length", "DRY Allowed Length"),
        ("dry_penalty_last_n", "DRY Penalty Last N"),
        ("xtc_probability", "XTC Probability"),
        ("xtc_threshold", "XTC Threshold"),
    ],
    "Generation": [
        ("n_predict", "Max Tokens (n_predict)"),
        ("seed", "Seed (-1 = random)"),
        ("mirostat", "Mirostat (0/1/2)"),
        ("mirostat_tau", "Mirostat Tau"),
        ("mirostat_eta", "Mirostat Eta"),
        ("tfs_z", "Tail-Free Sampling (z)"),
        ("use_bos", "Add BOS Token"),
        ("ignore_eos", "Ignore EOS Token"),
        ("cache_prompt", "Cache Prompt"),
        ("stop", "Stop Strings (one per line)"),
    ],
    "Grammar": [
        ("grammar", "GBNF Grammar"),
        ("json_schema", "JSON Schema"),
    ],
}

# Map config keys → llama-server CLI flags
_SAMPLER_FLAGS = {
    "temperature": "--temp",
    "top_k": "--top-k",
    "top_p": "--top-p",
    "min_p": "--min-p",
    "typical_p": "--typical",
    "repeat_penalty": "--repeat-penalty",
    "repeat_last_n": "--repeat-last-n",
    "frequency_penalty": "--frequency-penalty",
    "presence_penalty": "--presence-penalty",
    "mirostat": "--mirostat",
    "mirostat_tau": "--mirostat-tau",
    "mirostat_eta": "--mirostat-eta",
    "tfs_z": "--tfs",
    "seed": "--seed",
    "n_predict": "--n-predict",
    "dynatemp_range": "--dynatemp-range",
    "dynatemp_exponent": "--dynatemp-exponent",
    "top_n_sigma": "--top-n-sigma",
    "penalize_nl": "--penalize-nl",
    "dry_multiplier": "--dry-multiplier",
    "dry_base": "--dry-base",
    "dry_allowed_length": "--dry-allowed-length",
    "dry_penalty_last_n": "--dry-penalty-last-n",
    "xtc_probability": "--xtc-probability",
    "xtc_threshold": "--xtc-threshold",
    "samplers": "--samplers",
    "ignore_eos": "--ignore-eos",
    "cache_prompt": "--cache-prompt",
}

# Fields that are multiline text areas
_MULTILINE_FIELDS = {"stop", "grammar", "json_schema"}

# Fields that are boolean checkboxes
_BOOL_FIELDS = {"penalize_nl", "use_bos", "ignore_eos", "cache_prompt"}


_CMD_POLYGLOT_HEADER = """\
0<0# : ^
\"\"\"
@echo off
where python >nul 2>&1 && (python "%~f0" %* & exit /b)
where python3 >nul 2>&1 && (python3 "%~f0" %* & exit /b)
where py >nul 2>&1 && (py -3 "%~f0" %* & exit /b)
echo Python not found. Install from https://python.org
pause
exit /b 1
\"\"\"
"""


def _load_config():
    """Load unified config, migrating from old workers.json if needed."""
    os.makedirs(CONFIG_DIR, exist_ok=True)

    # Migration: old workers.json (bare list) → new config format
    if not os.path.exists(CONFIG_FILE) and os.path.exists(_OLD_WORKERS_FILE):
        try:
            with open(_OLD_WORKERS_FILE, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                config = dict(_DEFAULT_CONFIG)
                config["workers"] = data
                _save_config(config)
                return config
        except (json.JSONDecodeError, OSError):
            pass

    if not os.path.exists(CONFIG_FILE):
        config = dict(_DEFAULT_CONFIG)
        config["workers"] = list(_DEFAULT_CONFIG["workers"])
        _save_config(config)
        return config

    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError):
        config = {}

    # Fill missing keys from defaults
    for key, default_val in _DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = default_val if not isinstance(default_val, list) else list(default_val)

    return config


def _save_config(config):
    """Persist full config dict to config file."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
    except OSError as e:
        # Deferred warning — caller may show QMessageBox if app is running
        print(f"Warning: could not save config: {e}", file=sys.stderr)


def _discover_llama_server(configured_path):
    """Find llama-server binary. Returns (path, found_bool)."""
    expanded = os.path.expanduser(configured_path)
    if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
        return expanded, True

    which_path = shutil.which("llama-server")
    if which_path:
        return which_path, True

    common = [
        os.path.expanduser("~/llama.cpp/build/bin/llama-server"),
        "/usr/local/bin/llama-server",
        "/usr/bin/llama-server",
        os.path.expanduser("~/.local/bin/llama-server"),
    ]
    for p in common:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p, True

    return expanded, False


# ─────────────────────────────────────────────────────────────────────────────


def local_vram():
    """Query local nvidia-smi. Returns (free_mb, total_mb) or (None, None).

    Sums across all GPUs if multiple are present.
    """
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


class WorkerPinger(QThread):
    """Background thread that TCP-pings each worker every few seconds."""

    results = pyqtSignal(list, int)  # [(host, port, name, alive, latency_ms), ...], generation

    def __init__(self, workers, agent_port, generation, parent=None):
        super().__init__(parent)
        self.workers = workers
        self._agent_port = agent_port
        self._generation = generation
        self._running = True

    def run(self):
        while self._running:
            statuses = []
            for host, port, name in self.workers:
                if host == "local":
                    statuses.append((host, port, name, True, 0.0))
                else:
                    alive, latency = self._ping(host, port)
                    # RPC port busy (server holding connection) — check agent port
                    if not alive:
                        alive, latency = self._ping(host, self._agent_port)
                    statuses.append((host, port, name, alive, latency))
            self.results.emit(statuses, self._generation)
            self.msleep(PING_INTERVAL_MS)

    def stop(self):
        self._running = False
        self.wait()

    @staticmethod
    def _ping(host, port, timeout=1.0):
        try:
            t0 = time.monotonic()
            with socket.create_connection((host, port), timeout=timeout):
                latency = (time.monotonic() - t0) * 1000
            return True, round(latency, 1)
        except (OSError, socket.timeout):
            return False, 0.0


class WorkerAgentPoller(QThread):
    """Polls worker-agent HTTP endpoints + local nvidia-smi for VRAM/battery."""

    results = pyqtSignal(dict, int)  # {host: {vram_free_mb, vram_total_mb, ...}}, generation

    def __init__(self, workers, agent_port, generation, parent=None):
        super().__init__(parent)
        self.workers = workers
        self._agent_port = agent_port
        self._generation = generation
        self._running = True

    def run(self):
        cache = {}
        while self._running:
            info = {}
            for host, _port, _name in self.workers:
                if host == "local":
                    free, total = local_vram()
                    data = {
                        "vram_free_mb": free,
                        "vram_total_mb": total,
                        "battery_pct": None,
                        "power_source": "AC",
                    }
                else:
                    data = self._query(host, self._agent_port)
                # Keep last good data if this poll returned empty
                if data:
                    cache[host] = data
                info[host] = cache.get(host, {})
            self.results.emit(info, self._generation)
            self.msleep(PING_INTERVAL_MS)

    def stop(self):
        self._running = False
        self.wait()

    @staticmethod
    def _query(host, agent_port):
        # Quick TCP check first — avoids slow urllib timeout on unreachable hosts
        try:
            with socket.create_connection((host, agent_port), timeout=0.5):
                pass
        except (OSError, socket.timeout):
            return {}
        try:
            req = urllib.request.Request(f"http://{host}:{agent_port}/")
            with urllib.request.urlopen(req, timeout=1) as resp:
                return json.loads(resp.read())
        except Exception:
            return {}


class ServerMonitor(QThread):
    """Polls llama-server /slots endpoint for inference activity."""

    slots_updated = pyqtSignal(list)
    server_health = pyqtSignal(str)

    def __init__(self, base_url="http://localhost:8080", parent=None):
        super().__init__(parent)
        self.base_url = base_url
        self._running = True

    def run(self):
        while self._running:
            try:
                req = urllib.request.Request(f"{self.base_url}/slots")
                with urllib.request.urlopen(req, timeout=2) as resp:
                    slots = json.loads(resp.read())
                self.slots_updated.emit(slots)
                self.server_health.emit("ok")
            except Exception:
                self.slots_updated.emit([])
                self.server_health.emit("loading")
            self.msleep(SERVER_POLL_INTERVAL_MS)

    def stop(self):
        self._running = False
        self.wait()


class LogReader(QThread):
    """Reads a subprocess pipe and emits lines for the log panel."""

    line_ready = pyqtSignal(str)

    def __init__(self, pipe, parent=None):
        super().__init__(parent)
        self.pipe = pipe
        self._running = True

    def run(self):
        try:
            for line in iter(self.pipe.readline, b""):
                if not self._running:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    self.line_ready.emit(text)
        except (ValueError, OSError):
            pass  # pipe closed

    def stop(self):
        self._running = False
        self.wait(2000)


# ── Worker Input Dialog ──────────────────────────────────────────────────────

class _WorkerInputDialog(QDialog):
    """Small dialog to add or edit a single worker entry."""

    def __init__(self, parent=None, name="", ip="", port=50052):
        super().__init__(parent)
        self.setWindowTitle("Worker")
        layout = QFormLayout(self)

        self.name_edit = QLineEdit(name)
        self.ip_edit = QLineEdit(ip)
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(port)

        layout.addRow("Name:", self.name_edit)
        layout.addRow("IP:", self.ip_edit)
        layout.addRow("RPC Port:", self.port_spin)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def values(self):
        return self.name_edit.text().strip(), self.ip_edit.text().strip(), self.port_spin.value()


# ── Options Dialog (Tabbed) ──────────────────────────────────────────────────

class OptionsDialog(QDialog):
    """Tabbed dialog for workers, paths, ports, and theme settings."""

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Options")
        self.setMinimumWidth(560)
        self.config = dict(config)  # working copy
        self.config["workers"] = [dict(r) for r in config.get("workers", [])]

        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # ── Tab 1: Workers ──
        workers_tab = QWidget()
        wl = QVBoxLayout(workers_tab)

        self.table = QTableWidget(len(self.config["workers"]), 3)
        self.table.setHorizontalHeaderLabels(["Name", "IP", "RPC Port"])
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self._populate_table()
        wl.addWidget(self.table)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._on_add)
        edit_btn = QPushButton("Edit")
        edit_btn.clicked.connect(self._on_edit)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._on_remove)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(edit_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addStretch()
        wl.addLayout(btn_row)

        self.tabs.addTab(workers_tab, "Workers")

        # ── Tab 2: Paths & Ports ──
        paths_tab = QWidget()
        fl = QFormLayout(paths_tab)

        # Model directory
        model_row = QHBoxLayout()
        self.model_dir_edit = QLineEdit(config.get("model_dir", "~/models"))
        model_browse = QPushButton("Browse...")
        model_browse.setFixedWidth(80)
        model_browse.clicked.connect(self._browse_model_dir)
        model_row.addWidget(self.model_dir_edit)
        model_row.addWidget(model_browse)
        fl.addRow("Model directory:", model_row)

        # llama-server path
        server_row = QHBoxLayout()
        self.server_path_edit = QLineEdit(config.get("llama_server", ""))
        server_browse = QPushButton("Browse...")
        server_browse.setFixedWidth(80)
        server_browse.clicked.connect(self._browse_server_path)
        server_row.addWidget(self.server_path_edit)
        server_row.addWidget(server_browse)
        fl.addRow("llama-server path:", server_row)

        # Agent port
        self.agent_port_spin = QSpinBox()
        self.agent_port_spin.setRange(1, 65535)
        self.agent_port_spin.setValue(config.get("agent_port", 50053))
        fl.addRow("Agent port:", self.agent_port_spin)

        # Server listen port
        self.server_port_spin = QSpinBox()
        self.server_port_spin.setRange(1, 65535)
        self.server_port_spin.setValue(config.get("server_port", 8080))
        fl.addRow("Server listen port:", self.server_port_spin)

        # Local machine name
        self.local_name_edit = QLineEdit(config.get("local_name", "ag1 (local)"))
        fl.addRow("Local machine name:", self.local_name_edit)

        # Parallel slots
        self.parallel_spin = QSpinBox()
        self.parallel_spin.setRange(1, 64)
        self.parallel_spin.setValue(config.get("parallel_slots", 1))
        fl.addRow("Parallel slots:", self.parallel_spin)

        # Theme
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["dark", "light"])
        current_theme = config.get("theme", "dark")
        idx = self.theme_combo.findText(current_theme)
        if idx >= 0:
            self.theme_combo.setCurrentIndex(idx)
        fl.addRow("Theme:", self.theme_combo)

        self.tabs.addTab(paths_tab, "Paths && Ports")

        # ── Dialog buttons ──
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._apply_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _populate_table(self):
        workers = self.config["workers"]
        self.table.setRowCount(len(workers))
        for row, r in enumerate(workers):
            self.table.setItem(row, 0, QTableWidgetItem(r["name"]))
            self.table.setItem(row, 1, QTableWidgetItem(r["ip"]))
            item = QTableWidgetItem(str(r["port"]))
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 2, item)

    def _on_add(self):
        dlg = _WorkerInputDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            name, ip, port = dlg.values()
            if not ip:
                QMessageBox.warning(self, "Invalid", "IP address is required.")
                return
            self.config["workers"].append({"ip": ip, "port": port, "name": name or ip})
            self._populate_table()

    def _on_edit(self):
        row = self.table.currentRow()
        if row < 0:
            return
        r = self.config["workers"][row]
        dlg = _WorkerInputDialog(self, name=r["name"], ip=r["ip"], port=r["port"])
        if dlg.exec_() == QDialog.Accepted:
            name, ip, port = dlg.values()
            if not ip:
                QMessageBox.warning(self, "Invalid", "IP address is required.")
                return
            self.config["workers"][row] = {"ip": ip, "port": port, "name": name or ip}
            self._populate_table()

    def _on_remove(self):
        row = self.table.currentRow()
        if row < 0:
            return
        name = self.config["workers"][row].get("name", "this worker")
        reply = QMessageBox.question(
            self, "Confirm Remove",
            f"Remove worker \"{name}\"?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            del self.config["workers"][row]
            self._populate_table()

    def _browse_model_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Model Directory",
            os.path.expanduser(self.model_dir_edit.text()),
        )
        if d:
            self.model_dir_edit.setText(d)

    def _browse_server_path(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select llama-server Binary",
            os.path.expanduser(self.server_path_edit.text()),
        )
        if f:
            self.server_path_edit.setText(f)

    def _apply_and_accept(self):
        self.config["model_dir"] = self.model_dir_edit.text().strip() or "~/models"
        self.config["llama_server"] = self.server_path_edit.text().strip()
        self.config["agent_port"] = self.agent_port_spin.value()
        self.config["server_port"] = self.server_port_spin.value()
        self.config["local_name"] = self.local_name_edit.text().strip() or "ag1 (local)"
        self.config["parallel_slots"] = self.parallel_spin.value()
        self.config["theme"] = self.theme_combo.currentText()
        self.accept()


# ── Sampler Dialog ────────────────────────────────────────────────────────────

class SamplerDialog(QDialog):
    """Tabbed dialog for sampler settings with profile CRUD and model associations."""

    def __init__(self, config, model_list=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sampler Settings")
        self.setMinimumSize(540, 600)
        self.config = dict(config)
        self._model_list = model_list or []

        # Ensure sampler config keys exist
        if "sampler_profiles" not in self.config:
            self.config["sampler_profiles"] = {"Default": dict(_DEFAULT_SAMPLER_PROFILE)}
        profiles = self.config["sampler_profiles"]
        if "Default" not in profiles:
            profiles["Default"] = dict(_DEFAULT_SAMPLER_PROFILE)
        if "active_sampler_profile" not in self.config:
            self.config["active_sampler_profile"] = "Default"
        if "model_profile_map" not in self.config:
            self.config["model_profile_map"] = {}

        layout = QVBoxLayout(self)

        # ── Profile selector row ──
        profile_row = QHBoxLayout()
        profile_row.addWidget(QLabel("Profile:"))
        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumWidth(160)
        self._refresh_profile_combo()
        self.profile_combo.currentTextChanged.connect(self._on_profile_changed)
        profile_row.addWidget(self.profile_combo, 1)

        new_btn = QPushButton("New")
        new_btn.setFixedWidth(55)
        new_btn.clicked.connect(self._on_new_profile)
        profile_row.addWidget(new_btn)

        self.save_btn = QPushButton("Save")
        self.save_btn.setFixedWidth(55)
        self.save_btn.clicked.connect(self._on_save_profile)
        profile_row.addWidget(self.save_btn)

        self.rename_btn = QPushButton("Rename")
        self.rename_btn.setFixedWidth(65)
        self.rename_btn.clicked.connect(self._on_rename_profile)
        profile_row.addWidget(self.rename_btn)

        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setFixedWidth(60)
        self.delete_btn.clicked.connect(self._on_delete_profile)
        profile_row.addWidget(self.delete_btn)

        layout.addLayout(profile_row)

        # ── Tabbed settings area ──
        self.tabs = QTabWidget()
        self._fields = {}  # key → QLineEdit / QCheckBox / QTextEdit

        for tab_name, fields in _SAMPLER_TABS.items():
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            container = QWidget()
            form = QFormLayout(container)
            form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

            for key, label in fields:
                default_val = _DEFAULT_SAMPLER_PROFILE.get(key, "")
                if key in _BOOL_FIELDS:
                    widget = QCheckBox()
                    widget.setChecked(bool(default_val))
                elif key in _MULTILINE_FIELDS:
                    widget = QTextEdit()
                    widget.setMaximumHeight(100)
                    widget.setPlainText(str(default_val) if default_val else "")
                else:
                    widget = QLineEdit(str(default_val))

                self._fields[key] = widget
                form.addRow(label + ":", widget)

            scroll.setWidget(container)
            self.tabs.addTab(scroll, tab_name)

        # ── Model Associations tab ──
        assoc_widget = QWidget()
        assoc_layout = QVBoxLayout(assoc_widget)
        assoc_layout.addWidget(QLabel(
            "Assign sampler profiles to models. When a model is launched,\n"
            "its assigned profile will be used automatically."
        ))

        self.assoc_table = QTableWidget(0, 3)
        self.assoc_table.setHorizontalHeaderLabels(["Model", "Profile", ""])
        hdr = self.assoc_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.assoc_table.verticalHeader().setVisible(False)
        self.assoc_table.setEditTriggers(QTableWidget.NoEditTriggers)
        assoc_layout.addWidget(self.assoc_table)

        assoc_btn_row = QHBoxLayout()
        add_assoc_btn = QPushButton("Add Association")
        add_assoc_btn.clicked.connect(self._on_add_association)
        assoc_btn_row.addWidget(add_assoc_btn)
        assoc_btn_row.addStretch()
        assoc_layout.addLayout(assoc_btn_row)

        self.tabs.addTab(assoc_widget, "Model Associations")

        layout.addWidget(self.tabs)

        # ── Dialog buttons ──
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._apply_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Load the active profile into fields
        self._populate_fields(self.config["active_sampler_profile"])
        self._populate_assoc_table()
        self._update_default_protection()

    # ── Profile combo helpers ──

    def _refresh_profile_combo(self):
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        profiles = self.config["sampler_profiles"]
        # Always put Default first
        self.profile_combo.addItem("Default")
        for name in sorted(profiles.keys()):
            if name != "Default":
                self.profile_combo.addItem(name)
        active = self.config.get("active_sampler_profile", "Default")
        idx = self.profile_combo.findText(active)
        if idx >= 0:
            self.profile_combo.setCurrentIndex(idx)
        self.profile_combo.blockSignals(False)

    def _update_default_protection(self):
        is_default = self.profile_combo.currentText() == "Default"
        self.save_btn.setEnabled(not is_default)
        self.delete_btn.setEnabled(not is_default)
        self.rename_btn.setEnabled(not is_default)
        # Make fields read-only for Default
        for key, widget in self._fields.items():
            if isinstance(widget, QCheckBox):
                widget.setEnabled(not is_default)
            elif isinstance(widget, QTextEdit):
                widget.setReadOnly(is_default)
            else:
                widget.setReadOnly(is_default)

    def _populate_fields(self, profile_name):
        profiles = self.config["sampler_profiles"]
        profile = profiles.get(profile_name, _DEFAULT_SAMPLER_PROFILE)
        for key, widget in self._fields.items():
            val = profile.get(key, _DEFAULT_SAMPLER_PROFILE.get(key, ""))
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(val))
            elif isinstance(widget, QTextEdit):
                widget.setPlainText(str(val) if val else "")
            else:
                widget.setText(str(val))

    def _read_fields(self):
        """Read all field values, returning a dict with properly typed values."""
        result = {}
        for key, widget in self._fields.items():
            if isinstance(widget, QCheckBox):
                result[key] = widget.isChecked()
            elif isinstance(widget, QTextEdit):
                result[key] = widget.toPlainText().strip()
            else:
                text = widget.text().strip()
                # Try to parse as number
                if key not in _MULTILINE_FIELDS:
                    try:
                        result[key] = int(text)
                        continue
                    except ValueError:
                        pass
                    try:
                        result[key] = float(text)
                        continue
                    except ValueError:
                        pass
                result[key] = text
        return result

    # ── Profile CRUD ──

    def _on_profile_changed(self, name):
        if not name:
            return
        self._populate_fields(name)
        self._update_default_protection()

    def _on_new_profile(self):
        name, ok = QInputDialog.getText(self, "New Profile", "Profile name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        if name.lower() == "default":
            QMessageBox.warning(self, "Reserved Name",
                                "Cannot create a profile named 'Default'.")
            return
        if name in self.config["sampler_profiles"]:
            QMessageBox.warning(self, "Duplicate",
                                f"Profile '{name}' already exists.")
            return
        # Clone current field values into new profile
        self.config["sampler_profiles"][name] = self._read_fields()
        self._refresh_profile_combo()
        self.profile_combo.setCurrentText(name)

    def _on_save_profile(self):
        name = self.profile_combo.currentText()
        if name == "Default":
            QMessageBox.warning(self, "Protected",
                                "The Default profile cannot be overwritten.")
            return
        self.config["sampler_profiles"][name] = self._read_fields()
        QMessageBox.information(self, "Saved", f"Profile '{name}' saved.")

    def _on_rename_profile(self):
        old_name = self.profile_combo.currentText()
        if old_name == "Default":
            return
        new_name, ok = QInputDialog.getText(self, "Rename Profile",
                                            "New name:", text=old_name)
        if not ok or not new_name.strip():
            return
        new_name = new_name.strip()
        if new_name.lower() == "default":
            QMessageBox.warning(self, "Reserved Name",
                                "Cannot rename to 'Default'.")
            return
        if new_name in self.config["sampler_profiles"]:
            QMessageBox.warning(self, "Duplicate",
                                f"Profile '{new_name}' already exists.")
            return
        profiles = self.config["sampler_profiles"]
        profiles[new_name] = profiles.pop(old_name)
        # Update model associations
        model_map = self.config.get("model_profile_map", {})
        for model, prof in list(model_map.items()):
            if prof == old_name:
                model_map[model] = new_name
        if self.config.get("active_sampler_profile") == old_name:
            self.config["active_sampler_profile"] = new_name
        self._refresh_profile_combo()
        self.profile_combo.setCurrentText(new_name)
        self._populate_assoc_table()

    def _on_delete_profile(self):
        name = self.profile_combo.currentText()
        if name == "Default":
            QMessageBox.warning(self, "Protected",
                                "The Default profile cannot be deleted.")
            return
        reply = QMessageBox.question(
            self, "Delete Profile",
            f"Delete profile '{name}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        del self.config["sampler_profiles"][name]
        # Clean up model associations pointing to this profile
        model_map = self.config.get("model_profile_map", {})
        for model in [m for m, p in model_map.items() if p == name]:
            del model_map[model]
        if self.config.get("active_sampler_profile") == name:
            self.config["active_sampler_profile"] = "Default"
        self._refresh_profile_combo()
        self.profile_combo.setCurrentText("Default")
        self._populate_assoc_table()

    # ── Model Associations ──

    def _populate_assoc_table(self):
        model_map = self.config.get("model_profile_map", {})
        self.assoc_table.setRowCount(len(model_map))
        for row, (model, profile) in enumerate(sorted(model_map.items())):
            self.assoc_table.setItem(row, 0, QTableWidgetItem(model))
            self.assoc_table.setItem(row, 1, QTableWidgetItem(profile))
            remove_btn = QPushButton("Remove")
            remove_btn.clicked.connect(lambda checked, m=model: self._remove_association(m))
            self.assoc_table.setCellWidget(row, 2, remove_btn)

    def _on_add_association(self):
        if not self._model_list:
            QMessageBox.information(self, "No Models",
                                    "No models found. Add models to your model directory first.")
            return
        model, ok = QInputDialog.getItem(
            self, "Select Model", "Model:", self._model_list, 0, False)
        if not ok or not model:
            return
        profile_names = list(self.config["sampler_profiles"].keys())
        profile, ok = QInputDialog.getItem(
            self, "Select Profile", "Sampler profile:", profile_names, 0, False)
        if not ok or not profile:
            return
        self.config.setdefault("model_profile_map", {})[model] = profile
        self._populate_assoc_table()

    def _remove_association(self, model):
        model_map = self.config.get("model_profile_map", {})
        if model in model_map:
            del model_map[model]
        self._populate_assoc_table()

    # ── Apply ──

    def _apply_and_accept(self):
        name = self.profile_combo.currentText()
        # Auto-save current edits if not Default
        if name != "Default":
            self.config["sampler_profiles"][name] = self._read_fields()
        self.config["active_sampler_profile"] = name
        self.accept()


# ── Generate Worker Dialog ────────────────────────────────────────────────────

class GenerateWorkerDialog(QDialog):
    """Dialog to generate a pre-configured worker script for any OS."""

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generate Worker Script")
        self.setMinimumWidth(400)
        self._config = config

        layout = QFormLayout(self)

        # Worker selector
        self.worker_combo = QComboBox()
        for w in config.get("workers", []):
            self.worker_combo.addItem(w.get("name", w["ip"]), w)
        self.worker_combo.currentIndexChanged.connect(self._on_worker_changed)
        layout.addRow("Worker:", self.worker_combo)

        # RPC Port
        self.rpc_port_spin = QSpinBox()
        self.rpc_port_spin.setRange(1, 65535)
        layout.addRow("RPC Port:", self.rpc_port_spin)

        # Agent Port
        self.agent_port_spin = QSpinBox()
        self.agent_port_spin.setRange(1, 65535)
        self.agent_port_spin.setValue(config.get("agent_port", 50053))
        layout.addRow("Agent Port:", self.agent_port_spin)

        # Target OS
        self.os_combo = QComboBox()
        self.os_combo.addItems(["Auto-detect", "Windows", "Linux", "macOS"])
        layout.addRow("Target OS:", self.os_combo)

        # Buttons
        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save As...")
        save_btn.clicked.connect(self._on_save)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(cancel_btn)
        layout.addRow(btn_row)

        # Initialize ports from first worker
        self._on_worker_changed(0)

    def _on_worker_changed(self, index):
        data = self.worker_combo.currentData()
        if data:
            self.rpc_port_spin.setValue(data.get("port", 50052))

    def _on_save(self):
        target_os = self.os_combo.currentText()
        if target_os == "Auto-detect":
            target_os = platform.system()
            if target_os == "Darwin":
                target_os = "macOS"

        # Read template
        template_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "workers", "llama-worker.py"
        )
        try:
            with open(template_path, "r") as f:
                content = f.read()
        except OSError as e:
            QMessageBox.critical(self, "Error", f"Could not read template:\n{e}")
            return

        # Substitute ports
        content = content.replace(
            "DEFAULT_RPC_PORT = 50052",
            f"DEFAULT_RPC_PORT = {self.rpc_port_spin.value()}",
        )
        content = content.replace(
            "DEFAULT_AGENT_PORT = 50053",
            f"DEFAULT_AGENT_PORT = {self.agent_port_spin.value()}",
        )

        if target_os == "Windows":
            # Strip shebang, prepend polyglot header, save as .cmd
            if content.startswith("#!/"):
                content = content[content.index("\n") + 1:]
            content = _CMD_POLYGLOT_HEADER + content
            default_name = "llama-worker.cmd"
            filter_str = "CMD Script (*.cmd);;All Files (*)"
        else:
            default_name = "llama-worker.py"
            filter_str = "Python Script (*.py);;All Files (*)"

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Worker Script", default_name, filter_str,
        )
        if not path:
            return

        try:
            with open(path, "w", newline="\n") as f:
                f.write(content)
            if target_os != "Windows":
                os.chmod(path, 0o755)
            QMessageBox.information(
                self, "Success",
                f"Worker script saved to:\n{path}\n\n"
                f"RPC port: {self.rpc_port_spin.value()}\n"
                f"Agent port: {self.agent_port_spin.value()}\n"
                f"Target OS: {target_os}",
            )
            self.accept()
        except OSError as e:
            QMessageBox.critical(self, "Error", f"Could not save file:\n{e}")


# ── SSH Deploy Thread ─────────────────────────────────────────────────────────

class SSHDeployThread(QThread):
    """Background thread that deploys worker script to a remote machine via SSH."""

    log_line = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    _SSH_OPTS = [
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=accept-new",
    ]

    def __init__(self, host, username, key_file, rpc_port, agent_port,
                 remote_path, template_path, parent=None):
        super().__init__(parent)
        self._host = host
        self._username = username
        self._key_file = key_file
        self._rpc_port = rpc_port
        self._agent_port = agent_port
        self._remote_path = remote_path
        self._template_path = template_path

    def _ssh_base(self):
        cmd = ["ssh"] + self._SSH_OPTS
        if self._key_file:
            cmd += ["-i", self._key_file]
        cmd.append(f"{self._username}@{self._host}")
        return cmd

    def _run_cmd(self, cmd, description):
        self.log_line.emit(f"  $ {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
            )
            if result.stdout.strip():
                self.log_line.emit(f"  {result.stdout.strip()}")
            if result.stderr.strip():
                self.log_line.emit(f"  {result.stderr.strip()}")
            return result
        except subprocess.TimeoutExpired:
            self.log_line.emit(f"  TIMEOUT: {description}")
            return None
        except FileNotFoundError:
            self.log_line.emit(f"  ERROR: command not found")
            return None

    def run(self):
        try:
            # Step 1: Detect remote OS
            self.log_line.emit("[1/6] Detecting remote OS...")
            result = self._run_cmd(self._ssh_base() + ["uname", "-s"], "uname")
            if not result or result.returncode != 0:
                self.finished_signal.emit(False, "Failed to connect or detect OS")
                return

            uname = result.stdout.strip()
            if uname.startswith(("MINGW", "MSYS", "CYGWIN")):
                remote_os = "Windows"
            elif uname == "Darwin":
                remote_os = "macOS"
            else:
                remote_os = "Linux"
            self.log_line.emit(f"  Remote OS: {remote_os} ({uname})")

            # Step 2: Read template and substitute ports
            self.log_line.emit("[2/6] Preparing worker script...")
            try:
                with open(self._template_path, "r") as f:
                    content = f.read()
            except OSError as e:
                self.finished_signal.emit(False, f"Cannot read template: {e}")
                return

            content = content.replace(
                "DEFAULT_RPC_PORT = 50052",
                f"DEFAULT_RPC_PORT = {self._rpc_port}",
            )
            content = content.replace(
                "DEFAULT_AGENT_PORT = 50053",
                f"DEFAULT_AGENT_PORT = {self._agent_port}",
            )

            if remote_os == "Windows":
                if content.startswith("#!/"):
                    content = content[content.index("\n") + 1:]
                content = _CMD_POLYGLOT_HEADER + content

            self.log_line.emit(f"  Ports: RPC={self._rpc_port}, Agent={self._agent_port}")

            # Step 3: Write to temp file
            suffix = ".cmd" if remote_os == "Windows" else ".py"
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=suffix, delete=False, newline="\n",
            )
            tmp.write(content)
            tmp.close()

            # Step 4: Kill existing worker on remote
            self.log_line.emit("[3/6] Stopping existing worker (if any)...")
            self._run_cmd(
                self._ssh_base() + ["pkill", "-f", "llama-worker", "||", "true"],
                "pkill",
            )

            # Step 5: SCP file to remote
            self.log_line.emit(f"[4/6] Uploading to {self._remote_path}...")
            scp_cmd = ["scp"] + self._SSH_OPTS
            if self._key_file:
                scp_cmd += ["-i", self._key_file]
            scp_cmd += [tmp.name, f"{self._username}@{self._host}:{self._remote_path}"]
            result = self._run_cmd(scp_cmd, "scp")
            os.unlink(tmp.name)
            if not result or result.returncode != 0:
                self.finished_signal.emit(False, "SCP upload failed")
                return

            # Step 6: chmod +x (non-Windows)
            if remote_os != "Windows":
                self.log_line.emit("[5/6] Setting executable permission...")
                self._run_cmd(
                    self._ssh_base() + ["chmod", "+x", self._remote_path],
                    "chmod",
                )
            else:
                self.log_line.emit("[5/6] Skipped chmod (Windows)")

            # Step 7: Start worker
            self.log_line.emit("[6/6] Starting worker...")
            if remote_os == "Windows":
                start_cmd = self._ssh_base() + [
                    f"start /B python {self._remote_path}",
                ]
            else:
                start_cmd = self._ssh_base() + [
                    f"nohup python3 {self._remote_path} > /dev/null 2>&1 &",
                ]
            self._run_cmd(start_cmd, "start worker")

            self.finished_signal.emit(True, f"Worker deployed to {self._host}")

        except Exception as e:
            self.finished_signal.emit(False, str(e))


# ── Deploy SSH Dialog ─────────────────────────────────────────────────────────

class DeploySSHDialog(QDialog):
    """Dialog to deploy a worker script to a remote machine via SSH."""

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Deploy Worker via SSH")
        self.setMinimumWidth(520)
        self.setMinimumHeight(400)
        self._config = config
        self._deploy_thread = None

        layout = QVBoxLayout(self)

        form = QFormLayout()

        # Worker selector
        self.worker_combo = QComboBox()
        for w in config.get("workers", []):
            self.worker_combo.addItem(
                f"{w.get('name', w['ip'])}  ({w['ip']})", w,
            )
        self.worker_combo.currentIndexChanged.connect(self._on_worker_changed)
        form.addRow("Worker:", self.worker_combo)

        # RPC Port
        self.rpc_port_spin = QSpinBox()
        self.rpc_port_spin.setRange(1, 65535)
        form.addRow("RPC Port:", self.rpc_port_spin)

        # Agent Port
        self.agent_port_spin = QSpinBox()
        self.agent_port_spin.setRange(1, 65535)
        self.agent_port_spin.setValue(config.get("agent_port", 50053))
        form.addRow("Agent Port:", self.agent_port_spin)

        # SSH Username
        self.username_edit = QLineEdit(os.environ.get("USER", ""))
        form.addRow("SSH Username:", self.username_edit)

        # SSH Key File
        key_row = QHBoxLayout()
        default_key = os.path.expanduser("~/.ssh/id_rsa")
        if not os.path.isfile(default_key):
            ed25519 = os.path.expanduser("~/.ssh/id_ed25519")
            if os.path.isfile(ed25519):
                default_key = ed25519
        self.key_edit = QLineEdit(default_key)
        key_browse = QPushButton("Browse...")
        key_browse.setFixedWidth(80)
        key_browse.clicked.connect(self._browse_key)
        key_row.addWidget(self.key_edit)
        key_row.addWidget(key_browse)
        form.addRow("SSH Key File:", key_row)

        # Remote Path
        self.remote_path_edit = QLineEdit("~/llama-worker.py")
        form.addRow("Remote Path:", self.remote_path_edit)

        layout.addLayout(form)

        # Buttons
        btn_row = QHBoxLayout()
        self.test_btn = QPushButton("Test Connection")
        self.test_btn.clicked.connect(self._on_test)
        self.deploy_btn = QPushButton("Deploy")
        self.deploy_btn.clicked.connect(self._on_deploy)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.test_btn)
        btn_row.addWidget(self.deploy_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        # Log area
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFont(QFont("Monospace", 9))
        self.log_area.setStyleSheet(
            "QTextEdit { background-color: #1e1e1e; color: #dcdcdc; }"
        )
        layout.addWidget(self.log_area)

        # Initialize from first worker
        self._on_worker_changed(0)

    def _on_worker_changed(self, index):
        data = self.worker_combo.currentData()
        if data:
            self.rpc_port_spin.setValue(data.get("port", 50052))

    def _browse_key(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select SSH Key",
            os.path.expanduser("~/.ssh/"),
        )
        if f:
            self.key_edit.setText(f)

    def _get_host(self):
        data = self.worker_combo.currentData()
        return data["ip"] if data else ""

    def _on_test(self):
        host = self._get_host()
        username = self.username_edit.text().strip()
        key_file = self.key_edit.text().strip()
        if not host or not username:
            self.log_area.append("ERROR: Worker and username are required.")
            return

        self.log_area.append(f"Testing connection to {username}@{host}...")
        cmd = ["ssh",
               "-o", "BatchMode=yes",
               "-o", "ConnectTimeout=10",
               "-o", "StrictHostKeyChecking=accept-new"]
        if key_file:
            cmd += ["-i", key_file]
        cmd += [f"{username}@{host}", "echo", "ok"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode == 0 and "ok" in result.stdout:
                self.log_area.append("  Connection successful!")
            else:
                self.log_area.append(f"  Connection failed (code {result.returncode})")
                if result.stderr.strip():
                    self.log_area.append(f"  {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            self.log_area.append("  Connection timed out.")
        except FileNotFoundError:
            self.log_area.append("  ERROR: ssh command not found.")

    def _on_deploy(self):
        host = self._get_host()
        username = self.username_edit.text().strip()
        key_file = self.key_edit.text().strip()
        remote_path = self.remote_path_edit.text().strip()
        if not host or not username or not remote_path:
            self.log_area.append("ERROR: Worker, username, and remote path are required.")
            return

        template_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "workers", "llama-worker.py"
        )
        if not os.path.isfile(template_path):
            self.log_area.append(f"ERROR: Template not found at {template_path}")
            return

        self.log_area.clear()
        self.log_area.append(f"Deploying to {username}@{host}:{remote_path}")
        self.log_area.append("")

        # Disable buttons during deploy
        self.test_btn.setEnabled(False)
        self.deploy_btn.setEnabled(False)

        self._deploy_thread = SSHDeployThread(
            host=host,
            username=username,
            key_file=key_file,
            rpc_port=self.rpc_port_spin.value(),
            agent_port=self.agent_port_spin.value(),
            remote_path=remote_path,
            template_path=template_path,
            parent=self,
        )
        self._deploy_thread.log_line.connect(self._on_log_line)
        self._deploy_thread.finished_signal.connect(self._on_deploy_finished)
        self._deploy_thread.start()

    def _on_log_line(self, text):
        self.log_area.append(text)
        sb = self.log_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_deploy_finished(self, success, message):
        self.test_btn.setEnabled(True)
        self.deploy_btn.setEnabled(True)
        self.log_area.append("")
        if success:
            self.log_area.append(f"SUCCESS: {message}")
        else:
            self.log_area.append(f"FAILED: {message}")


# ── Theme ────────────────────────────────────────────────────────────────────

def _apply_theme(app, theme_name):
    """Apply dark or light Fusion theme to the application."""
    app.setStyle("Fusion")
    if theme_name == "dark":
        p = QPalette()
        p.setColor(QPalette.Window, QColor(53, 53, 53))
        p.setColor(QPalette.WindowText, Qt.white)
        p.setColor(QPalette.Base, QColor(35, 35, 35))
        p.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        p.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
        p.setColor(QPalette.ToolTipText, Qt.white)
        p.setColor(QPalette.Text, Qt.white)
        p.setColor(QPalette.Button, QColor(53, 53, 53))
        p.setColor(QPalette.ButtonText, Qt.white)
        p.setColor(QPalette.BrightText, Qt.red)
        p.setColor(QPalette.Link, QColor(42, 130, 218))
        p.setColor(QPalette.Highlight, QColor(42, 130, 218))
        p.setColor(QPalette.HighlightedText, QColor(35, 35, 35))
        p.setColor(QPalette.Disabled, QPalette.Text, QColor(127, 127, 127))
        p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(127, 127, 127))
        app.setPalette(p)
    else:
        app.setPalette(app.style().standardPalette())


# ── Help Window (non-blocking) ────────────────────────────────────────────────

class HelpWindow(QDialog):
    """Non-modal scrollable README viewer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LlamaHerder Help")
        self.setMinimumSize(620, 500)
        self.resize(680, 600)
        self.setModal(False)

        layout = QVBoxLayout(self)

        self.text_view = QTextEdit()
        self.text_view.setReadOnly(True)
        self.text_view.setFont(QFont("Monospace", 10))

        # Load README.md from the same directory as the script
        readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
        try:
            with open(readme_path, "r") as f:
                content = f.read()
            self.text_view.setMarkdown(content)
        except OSError:
            self.text_view.setPlainText("README.md not found.")

        layout.addWidget(self.text_view)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)


# ── Main Window ──────────────────────────────────────────────────────────────

class LauncherWindow(QMainWindow):
    def __init__(self, config=None):
        super().__init__()
        self.server_proc = None
        self.server_monitor = None
        self.log_reader = None
        self._connected_workers = []
        self._agent_data = {}
        self._worker_generation = 0
        self._tray_exiting = False
        self._tray_notified = False

        self._config = config if config is not None else _load_config()
        self._remotes = self._config.get("workers", [])
        self._llama_server_path, self._llama_server_found = _discover_llama_server(
            self._config.get("llama_server", "")
        )

        self._rebuild_workers()
        self._init_ui()
        self._setup_tray()
        self._start_pinger()
        self._start_agent_poller()

        if not self._llama_server_found:
            QTimer.singleShot(500, self._warn_no_server)

    def _warn_no_server(self):
        QMessageBox.warning(
            self, "llama-server Not Found",
            f"Could not find llama-server at:\n{self._llama_server_path}\n\n"
            "You can set the correct path in File > Options.",
        )

    def _rebuild_workers(self):
        """Rebuild self.workers from local + loaded remotes."""
        local_name = self._config.get("local_name", "ag1 (local)")
        self.workers = [("local", 0, local_name)] + [
            (r["ip"], r["port"], r["name"]) for r in self._remotes
        ]

    # ── UI ────────────────────────────────────────────────────────────────

    def _init_ui(self):
        self.setWindowTitle("LlamaHerder Cluster Manager")
        self.setMinimumWidth(1100)
        self.resize(1100, 800)

        # ── Menu bar ──
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")

        options_action = QAction("Options", self)
        options_action.triggered.connect(self._open_options)
        file_menu.addAction(options_action)

        sampler_action = QAction("Sampler", self)
        sampler_action.triggered.connect(self._open_sampler)
        file_menu.addAction(sampler_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self._tray_exit)
        file_menu.addAction(exit_action)

        # ── Workers menu ──
        workers_menu = menu_bar.addMenu("Workers")

        gen_worker_action = QAction("Generate Worker...", self)
        gen_worker_action.triggered.connect(self._open_generate_worker)
        workers_menu.addAction(gen_worker_action)

        deploy_ssh_action = QAction("Deploy via SSH...", self)
        deploy_ssh_action.triggered.connect(self._open_deploy_ssh)
        workers_menu.addAction(deploy_ssh_action)

        # ── Help button (direct action, no dropdown) ──
        help_action = QAction("Help", self)
        help_action.triggered.connect(self._open_help)
        menu_bar.addAction(help_action)

        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter)

        # ── Left panel (controls) ──
        left_widget = QWidget()
        self.main_layout = QVBoxLayout(left_widget)
        self.main_layout.setSpacing(6)
        self.main_layout.setContentsMargins(6, 6, 6, 6)

        # ── Right panel (server log) ──
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(4)
        right_layout.addWidget(self._make_label("Server Output"))

        # Log search bar
        search_row = QHBoxLayout()
        self.log_search_edit = QLineEdit()
        self.log_search_edit.setPlaceholderText("Search log...")
        self.log_search_edit.textChanged.connect(self._log_search_reset)
        self.log_search_edit.returnPressed.connect(self._log_search_next)
        search_prev_btn = QPushButton("Prev")
        search_prev_btn.setFixedWidth(50)
        search_prev_btn.clicked.connect(self._log_search_prev)
        search_next_btn = QPushButton("Next")
        search_next_btn.setFixedWidth(50)
        search_next_btn.clicked.connect(self._log_search_next)
        search_row.addWidget(self.log_search_edit)
        search_row.addWidget(search_prev_btn)
        search_row.addWidget(search_next_btn)
        right_layout.addLayout(search_row)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Monospace", 9))
        self.log_view.setStyleSheet(
            "QTextEdit { background-color: #1e1e1e; color: #dcdcdc; }"
        )
        right_layout.addWidget(self.log_view)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([550, 550])

        # Worker status table — 7 columns
        self.main_layout.addWidget(self._make_label("RPC Workers"))
        self.table = QTableWidget(len(self.workers), 7)
        self.table.setHorizontalHeaderLabels(
            ["Host", "Port", "Status", "Latency", "VRAM", "Battery", "Power"]
        )
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 7):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.verticalHeader().setVisible(False)
        self._resize_worker_table()
        self._populate_worker_table()
        self.main_layout.addWidget(self.table)

        # Model + Context headers
        header_row = QHBoxLayout()
        header_row.addWidget(self._make_label("Model"), 1)
        ctx_header = self._make_label("Context")
        ctx_header.setFixedWidth(80)
        header_row.addWidget(ctx_header)
        spacer_label = QLabel("")
        spacer_label.setFixedWidth(80)
        header_row.addWidget(spacer_label)
        self.main_layout.addLayout(header_row)

        # Model dropdown, Context dropdown, Refresh — one line, matched to Launch height
        selector_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setFixedHeight(36)
        self.model_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.model_combo.currentIndexChanged.connect(self._update_context_options)
        selector_row.addWidget(self.model_combo, 1)
        self.ctx_combo = QComboBox()
        self.ctx_combo.setFixedHeight(36)
        self.ctx_combo.setFixedWidth(80)
        self.ctx_combo.currentIndexChanged.connect(self._update_overhead)
        selector_row.addWidget(self.ctx_combo)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedWidth(80)
        refresh_btn.setFixedHeight(36)
        refresh_btn.clicked.connect(self._scan_models)
        selector_row.addWidget(refresh_btn)
        self.main_layout.addLayout(selector_row)

        # Estimated Overhead line
        self.overhead_label = QLabel("")
        self.overhead_label.setFont(QFont("sans-serif", 11, QFont.Bold))
        self.main_layout.addWidget(self.overhead_label)

        # Recommendation (below overhead)
        self.cluster_rec_label = QLabel("")
        self.cluster_rec_label.setWordWrap(True)
        self.cluster_rec_label.setFont(QFont("sans-serif", 11, QFont.Bold))
        self.cluster_rec_label.setStyleSheet("color: #2ecc71;")
        self.main_layout.addWidget(self.cluster_rec_label)

        # Status row (label + browser button)
        status_row = QHBoxLayout()
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        status_row.addWidget(self.status_label, 1)
        self.browser_btn = QPushButton("Launch Browser")
        self.browser_btn.setFixedWidth(120)
        self.browser_btn.setFixedHeight(28)
        self.browser_btn.setVisible(False)
        self.browser_btn.clicked.connect(self._open_browser)
        status_row.addWidget(self.browser_btn)
        self.main_layout.addLayout(status_row)

        # Action buttons
        btn_row = QHBoxLayout()
        self.launch_btn = QPushButton("Launch")
        self.launch_btn.setFixedHeight(36)
        self.launch_btn.clicked.connect(self._on_launch)
        btn_row.addWidget(self.launch_btn)

        self.unload_btn = QPushButton("Unload")
        self.unload_btn.setFixedHeight(36)
        self.unload_btn.setToolTip("Kill all llama-server processes to free VRAM")
        self.unload_btn.clicked.connect(self._on_unload)
        btn_row.addWidget(self.unload_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setFixedHeight(36)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self.stop_btn)

        exit_btn = QPushButton("Exit")
        exit_btn.setFixedHeight(36)
        exit_btn.clicked.connect(self._tray_exit)
        btn_row.addWidget(exit_btn)
        self.main_layout.addLayout(btn_row)

        # ── Inference monitor panel (hidden until Launch) ────────────────
        self.monitor_frame = QFrame()
        self.monitor_frame.setFrameShape(QFrame.StyledPanel)
        self.monitor_frame.setVisible(False)
        mon = QVBoxLayout(self.monitor_frame)

        mon.addWidget(self._make_label("Inference Monitor"))

        self.server_status_label = QLabel("Waiting for server...")
        self.server_status_label.setAlignment(Qt.AlignCenter)
        mon.addWidget(self.server_status_label)

        self.slots_table = QTableWidget(0, 5)
        self.slots_table.setHorizontalHeaderLabels(
            ["Slot", "State", "Prompt", "Tokens", "Speed"]
        )
        shdr = self.slots_table.horizontalHeader()
        shdr.setSectionResizeMode(2, QHeaderView.Stretch)
        for col in (0, 1, 3, 4):
            shdr.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.slots_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.slots_table.setSelectionMode(QTableWidget.NoSelection)
        self.slots_table.verticalHeader().setVisible(False)
        mon.addWidget(self.slots_table)

        self.worker_load_label = QLabel("")
        self.worker_load_label.setWordWrap(True)
        mon.addWidget(self.worker_load_label)

        self.main_layout.addWidget(self.monitor_frame)
        self.main_layout.addStretch(1)

        self._scan_models()

    # ── Log search ────────────────────────────────────────────────────────

    def _log_search_reset(self):
        cursor = self.log_view.textCursor()
        cursor.movePosition(cursor.Start)
        self.log_view.setTextCursor(cursor)

    def _log_search_next(self):
        text = self.log_search_edit.text()
        if not text:
            return
        if not self.log_view.find(text):
            # Wrap around
            cursor = self.log_view.textCursor()
            cursor.movePosition(cursor.Start)
            self.log_view.setTextCursor(cursor)
            self.log_view.find(text)

    def _log_search_prev(self):
        text = self.log_search_edit.text()
        if not text:
            return
        if not self.log_view.find(text, QTextDocument.FindBackward):
            # Wrap around
            cursor = self.log_view.textCursor()
            cursor.movePosition(cursor.End)
            self.log_view.setTextCursor(cursor)
            self.log_view.find(text, QTextDocument.FindBackward)

    # ── System tray ──────────────────────────────────────────────────────

    @staticmethod
    def _make_tray_icon(color):
        """Create a simple colored circle icon for the tray."""
        pix = QPixmap(32, 32)
        pix.fill(Qt.transparent)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(color))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(2, 2, 28, 28)
        painter.end()
        return QIcon(pix)

    def _setup_tray(self):
        """Create system tray icon if available."""
        self._tray = None
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return

        self._tray = QSystemTrayIcon(self._make_tray_icon("#e74c3c"), self)
        self._tray.setToolTip("LlamaHerder")
        self._tray.activated.connect(self._tray_activated)

        menu = QMenu()
        show_action = menu.addAction("Show/Hide")
        show_action.triggered.connect(self._toggle_visibility)
        menu.addSeparator()
        exit_action = menu.addAction("Exit")
        exit_action.triggered.connect(self._tray_exit)
        self._tray.setContextMenu(menu)
        self._tray.show()

    def _tray_activated(self, reason):
        if reason == QSystemTrayIcon.DoubleClick:
            self._toggle_visibility()

    def _toggle_visibility(self):
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.activateWindow()

    def _tray_exit(self):
        self._tray_exiting = True
        self.close()

    def _update_tray_icon(self, server_running):
        if self._tray:
            color = "#2ecc71" if server_running else "#e74c3c"
            self._tray.setIcon(self._make_tray_icon(color))

    # ── Worker table helpers ─────────────────────────────────────────────

    def _resize_worker_table(self, expanded=False):
        # +1 for the summary row
        n_rows = len(self.workers) + 1
        h = 36 + n_rows * 32
        self.table.setMinimumHeight(h)
        if expanded:
            self.table.setMaximumHeight(16777215)  # QWIDGETSIZE_MAX
        else:
            self.table.setMaximumHeight(h)

    def _populate_worker_table(self):
        n_workers = len(self.workers)
        self.table.setRowCount(n_workers + 1)  # +1 for summary row
        for row, (host, port, name) in enumerate(self.workers):
            self.table.setItem(row, 0, QTableWidgetItem(name))
            port_text = str(port) if host != "local" else "--"
            self.table.setItem(row, 1, self._centered(port_text))
            self.table.setItem(row, 2, self._status_dot(host == "local"))
            latency_text = "0 ms" if host == "local" else "--"
            self.table.setItem(row, 3, self._centered(latency_text))
            self.table.setItem(row, 4, self._centered("--"))
            self.table.setItem(row, 5, self._centered("--"))
            self.table.setItem(row, 6, self._centered("--"))
        # Summary row
        self._update_vram_summary_row()
        self._resize_worker_table()

    def _update_vram_summary_row(self):
        """Write the Total VRAM summary into the last row of the worker table."""
        row = self.table.rowCount() - 1
        if row < 0:
            return
        total_free = 0
        total_cap = 0
        for host, _port, _name in self.workers:
            data = self._agent_data.get(host, {})
            free = data.get("vram_free_mb")
            total = data.get("vram_total_mb")
            if free is not None and total is not None:
                total_free += free
                total_cap += total

        label_item = QTableWidgetItem("Total VRAM")
        label_item.setFont(QFont("sans-serif", 9, QFont.Bold))
        label_item.setForeground(QColor("#aaa"))
        self.table.setItem(row, 0, label_item)

        if total_cap > 0:
            vram_item = self._vram_item(total_free, total_cap)
        else:
            vram_item = self._centered("--")
            vram_item.setForeground(QColor("#aaa"))
        self.table.setItem(row, 4, vram_item)

        # Clear other cells in summary row
        for col in (1, 2, 3, 5, 6):
            self.table.setItem(row, col, QTableWidgetItem(""))

    # ── Options dialog ───────────────────────────────────────────────────

    def _open_options(self):
        old_theme = self._config.get("theme", "dark")
        dlg = OptionsDialog(self._config, self)
        if dlg.exec_() == QDialog.Accepted:
            self._config = dlg.config
            self._remotes = self._config.get("workers", [])
            _save_config(self._config)

            self._rebuild_workers()
            self._populate_worker_table()

            # Bump generation so stale thread results are ignored
            self._worker_generation += 1

            # Restart pinger & agent poller with new worker list / port
            self.pinger.stop()
            self._start_pinger()
            self.agent_poller.stop()
            self._start_agent_poller()

            # Re-discover server path
            self._llama_server_path, self._llama_server_found = _discover_llama_server(
                self._config.get("llama_server", "")
            )

            # Re-scan models from new directory
            self._scan_models()

            # Apply theme if changed
            new_theme = self._config.get("theme", "dark")
            if new_theme != old_theme:
                _apply_theme(QApplication.instance(), new_theme)

            # Warn about running server
            if self.server_proc:
                QMessageBox.information(
                    self, "Server Running",
                    "Some changes will take effect after restarting the server.",
                )

    def _open_sampler(self):
        # Gather current model list for associations
        model_list = []
        for i in range(self.model_combo.count()):
            txt = self.model_combo.itemText(i)
            if txt and txt != "No models found":
                model_list.append(txt)

        dlg = SamplerDialog(self._config, model_list=model_list, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self._config = dlg.config
            _save_config(self._config)

    def _open_help(self):
        # Keep a reference so it isn't garbage-collected
        self._help_window = HelpWindow(self)
        self._help_window.show()

    # ── Workers menu actions ─────────────────────────────────────────────

    def _open_generate_worker(self):
        if not self._config.get("workers"):
            QMessageBox.warning(
                self, "No Workers",
                "Add at least one worker in File > Options first.",
            )
            return
        dlg = GenerateWorkerDialog(self._config, self)
        dlg.exec_()

    def _open_deploy_ssh(self):
        if not self._config.get("workers"):
            QMessageBox.warning(
                self, "No Workers",
                "Add at least one worker in File > Options first.",
            )
            return
        dlg = DeploySSHDialog(self._config, self)
        dlg.exec_()

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _make_label(text):
        lbl = QLabel(text)
        lbl.setFont(QFont("sans-serif", 11, QFont.Bold))
        return lbl

    @staticmethod
    def _centered(text):
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        return item

    @staticmethod
    def _status_dot(alive):
        item = QTableWidgetItem("●")
        item.setTextAlignment(Qt.AlignCenter)
        item.setForeground(QColor("#2ecc71") if alive else QColor("#e74c3c"))
        return item

    def _vram_item(self, free_mb, total_mb):
        if free_mb is None or total_mb is None:
            return self._centered("N/A")
        if total_mb >= 1024:
            text = f"{free_mb / 1024:.1f} / {total_mb / 1024:.1f} GB"
        else:
            text = f"{free_mb} / {total_mb} MB"
        item = self._centered(text)
        ratio = free_mb / total_mb if total_mb else 0
        if ratio < 0.15:
            item.setForeground(QColor("#e74c3c"))
        elif ratio < 0.4:
            item.setForeground(QColor("#f39c12"))
        else:
            item.setForeground(QColor("#2ecc71"))
        return item

    def _battery_item(self, pct):
        if pct is None:
            return self._centered("N/A")
        item = self._centered(f"{pct}%")
        if pct <= 20:
            item.setForeground(QColor("#e74c3c"))
        elif pct <= 50:
            item.setForeground(QColor("#f39c12"))
        else:
            item.setForeground(QColor("#2ecc71"))
        return item

    def _power_item(self, source):
        if source is None:
            return self._centered("N/A")
        display = {
            "AC": "AC",
            "Charging": "AC (charging)",
            "Full": "AC (full)",
            "Not charging": "AC (held)",
            "Discharging": "\U0001F50B",
        }
        return self._centered(display.get(source, "N/A"))

    def _scan_models(self):
        current = self.model_combo.currentText()
        self.model_combo.clear()
        model_dir = os.path.expanduser(self._config.get("model_dir", "~/models"))
        if not os.path.isdir(model_dir):
            self.model_combo.addItem("No models found")
            self.model_combo.setEnabled(False)
            return
        gguf_files = sorted(
            f for f in os.listdir(model_dir) if f.endswith(".gguf")
        )
        if gguf_files:
            self.model_combo.setEnabled(True)
            self.model_combo.addItems(gguf_files)
            idx = self.model_combo.findText(current)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        else:
            self.model_combo.addItem("No models found")
            self.model_combo.setEnabled(False)

    # ── Context size ──────────────────────────────────────────────────────

    def _model_file_size_gb(self):
        """Return selected model's file size in GB, or None."""
        name = self.model_combo.currentText()
        if not name or name == "No models found":
            return None
        model_dir = os.path.expanduser(self._config.get("model_dir", "~/models"))
        path = os.path.join(model_dir, name)
        try:
            return os.path.getsize(path) / (1024 ** 3)
        except OSError:
            return None

    def _cluster_free_gb(self):
        """Return total cluster free VRAM in GB."""
        total = 0
        for host, _port, _name in self.workers:
            data = self._agent_data.get(host, {})
            free = data.get("vram_free_mb")
            if free is not None:
                total += free
        return total / 1024

    @staticmethod
    def _estimate_kv_gb(model_file_gb, ctx_tokens, n_slots=1):
        """Estimate KV cache VRAM in GB for a given context size.

        Each parallel slot gets its own KV cache, so total = per-slot * n_slots.
        """
        n_params_b = model_file_gb * 1.3
        return n_params_b * ctx_tokens / 200_000 * n_slots

    def _update_context_options(self):
        current_ctx = self.ctx_combo.currentData()
        self.ctx_combo.clear()

        model_gb = self._model_file_size_gb()
        free_gb = self._cluster_free_gb()
        n_slots = self._config.get("parallel_slots", 1)

        if model_gb is None or free_gb == 0:
            self.ctx_combo.addItem("2K", 2048)
            return

        headroom = free_gb - model_gb
        if headroom < 0:
            headroom = 0

        CTX_OPTIONS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
        best_comfortable = 0
        best_tight = -1
        for ctx in CTX_OPTIONS:
            kv = self._estimate_kv_gb(model_gb, ctx, n_slots)
            if kv <= headroom * 0.95:
                tag = "comfortable"
            elif kv <= headroom:
                tag = "tight"
            else:
                tag = "won't fit"

            ctx_k = f"{ctx // 1024}K" if ctx >= 1024 else str(ctx)
            self.ctx_combo.addItem(ctx_k, ctx)

            idx = self.ctx_combo.count() - 1
            if tag == "comfortable":
                best_comfortable = idx
            elif tag == "tight" and best_tight < 0:
                best_tight = idx

        # Prefer largest comfortable; fall back to first tight fit
        best_idx = best_comfortable

        # Restore previous selection or pick best fit
        if current_ctx:
            for i in range(self.ctx_combo.count()):
                if self.ctx_combo.itemData(i) == current_ctx:
                    self.ctx_combo.setCurrentIndex(i)
                    return
        self.ctx_combo.setCurrentIndex(best_idx)

    def _update_overhead(self):
        """Update the Estimated Overhead label based on model + context selection."""
        model_gb = self._model_file_size_gb()
        free_gb = self._cluster_free_gb()
        n_slots = self._config.get("parallel_slots", 1)
        ctx = self.ctx_combo.currentData()

        if model_gb is None or free_gb == 0 or ctx is None:
            self.overhead_label.setText("")
            return

        kv_gb = self._estimate_kv_gb(model_gb, ctx, n_slots)
        total_need = model_gb + kv_gb

        # Color based on how well it fits
        if total_need <= free_gb * 0.85:
            color = "#2ecc71"  # green — comfortable
            suffix = ""
        elif total_need <= free_gb * 0.95:
            color = "#f1c40f"  # yellow — snug
            suffix = ""
        elif total_need <= free_gb:
            color = "#f39c12"  # orange — tight
            suffix = ""
        else:
            color = "#e74c3c"  # red — won't fit
            suffix = "  WON'T FIT"

        text = (
            f"Estimated Overhead: {total_need:.1f} GB"
            f"  [{model_gb:.1f} GB model + {kv_gb:.1f} GB context]"
            f"{suffix}"
        )
        self.overhead_label.setText(text)
        self.overhead_label.setStyleSheet(f"color: {color};")

    # ── Cluster summary ──────────────────────────────────────────────────

    def _update_cluster(self):
        # Update the summary row in the worker table
        self._update_vram_summary_row()

        total_free = 0
        total_cap = 0
        for host, _port, _name in self.workers:
            data = self._agent_data.get(host, {})
            free = data.get("vram_free_mb")
            total = data.get("vram_total_mb")
            if free is not None and total is not None:
                total_free += free
                total_cap += total

        if total_cap == 0:
            self.cluster_rec_label.setText("")
            return

        free_gb = total_free / 1024

        # Best fit = largest model that leaves ~15-25% headroom for KV cache
        best = None
        for label, need_gb in reversed(MODEL_SIZES):
            ratio = need_gb / free_gb if free_gb else 999
            if 0.70 <= ratio <= 0.90:
                best = (label, need_gb)
                break
        if not best:
            for label, need_gb in reversed(MODEL_SIZES):
                if need_gb <= free_gb * 0.95:
                    best = (label, need_gb)
                    break

        if best:
            label, need_gb = best
            headroom = free_gb - need_gb
            self.cluster_rec_label.setText(
                f"Recommendation:  {label}  "
                f"[{need_gb:.0f} GB, leaves {headroom:.1f} GB for context]"
            )
        else:
            self.cluster_rec_label.setText("")

        # Refresh context options when VRAM changes (skip while server is running)
        if not self.server_proc:
            new_free = round(free_gb, 1)
            if not hasattr(self, "_last_ctx_free") or self._last_ctx_free != new_free:
                self._last_ctx_free = new_free
                self._update_context_options()

        self._update_overhead()

    # ── Background threads ───────────────────────────────────────────────

    def _start_pinger(self):
        self.pinger = WorkerPinger(
            list(self.workers),
            self._config.get("agent_port", 50053),
            self._worker_generation,
            self,
        )
        self.pinger.results.connect(self._update_pings)
        self.pinger.start()

    def _start_agent_poller(self):
        self.agent_poller = WorkerAgentPoller(
            list(self.workers),
            self._config.get("agent_port", 50053),
            self._worker_generation,
            self,
        )
        self.agent_poller.results.connect(self._update_agent_info)
        self.agent_poller.start()

    def _update_pings(self, statuses, generation):
        if generation != self._worker_generation:
            return
        for row, (host, port, name, alive, latency) in enumerate(statuses):
            self.table.setItem(row, 2, self._status_dot(alive))
            if host == "local":
                self.table.setItem(row, 3, self._centered("0 ms"))
            else:
                self.table.setItem(
                    row, 3, self._centered(f"{latency} ms" if alive else "--")
                )

    def _update_agent_info(self, info, generation):
        if generation != self._worker_generation:
            return
        self._agent_data = info
        for row, (host, _port, _name) in enumerate(self.workers):
            data = info.get(host, {})
            vram_free = data.get("vram_free_mb")
            vram_total = data.get("vram_total_mb")
            self.table.setItem(row, 4, self._vram_item(vram_free, vram_total))
            self.table.setItem(row, 5, self._battery_item(data.get("battery_pct")))
            self.table.setItem(row, 6, self._power_item(data.get("power_source")))
        self._update_cluster()

    # ── Launch / Stop ────────────────────────────────────────────────────

    def _healthy_workers(self):
        """Return list of (ip:port, display_name) for reachable remote workers."""
        healthy = []
        for row, (host, port, name) in enumerate(self.workers):
            if host == "local":
                continue
            dot = self.table.item(row, 2)
            if dot and dot.foreground().color() == QColor("#2ecc71"):
                healthy.append((f"{host}:{port}", name))
        return healthy

    def _on_launch(self):
        model_name = self.model_combo.currentText()
        if not model_name or model_name == "No models found":
            self.status_label.setText("Select a model first.")
            return

        # If server is already running, stop it first (switch model)
        if self.server_proc:
            self.status_label.setText("Switching model...")
            self._kill_server()

        model_dir = os.path.expanduser(self._config.get("model_dir", "~/models"))
        model_path = os.path.join(model_dir, model_name)
        healthy = self._healthy_workers()
        rpc_addrs = [addr for addr, _name in healthy]
        rpc_names = [name for _addr, name in healthy]

        server_port = self._config.get("server_port", 8080)
        parallel_slots = self._config.get("parallel_slots", 1)
        ctx_size = self.ctx_combo.currentData() or 2048
        cmd = [
            self._llama_server_path,
            "-m", model_path,
            "-c", str(ctx_size),
            "-ngl", "999",         # fully offload to GPU
            "--no-warmup",         # skip warmup for faster start
            "--slots",             # enable /slots endpoint for inference monitor
            "--host", "0.0.0.0",   # allow remote connections
            "--port", str(server_port),
        ]
        if parallel_slots > 1:
            cmd += ["-np", str(parallel_slots)]
        if rpc_addrs:
            cmd += ["--rpc", ",".join(rpc_addrs)]

        # ── Sampler parameters ──
        # Check for model-specific profile, fall back to active profile
        model_map = self._config.get("model_profile_map", {})
        active_profile = model_map.get(
            model_name,
            self._config.get("active_sampler_profile", "Default"),
        )
        profiles = self._config.get("sampler_profiles", {})
        sampler = profiles.get(active_profile, _DEFAULT_SAMPLER_PROFILE)

        for key, flag in _SAMPLER_FLAGS.items():
            val = sampler.get(key)
            if val is None:
                continue
            if isinstance(val, bool):
                if val:
                    cmd.append(flag)
            elif isinstance(val, str):
                if val:  # skip empty strings
                    cmd += [flag, val]
            else:
                cmd += [flag, str(val)]

        # Stop strings (one per line → multiple --stop flags)
        stop_text = sampler.get("stop", "")
        if stop_text:
            for line in stop_text.splitlines():
                line = line.strip()
                if line:
                    cmd += ["--stop", line]

        # Grammar (passed inline)
        grammar = sampler.get("grammar", "")
        if grammar.strip():
            cmd += ["--grammar", grammar.strip()]

        # JSON schema
        json_schema = sampler.get("json_schema", "")
        if json_schema.strip():
            cmd += ["--json-schema", json_schema.strip()]

        try:
            self.server_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
            )
        except FileNotFoundError:
            self.status_label.setText(
                f"Error: llama-server not found at {self._llama_server_path}"
            )
            return

        # Start log reader threads (stdout + stderr)
        self._stop_log_reader()
        self.log_view.clear()
        self.log_view.append(f"$ {' '.join(cmd)}\n")
        self.log_reader = LogReader(self.server_proc.stderr, self)
        self.log_reader.line_ready.connect(self._append_log)
        self.log_reader.start()
        self.log_reader_out = LogReader(self.server_proc.stdout, self)
        self.log_reader_out.line_ready.connect(self._append_log)
        self.log_reader_out.start()

        rpc_note = (
            f"  (RPC: {', '.join(rpc_names)})" if rpc_names else "  (no RPC workers)"
        )
        self.status_label.setText(
            f"Server running — http://localhost:{server_port}{rpc_note}"
        )
        self.launch_btn.setText("Switch Model")
        self.stop_btn.setEnabled(True)
        self.browser_btn.setVisible(True)

        # Let worker table expand while server is running
        self._resize_worker_table(expanded=True)
        self.overhead_label.setVisible(False)
        self.cluster_rec_label.setVisible(False)

        # Expand monitor panel
        self._connected_workers = list(zip(rpc_addrs, rpc_names))
        self.monitor_frame.setVisible(True)
        self.server_status_label.setText("Waiting for server...")
        self.server_status_label.setStyleSheet("color: #f39c12; font-weight: bold;")

        self._stop_monitor()
        base_url = f"http://localhost:{server_port}"
        self.server_monitor = ServerMonitor(base_url=base_url, parent=self)
        self.server_monitor.slots_updated.connect(self._update_slots)
        self.server_monitor.server_health.connect(self._update_health)
        self.server_monitor.start()

        if not hasattr(self, "_poll_timer"):
            self._poll_timer = QTimer(self)
            self._poll_timer.timeout.connect(self._check_server)
        self._poll_timer.start(1000)

        self._update_tray_icon(True)

    _LOG_NOISE = ("all slots are idle", "GET /slots", "GET /health")

    def _append_log(self, text):
        if any(n in text for n in self._LOG_NOISE):
            return
        self.log_view.append(text)
        # Auto-scroll to bottom
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _stop_log_reader(self):
        if self.log_reader:
            self.log_reader.stop()
            self.log_reader = None
        if hasattr(self, "log_reader_out") and self.log_reader_out:
            self.log_reader_out.stop()
            self.log_reader_out = None

    def _kill_server(self):
        """Kill server process without touching UI state."""
        self._stop_monitor()
        self._stop_log_reader()
        if self.server_proc:
            try:
                pgid = os.getpgid(self.server_proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                try:
                    self.server_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                        self.server_proc.wait(timeout=3)
                    except (ProcessLookupError, OSError):
                        pass
            except ProcessLookupError:
                pass
            self.server_proc = None
        if hasattr(self, "_poll_timer"):
            self._poll_timer.stop()
        self._update_tray_icon(False)

    def _check_server(self):
        if self.server_proc and self.server_proc.poll() is not None:
            self._poll_timer.stop()
            rc = self.server_proc.returncode
            self.status_label.setText(f"Server exited (code {rc})")
            self.server_proc = None
            self.launch_btn.setText("Launch")
            self.stop_btn.setEnabled(False)
            self.browser_btn.setVisible(False)
            self._resize_worker_table(expanded=False)
            self.overhead_label.setVisible(True)
            self.cluster_rec_label.setVisible(True)
            self._stop_monitor()
            self._update_tray_icon(False)

    def _update_health(self, health):
        if health == "ok":
            self.server_status_label.setText("Server responding")
            self.server_status_label.setStyleSheet(
                "color: #2ecc71; font-weight: bold;"
            )
        else:
            self.server_status_label.setText("Server loading model...")
            self.server_status_label.setStyleSheet(
                "color: #f39c12; font-weight: bold;"
            )

    def _update_slots(self, slots):
        self.slots_table.setRowCount(len(slots))
        active = 0
        for row, slot in enumerate(slots):
            sid = str(slot.get("id", row))
            is_busy = slot.get("is_processing", False) or slot.get("state", 0) != 0
            if is_busy:
                active += 1

            # next_token may be a list (new API) or dict (old API)
            next_tok_raw = slot.get("next_token", {})
            if isinstance(next_tok_raw, list):
                next_tok = next_tok_raw[0] if next_tok_raw else {}
            else:
                next_tok = next_tok_raw

            n_decoded = next_tok.get("n_decoded", 0)
            n_remain = next_tok.get("n_remain", 0)

            # n_predicted (old API) or n_decoded (new API)
            n_pred = slot.get("n_predicted", 0) or n_decoded
            n_ctx = slot.get("n_ctx", 0)

            # Prompt — may be missing in new API
            prompt = slot.get("prompt", "")
            if isinstance(prompt, list):
                prompt = str(prompt[0]) if prompt else ""
            if not prompt:
                params = slot.get("params", {})
                max_tok = params.get("max_tokens", 0)
                if max_tok and n_decoded:
                    prompt = f"[{n_decoded}/{max_tok} tokens]"
            if isinstance(prompt, str) and len(prompt) > 80:
                prompt = prompt[:80] + "..."

            # Speed calc — try t_token_generation, fall back to timings
            tg_ms = slot.get("t_token_generation", 0)
            if tg_ms and tg_ms > 0 and n_pred:
                speed = f"{n_pred / (tg_ms / 1000):.1f} t/s"
            else:
                speed = "--"

            if is_busy:
                state_text = "Processing"
                state_color = "#2ecc71"
            elif n_pred > 0:
                state_text = "Done"
                state_color = "#3498db"
            else:
                state_text = "Idle"
                state_color = "#888"

            state_item = self._centered(state_text)
            state_item.setForeground(QColor(state_color))

            if is_busy:
                ctx_text = f"{n_decoded}+{n_remain}"
            elif n_pred > 0:
                ctx_text = f"{n_pred}"
            else:
                ctx_text = f"0/{n_ctx}" if n_ctx else "--"

            self.slots_table.setItem(row, 0, self._centered(sid))
            self.slots_table.setItem(row, 1, state_item)
            self.slots_table.setItem(row, 2, QTableWidgetItem(prompt or "--"))
            self.slots_table.setItem(row, 3, self._centered(ctx_text))
            self.slots_table.setItem(row, 4, self._centered(speed))

        if self._connected_workers:
            tags = []
            for _addr, name in self._connected_workers:
                marker = "[active]" if active > 0 else "[idle]"
                tags.append(f"{name} {marker}")
            self.worker_load_label.setText(
                f"Slots active: {active}/{len(slots)}   |   "
                + "   ".join(tags)
            )
        else:
            self.worker_load_label.setText(
                f"Local-only — slots active: {active}/{len(slots)}"
            )

    def _open_browser(self):
        import webbrowser
        port = self._config.get("server_port", 8080)
        webbrowser.open(f"http://localhost:{port}")

    def _on_unload(self):
        """Kill all llama-server processes system-wide to free VRAM."""
        # First, stop our own managed server if running
        if self.server_proc:
            self._on_stop()

        # Then kill any remaining llama-server processes
        killed = 0
        try:
            result = subprocess.run(
                ["pgrep", "-f", "llama-server"],
                capture_output=True, text=True,
            )
            pids = result.stdout.strip().splitlines()
            for pid in pids:
                pid = pid.strip()
                if not pid:
                    continue
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    killed += 1
                except (ProcessLookupError, PermissionError):
                    pass
        except FileNotFoundError:
            # pgrep not available, try killall
            subprocess.run(
                ["killall", "-9", "llama-server"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            killed = -1  # unknown count

        if killed > 0:
            self.status_label.setText(f"Killed {killed} llama-server process(es).")
        elif killed == 0:
            self.status_label.setText("No llama-server processes found.")
        else:
            self.status_label.setText("Sent kill signal to llama-server.")

    def _on_stop(self):
        self._kill_server()
        self.monitor_frame.setVisible(False)
        self.status_label.setText("Server stopped.")
        self.launch_btn.setText("Launch")
        self.stop_btn.setEnabled(False)
        self.browser_btn.setVisible(False)
        self._resize_worker_table(expanded=False)
        self.overhead_label.setVisible(True)
        self.cluster_rec_label.setVisible(True)

    def _stop_monitor(self):
        if self.server_monitor:
            self.server_monitor.stop()
            self.server_monitor = None

    # ── Cleanup ──────────────────────────────────────────────────────────

    def closeEvent(self, event):
        # Minimize to tray instead of quitting (unless Exit was clicked)
        if self._tray and not self._tray_exiting:
            event.ignore()
            self.hide()
            if not self._tray_notified:
                self._tray.showMessage(
                    "LlamaHerder",
                    "Minimized to system tray. Use tray icon to restore or exit.",
                    QSystemTrayIcon.Information,
                    2000,
                )
                self._tray_notified = True
            return

        self._on_stop()
        self._stop_log_reader()
        self.pinger.stop()
        self.agent_poller.stop()
        if self._tray:
            self._tray.hide()
        event.accept()


def main():
    app = QApplication(sys.argv)
    config = _load_config()
    _apply_theme(app, config.get("theme", "dark"))
    window = LauncherWindow(config)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
