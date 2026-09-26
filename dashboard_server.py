"""
dashboard_server.py — Interactive workspace dashboard backend.

Serves the dashboard HTML and provides API endpoints to launch, stop,
and monitor tools from the browser.

    cd <workspace_dir>
    .venv\\Scripts\\python dashboard_server.py

    Then open http://localhost:9000
"""

import asyncio
import io
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import httpx
import platform
import shutil

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    _HAS_PSUTIL = False

WORKSPACE = Path(__file__).parent.resolve()

# ── Tool Registry ──────────────────────────────────────────────────
# Only these commands can be executed. No arbitrary input accepted.
TOOLS = {
    # AI Systems
    "artemis": {
        "name": "ARTEMIS (Sage)",
        "cmd": [sys.executable, "launch.py"],
        "cwd": "toolbox/artemis",
    },
    "ears": {
        "name": "ARTEMIS Ears",
        "cmd": [sys.executable, "listen.py"],
        "cwd": "toolbox/ears",
    },
    "csm": {
        "name": "CSM Voice",
        "cmd": [sys.executable, "run_csm.py"],
        "cwd": "toolbox/csm",
    },
    "neural-sim": {
        "name": "Neural Simulation",
        "cmd": ["wsl", "python3", "launch.py"],
        "cwd": "toolbox/neural-sim",
        "url": "http://localhost:8765",
    },
    # Built Tools
    "artemis-server": {
        "name": "Artemis Server",
        "cmd": [sys.executable, "artemis-server.pyw"],
        "cwd": "toolbox/artemis-server",
    },
    "comms": {
        "name": "Comms",
        "cmd": [sys.executable, "server.py"],
        "cwd": "toolbox/comms",
    },
    "converter-gui": {
        "name": "Converter GUI",
        "cmd": [sys.executable, "converter.pyw"],
        "cwd": "toolbox/converter",
    },
    "gateway": {
        "name": "Gateway",
        "cmd": [sys.executable, "gateway.py"],
        "cwd": "toolbox/gateway",
    },
    "mirror": {
        "name": "Mirror",
        "cmd": [sys.executable, "server.py"],
        "cwd": "toolbox/mirror",
    },
    # Analysis
    "ghidra": {
        "name": "Ghidra",
        "cmd": ["cmd", "/c", "ghidraRun.bat"],
        "cwd": "ghidra_12.0.4_PUBLIC",
    },
    # Scripts
    "resource-guardian": {
        "name": "Resource Guardian",
        "cmd": ["powershell", "-ExecutionPolicy", "Bypass", "-File", "resource_guardian.ps1"],
        "cwd": "scripts",
    },
    "quick-capture": {
        "name": "Quick Capture",
        "cmd": ["powershell", "-ExecutionPolicy", "Bypass", "-File", "quick_capture.ps1"],
        "cwd": "scripts",
    },
    # Playbooks
    "pb-guardian": {
        "name": "Guardian Start",
        "cmd": [sys.executable, "playbook.py", "replay", "guardian-start"],
        "cwd": "scripts",
    },
    "pb-neural-sim": {
        "name": "Neural Sim Launch",
        "cmd": [sys.executable, "playbook.py", "replay", "neural-sim-launch"],
        "cwd": "scripts",
    },
    "pb-wsl-dep": {
        "name": "WSL Dep Check",
        "cmd": [sys.executable, "playbook.py", "replay", "wsl-dep-check"],
        "cwd": "scripts",
    },
    "pb-wsl-stop": {
        "name": "WSL Emergency Stop",
        "cmd": [sys.executable, "playbook.py", "replay", "wsl-emergency-stop"],
        "cwd": "scripts",
    },
}

CATALOG = [
    {
        "id": "dashboard",
        "name": "Workspace Dashboard",
        "lane": "usable",
        "category": "control",
        "state": "operational",
        "summary": "Primary browser GUI for operating the workspace without relying on VS Code for day-to-day use.",
        "how_to_use": "Start dashboard_server.py, then open http://127.0.0.1:9000 in a browser.",
        "dependencies": [".venv", "FastAPI", "Browser"],
        "verification": "Production-ready; full browser/API validation recorded in session 049.",
        "copy_cmd": ".venv\\Scripts\\python dashboard_server.py",
        "url": "http://127.0.0.1:9000",
        "anchor": "overview",
    },
    {
        "id": "ie-explorer",
        "name": "IE Explorer",
        "lane": "usable",
        "category": "tool",
        "state": "operational",
        "summary": "Index Explorer for scans, trees, lookup, and local Ollama-assisted file understanding.",
        "how_to_use": "Run IE.py in GUI mode or use its CLI subcommands like tree, lookup, db-stats, and ai.",
        "dependencies": ["Python", "Tkinter", "Optional: Ollama on 127.0.0.1:11434"],
        "verification": "Core CLI and scan/index workflows were completed in sessions 052 through 055.",
        "copy_cmd": ".venv\\Scripts\\python IE.py help",
        "anchor": "workspace",
    },
    {
        "id": "voice-chat",
        "name": "Voice Chat Surface",
        "lane": "usable",
        "category": "interaction",
        "state": "operational",
        "summary": "Browser voice interaction pipeline using Whisper STT, Ollama, and Edge TTS directly from the dashboard.",
        "how_to_use": "Open the voice chat panel from the dashboard and talk or type to the selected Ollama model.",
        "dependencies": ["Microphone permission", "Ollama", "faster-whisper", "edge-tts"],
        "verification": "Voice chat was previously integrated directly into the dashboard backend and UI.",
        "action": "voice-chat",
        "anchor": "ai-systems",
    },
    {
        "id": "neural-sim",
        "name": "Neural Simulation Dashboard",
        "lane": "usable",
        "category": "tool",
        "state": "functional",
        "summary": "3D browser dashboard for the neural simulation stack with live training and diagnostics.",
        "how_to_use": "Launch the WSL service, then open the browser dashboard on localhost:8765.",
        "dependencies": ["WSL", "Python3", "CUDA stack in WSL"],
        "verification": "Marked functional in repo state with training, metrics, and self-modify working.",
        "tool_id": "neural-sim",
        "copy_cmd": "wsl python3 toolbox/neural-sim/launch.py",
        "url": "http://localhost:8765",
        "anchor": "ai-systems",
    },
    {
        "id": "ghidra",
        "name": "Ghidra 12.0.4",
        "lane": "usable",
        "category": "analysis",
        "state": "ready",
        "summary": "Full static analysis distribution for binaries, APKs, DEX, ELF, and PE files.",
        "how_to_use": "Launch Ghidra from the dashboard or run ghidraRun.bat directly.",
        "dependencies": ["Bundled Ghidra", "Java runtime support"],
        "verification": "Workspace-local Ghidra distribution is installed and documented as ready to use.",
        "tool_id": "ghidra",
        "copy_cmd": "ghidra_12.0.4_PUBLIC\\ghidraRun.bat",
        "anchor": "analysis",
    },
    {
        "id": "android-analysis",
        "name": "Android Analysis Stack",
        "lane": "usable",
        "category": "analysis",
        "state": "ready",
        "summary": "Workspace-local ADB, Frida server binaries, Objection, Wireshark, Zeek, and nDPI commands.",
        "how_to_use": "Use the copied commands for ADB, Frida, tshark, Zeek, or nDPI from the dashboard.",
        "dependencies": ["adb/ local copy", "Optional Android device", "Optional WSL for Zeek/nDPI"],
        "verification": "Tool paths and usage notes are already wired into the dashboard and repo instructions.",
        "copy_cmd": "adb\\adb.exe devices",
        "anchor": "analysis",
    },
    {
        "id": "playbooks",
        "name": "Playbook Runner",
        "lane": "usable",
        "category": "automation",
        "state": "ready",
        "summary": "Approved automation playbooks for guardian start, WSL dependency checks, and emergency stop live in one place.",
        "how_to_use": "Run a playbook from the dashboard when you want a predefined automation path instead of typing commands manually.",
        "dependencies": ["Python .venv", "scripts/playbook.py", "Relevant target environment"],
        "verification": "Playbook invocations were fixed and validated during earlier dashboard testing.",
        "anchor": "playbooks",
    },
    {
        "id": "workspace-memory",
        "name": "Workspace Memory And Archive",
        "lane": "usable",
        "category": "support",
        "state": "ready",
        "summary": "Project memory, archive workflow, and session history are present as support surfaces and should remain accessible from the GUI.",
        "how_to_use": "Use this area when you need workspace state, archive workflow, or project memory references without hunting through folders manually.",
        "dependencies": ["conversation-archive/", "memory/", "repo state discipline"],
        "verification": "Archive and project-state workflow are already established and documented in this workspace.",
        "anchor": "workspace",
    },
    {
        "id": "links-library",
        "name": "Links Library",
        "lane": "usable",
        "category": "reference",
        "state": "ready",
        "summary": "Curated topic files of reference URLs remain useful as a finished support surface rather than active development work.",
        "how_to_use": "Browse topic files when you need reference material or external resources relevant to a project area.",
        "dependencies": ["Links/ topic files"],
        "verification": "The links library is documented as a standing workspace asset, not an in-progress feature.",
        "anchor": "workspace",
    },
    {
        "id": "reference-library",
        "name": "Reference APK Library",
        "lane": "usable",
        "category": "reference",
        "state": "ready",
        "summary": "The decompiled APK study library is a finished reference surface even though it is not an end-user runtime tool.",
        "how_to_use": "Use it for architectural study and design reference, not as a runnable product.",
        "dependencies": ["toolbox reference APK folders"],
        "verification": "The library is already documented as study-only and structurally present in the workspace.",
        "anchor": "reference-apks",
    },
    {
        "id": "artemis",
        "name": "ARTEMIS Runtime",
        "lane": "workbench",
        "category": "runtime",
        "state": "active-chain",
        "summary": "Primary AI runtime under active integration work. Current focus is replacing the stale 8000 runtime and continuing live hearing integration.",
        "how_to_use": "Treat this as active development. Relaunch the fixed runtime, verify /think on qwen3:0.6b, then continue capability wiring.",
        "dependencies": ["Ollama on 127.0.0.1:11434", "qwen3:0.6b", "Port 8000 availability"],
        "verification": "Validated on a clean alternate runtime previously; primary 8000 instance still needs replacement.",
        "tool_id": "artemis",
        "copy_cmd": "cd toolbox/artemis ; python launch.py",
        "anchor": "ai-systems",
        "promotion_ready": True,
        "promotion_note": "Promote after the fixed runtime replaces the stale 8000 instance and /think is re-verified.",
    },
    {
        "id": "ears",
        "name": "ARTEMIS Ears",
        "lane": "workbench",
        "category": "runtime",
        "state": "integration",
        "summary": "Audio perception stack that must be wired into the live ARTEMIS runtime rather than operated as an isolated side tool.",
        "how_to_use": "Launch only for validation; the real goal is invoking its hearing path from ARTEMIS.",
        "dependencies": ["Audio input", "faster-whisper", "ARTEMIS runtime integration"],
        "verification": "Dependency preflight was hardened; hearing path wiring remains active work.",
        "tool_id": "ears",
        "copy_cmd": "cd toolbox/ears ; python listen.py",
        "anchor": "ai-systems",
        "promotion_ready": True,
        "promotion_note": "Promote only after ARTEMIS can invoke the hearing path directly instead of running it as a side tool.",
    },
    {
        "id": "csm",
        "name": "CSM Voice",
        "lane": "workbench",
        "category": "runtime",
        "state": "blocked",
        "summary": "Speech synthesis stack with the correct long-term direction in WSL, but not ready on the Windows .venv runtime.",
        "how_to_use": "Do not treat as operational yet. Resolve the WSL Python 3.12 virtualenv blocker before further runtime work.",
        "dependencies": ["WSL Ubuntu 24.04", "CUDA Torch stack", "python3.12-venv or ensurepip"],
        "verification": "Current blocker is explicitly documented in repo state and checkpoint handoff.",
        "tool_id": "csm",
        "copy_cmd": "cd toolbox/csm ; python run_csm.py",
        "anchor": "ai-systems",
        "promotion_ready": False,
        "promotion_note": "Blocked until the WSL Python 3.12 virtual environment issue is resolved.",
    },
    {
        "id": "artemis-server",
        "name": "Artemis Server",
        "lane": "workbench",
        "category": "tool",
        "state": "built",
        "summary": "Portable file server and tunnel tooling that exists, but should remain in the workbench until re-verified from this dashboard-first operating model.",
        "how_to_use": "Launch when you need the server path; keep it in workbench status until it is re-verified through the unified GUI flow.",
        "dependencies": ["Python .venv", "Go build artifacts where applicable"],
        "verification": "Existing implementation is present in toolbox/artemis-server.",
        "tool_id": "artemis-server",
        "copy_cmd": "python toolbox/artemis-server/artemis-server.pyw",
        "anchor": "built-tools",
        "promotion_ready": True,
        "promotion_note": "Promote after a full launch and daily-use workflow is re-verified from the dashboard-first path.",
    },
    {
        "id": "comms",
        "name": "Comms",
        "lane": "workbench",
        "category": "tool",
        "state": "built",
        "summary": "Node-to-node communication platform that exists but still belongs in the experimental lane until it is re-verified as an operator-facing surface.",
        "how_to_use": "Launch for development and validation runs, not as a finished daily-use tool yet.",
        "dependencies": ["Python .venv", "Optional AI and voice dependencies"],
        "verification": "Server and AI engine preflights were recently hardened.",
        "tool_id": "comms",
        "copy_cmd": "cd toolbox/comms ; python server.py",
        "anchor": "built-tools",
        "promotion_ready": True,
        "promotion_note": "Promote after the operator-facing startup and communication flow is re-verified from the GUI.",
    },
    {
        "id": "converter-gui",
        "name": "Bat_To_Exe Converter",
        "lane": "workbench",
        "category": "tool",
        "state": "built",
        "summary": "Reimplementation with its own GUI and C stub pipeline. Present and launchable, but not yet promoted to the operator-facing finished lane.",
        "how_to_use": "Launch the converter when needed, but keep it separated from the verified finished surface for now.",
        "dependencies": ["Python .venv", "Stub toolchain for builds"],
        "verification": "Included in the dashboard tool registry and prior dashboard test passes.",
        "tool_id": "converter-gui",
        "copy_cmd": "python toolbox/converter/converter.pyw",
        "anchor": "built-tools",
        "promotion_ready": True,
        "promotion_note": "Likely promotable after one clean end-user conversion pass is re-verified from the dashboard.",
    },
    {
        "id": "gateway",
        "name": "Gateway",
        "lane": "workbench",
        "category": "tool",
        "state": "built",
        "summary": "Dual-node virtual router tooling that exists, but still belongs in the separate workbench lane until its end-user operating flow is finalized.",
        "how_to_use": "Use for controlled runs and validation, not yet as a polished finished surface.",
        "dependencies": ["Python .venv", "WireGuard context"],
        "verification": "No-arg flow was previously fixed and validated in dashboard testing.",
        "tool_id": "gateway",
        "copy_cmd": "cd toolbox/gateway ; python gateway.py",
        "anchor": "built-tools",
        "promotion_ready": True,
        "promotion_note": "Promote after its end-user operating path is re-verified instead of just the developer launch path.",
    },
    {
        "id": "mirror",
        "name": "Mirror",
        "lane": "workbench",
        "category": "tool",
        "state": "built",
        "summary": "ADB screen mirror service. Present and runnable, but kept separate from the finished lane until its user flow is re-verified through the unified GUI.",
        "how_to_use": "Launch for device mirroring sessions when needed.",
        "dependencies": ["Python .venv", "ADB", "Android device"],
        "verification": "Main entrypoint and runtime issues were previously fixed and validated.",
        "tool_id": "mirror",
        "copy_cmd": "cd toolbox/mirror ; python server.py",
        "anchor": "built-tools",
        "promotion_ready": True,
        "promotion_note": "Promote after a device-connected browser viewing session is re-verified through the dashboard.",
    },
    {
        "id": "tethering",
        "name": "Tethering Project",
        "lane": "workbench",
        "category": "research",
        "state": "active",
        "summary": "PdaNet, FoxFi, and Tetrd reverse engineering plus Go and Python rebuilds. Valuable, but still clearly active development.",
        "how_to_use": "Treat as research and build work, not a finished operator-facing product surface yet.",
        "dependencies": ["Go toolchain", "Python", "Reference APK study", "Android validation"],
        "verification": "Recent state notes report pytether built and the Go engine compiling.",
        "copy_cmd": "cd toolbox/tethering/engine ; go build ./...",
        "anchor": "built-tools",
        "promotion_ready": False,
        "promotion_note": "Keep in workbench until the project has a stable operator-facing flow rather than an active research/build loop.",
    },
    {
        "id": "cloud-server",
        "name": "Cloud Server Project",
        "lane": "workbench",
        "category": "research",
        "state": "planning",
        "summary": "Self-hosted phone-based cloud server direction. Important, but still in planning and external validation stages.",
        "how_to_use": "Do not treat as runnable yet. Next step is IPv6 inbound validation from an external probe.",
        "dependencies": ["Phone IPv6 reachability", "Termux path", "Network validation"],
        "verification": "Project state marks this as planning/research only.",
        "anchor": "workspace",
        "promotion_ready": False,
        "promotion_note": "Planning only; not a promotion candidate yet.",
    },
    {
        "id": "scripts-engine",
        "name": "Scripts And Ad-Hoc Analysis",
        "lane": "workbench",
        "category": "support",
        "state": "built",
        "summary": "The scripts area is valuable, but it remains a builder and analyst workbench rather than a polished operator-facing surface.",
        "how_to_use": "Use these when you are doing directed analysis or engineering work, not when you need a simple finished product flow.",
        "dependencies": ["Python", "PowerShell", "Ghidra or packet tooling depending on script"],
        "verification": "Scripts are present and linked throughout the dashboard, but the area is still intentionally tool-oriented.",
        "anchor": "scripts",
        "promotion_ready": False,
        "promotion_note": "Keep separate until a smaller curated operator subset emerges from the scripts area.",
    },
]


def _catalog_with_runtime_state():
    items = []
    for item in CATALOG:
        enriched = dict(item)
        tool_id = item.get("tool_id")
        proc_info = _processes.get(tool_id) if tool_id else None
        running = False
        pid = None
        uptime = None
        if proc_info and hasattr(proc_info["proc"], "returncode") and proc_info["proc"].returncode is None:
            running = True
            pid = proc_info["proc"].pid
            uptime = round(time.time() - proc_info["started"])
        enriched["running"] = running
        enriched["pid"] = pid
        enriched["uptime_s"] = uptime
        if enriched["lane"] == "usable":
            if enriched.get("category") in {"reference", "support"}:
                enriched["use_mode"] = "Reference / support surface"
                enriched["use_boundary"] = "Safe to use as a workspace support area, but not a runnable product flow."
            else:
                enriched["use_mode"] = "Normal operator use"
                enriched["use_boundary"] = "Safe to treat as part of the finished day-to-day operating surface."
        else:
            enriched["use_mode"] = "Development / validation only"
            enriched["use_boundary"] = enriched.get("promotion_note") or "Keep this separated from finished use until it is re-verified and promoted."
        items.append(enriched)
    return items


def _command_available(command: str) -> bool:
    return shutil.which(command) is not None


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _safe_subprocess_ok(cmd: list[str], timeout: float = 3.0) -> bool:
    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
        return completed.returncode == 0
    except Exception:
        return False


def _build_dependency_health():
    checks = []

    ollama_ok = _port_open("127.0.0.1", 11434)
    checks.append({
        "id": "ollama",
        "name": "Ollama",
        "status": "ready" if ollama_ok else "attention",
        "summary": "Required for ARTEMIS, voice chat, and local AI paths.",
        "detail": "Listening on 127.0.0.1:11434." if ollama_ok else "Not reachable on 127.0.0.1:11434.",
    })

    adb_ok = (WORKSPACE / "adb" / "adb.exe").exists()
    checks.append({
        "id": "adb",
        "name": "ADB",
        "status": "ready" if adb_ok else "attention",
        "summary": "Required for Android device access, mirror, Frida push flows, and device shell use.",
        "detail": "Workspace-local adb.exe is present." if adb_ok else "Workspace-local adb.exe is missing.",
    })

    wsl_ok = _command_available("wsl") and _safe_subprocess_ok(["wsl", "--status"])
    checks.append({
        "id": "wsl",
        "name": "WSL",
        "status": "ready" if wsl_ok else "attention",
        "summary": "Required for Neural Sim, CUDA paths, and the intended CSM runtime direction.",
        "detail": "WSL responds normally." if wsl_ok else "WSL is unavailable or not responding.",
    })

    ghidra_ok = (WORKSPACE / "ghidra_12.0.4_PUBLIC" / "ghidraRun.bat").exists()
    checks.append({
        "id": "ghidra",
        "name": "Ghidra",
        "status": "ready" if ghidra_ok else "attention",
        "summary": "Required for static binary and APK analysis.",
        "detail": "ghidraRun.bat is present." if ghidra_ok else "Ghidra launcher is missing.",
    })

    port_8000_in_use = _port_open("127.0.0.1", 8000)
    checks.append({
        "id": "port-8000",
        "name": "Port 8000",
        "status": "attention" if port_8000_in_use else "ready",
        "summary": "Primary ARTEMIS runtime target port.",
        "detail": "Something is already listening on 127.0.0.1:8000." if port_8000_in_use else "Port 8000 is free for the fixed ARTEMIS runtime.",
    })

    ffmpeg_ok = _command_available("ffmpeg")
    checks.append({
        "id": "ffmpeg",
        "name": "ffmpeg",
        "status": "ready" if ffmpeg_ok else "attention",
        "summary": "Used for voice-chat audio conversion when browser audio arrives as WebM/Opus.",
        "detail": "ffmpeg is available on PATH." if ffmpeg_ok else "ffmpeg is not available on PATH.",
    })

    return checks

# ── Process Tracker ────────────────────────────────────────────────
_processes: dict[str, dict] = {}  # tool_id -> {proc, started, output_lines}
MAX_OUTPUT_LINES = 500


# ── Graceful Shutdown ──────────────────────────────────────────────
async def _cleanup_processes():
    """Terminate all tracked child processes on server shutdown."""
    for tid, info in _processes.items():
        proc = info["proc"]
        if hasattr(proc, "terminate") and proc.returncode is None:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            except Exception:
                pass


# ── FastAPI App ────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app):
    yield
    await _cleanup_processes()

app = FastAPI(title="RE Workspace Dashboard", lifespan=lifespan)

# Allow same-host origins (localhost vs 127.0.0.1 mismatch)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index():
    return FileResponse(WORKSPACE / "dashboard.html")


@app.get("/3d")
async def index_3d():
    return FileResponse(WORKSPACE / "dashboard_3d.html")


@app.get("/galaxy_5k.json")
async def galaxy_5k():
    return FileResponse(WORKSPACE / "galaxy_5k.json")


@app.get("/galaxy_25k.json")
async def galaxy_25k():
    return FileResponse(WORKSPACE / "galaxy_25k.json")


@app.get("/galaxy_50k.json")
async def galaxy_50k():
    p = WORKSPACE / "galaxy_50k.json"
    if not p.exists():
        raise HTTPException(404, "galaxy_50k.json not present")
    return FileResponse(p)


@app.get("/assets/{asset_name}")
async def dashboard_asset(asset_name: str):
    """Serve explicit dashboard assets from the workspace root."""
    target = (WORKSPACE / asset_name).resolve()
    if not target.is_relative_to(WORKSPACE):
        raise HTTPException(400, "Path outside workspace")
    if not target.exists() or not target.is_file():
        raise HTTPException(404, f"Asset not found: {asset_name}")
    return FileResponse(target)


@app.get("/api/status")
async def all_status():
    """Return running state of every registered tool."""
    result = {}
    for tid, tool in TOOLS.items():
        proc_info = _processes.get(tid)
        running = False
        pid = None
        if proc_info and proc_info["proc"].returncode is None:
            running = True
            pid = proc_info["proc"].pid
        result[tid] = {
            "name": tool["name"],
            "running": running,
            "pid": pid,
            "url": tool.get("url"),
        }
    return result


class LaunchReq(BaseModel):
    tool_id: str


@app.post("/api/launch")
async def launch_tool(req: LaunchReq):
    """Launch a registered tool."""
    tid = req.tool_id
    if tid not in TOOLS:
        raise HTTPException(400, f"Unknown tool: {tid}")

    # Already running?
    if tid in _processes:
        proc = _processes[tid]["proc"]
        if proc.returncode is None:
            return {"status": "already_running", "pid": proc.pid}

    tool = TOOLS[tid]
    cwd = WORKSPACE / tool["cwd"]

    env = os.environ.copy()
    # Ensure .venv is on PATH for Python tools
    venv_scripts = WORKSPACE / ".venv" / "Scripts"
    if venv_scripts.exists():
        env["PATH"] = str(venv_scripts) + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = str(WORKSPACE / ".venv")

    try:
        proc = await asyncio.create_subprocess_exec(
            *tool["cmd"],
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to start {tid}: {e}")

    info = {"proc": proc, "started": time.time(), "output": []}
    _processes[tid] = info

    # Background task to collect output
    asyncio.create_task(_collect_output(tid, proc, info))

    return {"status": "launched", "pid": proc.pid}


async def _collect_output(tid: str, proc, info: dict):
    """Read stdout lines in background, store in buffer."""
    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            info["output"].append(text)
            if len(info["output"]) > MAX_OUTPUT_LINES:
                info["output"] = info["output"][-MAX_OUTPUT_LINES:]
    except Exception:
        pass


@app.post("/api/stop")
async def stop_tool(req: LaunchReq):
    """Stop a running tool."""
    tid = req.tool_id
    if tid not in _processes:
        raise HTTPException(404, f"Not running: {tid}")

    proc = _processes[tid]["proc"]
    if proc.returncode is not None:
        return {"status": "already_stopped", "code": proc.returncode}

    try:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            proc.kill()
    except Exception as e:
        raise HTTPException(500, f"Failed to stop {tid}: {e}")

    return {"status": "stopped", "code": proc.returncode}


@app.get("/api/output/{tool_id}")
async def get_output(tool_id: str, since: int = 0):
    """Get recent output lines from a tool."""
    if tool_id not in _processes:
        return {"lines": [], "running": False}
    info = _processes[tool_id]
    lines = info["output"][since:]
    running = info["proc"].returncode is None
    return {"lines": lines, "total": len(info["output"]), "running": running}


@app.post("/api/open-vscode")
async def open_in_vscode(req: dict):
    """Open a file in VS Code."""
    rel_path = req.get("path", "")
    # Sanitize: must be under workspace, no .. traversal
    target = (WORKSPACE / rel_path).resolve()
    if not target.is_relative_to(WORKSPACE):
        raise HTTPException(400, "Path outside workspace")
    if not target.exists():
        raise HTTPException(404, f"Not found: {rel_path}")

    await asyncio.create_subprocess_exec(
        "code", "--goto", str(target),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    return {"status": "opened"}


# ── Interactive Terminal ───────────────────────────────────────────
# Runs commands via PowerShell in the workspace directory.
# Each "session" is a named process slot in _processes.

_run_counter = 0
_terminal_cwd = str(WORKSPACE)  # mutable working directory


class RunReq(BaseModel):
    command: str
    session: str = ""  # optional session name; auto-assigned if empty


@app.post("/api/run")
async def run_command(req: RunReq):
    """Run a shell command in the workspace directory."""
    global _run_counter
    cmd = req.command.strip()
    if not cmd:
        raise HTTPException(400, "Empty command")

    # Assign session ID
    sid = req.session or f"run-{_run_counter}"
    _run_counter += 1

    # If session already active and running, kill it first
    if sid in _processes:
        old = _processes[sid]["proc"]
        if old.returncode is None:
            old.terminate()
            try:
                await asyncio.wait_for(old.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                old.kill()

    env = os.environ.copy()
    venv_scripts = WORKSPACE / ".venv" / "Scripts"
    if venv_scripts.exists():
        env["PATH"] = str(venv_scripts) + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = str(WORKSPACE / ".venv")

    # Handle cd/pushd — update tracked cwd
    cd_target = _parse_cd(cmd)
    if cd_target is not None:
        return await _handle_cd(cd_target, sid)

    try:
        proc = await asyncio.create_subprocess_exec(
            "powershell", "-NoProfile", "-Command", cmd,
            cwd=_terminal_cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to run: {e}")

    info = {"proc": proc, "started": time.time(), "output": [], "cmd": cmd}
    _processes[sid] = info
    asyncio.create_task(_collect_output(sid, proc, info))

    return {"status": "running", "session": sid, "pid": proc.pid, "cwd": _terminal_cwd}


def _parse_cd(cmd: str):
    """Return target dir if cmd is a cd/pushd/Set-Location, else None."""
    stripped = cmd.strip()
    for prefix in ("cd ", "cd\\", "pushd ", "Set-Location ", "sl "):
        if stripped.lower().startswith(prefix.lower()):
            return stripped[len(prefix):].strip().strip('"').strip("'")
    if stripped.lower() in ("cd", "cd.", "cd ~"):
        return str(WORKSPACE)
    return None


async def _handle_cd(target: str, sid: str):
    """Change tracked working directory."""
    global _terminal_cwd
    if os.path.isabs(target):
        new_cwd = os.path.realpath(target)
    else:
        new_cwd = os.path.realpath(os.path.join(_terminal_cwd, target))

    if not os.path.isdir(new_cwd):
        return {"status": "error", "session": sid, "error": f"Not a directory: {new_cwd}"}

    _terminal_cwd = new_cwd
    # Put a synthetic output line so the UI shows feedback
    _processes[sid] = {
        "proc": type("FakeProc", (), {"returncode": 0, "pid": 0})(),
        "started": time.time(),
        "output": [f"Changed directory to: {new_cwd}"],
        "cmd": f"cd {target}",
    }
    return {"status": "done", "session": sid, "cwd": _terminal_cwd}


@app.get("/api/cwd")
async def get_cwd():
    return {"cwd": _terminal_cwd}


@app.post("/api/kill")
async def kill_session(req: dict):
    """Kill a running session."""
    sid = req.get("session", "")
    if sid not in _processes:
        raise HTTPException(404, f"No session: {sid}")
    proc = _processes[sid]["proc"]
    if hasattr(proc, "terminate") and proc.returncode is None:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            proc.kill()
    return {"status": "killed"}


# ── Additional REST Endpoints ──────────────────────────────────────

@app.get("/api/tools")
async def list_tools():
    """List all registered tools with config and status."""
    result = {}
    for tid, tool in TOOLS.items():
        proc_info = _processes.get(tid)
        running = False
        pid = None
        uptime = None
        if proc_info:
            if hasattr(proc_info["proc"], "returncode") and proc_info["proc"].returncode is None:
                running = True
                pid = proc_info["proc"].pid
                uptime = round(time.time() - proc_info["started"])
        result[tid] = {
            "name": tool["name"],
            "cwd": tool["cwd"],
            "url": tool.get("url"),
            "running": running,
            "pid": pid,
            "uptime_s": uptime,
        }
    return result


@app.get("/api/catalog")
async def workspace_catalog():
    """Return the operator-facing workspace catalog with live runtime state."""
    items = _catalog_with_runtime_state()
    counts = {
        "usable": sum(1 for item in items if item["lane"] == "usable"),
        "workbench": sum(1 for item in items if item["lane"] == "workbench"),
        "running": sum(1 for item in items if item.get("running")),
    }
    categories = {}
    for item in items:
        key = item.get("category", "other")
        categories[key] = categories.get(key, 0) + 1
    promotion_queue = [
        {
            "id": item["id"],
            "name": item["name"],
            "state": item["state"],
            "promotion_note": item.get("promotion_note", "Needs explicit re-verification before promotion."),
            "anchor": item.get("anchor"),
            "tool_id": item.get("tool_id"),
        }
        for item in items
        if item["lane"] == "workbench" and item.get("promotion_ready")
    ]
    return {"items": items, "counts": counts, "categories": categories, "promotion_queue": promotion_queue}


@app.get("/api/health")
async def dependency_health():
    """Return critical dependency readiness for the operator surface."""
    checks = await asyncio.to_thread(_build_dependency_health)
    counts = {
        "ready": sum(1 for check in checks if check["status"] == "ready"),
        "attention": sum(1 for check in checks if check["status"] != "ready"),
    }
    return {"checks": checks, "counts": counts}


def _verify_catalog_item(item: dict) -> dict:
    """Run lightweight, non-destructive verification checks for a catalog item.

    Returns a dict with overall 'ok' boolean and a list of 'checks'.
    """
    results = []
    overall_ok = True

    # Dependency-based quick checks
    for dep in item.get("dependencies", [])[:]:
        d = dep.lower()
        if "ollama" in d:
            ok = _port_open("127.0.0.1", 11434)
            results.append({"check": "ollama", "ok": ok, "detail": "127.0.0.1:11434 reachable" if ok else "not reachable"})
            overall_ok &= ok
        elif "wsl" in d:
            ok = _command_available("wsl") and _safe_subprocess_ok(["wsl", "--status"])
            results.append({"check": "wsl", "ok": ok, "detail": "WSL responsive" if ok else "not responsive"})
            overall_ok &= ok
        elif "adb" in d:
            ok = (WORKSPACE / "adb" / "adb.exe").exists()
            results.append({"check": "adb", "ok": ok, "detail": "adb.exe present" if ok else "missing adb.exe"})
            overall_ok &= ok
        elif "ghidra" in d:
            ok = (WORKSPACE / "ghidra_12.0.4_PUBLIC" / "ghidraRun.bat").exists()
            results.append({"check": "ghidra", "ok": ok, "detail": "ghidraRun.bat present" if ok else "missing ghidraRun.bat"})
            overall_ok &= ok

    # Tool-specific checks (if the item maps to a registered tool)
    tid = item.get("tool_id")
    if tid and tid in TOOLS:
        tool = TOOLS[tid]
        cwd = WORKSPACE / tool.get("cwd", "")
        cwd_ok = cwd.exists()
        results.append({"check": "cwd_exists", "ok": cwd_ok, "detail": str(cwd)})
        overall_ok &= cwd_ok

        cmd = tool.get("cmd", [])
        if cmd:
            exe = cmd[0]
            # If the command is Python, check the script path
            if exe in (sys.executable, "python", "python3") and len(cmd) > 1:
                script = cwd / cmd[1]
                script_ok = script.exists()
                results.append({"check": "script_exists", "ok": script_ok, "detail": str(script)})
                overall_ok &= script_ok
            else:
                found = shutil.which(exe) is not None
                results.append({"check": "executable_on_path", "ok": found, "detail": exe})
                overall_ok &= found

        # If the tool exposes a URL, try a quick HTTP probe (non-blocking friendly)
        url = tool.get("url")
        if url:
            try:
                r = httpx.get(url, timeout=2.0)
                ok = r.status_code < 400
            except Exception:
                ok = False
            results.append({"check": "tool_url", "ok": ok, "detail": url})
            overall_ok &= ok

    # If the item itself has a URL, probe that too
    item_url = item.get("url")
    if item_url:
        try:
            r = httpx.get(item_url, timeout=2.0)
            ok = r.status_code < 400
        except Exception:
            ok = False
        results.append({"check": "item_url", "ok": ok, "detail": item_url})
        overall_ok &= ok

    return {"ok": overall_ok, "checks": results}


@app.get("/api/verify/{item_id}")
async def verify_item(item_id: str):
    """Run a lightweight verification for a catalog item and return results."""
    found = None
    for item in CATALOG:
        if item.get("id") == item_id:
            found = item
            break
    if not found:
        raise HTTPException(404, f"Catalog item not found: {item_id}")

    # Run verification off the event loop
    result = await asyncio.to_thread(_verify_catalog_item, found)
    return JSONResponse(result)


@app.post("/api/promote/{item_id}")
async def promote_item(item_id: str):
    """Promote a workbench item into the usable lane if verification passes and it's allowed.

    This modifies the in-memory CATALOG only for the running server instance.
    """
    for item in CATALOG:
        if item.get("id") == item_id:
            target = item
            break
    else:
        raise HTTPException(404, f"Catalog item not found: {item_id}")

    if not target.get("promotion_ready"):
        raise HTTPException(400, "Item is not marked promotion-ready")

    # Verify before promoting
    result = await asyncio.to_thread(_verify_catalog_item, target)
    if not result.get("ok"):
        return JSONResponse(status_code=400, content={"detail": "Verification failed", "result": result})

    # Perform in-memory promotion
    target["lane"] = "usable"
    target["promotion_note"] = f"Promoted on {time.strftime('%Y-%m-%d %H:%M:%S')} from dashboard"

    return {"status": "promoted", "id": item_id, "item": target}


@app.post("/api/restart")
async def restart_tool(req: LaunchReq):
    """Stop a tool (if running) and relaunch it."""
    tid = req.tool_id
    if tid not in TOOLS:
        raise HTTPException(400, f"Unknown tool: {tid}")
    if tid in _processes:
        proc = _processes[tid]["proc"]
        if hasattr(proc, "returncode") and proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                proc.kill()
            await asyncio.sleep(0.2)
    return await launch_tool(LaunchReq(tool_id=tid))


@app.get("/api/sysinfo")
async def sysinfo():
    """System resource info."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "workspace": str(WORKSPACE),
    }
    try:
        usage = shutil.disk_usage(WORKSPACE)
        info["disk"] = {
            "total_gb": round(usage.total / (1024**3), 1),
            "used_gb": round(usage.used / (1024**3), 1),
            "free_gb": round(usage.free / (1024**3), 1),
            "percent": round(usage.used / usage.total * 100, 1),
        }
    except Exception:
        pass
    if psutil is not None:
        info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        info["memory"] = {
            "total_gb": round(mem.total / (1024**3), 1),
            "used_gb": round(mem.used / (1024**3), 1),
            "percent": mem.percent,
        }
    info["tracked_processes"] = sum(
        1 for p in _processes.values()
        if hasattr(p["proc"], "returncode") and p["proc"].returncode is None
    )
    return info


# ── WebSocket: Live Output Streaming ──────────────────────────────

@app.websocket("/ws/output/{tool_id}")
async def ws_output(websocket: WebSocket, tool_id: str):
    """Stream tool output in real time via WebSocket."""
    await websocket.accept()
    offset = 0
    try:
        while True:
            if tool_id in _processes:
                info = _processes[tool_id]
                lines = info["output"][offset:]
                if lines:
                    await websocket.send_json({"lines": lines, "total": len(info["output"])})
                    offset = len(info["output"])
                running = info["proc"].returncode is None
                if not running and not lines and offset > 0:
                    await websocket.send_json({"done": True, "code": info["proc"].returncode})
                    break
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ── WebSocket: Persistent Terminal ─────────────────────────────────

_CWD_MARKER = "__PSCWD__"


@app.websocket("/ws/terminal")
async def ws_terminal(websocket: WebSocket):
    """Persistent PowerShell session via WebSocket."""
    await websocket.accept()

    env = os.environ.copy()
    venv_scripts = WORKSPACE / ".venv" / "Scripts"
    if venv_scripts.exists():
        env["PATH"] = str(venv_scripts) + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = str(WORKSPACE / ".venv")

    try:
        proc = await asyncio.create_subprocess_exec(
            "powershell", "-NoProfile", "-NoLogo", "-Command", "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(WORKSPACE),
            env=env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )
    except Exception as e:
        await websocket.send_json({"type": "error", "text": f"Failed to start shell: {e}"})
        await websocket.close()
        return

    assert proc.stdin is not None
    assert proc.stdout is not None

    # Emit initial working directory
    proc.stdin.write(f'Write-Host "{_CWD_MARKER}$((Get-Location).Path)"\n'.encode())
    await proc.stdin.drain()

    async def _read_stdout():
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    try:
                        await websocket.send_json({"type": "exit"})
                    except Exception:
                        pass
                    break
                text = line.decode("utf-8", errors="replace").rstrip("\r\n")
                if text.startswith(_CWD_MARKER):
                    await websocket.send_json({"type": "cwd", "path": text[len(_CWD_MARKER):]})
                else:
                    await websocket.send_json({"type": "output", "text": text})
        except (WebSocketDisconnect, ConnectionError):
            pass
        except Exception:
            pass

    reader_task = asyncio.create_task(_read_stdout())

    try:
        while True:
            msg = await websocket.receive_json()
            cmd_type = msg.get("type", "command")

            if cmd_type == "command":
                cmd = msg.get("cmd", "").strip()
                if not cmd:
                    continue
                if proc.returncode is not None:
                    await websocket.send_json({"type": "error", "text": "Shell exited. Close and reopen the terminal."})
                    break
                full = f'{cmd}\nWrite-Host "{_CWD_MARKER}$((Get-Location).Path)"\n'
                proc.stdin.write(full.encode("utf-8"))
                await proc.stdin.drain()

            elif cmd_type == "kill":
                if proc.returncode is None:
                    try:
                        proc.send_signal(
                            signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGINT
                        )
                    except Exception:
                        pass
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        reader_task.cancel()
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    proc.kill()
                except Exception:
                    pass


# ── Voice Chat ─────────────────────────────────────────────────────
# STT (faster-whisper), LLM (Ollama), TTS (edge-tts) — all server-side.
# Browser sends raw audio via WebSocket, receives audio + transcript back.

WHISPER_MODELS_DIR = WORKSPACE / "models" / "whisper"
_whisper_model = None
_whisper_lock = asyncio.Lock()


def _load_whisper(size: str = "base.en"):
    """Load a faster-whisper model (blocking — run in thread)."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    from faster_whisper import WhisperModel  # type: ignore[import-not-found]
    local = WHISPER_MODELS_DIR / f"faster-whisper-{size}"
    if local.exists():
        _whisper_model = WhisperModel(str(local), device="cuda", compute_type="float16")
    else:
        _whisper_model = WhisperModel(size, device="cuda", compute_type="float16")
    return _whisper_model


def _transcribe_audio(audio_np: np.ndarray) -> str:
    """Transcribe float32 16kHz mono audio to text (blocking)."""
    model = _load_whisper()
    segments, _ = model.transcribe(
        audio_np, beam_size=5, language="en",
        vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500),
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


async def _tts_edge(text: str, voice: str = "en-US-GuyNeural") -> bytes:
    """Generate MP3 audio bytes via edge-tts."""
    import edge_tts
    comm = edge_tts.Communicate(text, voice=voice)
    buf = io.BytesIO()
    async for chunk in comm.stream():
        if chunk["type"] == "audio" and "data" in chunk:
            buf.write(chunk["data"])  # type: ignore[typeddict-item]
    return buf.getvalue()


async def _chat_ollama(messages: list[dict], model: str,
                       host: str = "http://localhost:11434") -> str:
    """Send chat to Ollama, return response text."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            f"{host}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
        )
        r.raise_for_status()
        return r.json()["message"]["content"]


@app.get("/api/voice/models")
async def voice_models():
    """List available Ollama models for voice chat."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get("http://localhost:11434/api/tags")
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            return {"models": models}
    except Exception as e:
        return {"models": [], "error": str(e)}


@app.get("/api/voice/whisper-models")
async def voice_whisper_models():
    """List locally available whisper model sizes."""
    sizes = []
    if WHISPER_MODELS_DIR.exists():
        for d in sorted(WHISPER_MODELS_DIR.iterdir()):
            if d.is_dir() and d.name.startswith("faster-whisper-"):
                sizes.append(d.name.replace("faster-whisper-", ""))
    return {"models": sizes}


@app.get("/api/voice/tts-voices")
async def voice_tts_voices():
    """List edge-tts voices."""
    try:
        import edge_tts
        voices = await edge_tts.list_voices()
        result = [
            {"name": v["ShortName"], "gender": v["Gender"], "locale": v["Locale"]}
            for v in voices if v["Locale"].startswith("en-")
        ]
        return {"voices": result}
    except Exception as e:
        return {"voices": [], "error": str(e)}


def _wav_bytes_to_float32(wav_data: bytes) -> np.ndarray:
    """Convert WAV bytes to float32 numpy array at 16kHz mono."""
    with wave.open(io.BytesIO(wav_data), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(sw, np.int16)
    audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    audio /= np.iinfo(dtype).max

    # Mono
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)

    # Resample to 16kHz if needed
    if sr != 16000:
        duration = len(audio) / sr
        target_len = int(duration * 16000)
        indices = np.linspace(0, len(audio) - 1, target_len).astype(int)
        audio = audio[indices]

    return audio


def _webm_to_float32(webm_data: bytes) -> np.ndarray | None:
    """Convert WebM/Opus audio to float32 16kHz mono via ffmpeg."""
    tmp_in = Path(tempfile.gettempdir()) / "voice_chat_in.webm"
    tmp_out = Path(tempfile.gettempdir()) / "voice_chat_out.wav"
    tmp_in.write_bytes(webm_data)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp_in), "-ar", "16000", "-ac", "1",
             "-f", "wav", str(tmp_out)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=10,
        )
        if tmp_out.exists():
            return _wav_bytes_to_float32(tmp_out.read_bytes())
    except Exception:
        pass
    finally:
        tmp_in.unlink(missing_ok=True)
        tmp_out.unlink(missing_ok=True)
    return None


@app.websocket("/ws/voice-chat")
async def ws_voice_chat(websocket: WebSocket):
    """
    Full voice chat pipeline over WebSocket.

    Client sends JSON config, then binary audio frames (WebM/Opus from MediaRecorder).
    Server responds with JSON transcripts and binary audio (MP3).

    Protocol:
      Client → {"type":"config", "model":"...", "voice":"...", "system":"..."}
      Client → {"type":"audio", "data":"<base64 WAV/WebM>"}
      Server → {"type":"transcript", "role":"user", "text":"..."}
      Server → {"type":"transcript", "role":"assistant", "text":"..."}
      Server → {"type":"audio", "data":"<base64 MP3>", "format":"mp3"}
      Server → {"type":"status", "text":"..."}
      Server → {"type":"error", "text":"..."}
    """
    await websocket.accept()

    model = "sage:latest"
    voice = "en-US-GuyNeural"
    system_prompt = (
        "You are a helpful voice assistant. Keep responses concise and conversational — "
        "they will be spoken aloud. Avoid markdown formatting, code blocks, or bullet points "
        "unless specifically asked. Respond naturally as if speaking to someone."
    )
    messages: list[dict] = []
    whisper_loaded = False

    try:
        while True:
            raw = await websocket.receive()

            # Handle text messages (JSON)
            if "text" in raw:
                msg = json.loads(raw["text"])
                msg_type = msg.get("type", "")

                if msg_type == "config":
                    model = msg.get("model", model)
                    voice = msg.get("voice", voice)
                    if msg.get("system"):
                        system_prompt = msg["system"]
                    messages = [{"role": "system", "content": system_prompt}]
                    await websocket.send_json({"type": "status", "text": f"Configured: {model}"})

                    # Pre-load whisper in background
                    if not whisper_loaded:
                        await websocket.send_json({"type": "status", "text": "Loading Whisper STT..."})
                        await asyncio.to_thread(_load_whisper)
                        whisper_loaded = True
                        await websocket.send_json({"type": "status", "text": "Whisper ready"})

                elif msg_type == "audio":
                    import base64
                    audio_b64 = msg.get("data", "")
                    audio_bytes = base64.b64decode(audio_b64)

                    if not messages:
                        messages = [{"role": "system", "content": system_prompt}]

                    # Load whisper if not done
                    if not whisper_loaded:
                        await websocket.send_json({"type": "status", "text": "Loading Whisper..."})
                        await asyncio.to_thread(_load_whisper)
                        whisper_loaded = True

                    # Detect format and convert
                    await websocket.send_json({"type": "status", "text": "Transcribing..."})
                    if audio_bytes[:4] == b'RIFF':
                        audio_np = _wav_bytes_to_float32(audio_bytes)
                    else:
                        audio_np = await asyncio.to_thread(_webm_to_float32, audio_bytes)

                    if audio_np is None or len(audio_np) < 4800:  # < 0.3s
                        await websocket.send_json({"type": "error", "text": "Audio too short or unreadable"})
                        continue

                    # STT
                    text = await asyncio.to_thread(_transcribe_audio, audio_np)
                    if not text:
                        await websocket.send_json({"type": "error", "text": "Couldn't understand audio"})
                        continue

                    await websocket.send_json({"type": "transcript", "role": "user", "text": text})

                    # LLM
                    await websocket.send_json({"type": "status", "text": "Thinking..."})
                    messages.append({"role": "user", "content": text})
                    try:
                        response = await _chat_ollama(messages, model)
                    except Exception as e:
                        await websocket.send_json({"type": "error", "text": f"LLM error: {e}"})
                        messages.pop()  # remove failed user message
                        continue

                    messages.append({"role": "assistant", "content": response})
                    await websocket.send_json({"type": "transcript", "role": "assistant", "text": response})

                    # TTS
                    await websocket.send_json({"type": "status", "text": "Speaking..."})
                    try:
                        audio_out = await _tts_edge(response, voice=voice)
                        audio_out_b64 = base64.b64encode(audio_out).decode()
                        await websocket.send_json({
                            "type": "audio",
                            "data": audio_out_b64,
                            "format": "mp3",
                        })
                    except Exception as e:
                        await websocket.send_json({"type": "error", "text": f"TTS error: {e}"})

                    await websocket.send_json({"type": "status", "text": "Ready"})

                elif msg_type == "text":
                    # Text-only chat (no mic needed)
                    user_text = msg.get("text", "").strip()
                    if not user_text:
                        continue

                    if not messages:
                        messages = [{"role": "system", "content": system_prompt}]

                    await websocket.send_json({"type": "transcript", "role": "user", "text": user_text})

                    await websocket.send_json({"type": "status", "text": "Thinking..."})
                    messages.append({"role": "user", "content": user_text})
                    try:
                        response = await _chat_ollama(messages, model)
                    except Exception as e:
                        await websocket.send_json({"type": "error", "text": f"LLM error: {e}"})
                        messages.pop()
                        continue

                    messages.append({"role": "assistant", "content": response})
                    await websocket.send_json({"type": "transcript", "role": "assistant", "text": response})

                    # TTS
                    await websocket.send_json({"type": "status", "text": "Speaking..."})
                    try:
                        import base64 as b64mod
                        audio_out = await _tts_edge(response, voice=voice)
                        audio_out_b64 = b64mod.b64encode(audio_out).decode()
                        await websocket.send_json({
                            "type": "audio", "data": audio_out_b64, "format": "mp3",
                        })
                    except Exception as e:
                        await websocket.send_json({"type": "error", "text": f"TTS error: {e}"})

                    await websocket.send_json({"type": "status", "text": "Ready"})

                elif msg_type == "clear":
                    messages = [{"role": "system", "content": system_prompt}]
                    await websocket.send_json({"type": "status", "text": "Conversation cleared"})

    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ── Entry Point ────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# DATASET ANALYSIS ENDPOINT + WEBSOCKET BRAIN STREAMING
# ══════════════════════════════════════════════════════════════════════════════

from fastapi import UploadFile, File
from fastapi.responses import StreamingResponse

# Import our enhanced dataset analyzer
import sys as _sys
_sys.path.insert(0, str(WORKSPACE / "enhanced"))
try:
    from dataset_analyzer import DatasetAnalyzer
    _HAS_DATASET_ANALYZER = True
except ImportError:
    _HAS_DATASET_ANALYZER = False

# ── WebSocket: Live Brain Cognitive Streaming ─────────────────────────────
_brain_analysis_clients: set = set()

@app.websocket("/ws/brain-analysis")
async def ws_brain_analysis(websocket: WebSocket):
    """WebSocket endpoint — dashboard connects to receive streaming cognitive frames."""
    await websocket.accept()
    _brain_analysis_clients.add(websocket)
    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_text('{"type":"ping"}')
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _brain_analysis_clients.discard(websocket)


async def _broadcast_brain_frame(frame_json: str):
    """Send a cognitive frame to all connected brain-analysis WS clients."""
    dead = set()
    for ws in list(_brain_analysis_clients):
        try:
            await ws.send_text(frame_json)
        except Exception:
            dead.add(ws)
    _brain_analysis_clients -= dead


# ── REST: Upload + Analyze Dataset ───────────────────────────────────────────

@app.post("/api/analyze-dataset")
async def analyze_dataset(file: UploadFile = File(...)):
    """
    Accepts a dataset file (CSV, JSON, TXT, JSONL, TSV, XML, YAML).
    Runs full cognitive analysis. Returns all frames + summary.
    Also broadcasts frames live to /ws/brain-analysis clients.
    """
    if not _HAS_DATASET_ANALYZER:
        raise HTTPException(status_code=503, detail="dataset_analyzer module not found")

    content  = await file.read()
    filename = file.filename or "dataset.txt"

    collected_frames = []

    async def ws_send(json_str: str):
        collected_frames.append(json.loads(json_str))
        asyncio.create_task(_broadcast_brain_frame(json_str))

    analyzer = DatasetAnalyzer()
    await analyzer.analyze(content, filename, ws_send)
    summary = analyzer.get_summary()

    return JSONResponse({
        "status": "ok",
        "filename": filename,
        "frames": collected_frames,
        "summary": summary,
        "frame_count": len(collected_frames),
    })


@app.post("/api/analyze-dataset-stream")
async def analyze_dataset_stream(file: UploadFile = File(...)):
    """Server-Sent Events version — streams cognitive frames as generated."""
    if not _HAS_DATASET_ANALYZER:
        raise HTTPException(status_code=503, detail="dataset_analyzer not available")

    content  = await file.read()
    filename = file.filename or "dataset.txt"

    async def event_gen():
        frames = []
        async def collect(json_str: str):
            frames.append(json_str)
            asyncio.create_task(_broadcast_brain_frame(json_str))

        analyzer = DatasetAnalyzer()
        await analyzer.analyze(content, filename, collect)
        summary = analyzer.get_summary()

        for frame_json in frames:
            yield f"data: {frame_json}\n\n"
            await asyncio.sleep(0.001)

        yield f"data: {json.dumps({'type': 'analysis_complete', 'summary': summary})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache",
                                       "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn

    HOST, PORT = "127.0.0.1", 9000

    # Check if port is already in use
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex((HOST, PORT)) == 0:
            print(f"ERROR: Port {PORT} is already in use.")
            print(f"  Another instance may be running. Kill it first or choose a different port.")
            sys.exit(1)

    print("=" * 50)
    print("  RE WORKSPACE DASHBOARD")
    print(f"  http://{HOST}:{PORT}")
    print("=" * 50)

    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
