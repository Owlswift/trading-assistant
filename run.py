"""Start / restart the Trading Assistant.

    python run.py            start the app; it RESTARTS ITSELF when a .py file or .env changes
    python run.py restart    stop whatever is already on the port, then start (use if it is running elsewhere)
    python run.py stop       stop whatever is on the port

Port: 8000, or set PORT in the environment. Frontend files (frontend/) are served from disk and need
no restart, only a browser refresh.
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent
PORT = int(os.getenv("PORT", "8000"))


def pids_on_port(port: int) -> list[int]:
    out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], capture_output=True, text=True).stdout
    return [int(p) for p in out.split() if p.strip().isdigit()]


def stop(port: int) -> None:
    pids = pids_on_port(port)
    for pid in pids:
        os.kill(pid, signal.SIGTERM)
    for _ in range(40):  # wait up to ~4 s for the port to free up
        if not pids_on_port(port):
            break
        time.sleep(0.1)
    print(f"stopped {len(pids)} process(es) on port {port}" if pids else f"nothing was running on port {port}")


def start(port: int) -> None:
    uvicorn.run(
        "backend.app.main:app",
        host="127.0.0.1",
        port=port,
        reload=True,
        # Open browser WebSockets (live ticks) would otherwise block a reload forever
        timeout_graceful_shutdown=2,
        reload_dirs=[str(ROOT)],
        reload_includes=["*.py", ".env"],
        # data/ is written constantly by the MT5 feed; never restart for it
        reload_excludes=["data/*", ".venv/*", "frontend/*", "mt5/*", "*/__pycache__/*"],
    )


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "start"
    if cmd == "stop":
        stop(PORT)
    elif cmd == "restart":
        stop(PORT)
        start(PORT)
    elif cmd == "start":
        start(PORT)
    else:
        sys.exit(__doc__)
