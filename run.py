"""
run.py — Ultimate launcher for Dell FutureMinds webapp.

  python run.py

Fixes THREE problems automatically:

  1. DEPENDENCY FIX
     Checks if flask / python-dotenv are importable. If not, runs
     `uv add flask python-dotenv` which adds them to pyproject.toml
     AND installs them — so uv sync picks them up forever after.

  2. 404 JOB STATUS FIX (THE MAIN BUG)
     Root cause: Werkzeug's watchdog watches ALL of sys.path, including
     .venv/site-packages. PyTorch writes to torch/_inductor/config.py
     at import time. Werkzeug sees this as a code change → restarts Flask
     → in-memory _jobs dict wiped → /api/status/<job_id> returns 404.

     Fix: use reloader_type="stat" which only polls files we explicitly
     list in extra_files — it never scans site-packages, so PyTorch
     cache changes no longer trigger restarts.

  3. THREADING FIX  (python -m webapp.app issue)
     Flask's default single-threaded mode means the background ingestion
     thread and the browser's polling requests block each other. Setting
     threaded=True lets both run concurrently so polling works fine.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — ensure flask + python-dotenv are in pyproject.toml AND installed
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_deps() -> None:
    missing: list[str] = []

    try:
        import flask  # noqa: F401
    except ImportError:
        missing.append("flask>=3.1")

    try:
        import dotenv  # noqa: F401
    except ImportError:
        missing.append("python-dotenv>=1.2")

    if missing:
        print(f"\n[run.py] Missing packages: {missing}")
        print("[run.py] Running: uv add " + " ".join(missing))
        subprocess.check_call(
            ["uv", "add"] + missing,
            cwd=str(ROOT),
        )
        print("[run.py] Done. Re-launching with updated environment...\n")
        # Re-exec so the newly installed packages are importable
        os.execv(sys.executable, [sys.executable, str(__file__)] + sys.argv[1:])


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — start Flask with a safe reloader
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _ensure_deps()

    # Add project root to sys.path so `from webapp import ...` works
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    # Load .env BEFORE importing webapp modules (they read env vars at import)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    from webapp.app import app
    from webapp import config

    # ── Which source files should trigger a reload? ───────────────────────
    # Only watch files in webapp/  (our actual source code).
    # Do NOT watch output/, data/, .venv/ — changes there are pipeline
    # artefacts and package internals, NOT code changes.
    watch_files: list[str] = [
        str(f)
        for f in (ROOT / "webapp").rglob("*")
        if f.suffix in (".py", ".html", ".css", ".js") and ".pyc" not in str(f)
    ]

    host = config.FLASK_HOST
    port = config.FLASK_PORT

    print("\n" + "═" * 55)
    print("  Dell FutureMinds — GraphRAG Webapp")
    print("═" * 55)
    print(f"  URL      :  http://localhost:{port}")
    print(f"  Debug    :  ON")
    print(f"  Threaded :  ON  (background ingestion + polling work together)")
    print(f"  Reloader :  stat  (watches webapp/ only, ignores .venv/)")
    print(f"  Press Ctrl+C to stop")
    print("═" * 55 + "\n")

    from werkzeug.serving import run_simple

    run_simple(
        hostname       = host,
        port           = port,
        application    = app,
        use_reloader   = True,
        use_debugger   = True,
        threaded       = True,               # Fix 3: concurrent requests
        extra_files    = watch_files,        # Fix 2a: only watch webapp/
        reloader_type  = "stat",             # Fix 2b: stat never scans venv
    )


if __name__ == "__main__":
    main()