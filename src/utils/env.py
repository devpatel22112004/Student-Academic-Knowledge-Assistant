import os
import sys
from pathlib import Path


def running_inside_streamlit():
    """Return True when the app is running inside the Streamlit runtime."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def launch_streamlit_app():
    """Re-launch the current app using Streamlit when started with plain Python."""
    app_path = Path(__file__).resolve().parents[2] / "app.py"
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
        ],
    )