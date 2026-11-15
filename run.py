"""
Convenience launcher for CineMatch AI Movie Recommender System.
Runs Streamlit with optimal settings whether invoked from root or subfolder.
"""

import os
from pathlib import Path
import subprocess
import sys


def main():
    base_dir = Path(__file__).resolve().parent
    app_path = base_dir / "app.py"

    venv_streamlit = base_dir / ".venv" / "Scripts" / "streamlit.exe"
    cmd = [str(venv_streamlit) if venv_streamlit.exists() else "streamlit", "run", str(app_path)]

    print(f"[🎬] Starting CineMatch AI Movie Recommender...")
    print(f"[>] App file: {app_path}")
    print(f"[>] Executing: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[✓] CineMatch AI stopped cleanly.")
    except Exception as e:
        print(f"[!] Error launching Streamlit: {e}")
        print("    Try running directly: streamlit run app.py")


if __name__ == "__main__":
    main()

