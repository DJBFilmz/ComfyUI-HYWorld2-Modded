import subprocess
import sys
from pathlib import Path

build_script = Path(__file__).parent / "scripts" / "build_gsplat.py"

print("[HYWorld2] Installing HY-World native wheels (gsplat, recast, PyTorch3D)...")
try:
    subprocess.check_call([sys.executable, str(build_script)])
    print("[HYWorld2] HY-World native wheels installed successfully.")
except subprocess.CalledProcessError as e:
    print(f"[HYWorld2] WARNING: HY-World native wheel build failed (exit code {e.returncode}).")
    print("[HYWorld2] You can retry manually: run scripts/pipinstall.bat")
