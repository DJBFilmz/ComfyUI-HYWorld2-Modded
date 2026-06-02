import subprocess
import sys
from pathlib import Path

build_script = Path(__file__).parent / "scripts" / "build_gsplat.py"

print("[HYWorld2] Installing HY-World gsplat_maskgaussian fork...")
try:
    subprocess.check_call([sys.executable, str(build_script)])
    print("[HYWorld2] HY-World gsplat_maskgaussian installed successfully.")
except subprocess.CalledProcessError as e:
    print(f"[HYWorld2] WARNING: HY-World gsplat_maskgaussian build failed (exit code {e.returncode}).")
    print("[HYWorld2] You can retry manually: run scripts/pipinstall.bat")
