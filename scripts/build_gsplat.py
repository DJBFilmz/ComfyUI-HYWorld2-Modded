from __future__ import annotations

import inspect
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from packaging import tags


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FORK_DIR = PROJECT_ROOT / "hyworld2" / "worldgen" / "third_party" / "gsplat_maskgaussian"
WHEELS_DIR = PROJECT_ROOT / "gsplat"


def run_command(args: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("[CMD]", " ".join(f'"{a}"' if " " in a else a for a in args))
    subprocess.check_call(args, cwd=str(cwd) if cwd else None, env=env)


def torch_build_label() -> str:
    import torch

    torch_version = torch.__version__.split("+", 1)[0].split(".")
    torch_label = "pt" + "".join(torch_version[:3])
    cuda_version = torch.version.cuda or "cpu"
    cuda_label = "cpu" if cuda_version == "cpu" else "cu" + cuda_version.replace(".", "")
    return f"hyworld.{torch_label}.{cuda_label}"


def supported_python_tags() -> set[str]:
    return {tag.interpreter for tag in tags.sys_tags()}


def find_compatible_wheel(local_label: str) -> Path | None:
    py_tags = supported_python_tags()
    for wheel_path in sorted(WHEELS_DIR.glob(f"gsplat-*+{local_label}-*.whl"), key=os.path.getmtime, reverse=True):
        parts = wheel_path.name.split("-")
        if len(parts) >= 5 and parts[-3] in py_tags:
            return wheel_path
    return None


def build_wheel(local_label: str) -> Path:
    WHEELS_DIR.mkdir(exist_ok=True)
    build_dir = FORK_DIR / "build"
    if build_dir.exists():
        print(f"[INFO] Removing stale build directory: {build_dir}")
        shutil.rmtree(build_dir)
    env = os.environ.copy()
    env.setdefault("MAX_JOBS", "10")
    env["GSPLAT_LOCAL_VERSION"] = local_label
    version_path = FORK_DIR / "gsplat" / "version.py"
    original_version = version_path.read_text(encoding="utf-8")
    try:
        version_path.write_text(f'__version__ = "1.5.3+{local_label}"\n', encoding="utf-8")
        run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-build-isolation",
                "--no-deps",
                "-w",
                str(WHEELS_DIR),
                str(FORK_DIR),
            ],
            cwd=FORK_DIR,
            env=env,
        )
    finally:
        version_path.write_text(original_version, encoding="utf-8")
    wheel = find_compatible_wheel(local_label)
    if wheel is None:
        raise RuntimeError(f"HY-World gsplat wheel build finished, but no compatible wheel found in {WHEELS_DIR}")
    print(f"[OK] Built HY-World gsplat wheel: {wheel}")
    return wheel


def verify_hyworld_gsplat() -> None:
    import torch
    import gsplat
    from gsplat.rendering import rasterization

    sig = inspect.signature(rasterization)
    required = {"distloss", "gauss_masks"}
    missing = sorted(required - set(sig.parameters))
    if missing:
        raise RuntimeError(
            "Installed gsplat is not the HY-World gsplat_maskgaussian fork; "
            f"missing rasterization kwargs: {missing}"
        )
    print(f"[OK] Imported gsplat {getattr(gsplat, '__version__', 'unknown')} from {gsplat.__file__}")

    if not torch.cuda.is_available():
        print("[WARN] CUDA is not available; skipped rasterization smoke test.")
        return

    smoke = r'''
import torch
from gsplat.rendering import rasterization

device = "cuda"
means = torch.tensor([[0.0, 0.0, 2.0]], device=device)
quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
scales = torch.tensor([[0.1, 0.1, 0.1]], device=device)
opacities = torch.tensor([0.9], device=device)
colors = torch.tensor([[[0.8, 0.2, 0.1]]], device=device)
viewmats = torch.eye(4, device=device)[None]
Ks = torch.tensor([[[64.0, 0.0, 32.0], [0.0, 64.0, 32.0], [0.0, 0.0, 1.0]]], device=device)
render, alpha, info = rasterization(
    means=means,
    quats=quats,
    scales=scales,
    opacities=opacities,
    colors=colors,
    viewmats=viewmats,
    Ks=Ks,
    width=64,
    height=64,
    sh_degree=0,
    packed=False,
    distloss=True,
    gauss_masks=torch.ones((1,), device=device),
)
assert alpha.max().item() > 0.0, "HY-World gsplat smoke test rendered empty alpha"
print("[OK] HY-World gsplat_maskgaussian CUDA smoke test passed.")
'''
    with tempfile.NamedTemporaryFile("w", suffix="_hyworld_gsplat_smoke.py", delete=False, encoding="utf-8") as fh:
        fh.write(smoke)
        smoke_path = fh.name
    try:
        run_command([sys.executable, smoke_path])
    finally:
        try:
            os.remove(smoke_path)
        except OSError:
            pass


def main() -> None:
    if not FORK_DIR.exists():
        raise FileNotFoundError(f"HY-World gsplat_maskgaussian fork not found: {FORK_DIR}")

    print("============================================================")
    print("   HY-World gsplat_maskgaussian installer")
    print("============================================================")
    print(f"[INFO] Fork directory: {FORK_DIR}")
    local_label = torch_build_label()
    print(f"[INFO] Wheel local version label: {local_label}")

    wheel = find_compatible_wheel(local_label)
    if wheel is None:
        wheel = build_wheel(local_label)
    else:
        print(f"[INFO] Using cached HY-World gsplat wheel: {wheel}")

    run_command([sys.executable, "-m", "pip", "uninstall", "-y", "gsplat"])
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            str(wheel),
        ],
    )
    verify_hyworld_gsplat()


if __name__ == "__main__":
    main()
