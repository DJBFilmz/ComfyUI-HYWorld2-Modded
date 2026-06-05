from __future__ import annotations

import argparse
import base64
import hashlib
import inspect
import os
import re
import stat
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from packaging import tags


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FORK_DIR = PROJECT_ROOT / "hyworld2" / "worldgen" / "third_party" / "gsplat_maskgaussian"
NAVMESH_DIR = PROJECT_ROOT / "hyworld2" / "worldgen" / "third_party" / "navmesh"
RECASTNAV_DIR = PROJECT_ROOT / "hyworld2" / "worldgen" / "third_party" / "recastnavigation"
WHEELS_DIR = PROJECT_ROOT / "gsplat"
PYTORCH3D_BUILD_DIR = SCRIPT_DIR / "pytorch3d_build"
FUSED_SSIM_BUILD_DIR = SCRIPT_DIR / "fused_ssim_build"
PYTORCH3D_REPO_URL = "https://github.com/facebookresearch/pytorch3d.git"
PYTORCH3D_REF = "stable"
FUSED_SSIM_REPO_URL = "https://github.com/rahul-goel/fused-ssim/"
RECASTNAV_REPO_URL = "https://github.com/recastnavigation/recastnavigation.git"
CUDA_130_HOME = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0")


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


def pytorch3d_build_label() -> str:
    import torch

    torch_version = torch.__version__.split("+", 1)[0].split(".")
    torch_label = "pt" + "".join(torch_version[:3])
    cuda_version = torch.version.cuda or "cpu"
    cuda_label = "cpu" if cuda_version == "cpu" else "cu" + cuda_version.replace(".", "")
    python_label = f"cp{sys.version_info.major}{sys.version_info.minor}"
    return f"{torch_label}.{cuda_label}.{python_label}.nopulsar"


def supported_python_tags() -> set[str]:
    return {tag.interpreter for tag in tags.sys_tags()}


def find_compatible_wheel(local_label: str) -> Path | None:
    py_tags = supported_python_tags()
    for wheel_path in sorted(WHEELS_DIR.glob(f"gsplat-*+{local_label}-*.whl"), key=os.path.getmtime, reverse=True):
        parts = wheel_path.name.split("-")
        if len(parts) >= 5 and parts[-3] in py_tags:
            return wheel_path
    return None


def find_compatible_pytorch3d_wheel(local_label: str) -> Path | None:
    py_tags = supported_python_tags()
    for wheel_path in sorted(WHEELS_DIR.glob(f"pytorch3d-*+{local_label}-*.whl"), key=os.path.getmtime, reverse=True):
        parts = wheel_path.name.split("-")
        if len(parts) >= 5 and parts[-3] in py_tags:
            return wheel_path
    return None


def find_compatible_fused_ssim_wheel() -> Path | None:
    py_tags = supported_python_tags()
    for wheel_path in sorted(WHEELS_DIR.glob("fused_ssim-*.whl"), key=os.path.getmtime, reverse=True):
        parts = wheel_path.name.split("-")
        if len(parts) >= 5 and parts[-3] in py_tags:
            return wheel_path
    return None


def find_compatible_recast_wheel() -> Path | None:
    py_tags = supported_python_tags()
    for wheel_path in sorted(WHEELS_DIR.glob("recast-*.whl"), key=os.path.getmtime, reverse=True):
        parts = wheel_path.name.split("-")
        if len(parts) >= 5 and parts[-3] in py_tags:
            return wheel_path
    return None


def assert_build_dir_is_safe(path: Path) -> None:
    resolved = path.resolve()
    allowed_parent = SCRIPT_DIR.resolve()
    if resolved == allowed_parent or allowed_parent not in resolved.parents:
        raise RuntimeError(f"Refusing to reset build directory outside scripts/: {resolved}")


def reset_pytorch3d_build_dir() -> None:
    assert_build_dir_is_safe(PYTORCH3D_BUILD_DIR)
    if PYTORCH3D_BUILD_DIR.exists():
        print(f"[INFO] Removing stale PyTorch3D build directory: {PYTORCH3D_BUILD_DIR}")
        def handle_remove_readonly(function, path, exc_info):
            os.chmod(path, stat.S_IWRITE)
            function(path)

        shutil.rmtree(PYTORCH3D_BUILD_DIR, onerror=handle_remove_readonly)


def reset_fused_ssim_build_dir() -> None:
    assert_build_dir_is_safe(FUSED_SSIM_BUILD_DIR)
    if FUSED_SSIM_BUILD_DIR.exists():
        print(f"[INFO] Removing stale fused-ssim build directory: {FUSED_SSIM_BUILD_DIR}")
        def handle_remove_readonly(function, path, exc_info):
            os.chmod(path, stat.S_IWRITE)
            function(path)

        shutil.rmtree(FUSED_SSIM_BUILD_DIR, onerror=handle_remove_readonly)


def directory_has_files(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def ensure_recastnavigation_source() -> None:
    if (RECASTNAV_DIR / "Recast" / "Source" / "Recast.cpp").exists() and (
        RECASTNAV_DIR / "Detour" / "Source" / "DetourAlloc.cpp"
    ).exists():
        return
    if directory_has_files(RECASTNAV_DIR):
        print(f"[WARN] RecastNavigation source looks incomplete; refreshing: {RECASTNAV_DIR}")
        def handle_remove_readonly(function, path, exc_info):
            os.chmod(path, stat.S_IWRITE)
            function(path)

        shutil.rmtree(RECASTNAV_DIR, onerror=handle_remove_readonly)
    RECASTNAV_DIR.parent.mkdir(parents=True, exist_ok=True)
    run_command([
        "git",
        "clone",
        "--depth",
        "1",
        RECASTNAV_REPO_URL,
        str(RECASTNAV_DIR),
    ])


def clone_pytorch3d_source() -> None:
    reset_pytorch3d_build_dir()
    run_command([
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        PYTORCH3D_REF,
        PYTORCH3D_REPO_URL,
        str(PYTORCH3D_BUILD_DIR),
    ])
    patch_pytorch3d_windows_build()


def clone_fused_ssim_source() -> None:
    reset_fused_ssim_build_dir()
    run_command([
        "git",
        "clone",
        "--depth",
        "1",
        FUSED_SSIM_REPO_URL,
        str(FUSED_SSIM_BUILD_DIR),
    ])
    patch_fused_ssim_windows_build()


def patch_pytorch3d_windows_build() -> Path | None:
    if os.name != "nt":
        return None
    shim_header = PYTORCH3D_BUILD_DIR / "hyworld2_undef_windows_small.h"
    shim_header.write_text(
        "#pragma once\n"
        "#include <windows.h>\n"
        "#ifdef small\n"
        "#undef small\n"
        "#endif\n",
        encoding="utf-8",
    )
    setup_path = PYTORCH3D_BUILD_DIR / "setup.py"
    setup_text = setup_path.read_text(encoding="utf-8")
    needle = '    extra_compile_args = {"cxx": ["-std=c++17"]}\n'
    replacement = (
        '    extra_compile_args = {"cxx": ["-std=c++17"]}\n'
        "    if os.name == \"nt\":\n"
        "        extra_compile_args[\"cxx\"].append(\"/DNOMINMAX\")\n"
    )
    if needle in setup_text and "/DNOMINMAX" not in setup_text:
        setup_path.write_text(setup_text.replace(needle, replacement, 1), encoding="utf-8")
        setup_text = setup_path.read_text(encoding="utf-8")
    filter_needle = '    extension = CppExtension\n'
    filter_replacement = (
        '    extension = CppExtension\n'
        '    if os.name == "nt":\n'
        '        sources = [path for path in sources if "pytorch3d{}csrc{}pulsar".format(os.sep, os.sep) not in path]\n'
        '        source_cuda = [path for path in source_cuda if "pytorch3d{}csrc{}pulsar".format(os.sep, os.sep) not in path]\n'
    )
    if filter_needle in setup_text and "csrc{}pulsar" not in setup_text:
        setup_path.write_text(setup_text.replace(filter_needle, filter_replacement, 1), encoding="utf-8")
    ext_path = PYTORCH3D_BUILD_DIR / "pytorch3d" / "csrc" / "ext.cpp"
    ext_text = ext_path.read_text(encoding="utf-8")
    ext_text = re.sub(
        r'#if !defined\(USE_ROCM\)\n#include "\./pulsar/global\.h".*?\n#endif\n',
        "",
        ext_text,
        flags=re.DOTALL,
    )
    ext_text = re.sub(
        r'#if !defined\(USE_ROCM\)\n#include "\./pulsar/pytorch/renderer\.h"\n#include "\./pulsar/pytorch/tensor_util\.h"\n#endif\n',
        "",
        ext_text,
        flags=re.DOTALL,
    )
    pulsar_start = ext_text.find("  // Pulsar.")
    if pulsar_start != -1:
        pulsar_end = ext_text.find("\n#endif\n}", pulsar_start)
        if pulsar_end == -1:
            raise RuntimeError("Could not locate PyTorch3D Pulsar binding block end")
        ext_text = ext_text[:pulsar_start] + "  // Pulsar disabled for HYWorld2 Windows nopulsar build.\n" + ext_text[pulsar_end + len("\n#endif"):]
    ext_path.write_text(ext_text, encoding="utf-8")
    return shim_header


def patch_fused_ssim_windows_build() -> None:
    if os.name != "nt":
        return
    shim_header = FUSED_SSIM_BUILD_DIR / "hyworld2_undef_windows_small.h"
    shim_header.write_text(
        "#pragma once\n"
        "#include <windows.h>\n"
        "#ifdef small\n"
        "#undef small\n"
        "#endif\n",
        encoding="utf-8",
    )
    setup_path = FUSED_SSIM_BUILD_DIR / "setup.py"
    setup_text = setup_path.read_text(encoding="utf-8")
    needle = 'compiler_args = {"cxx": ["-O3", "-DFUSED_SSIM_CUDA"], "nvcc": ["-O3", "-DFUSED_SSIM_CUDA"]}\n'
    shim_path = str(shim_header).replace("\\", "/")
    replacement = (
        'compiler_args = {"cxx": ["-O3", "-DFUSED_SSIM_CUDA"], '
        f'"nvcc": ["-O3", "-DFUSED_SSIM_CUDA", "-allow-unsupported-compiler", "--pre-include", r"{shim_path}"]}}\n'
    )
    if needle in setup_text and "-allow-unsupported-compiler" not in setup_text:
        setup_path.write_text(setup_text.replace(needle, replacement, 1), encoding="utf-8")
    for header_name in ("ssim.h", "ssim3d.h"):
        header_path = FUSED_SSIM_BUILD_DIR / header_name
        header_text = header_path.read_text(encoding="utf-8")
        header_text = header_text.replace("#include <torch/extension.h>", "#include <torch/types.h>")
        header_path.write_text(header_text, encoding="utf-8")
    for source_name, header_name in (("ssim.cu", "ssim.h"), ("ssim3d.cu", "ssim3d.h")):
        source_path = FUSED_SSIM_BUILD_DIR / source_name
        source_text = source_path.read_text(encoding="utf-8")
        source_text = source_text.replace(
            "#include <torch/extension.h>",
            f'#include "{header_name}"\n#include <ATen/Functions.h>',
        )
        source_text = source_text.replace("torch::zeros_like", "at::zeros_like")
        source_text = source_text.replace("torch::empty", "at::empty")
        source_path.write_text(source_text, encoding="utf-8")


def relabel_wheel(wheel: Path, local_label: str) -> Path:
    if "+" in wheel.name:
        return wheel
    parts = wheel.name.split("-", 2)
    if len(parts) != 3:
        return wheel
    new_name = f"{parts[0]}-{parts[1]}+{local_label}-{parts[2]}"
    new_path = wheel.with_name(new_name)
    if new_path != wheel:
        shutil.copy2(wheel, new_path)
    return new_path


def cuda_130_build_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MAX_JOBS", "10")
    env.setdefault("FORCE_CUDA", "1")
    if CUDA_130_HOME.exists():
        env["CUDA_HOME"] = str(CUDA_130_HOME)
        env["CUDA_PATH"] = str(CUDA_130_HOME)
        env["CUB_HOME"] = str(CUDA_130_HOME / "include")
    env["NVCC_FLAGS"] = " ".join(
        part for part in [env.get("NVCC_FLAGS", ""), "-allow-unsupported-compiler"] if part
    )
    return env


def patch_pytorch3d_source_nopulsar() -> None:
    points_init = PYTORCH3D_BUILD_DIR / "pytorch3d" / "renderer" / "points" / "__init__.py"
    points_text = points_init.read_text(encoding="utf-8")
    points_text = re.sub(
        r"\r?\n# Pulsar not enabled on amd\.\r?\nif not torch\.version\.hip:\r?\n    from \.pulsar\.unified import PulsarPointsRenderer\r?\n",
        "\n# Pulsar disabled for HYWorld2 Windows nopulsar build.\n",
        points_text,
    )
    points_init.write_text(points_text, encoding="utf-8")

    renderer_init = PYTORCH3D_BUILD_DIR / "pytorch3d" / "renderer" / "__init__.py"
    renderer_text = renderer_init.read_text(encoding="utf-8")
    renderer_text = re.sub(
        r"\r?\n# Pulsar is not enabled on amd\.\r?\nif not torch\.version\.hip:\r?\n    from \.points import PulsarPointsRenderer\r?\n",
        "\n# Pulsar disabled for HYWorld2 Windows nopulsar build.\n",
        renderer_text,
    )
    renderer_init.write_text(renderer_text, encoding="utf-8")


def patch_pytorch3d_wheel_nopulsar(wheel: Path) -> Path:
    replacements = {
        "pytorch3d/renderer/points/__init__.py": lambda text: re.sub(
            r"\r?\n# Pulsar not enabled on amd\.\r?\nif not torch\.version\.hip:\r?\n    from \.pulsar\.unified import PulsarPointsRenderer\r?\n",
            "\n# Pulsar disabled for HYWorld2 Windows nopulsar build.\n",
            text,
        ),
        "pytorch3d/renderer/__init__.py": lambda text: re.sub(
            r"\r?\n# Pulsar is not enabled on amd\.\r?\nif not torch\.version\.hip:\r?\n    from \.points import PulsarPointsRenderer\r?\n",
            "\n# Pulsar disabled for HYWorld2 Windows nopulsar build.\n",
            text,
        ),
    }
    tmp_wheel = wheel.with_suffix(".patched.whl")
    record_path = None
    entries: dict[str, bytes] = {}
    infos = {}
    with zipfile.ZipFile(wheel, "r") as zin:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename in replacements:
                data = replacements[info.filename](data.decode("utf-8")).encode("utf-8")
            if info.filename.endswith(".dist-info/RECORD"):
                record_path = info.filename
            entries[info.filename] = data
            infos[info.filename] = info
    if record_path is None:
        raise RuntimeError(f"PyTorch3D wheel RECORD not found: {wheel}")
    record_lines = []
    for raw_line in entries[record_path].decode("utf-8").splitlines():
        parts = raw_line.split(",")
        path = parts[0]
        if path == record_path:
            record_lines.append(f"{path},,")
            continue
        data = entries.get(path)
        if data is None:
            record_lines.append(raw_line)
            continue
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode("ascii").rstrip("=")
        record_lines.append(f"{path},sha256={digest},{len(data)}")
    entries[record_path] = ("\n".join(record_lines) + "\n").encode("utf-8")

    with zipfile.ZipFile(tmp_wheel, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for path, data in entries.items():
            info = infos[path]
            zout.writestr(info, data)
    shutil.move(str(tmp_wheel), str(wheel))
    return wheel


def build_pytorch3d_wheel(local_label: str) -> Path:
    WHEELS_DIR.mkdir(exist_ok=True)
    clone_pytorch3d_source()
    patch_pytorch3d_source_nopulsar()
    env = cuda_130_build_env()
    env.setdefault("PYTORCH3D_NO_NINJA", "1")
    env.setdefault("PYTORCH3D_NO_PULSAR", "1")
    env.setdefault("PYTORCH3D_DISABLE_PULSAR", "1")
    shim_header = PYTORCH3D_BUILD_DIR / "hyworld2_undef_windows_small.h"
    preinclude_flags = f"--pre-include {shim_header}" if shim_header.exists() else ""
    env["NVCC_FLAGS"] = " ".join(
        part for part in [env.get("NVCC_FLAGS", ""), "-allow-unsupported-compiler", "-DNOMINMAX", preinclude_flags] if part
    )
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
            str(PYTORCH3D_BUILD_DIR),
        ],
        cwd=PYTORCH3D_BUILD_DIR,
        env=env,
    )
    candidates = sorted(WHEELS_DIR.glob("pytorch3d-*.whl"), key=os.path.getmtime, reverse=True)
    if not candidates:
        raise RuntimeError(f"PyTorch3D wheel build finished, but no wheel found in {WHEELS_DIR}")
    wheel = relabel_wheel(candidates[0], local_label)
    patch_pytorch3d_wheel_nopulsar(wheel)
    print(f"[OK] Built PyTorch3D wheel: {wheel}")
    return wheel


def build_fused_ssim_wheel() -> Path:
    import torch

    if torch.version.cuda is None:
        raise RuntimeError(
            "fused-ssim must be built with the same CUDA-enabled Python/Torch environment used by ComfyUI. "
            f"Current Python has CPU-only torch: {sys.executable}"
        )
    WHEELS_DIR.mkdir(exist_ok=True)
    clone_fused_ssim_source()
    before = set(WHEELS_DIR.glob("fused_ssim-*.whl"))
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            "--no-cache-dir",
            "-w",
            str(WHEELS_DIR),
            str(FUSED_SSIM_BUILD_DIR),
        ],
        cwd=FUSED_SSIM_BUILD_DIR,
        env=cuda_130_build_env(),
    )
    after = set(WHEELS_DIR.glob("fused_ssim-*.whl"))
    candidates = sorted(after - before, key=os.path.getmtime, reverse=True)
    if not candidates:
        candidates = sorted(after, key=os.path.getmtime, reverse=True)
    for wheel in candidates:
        parts = wheel.name.split("-")
        if len(parts) >= 5 and parts[-3] in supported_python_tags():
            print(f"[OK] Built fused-ssim wheel: {wheel}")
            return wheel
    raise RuntimeError(f"fused-ssim wheel build finished, but no compatible wheel found in {WHEELS_DIR}")


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


def verify_pytorch3d() -> None:
    import torch
    import pytorch3d
    from pytorch3d.renderer.cameras import look_at_rotation

    rotations = look_at_rotation(torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32))
    if rotations.shape != (1, 3, 3):
        raise RuntimeError(f"Unexpected PyTorch3D smoke shape: {tuple(rotations.shape)}")
    print(f"[OK] Imported pytorch3d {getattr(pytorch3d, '__version__', 'unknown')} from {pytorch3d.__file__}")
    print("[OK] PyTorch3D renderer camera smoke test passed.")


def verify_recast() -> None:
    import recast

    module_path = getattr(recast, "__file__", "unknown")
    print(f"[OK] Imported recast from {module_path}")


def verify_fused_ssim() -> None:
    import fused_ssim

    module_path = getattr(fused_ssim, "__file__", "unknown")
    print(f"[OK] Imported fused_ssim from {module_path}")


def build_recast_wheel() -> Path:
    if not NAVMESH_DIR.exists():
        raise FileNotFoundError(f"HY-World navmesh bindings not found: {NAVMESH_DIR}")
    ensure_recastnavigation_source()
    WHEELS_DIR.mkdir(exist_ok=True)
    build_dir = NAVMESH_DIR / "build"
    if build_dir.exists():
        print(f"[INFO] Removing stale recast build directory: {build_dir}")
        shutil.rmtree(build_dir)
    env = os.environ.copy()
    env["RECAST_PATH"] = str(RECASTNAV_DIR)
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "-w",
            str(WHEELS_DIR),
            str(NAVMESH_DIR),
        ],
        cwd=NAVMESH_DIR,
        env=env,
    )
    wheel = find_compatible_recast_wheel()
    if wheel is None:
        raise RuntimeError(f"recast wheel build finished, but no compatible wheel found in {WHEELS_DIR}")
    print(f"[OK] Built recast wheel: {wheel}")
    return wheel


def install_recast(skip_build_if_cached: bool = False) -> Path:
    print("============================================================")
    print("   Recast navmesh bindings installer")
    print("============================================================")
    wheel = find_compatible_recast_wheel() if skip_build_if_cached else None
    if wheel is None:
        wheel = build_recast_wheel()
    else:
        print(f"[INFO] Using cached recast wheel: {wheel}")
    try:
        run_command([sys.executable, "-m", "pip", "uninstall", "-y", "recast"])
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
        verify_recast()
    except Exception as exc:
        if not skip_build_if_cached:
            raise
        print(f"[WARN] Cached recast wheel install/verify failed: {type(exc).__name__}: {exc}")
        print("[INFO] Falling back to a source build for this Python/platform.")
        wheel = build_recast_wheel()
        run_command([sys.executable, "-m", "pip", "uninstall", "-y", "recast"])
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
        verify_recast()
    return wheel


def install_pytorch3d(skip_build_if_cached: bool = False) -> Path:
    local_label = pytorch3d_build_label()
    print(f"[INFO] PyTorch3D wheel local version label: {local_label}")
    wheel = find_compatible_pytorch3d_wheel(local_label) if skip_build_if_cached else None
    if wheel is None:
        wheel = build_pytorch3d_wheel(local_label)
    else:
        print(f"[INFO] Using cached PyTorch3D wheel: {wheel}")
        patch_pytorch3d_wheel_nopulsar(wheel)
    run_command([sys.executable, "-m", "pip", "uninstall", "-y", "pytorch3d"])
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
    verify_pytorch3d()
    return wheel


def install_fused_ssim(skip_build_if_cached: bool = False) -> Path:
    print("============================================================")
    print("   fused-ssim installer")
    print("============================================================")
    if CUDA_130_HOME.exists():
        print(f"[INFO] CUDA build root: {CUDA_130_HOME}")
    else:
        print(f"[WARN] CUDA 13.0 build root not found: {CUDA_130_HOME}")
    wheel = find_compatible_fused_ssim_wheel() if skip_build_if_cached else None
    if wheel is None:
        wheel = build_fused_ssim_wheel()
    else:
        print(f"[INFO] Using cached fused-ssim wheel: {wheel}")
    run_command([sys.executable, "-m", "pip", "uninstall", "-y", "fused-ssim"])
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
    verify_fused_ssim()
    return wheel


def install_hyworld_gsplat() -> Path:
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
    return wheel


def main() -> None:
    parser = argparse.ArgumentParser(description="Build native HY-World wheels.")
    parser.add_argument("--pytorch3d-only", action="store_true", help="Build/install PyTorch3D only; do not reinstall gsplat.")
    parser.add_argument("--gsplat-only", action="store_true", help="Build/install HY-World gsplat only.")
    parser.add_argument("--recast-only", action="store_true", help="Build/install Recast navmesh bindings only.")
    parser.add_argument("--fused-ssim-only", action="store_true", help="Build/install fused-ssim only.")
    parser.add_argument("--skip-recast-build-if-cached", action="store_true", help="Reuse a matching recast wheel if it already exists.")
    parser.add_argument("--skip-fused-ssim-build-if-cached", action="store_true", help="Reuse a matching fused-ssim wheel if it already exists.")
    parser.add_argument("--skip-pytorch3d-build-if-cached", action="store_true", help="Reuse a matching PyTorch3D wheel if it already exists.")
    args = parser.parse_args()

    selected_only = [args.pytorch3d_only, args.gsplat_only, args.recast_only, args.fused_ssim_only]
    if sum(bool(item) for item in selected_only) > 1:
        raise ValueError("--pytorch3d-only, --gsplat-only, --recast-only, and --fused-ssim-only are mutually exclusive")

    if args.pytorch3d_only:
        print("============================================================")
        print("   PyTorch3D builder (gsplat reinstall disabled)")
        print("============================================================")
        install_pytorch3d(skip_build_if_cached=args.skip_pytorch3d_build_if_cached)
        return

    if args.gsplat_only:
        install_hyworld_gsplat()
        return

    if args.recast_only:
        install_recast(skip_build_if_cached=args.skip_recast_build_if_cached)
        return

    if args.fused_ssim_only:
        install_fused_ssim(skip_build_if_cached=args.skip_fused_ssim_build_if_cached)
        return

    install_hyworld_gsplat()
    install_fused_ssim(skip_build_if_cached=args.skip_fused_ssim_build_if_cached)
    install_recast(skip_build_if_cached=True)
    install_pytorch3d(skip_build_if_cached=args.skip_pytorch3d_build_if_cached)


if __name__ == "__main__":
    main()
