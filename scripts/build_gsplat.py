import os
import subprocess
import sys
import shutil
import torch
import stat
import tempfile
from pathlib import Path

def on_rm_error(func, path, exc_info):
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    else:
        raise

def _print_log_hint(log_path: Path):
    print(f"[INFO] Full command log saved to: {log_path}")


def run_command(cmd, cwd=None, env=None, check=True, log_path: Path | None = None):
    print(f"[RUN] {cmd}")
    sys.stdout.flush()
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            process = subprocess.Popen(
                cmd,
                shell=True,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
            return_code = process.wait()
        _print_log_hint(log_path)
        if return_code != 0:
            print(f"[ERROR] Command failed with error code {return_code}")
            print(f"[ERROR] Inspect the full log above, especially the first 'FAILED:' block: {log_path}")
            if check:
                sys.exit(1)
        return return_code
    try:
        if check:
            subprocess.check_call(cmd, shell=True, cwd=cwd, env=env)
            return 0
        else:
            return subprocess.call(cmd, shell=True, cwd=cwd, env=env)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with error code {e.returncode}")
        if check:
            sys.exit(1)
        return e.returncode

def get_cuda_version():
    return torch.version.cuda


def _cuda_version_tag(cuda_ver: str):
    parts = (cuda_ver or "").split(".")
    if len(parts) < 2:
        return None
    return f"v{parts[0]}.{parts[1]}"


def _cuda_env_var_name(cuda_ver: str):
    parts = (cuda_ver or "").split(".")
    if len(parts) < 2:
        return None
    return f"CUDA_PATH_V{parts[0]}_{parts[1]}"


def _nvcc_version_for_home(cuda_home: Path):
    nvcc = cuda_home / "bin" / "nvcc.exe"
    if not nvcc.exists():
        nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.exists():
        return None
    try:
        output = subprocess.check_output(
            f'"{nvcc}" --version',
            shell=True,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
    except Exception:
        return None
    import re
    match = re.search(r"release\s+(\d+\.\d+)", output)
    return match.group(1) if match else None


def find_matching_cuda_home(cuda_ver: str, script_dir: Path):
    wanted = _cuda_version_tag(cuda_ver)
    if not wanted:
        return None
    wanted_version = wanted[1:]

    candidates = []
    exact_env = _cuda_env_var_name(cuda_ver)
    if exact_env and os.environ.get(exact_env):
        candidates.append(Path(os.environ[exact_env]))

    candidates.extend([
        script_dir / "portable_cuda" / wanted,
        script_dir / "cuda" / wanted,
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA") / wanted,
    ])

    # Generic env vars are only acceptable if their nvcc version is exactly the
    # one PyTorch was built against. Never use "latest" just because it is first.
    for env_key in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(env_key)
        if value:
            candidates.append(Path(value))

    for candidate in candidates:
        if _nvcc_version_for_home(candidate) == wanted_version:
            return candidate.resolve()

    return None


def apply_cuda_home_to_env(env: dict, cuda_home: Path | None):
    if cuda_home is None:
        return env
    env = env.copy()
    env["CUDA_HOME"] = str(cuda_home)
    env["CUDA_PATH"] = str(cuda_home)
    bin_dir = str(cuda_home / "bin")
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    return env


def add_python_scripts_to_env(env: dict):
    env = env.copy()
    scripts_dir = Path(sys.executable).resolve().parent / "Scripts"
    if scripts_dir.exists():
        env["PATH"] = str(scripts_dir) + os.pathsep + env.get("PATH", "")
    return env


def nvcc_version_from_env(env=None):
    try:
        output = subprocess.check_output(
            "nvcc --version",
            shell=True,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            errors="replace",
        )
    except Exception:
        return None, None
    import re
    match = re.search(r"release\s+(\d+\.\d+)", output)
    return (match.group(1) if match else None), output


def check_compiler():
    nvcc_ok = False
    cl_ok = False
    try:
        subprocess.check_output("nvcc --version", shell=True, stderr=subprocess.STDOUT)
        nvcc_ok = True
    except: pass

    if shutil.which("cl.exe"):
        cl_ok = True
    else:
        try:
            subprocess.check_output("cl", shell=True, stderr=subprocess.STDOUT)
            cl_ok = True
        except: pass

    return nvcc_ok, cl_ok


def get_portable_msvc_activator(msvc_dir: Path):
    msvc_installed = msvc_dir / "MSVC"
    vcvars = list(msvc_installed.rglob("vcvars64.bat"))

    if not vcvars and (msvc_dir / "MSVC-Portable.bat").exists():
        print("[INFO] Initializing Portable MSVC...")
        subprocess.check_call(f'"{msvc_dir}/MSVC-Portable.bat"', shell=True, cwd=str(msvc_dir),
                              stdin=subprocess.DEVNULL)
        vcvars = list(msvc_installed.rglob("vcvars64.bat"))

    return vcvars[0] if vcvars else None


def run_with_optional_msvc(command: str, *, cwd=None, env=None, use_portable_msvc=False, msvc_dir: Path | None = None, check=True, log_path: Path | None = None):
    if use_portable_msvc:
        activator = get_portable_msvc_activator(msvc_dir)
        if activator is None:
            print("[ERROR] Compiler setup failed.")
            sys.exit(1)
        command = f'"{activator}" && {command}'
    return run_command(command, cwd=cwd, env=env, check=check, log_path=log_path)


GSPLAT_SMOKE_TEST = r'''
import torch
from gsplat.rendering import rasterization

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

device = torch.device("cuda")
means = torch.tensor([[0.0, 0.0, 2.0]], device=device)
quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
scales = torch.tensor([[0.05, 0.05, 0.05]], device=device)
opacities = torch.tensor([1.0], device=device)
colors = torch.tensor([[1.0, 0.0, 0.0]], device=device)
viewmats = torch.eye(4, device=device).unsqueeze(0)
Ks = torch.tensor([[[32.0, 0.0, 16.0], [0.0, 32.0, 16.0], [0.0, 0.0, 1.0]]], device=device)

rasterization(
    means=means,
    quats=quats,
    scales=scales,
    opacities=opacities,
    colors=colors,
    viewmats=viewmats,
    Ks=Ks,
    width=32,
    height=32,
    packed=False,
)
print("[OK] gsplat CUDA rasterization smoke test passed.")
'''


def gsplat_smoke_test(use_portable_msvc=False, msvc_dir: Path | None = None):
    with tempfile.NamedTemporaryFile("w", suffix="_gsplat_smoke.py", delete=False, encoding="utf-8") as fh:
        fh.write(GSPLAT_SMOKE_TEST)
        smoke_path = fh.name
    try:
        cmd = f'"{sys.executable}" "{smoke_path}"'
        return run_with_optional_msvc(
            cmd,
            use_portable_msvc=use_portable_msvc,
            msvc_dir=msvc_dir,
            check=False,
        ) == 0
    finally:
        try:
            os.remove(smoke_path)
        except OSError:
            pass


def install_pypi_gsplat(use_portable_msvc=False, msvc_dir: Path | None = None):
    print("\n[INFO] Trying official PyPI gsplat package (JIT builds CUDA on first run)...")
    if run_command(f"{sys.executable} -m pip install gsplat", check=False) != 0:
        return False
    return verify_install(use_portable_msvc=use_portable_msvc, msvc_dir=msvc_dir, require_smoke=True)


def verify_pytorch3d():
    try:
        import pytorch3d
        from pytorch3d.renderer import PointsRasterizer, PointsRenderer, look_at_rotation
        from pytorch3d.structures import Pointclouds
        assert PointsRasterizer and PointsRenderer and look_at_rotation and Pointclouds
        version = getattr(pytorch3d, "__version__", "unknown")
        print(f"[OK] pytorch3d {version} renderer imports are usable.")
        return True
    except Exception as e:
        print(f"[INFO] pytorch3d is not importable yet: {e}")
        return False


def _version_digits(version: str):
    return "".join(ch for ch in version.split("+")[0] if ch.isdigit())


def pytorch3d_wheel_label():
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    pt_tag = f"pt{_version_digits(torch.__version__)}"
    cuda_ver = get_cuda_version() or "cpu"
    cu_tag = "cpu" if cuda_ver == "cpu" else f"cu{''.join(cuda_ver.split('.')[:2])}"
    return f"{pt_tag}.{cu_tag}.{py_tag}.nopulsar"


def label_pytorch3d_wheel(wheel_path: Path, wheels_dir: Path):
    label = pytorch3d_wheel_label()
    stem = wheel_path.name[:-4]
    if label in stem:
        return wheel_path

    parts = stem.rsplit("-", 3)
    if len(parts) != 4:
        return wheel_path
    package_and_version, py_tag, abi_tag, platform_tag = parts
    name_parts = package_and_version.split("-", 1)
    if len(name_parts) != 2:
        return wheel_path
    package, version = name_parts
    version = version.split("+", 1)[0]
    labeled = wheels_dir / f"{package}-{version}+{label}-{py_tag}-{abi_tag}-{platform_tag}.whl"
    if labeled.exists():
        labeled.unlink()
    wheel_path.replace(labeled)
    print(f"[INFO] Labeled PyTorch3D wheel: {labeled.name}")
    return labeled


def clean_invalid_pytorch3d_wheels(wheels_dir: Path):
    for wheel_path in wheels_dir.glob("pytorch3d-*+pt*-cu*-cp*-nopulsar-*.whl"):
        print(f"[INFO] Removing invalid legacy PyTorch3D wheel filename: {wheel_path.name}")
        try:
            wheel_path.unlink()
        except OSError as e:
            print(f"[WARN] Could not remove invalid wheel {wheel_path}: {e}")


def is_compatible_pytorch3d_wheel(wheel_path: Path):
    name = wheel_path.name
    label = pytorch3d_wheel_label()
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if not name.startswith("pytorch3d-") or not name.endswith(".whl"):
        return False
    if f"+{label}-" not in name:
        return False
    if f"-{py_tag}-{py_tag}-" not in name:
        return False
    if "win_amd64" not in name and os.name == "nt":
        return False
    return True


def find_local_pytorch3d_wheels(script_dir: Path):
    repo_root = script_dir.parent
    search_dirs = [
        repo_root / "gsplat",
        script_dir / "wheels",
    ]
    candidates = []
    for directory in search_dirs:
        if directory.exists():
            candidates.extend(directory.glob("pytorch3d-*.whl"))
    return sorted(candidates, key=os.path.getmtime, reverse=True)


def install_local_pytorch3d_wheel(script_dir: Path, use_portable_msvc=False, msvc_dir: Path | None = None):
    for wheel_path in find_local_pytorch3d_wheels(script_dir):
        if not is_compatible_pytorch3d_wheel(wheel_path):
            print(f"[INFO] Skipping incompatible PyTorch3D wheel: {wheel_path.name}")
            continue
        print(f"[INFO] Trying local PyTorch3D wheel: {wheel_path}")
        cmd = f'{sys.executable} -m pip install --force-reinstall --no-deps "{wheel_path}"'
        if run_with_optional_msvc(
            cmd,
            use_portable_msvc=use_portable_msvc,
            msvc_dir=msvc_dir,
            check=False,
        ) == 0 and verify_pytorch3d():
            print("[OK] Installed compatible local PyTorch3D wheel.")
            return True
        print(f"[WARN] Local PyTorch3D wheel did not verify: {wheel_path.name}")
    return False


def patch_pytorch3d_source_for_cuda13(source_dir: Path):
    setup_path = source_dir / "setup.py"
    ext_path = source_dir / "pytorch3d" / "csrc" / "ext.cpp"
    points_init_path = source_dir / "pytorch3d" / "renderer" / "points" / "__init__.py"
    renderer_init_path = source_dir / "pytorch3d" / "renderer" / "__init__.py"
    if not setup_path.exists() or not ext_path.exists():
        raise FileNotFoundError(f"Unexpected PyTorch3D source layout: {source_dir}")

    setup_text = setup_path.read_text(encoding="utf-8")
    if "PYTORCH3D_DISABLE_PULSAR" not in setup_text:
        setup_text = setup_text.replace(
            '    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu"), recursive=True)\n',
            '    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu"), recursive=True)\n'
            '    disable_pulsar = os.getenv("PYTORCH3D_DISABLE_PULSAR", "0") == "1"\n'
            '    if disable_pulsar:\n'
            '        print("HYWorld2: building PyTorch3D with Pulsar disabled for CUDA 13 compatibility.")\n'
            '        is_pulsar = lambda s: "/pulsar/" in s.replace("\\\\", "/")\n'
            '        sources = [s for s in sources if not is_pulsar(s)]\n'
            '        source_cuda = [s for s in source_cuda if not is_pulsar(s)]\n',
        )
    if '("PYTORCH3D_DISABLE_PULSAR", None)' not in setup_text:
        setup_text = setup_text.replace(
            '        define_macros += [("WITH_CUDA", None)]\n',
            '        define_macros += [("WITH_CUDA", None)]\n'
            '        if disable_pulsar:\n'
            '            define_macros += [("PYTORCH3D_DISABLE_PULSAR", None)]\n',
        )
    if "PYTORCH3D_DISABLE_PULSAR" not in setup_text:
        raise RuntimeError("Could not patch PyTorch3D setup.py to disable Pulsar.")
    setup_path.write_text(setup_text, encoding="utf-8")

    ext_text = ext_path.read_text(encoding="utf-8")
    if "PYTORCH3D_DISABLE_PULSAR" not in ext_text:
        ext_text = ext_text.replace(
            "#if !defined(USE_ROCM)",
            "#if !defined(USE_ROCM) && !defined(PYTORCH3D_DISABLE_PULSAR)",
        )
        ext_path.write_text(ext_text, encoding="utf-8")

    if not points_init_path.exists() or not renderer_init_path.exists():
        raise FileNotFoundError(f"Unexpected PyTorch3D renderer layout: {source_dir}")

    points_init = points_init_path.read_text(encoding="utf-8")
    if "HYWorld2: Pulsar disabled" not in points_init:
        points_init = points_init.replace(
            "# Pulsar not enabled on amd.\nif not torch.version.hip:\n    from .pulsar.unified import PulsarPointsRenderer\n",
            "# HYWorld2: Pulsar disabled for CUDA 13 compatibility.\n",
        )
        points_init_path.write_text(points_init, encoding="utf-8")

    renderer_init = renderer_init_path.read_text(encoding="utf-8")
    if "HYWorld2: Pulsar disabled" not in renderer_init:
        renderer_init = renderer_init.replace(
            "# Pulsar is not enabled on amd.\nif not torch.version.hip:\n    from .points import PulsarPointsRenderer\n",
            "# HYWorld2: Pulsar disabled for CUDA 13 compatibility.\n",
        )
        renderer_init_path.write_text(renderer_init, encoding="utf-8")


def prepare_pytorch3d_source(script_dir: Path):
    source_dir = Path(
        os.environ.get(
            "HYWORLD2_PYTORCH3D_BUILD_DIR",
            str(Path(tempfile.gettempdir()) / "hyworld2_pytorch3d_build"),
        )
    )
    if not (source_dir / ".git").exists():
        if source_dir.exists():
            shutil.rmtree(source_dir, onerror=on_rm_error)
        print("[INFO] Cloning PyTorch3D source...")
        run_command(f'git clone --depth 1 --branch stable https://github.com/facebookresearch/pytorch3d.git "{source_dir}"')
    else:
        print("[INFO] PyTorch3D source cache found; refreshing stable branch...")
        run_command("git fetch --depth 1 origin stable", cwd=str(source_dir))
        run_command("git checkout -B hyworld2-stable FETCH_HEAD", cwd=str(source_dir))

    print(f"[INFO] PyTorch3D source/build directory: {source_dir}")
    patch_pytorch3d_source_for_cuda13(source_dir)
    return source_dir


def install_pytorch3d(use_portable_msvc=False, msvc_dir: Path | None = None, cuda_home: Path | None = None):
    print("\n==================================================")
    print("   PyTorch3D Installer")
    print("==================================================")

    if verify_pytorch3d():
        return

    script_dir = Path(__file__).parent.resolve()
    if install_local_pytorch3d_wheel(script_dir, use_portable_msvc=use_portable_msvc, msvc_dir=msvc_dir):
        return

    env = add_python_scripts_to_env(apply_cuda_home_to_env(os.environ.copy(), cuda_home))
    env.setdefault("FORCE_CUDA", "1")
    env.setdefault("PYTORCH3D_NO_NINJA", "0")
    env.setdefault("PYTORCH3D_DISABLE_PULSAR", "1")
    env.setdefault("MAX_JOBS", "4")
    torch_extensions_dir = script_dir / "torch_extensions"
    torch_extensions_dir.mkdir(exist_ok=True)
    env.setdefault("TORCH_EXTENSIONS_DIR", str(torch_extensions_dir))
    if cuda_home is not None and (cuda_home / "include" / "cub").exists():
        env.setdefault("CUB_HOME", str(cuda_home / "include"))
    existing_nvcc_flags = env.get("NVCC_FLAGS", "")
    if "-allow-unsupported-compiler" not in existing_nvcc_flags:
        env["NVCC_FLAGS"] = f"{existing_nvcc_flags} -allow-unsupported-compiler".strip()

    torch_cuda = get_cuda_version()
    nvcc_version, nvcc_output = nvcc_version_from_env(env=env)
    print(f"[INFO] PyTorch CUDA: {torch_cuda}; nvcc: {nvcc_version or 'not found'}")
    if torch_cuda and nvcc_version and not torch_cuda.startswith(nvcc_version):
        print("[WARN] Skipping PyTorch3D build because nvcc does not match PyTorch CUDA.")
        print("[WARN] Install a CUDA Toolkit matching PyTorch CUDA or place it under scripts/portable_cuda/vX.Y.")
        if nvcc_output:
            print(nvcc_output)
        return

    try:
        import ninja
    except:
        print("[INFO] Installing ninja build tool...")
        run_command(f"{sys.executable} -m pip install ninja")

    source_dir = prepare_pytorch3d_source(script_dir)
    wheels_dir = script_dir / "wheels"
    wheels_dir.mkdir(exist_ok=True)
    clean_invalid_pytorch3d_wheels(wheels_dir)
    log_path = script_dir / "logs" / "pytorch3d_build.log"
    print(f"[INFO] PyTorch3D build log will be written to: {log_path}")
    print(f"[INFO] PyTorch3D wheel will be saved to: {wheels_dir}")
    wheel_cmd = (
        f'{sys.executable} -m pip wheel '
        f'--no-build-isolation -v --no-deps '
        f'-w "{wheels_dir}" '
        f'"{source_dir}"'
    )
    run_with_optional_msvc(
        wheel_cmd,
        cwd=str(source_dir),
        env=env,
        use_portable_msvc=use_portable_msvc,
        msvc_dir=msvc_dir,
        log_path=log_path,
    )

    try:
        wheel_path = sorted(wheels_dir.glob("pytorch3d-*.whl"), key=os.path.getmtime)[-1]
    except IndexError:
        print(f"[ERROR] PyTorch3D build finished but no wheel was found in {wheels_dir}")
        sys.exit(1)
    wheel_path = label_pytorch3d_wheel(wheel_path, wheels_dir)

    print(f"[INFO] Installing cached PyTorch3D wheel: {wheel_path}")
    install_cmd = (
        f'{sys.executable} -m pip install '
        f'--force-reinstall --no-deps '
        f'"{wheel_path}"'
    )
    run_with_optional_msvc(
        install_cmd,
        env=env,
        use_portable_msvc=use_portable_msvc,
        msvc_dir=msvc_dir,
    )

    verify_pytorch3d()


def build_gsplat():
    print("\n==================================================")
    print("   Safe gsplat Installer (Surgical Mode)")
    print("==================================================")

    print(f"[OK] Python: {sys.version.split()[0]}")
    print(f"[OK] PyTorch: {torch.__version__}")

    cuda_ver = get_cuda_version()
    if not cuda_ver:
        print("[ERROR] CUDA not found! gsplat requires a CUDA-enabled PyTorch.")
        sys.exit(1)
    print(f"[OK] PyTorch CUDA: {cuda_ver}")

    print("\n[INFO] Checking existing gsplat installation before installing/building anything...")
    existing_gsplat_usable = verify_install(require_smoke=True)
    if existing_gsplat_usable:
        print("[OK] Existing gsplat installation is usable.")
        if verify_pytorch3d():
            print("[OK] Existing pytorch3d installation is usable.")
            return
        print("[INFO] pytorch3d is missing; compiler setup will be prepared only for PyTorch3D.")

    script_dir = Path(__file__).parent.resolve()
    repo_root = script_dir.parent
    msvc_dir = script_dir / "portable_msvc"
    use_portable_msvc = False
    cuda_home = find_matching_cuda_home(cuda_ver, script_dir)
    if cuda_home is not None:
        print(f"[INFO] Using CUDA toolkit matching PyTorch: {cuda_home}")
        os.environ.update(apply_cuda_home_to_env(os.environ.copy(), cuda_home))
    else:
        exact_env = _cuda_env_var_name(cuda_ver)
        wanted = _cuda_version_tag(cuda_ver)
        print(f"[ERROR] No CUDA toolkit matching PyTorch CUDA {cuda_ver} was found.")
        if exact_env:
            print(f"[ERROR] Expected env var like {exact_env}=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\{wanted}")
        print(f"[ERROR] Refusing to use CUDA_HOME/CUDA_PATH/latest nvcc because it may not match PyTorch.")
        sys.exit(1)

    nvcc_ok, cl_ok = check_compiler()
    if not cl_ok:
        if (msvc_dir / "MSVC").exists() or (msvc_dir / "MSVC-Portable.bat").exists():
            use_portable_msvc = True

    print(f"[INFO] Compiler check:")
    print(f"   - NVCC: {'[OK] Found' if nvcc_ok else '[MISSING]'}")
    print(f"   - CL:   {'[OK] Found (System)' if cl_ok else ('[OK] Found (Portable)' if use_portable_msvc else '[MISSING]')}")

    if not nvcc_ok:
        print("\n[ERROR] NVCC (CUDA Compiler) is missing.")
        print("Please install the CUDA Toolkit that matches your PyTorch version.")
        sys.exit(1)

    if not cl_ok and not use_portable_msvc:
        print("\n[INFO] MSVC Compiler missing. Attempting to download Portable MSVC (600MB)...")
        try:
            run_command(f"git clone https://github.com/Delphier/MSVC {msvc_dir}")
            use_portable_msvc = True
        except:
            print("[ERROR] Failed to download compiler. Please install Visual Studio Build Tools manually.")
            sys.exit(1)

    if existing_gsplat_usable:
        print("[INFO] Skipping gsplat install/build because the existing installation passed verification.")
        install_pytorch3d(use_portable_msvc=use_portable_msvc, msvc_dir=msvc_dir, cuda_home=cuda_home)
        return

    if install_pypi_gsplat(use_portable_msvc=use_portable_msvc, msvc_dir=msvc_dir):
        install_pytorch3d(use_portable_msvc=use_portable_msvc, msvc_dir=msvc_dir, cuda_home=cuda_home)
        return

    for whl in sorted((repo_root / "gsplat").glob("gsplat*.whl"), key=os.path.getmtime, reverse=True):
        print(f"[INFO] Found local wheel: {whl.name}")
        if run_command(f"{sys.executable} -m pip install {whl} --force-reinstall --no-deps", check=False) == 0:
            if verify_install(use_portable_msvc=use_portable_msvc, msvc_dir=msvc_dir, require_smoke=True):
                install_pytorch3d(use_portable_msvc=use_portable_msvc, msvc_dir=msvc_dir, cuda_home=cuda_home)
                return

    # Check for pre-built wheel
    print("\n[INFO] Checking for pre-built wheel...")
    cu_tag = f"cu{cuda_ver.replace('.', '')}"
    pt_ver = torch.__version__.split('+')[0].replace('.', '')
    pt_tag = f"pt{pt_ver[:2]}"

    wheel_urls = [
        f"https://docs.gsplat.studio/whl/{cu_tag}",
        f"https://docs.gsplat.studio/whl/{pt_tag}{cu_tag}",
        f"https://docs.gsplat.studio/whl/nightly/{cu_tag}",
    ]

    for index_url in wheel_urls:
        print(f"[INFO] Trying: {index_url}")
        cmd = f"{sys.executable} -m pip install gsplat --index-url {index_url} --no-deps"
        if run_command(cmd, check=False) == 0:
            print("[OK] Installed from official wheel!")
            if verify_install(use_portable_msvc=use_portable_msvc, msvc_dir=msvc_dir, require_smoke=True):
                install_pytorch3d(use_portable_msvc=use_portable_msvc, msvc_dir=msvc_dir, cuda_home=cuda_home)
                return

    print("[INFO] No pre-built wheel found. Proceeding to build from source...")

    # Clone source
    build_dir = script_dir / "gsplat_build"
    if not build_dir.exists():
        print(f"[INFO] Cloning gsplat source...")
        run_command(f"git clone --recursive https://github.com/nerfstudio-project/gsplat.git {build_dir}")
    else:
        print("[INFO] Source cache found.")

    # Build wheel
    print("\n[INFO] Compiling gsplat...")
    dist_dir = script_dir / "dist"
    dist_dir.mkdir(exist_ok=True)

    env = os.environ.copy()

    try: import ninja
    except:
        print("[INFO] Installing ninja build tool...")
        run_command(f"{sys.executable} -m pip install ninja")

    wheel_cmd = f"{sys.executable} -m pip wheel . -w {dist_dir} --verbose --no-build-isolation"

    if use_portable_msvc:
        activator = get_portable_msvc_activator(msvc_dir)
        if activator is None:
            print("[ERROR] Compiler setup failed.")
            sys.exit(1)
        full_cmd = f'"{activator}" && {wheel_cmd}'
    else:
        full_cmd = wheel_cmd

    run_command(full_cmd, cwd=str(build_dir), env=env)

    # Install wheel
    try:
        whl = sorted(list(dist_dir.glob("*.whl")), key=os.path.getmtime)[-1]
    except IndexError:
        print("[ERROR] Build failed to produce a .whl file.")
        sys.exit(1)

    print(f"\n[INFO] Installing {whl.name}...")
    install_cmd = f"{sys.executable} -m pip install {whl} --force-reinstall --no-deps"
    run_command(install_cmd)

    print("\n==================================================")
    print("[OK] SUCCESS")
    print("==================================================")
    if verify_install(use_portable_msvc=use_portable_msvc, msvc_dir=msvc_dir, require_smoke=True):
        install_pytorch3d(use_portable_msvc=use_portable_msvc, msvc_dir=msvc_dir, cuda_home=cuda_home)

def verify_install(use_portable_msvc=False, msvc_dir: Path | None = None, require_smoke=False):
    try:
        import gsplat
        print(f"[OK] gsplat {gsplat.__version__} is importable.")
        if require_smoke and not gsplat_smoke_test(use_portable_msvc=use_portable_msvc, msvc_dir=msvc_dir):
            print("[WARN] gsplat imports, but CUDA rasterization smoke test failed.")
            return False
        return True
    except Exception as e:
        print(f"[WARN] Installed but import failed: {e}")
        return False

if __name__ == "__main__":
    build_gsplat()
