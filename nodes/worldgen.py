import gc
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    import comfy.model_management as comfy_model_management
except ImportError:
    comfy_model_management = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORLDGEN_DIR = PROJECT_ROOT / "hyworld2" / "worldgen"
SAM3_REPO_ID = "MIUProject/sam3"
SAM3_LOCAL_DIRNAME = "MIUProject_sam3"
SH_C0 = 0.28209479177387814
WORLDSTEREO_LIGHT_SINGLE_MODELS = {
    "worldstereo-memory-dmd": "vnccs-worldstereo-memory-dmd-int4.safetensors",
    "worldstereo-camera": "vnccs-worldstereo-camera-light-int4.safetensors",
}


def _default_sam3_path() -> str:
    if folder_paths is not None:
        models_dir = Path(folder_paths.models_dir)
        local_path = models_dir / "sam3" / SAM3_LOCAL_DIRNAME
        if local_path.exists():
            return str(local_path)
    return SAM3_REPO_ID


def _output_root() -> Path:
    if folder_paths is not None:
        return Path(folder_paths.get_output_directory())
    return PROJECT_ROOT / "output"


def _models_root() -> Path:
    if folder_paths is not None:
        return Path(folder_paths.models_dir)
    return PROJECT_ROOT / "models"


def _release_comfy_models_for_external_process():
    if comfy_model_management is None:
        return
    print("[WorldGen] Releasing ComfyUI models before external WorldStereo process...")
    comfy_model_management.unload_all_models()
    comfy_model_management.cleanup_models_gc()
    comfy_model_management.soft_empty_cache(force=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _sanitize_name(value, fallback="scene"):
    base = os.path.basename(str(value or fallback).replace("\\", "/"))
    base = os.path.splitext(base)[0]
    base = re.sub(r"[^A-Za-z0-9._ -]+", "_", base).strip(" ._")
    return base or fallback


def _ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _path_is_relative_to(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def _reset_dir(path, label="directory"):
    path = Path(path)
    resolved = path.resolve()
    if resolved == resolved.anchor:
        raise ValueError(f"Refusing to reset drive/root {label}: {resolved}")
    protected = {PROJECT_ROOT.resolve(), WORLDGEN_DIR.resolve(), Path.home().resolve()}
    if resolved in protected:
        raise ValueError(f"Refusing to reset protected {label}: {resolved}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _as_cpu_float_tensor(value):
    if not isinstance(value, torch.Tensor):
        return None
    return value.detach().cpu().float()


def _normalize_image_tensor(images):
    images = _as_cpu_float_tensor(images)
    if images is None:
        return None
    if images.dim() == 5 and images.shape[0] == 1 and images.shape[2] in (1, 3, 4):
        images = images[0].permute(0, 2, 3, 1)
    elif images.dim() == 4 and images.shape[1] in (1, 3, 4) and images.shape[-1] not in (1, 3, 4):
        images = images.permute(0, 2, 3, 1)
    if images.dim() != 4:
        return None
    if images.shape[-1] == 1:
        images = images.repeat(1, 1, 1, 3)
    return images[..., :3].clamp(0.0, 1.0).contiguous()


def _normalize_pose_tensor(camera_poses):
    poses = _as_cpu_float_tensor(camera_poses)
    if poses is None:
        return None
    if poses.dim() == 4 and poses.shape[0] == 1:
        poses = poses[0]
    if poses.dim() == 3 and poses.shape[-2:] == (4, 4):
        return poses
    return None


def _normalize_intrinsics_tensor(camera_intrinsics):
    intrs = _as_cpu_float_tensor(camera_intrinsics)
    if intrs is None:
        return None
    if intrs.dim() == 4 and intrs.shape[0] == 1:
        intrs = intrs[0]
    if intrs.dim() == 3 and intrs.shape[-2:] == (3, 3):
        return intrs
    return None


def _camera_translation_std(poses):
    poses = _normalize_pose_tensor(poses)
    if poses is None or poses.shape[0] == 0:
        return 0.0
    return float(poses[:, :3, 3].std().item())


def _scale_intrinsic_for_image(intrinsic, source_h, source_w, image_h, image_w):
    k = intrinsic.detach().cpu().float().clone()
    if source_h is None or source_w is None:
        return k
    source_h = float(source_h)
    source_w = float(source_w)
    if source_h <= 0 or source_w <= 0:
        return k
    sx = float(image_w) / source_w
    sy = float(image_h) / source_h
    if abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6:
        return k
    k[0, 0] *= sx
    k[0, 2] *= sx
    k[1, 1] *= sy
    k[1, 2] *= sy
    return k


def _normalize_points_tensor(points):
    points = _as_cpu_float_tensor(points)
    if points is None:
        return None
    if points.dim() == 5 and points.shape[0] == 1:
        points = points[0]
    if points.dim() == 4 and points.shape[-1] == 3:
        return points
    if points.dim() == 2 and points.shape[-1] == 3:
        return points
    return None


def _points_and_colors_from_ply_data(ply_data, images=None, prefer_filtered=False, max_points=0):
    if not isinstance(ply_data, dict):
        raise ValueError("ply_data must be a PLY_DATA dictionary.")

    source = ply_data.get("pts3d_filtered") if prefer_filtered else None
    points = _normalize_points_tensor(source)
    if points is None:
        points = _normalize_points_tensor(ply_data.get("pts3d"))
    if points is None:
        splats = ply_data.get("splats")
        if isinstance(splats, dict):
            means = splats.get("means")
            if isinstance(means, list):
                means = means[0]
            points = _normalize_points_tensor(means)
    if points is None:
        raise ValueError("PLY_DATA has no pts3d/pts3d_filtered/splats.means points.")

    image_tensor = _normalize_image_tensor(images)
    if image_tensor is None:
        image_tensor = _normalize_image_tensor(ply_data.get("images"))

    if points.dim() == 4:
        flat_points = points.reshape(-1, 3)
        if image_tensor is not None and image_tensor.shape[0] == points.shape[0]:
            if image_tensor.shape[1:3] != points.shape[1:3]:
                resized = [
                    _resize_hwc_tensor(image_tensor[i], int(points.shape[1]), int(points.shape[2]))
                    for i in range(points.shape[0])
                ]
                image_tensor = torch.stack(resized, dim=0)
            flat_colors = image_tensor.reshape(-1, 3)
        else:
            flat_colors = torch.full_like(flat_points, 0.5)
    else:
        flat_points = points.reshape(-1, 3)
        flat_colors = torch.full_like(flat_points, 0.5)

    finite = torch.isfinite(flat_points).all(dim=1)
    flat_points = flat_points[finite]
    flat_colors = flat_colors[finite]

    if max_points and max_points > 0 and flat_points.shape[0] > max_points:
        generator = torch.Generator().manual_seed(42)
        idx = torch.randperm(flat_points.shape[0], generator=generator)[:max_points]
        flat_points = flat_points[idx]
        flat_colors = flat_colors[idx]

    colors_u8 = (flat_colors.clamp(0.0, 1.0).numpy() * 255.0 + 0.5).astype(np.uint8)
    return flat_points.numpy().astype(np.float32), colors_u8


def _write_point_ply(path, points, colors):
    path = Path(path)
    _ensure_dir(path.parent)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    n = min(points.shape[0], colors.shape[0])
    points, colors = points[:n], colors[:n]

    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vertices = np.empty(n, dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertices["red"], vertices["green"], vertices["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as handle:
        handle.write(header)
        vertices.tofile(handle)
    os.replace(tmp_path, path)
    return str(path)


def _save_rgb_image(path, image):
    arr = (image.clamp(0.0, 1.0).numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _save_depth16(path, depth):
    depth_np = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    depth_u16 = depth_np.astype(np.float16).view(np.uint16)
    Image.fromarray(depth_u16).save(path)


def _resize_2d_tensor(value, height, width, mode="bilinear"):
    if value.shape[-2:] == (height, width):
        return value
    interp_kwargs = {"mode": mode}
    if mode in {"bilinear", "bicubic"}:
        interp_kwargs["align_corners"] = False
    return F.interpolate(value[None, None].float(), size=(height, width), **interp_kwargs)[0, 0]


def _resize_hwc_tensor(value, height, width, mode="bilinear"):
    if value.shape[:2] == (height, width):
        return value
    interp_kwargs = {"mode": mode}
    if mode in {"bilinear", "bicubic"}:
        interp_kwargs["align_corners"] = False
    chw = value.permute(2, 0, 1)[None].float()
    resized = F.interpolate(chw, size=(height, width), **interp_kwargs)[0]
    return resized.permute(1, 2, 0)


def _depths_from_points(points, poses):
    points = _normalize_points_tensor(points)
    poses = _normalize_pose_tensor(poses)
    if points is None or poses is None or points.dim() != 4:
        return None
    if points.shape[0] != poses.shape[0]:
        return None
    depths = []
    for i in range(points.shape[0]):
        pts = points[i].reshape(-1, 3)
        c2w = poses[i]
        w2c = torch.linalg.inv(c2w)
        homog = torch.cat([pts, torch.ones((pts.shape[0], 1), dtype=pts.dtype)], dim=1)
        cam = (homog @ w2c.T)[:, :3]
        depth = cam[:, 2].reshape(points.shape[1], points.shape[2])
        depths.append(depth.clamp_min(0.0))
    return torch.stack(depths, dim=0)


def _depth_maps_to_metric_tensor(depth_maps):
    depths = _as_cpu_float_tensor(depth_maps)
    if depths is None:
        return None
    if depths.dim() == 5 and depths.shape[0] == 1 and depths.shape[-1] == 1:
        depths = depths[0, ..., 0]
    elif depths.dim() == 4 and depths.shape[-1] == 1:
        depths = depths[..., 0]
    elif depths.dim() == 4 and depths.shape[0] == 1 and depths.shape[-1] not in (1, 3, 4):
        depths = depths[0]
    elif depths.dim() == 4 and depths.shape[-1] in (3, 4):
        depths = depths[..., 0]
    if depths.dim() != 3:
        return None
    depths = torch.nan_to_num(depths.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if not torch.isfinite(depths).all() or float(depths.max().item()) <= 0.0:
        return None
    return depths.contiguous()


def _normal_maps_to_tensor(normal_maps):
    normals = _normalize_image_tensor(normal_maps)
    if normals is None:
        return None
    return normals.clamp(0.0, 1.0)


def _normal_maps_to_encoded_tensor(normal_maps):
    normals = _as_cpu_float_tensor(normal_maps)
    if normals is None:
        return None
    if normals.dim() == 5 and normals.shape[0] == 1:
        normals = normals[0]
    if normals.dim() == 4 and normals.shape[1] == 3 and normals.shape[-1] != 3:
        normals = normals.permute(0, 2, 3, 1)
    if normals.dim() != 4 or normals.shape[-1] != 3:
        return None
    if float(normals.min().item()) < -0.05:
        normals = (normals.clamp(-1.0, 1.0) + 1.0) * 0.5
    return normals[..., :3].clamp(0.0, 1.0).contiguous()


def _raw_prediction_dict(raw_splats):
    return raw_splats if isinstance(raw_splats, dict) else {}


def _normal_tensor_is_usable(normals):
    if normals is None or normals.numel() == 0:
        return False
    sample = normals.float()
    if not torch.isfinite(sample).all():
        return False
    raw_min = float(sample.min().item())
    raw_max = float(sample.max().item())
    raw_std = float(sample.std().item())
    if raw_max <= 0.02 or raw_std <= 0.005:
        return False
    decoded = sample * 2.0 - 1.0
    lengths = decoded.norm(dim=-1)
    valid_ratio = float(((lengths > 0.25) & (lengths < 1.75)).float().mean().item())
    return raw_max > raw_min and valid_ratio > 0.05


def _clear_png_dir(path):
    path = Path(path)
    if not path.exists():
        return
    for png_path in path.glob("*.png"):
        try:
            png_path.unlink()
        except OSError:
            pass


def _deep_log(enabled, message):
    if enabled:
        print(f"[WorldGen][deep] {message}")


def _run_command(command, cwd, env=None, deep_logging=False, log_path=None):
    process_env = os.environ.copy()
    process_env["USE_LIBUV"] = "0"
    process_env.setdefault("PYTHONIOENCODING", "utf-8")
    process_env.setdefault("PYTHONUTF8", "1")
    process_env.setdefault("MASTER_ADDR", "127.0.0.1")
    process_env.setdefault("MASTER_PORT", "29500")
    process_env.setdefault("RANK", "0")
    process_env.setdefault("WORLD_SIZE", "1")
    process_env.setdefault("LOCAL_RANK", "0")
    process_env.setdefault("COMFYUI_MODELS_DIR", str(_models_root()))
    if env:
        process_env.update(env)
    process_env["PYTHONPATH"] = os.pathsep.join([
        str(WORLDGEN_DIR),
        str(PROJECT_ROOT),
        process_env.get("PYTHONPATH", ""),
    ])
    print(f"[WorldGen] Running: {' '.join(map(str, command))}")
    _deep_log(deep_logging, f"cwd={cwd}")
    _deep_log(deep_logging, f"PROJECT_ROOT={PROJECT_ROOT}")
    _deep_log(deep_logging, f"WORLDGEN_DIR={WORLDGEN_DIR}")
    _deep_log(deep_logging, f"PYTHONPATH={process_env.get('PYTHONPATH', '')}")
    log_handle = None
    if log_path:
        log_path = Path(log_path)
        _ensure_dir(log_path.parent)
        log_handle = open(log_path, "w", encoding="utf-8", errors="replace")
        log_handle.write(f"[WorldGen] command={' '.join(map(str, command))}\n")
        log_handle.write(f"[WorldGen] cwd={cwd}\n")
        log_handle.write(f"[WorldGen] PROJECT_ROOT={PROJECT_ROOT}\n")
        log_handle.write(f"[WorldGen] WORLDGEN_DIR={WORLDGEN_DIR}\n")
        log_handle.write(f"[WorldGen] PYTHONPATH={process_env.get('PYTHONPATH', '')}\n")
        log_handle.flush()
    process = subprocess.Popen(
        [str(x) for x in command],
        cwd=str(cwd),
        env=process_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    lines = []
    assert process.stdout is not None
    try:
        for line in process.stdout:
            line = line.rstrip()
            print(f"[WorldGen] {line}")
            if log_handle is not None:
                log_handle.write(f"{line}\n")
                log_handle.flush()
            lines.append(line)
            max_lines = 2000 if deep_logging else 500
            if len(lines) > max_lines:
                lines = lines[-max_lines:]
        rc = process.wait()
    finally:
        if log_handle is not None:
            log_handle.close()
    if rc != 0:
        log_note = f"\nFull log: {log_path}" if log_path else ""
        raise RuntimeError(f"WorldGen command failed with exit code {rc}.{log_note}\n" + "\n".join(lines[-80:]))
    result = "\n".join(lines if deep_logging else lines[-120:])
    if log_path:
        result = f"[WorldGen] Full log saved: {log_path}\n{result}"
    return result


def _official_strategy_args(max_steps):
    refine_stop = max(1, min(750, int(max_steps) - 1))
    return [
        "--strategy.refine-start-iter", "150",
        "--strategy.refine-stop-iter", str(refine_stop),
        "--strategy.refine-every", "100",
        "--strategy.refine-scale2d-stop-iter", str(refine_stop),
        "--strategy.reset-every", "99990",
        "--strategy.grow-grad2d", "0.0001",
        "--strategy.prune-scale3d", "0.1",
    ]


def _find_latest_ply(result_dir):
    result_dir = Path(result_dir)
    candidates = []
    preferred = result_dir / "ply"
    if preferred.exists():
        candidates.extend(preferred.glob("*.ply"))
    candidates.extend(path for path in result_dir.rglob("*.ply") if path not in candidates)
    if not candidates:
        return ""
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return str(candidates[0])


def _find_latest_trainer_cameras(result_dir):
    result_dir = Path(result_dir)
    candidates = []
    preferred = result_dir / "ply"
    if preferred.exists():
        candidates.extend(preferred.glob("trainer_cameras_*.json"))
        generic = preferred / "trainer_cameras.json"
        if generic.exists():
            candidates.append(generic)
    candidates.extend(path for path in result_dir.rglob("trainer_cameras*.json") if path not in candidates)
    if not candidates:
        return ""
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return str(candidates[0])


def _load_camera_tensors_from_json(path):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if "cameras" in data:
        cameras = data["cameras"]
        order = data.get("camera_order") or sorted(cameras.keys())
        poses = []
        intrs = []
        for camera_id in order:
            entry = cameras[str(camera_id)]
            if "camera_pose" in entry:
                c2w = np.asarray(entry["camera_pose"], dtype=np.float32)
            else:
                c2w = np.linalg.inv(np.asarray(entry["extrinsic"], dtype=np.float32))
            poses.append(c2w)
            intrs.append(np.asarray(entry["intrinsic"], dtype=np.float32))
    else:
        poses = []
        intrs = []
        for camera_id in sorted(data.keys()):
            entry = data[camera_id]
            if not isinstance(entry, dict) or "extrinsic" not in entry or "intrinsic" not in entry:
                continue
            poses.append(np.linalg.inv(np.asarray(entry["extrinsic"], dtype=np.float32)))
            intrs.append(np.asarray(entry["intrinsic"], dtype=np.float32))
    if not poses or not intrs:
        raise ValueError(f"No cameras found in {path}")
    return torch.from_numpy(np.stack(poses)).float(), torch.from_numpy(np.stack(intrs)).float()


def _load_train_camera_tensors(result_dir, data_dir):
    trainer_cameras = _find_latest_trainer_cameras(result_dir)
    if trainer_cameras:
        return (*_load_camera_tensors_from_json(trainer_cameras), trainer_cameras)
    gs_cameras = Path(data_dir) / "cameras.json"
    if gs_cameras.exists():
        return (*_load_camera_tensors_from_json(gs_cameras), str(gs_cameras))
    return torch.empty((0, 4, 4), dtype=torch.float32), torch.empty((0, 3, 3), dtype=torch.float32), ""


def _gs_data_camera_translation_std(data_dir):
    cameras_path = Path(data_dir) / "cameras.json"
    if not cameras_path.exists():
        return None
    poses, _intrs = _load_camera_tensors_from_json(cameras_path)
    return _camera_translation_std(poses)


def _ensure_scene_type_meta(data_dir, scene_type="unknown"):
    data_dir = Path(data_dir)
    candidates = [data_dir.parent / "meta_info.json", data_dir / "meta_info.json"]
    target = None
    meta = {}
    for path in candidates:
        if path.exists():
            target = path
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, dict):
                    meta = loaded
            except Exception:
                meta = {}
            break
    if target is None:
        target = data_dir / "meta_info.json"
    if not meta.get("scene_type"):
        meta["scene_type"] = scene_type
        _ensure_dir(target.parent)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
    return str(target)


def _has_valid_normal_files(data_dir, max_files=3):
    normals_dir = Path(data_dir) / "normals"
    if not normals_dir.exists():
        return False
    normal_files = sorted(normals_dir.glob("*.png"))[:max_files]
    if not normal_files:
        return False
    for path in normal_files:
        try:
            arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        except Exception:
            continue
        tensor = torch.from_numpy(arr)
        if _normal_tensor_is_usable(tensor):
            return True
    return False


def _load_metric_depth16_preview(path):
    with Image.open(path) as depth_pil:
        arr = np.asarray(depth_pil)
    if arr.ndim != 2 or arr.dtype.itemsize < 2:
        return None
    depth = np.frombuffer(arr.astype(np.uint16, copy=False), dtype=np.float16).astype(np.float32)
    depth = depth.reshape(arr.shape)
    return depth


def _has_valid_depth_files(data_dir, max_files=3):
    depths_dir = Path(data_dir) / "depths"
    if not depths_dir.exists():
        return False
    depth_files = sorted(depths_dir.glob("*.png"))[:max_files]
    if not depth_files:
        return False
    for path in depth_files:
        try:
            depth = _load_metric_depth16_preview(path)
        except Exception:
            continue
        if depth is None:
            continue
        finite = np.isfinite(depth)
        if float(finite.mean()) < 0.999:
            continue
        valid = depth[finite & (depth > 1e-4)]
        if valid.size == 0:
            continue
        vmax = float(valid.max())
        if vmax > 1e-3 and vmax < 1e6:
            return True
    return False


def _shell_split(value, boolean_flags=None):
    if isinstance(value, bool):
        return []
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"false", "none", "null"}:
        return []
    tokens = shlex.split(raw, posix=os.name != "nt")
    boolean_flags = set(boolean_flags or ())
    normalized = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.lower() in {"false", "none", "null"}:
            i += 1
            continue
        if token in boolean_flags and i + 1 < len(tokens):
            next_token = tokens[i + 1].lower()
            if next_token in {"false", "0", "no", "off"}:
                i += 2
                continue
            if next_token in {"true", "1", "yes", "on"}:
                normalized.append(token)
                i += 2
                continue
        normalized.append(token)
        i += 1
    return normalized


def _torchrun_command(script_name, nproc_per_node):
    if int(nproc_per_node) <= 1:
        return _script_command(script_name)
    return [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node", str(int(nproc_per_node)),
        str(WORLDGEN_DIR / script_name),
    ]


def _script_command(script_name):
    script_path = WORLDGEN_DIR / script_name
    bootstrap = (
        "import runpy, sys; "
        f"sys.path.insert(0, {str(WORLDGEN_DIR)!r}); "
        f"runpy.run_path({str(script_path)!r}, run_name='__main__')"
    )
    return [sys.executable, "-c", bootstrap]


def _scene_list_from_scene_dir(scene_dir):
    scene = Path(scene_dir)
    if not scene.exists():
        raise FileNotFoundError(f"scene_dir not found: {scene}")
    if (scene / "panorama.png").exists():
        return [scene]
    return sorted(path for path in scene.iterdir() if path.is_dir() and (path / "panorama.png").exists())


def _write_manual_traj_prompts(scene_dir, prompt):
    prompt = str(prompt or "").strip()
    if not prompt:
        return 0
    written = 0
    for scene in _scene_list_from_scene_dir(scene_dir):
        render_root = scene / "render_results"
        if not render_root.exists():
            continue
        for render_path in render_root.glob("*/traj*/render.mp4"):
            caption_path = render_path.with_name("traj_caption.json")
            if caption_path.exists():
                continue
            with open(caption_path, "w", encoding="utf-8") as handle:
                json.dump({"prompt": prompt}, handle, indent=2)
            written += 1
        prompt_path = render_root / "prompt.json"
        if not prompt_path.exists():
            with open(prompt_path, "w", encoding="utf-8") as handle:
                json.dump({"prompt": prompt}, handle, indent=2)
    return written


def _missing_traj_caption_paths(scene_dir):
    missing = []
    for scene in _scene_list_from_scene_dir(scene_dir):
        render_root = scene / "render_results"
        if not render_root.exists():
            continue
        for render_path in render_root.glob("*/traj*/render.mp4"):
            caption_path = render_path.with_name("traj_caption.json")
            if not caption_path.exists():
                missing.append(caption_path)
    return missing


def _worldstereo_cli_args(worldstereo_model, model_type):
    if not isinstance(worldstereo_model, dict):
        args = []
        models_root = _models_root()
        pretrained_path = models_root / "WorldStereo"
        if pretrained_path.exists():
            args.extend(["--pretrained_path", str(pretrained_path).replace("\\", "/")])
            args.append("--local_files_only")
        single_model_name = WORLDSTEREO_LIGHT_SINGLE_MODELS.get(model_type)
        if single_model_name:
            single_model_path = models_root / "WorldStereoLight" / single_model_name
            if single_model_path.exists():
                args.extend(["--single_model_path", str(single_model_path).replace("\\", "/")])
        moge_dir = models_root / "MoGe"
        if moge_dir.exists():
            args.extend(["--moge_path", str(moge_dir).replace("\\", "/")])
        return model_type, args
    resolved_model_type = worldstereo_model.get("model_type") or model_type
    args = []
    pretrained_path = worldstereo_model.get("pretrained_path")
    if pretrained_path:
        args.extend(["--pretrained_path", str(pretrained_path).replace("\\", "/")])
        args.append("--local_files_only")
    single_model_path = worldstereo_model.get("single_model_path")
    if single_model_path:
        args.extend(["--single_model_path", str(single_model_path).replace("\\", "/")])
    moge_dir = worldstereo_model.get("moge_dir")
    if moge_dir:
        args.extend(["--moge_path", str(moge_dir).replace("\\", "/")])
    if resolved_model_type not in ("worldstereo-memory", "worldstereo-memory-dmd"):
        raise ValueError(
            f"video_gen.py supports worldstereo-memory/worldstereo-memory-dmd, got {resolved_model_type!r}. "
            "Use Load WorldStereo Model with a memory model type for WorldGen WorldStereo Video."
        )
    return resolved_model_type, args


class VNCCS_WorldGenPreparePanoramaScene:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panorama": ("IMAGE",),
                "workspace_name": ("STRING", {"default": "comfy_worldgen"}),
            },
            "optional": {
                "root_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Base output folder. Empty uses ComfyUI/output/hyworld2_worldgen.",
                }),
                "scene_type": (["unknown", "indoor", "outdoor"], {"default": "unknown"}),
                "overwrite": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("scene_dir", "info")
    FUNCTION = "prepare_scene"
    CATEGORY = "VNCCS/WorldGen"
    OUTPUT_NODE = True

    def prepare_scene(self, panorama, workspace_name="comfy_worldgen", root_dir="", scene_type="unknown", overwrite=True):
        root = Path(root_dir) if str(root_dir).strip() else _output_root() / "hyworld2_worldgen"
        scene_dir = root / _sanitize_name(workspace_name)
        if overwrite:
            _reset_dir(scene_dir, "WorldGen panorama scene")
        else:
            _ensure_dir(scene_dir)
        image_tensor = _normalize_image_tensor(panorama)
        if image_tensor is None or image_tensor.shape[0] < 1:
            raise ValueError("panorama must be a ComfyUI IMAGE tensor.")
        pano = image_tensor[0]
        _save_rgb_image(scene_dir / "panorama.png", pano)
        meta = {"scene_type": None if scene_type == "unknown" else scene_type}
        if meta["scene_type"] is not None:
            with open(scene_dir / "meta_info.json", "w", encoding="utf-8") as handle:
                json.dump(meta, handle, indent=2)
        info = {
            "scene_dir": str(scene_dir),
            "panorama_path": str(scene_dir / "panorama.png"),
            "scene_type": meta["scene_type"] or "auto/vlm",
            "image_shape": list(image_tensor.shape),
        }
        return (str(scene_dir), json.dumps(info, indent=2))


class VNCCS_WorldGenGenerateTrajectories:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_dir": ("STRING", {"default": ""}),
            },
            "optional": {
                "fov_x": ("FLOAT", {"default": 120.0, "min": 1.0, "max": 179.0, "step": 1.0}),
                "fov_y": ("FLOAT", {"default": 90.0, "min": 1.0, "max": 179.0, "step": 1.0}),
                "seed": ("INT", {"default": 1024, "min": 0, "max": 2**31 - 1, "step": 1}),
                "split_view_num": ("INT", {"default": 3, "min": 1, "max": 16, "step": 1}),
                "splitted_resolution": ("INT", {"default": 480, "min": 64, "max": 4096, "step": 16}),
                "nframe": ("INT", {"default": 21, "min": 2, "max": 257, "step": 1}),
                "apply_nav_traj": ("BOOLEAN", {"default": True}),
                "apply_up_route": ("BOOLEAN", {"default": True}),
                "apply_recon_iteration": ("BOOLEAN", {"default": True}),
                "force_vlm": ("BOOLEAN", {"default": False}),
                "skip_exist": ("BOOLEAN", {"default": False}),
                "llm_addr": ("STRING", {"default": "localhost"}),
                "llm_port": ("INT", {"default": 8000, "min": 1, "max": 65535, "step": 1}),
                "llm_name": ("STRING", {"default": "Qwen/Qwen3-VL-8B-Instruct"}),
                "sam3_path": ("STRING", {
                    "default": _default_sam3_path(),
                    "tooltip": "SAM3 repo id or local checkpoint path.",
                }),
                "local_files_only": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use local Hugging Face cache/paths only. Enable when sam3_path points to a local checkpoint.",
                }),
                "extra_args": ("STRING", {"default": "", "multiline": False}),
                "deep_logging": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("scene_dir", "log")
    FUNCTION = "generate"
    CATEGORY = "VNCCS/WorldGen"
    OUTPUT_NODE = True

    def generate(
        self,
        scene_dir,
        fov_x=120.0,
        fov_y=90.0,
        seed=1024,
        split_view_num=3,
        splitted_resolution=480,
        nframe=21,
        apply_nav_traj=True,
        apply_up_route=True,
        apply_recon_iteration=True,
        force_vlm=False,
        skip_exist=False,
        llm_addr="localhost",
        llm_port=8000,
        llm_name="Qwen/Qwen3-VL-8B-Instruct",
        sam3_path=None,
        local_files_only=False,
        extra_args="",
        deep_logging=False,
    ):
        scene = Path(scene_dir)
        if not scene.exists():
            raise FileNotFoundError(f"scene_dir not found: {scene}")
        cmd = [
            *_script_command("traj_generate.py"),
            "--target_path", str(scene),
            "--fov_x", str(float(fov_x)),
            "--fov_y", str(float(fov_y)),
            "--seed", str(int(seed)),
            "--split_view_num", str(int(split_view_num)),
            "--splitted_resolution", str(int(splitted_resolution)),
            "--nframe", str(int(nframe)),
            "--llm_addr", llm_addr,
            "--llm_port", str(int(llm_port)),
            "--llm_name", llm_name,
            "--sam3_path", sam3_path or _default_sam3_path(),
        ]
        if local_files_only:
            cmd.append("--local_files_only")
        if apply_nav_traj:
            cmd.append("--apply_nav_traj")
        if apply_up_route:
            cmd.append("--apply_up_route")
        if apply_recon_iteration:
            cmd.append("--apply_recon_iteration")
        if force_vlm:
            cmd.append("--force_vlm")
        if skip_exist:
            cmd.append("--skip_exist")
        cmd.extend(_shell_split(extra_args))
        log = _run_command(cmd, WORLDGEN_DIR, deep_logging=deep_logging, log_path=scene / "worldgen_traj_generate_full.log")
        return (str(scene), log)


class VNCCS_WorldGenRenderTrajectories:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_dir": ("STRING", {"default": ""}),
            },
            "optional": {
                "nproc_per_node": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "seed": ("INT", {"default": 1024, "min": 0, "max": 2**31 - 1, "step": 1}),
                "llm_addr": ("STRING", {"default": "localhost"}),
                "llm_port": ("INT", {"default": 8000, "min": 1, "max": 65535, "step": 1}),
                "llm_name": ("STRING", {"default": "Qwen/Qwen3-VL-8B-Instruct"}),
                "enable_vlm_caption": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Temporary test default: off. Re-enable VLM/LLM captions for final WorldGen prompts.",
                }),
                "manual_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Used to create traj_caption.json files when VLM/LLM captioning is disabled.",
                }),
                "extra_args": ("STRING", {"default": "", "multiline": False}),
                "deep_logging": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("scene_dir", "log")
    FUNCTION = "render"
    CATEGORY = "VNCCS/WorldGen"
    OUTPUT_NODE = True

    def render(
        self,
        scene_dir,
        nproc_per_node=1,
        seed=1024,
        llm_addr="localhost",
        llm_port=8000,
        llm_name="Qwen/Qwen3-VL-8B-Instruct",
        enable_vlm_caption=False,
        manual_prompt="",
        extra_args="",
        deep_logging=False,
    ):
        scene = Path(scene_dir)
        if not scene.exists():
            raise FileNotFoundError(f"scene_dir not found: {scene}")
        cmd = _torchrun_command("traj_render.py", nproc_per_node)
        cmd.extend([
            "--target_path", str(scene),
            "--seed", str(int(seed)),
            "--llm_addr", llm_addr,
            "--llm_port", str(int(llm_port)),
            "--llm_name", llm_name,
        ])
        if not enable_vlm_caption:
            # TEMPORARY TEST MODE: LLM/VLM captioning is disabled.
            # Turn enable_vlm_caption back on for production-quality trajectory prompts.
            cmd.append("--disable_vlm_caption")
        cmd.extend(_shell_split(extra_args, boolean_flags={"--disable_vlm_caption"}))
        log = _run_command(cmd, WORLDGEN_DIR, deep_logging=deep_logging, log_path=scene / "worldgen_traj_render_full.log")
        manual_prompt_count = 0
        if not enable_vlm_caption:
            manual_prompt_count = _write_manual_traj_prompts(scene, manual_prompt)
            missing_captions = _missing_traj_caption_paths(scene)
            if missing_captions:
                sample = "\n".join(str(path) for path in missing_captions[:8])
                raise ValueError(
                    "VLM captioning is disabled, but manual_prompt is empty or did not cover all rendered "
                    "trajectories. Fill manual_prompt in WorldGen Render Trajectories or enable VLM captioning. "
                    f"Missing {len(missing_captions)} traj_caption.json file(s):\n{sample}"
                )
        if manual_prompt_count:
            log = (
                f"[WorldGen] Manual prompt test mode wrote {manual_prompt_count} missing traj_caption.json file(s). "
                "Re-enable VLM/LLM captions for final production prompts.\n"
                f"{log}"
            )
        return (str(scene), log)


class VNCCS_WorldGenWorldStereoVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_dir": ("STRING", {"default": ""}),
            },
            "optional": {
                "worldstereo_model": ("WORLDSTEREO_MODEL", {
                    "tooltip": "Optional. Connect Load WorldStereo Model only for custom paths; otherwise WorldGen resolves local model folders without loading a model in ComfyUI.",
                }),
                "model_type": (["worldstereo-memory", "worldstereo-memory-dmd"], {"default": "worldstereo-memory-dmd"}),
                "nproc_per_node": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "seed": ("INT", {"default": 1024, "min": 0, "max": 2**31 - 1, "step": 1}),
                "nframe": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 81,
                    "step": 1,
                    "tooltip": "0 keeps the model config. Lower values such as 41 or 21 reduce WorldGen video VRAM for testing.",
                }),
                "align_nframe": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
                "max_reference": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
                "downsampled_pts": ("INT", {"default": 2_000_000, "min": 1, "max": 50_000_000, "step": 100_000}),
                "fsdp": ("BOOLEAN", {"default": False}),
                "skip_exist": ("BOOLEAN", {"default": False}),
                "sam3_path": ("STRING", {
                    "default": _default_sam3_path(),
                    "tooltip": "SAM3 repo id or local checkpoint path.",
                }),
                "local_files_only": ("BOOLEAN", {"default": True}),
                "manual_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Temporary test prompt used to create missing traj_caption.json files when VLM/LLM captions are disabled.",
                }),
                "extra_args": ("STRING", {"default": "", "multiline": False}),
                "deep_logging": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("scene_dir", "generation_bank_dir", "log")
    FUNCTION = "generate_video"
    CATEGORY = "VNCCS/WorldGen"
    OUTPUT_NODE = True

    def generate_video(
        self,
        scene_dir,
        worldstereo_model=None,
        model_type="worldstereo-memory-dmd",
        nproc_per_node=1,
        seed=1024,
        nframe=0,
        align_nframe=8,
        max_reference=8,
        downsampled_pts=2_000_000,
        fsdp=False,
        skip_exist=False,
        sam3_path=None,
        local_files_only=True,
        manual_prompt="",
        extra_args="",
        deep_logging=False,
    ):
        scene = Path(scene_dir)
        if not scene.exists():
            raise FileNotFoundError(f"scene_dir not found: {scene}")
        manual_prompt_count = _write_manual_traj_prompts(scene, manual_prompt)
        if manual_prompt_count:
            print(
                "[WorldGen] Wrote "
                f"{manual_prompt_count} manual traj_caption.json file(s) from WorldStereo manual_prompt. "
                "Re-enable VLM/LLM captions for final production prompts."
            )
        resolved_model_type, model_args = _worldstereo_cli_args(worldstereo_model, model_type)
        worldstereo_model = None
        _release_comfy_models_for_external_process()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cmd = _torchrun_command("video_gen.py", nproc_per_node)
        cmd.extend([
            "--target_path", str(scene),
            "--model_type", resolved_model_type,
            "--seed", str(int(seed)),
            "--nframe", str(int(nframe)),
            "--align_nframe", str(int(align_nframe)),
            "--max_reference", str(int(max_reference)),
            "--downsampled_pts", str(int(downsampled_pts)),
            "--sam3_path", sam3_path or _default_sam3_path(),
        ])
        cmd.extend(model_args)
        if fsdp:
            cmd.append("--fsdp")
        if skip_exist:
            cmd.append("--skip_exist")
        if local_files_only and "--local_files_only" not in cmd:
            cmd.append("--local_files_only")
        cmd.extend(_shell_split(extra_args))
        log = _run_command(cmd, WORLDGEN_DIR, deep_logging=deep_logging, log_path=scene / "worldgen_video_gen_full.log")
        if manual_prompt_count:
            log = (
                f"[WorldGen] Manual prompt test mode wrote {manual_prompt_count} missing traj_caption.json file(s). "
                "Re-enable VLM/LLM captions for final production prompts.\n"
                f"{log}"
            )
        scene_list = _scene_list_from_scene_dir(scene)
        bank_dirs = [
            str(scene / "render_results" / f"generation_bank_{resolved_model_type}")
            for scene in scene_list
        ]
        bank_dir = bank_dirs[0] if len(bank_dirs) == 1 else "\n".join(bank_dirs)
        return (str(scene), bank_dir, log)


class VNCCS_WorldGenExportBankFromPLY:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_data": ("PLY_DATA",),
                "images": ("IMAGE",),
                "workspace_name": ("STRING", {
                    "default": "comfy_worldgen",
                    "tooltip": "Name for this ComfyUI worldgen workspace under root_dir.",
                }),
            },
            "optional": {
                "panorama": ("IMAGE", {
                    "tooltip": "Optional source panorama for the same scene. When connected, this workspace can continue through Generate/Render/WorldStereo.",
                }),
                "root_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Base output folder. Empty uses ComfyUI/output/hyworld2_worldgen.",
                }),
                "scene_type": (["unknown", "indoor", "outdoor"], {"default": "unknown"}),
                "bank_name": ("STRING", {
                    "default": "comfy-worldmirror",
                    "tooltip": "Name suffix for render_results/generation_bank_<bank_name>. Official runs use worldstereo-memory-dmd.",
                }),
                "global_max_points": ("INT", {"default": 3_000_000, "min": 0, "max": 50_000_000, "step": 100_000}),
                "aligned_max_points": ("INT", {"default": 2_000_000, "min": 0, "max": 50_000_000, "step": 100_000}),
                "write_aligned_from_filtered": ("BOOLEAN", {"default": True}),
                "deep_logging": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("scene_dir", "info")
    FUNCTION = "export_bank"
    CATEGORY = "VNCCS/WorldGen"
    OUTPUT_NODE = True

    def export_bank(
        self,
        ply_data,
        images,
        workspace_name="comfy_worldgen",
        panorama=None,
        root_dir="",
        scene_type="unknown",
        bank_name="comfy-worldmirror",
        global_max_points=3_000_000,
        aligned_max_points=2_000_000,
        write_aligned_from_filtered=True,
        deep_logging=False,
    ):
        root = Path(root_dir) if root_dir.strip() else _output_root() / "hyworld2_worldgen"
        workspace_dir = _ensure_dir(root / _sanitize_name(workspace_name))
        render_results_dir = _ensure_dir(workspace_dir / "render_results")
        bank_dir = _reset_dir(
            render_results_dir / f"generation_bank_{_sanitize_name(bank_name, 'comfy-worldmirror')}",
            "WorldGen generation bank",
        )
        image_tensor = _normalize_image_tensor(images)
        panorama_tensor = _normalize_image_tensor(panorama) if panorama is not None else None
        _deep_log(deep_logging, f"export workspace_dir={workspace_dir}")
        _deep_log(deep_logging, f"export bank_dir={bank_dir}")
        _deep_log(deep_logging, f"export image_shape={tuple(image_tensor.shape) if image_tensor is not None else None}")
        _deep_log(deep_logging, f"export panorama_shape={tuple(panorama_tensor.shape) if panorama_tensor is not None else None}")
        _deep_log(deep_logging, f"export ply_keys={sorted(ply_data.keys()) if isinstance(ply_data, dict) else type(ply_data)}")

        global_points, global_colors = _points_and_colors_from_ply_data(
            ply_data, images=images, prefer_filtered=False, max_points=int(global_max_points)
        )
        aligned_points, aligned_colors = _points_and_colors_from_ply_data(
            ply_data, images=images, prefer_filtered=bool(write_aligned_from_filtered), max_points=int(aligned_max_points)
        )
        _deep_log(deep_logging, f"export global_points={tuple(global_points.shape)} aligned_points={tuple(aligned_points.shape)}")

        global_path = _write_point_ply(bank_dir / "global_pcd.ply", global_points, global_colors)
        aligned_path = _write_point_ply(bank_dir / "aligned_pcd.ply", aligned_points, aligned_colors)
        panorama_path = None
        if panorama_tensor is not None and panorama_tensor.shape[0] > 0:
            panorama_path = str(workspace_dir / "panorama.png")
            _save_rgb_image(workspace_dir / "panorama.png", panorama_tensor[0])
        if scene_type != "unknown":
            meta = {"scene_type": scene_type}
            with open(workspace_dir / "meta_info.json", "w", encoding="utf-8") as handle:
                json.dump(meta, handle, indent=2)
        info = {
            "scene_dir": str(workspace_dir),
            "workspace_dir": str(workspace_dir),
            "bank_dir": str(bank_dir),
            "bank_name": bank_name,
            "panorama_path": panorama_path,
            "global_points": int(global_points.shape[0]),
            "aligned_points": int(aligned_points.shape[0]),
            "note": "Workspace prepared from WorldMirror PLY_DATA. Official trajectory generation will create render_results/global_pcd.ply from panorama.png.",
        }
        with open(bank_dir / "pcd_info.json", "w", encoding="utf-8") as handle:
            json.dump(info, handle, indent=2)
        return (str(workspace_dir), json.dumps(info, indent=2))


class VNCCS_WorldGenBuildGSDataFromWorldMirror:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_data": ("PLY_DATA",),
                "images": ("IMAGE",),
                "camera_poses": ("TENSOR",),
                "camera_intrinsics": ("TENSOR",),
            },
            "optional": {
                "workspace_dir": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Optional workspace_dir from WorldGen Export Generation Bank. If omitted, a default output workspace is created.",
                }),
                "depth_maps": ("IMAGE",),
                "normal_maps": ("IMAGE",),
                "raw_splats": ("VNCCS_SPLAT",),
                "out_name": ("STRING", {
                    "default": "gs_data",
                    "tooltip": "Subfolder inside workspace_dir where the trainer dataset is written.",
                }),
                "camera_bundle": (["connected_inputs", "model_predicted_when_input_zero"], {
                    "default": "model_predicted_when_input_zero",
                    "tooltip": "Use WorldMirror predicted cameras/geometry when connected cameras have zero translation. Native 3DGS needs camera baseline; rotation-only cameras collapse depth.",
                }),
                "points_max": ("INT", {"default": 3_000_000, "min": 0, "max": 50_000_000, "step": 100_000}),
                "write_normals": ("BOOLEAN", {"default": True}),
                "deep_logging": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("gs_data_dir", "info")
    FUNCTION = "build_gs_data"
    CATEGORY = "VNCCS/WorldGen"
    OUTPUT_NODE = True

    def build_gs_data(
        self,
        ply_data,
        images,
        camera_poses,
        camera_intrinsics,
        workspace_dir="",
        depth_maps=None,
        normal_maps=None,
        raw_splats=None,
        out_name="gs_data",
        camera_bundle="model_predicted_when_input_zero",
        points_max=3_000_000,
        write_normals=True,
        deep_logging=False,
    ):
        workspace = Path(workspace_dir) if str(workspace_dir).strip() else _output_root() / "hyworld2_worldgen" / "comfy_worldgen"
        gs_dir = _reset_dir(workspace / _sanitize_name(out_name, "gs_data"), "WorldGen gs_data")
        images_dir = _ensure_dir(gs_dir / "images")
        depths_dir = _ensure_dir(gs_dir / "depths")
        normals_dir = _ensure_dir(gs_dir / "normals")
        _deep_log(deep_logging, f"build workspace={workspace}")
        _deep_log(deep_logging, f"build gs_dir={gs_dir}")
        _deep_log(deep_logging, "build reset_existing_gs_data=true")

        image_tensor = _normalize_image_tensor(images)
        poses = _normalize_pose_tensor(camera_poses)
        intrs = _normalize_intrinsics_tensor(camera_intrinsics)
        if image_tensor is None:
            raise ValueError("images must be a ComfyUI IMAGE tensor.")
        if poses is None or intrs is None:
            raise ValueError("camera_poses and camera_intrinsics must be valid tensors.")
        camera_source = "input"
        geometry_source = "input"
        model_poses = _normalize_pose_tensor(ply_data.get("model_camera_poses") if isinstance(ply_data, dict) else None)
        model_intrs = _normalize_intrinsics_tensor(ply_data.get("model_camera_intrs") if isinstance(ply_data, dict) else None)
        input_translation_std = _camera_translation_std(poses)
        model_translation_std = _camera_translation_std(model_poses)
        if (
            model_poses is not None
            and model_intrs is not None
            and input_translation_std < 1e-6
            and model_translation_std > 1e-5
        ):
            poses = model_poses
            intrs = model_intrs
            camera_source = "worldmirror_model_predicted"
            geometry_source = "worldmirror_model_predicted"
            print(
                "[WorldGen] Connected camera_poses have zero translation; using "
                "WorldMirror predicted cameras/geometry for native 3DGS."
            )
        n = min(image_tensor.shape[0], poses.shape[0], intrs.shape[0])
        if n <= 0:
            raise ValueError("No frames available for gs_data.")
        active_translation_std = _camera_translation_std(poses[:n])
        _deep_log(deep_logging, f"build image_shape={tuple(image_tensor.shape)}")
        _deep_log(deep_logging, f"build poses_shape={tuple(poses.shape)} intrinsics_shape={tuple(intrs.shape)} frames={n}")
        _deep_log(
            deep_logging,
            f"build camera_source={camera_source} input_translation_std={input_translation_std:.8f} "
            f"model_translation_std={model_translation_std:.8f} active_translation_std={active_translation_std:.8f} "
            f"camera_bundle={camera_bundle}",
        )
        if active_translation_std < 1e-6:
            raise ValueError(
                "gs_data cameras have zero translation and no usable WorldMirror "
                "predicted camera bundle was selected. Native 3DGS needs camera baseline; "
                "rotation-only cameras collapse depth and produce holes/noise."
            )

        if geometry_source == "worldmirror_model_predicted" and isinstance(ply_data, dict):
            pts_grid = _normalize_points_tensor(ply_data.get("model_pts3d"))
        else:
            pts_grid = _normalize_points_tensor(ply_data.get("pts3d") if isinstance(ply_data, dict) else None)
        raw_predictions = _raw_prediction_dict(raw_splats)
        raw_depths = _depth_maps_to_metric_tensor(raw_predictions.get("depth"))
        if raw_depths is None and isinstance(ply_data, dict):
            raw_depths = _depth_maps_to_metric_tensor(ply_data.get("depth"))
        computed_depths = raw_depths if raw_depths is not None else _depths_from_points(pts_grid, poses)
        wm_image_tensor = _normalize_image_tensor(ply_data.get("images") if isinstance(ply_data, dict) else None)
        depth_image_tensor = _normalize_image_tensor(depth_maps) if depth_maps is not None else None
        normal_tensor = None
        if write_normals:
            normal_tensor = _normal_maps_to_encoded_tensor(raw_predictions.get("normals"))
            if normal_tensor is None and isinstance(ply_data, dict):
                normal_tensor = _normal_maps_to_encoded_tensor(ply_data.get("normals"))
            if normal_tensor is None:
                normal_tensor = _normal_maps_to_tensor(normal_maps)
        normal_source = "normal_maps" if _normal_tensor_is_usable(normal_tensor) else "none"
        if normal_tensor is not None and normal_source == "none":
            _clear_png_dir(normals_dir)
            normal_tensor = None
        if raw_depths is not None:
            depth_source = "raw metric depth"
        elif computed_depths is not None:
            depth_source = "pts3d camera z-depth"
        else:
            depth_source = "none"
        if computed_depths is None:
            _clear_png_dir(depths_dir)
        _deep_log(deep_logging, f"build pts3d_shape={tuple(pts_grid.shape) if pts_grid is not None else None}")
        _deep_log(deep_logging, f"build computed_depths_shape={tuple(computed_depths.shape) if computed_depths is not None else None}")
        _deep_log(deep_logging, f"build wm_image_shape={tuple(wm_image_tensor.shape) if wm_image_tensor is not None else None}")
        _deep_log(deep_logging, f"build depth_image_shape={tuple(depth_image_tensor.shape) if depth_image_tensor is not None else None}")
        _deep_log(deep_logging, f"build depth_source={depth_source}")
        _deep_log(deep_logging, f"build normal_shape={tuple(normal_tensor.shape) if normal_tensor is not None else None}")
        _deep_log(deep_logging, f"build normal_source={normal_source}")

        cameras = {}
        for i in range(n):
            name = f"frame_{i:06d}"
            image_h, image_w = int(image_tensor[i].shape[0]), int(image_tensor[i].shape[1])
            source_h, source_w = image_h, image_w
            if wm_image_tensor is not None and i < wm_image_tensor.shape[0]:
                source_h, source_w = int(wm_image_tensor[i].shape[0]), int(wm_image_tensor[i].shape[1])
            elif depth_image_tensor is not None and i < depth_image_tensor.shape[0]:
                source_h, source_w = int(depth_image_tensor[i].shape[0]), int(depth_image_tensor[i].shape[1])
            _save_rgb_image(images_dir / f"{name}.png", image_tensor[i])
            if computed_depths is not None and i < computed_depths.shape[0]:
                depth = _resize_2d_tensor(computed_depths[i], image_h, image_w)
                _save_depth16(depths_dir / f"{name}.png", depth.numpy())
                if i == 0:
                    _deep_log(deep_logging, f"build saved_depth_shape={tuple(depth.shape)} source=computed")
            if normal_tensor is not None and i < normal_tensor.shape[0]:
                normal = _resize_hwc_tensor(normal_tensor[i], image_h, image_w)
                normal_arr = (normal.clamp(0.0, 1.0).numpy() * 255.0 + 0.5).astype(np.uint8)
                Image.fromarray(normal_arr).save(normals_dir / f"{name}.png")
                if i == 0:
                    _deep_log(deep_logging, f"build saved_normal_shape={tuple(normal.shape)}")

            w2c = torch.linalg.inv(poses[i]).numpy().tolist()
            scaled_intrinsic = _scale_intrinsic_for_image(
                intrs[i],
                source_h,
                source_w,
                image_h,
                image_w,
            )
            if i == 0:
                _deep_log(
                    deep_logging,
                    "build intrinsic_scale="
                    f"source={source_w}x{source_h} image={image_w}x{image_h} "
                    f"fx {float(intrs[i, 0, 0]):.4f}->{float(scaled_intrinsic[0, 0]):.4f} "
                    f"fy {float(intrs[i, 1, 1]):.4f}->{float(scaled_intrinsic[1, 1]):.4f}",
                )
            cameras[name] = {
                "extrinsic": w2c,
                "intrinsic": scaled_intrinsic.numpy().tolist(),
            }

        with open(gs_dir / "cameras.json", "w", encoding="utf-8") as handle:
            json.dump(cameras, handle, indent=2)

        point_data = ply_data
        if geometry_source == "worldmirror_model_predicted" and isinstance(ply_data, dict):
            point_data = dict(ply_data)
            if point_data.get("model_pts3d") is not None:
                point_data["pts3d"] = point_data.get("model_pts3d")
            if point_data.get("model_pts3d_filtered") is not None:
                point_data["pts3d_filtered"] = point_data.get("model_pts3d_filtered")
        points, colors = _points_and_colors_from_ply_data(
            point_data, images=images, prefer_filtered=False, max_points=int(points_max)
        )
        points_path = _write_point_ply(gs_dir / "points.ply", points, colors)
        _deep_log(deep_logging, f"build points_path={points_path} points={tuple(points.shape)} colors={tuple(colors.shape)}")
        meta = {
            "source": "ComfyUI WorldMirror shortcut",
            "scene_type": "unknown",
            "frames": n,
            "points": int(points.shape[0]),
            "depth_source": depth_source,
            "normal_source": normal_source,
            "camera_source": camera_source,
            "geometry_source": geometry_source,
            "active_translation_std": active_translation_std,
            "points_path": points_path,
        }
        with open(gs_dir / "meta_info.json", "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        manifest = {
            "workspace": str(workspace),
            "gs_dir": str(gs_dir),
            "reset_existing_gs_data": True,
            "image_shape": list(image_tensor.shape),
            "pose_shape": list(poses.shape),
            "intrinsic_shape": list(intrs.shape),
            "frames_written": n,
            "camera_source": camera_source,
            "geometry_source": geometry_source,
            "input_translation_std": input_translation_std,
            "model_translation_std": model_translation_std,
            "active_translation_std": active_translation_std,
            "pts3d_shape": list(pts_grid.shape) if pts_grid is not None else None,
            "computed_depths_shape": list(computed_depths.shape) if computed_depths is not None else None,
            "depth_source": depth_source,
            "normal_shape": list(normal_tensor.shape) if normal_tensor is not None else None,
            "normal_source": normal_source,
            "points_path": points_path,
            "points": int(points.shape[0]),
        }
        with open(gs_dir / "build_manifest.json", "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        return (str(gs_dir), json.dumps(meta, indent=2))


class VNCCS_WorldGenRunOfficialGSData:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_dir": ("STRING", {"default": ""}),
            },
            "optional": {
                "result_name": ("STRING", {"default": "worldstereo-memory-dmd"}),
                "out_name": ("STRING", {"default": "gs_data"}),
                "nproc_per_node": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "save_normal": ("BOOLEAN", {"default": True}),
                "split_sky": ("BOOLEAN", {"default": True}),
                "split_align": ("BOOLEAN", {"default": False}),
                "extra_args": ("STRING", {"default": "", "multiline": False}),
                "deep_logging": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("gs_data_dir", "log")
    FUNCTION = "run_gen_gs_data"
    CATEGORY = "VNCCS/WorldGen"
    OUTPUT_NODE = True

    def run_gen_gs_data(
        self,
        scene_dir,
        result_name="worldstereo-memory-dmd",
        out_name="gs_data",
        nproc_per_node=1,
        save_normal=True,
        split_sky=True,
        split_align=False,
        extra_args="",
        deep_logging=False,
    ):
        scene = Path(scene_dir)
        if not scene.exists():
            raise FileNotFoundError(f"scene_dir not found: {scene}")
        gs_out_dir = scene / _sanitize_name(out_name, "gs_data")
        _reset_dir(gs_out_dir, "official WorldGen gs_data")
        cmd = _torchrun_command("gen_gs_data.py", nproc_per_node)
        cmd.extend([
            "--root_path", str(scene),
            "--out_name", out_name,
            "--result_name", result_name,
        ])
        if save_normal:
            cmd.append("--save_normal")
        if split_sky:
            cmd.append("--split_sky")
        if split_align:
            cmd.append("--split_align")
        if extra_args.strip():
            cmd.extend(_shell_split(extra_args))
        _deep_log(deep_logging, f"official gen_gs_data scene={scene}")
        log = _run_command(cmd, WORLDGEN_DIR, deep_logging=deep_logging, log_path=gs_out_dir / "worldgen_gen_gs_data_full.log")
        return (str(gs_out_dir), log)


class VNCCS_WorldGenTrain3DGS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gs_data_dir": ("STRING", {"default": ""}),
            },
            "optional": {
                "train_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Folder for native 3DGS training output. Empty writes next to gs_data as gs_results.",
                }),
                "max_steps": ("INT", {"default": 1500, "min": 1, "max": 100000, "step": 100}),
                "save_steps": ("STRING", {"default": "1500"}),
                "eval_steps": ("STRING", {"default": "1500"}),
                "ply_steps": ("STRING", {"default": "1500"}),
                "save_ply": ("BOOLEAN", {"default": True}),
                "disable_video": ("BOOLEAN", {"default": True}),
                "disable_viewer": ("BOOLEAN", {"default": True}),
                "depth_loss": ("BOOLEAN", {"default": True}),
                "normal_loss": ("BOOLEAN", {"default": True}),
                "use_scale_regularization": ("BOOLEAN", {"default": True}),
                "use_mask_gaussian": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "MaskGaussian can stochastically prune exported splats and create holes. Keep off for the issue6/manual optimization path.",
                }),
                "mask_export_stochastic": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Only used when MaskGaussian is enabled. Disable to avoid random holes in the exported PLY.",
                }),
                "antialiased": ("BOOLEAN", {"default": True}),
                "official_strategy_preset": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use HY-World's stable training strategy: short densify/refine window, no opacity resets, conservative grow/prune thresholds.",
                }),
                "normalize_world_space": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Official trainer normalization. Keep enabled for native 3DGS; disabling weakens depth scaling and can warp geometry.",
                }),
                "perceptual_loss": (["lpips_vgg", "lpips_alex", "simple", "off"], {"default": "lpips_vgg"}),
                "extra_args": ("STRING", {"default": "", "multiline": False}),
                "deep_logging": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "TENSOR", "TENSOR", "STRING", "STRING")
    RETURN_NAMES = ("ply_path", "camera_poses", "camera_intrinsics", "train_dir", "log")
    FUNCTION = "train"
    CATEGORY = "VNCCS/WorldGen"
    OUTPUT_NODE = True

    def train(
        self,
        gs_data_dir,
        train_dir="",
        max_steps=1500,
        save_steps="1500",
        eval_steps="1500",
        ply_steps="1500",
        save_ply=True,
        disable_video=True,
        disable_viewer=True,
        depth_loss=True,
        normal_loss=True,
        use_scale_regularization=True,
        use_mask_gaussian=False,
        mask_export_stochastic=False,
        antialiased=True,
        official_strategy_preset=True,
        normalize_world_space=True,
        perceptual_loss="lpips_vgg",
        extra_args="",
        deep_logging=False,
    ):
        data_dir = Path(gs_data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"gs_data_dir not found: {data_dir}")
        translation_std = _gs_data_camera_translation_std(data_dir)
        if translation_std is not None and translation_std < 1e-6:
            raise ValueError(
                f"gs_data cameras have zero translation (translation_std={translation_std:.8f}). "
                "Native 3DGS cannot recover depth from rotation-only cameras; rebuild gs_data "
                "with WorldMirror predicted cameras/geometry."
            )
        out_dir = Path(train_dir) if str(train_dir).strip() else data_dir.parent / "gs_results"
        data_resolved = data_dir.resolve()
        out_resolved = out_dir.resolve()
        if out_resolved == data_resolved or _path_is_relative_to(data_resolved, out_resolved):
            raise ValueError(
                f"Refusing to reset train_dir because it contains gs_data_dir. "
                f"train_dir={out_dir} gs_data_dir={data_dir}"
            )
        _reset_dir(out_dir, "WorldGen train_dir")
        meta_path = _ensure_scene_type_meta(data_dir)
        trainer_script = WORLDGEN_DIR / "world_gs_trainer.py"
        if not trainer_script.exists():
            raise FileNotFoundError(f"world_gs_trainer.py not found: {trainer_script}")
        _deep_log(deep_logging, f"train data_dir={data_dir}")
        _deep_log(deep_logging, f"train result_dir={out_dir}")
        _deep_log(deep_logging, f"train meta_info={meta_path}")
        _deep_log(deep_logging, f"train trainer_script={trainer_script}")

        cmd = [
            sys.executable, str(trainer_script), "default",
            "--data_dir", str(data_dir),
            "--result_dir", str(out_dir),
            "--max_steps", str(int(max_steps)),
            "--save_steps", *save_steps.split(),
            "--eval_steps", *eval_steps.split(),
            "--ply_steps", *ply_steps.split(),
        ]
        if save_ply:
            cmd.append("--save_ply")
        if disable_video:
            cmd.append("--disable_video")
        if disable_viewer:
            cmd.append("--disable_viewer")
        depth_files_valid = _has_valid_depth_files(data_dir)
        if depth_loss and depth_files_valid:
            cmd.append("--depth_loss")
        elif depth_loss:
            print(f"[WorldGen] depth_loss requested but valid metric float16-packed depths are missing under {data_dir / 'depths'}; disabling depth_loss.")
        normal_files_valid = _has_valid_normal_files(data_dir)
        if normal_loss and normal_files_valid:
            cmd.append("--normal_loss")
        elif normal_loss:
            print(f"[WorldGen] normal_loss requested but normals are missing/constant under {data_dir / 'normals'}; disabling normal_loss.")
        if use_scale_regularization:
            cmd.append("--use_scale_regularization")
        if use_mask_gaussian:
            cmd.append("--use_mask_gaussian")
            if mask_export_stochastic:
                cmd.append("--mask_export_stochastic")
            else:
                cmd.append("--no-mask_export_stochastic")
        if antialiased:
            cmd.append("--antialiased")
        if official_strategy_preset:
            cmd.extend(_official_strategy_args(max_steps))
        if not normalize_world_space:
            cmd.append("--no-normalize")
        if perceptual_loss == "lpips_vgg":
            cmd.extend(["--lpips_net", "vgg"])
        elif perceptual_loss == "lpips_alex":
            cmd.extend(["--lpips_net", "alex"])
        elif perceptual_loss == "simple":
            cmd.extend(["--lpips_net", "simple"])
        elif perceptual_loss == "off":
            cmd.extend(["--lpips_net", "none", "--lpips_lambda1", "0", "--lpips_lambda2", "0"])
        if extra_args.strip():
            cmd.extend(_shell_split(extra_args))

        command_info = {
            "data_dir": str(data_dir),
            "result_dir": str(out_dir),
            "trainer_script": str(trainer_script),
            "command": [str(x) for x in cmd],
            "max_steps": int(max_steps),
            "save_steps": save_steps.split(),
            "eval_steps": eval_steps.split(),
            "ply_steps": ply_steps.split(),
            "save_ply": bool(save_ply),
            "depth_loss_requested": bool(depth_loss),
            "depth_loss_enabled": bool(depth_loss and depth_files_valid),
            "normal_loss_requested": bool(normal_loss),
            "normal_loss_enabled": bool(normal_loss and normal_files_valid),
            "use_mask_gaussian": bool(use_mask_gaussian),
            "mask_export_stochastic": bool(mask_export_stochastic),
            "official_strategy_preset": bool(official_strategy_preset),
            "normalize_world_space": bool(normalize_world_space),
            "perceptual_loss": perceptual_loss,
            "reset_existing_train_dir": True,
            "full_log": str(out_dir / "worldgen_train_full.log"),
        }
        with open(out_dir / "train_command.json", "w", encoding="utf-8") as handle:
            json.dump(command_info, handle, indent=2)

        log = _run_command(cmd, WORLDGEN_DIR, deep_logging=deep_logging, log_path=out_dir / "worldgen_train_full.log")
        ply_path = _find_latest_ply(out_dir)
        if not ply_path:
            log = f"{log}\n[WorldGen] No PLY found under {out_dir}. Ensure save_ply=true and ply_steps includes a reached step."
        camera_poses, camera_intrinsics, camera_source = _load_train_camera_tensors(out_dir, data_dir)
        if camera_source:
            log = f"{log}\n[WorldGen] Camera export loaded: {camera_source}"
        else:
            log = f"{log}\n[WorldGen] No camera export found for preview."
        return (ply_path, camera_poses, camera_intrinsics, str(out_dir), log)


NODE_CLASS_MAPPINGS = {
    "VNCCS_WorldGenExportBankFromPLY": VNCCS_WorldGenExportBankFromPLY,
    "VNCCS_WorldGenBuildGSDataFromWorldMirror": VNCCS_WorldGenBuildGSDataFromWorldMirror,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_WorldGenExportBankFromPLY": "WorldGen Prepare Workspace",
    "VNCCS_WorldGenBuildGSDataFromWorldMirror": "WorldGen Build GS Data From WorldMirror",
}
