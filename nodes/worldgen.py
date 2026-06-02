import json
import os
import re
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORLDGEN_DIR = PROJECT_ROOT / "hyworld2" / "worldgen"
SH_C0 = 0.28209479177387814


def _output_root() -> Path:
    if folder_paths is not None:
        return Path(folder_paths.get_output_directory())
    return PROJECT_ROOT / "output"


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
        depth = cam.norm(dim=1).reshape(points.shape[1], points.shape[2])
        depths.append(depth.clamp_min(0.0))
    return torch.stack(depths, dim=0)


def _normal_maps_to_tensor(normal_maps):
    normals = _normalize_image_tensor(normal_maps)
    if normals is None:
        return None
    return normals.clamp(0.0, 1.0)


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


def _has_depth_files(data_dir):
    depths_dir = Path(data_dir) / "depths"
    return depths_dir.exists() and any(depths_dir.glob("*.png"))


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
                "root_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Base output folder. Empty uses ComfyUI/output/hyworld2_worldgen.",
                }),
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
    RETURN_NAMES = ("workspace_dir", "info")
    FUNCTION = "export_bank"
    CATEGORY = "VNCCS/WorldGen"
    OUTPUT_NODE = True

    def export_bank(
        self,
        ply_data,
        images,
        workspace_name="comfy_worldgen",
        root_dir="",
        bank_name="comfy-worldmirror",
        global_max_points=3_000_000,
        aligned_max_points=2_000_000,
        write_aligned_from_filtered=True,
        deep_logging=False,
    ):
        root = Path(root_dir) if root_dir.strip() else _output_root() / "hyworld2_worldgen"
        workspace_dir = _ensure_dir(root / _sanitize_name(workspace_name))
        bank_dir = _reset_dir(
            workspace_dir / "render_results" / f"generation_bank_{_sanitize_name(bank_name, 'comfy-worldmirror')}",
            "WorldGen generation bank",
        )
        image_tensor = _normalize_image_tensor(images)
        _deep_log(deep_logging, f"export workspace_dir={workspace_dir}")
        _deep_log(deep_logging, f"export bank_dir={bank_dir}")
        _deep_log(deep_logging, f"export image_shape={tuple(image_tensor.shape) if image_tensor is not None else None}")
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
        info = {
            "workspace_dir": str(workspace_dir),
            "bank_dir": str(bank_dir),
            "bank_name": bank_name,
            "global_points": int(global_points.shape[0]),
            "aligned_points": int(aligned_points.shape[0]),
            "note": "Official-like generation bank exported from current WorldMirror PLY_DATA; sky_pcd is not generated by this shortcut node.",
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
                "out_name": ("STRING", {
                    "default": "gs_data",
                    "tooltip": "Subfolder inside workspace_dir where the trainer dataset is written.",
                }),
                "camera_bundle": (["connected_inputs", "model_predicted_when_input_zero"], {
                    "default": "connected_inputs",
                    "tooltip": "connected_inputs preserves the exact cameras/depth/points currently wired from WorldMirror. model_predicted_when_input_zero is experimental.",
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
        out_name="gs_data",
        camera_bundle="connected_inputs",
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
            camera_bundle == "model_predicted_when_input_zero"
            and
            model_poses is not None
            and model_intrs is not None
            and input_translation_std < 1e-6
            and model_translation_std > 1e-5
        ):
            poses = model_poses
            intrs = model_intrs
            camera_source = "worldmirror_model_predicted"
            geometry_source = "worldmirror_model_predicted"
        n = min(image_tensor.shape[0], poses.shape[0], intrs.shape[0])
        if n <= 0:
            raise ValueError("No frames available for gs_data.")
        _deep_log(deep_logging, f"build image_shape={tuple(image_tensor.shape)}")
        _deep_log(deep_logging, f"build poses_shape={tuple(poses.shape)} intrinsics_shape={tuple(intrs.shape)} frames={n}")
        _deep_log(
            deep_logging,
            f"build camera_source={camera_source} input_translation_std={input_translation_std:.8f} "
            f"model_translation_std={model_translation_std:.8f} camera_bundle={camera_bundle}",
        )
        if input_translation_std < 1e-6 and camera_source == "input":
            print(
                "[WorldGen] WARNING: connected camera_poses have zero translation. "
                "This is expected for equirect panorama view slices, but native 3DGS training "
                "has no parallax and may collapse or smear geometry."
            )

        if geometry_source == "worldmirror_model_predicted" and isinstance(ply_data, dict):
            pts_grid = _normalize_points_tensor(ply_data.get("model_pts3d"))
        else:
            pts_grid = _normalize_points_tensor(ply_data.get("pts3d") if isinstance(ply_data, dict) else None)
        computed_depths = _depths_from_points(pts_grid, poses)
        wm_image_tensor = _normalize_image_tensor(ply_data.get("images") if isinstance(ply_data, dict) else None)
        depth_image_tensor = _normalize_image_tensor(depth_maps) if depth_maps is not None else None
        normal_tensor = _normal_maps_to_tensor(normal_maps) if write_normals else None
        normal_source = "normal_maps" if _normal_tensor_is_usable(normal_tensor) else "none"
        if normal_tensor is not None and normal_source == "none":
            _clear_png_dir(normals_dir)
            normal_tensor = None
        depth_source = "pts3d camera distance" if computed_depths is not None else "none"
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
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node", str(int(nproc_per_node)),
            "gen_gs_data.py",
            "--root_path", str(scene),
            "--out_name", out_name,
            "--result_name", result_name,
        ]
        if save_normal:
            cmd.append("--save_normal")
        if split_sky:
            cmd.append("--split_sky")
        if split_align:
            cmd.append("--split_align")
        if extra_args.strip():
            cmd.extend(extra_args.split())
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
                "antialiased": ("BOOLEAN", {"default": True}),
                "normalize_world_space": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Official trainer normalization. Disable for ComfyUI preview with original WorldMirror camera_poses.",
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
        antialiased=True,
        normalize_world_space=False,
        perceptual_loss="lpips_vgg",
        extra_args="",
        deep_logging=False,
    ):
        data_dir = Path(gs_data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"gs_data_dir not found: {data_dir}")
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
        depth_files_valid = _has_depth_files(data_dir)
        if depth_loss and depth_files_valid:
            cmd.append("--depth_loss")
        elif depth_loss:
            print(f"[WorldGen] depth_loss requested but metric depths are missing under {data_dir / 'depths'}; disabling depth_loss.")
        normal_files_valid = _has_valid_normal_files(data_dir)
        if normal_loss and normal_files_valid:
            cmd.append("--normal_loss")
        elif normal_loss:
            print(f"[WorldGen] normal_loss requested but normals are missing/constant under {data_dir / 'normals'}; disabling normal_loss.")
        if use_scale_regularization:
            cmd.append("--use_scale_regularization")
        if antialiased:
            cmd.append("--antialiased")
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
            cmd.extend(extra_args.split())

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
    "VNCCS_WorldGenRunOfficialGSData": VNCCS_WorldGenRunOfficialGSData,
    "VNCCS_WorldGenTrain3DGS": VNCCS_WorldGenTrain3DGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_WorldGenExportBankFromPLY": "WorldGen Prepare Workspace",
    "VNCCS_WorldGenBuildGSDataFromWorldMirror": "WorldGen Build GS Data From WorldMirror",
    "VNCCS_WorldGenRunOfficialGSData": "WorldGen Run Official gen_gs_data",
    "VNCCS_WorldGenTrain3DGS": "WorldGen Train Native 3DGS",
}
