import contextlib
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

try:
    import comfy.model_management as comfy_model_management
except Exception:
    comfy_model_management = None
import torch.nn.functional as F
from PIL import Image

try:
    import folder_paths
except ImportError:
    folder_paths = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORLDGEN_DIR = PROJECT_ROOT / "hyworld2" / "worldgen"


def _ensure_worldgen_path():
    for path in (str(PROJECT_ROOT), str(WORLDGEN_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _output_root() -> Path:
    if folder_paths is not None:
        return Path(folder_paths.get_output_directory())
    return PROJECT_ROOT / "output"


def _sanitize_name(value, fallback="scene"):
    import re

    base = os.path.basename(str(value or fallback).replace("\\", "/"))
    base = os.path.splitext(base)[0]
    base = re.sub(r"[^A-Za-z0-9._ -]+", "_", base).strip(" ._")
    return base or fallback


def _ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _release_model_memory(label="HYWorld2"):
    gc.collect()
    if comfy_model_management is not None:
        try:
            comfy_model_management.unload_all_models()
            comfy_model_management.cleanup_models_gc()
            comfy_model_management.soft_empty_cache(force=True)
        except Exception as exc:
            print(f"[{label}] Comfy model memory cleanup skipped ({type(exc).__name__}: {exc})")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


class _SingleProcessDist:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def is_initialized():
        return True

    @staticmethod
    def get_rank():
        return 0

    @staticmethod
    def get_world_size():
        return 1

    @staticmethod
    def barrier(*args, **kwargs):
        return None

    @staticmethod
    def all_gather_object(object_list, obj, *args, **kwargs):
        if object_list:
            object_list[0] = obj
        return None


def _ensure_single_process_dist(bank=None):
    if dist.is_available() and dist.is_initialized():
        return
    shim = _SingleProcessDist()
    module_names = {"hyworld2.worldgen.src.retrieval_wm", "src.retrieval_wm"}
    if bank is not None:
        module_names.add(bank.__class__.__module__)
    for module_name in module_names:
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "dist"):
            module.dist = shim


def _reset_dir(path, label="directory"):
    import shutil

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


def _image_tensor_to_pil_list(images):
    if not isinstance(images, torch.Tensor):
        return []
    tensor = images.detach().cpu().float()
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 4 and tensor.shape[1] in (1, 3, 4) and tensor.shape[-1] not in (1, 3, 4):
        tensor = tensor.permute(0, 2, 3, 1)
    result = []
    for frame in tensor:
        arr = (frame[..., :3].clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8)
        result.append(Image.fromarray(arr))
    return result


def _pil_list_to_image_tensor(images):
    frames = []
    for image in images:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        frames.append(torch.from_numpy(arr))
    if not frames:
        return torch.empty((0, 1, 1, 3), dtype=torch.float32)
    return torch.stack(frames, dim=0).contiguous()


def _save_rgb_image(path, image):
    arr = (image.detach().cpu().float().clamp(0.0, 1.0).numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr[..., :3]).save(path)


def _depth_tensor_to_numpy(depth_maps):
    if not isinstance(depth_maps, torch.Tensor):
        return []
    depth = depth_maps.detach().cpu().float()
    if depth.dim() == 5 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.dim() == 4 and depth.shape[0] == 1 and depth.shape[-1] not in (1, 3, 4):
        depth = depth[0]
    if depth.dim() == 4 and depth.shape[-1] in (1, 3, 4):
        depth = depth[..., 0]
    elif depth.dim() == 4 and depth.shape[1] in (1, 3, 4):
        depth = depth[:, 0]
    if depth.dim() != 3:
        return []
    return [np.nan_to_num(d.numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0) for d in depth]


def _depth_maps_to_numpy(depth_maps):
    return _depth_tensor_to_numpy(depth_maps)


def _raw_worldmirror_depths_to_numpy(raw_splats):
    if not isinstance(raw_splats, dict):
        return [], ""
    for key in ("gs_depth", "depth"):
        depths = _depth_tensor_to_numpy(raw_splats.get(key))
        if depths:
            return depths, f"raw_splats.{key}"
    return [], ""


def _first_existing_ply_path(*values):
    for value in values:
        if isinstance(value, dict):
            nested = _first_existing_ply_path(
                value.get("ply_path"),
                value.get("path"),
                value.get("file"),
                value.get("filepath"),
                value.get("gaussian_ply"),
                value.get("points_ply"),
            )
            if nested:
                return nested
        elif isinstance(value, (str, os.PathLike)) and str(value).lower().endswith(".ply"):
            path = Path(value)
            if path.exists():
                return str(path)
    return ""


def _first_splat_tensor(splats, key, dim):
    if not isinstance(splats, dict):
        return None
    value = splats.get(key)
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    if not isinstance(value, torch.Tensor):
        return None
    value = value.detach().cpu().float()
    if value.dim() >= 2 and value.shape[0] == 1 and value.shape[-1] == dim:
        value = value[0]
    if value.shape[-1] != dim:
        return None
    return value.reshape(-1, dim)


def _splat_tensor_any(splats, keys, dim):
    for key in keys:
        tensor = _first_splat_tensor(splats, key, dim)
        if tensor is not None:
            return tensor
    return None


def _normalize_bypass_points(points):
    if not isinstance(points, torch.Tensor):
        return None
    points = points.detach().cpu().float()
    if points.dim() == 5 and points.shape[0] == 1:
        points = points[0]
    if points.dim() == 4 and points.shape[-1] == 3:
        return points
    if points.dim() == 2 and points.shape[-1] == 3:
        return points
    return None


def _normalize_bypass_images(images):
    if not isinstance(images, torch.Tensor):
        return None
    images = images.detach().cpu().float()
    if images.dim() == 5 and images.shape[0] == 1:
        images = images[0]
    if images.dim() == 4 and images.shape[1] in (1, 3, 4) and images.shape[-1] not in (1, 3, 4):
        images = images.permute(0, 2, 3, 1)
    if images.dim() == 4 and images.shape[-1] in (1, 3, 4):
        return images[..., :3]
    return None


def _bypass_splat_points_and_colors(splats):
    means = _splat_tensor_any(splats, ("means", "xyz", "positions"), 3)
    if means is None:
        return None, None

    colors = _splat_tensor_any(splats, ("colors", "rgb", "rgbs"), 3)
    if colors is None:
        sh = _splat_tensor_any(splats, ("sh", "features_dc"), 3)
        if sh is not None:
            sh_c0 = 0.28209479177387814
            colors = (0.5 + sh_c0 * sh).clamp(0.0, 1.0)
    if colors is None or colors.shape[0] not in (1, means.shape[0]):
        colors = torch.full_like(means, 0.5)
    elif colors.shape[0] == 1 and means.shape[0] > 1:
        colors = colors.repeat(means.shape[0], 1)

    finite = torch.isfinite(means).all(dim=1)
    means = means[finite]
    colors = colors[finite]
    if means.numel() == 0:
        return None, None
    colors_u8 = (colors.clamp(0.0, 1.0).numpy() * 255.0 + 0.5).astype(np.uint8)
    return means.numpy().astype(np.float32), colors_u8


def _bypass_points_and_colors(ply_data, raw_splats=None):
    for container in (ply_data, raw_splats):
        if isinstance(container, dict):
            points, colors = _bypass_splat_points_and_colors(container.get("splats"))
            if points is not None and colors is not None:
                return points, colors

    sources = []
    if isinstance(ply_data, dict):
        sources.extend(
            [
                ply_data.get("pts3d_filtered"),
                ply_data.get("pts3d"),
                ply_data.get("model_pts3d_filtered"),
                ply_data.get("model_pts3d"),
            ]
        )
    if isinstance(raw_splats, dict):
        sources.extend([raw_splats.get("pts3d_filtered"), raw_splats.get("pts3d")])

    points = None
    for source in sources:
        points = _normalize_bypass_points(source)
        if points is not None:
            break
    if points is None:
        return None, None

    images = None
    if isinstance(ply_data, dict):
        images = _normalize_bypass_images(ply_data.get("images"))
    if images is None and isinstance(raw_splats, dict):
        images = _normalize_bypass_images(raw_splats.get("images"))

    if points.dim() == 4:
        flat_points = points.reshape(-1, 3)
        if images is not None and images.shape[0] == points.shape[0] and images.shape[1:3] == points.shape[1:3]:
            flat_colors = images.reshape(-1, 3)
        else:
            flat_colors = torch.full_like(flat_points, 0.5)
    else:
        flat_points = points.reshape(-1, 3)
        flat_colors = torch.full_like(flat_points, 0.5)

    finite = torch.isfinite(flat_points).all(dim=1)
    flat_points = flat_points[finite]
    flat_colors = flat_colors[finite]
    if flat_points.numel() == 0:
        return None, None
    colors_u8 = (flat_colors.clamp(0.0, 1.0).numpy() * 255.0 + 0.5).astype(np.uint8)
    return flat_points.numpy().astype(np.float32), colors_u8


def _write_bypass_point_ply(path, points, colors):
    from hyworld2.worldrecon.hyworldmirror.utils.save_utils import save_points_ply

    save_points_ply(Path(path), points, colors)
    return str(path)


def _export_bypass_memory_bank_pcds(bank, ply_data, raw_splats, downsampled_pts):
    points, colors = _bypass_points_and_colors(ply_data, raw_splats)
    if points is None or colors is None:
        raise ValueError(
            "HYWorld2 Memory Alignment bypass could not build aligned_pcd.ply: "
            "no Gaussian means or point geometry found in ply_data/raw_splats."
        )

    export_dir = Path(bank.root_path) / "render_results" / bank.results_path
    _ensure_dir(export_dir)
    bank.global_points = {
        "worldmirror_bypass": {
            "points": points,
            "colors": colors,
        }
    }
    bank.export_pcd(str(export_dir), N_points=max(1, int(downsampled_pts)))
    return str(export_dir / "aligned_pcd.ply"), int(points.shape[0])


def _save_worldmirror_ply_data_for_bypass(ply_data, output_path, raw_splats=None):
    existing = _first_existing_ply_path(ply_data, raw_splats)
    if existing:
        return existing

    if not isinstance(ply_data, dict):
        raise ValueError("HYWorld2 Memory Alignment bypass requires the WorldMirror PLY_DATA output connected to ply_data.")

    output_path = Path(output_path)
    _ensure_dir(output_path.parent)
    splats = ply_data.get("splats")
    if not isinstance(splats, dict) and isinstance(raw_splats, dict):
        splats = raw_splats.get("splats")
    means = _splat_tensor_any(splats, ("means", "xyz", "positions"), 3)
    scales = _splat_tensor_any(splats, ("scales", "scale"), 3)
    quats = _splat_tensor_any(splats, ("quats", "rotations", "rotation", "rots"), 4)
    opacities = _splat_tensor_any(splats, ("opacities", "opacity"), 1)
    colors = _splat_tensor_any(splats, ("sh", "features_dc"), 3)
    if colors is None:
        colors = _splat_tensor_any(splats, ("colors", "rgb", "rgbs"), 3)
        if colors is not None:
            sh_c0 = 0.28209479177387814
            colors = (colors - 0.5) / sh_c0
    if colors is not None and means is not None and colors.shape[0] == 1 and means.shape[0] > 1:
        colors = colors.repeat(means.shape[0], 1)

    if all(t is not None for t in (means, scales, quats, opacities, colors)):
        from hyworld2.worldrecon.hyworldmirror.utils.save_utils import _build_gs_ply_data

        count = min(means.shape[0], scales.shape[0], quats.shape[0], opacities.shape[0], colors.shape[0])
        ply = _build_gs_ply_data(
            means[:count],
            scales[:count].clamp_min(1e-8),
            quats[:count],
            colors[:count],
            opacities[:count].reshape(-1),
            quantile_threshold=1.0,
        )
        ply.write(str(output_path))
        return str(output_path)

    try:
        from .world_mirror_v1 import extract_splat_params
    except Exception:
        extract_splat_params = None
    if extract_splat_params is not None:
        params = extract_splat_params(ply_data)
        if params:
            from hyworld2.worldrecon.hyworldmirror.utils.save_utils import _build_gs_ply_data

            means, scales, quats, rgb, opacities = params
            sh_c0 = 0.28209479177387814
            colors = (rgb.detach().cpu().float() - 0.5) / sh_c0
            ply = _build_gs_ply_data(
                means.detach().cpu().float(),
                scales.detach().cpu().float().clamp_min(1e-8),
                quats.detach().cpu().float(),
                colors,
                opacities.detach().cpu().float().reshape(-1),
                quantile_threshold=1.0,
            )
            ply.write(str(output_path))
            return str(output_path)

    points, colors = _bypass_points_and_colors(ply_data, raw_splats)
    if points is not None and colors is not None:
        return _write_bypass_point_ply(output_path, points, colors)

    keys = sorted(str(k) for k in ply_data.keys())
    raw_keys = sorted(str(k) for k in raw_splats.keys()) if isinstance(raw_splats, dict) else []
    raise ValueError(
        "HYWorld2 Memory Alignment bypass could not find Gaussian splats or point geometry "
        f"in connected ply_data/raw_splats. ply_data keys={keys}, raw_splats keys={raw_keys}"
    )


def _to_c2w(poses):
    if not isinstance(poses, torch.Tensor):
        return torch.empty((0, 4, 4), dtype=torch.float32)
    poses = poses.detach().cpu().float()
    if poses.dim() == 4 and poses.shape[0] == 1:
        poses = poses[0]
    if poses.dim() == 2:
        poses = poses.unsqueeze(0)
    if poses.shape[-2:] == (3, 4):
        bottom = torch.tensor([0, 0, 0, 1], dtype=poses.dtype).view(1, 1, 4).repeat(poses.shape[0], 1, 1)
        poses = torch.cat([poses, bottom], dim=1)
    return poses


def _to_intrinsics(intrs):
    if not isinstance(intrs, torch.Tensor):
        return torch.empty((0, 3, 3), dtype=torch.float32)
    intrs = intrs.detach().cpu().float()
    if intrs.dim() == 4 and intrs.shape[0] == 1:
        intrs = intrs[0]
    if intrs.dim() == 2:
        intrs = intrs.unsqueeze(0)
    return intrs


_WORLDSTEREO_TO_WORLDMIRROR_BASIS = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=torch.float32,
)


def _worldstereo_w2c_to_worldmirror_c2w(w2c):
    """Convert WorldStereo Z-up W2C cameras to WorldMirror panorama C2W poses."""
    c2w = torch.linalg.inv(w2c.detach().cpu().float())
    return _worldstereo_c2w_to_worldmirror_c2w(c2w)


def _worldstereo_c2w_to_worldmirror_c2w(c2w):
    """Convert WorldStereo/worldgen Z-up C2W cameras to WorldMirror panorama C2W poses."""
    c2w = c2w.detach().cpu().float()
    basis = _WORLDSTEREO_TO_WORLDMIRROR_BASIS.to(dtype=c2w.dtype)
    return basis @ c2w


def _load_camera_tensors_from_json(path):
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
    if not poses:
        return torch.empty((0, 4, 4)), torch.empty((0, 3, 3))
    return torch.from_numpy(np.stack(poses)).float(), torch.from_numpy(np.stack(intrs)).float()


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


def _official_strategy_settings(max_steps):
    refine_stop = max(1, min(750, int(max_steps) - 1))
    return {
        "refine_start_iter": 150,
        "refine_stop_iter": refine_stop,
        "refine_every": 100,
        "refine_scale2d_stop_iter": refine_stop,
        "reset_every": 99990,
        "grow_grad2d": 0.0001,
        "prune_scale3d": 0.1,
    }


def _apply_official_strategy_preset(strategy, max_steps):
    for name, value in _official_strategy_settings(max_steps).items():
        if hasattr(strategy, name):
            setattr(strategy, name, value)


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
        if _normal_tensor_is_usable(torch.from_numpy(arr)):
            return True
    return False


def _load_metric_depth16_preview(path):
    with Image.open(path) as depth_pil:
        arr = np.asarray(depth_pil)
    if arr.ndim != 2 or arr.dtype.itemsize < 2:
        return None
    depth = np.frombuffer(arr.astype(np.uint16, copy=False), dtype=np.float16).astype(np.float32)
    return depth.reshape(arr.shape)


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


def _parse_int_list(value):
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(v) for v in str(value).replace(",", " ").split() if str(v).strip()]


def _worldstereo_keyframe_indices(num_frames, device=None):
    keyframe_count = max(1, (int(num_frames) - 1) // 4 + 1)
    indices = torch.linspace(0, int(num_frames) - 1, keyframe_count, device=device).round().long()
    return torch.unique_consecutive(indices.clamp(0, int(num_frames) - 1))


def _slice_render_conditioning_to_keyframes(pipeline_kwargs):
    render_video = pipeline_kwargs.get("render_video")
    num_frames = int(pipeline_kwargs.get("num_frames") or 0)
    if not isinstance(render_video, torch.Tensor) or num_frames <= 0:
        return
    keyframe_indices = _worldstereo_keyframe_indices(num_frames, device=render_video.device)
    if render_video.shape[2] == keyframe_indices.numel():
        return
    old_frames = int(render_video.shape[2])
    pipeline_kwargs["render_video"] = render_video.index_select(2, keyframe_indices).contiguous()
    for key in ("render_mask", "camera_embedding"):
        value = pipeline_kwargs.get(key)
        if isinstance(value, torch.Tensor) and value.shape[2] == old_frames:
            pipeline_kwargs[key] = value.index_select(2, keyframe_indices.to(value.device)).contiguous()
    camera_qt = pipeline_kwargs.get("camera_qt")
    if isinstance(camera_qt, torch.Tensor) and camera_qt.shape[1] == old_frames:
        pipeline_kwargs["camera_qt"] = camera_qt.index_select(1, keyframe_indices.to(camera_qt.device)).contiguous()
    ref_index = pipeline_kwargs.get("ref_index")
    max_ref_index = max(0, keyframe_indices.numel() - 2)
    if isinstance(ref_index, torch.Tensor) and ref_index.numel() > 0 and max_ref_index < 19:
        pipeline_kwargs["ref_index"] = torch.round(ref_index.float() * (float(max_ref_index) / 19.0)).long().clamp_(0, max_ref_index)
    print(f"[HYWorld2] Render VAE conditioning sliced to keyframes: {old_frames} -> {pipeline_kwargs['render_video'].shape[2]}")


def _sample_camera_tensors_to_frame_count(w2cs, Ks, frame_count):
    frame_count = int(frame_count)
    if frame_count <= 0 or w2cs.shape[0] == frame_count:
        return w2cs, Ks
    indices = np.linspace(0, w2cs.shape[0] - 1, frame_count, dtype=int)
    indices = torch.as_tensor(indices, dtype=torch.long, device=w2cs.device)
    return w2cs.index_select(0, indices), Ks.index_select(0, indices)


def _load_video_frames(path):
    _ensure_worldgen_path()
    from hyworld2.worldgen.src.general_utils import load_video

    return load_video(str(path))


def _export_video(frames, path, fps=16):
    from diffusers.utils import export_to_video

    path = Path(path)
    _ensure_dir(path.parent)
    export_to_video(frames, str(path), fps=fps)


def _encode_prompt_cache(pipeline, prompt, negative_prompt, do_classifier_free_guidance, device):
    execution_device = getattr(pipeline, "_execution_device", None)
    if callable(execution_device):
        execution_device = execution_device()
    if execution_device is None:
        execution_device = device
    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
            prompt=prompt if prompt else "",
            negative_prompt=negative_prompt if negative_prompt else None,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=1,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            max_sequence_length=512,
            device=torch.device(execution_device),
        )
    if hasattr(pipeline, "maybe_free_model_hooks"):
        with contextlib.suppress(Exception):
            pipeline.maybe_free_model_hooks()
    prompt_embeds = prompt_embeds.detach().to("cpu")
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.detach().to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prompt_embeds, negative_prompt_embeds


def _build_prompt_cache(worldstereo_model, workspace, render_list, model_type, device):
    prompt_cache = {}
    pipeline = worldstereo_model["pipeline"]
    cfg = getattr(pipeline, "cfg", None) or getattr(worldstereo_model.get("worldstereo"), "cfg", None)
    negative_prompt = ""
    if cfg is not None:
        negative_prompt = getattr(cfg, "negative_prompt", "") or ""
    do_cfg = model_type != "worldstereo-memory-dmd"
    render_root = Path(workspace["scene_dir"]) / "render_results"
    for render_path in render_list:
        parts = Path(render_path).parts
        view_id, traj_id = parts[-3], parts[-2]
        caption_path = render_root / view_id / traj_id / "traj_caption.json"
        if not caption_path.exists():
            raise FileNotFoundError(
                f"Missing {caption_path}. Run HYWorld2 QwenVL in trajectory_caption mode before World Expansion; fallback prompts are disabled."
            )
        with open(caption_path, "r", encoding="utf-8") as handle:
            prompt = json.load(handle).get("prompt", "")
        prompt_cache[(view_id, traj_id)] = _encode_prompt_cache(
            pipeline,
            prompt,
            negative_prompt,
            do_classifier_free_guidance=do_cfg,
            device=device,
        )
    return prompt_cache


def _worldstereo_cfg(worldstereo_model):
    pipeline = worldstereo_model["pipeline"]
    cfg = getattr(worldstereo_model.get("worldstereo"), "cfg", None)
    if cfg is None:
        cfg = getattr(pipeline, "cfg", None)
    if cfg is None:
        raise RuntimeError("WORLDSTEREO_MODEL has no cfg; cannot run HYWorld2 memory mode.")
    return cfg


def _safe_json_dumps(value):
    def default(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, torch.Tensor):
            return list(obj.shape)
        return str(obj)

    return json.dumps(value, indent=2, default=default)


class HYWorld2Workspace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["create", "load_existing", "resume"], {"default": "resume"}),
                "workspace_name": ("STRING", {"default": "comfy_worldgen"}),
            },
            "optional": {
                "root_dir": ("STRING", {"default": ""}),
                "scene_dir": ("STRING", {"default": ""}),
                "panorama": ("IMAGE",),
                "scene_type": (["unknown", "indoor", "outdoor"], {"default": "unknown"}),
                "result_name": ("STRING", {"default": "worldstereo-memory-dmd"}),
            },
        }

    RETURN_TYPES = ("HYWORLD2_WORKSPACE", "STRING")
    RETURN_NAMES = ("workspace", "info")
    FUNCTION = "build"
    CATEGORY = "VNCCS/HYWorld2"

    def build(self, mode, workspace_name, root_dir="", scene_dir="", panorama=None, scene_type="unknown", result_name="worldstereo-memory-dmd"):
        if mode in ("load_existing", "resume") and str(scene_dir).strip():
            scene = Path(scene_dir)
        else:
            root = Path(root_dir) if str(root_dir).strip() else _output_root() / "hyworld2_worldgen"
            scene = root / _sanitize_name(workspace_name, "comfy_worldgen")
        _ensure_dir(scene)
        _ensure_dir(scene / "render_results")
        if panorama is not None:
            frames = _image_tensor_to_pil_list(panorama)
            if frames:
                frames[0].save(scene / "panorama.png")
        meta_path = scene / "meta_info.json"
        meta = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                meta.update(loaded)
        if scene_type != "unknown" or "scene_type" not in meta:
            meta["scene_type"] = scene_type
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        workspace = {
            "scene_dir": str(scene),
            "render_results_dir": str(scene / "render_results"),
            "workspace_name": workspace_name,
            "result_name": result_name,
            "scene_type": meta.get("scene_type", "unknown"),
        }
        return (workspace, _safe_json_dumps(workspace))


class HYWorld2QwenVL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workspace": ("HYWORLD2_WORKSPACE",),
                "mode": (["scene_objects", "trajectory_caption", "prompt_refine"], {"default": "trajectory_caption"}),
                "model_id": ("STRING", {"default": "Qwen/Qwen2.5-VL-3B-Instruct"}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "images": ("IMAGE",),
                "trajectory_set": ("HYWORLD2_TRAJECTORY_SET",),
                "device": ("STRING", {"default": "cuda"}),
                "max_new_tokens": ("INT", {"default": 256, "min": 16, "max": 4096, "step": 16}),
                "write_results": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("HYWORLD2_LLM_CONTEXT", "STRING")
    RETURN_NAMES = ("llm_context", "text")
    FUNCTION = "run"
    CATEGORY = "VNCCS/HYWorld2"

    def _generate(self, model_id, prompt, images=None, device="cuda", max_new_tokens=256):
        try:
            from transformers import AutoProcessor
            try:
                from transformers import AutoModelForImageTextToText as AutoModel
            except ImportError:
                from transformers import AutoModelForVision2Seq as AutoModel
        except Exception as exc:
            raise ImportError("QwenVL requires transformers with vision-language model support. Install project requirements.") from exc
        selected_device = device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if selected_device.startswith("cuda") else torch.float32,
            device_map=selected_device if selected_device.startswith("cuda") else None,
            trust_remote_code=True,
        )
        if not selected_device.startswith("cuda"):
            model = model.to(selected_device)
        pil_images = _image_tensor_to_pil_list(images) if images is not None else []
        content = []
        for image in pil_images[:8]:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=pil_images[:8] or None, return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False)
        generated = generated[:, inputs["input_ids"].shape[1]:]
        result = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    def run(self, workspace, mode, model_id, prompt, images=None, trajectory_set=None, device="cuda", max_new_tokens=256, write_results=True):
        scene = Path(workspace["scene_dir"])
        if not prompt.strip():
            if mode == "scene_objects":
                prompt = "Analyze this panoramic scene. Return concise JSON with scene_type, objects, navigable_areas, and visual_style."
            elif mode == "trajectory_caption":
                prompt = "Describe the visible trajectory render as a concise image generation prompt. Return only the prompt text."
            else:
                raise ValueError("prompt_refine requires a non-empty prompt; fallback prompts are disabled.")
        text = self._generate(model_id, prompt, images=images, device=device, max_new_tokens=max_new_tokens)
        context = {"mode": mode, "text": text, "model_id": model_id}
        if write_results:
            if mode == "scene_objects":
                out_path = scene / "hyworld2_qwenvl_scene.json"
                try:
                    parsed = json.loads(text)
                except Exception:
                    parsed = {"raw": text}
                with open(out_path, "w", encoding="utf-8") as handle:
                    json.dump(parsed, handle, indent=2)
                context["scene_objects_path"] = str(out_path)
            elif mode == "trajectory_caption" and trajectory_set:
                render_list = trajectory_set.get("render_list", [])
                for render_path in render_list:
                    path = Path(render_path)
                    caption_path = path.parent / "traj_caption.json"
                    with open(caption_path, "w", encoding="utf-8") as handle:
                        json.dump({"prompt": text, "source": "HYWorld2 QwenVL"}, handle, indent=2)
                context["captions_written"] = len(render_list)
        return (context, text)


class HYWorld2Trajectories:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workspace": ("HYWORLD2_WORKSPACE",),
                "mode": (["sort_existing", "generate_from_plan", "select_range", "debug_single"], {"default": "sort_existing"}),
            },
            "optional": {
                "start_index": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "max_count": ("INT", {"default": 0, "min": 0, "max": 100000}),
            },
        }

    RETURN_TYPES = ("HYWORLD2_TRAJECTORY_SET", "STRING")
    RETURN_NAMES = ("trajectory_set", "info")
    FUNCTION = "run"
    CATEGORY = "VNCCS/HYWorld2"

    def run(self, workspace, mode, start_index=0, max_count=0):
        if mode == "generate_from_plan":
            raise NotImplementedError("Native trajectory generation from Qwen plan is not implemented yet; use sort_existing after HYWorld2 trajectory render data exists.")
        _ensure_worldgen_path()
        from hyworld2.worldgen.src.data_utils import sort_trajs

        render_root = Path(workspace["scene_dir"]) / "render_results"
        render_list = sort_trajs(str(render_root))
        if mode == "debug_single" and render_list:
            render_list = [render_list[int(start_index)]]
        elif mode == "select_range":
            end = None if int(max_count) <= 0 else int(start_index) + int(max_count)
            render_list = render_list[int(start_index):end]
        data = {"workspace": workspace, "render_list": render_list, "count": len(render_list)}
        return (data, _safe_json_dumps(data))


class HYWorld2MemoryBank:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workspace": ("HYWORLD2_WORKSPACE",),
                "mode": (["initialize", "load_cached"], {"default": "initialize"}),
            },
            "optional": {
                "image_width": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "image_height": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "nframe": ("INT", {"default": 0, "min": 0, "max": 257}),
                "max_reference": ("INT", {"default": 8, "min": 1, "max": 64}),
                "align_nframe": ("INT", {"default": 8, "min": 1, "max": 64}),
                "downsampled_pts": ("INT", {"default": 2_000_000, "min": 1, "max": 50_000_000, "step": 100000}),
                "kb_anomaly_percentile": ("FLOAT", {"default": 90.0, "min": 1.0, "max": 100.0, "step": 0.5}),
            },
        }

    RETURN_TYPES = ("HYWORLD2_MEMORY_BANK", "STRING")
    RETURN_NAMES = ("memory_bank", "info")
    FUNCTION = "run"
    CATEGORY = "VNCCS/HYWorld2"

    def run(self, workspace, mode, image_width=0, image_height=0, nframe=0, max_reference=8, align_nframe=8, downsampled_pts=2_000_000, kb_anomaly_percentile=90.0):
        _ensure_worldgen_path()
        from hyworld2.worldgen.src.retrieval_wm import PanoramaMemoryBank

        scene = Path(workspace["scene_dir"])
        if image_width <= 0 or image_height <= 0:
            from imagesize import get as image_size

            start_frames = sorted((scene / "render_results").glob("*/start_frame.png"))
            if start_frames:
                image_width, image_height = image_size(str(start_frames[0]))
            else:
                pano = scene / "panorama.png"
                image_width, image_height = image_size(str(pano))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        bank = PanoramaMemoryBank(
            root_path=str(scene),
            image_width=int(image_width),
            image_height=int(image_height),
            device=device,
            nframe=int(nframe) if int(nframe) > 0 else 21,
            max_reference=int(max_reference),
            align_nframe=int(align_nframe),
            rank=0,
            world_size=1,
            results_name=workspace.get("result_name", "worldstereo-memory-dmd"),
            valid_threshold=0.15,
            pts_num=int(downsampled_pts),
            kb_anomaly_percentile=float(kb_anomaly_percentile),
        )
        state = {"workspace": workspace, "bank": bank, "device": str(device), "image_width": int(image_width), "image_height": int(image_height)}
        info = {
            "scene_dir": str(scene),
            "device": str(device),
            "memory_size": int(bank.mem_size),
            "results_path": bank.results_path,
        }
        return (state, _safe_json_dumps(info))


class HYWorld2WorldExpansion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workspace": ("HYWORLD2_WORKSPACE",),
                "memory_bank": ("HYWORLD2_MEMORY_BANK",),
                "trajectory_set": ("HYWORLD2_TRAJECTORY_SET",),
                "model": ("WORLDSTEREO_MODEL",),
            },
            "optional": {
                "caption_mode": (["qwenvl_missing", "qwenvl_overwrite", "existing_files_only"], {"default": "qwenvl_missing"}),
                "qwen_model_id": ("STRING", {"default": "Qwen/Qwen2.5-VL-3B-Instruct"}),
                "qwen_device": ("STRING", {"default": "cuda"}),
                "qwen_max_new_tokens": ("INT", {"default": 192, "min": 16, "max": 2048, "step": 16}),
                "seed": ("INT", {"default": 1024, "min": 0, "max": 2**31 - 1}),
                "skip_existing": ("BOOLEAN", {"default": True}),
                "max_trajectories": ("INT", {"default": 0, "min": 0, "max": 100000}),
            },
        }

    RETURN_TYPES = ("HYWORLD2_MEMORY_BANK", "STRING")
    RETURN_NAMES = ("memory_bank", "info")
    FUNCTION = "run"
    CATEGORY = "VNCCS/HYWorld2"

    def _ensure_captions(self, workspace, render_list, caption_mode, qwen_model_id, qwen_device, qwen_max_new_tokens):
        if caption_mode == "existing_files_only":
            return []
        qwen = HYWorld2QwenVL()
        written = []
        for render_path in render_list:
            traj_dir = Path(render_path).parent
            caption_path = traj_dir / "traj_caption.json"
            if caption_path.exists() and caption_mode != "qwenvl_overwrite":
                continue
            frames = _load_video_frames(render_path)
            sample = []
            if frames:
                sample = [frames[0]]
                if len(frames) > 2:
                    sample.append(frames[len(frames) // 2])
                if len(frames) > 1:
                    sample.append(frames[-1])
            start_frame = traj_dir.parent / "start_frame.png"
            if start_frame.exists():
                sample.insert(0, Image.open(start_frame).convert("RGB"))
            if not sample:
                raise FileNotFoundError(f"Cannot caption trajectory; no render frames found for {render_path}")
            prompt = (
                "Create a concise photorealistic video generation prompt for this HYWorld2 camera "
                "trajectory. Describe stable scene layout, materials, lighting, and newly visible areas. "
                "Return only the prompt text, no JSON and no commentary."
            )
            text = qwen._generate(
                qwen_model_id,
                prompt,
                images=_pil_list_to_image_tensor(sample[:4]),
                device=qwen_device,
                max_new_tokens=qwen_max_new_tokens,
            )
            if not text.strip():
                raise RuntimeError(f"QwenVL returned an empty caption for {render_path}")
            with open(caption_path, "w", encoding="utf-8") as handle:
                json.dump({"prompt": text.strip(), "source": "HYWorld2 World Expansion QwenVL"}, handle, indent=2)
            written.append(str(caption_path))
        return written

    def run(
        self,
        workspace,
        memory_bank,
        trajectory_set,
        model,
        caption_mode="qwenvl_missing",
        qwen_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        qwen_device="cuda",
        qwen_max_new_tokens=192,
        seed=1024,
        skip_existing=True,
        max_trajectories=0,
    ):
        _ensure_worldgen_path()
        from hyworld2.worldgen.src.data_utils import load_mutli_traj_dataset

        bank = memory_bank["bank"]
        device = torch.device(memory_bank.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
        pipeline = model["pipeline"]
        cfg = _worldstereo_cfg(model)
        model_type = model.get("model_type") or workspace.get("result_name", "worldstereo-memory-dmd")
        if int(getattr(bank, "nframe", 0)) != int(getattr(cfg, "nframe", getattr(bank, "nframe", 21))):
            bank.nframe = int(getattr(cfg, "nframe", bank.nframe))
        render_list = list(trajectory_set.get("render_list", []))
        if int(max_trajectories) > 0:
            render_list = render_list[: int(max_trajectories)]
        captions_written = self._ensure_captions(
            workspace,
            render_list,
            caption_mode,
            qwen_model_id,
            qwen_device,
            int(qwen_max_new_tokens),
        )
        prompt_cache = _build_prompt_cache(model, workspace, render_list, model_type, device)
        generator = torch.Generator(device=device).manual_seed(int(seed))
        autocast_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        completed = []
        for render_path in render_list:
            render_parts = Path(render_path).parts
            view_id, traj_id = render_parts[-3], render_parts[-2]
            traj_dir = Path(workspace["scene_dir"]) / "render_results" / view_id / traj_id
            result_path = traj_dir / f"{model_type}_result.mp4"
            camera_data = json.load(open(traj_dir / "camera.json", "r", encoding="utf-8"))
            tar_w2cs = torch.from_numpy(np.asarray(camera_data["extrinsic"], dtype=np.float32)).to(device)
            tar_Ks = torch.from_numpy(np.asarray(camera_data["intrinsic"], dtype=np.float32)).to(device)
            if skip_existing and result_path.exists():
                frames = _load_video_frames(result_path)
                update_w2cs, update_Ks = _sample_camera_tensors_to_frame_count(tar_w2cs, tar_Ks, len(frames))
                bank.update_memory(frames, update_w2cs, update_Ks, view_id=view_id, traj_id=traj_id)
                completed.append(str(result_path))
                continue
            retrieved_frames, ref_index, ref_index_dict, ref_w2cs, _ = bank.retrieval(tar_w2cs, tar_Ks, view_id=view_id, traj_id=traj_id)
            memory_dir = traj_dir / "memory_inputs"
            _ensure_dir(memory_dir)
            _export_video(retrieved_frames / 255.0, memory_dir / f"{model_type}.mp4", fps=16)
            with open(memory_dir / f"{model_type}_ref_index.json", "w", encoding="utf-8") as handle:
                json.dump(ref_index_dict, handle, indent=2)
            with open(memory_dir / f"{model_type}_ref_w2cs.json", "w", encoding="utf-8") as handle:
                json.dump(ref_w2cs.detach().cpu().numpy().tolist(), handle, indent=2)
            meta_data = load_mutli_traj_dataset(
                cfg=cfg,
                input_path=str(Path(workspace["scene_dir"]) / "render_results"),
                output_path=str(Path(workspace["scene_dir"]) / "render_results"),
                view_id=view_id,
                traj_id=traj_id,
                device=device,
                ref_index=ref_index,
                model_type=model_type,
                task_type="panorama",
            )
            pipeline_kwargs = {k: v for k, v in meta_data.items() if v is not None}
            pipeline_kwargs.update(generator=generator, output_type="pt", latent_cond_mode=getattr(cfg, "latent_cond_mode", "first_frame_only"))
            cached_prompt_embeds, cached_negative_prompt_embeds = prompt_cache[(view_id, traj_id)]
            pipeline_kwargs.pop("prompt", None)
            pipeline_kwargs.update(
                prompt=None,
                negative_prompt=None,
                prompt_embeds=cached_prompt_embeds.to(device),
                negative_prompt_embeds=cached_negative_prompt_embeds.to(device) if cached_negative_prompt_embeds is not None else None,
            )
            if model_type == "worldstereo-memory-dmd":
                pipeline_kwargs["mode"] = "test"
                _slice_render_conditioning_to_keyframes(pipeline_kwargs)
            else:
                pipeline_kwargs["guidance_scale"] = 5.0
            with torch.no_grad(), torch.autocast(device.type, dtype=autocast_dtype, enabled=device.type == "cuda"):
                output = pipeline(**pipeline_kwargs).frames[0].float()
            frames_np = output.permute(0, 2, 3, 1).detach().cpu().clamp(0, 1).numpy()
            _export_video(frames_np, result_path, fps=16)
            gen_frames = _load_video_frames(result_path)
            update_w2cs, update_Ks = _sample_camera_tensors_to_frame_count(tar_w2cs, tar_Ks, len(gen_frames))
            bank.update_memory(gen_frames, update_w2cs, update_Ks, view_id=view_id, traj_id=traj_id)
            completed.append(str(result_path))
            del output, pipeline_kwargs, meta_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        memory_bank["bank"] = bank
        del pipeline
        _release_model_memory("HYWorld2 World Expansion")
        return (memory_bank, _safe_json_dumps({"completed": completed, "count": len(completed), "captions_written": captions_written}))


class HYWorld2PrepareWorldMirrorBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"memory_bank": ("HYWORLD2_MEMORY_BANK",)}}

    RETURN_TYPES = ("IMAGE", "TENSOR", "TENSOR", "HYWORLD2_WORLDMIRROR_BATCH", "STRING")
    RETURN_NAMES = ("images", "camera_poses", "camera_intrinsics", "worldmirror_batch", "info")
    FUNCTION = "run"
    CATEGORY = "VNCCS/HYWorld2"

    def run(self, memory_bank):
        bank = memory_bank["bank"]
        world_mirror_dir = Path(bank.root_path) / "render_results" / bank.results_path / "world_mirror_data"
        images_dir = _ensure_dir(world_mirror_dir / "images")
        cameras = {"num_cameras": 0, "extrinsics": [], "intrinsics": []}
        name_map = {}
        images = []
        poses = []
        intrs = []
        entries = []
        for gi, fname in enumerate(bank.fnames):
            view_id, traj_id, frame_id = fname.split("/")
            camera_id = f"pano-{frame_id}" if view_id.startswith("render_results") else f"{view_id}-{traj_id}-{frame_id}"
            entries.append((camera_id, fname, gi))

        # WorldMirror writes depth_NNNN by sorted image/camera id. Keep the tensor
        # batch, cameras.json, files, and name_map in that exact same order.
        entries.sort(key=lambda item: item[0])
        for index, (camera_id, fname, gi) in enumerate(entries):
            view_id, traj_id, frame_id = fname.split("/")
            image = bank.ref_frames[gi].convert("RGB")
            image.save(images_dir / f"{camera_id}.png")
            pose = _worldstereo_w2c_to_worldmirror_c2w(bank.ref_w2cs[gi])
            cameras["extrinsics"].append({"camera_id": camera_id, "matrix": pose.numpy().tolist()})
            cameras["intrinsics"].append({"camera_id": camera_id, "matrix": bank.ref_Ks[gi].detach().cpu().numpy().tolist()})
            images.append(image)
            poses.append(pose)
            intrs.append(bank.ref_Ks[gi].detach().cpu())
            name_map[fname] = str(index).zfill(4)
        cameras["num_cameras"] = len(images)
        with open(world_mirror_dir / "cameras.json", "w", encoding="utf-8") as handle:
            json.dump(cameras, handle, indent=2)
        with open(world_mirror_dir / "name_map.json", "w", encoding="utf-8") as handle:
            json.dump(name_map, handle, indent=2)
        bank.world_mirror_dir = str(world_mirror_dir)
        bank.name_map = name_map
        batch = {"memory_bank": memory_bank, "world_mirror_dir": str(world_mirror_dir), "name_map": name_map}
        return (_pil_list_to_image_tensor(images), torch.stack(poses).float(), torch.stack(intrs).float(), batch, _safe_json_dumps({"frames": len(images), "world_mirror_dir": world_mirror_dir}))


class HYWorld2MemoryAlignment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "worldmirror_batch": ("HYWORLD2_WORLDMIRROR_BATCH",),
                "raw_splats": ("VNCCS_SPLAT",),
                "mode": (["consume_worldmirror_depths", "align_and_export", "bypass"], {"default": "align_and_export"}),
            },
            "optional": {
                "ply_data": ("PLY_DATA",),
                "downsampled_pts": ("INT", {"default": 2_000_000, "min": 1, "max": 50_000_000, "step": 100000}),
                "debug_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("HYWORLD2_MEMORY_BANK", "STRING", "STRING")
    RETURN_NAMES = ("memory_bank", "aligned_ply", "info")
    FUNCTION = "run"
    CATEGORY = "VNCCS/HYWorld2"

    def run(self, worldmirror_batch, raw_splats, mode, ply_data=None, downsampled_pts=2_000_000, debug_mode=False):
        memory_bank = worldmirror_batch["memory_bank"]
        bank = memory_bank["bank"]
        world_mirror_dir = Path(worldmirror_batch["world_mirror_dir"])
        depth_dir = _ensure_dir(world_mirror_dir / "results" / "depth")
        depths, depth_source = _raw_worldmirror_depths_to_numpy(raw_splats)
        if not depths and mode != "bypass":
            raise ValueError("HYWorld2 Memory Alignment requires raw_splats with metric float depth: raw_splats.gs_depth or raw_splats.depth.")
        for index, depth in enumerate(depths):
            np.save(depth_dir / f"depth_{index:04d}.npy", depth)
        if mode == "align_and_export":
            _ensure_single_process_dist(bank)
            bank.alignment(debug_mode=bool(debug_mode))
            export_dir = Path(bank.root_path) / "render_results" / bank.results_path
            _ensure_dir(export_dir)
            bank.export_pcd(str(export_dir), N_points=int(downsampled_pts))
            aligned = str(export_dir / "aligned_pcd.ply")
            bypass_source_points = 0
        elif mode == "bypass":
            aligned, bypass_source_points = _export_bypass_memory_bank_pcds(bank, ply_data, raw_splats, downsampled_pts)
        else:
            aligned = ""
            bypass_source_points = 0
        memory_bank["bank"] = bank
        return (
            memory_bank,
            aligned,
            _safe_json_dumps(
                {
                    "mode": mode,
                    "depths_written": len(depths),
                    "depth_source": depth_source,
                    "aligned_ply": aligned,
                    "bypass_source_points": bypass_source_points,
                    "alignment_ran": mode == "align_and_export",
                }
            ),
        )


class HYWorld2GSData:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workspace": ("HYWORLD2_WORKSPACE",),
                "mode": (["build", "validate", "repair_metadata"], {"default": "build"}),
            },
            "optional": {
                "memory_bank": ("HYWORLD2_MEMORY_BANK",),
                "result_name": ("STRING", {"default": ""}),
                "out_name": ("STRING", {"default": "gs_data"}),
                "save_normal": ("BOOLEAN", {"default": True}),
                "split_sky": ("BOOLEAN", {"default": True}),
                "split_align": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("HYWORLD2_GS_DATA", "STRING")
    RETURN_NAMES = ("gs_data", "info")
    FUNCTION = "run"
    CATEGORY = "VNCCS/HYWorld2"

    def run(self, workspace, mode, memory_bank=None, result_name="", out_name="gs_data", save_normal=True, split_sky=True, split_align=False):
        scene = Path(workspace["scene_dir"])
        gs_dir = scene / _sanitize_name(out_name, "gs_data")
        if mode == "validate":
            required = [gs_dir / "cameras.json", gs_dir / "points.ply", gs_dir / "images"]
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise FileNotFoundError(f"HYWorld2 GS data missing required files: {missing}")
            return ({"workspace": workspace, "gs_data_dir": str(gs_dir)}, _safe_json_dumps({"valid": True, "gs_data_dir": gs_dir}))
        if mode == "repair_metadata":
            meta = gs_dir / "meta_info.json"
            if not meta.exists():
                with open(meta, "w", encoding="utf-8") as handle:
                    json.dump({"scene_type": workspace.get("scene_type", "unknown")}, handle, indent=2)
            return ({"workspace": workspace, "gs_data_dir": str(gs_dir)}, _safe_json_dumps({"repaired": True, "gs_data_dir": gs_dir}))
        _ensure_worldgen_path()
        import hyworld2.worldgen.gen_gs_data as gen_gs_data

        if not hasattr(gen_gs_data, "run_gen_gs_data"):
            raise RuntimeError("gen_gs_data.py must expose run_gen_gs_data for native node execution.")
        result = gen_gs_data.run_gen_gs_data(
            root_path=str(scene),
            out_name=out_name,
            result_name=result_name or workspace.get("result_name", "worldstereo-memory-dmd"),
            save_normal=bool(save_normal),
            split_sky=bool(split_sky),
            split_align=bool(split_align),
            world_size=1,
        )
        gs_dir = Path(result["output_path"])
        return ({"workspace": workspace, "gs_data_dir": str(gs_dir)}, _safe_json_dumps(result))


class HYWorld2Train3DGS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gs_data": ("HYWORLD2_GS_DATA",),
            },
            "optional": {
                "max_steps": ("INT", {"default": 1500, "min": 1, "max": 100000, "step": 100}),
                "save_steps": ("STRING", {"default": "1500"}),
                "eval_steps": ("STRING", {"default": "1500"}),
                "ply_steps": ("STRING", {"default": "1500"}),
                "downsample_pts_num": ("INT", {"default": 1_000_000, "min": 1, "max": 50_000_000, "step": 100000}),
                "save_ply": ("BOOLEAN", {"default": True}),
                "disable_video": ("BOOLEAN", {"default": True}),
                "disable_viewer": ("BOOLEAN", {"default": True}),
                "depth_loss": ("BOOLEAN", {"default": True}),
                "normal_loss": ("BOOLEAN", {"default": True}),
                "sky_depth_from_pcd": ("BOOLEAN", {"default": True}),
                "use_scale_regularization": ("BOOLEAN", {"default": True}),
                "use_mask_gaussian": ("BOOLEAN", {"default": False}),
                "mask_export_stochastic": ("BOOLEAN", {"default": False}),
                "antialiased": ("BOOLEAN", {"default": True}),
                "official_strategy_preset": ("BOOLEAN", {"default": True}),
                "normalize_world_space": ("BOOLEAN", {"default": True}),
                "perceptual_loss": (["lpips_vgg", "lpips_alex", "simple", "off"], {"default": "lpips_vgg"}),
            },
        }

    RETURN_TYPES = ("STRING", "TENSOR", "TENSOR", "STRING", "STRING")
    RETURN_NAMES = ("ply_path", "camera_poses", "camera_intrinsics", "train_dir", "info")
    FUNCTION = "run"
    CATEGORY = "VNCCS/HYWorld2"
    OUTPUT_NODE = True

    def run(self, gs_data, max_steps=1500, save_steps="1500", eval_steps="1500", ply_steps="1500", downsample_pts_num=1_000_000, save_ply=True, disable_video=True, disable_viewer=True, depth_loss=True, normal_loss=True, sky_depth_from_pcd=True, use_scale_regularization=True, use_mask_gaussian=False, mask_export_stochastic=False, antialiased=True, official_strategy_preset=True, normalize_world_space=True, perceptual_loss="lpips_vgg"):
        _ensure_worldgen_path()
        import hyworld2.worldgen.world_gs_trainer as trainer
        from gsplat.strategy import DefaultStrategy

        data_dir = Path(gs_data["gs_data_dir"])
        out_dir = data_dir.parent / "gs_results"
        _reset_dir(out_dir, "HYWorld2 train_dir")
        _ensure_scene_type_meta(data_dir)
        strategy = DefaultStrategy(verbose=True)
        if official_strategy_preset:
            _apply_official_strategy_preset(strategy, max_steps)
        cfg = trainer.Config(strategy=strategy)
        cfg.data_dir = str(data_dir)
        cfg.result_dir = str(out_dir)
        cfg.max_steps = int(max_steps)
        cfg.save_steps = _parse_int_list(save_steps)
        cfg.eval_steps = _parse_int_list(eval_steps)
        cfg.ply_steps = _parse_int_list(ply_steps)
        cfg.downsample_pts_num = int(downsample_pts_num)
        cfg.save_ply = bool(save_ply)
        cfg.disable_video = bool(disable_video)
        cfg.disable_viewer = bool(disable_viewer)
        if hasattr(cfg, "dataloader_num_workers"):
            cfg.dataloader_num_workers = 0
        depth_files_valid = _has_valid_depth_files(data_dir)
        normal_files_valid = _has_valid_normal_files(data_dir)
        cfg.depth_loss = bool(depth_loss and depth_files_valid)
        cfg.normal_loss = bool(normal_loss and normal_files_valid)
        cfg.sky_depth_from_pcd = bool(sky_depth_from_pcd and cfg.depth_loss and normal_files_valid)
        cfg.use_scale_regularization = bool(use_scale_regularization)
        cfg.use_mask_gaussian = bool(use_mask_gaussian)
        if hasattr(cfg, "mask_export_stochastic"):
            cfg.mask_export_stochastic = bool(mask_export_stochastic)
        cfg.antialiased = bool(antialiased)
        cfg.no_normalize = not bool(normalize_world_space)
        if perceptual_loss == "lpips_vgg":
            cfg.lpips_net = "vgg"
        elif perceptual_loss == "lpips_alex":
            cfg.lpips_net = "alex"
        elif perceptual_loss == "simple":
            cfg.lpips_net = "simple"
        else:
            cfg.lpips_net = "none"
            cfg.lpips_lambda1 = 0
            cfg.lpips_lambda2 = 0
        command_info = {
            "data_dir": str(data_dir),
            "result_dir": str(out_dir),
            "max_steps": int(max_steps),
            "save_steps": cfg.save_steps,
            "eval_steps": cfg.eval_steps,
            "ply_steps": cfg.ply_steps,
            "downsample_pts_num": int(cfg.downsample_pts_num),
            "save_ply": bool(cfg.save_ply),
            "disable_video": bool(cfg.disable_video),
            "disable_viewer": bool(cfg.disable_viewer),
            "dataloader_num_workers": int(getattr(cfg, "dataloader_num_workers", -1)),
            "depth_loss_requested": bool(depth_loss),
            "depth_loss_enabled": bool(cfg.depth_loss),
            "normal_loss_requested": bool(normal_loss),
            "normal_loss_enabled": bool(cfg.normal_loss),
            "sky_depth_from_pcd_requested": bool(sky_depth_from_pcd),
            "sky_depth_from_pcd_enabled": bool(cfg.sky_depth_from_pcd),
            "use_scale_regularization": bool(cfg.use_scale_regularization),
            "use_mask_gaussian": bool(cfg.use_mask_gaussian),
            "mask_export_stochastic": bool(getattr(cfg, "mask_export_stochastic", False)),
            "antialiased": bool(cfg.antialiased),
            "official_strategy_preset": bool(official_strategy_preset),
            "official_strategy_settings": _official_strategy_settings(max_steps) if official_strategy_preset else {},
            "normalize_world_space": bool(normalize_world_space),
            "perceptual_loss": perceptual_loss,
            "in_process": True,
        }
        with open(out_dir / "train_command.json", "w", encoding="utf-8") as handle:
            json.dump(command_info, handle, indent=2)
        if depth_loss and not cfg.depth_loss:
            print(f"[HYWorld2 Train 3DGS] depth_loss requested but valid metric float16-packed depths are missing under {data_dir / 'depths'}; disabling depth_loss.")
        if normal_loss and not cfg.normal_loss:
            print(f"[HYWorld2 Train 3DGS] normal_loss requested but normals are missing/constant under {data_dir / 'normals'}; disabling normal_loss.")
        if sky_depth_from_pcd and not cfg.sky_depth_from_pcd:
            print("[HYWorld2 Train 3DGS] sky_depth_from_pcd requested but depth/normal inputs are not usable; disabling sky_depth_from_pcd.")
        with torch.inference_mode(False), torch.enable_grad():
            trainer.main(0, 0, 1, cfg)
        ply_path = _find_latest_ply(out_dir)
        camera_json = out_dir / "ply" / "trainer_cameras.json"
        if not camera_json.exists():
            candidates = sorted((out_dir / "ply").glob("trainer_cameras_*.json")) if (out_dir / "ply").exists() else []
            camera_json = candidates[-1] if candidates else data_dir / "cameras.json"
        poses, intrs = _load_camera_tensors_from_json(camera_json) if camera_json.exists() else (torch.empty((0, 4, 4)), torch.empty((0, 3, 3)))
        info = {
            "ply_path": ply_path,
            "train_dir": str(out_dir),
            "camera_json": str(camera_json) if camera_json.exists() else "",
            "camera_pose_basis": "trainer_c2w",
        }
        return (ply_path, poses, intrs, str(out_dir), _safe_json_dumps(info))


NODE_CLASS_MAPPINGS = {
    "HYWorld2Workspace": HYWorld2Workspace,
    "HYWorld2QwenVL": HYWorld2QwenVL,
    "HYWorld2Trajectories": HYWorld2Trajectories,
    "HYWorld2MemoryBank": HYWorld2MemoryBank,
    "HYWorld2WorldExpansion": HYWorld2WorldExpansion,
    "HYWorld2PrepareWorldMirrorBatch": HYWorld2PrepareWorldMirrorBatch,
    "HYWorld2MemoryAlignment": HYWorld2MemoryAlignment,
    "HYWorld2GSData": HYWorld2GSData,
    "HYWorld2Train3DGS": HYWorld2Train3DGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HYWorld2Workspace": "HYWorld2 Workspace",
    "HYWorld2QwenVL": "HYWorld2 QwenVL",
    "HYWorld2Trajectories": "HYWorld2 Trajectories",
    "HYWorld2MemoryBank": "HYWorld2 Memory Bank",
    "HYWorld2WorldExpansion": "HYWorld2 World Expansion",
    "HYWorld2PrepareWorldMirrorBatch": "HYWorld2 Prepare WorldMirror Batch",
    "HYWorld2MemoryAlignment": "HYWorld2 Memory Alignment",
    "HYWorld2GSData": "HYWorld2 GS Data",
    "HYWorld2Train3DGS": "HYWorld2 Train 3DGS",
}
