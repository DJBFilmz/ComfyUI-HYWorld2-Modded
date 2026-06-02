"""
WorldMirror V2 ComfyUI nodes — uses HY-World-2.0 (tencent/HY-World-2.0).

Nodes:
  - VNCCS_LoadWorldMirrorV2Model   — download + load V2 model
  - VNCCS_WorldMirrorV2_3D         — V2 inference, PLY_DATA output
  - VNCCS_WorldMirrorV2_3D_Advanced — advanced V2 inference/debug copy
"""

import os
import sys
import copy
import numpy as np
import torch
from torchvision import transforms

# ── nodes/ -> repo root; hyworld2/ lives directly in repo root ────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── optional deps ──────────────────────────────────────────────────────────────
try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False

try:
    from gsplat.rendering import rasterization as _r
    GSPLAT_AVAILABLE = True
    del _r
except ImportError:
    GSPLAT_AVAILABLE = False

# ── V2 utilities ───────────────────────────────────────────────────────────────
try:
    from hyworld2.worldrecon.hyworldmirror.utils.inference_utils import (
        compute_filter_mask,           # high-level: pts_mask + gs_mask
        _compute_sky_mask_from_model,  # model-native sky mask (no ONNX needed)
        _voxel_prune_gaussians,
    )
    from hyworld2.worldrecon.hyworldmirror.utils.visual_util import (
        segment_sky,
        download_file_from_url,
    )
    from hyworld2.worldrecon.hyworldmirror.models.utils.geometry import depth_to_world_coords_points
    V2_UTILS_AVAILABLE = True
except Exception as _e:
    print(f"⚠️ [VNCCS V2] Could not import V2 utilities: {_e}")
    V2_UTILS_AVAILABLE = False

_PATCH_SIZE = 14


# ── image preprocessing (V2 logic: scale longest side) ────────────────────────
def _resize_to_tensor(pil_img, target_size):
    """Resize image so its longest side == target_size (multiple of 14). Returns CHW tensor."""
    orig_w, orig_h = pil_img.size
    if orig_w >= orig_h:
        new_w = target_size
        new_h = round(orig_h * (new_w / orig_w) / _PATCH_SIZE) * _PATCH_SIZE
    else:
        new_h = target_size
        new_w = round(orig_w * (new_h / orig_h) / _PATCH_SIZE) * _PATCH_SIZE
    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BICUBIC)
    return transforms.ToTensor()(pil_img), orig_w, orig_h, new_w, new_h


def _adaptive_target_size_from_images(images, max_target_size):
    """Match the official adaptive-size idea for in-memory ComfyUI images."""
    if images is None or images.shape[0] == 0:
        return max_target_size
    heights = images.shape[1]
    widths = images.shape[2]
    longest = max(int(widths), int(heights))
    target = min(max_target_size, longest)
    target = max(_PATCH_SIZE, (target // _PATCH_SIZE) * _PATCH_SIZE)
    return target


def _map_to_comfy_image(value, fallback_shape, normalize=True):
    """Convert [1,S,H,W], [1,S,H,W,1], [S,H,W], or [S,H,W,1] maps to [S,H,W,3]."""
    S, H, W = fallback_shape
    if value is None:
        return torch.zeros(S, H, W, 3, dtype=torch.float32)
    if isinstance(value, np.ndarray):
        t = torch.from_numpy(value)
    else:
        t = value.detach().cpu()
    t = t.float()
    if t.dim() == 5:
        t = t[0]
    if t.dim() == 4 and t.shape[-1] == 1:
        t = t[..., 0]
    elif t.dim() == 4 and t.shape[1] == 1:
        t = t[:, 0]
    if t.dim() == 2:
        t = t.unsqueeze(0)
    if t.dim() != 3:
        return torch.zeros(S, H, W, 3, dtype=torch.float32)
    if normalize:
        t_min = t.amin(dim=(1, 2), keepdim=True)
        t_max = t.amax(dim=(1, 2), keepdim=True)
        t = (t - t_min) / (t_max - t_min + 1e-8)
    else:
        t = t.clamp(0, 1)
    return t.unsqueeze(-1).repeat(1, 1, 1, 3).float()


def _resize_depth_prior(depth_prior, target_size, expected_count):
    """Resize optional ComfyUI depth IMAGE to [1,S,H,W] for WorldMirror depth conditioning."""
    if depth_prior is None:
        return None
    if depth_prior.shape[0] not in (1, expected_count):
        raise ValueError(
            f"depth_prior must have 1 frame or match images ({expected_count}); got {depth_prior.shape[0]}"
        )
    if depth_prior.shape[0] == 1 and expected_count > 1:
        depth_prior = depth_prior.repeat(expected_count, 1, 1, 1)

    depth_list = []
    for i in range(expected_count):
        depth_np = depth_prior[i].detach().cpu().numpy()
        if depth_np.ndim == 3:
            depth_np = depth_np[..., 0]
        depth_np = np.nan_to_num(depth_np.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if depth_np.max() <= 0:
            depth_np = np.ones_like(depth_np, dtype=np.float32)
        depth_min = depth_np[depth_np > 0].min() if np.any(depth_np > 0) else depth_np.min()
        depth_np = depth_np - depth_min
        if depth_np.max() > 0:
            depth_np = depth_np / depth_np.max()
        orig_h, orig_w = depth_np.shape[:2]
        if orig_w >= orig_h:
            new_w = target_size
            new_h = round(orig_h * (new_w / orig_w) / _PATCH_SIZE) * _PATCH_SIZE
        else:
            new_h = target_size
            new_w = round(orig_w * (new_h / orig_h) / _PATCH_SIZE) * _PATCH_SIZE
        d = torch.from_numpy(depth_np).float().view(1, 1, orig_h, orig_w)
        d = torch.nn.functional.interpolate(d, size=(new_h, new_w), mode="bilinear", align_corners=False)[0, 0]
        if new_h > target_size:
            crop = (new_h - target_size) // 2
            d = d[crop:crop + target_size, :]
        depth_list.append(d)
    return torch.stack(depth_list).unsqueeze(0)


def _align_tensor_sequence_to_count(tensor, target_count, label):
    if not isinstance(tensor, torch.Tensor) or target_count <= 0:
        return tensor
    seq_dim = None
    seq_count = None
    if tensor.dim() >= 4 and tensor.shape[0] == 1 and tensor.shape[1] != target_count:
        seq_dim = 1
        seq_count = tensor.shape[1]
    elif tensor.dim() >= 1 and tensor.shape[0] != target_count:
        seq_dim = 0
        seq_count = tensor.shape[0]
    if seq_dim is None or seq_count is None or seq_count <= 0 or seq_count == target_count:
        return tensor
    raise RuntimeError(
        f"WorldMirror V2 received {target_count} images but {seq_count} {label}. "
        "Upstream nodes must provide one camera pose/intrinsic per image; refusing to guess camera alignment."
    )


def _align_camera_priors_to_image_count(images, camera_poses, camera_intrinsics):
    if isinstance(images, torch.Tensor):
        target_count = int(images.shape[0])
    else:
        try:
            target_count = len(images)
        except Exception:
            target_count = 0
    return (
        _align_tensor_sequence_to_count(camera_poses, target_count, "camera_poses"),
        _align_tensor_sequence_to_count(camera_intrinsics, target_count, "camera_intrinsics"),
    )


def _normalize_camera_poses_to_first(camera_poses):
    """Match official pipeline prior-camera normalization: inv(first_pose) @ pose."""
    if not isinstance(camera_poses, torch.Tensor):
        return camera_poses

    poses = camera_poses
    had_batch = poses.dim() == 4 and poses.shape[0] == 1
    if had_batch:
        work = poses[0]
    elif poses.dim() == 3:
        work = poses
    else:
        return camera_poses

    if work.shape[-2:] == (4, 4):
        work4 = work
        return_3x4 = False
    elif work.shape[-2:] == (3, 4):
        bottom = torch.zeros(
            work.shape[0], 1, 4,
            dtype=work.dtype,
            device=work.device,
        )
        bottom[:, 0, 3] = 1.0
        work4 = torch.cat([work, bottom], dim=1)
        return_3x4 = True
    else:
        return camera_poses

    if work4.shape[0] == 0:
        return camera_poses

    try:
        first = work4[0]
        inv_first = torch.linalg.inv(first.float()).to(dtype=work4.dtype, device=work4.device)
        normalized = inv_first.unsqueeze(0) @ work4
    except Exception as exc:
        print(f"[V2] Official camera-pose normalization skipped: {type(exc).__name__}: {exc}")
        return camera_poses

    if return_3x4:
        normalized = normalized[:, :3, :]
    if had_batch:
        normalized = normalized.unsqueeze(0)
    return normalized


def _extract_pose_tensor(value):
    if not isinstance(value, torch.Tensor):
        return None
    poses = value.detach().cpu().float()
    if poses.dim() == 4 and poses.shape[0] == 1:
        poses = poses[0]
    if poses.dim() == 3 and poses.shape[-2:] == (4, 4):
        return poses
    return None


def _rescale_input_pose_translation_to_prediction(input_poses, predicted_poses, eps=1e-6):
    poses = _extract_pose_tensor(input_poses)
    pred = _extract_pose_tensor(predicted_poses)
    if poses is None or pred is None or poses.shape[0] != pred.shape[0]:
        return poses if poses is not None else input_poses

    out = poses.clone()
    input_t = poses[:, :3, 3]
    pred_t = pred[:, :3, 3]
    input_norm = input_t.norm(dim=1)

    starts = [0]
    for i in range(1, poses.shape[0]):
        if input_norm[i] <= eps:
            starts.append(i)
    starts = sorted(set(starts))

    changed = 0
    ratios = []
    for start_idx, end_idx in zip(starts, starts[1:] + [poses.shape[0]]):
        if end_idx - start_idx <= 1:
            continue
        base_input = input_t[start_idx]
        base_pred = pred_t[start_idx]
        rel_input = input_t[start_idx:end_idx] - base_input
        rel_pred = pred_t[start_idx:end_idx] - base_pred
        rel_input_norm = rel_input.norm(dim=1)
        rel_pred_norm = rel_pred.norm(dim=1)
        valid = rel_input_norm > eps
        if not bool(valid.any()):
            continue
        scale = torch.ones_like(rel_input_norm)
        scale[valid] = rel_pred_norm[valid] / rel_input_norm[valid].clamp_min(eps)
        out[start_idx:end_idx, :3, 3] = base_input + rel_input * scale[:, None]
        changed += int(valid.sum().item())
        ratios.extend(scale[valid].tolist())

    if changed > 0 and ratios:
        ratios_t = torch.tensor(ratios, dtype=torch.float32)
        print(
            "[V2 EXP] Calibrated input camera translation scale from predicted motion: "
            f"frames={changed}, median={ratios_t.median().item():.4f}, "
            f"min={ratios_t.min().item():.4f}, max={ratios_t.max().item():.4f}"
        )
    return out


def _suppress_generated_surfaces_seen_by_anchors(
    pts,
    depth_hi,
    poses,
    intrs,
    anchor_frames,
    abs_tol=0.035,
    rel_tol=0.035,
):
    if pts.dim() != 4 or depth_hi.dim() != 3 or poses.shape[0] != pts.shape[0]:
        return torch.ones(depth_hi.shape, dtype=torch.bool)

    S, H, W, _ = pts.shape
    if anchor_frames.shape[0] != S or not bool(anchor_frames.any()) or not bool((~anchor_frames).any()):
        return torch.ones((S, H, W), dtype=torch.bool)

    keep = torch.ones((S, H, W), dtype=torch.bool)
    anchor_ids = torch.where(anchor_frames)[0]
    generated_ids = torch.where(~anchor_frames)[0]
    total_generated = int(generated_ids.numel() * H * W)
    suppressed = 0

    for gen_id_t in generated_ids:
        gen_id = int(gen_id_t.item())
        points = pts[gen_id].reshape(-1, 3).float()
        covered = torch.zeros(points.shape[0], dtype=torch.bool)

        for anchor_id_t in anchor_ids:
            anchor_id = int(anchor_id_t.item())
            R = poses[anchor_id, :3, :3].float()
            t = poses[anchor_id, :3, 3].float()
            cam = (points - t) @ R
            z = cam[:, 2]
            valid = z > 1e-6
            if not bool(valid.any()):
                continue

            K = intrs[anchor_id].float()
            u = torch.round(cam[:, 0] * K[0, 0] / z.clamp_min(1e-6) + K[0, 2]).long()
            v = torch.round(cam[:, 1] * K[1, 1] / z.clamp_min(1e-6) + K[1, 2]).long()
            valid = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if not bool(valid.any()):
                continue

            valid_ids = torch.where(valid)[0]
            anchor_depth = depth_hi[anchor_id, v[valid_ids], u[valid_ids]].float()
            tol = float(abs_tol) + float(rel_tol) * anchor_depth.abs()
            same_or_in_front = z[valid_ids] <= anchor_depth + tol
            covered[valid_ids[same_or_in_front]] = True

        frame_keep = ~covered.reshape(H, W)
        keep[gen_id] = frame_keep
        suppressed += int(covered.sum().item())

    if suppressed > 0:
        print(
            "[V2 EXP] Suppressed low-res stereo splats already covered by high-res anchors: "
            f"{suppressed}/{total_generated} ({100.0 * suppressed / max(total_generated, 1):.2f}%)"
        )
    return keep


def _apply_mask_to_splats(splats, mask_np):
    if splats is None or mask_np is None:
        return splats
    flat_mask = torch.from_numpy(mask_np.reshape(-1)).bool()

    def _filter_splat_tensor(t):
        mask = flat_mask.to(t.device)
        if t.dim() >= 2 and t.shape[1] == mask.shape[0]:
            return t[:, mask, ...]
        if t.dim() >= 1 and t.shape[0] == mask.shape[0]:
            return t[mask, ...]
        return t

    filtered = {}
    for k, v in splats.items():
        if isinstance(v, list):
            filtered[k] = [
                _filter_splat_tensor(t) if isinstance(t, torch.Tensor) else t
                for t in v
            ]
        elif isinstance(v, torch.Tensor):
            filtered[k] = _filter_splat_tensor(v)
        else:
            filtered[k] = v
    return filtered


def _voxel_prune_splats_dict(splats, voxel_size):
    if splats is None or voxel_size <= 0 or not V2_UTILS_AVAILABLE:
        return splats
    required = ("means", "scales", "quats", "opacities")
    if not isinstance(splats, dict) or any(k not in splats for k in required):
        return splats
    out = dict(splats)
    batches = len(splats["means"]) if isinstance(splats["means"], list) else splats["means"].shape[0]
    pruned = {k: [] for k in ("means", "scales", "quats", "opacities")}
    color_key = "sh" if "sh" in splats else "colors" if "colors" in splats else None
    if color_key is None:
        return splats
    pruned[color_key] = []

    for i in range(batches):
        def get_item(key):
            v = splats[key]
            return v[i] if isinstance(v, list) else v[i]

        means = get_item("means").detach().cpu()
        scales = get_item("scales").detach().cpu()
        quats = get_item("quats").detach().cpu()
        opacities = get_item("opacities").detach().cpu().reshape(-1)
        opacity_max = opacities.max().detach().clone() if opacities.numel() else torch.tensor(1.0)
        colors = get_item(color_key).detach().cpu()
        if colors.dim() == 3 and colors.shape[1] == 1:
            colors_for_prune = colors[:, 0, :]
        else:
            colors_for_prune = colors.reshape(colors.shape[0], -1)[:, :3]
        weights = get_item("weights").detach().cpu().reshape(-1) if "weights" in splats else torch.ones_like(opacities)
        if weights.shape[0] != opacities.shape[0]:
            print(
                f"⚠️ [V2] Ignoring stale splat weights during voxel prune: "
                f"weights={weights.shape[0]}, opacities={opacities.shape[0]}"
            )
            weights = torch.ones_like(opacities)
        means, scales, quats, colors_new, opacities = _voxel_prune_gaussians(
            means, scales, quats, colors_for_prune, opacities, weights, voxel_size=voxel_size
        )
        opacities = opacities.clamp_max(opacity_max)
        pruned["means"].append(means)
        pruned["scales"].append(scales)
        pruned["quats"].append(quats)
        pruned["opacities"].append(opacities)
        if color_key == "sh":
            pruned[color_key].append(colors_new[:, None, :])
        else:
            pruned[color_key].append(colors_new)
    out.update(pruned)
    # Weights are only used as a merge/prune helper. After voxel merging they no
    # longer map 1:1 to the pruned gaussians, so keeping them causes later prune
    # passes to mix tensors with incompatible lengths.
    out.pop("weights", None)
    return out


def _replace_splat_colors_from_images(splats, imgs_tensor):
    if not isinstance(splats, dict) or imgs_tensor is None:
        return splats
    means = splats.get("means")
    if means is None:
        return splats

    B, S, _, H, W = imgs_tensor.shape
    rgb = imgs_tensor.permute(0, 1, 3, 4, 2).reshape(B, S * H * W, 3)
    sh_dc = (rgb - 0.5) / 0.28209479177387814

    out = dict(splats)
    if isinstance(means, list):
        out["sh"] = [sh_dc[i, :m.shape[0]].to(m.device, dtype=m.dtype)[:, None, :] for i, m in enumerate(means)]
    elif isinstance(means, torch.Tensor):
        out["sh"] = sh_dc[:, :means.shape[1]].to(means.device, dtype=means.dtype)[:, :, None, :]
    else:
        return splats
    return out


def _tune_splats(splats, scale_multiplier=1.0, opacity_floor=0.0):
    if not isinstance(splats, dict):
        return splats
    out = dict(splats)

    def map_value(value, fn):
        if isinstance(value, list):
            return [fn(v) if isinstance(v, torch.Tensor) else v for v in value]
        if isinstance(value, torch.Tensor):
            return fn(value)
        return value

    if scale_multiplier != 1.0 and "scales" in out:
        out["scales"] = map_value(out["scales"], lambda t: t * float(scale_multiplier))
    if opacity_floor > 0.0 and "opacities" in out:
        floor = float(opacity_floor)
        out["opacities"] = map_value(out["opacities"], lambda t: t.clamp_min(floor))
    return out


def _rotation_matrix_to_quat_wxyz(R, dtype=None, device=None):
    R = R.detach().to(device=device, dtype=torch.float32)
    trace = R.trace()
    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = torch.stack([w, x, y, z])
    q = q / q.norm().clamp_min(1e-8)
    return q.to(dtype=dtype or R.dtype, device=device or R.device)


def _quat_mul_wxyz(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def _estimate_similarity_transform(src_points, dst_points, eps=1e-8):
    """Estimate dst ~= scale * R * src + t for matched 3D point sets."""
    if src_points.shape != dst_points.shape or src_points.shape[0] < 3:
        return None
    X = src_points.detach().cpu().float()
    Y = dst_points.detach().cpu().float()
    finite = torch.isfinite(X).all(dim=1) & torch.isfinite(Y).all(dim=1)
    X, Y = X[finite], Y[finite]
    if X.shape[0] < 3:
        return None

    mu_x = X.mean(dim=0)
    mu_y = Y.mean(dim=0)
    Xc = X - mu_x
    Yc = Y - mu_y
    var_x = (Xc.square().sum(dim=1)).mean()
    if var_x <= eps:
        return None

    cov = (Yc.T @ Xc) / X.shape[0]
    try:
        U, D, Vh = torch.linalg.svd(cov)
    except RuntimeError:
        return None
    S = torch.eye(3, dtype=torch.float32)
    if torch.linalg.det(U @ Vh) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vh
    scale = torch.trace(torch.diag(D) @ S) / var_x
    if not torch.isfinite(scale) or scale <= eps:
        return None
    t = mu_y - scale * (R @ mu_x)
    return {"scale": scale.float(), "rotation": R.float(), "translation": t.float()}


def _compose_similarity(a, b):
    """Compose transforms a(b(x)); each transform is scale, rotation, translation."""
    s = a["scale"] * b["scale"]
    R = a["rotation"] @ b["rotation"]
    t = a["scale"] * (a["rotation"] @ b["translation"]) + a["translation"]
    return {"scale": s.float(), "rotation": R.float(), "translation": t.float()}


def _transform_points_similarity(points, transform):
    R = transform["rotation"].to(points.device, dtype=points.dtype)
    t = transform["translation"].to(points.device, dtype=points.dtype)
    s = transform["scale"].to(points.device, dtype=points.dtype)
    return (points @ R.T) * s + t


def _camera_pose_icp_cloud(poses, intrinsics=None, image_hw=None):
    """Build an ICP point set from camera positions and pose-derived ray samples."""
    p = _extract_pose_tensor(poses)
    if p is None or p.shape[0] < 3:
        return None

    intr = None
    if isinstance(intrinsics, torch.Tensor):
        intr = intrinsics.detach().cpu().float()
        if intr.dim() == 4 and intr.shape[0] == 1:
            intr = intr[0]
        if intr.dim() != 3 or intr.shape[0] != p.shape[0] or intr.shape[-2:] != (3, 3):
            intr = None

    positions = p[:, :3, 3]
    finite = torch.isfinite(positions).all(dim=1)
    positions = positions[finite]
    rotations = p[finite, :3, :3]
    if positions.shape[0] < 3:
        return None

    span = positions.max(dim=0).values - positions.min(dim=0).values
    ray_len = max(float(torch.linalg.norm(span).item()) * 0.05, 1e-3)
    points = [positions]
    for axis in range(3):
        points.append(positions + rotations[:, :, axis] * ray_len)

    if intr is not None:
        intr = intr[finite]
        if image_hw is None:
            h = w = 1.0
        else:
            h, w = float(image_hw[0]), float(image_hw[1])
        uv = torch.tensor(
            [[0.0, 0.0], [w - 1.0, 0.0], [0.0, h - 1.0], [w - 1.0, h - 1.0], [w * 0.5, h * 0.5]],
            dtype=torch.float32,
        )
        rays = []
        for i in range(intr.shape[0]):
            fx = intr[i, 0, 0].clamp_min(1e-6)
            fy = intr[i, 1, 1].clamp_min(1e-6)
            cx = intr[i, 0, 2]
            cy = intr[i, 1, 2]
            cam = torch.stack([
                (uv[:, 0] - cx) / fx,
                (uv[:, 1] - cy) / fy,
                torch.ones(uv.shape[0], dtype=torch.float32),
            ], dim=1)
            cam = cam / cam.norm(dim=1, keepdim=True).clamp_min(1e-8)
            rays.append(positions[i][None, :] + (cam @ rotations[i].T) * ray_len)
        points.append(torch.cat(rays, dim=0))

    cloud = torch.cat(points, dim=0)
    if cloud.shape[0] < 3 or (cloud.max(dim=0).values - cloud.min(dim=0).values).norm() <= 1e-8:
        return None
    return cloud


def _camera_positions_degenerate(poses, eps=1e-6):
    p = _extract_pose_tensor(poses)
    if p is None or p.shape[0] < 2:
        return True
    positions = p[:, :3, 3]
    finite = torch.isfinite(positions).all(dim=1)
    positions = positions[finite]
    if positions.shape[0] < 2:
        return True
    span = positions.max(dim=0).values - positions.min(dim=0).values
    return float(torch.linalg.norm(span).item()) <= float(eps)


def _estimate_orientation_alignment(source_poses, target_poses):
    src = _extract_pose_tensor(source_poses)
    dst = _extract_pose_tensor(target_poses)
    if src is None or dst is None or src.shape[0] != dst.shape[0] or src.shape[0] < 1:
        return None
    src_r = src[:, :3, :3].detach().cpu().float()
    dst_r = dst[:, :3, :3].detach().cpu().float()
    src_vecs = src_r.permute(0, 2, 1).reshape(-1, 3)
    dst_vecs = dst_r.permute(0, 2, 1).reshape(-1, 3)
    finite = torch.isfinite(src_vecs).all(dim=1) & torch.isfinite(dst_vecs).all(dim=1)
    src_vecs = src_vecs[finite]
    dst_vecs = dst_vecs[finite]
    if src_vecs.shape[0] < 3:
        return None
    try:
        U, _, Vh = torch.linalg.svd(dst_vecs.T @ src_vecs)
    except RuntimeError:
        return None
    S = torch.eye(3, dtype=torch.float32)
    if torch.linalg.det(U @ Vh) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vh
    aligned = (src_vecs @ R.T)
    angular = torch.acos((aligned * dst_vecs).sum(dim=1).clamp(-1.0, 1.0)) * (180.0 / torch.pi)
    return R.float(), float(angular.mean().item()), float(angular.max().item())


def _icp_similarity(source_points, target_points, max_iterations=40, tolerance=1e-7):
    if source_points is None or target_points is None or source_points.shape[0] < 3 or target_points.shape[0] < 3:
        return None

    source = source_points.detach().cpu().float()
    target = target_points.detach().cpu().float()
    transform = {
        "scale": torch.tensor(1.0, dtype=torch.float32),
        "rotation": torch.eye(3, dtype=torch.float32),
        "translation": torch.zeros(3, dtype=torch.float32),
    }
    initial = None
    if source.shape[0] == target.shape[0]:
        initial = _estimate_similarity_transform(source, target)
    if initial is not None:
        transform = initial
        transformed = _transform_points_similarity(source, transform)
    else:
        transformed = source
    prev_rmse = None
    best_transform = transform
    best_rmse = None

    for _ in range(int(max_iterations)):
        dists = torch.cdist(transformed, target)
        nn_dist, nn_idx = torch.min(dists, dim=1)
        matched = target[nn_idx]
        if nn_dist.numel() >= 8:
            cutoff = torch.quantile(nn_dist, 0.90)
            keep = nn_dist <= cutoff
            if int(keep.sum().item()) >= 3:
                src_fit = transformed[keep]
                dst_fit = matched[keep]
                rmse = torch.sqrt((nn_dist[keep].square()).mean())
            else:
                src_fit, dst_fit = transformed, matched
                rmse = torch.sqrt((nn_dist.square()).mean())
        else:
            src_fit, dst_fit = transformed, matched
            rmse = torch.sqrt((nn_dist.square()).mean())

        delta = _estimate_similarity_transform(src_fit, dst_fit)
        if delta is None:
            return None
        final_rmse = float(rmse.item())
        if best_rmse is None or final_rmse < best_rmse:
            best_rmse = final_rmse
            best_transform = transform
        elif final_rmse > best_rmse + tolerance:
            break
        next_transformed = _transform_points_similarity(transformed, delta)
        next_transform = _compose_similarity(delta, transform)
        next_nn = torch.min(torch.cdist(next_transformed, target), dim=1).values
        next_rmse = float(torch.sqrt(next_nn.square().mean()).item())
        if next_rmse > final_rmse + tolerance:
            break
        transformed = next_transformed
        transform = next_transform
        if prev_rmse is not None and abs(prev_rmse - final_rmse) <= tolerance:
            break
        prev_rmse = final_rmse

    best_transform["rmse"] = float(best_rmse if best_rmse is not None else 0.0)
    return best_transform


def _transform_pose_tensor_similarity(poses, transform):
    if not isinstance(poses, torch.Tensor):
        return poses
    out = poses.clone()
    view = out.reshape(-1, *out.shape[-2:]) if out.dim() == 4 else out
    if view.dim() != 3 or view.shape[-2:] != (4, 4):
        return poses
    R = transform["rotation"].to(out.device, dtype=out.dtype)
    t = transform["translation"].to(out.device, dtype=out.dtype)
    s = transform["scale"].to(out.device, dtype=out.dtype)
    view[:, :3, :3] = R @ view[:, :3, :3]
    view[:, :3, 3] = (view[:, :3, 3] @ R.T) * s + t
    return out


def _transform_splats_similarity(splats, transform):
    if not isinstance(splats, dict):
        return splats
    out = dict(splats)
    scale_value = float(transform["scale"].abs().item())

    def map_value(value, fn):
        if isinstance(value, list):
            return [fn(v) if isinstance(v, torch.Tensor) else v for v in value]
        if isinstance(value, torch.Tensor):
            return fn(value)
        return value

    if "means" in out:
        out["means"] = map_value(out["means"], lambda x: _transform_points_similarity(x, transform))
    if "scales" in out:
        out["scales"] = map_value(out["scales"], lambda x: x * scale_value)
    if "quats" in out:
        def rotate_quats(q):
            q_rot = _rotation_matrix_to_quat_wxyz(
                transform["rotation"], dtype=q.dtype, device=q.device
            ).view(*([1] * (q.dim() - 1)), 4)
            q_out = _quat_mul_wxyz(q_rot.expand_as(q), q)
            return q_out / q_out.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        out["quats"] = map_value(out["quats"], rotate_quats)
    return out


def _align_predictions_to_input_cameras_icp(predictions, input_camera_poses, input_camera_intrinsics, image_hw):
    pred_cloud = _camera_pose_icp_cloud(
        predictions.get("camera_poses"),
        predictions.get("camera_intrs"),
        image_hw=image_hw,
    )
    input_cloud = _camera_pose_icp_cloud(
        input_camera_poses,
        input_camera_intrinsics,
        image_hw=image_hw,
    )
    transform = _icp_similarity(pred_cloud, input_cloud)
    if transform is None:
        print("[V2] Input-camera ICP alignment skipped: insufficient or degenerate camera pose point sets.")
        return predictions, None

    aligned = dict(predictions)
    if "camera_poses" in aligned:
        aligned["camera_poses"] = _transform_pose_tensor_similarity(aligned["camera_poses"], transform)
    if isinstance(aligned.get("pts3d"), torch.Tensor):
        aligned["pts3d"] = _transform_points_similarity(aligned["pts3d"], transform)
    if isinstance(aligned.get("splats"), dict):
        aligned["splats"] = _transform_splats_similarity(aligned["splats"], transform)

    print(
        "[V2] ICP-aligned predicted reconstruction to input cameras: "
        f"scale={float(transform['scale'].item()):.6f}, rmse={transform['rmse']:.6f}, "
        f"source_points={pred_cloud.shape[0]}, target_points={input_cloud.shape[0]}"
    )
    return aligned, transform


def _align_predictions_to_input_cameras_global(predictions, input_camera_poses, input_camera_intrinsics, image_hw):
    pred_poses = predictions.get("camera_poses")
    pred_pose_tensor = _extract_pose_tensor(pred_poses)
    input_pose_tensor = _extract_pose_tensor(input_camera_poses)
    if pred_pose_tensor is None or input_pose_tensor is None or pred_pose_tensor.shape[0] != input_pose_tensor.shape[0]:
        print("[V2] Input-align skipped: predicted/input camera poses are missing or mismatched.")
        return predictions, None

    transform = None
    if not _camera_positions_degenerate(input_camera_poses):
        pred_cloud = _camera_pose_icp_cloud(
            pred_poses,
            predictions.get("camera_intrs"),
            image_hw=image_hw,
        )
        input_cloud = _camera_pose_icp_cloud(
            input_camera_poses,
            input_camera_intrinsics,
            image_hw=image_hw,
        )
        transform = _icp_similarity(pred_cloud, input_cloud)
        if transform is not None:
            print(
                "[V2] Input-align: Sim3 aligned complete predicted reconstruction to input cameras: "
                f"scale={float(transform['scale'].item()):.6f}, rmse={transform['rmse']:.6f}, "
                f"source_points={pred_cloud.shape[0]}, target_points={input_cloud.shape[0]}"
            )

    if transform is None:
        orient = _estimate_orientation_alignment(pred_pose_tensor, input_pose_tensor)
        if orient is None:
            print("[V2] Input-align skipped: could not estimate orientation alignment.")
            return predictions, None
        R, mean_angle, max_angle = orient
        pred_center = pred_pose_tensor[:, :3, 3].median(dim=0).values
        input_center = input_pose_tensor[:, :3, 3].median(dim=0).values
        transform = {
            "scale": torch.tensor(1.0, dtype=torch.float32),
            "rotation": R,
            "translation": input_center - (R @ pred_center),
            "rmse": None,
            "orientation_mean_deg": mean_angle,
            "orientation_max_deg": max_angle,
            "orientation_only": True,
        }
        print(
            "[V2] Input-align: SO3 aligned complete predicted reconstruction to co-located input cameras: "
            f"mean_angle={mean_angle:.4f} deg, max_angle={max_angle:.4f} deg"
        )

    aligned = dict(predictions)
    if "camera_poses" in aligned:
        aligned["camera_poses"] = _transform_pose_tensor_similarity(aligned["camera_poses"], transform)
    if isinstance(aligned.get("pts3d"), torch.Tensor):
        aligned["pts3d"] = _transform_points_similarity(aligned["pts3d"], transform)
    if isinstance(aligned.get("splats"), dict):
        aligned["splats"] = _transform_splats_similarity(aligned["splats"], transform)
    aligned["_splat_camera_source"] = "input_align"
    aligned["_input_align_transform"] = transform
    return aligned, transform


def _depth_maps_s_h_w(depth):
    if not isinstance(depth, torch.Tensor):
        return None
    if depth.dim() == 5 and depth.shape[0] == 1 and depth.shape[-1] == 1:
        return depth[0, ..., 0]
    if depth.dim() == 4 and depth.shape[-1] == 1:
        return depth[..., 0]
    if depth.dim() == 4 and depth.shape[0] == 1:
        return depth[0]
    if depth.dim() == 3:
        return depth
    return None


def _backproject_depth_from_input_cameras(depth_maps, input_camera_poses, input_camera_intrinsics):
    poses = _extract_pose_tensor(input_camera_poses)
    if poses is None or not isinstance(input_camera_intrinsics, torch.Tensor):
        return None
    intrs = input_camera_intrinsics.detach().float()
    if intrs.dim() == 4 and intrs.shape[0] == 1:
        intrs = intrs[0]
    if intrs.dim() != 3 or intrs.shape[-2:] != (3, 3):
        return None
    if poses.shape[0] != depth_maps.shape[0] or intrs.shape[0] != depth_maps.shape[0]:
        return None

    device = depth_maps.device
    depth_f = depth_maps.float()
    poses = poses.to(device=device, dtype=torch.float32)
    intrs = intrs.to(device=device, dtype=torch.float32)
    pts, _, _ = depth_to_world_coords_points(depth_f, poses, intrs)
    return pts


def _map_splat_value(value, fn):
    if isinstance(value, list):
        return [fn(v) if isinstance(v, torch.Tensor) else v for v in value]
    if isinstance(value, torch.Tensor):
        return fn(value)
    return value


def _replace_splat_means_from_points(splats, points):
    if not isinstance(splats, dict) or not isinstance(points, torch.Tensor):
        return splats
    out = dict(splats)
    flat = points.reshape(-1, 3)
    means = out.get("means")
    if isinstance(means, list):
        out["means"] = [
            flat.to(device=m.device, dtype=m.dtype) if isinstance(m, torch.Tensor) else m
            for m in means
        ]
    elif isinstance(means, torch.Tensor):
        repl = flat.to(device=means.device, dtype=means.dtype)
        out["means"] = repl.unsqueeze(0) if means.dim() >= 3 and means.shape[0] == 1 else repl
    else:
        out["means"] = flat.unsqueeze(0)
    return out


def _splat_means_count(splats):
    if not isinstance(splats, dict):
        return None
    means = splats.get("means")
    if isinstance(means, list):
        means = means[0] if means and isinstance(means[0], torch.Tensor) else None
    if not isinstance(means, torch.Tensor):
        return None
    if means.dim() >= 3 and means.shape[0] == 1:
        return int(means.shape[1])
    if means.dim() >= 2:
        return int(means.shape[0])
    return None


def _scale_splat_scales(splats, scale):
    if not isinstance(splats, dict) or "scales" not in splats:
        return splats
    out = dict(splats)
    scale_value = float(abs(scale))
    out["scales"] = _map_splat_value(out["scales"], lambda x: x * scale_value)
    return out


def _stabilize_camera_intrinsics(intrinsics, mode, image_hw=None):
    if not isinstance(intrinsics, torch.Tensor):
        return intrinsics
    intrs = intrinsics.detach().clone().float()
    had_batch = intrs.dim() == 4 and intrs.shape[0] == 1
    if had_batch:
        work = intrs[0]
    elif intrs.dim() == 3:
        work = intrs
    else:
        return intrinsics
    if work.shape[-2:] != (3, 3) or work.shape[0] == 0:
        return intrinsics

    stable = work.clone()
    finite = torch.isfinite(stable).all(dim=(1, 2))
    if not bool(finite.any()):
        return intrinsics
    valid = stable[finite]
    med = valid.median(dim=0).values

    stable[:, 0, 0] = med[0, 0]
    stable[:, 1, 1] = med[1, 1]
    if mode in ("median_focal_center", "median_all"):
        if image_hw is not None:
            h, w = image_hw
            stable[:, 0, 2] = float(w) * 0.5
            stable[:, 1, 2] = float(h) * 0.5
        else:
            stable[:, 0, 2] = med[0, 2]
            stable[:, 1, 2] = med[1, 2]
    elif mode == "median_all":
        stable[:, 0, 2] = med[0, 2]
        stable[:, 1, 2] = med[1, 2]

    return stable.unsqueeze(0) if had_batch else stable


def _auto_stabilize_missing_input_cameras(predictions, mode, image_hw):
    if mode == "off":
        return None, None, None
    pred_poses = predictions.get("camera_poses")
    pred_intrs = predictions.get("camera_intrs")
    poses = _extract_pose_tensor(pred_poses)
    if poses is None or not isinstance(pred_intrs, torch.Tensor):
        print("[V2] Missing-camera stabilization skipped: predicted cameras are unavailable.")
        return None, None, None

    intrs = pred_intrs.detach().float()
    if intrs.dim() == 4 and intrs.shape[0] == 1:
        intrs = intrs[0]
    if intrs.dim() != 3 or intrs.shape[-2:] != (3, 3) or intrs.shape[0] != poses.shape[0]:
        print("[V2] Missing-camera stabilization skipped: predicted intrinsics are invalid.")
        return None, None, None

    stable_intrs = _stabilize_camera_intrinsics(intrs, "median_focal_center", image_hw=image_hw)
    stable_poses = poses.detach().clone().float()
    info = {
        "source": "predicted_cameras",
        "mode": mode,
        "fx_min": float(intrs[:, 0, 0].min().item()),
        "fx_max": float(intrs[:, 0, 0].max().item()),
        "fx_stable": float(stable_intrs[:, 0, 0].median().item()),
        "fy_min": float(intrs[:, 1, 1].min().item()),
        "fy_max": float(intrs[:, 1, 1].max().item()),
        "fy_stable": float(stable_intrs[:, 1, 1].median().item()),
    }
    print(
        "[V2] Missing-camera stabilization: "
        f"mode={mode}, fx={info['fx_min']:.3f}..{info['fx_max']:.3f}->{info['fx_stable']:.3f}, "
        f"fy={info['fy_min']:.3f}..{info['fy_max']:.3f}->{info['fy_stable']:.3f}"
    )
    return stable_poses, stable_intrs, info


def _project_predictions_from_input_cameras(predictions, input_camera_poses, input_camera_intrinsics, scale_value):
    projected = dict(predictions)
    depth_device = projected["depth"].device if isinstance(projected.get("depth"), torch.Tensor) else torch.device("cpu")
    depth_points = None

    if isinstance(projected.get("depth"), torch.Tensor):
        projected["depth"] = projected["depth"] * float(scale_value)
        depth_maps = _depth_maps_s_h_w(projected["depth"])
        if depth_maps is not None:
            pts = _backproject_depth_from_input_cameras(depth_maps, input_camera_poses, input_camera_intrinsics)
            if pts is not None:
                depth_points = pts
                projected["pts3d"] = pts.unsqueeze(0).to(
                    device=projected["depth"].device,
                    dtype=projected["depth"].dtype,
                )

    if isinstance(projected.get("gs_depth"), torch.Tensor):
        projected["gs_depth"] = projected["gs_depth"] * float(scale_value)
        gs_depth_maps = _depth_maps_s_h_w(projected["gs_depth"])
        if gs_depth_maps is not None and isinstance(projected.get("splats"), dict):
            gs_pts = _backproject_depth_from_input_cameras(gs_depth_maps, input_camera_poses, input_camera_intrinsics)
            if gs_pts is not None:
                projected["splats"] = _replace_splat_means_from_points(projected["splats"], gs_pts)
                projected["splats"] = _scale_splat_scales(projected["splats"], scale_value)
                print(
                    "[V2] Input-camera projection: replaced splat means from gs_depth "
                    f"({int(gs_pts.reshape(-1, 3).shape[0])} points)."
                )
    elif isinstance(projected.get("splats"), dict) and isinstance(depth_points, torch.Tensor):
        splat_count = _splat_means_count(projected["splats"])
        depth_count = int(depth_points.reshape(-1, 3).shape[0])
        if splat_count == depth_count:
            projected["splats"] = _replace_splat_means_from_points(projected["splats"], depth_points)
            projected["splats"] = _scale_splat_scales(projected["splats"], scale_value)
            print(
                "[V2] Input-camera projection: gs_depth missing; replaced splat means "
                f"from regular depth grid ({depth_count} points)."
            )
        else:
            print(
                "[V2] Input-camera projection: gs_depth missing; kept model splat means "
                f"because splat_count={splat_count} and depth_count={depth_count} do not match."
            )

    if isinstance(input_camera_poses, torch.Tensor):
        projected["camera_poses"] = input_camera_poses.unsqueeze(0).to(
            device=depth_device,
            dtype=torch.float32,
        )
    if isinstance(input_camera_intrinsics, torch.Tensor):
        projected["camera_intrs"] = input_camera_intrinsics.unsqueeze(0).to(
            device=depth_device,
            dtype=torch.float32,
        )
    return projected


def _reproject_predictions_from_input_cameras_with_icp_scale(
    predictions,
    input_camera_poses,
    input_camera_intrinsics,
    image_hw,
):
    if not V2_UTILS_AVAILABLE:
        print("[V2] Input-camera ICP depth-scale projection skipped: V2 geometry utilities unavailable.")
        return predictions, None

    if _camera_positions_degenerate(input_camera_poses):
        print(
            "[V2] Input-camera projection: input camera positions are co-located, "
            "so ICP scale is not observable. Using scale=1.0 and backprojecting depths "
            "through input rotations/intrinsics."
        )
        projected = _project_predictions_from_input_cameras(
            predictions,
            input_camera_poses,
            input_camera_intrinsics,
            scale_value=1.0,
        )
        return projected, {
            "scale": torch.tensor(1.0, dtype=torch.float32),
            "rotation": torch.eye(3, dtype=torch.float32),
            "translation": torch.zeros(3, dtype=torch.float32),
            "rmse": None,
            "degenerate_input_camera_positions": True,
        }

    pred_cloud = _camera_pose_icp_cloud(
        predictions.get("camera_poses"),
        predictions.get("camera_intrs"),
        image_hw=image_hw,
    )
    input_cloud = _camera_pose_icp_cloud(
        input_camera_poses,
        input_camera_intrinsics,
        image_hw=image_hw,
    )
    transform = _icp_similarity(pred_cloud, input_cloud)
    if transform is None:
        print("[V2] Input-camera ICP depth-scale projection skipped: insufficient or degenerate camera pose point sets.")
        return predictions, None

    scale_value = float(abs(transform["scale"].item()))
    projected = _project_predictions_from_input_cameras(
        predictions,
        input_camera_poses,
        input_camera_intrinsics,
        scale_value=scale_value,
    )

    print(
        "[V2] ICP-scaled depths and reprojected from input cameras: "
        f"scale={scale_value:.6f}, rmse={transform['rmse']:.6f}, "
        f"source_points={pred_cloud.shape[0]}, target_points={input_cloud.shape[0]}"
    )
    return projected, transform


def _log_splat_stats(splats):
    if not isinstance(splats, dict):
        print("[V2 DEBUG] splats stats: none")
        return

    def first_tensor(key):
        value = splats.get(key)
        if isinstance(value, list):
            return value[0] if value and isinstance(value[0], torch.Tensor) else None
        return value if isinstance(value, torch.Tensor) else None

    for key in ("scales", "opacities", "weights"):
        value = first_tensor(key)
        if value is not None:
            print("[V2 DEBUG] splats." + _tensor_stats_line(key, value))


def _module_has_meta_tensors(module):
    return any(p.device.type == "meta" for p in module.parameters())


def _first_real_device(module, fallback=torch.device("cpu")):
    for param in module.parameters():
        if param.device.type != "meta":
            return param.device
    return fallback


def _move_worldmirror(module, device):
    if _module_has_meta_tensors(module):
        raise RuntimeError(
            "WorldMirror V2 model contains meta tensors, most likely from a previous "
            "model_cpu_offload run. Reload the Load WorldMirror V2 Model node, or keep "
            "offload_scheme=model_cpu_offload for this model instance."
        )
    module.to(device)


def _register_bf16_leaf_cast_hooks(model):
    def _input_cast_hook(module, args):
        if not args:
            return args
        dtype = next((p.dtype for p in module.parameters(recurse=False)), None)
        if dtype is None:
            return args
        return tuple(
            a.to(dtype) if isinstance(a, torch.Tensor) and a.is_floating_point() and a.dtype != dtype else a
            for a in args
        )

    for _, module in model.named_modules():
        if not any(True for _ in module.children()):
            own = list(module.parameters(recurse=False))
            if own and all(p.dtype == torch.bfloat16 for p in own):
                module.register_forward_pre_hook(_input_cast_hook)


def _prepare_worldmirror_bf16(model):
    from hyworld2.worldrecon.pipeline import _collect_fp32_critical_modules

    crit = _collect_fp32_critical_modules(model)
    model.to(torch.bfloat16)
    for mod in crit:
        mod.to(torch.float32)
    _register_bf16_leaf_cast_hooks(model)
    model.enable_bf16 = True
    model.to = model._bf16_to
    return crit


def _get_torchao_fp8_weight_only_config():
    try:
        from torchao.quantization import quantize_, float8_weight_only
        return quantize_, float8_weight_only()
    except ImportError as first_error:
        try:
            from torchao.quantization import quantize_, Float8WeightOnlyConfig
            return quantize_, Float8WeightOnlyConfig()
        except ImportError:
            raise ImportError(
                "torchao is required for fp8. Install with: pip install torchao"
            ) from first_error


def _quantize_bf16_linear_weights_fp8(model):
    import torch.nn as nn

    quantize_, config = _get_torchao_fp8_weight_only_config()
    target_linear_ids = {
        id(module)
        for module in model.modules()
        if isinstance(module, nn.Linear)
        and (params := list(module.parameters(recurse=False)))
        and all(p.dtype == torch.bfloat16 for p in params)
    }

    def _filter_bf16_linear(module, _name):
        return id(module) in target_linear_ids

    quantize_(model, config, filter_fn=_filter_bf16_linear)
    print(f"✅ [V2] fp8 weight quantization applied to {len(target_linear_ids)} bf16 Linear layers")


def _set_worldmirror_head_frame_chunk_size(model, chunk_size):
    chunk_size = max(1, int(chunk_size))
    for name in ("depth_head", "pts_head", "norm_head", "gs_head"):
        head = getattr(model, name, None)
        if head is not None:
            setattr(head, "frames_chunk_size", chunk_size)


def _set_worldmirror_gs_param_chunk_size(model, chunk_size):
    chunk_size = max(1, int(chunk_size))
    gs_renderer = getattr(model, "gs_renderer", None)
    if gs_renderer is not None:
        setattr(gs_renderer, "gs_param_chunk_size", chunk_size)


def _set_worldmirror_transformer_mlp_chunk_size(model, chunk_size):
    chunk_size = max(0, int(chunk_size))
    updated = 0
    for module in model.modules():
        if hasattr(module, "inference_chunk_size") and hasattr(module, "fc1") and hasattr(module, "fc2"):
            setattr(module, "inference_chunk_size", chunk_size)
            updated += 1
    if chunk_size > 0:
        print(f"ℹ️ [V2] Transformer MLP chunking enabled: {updated} MLPs, chunk_size={chunk_size}")


def _has_torchao_tensor_subclasses(model):
    for tensor in list(model.parameters()) + list(model.buffers()):
        tensor_type = type(tensor)
        data_type = type(getattr(tensor, "data", tensor))
        if tensor_type.__module__.startswith("torchao.") or data_type.__module__.startswith("torchao."):
            return True
    return False


def _apply_worldmirror_head_compute_mode(model, mode, use_gsplat):
    original = {
        "enable_pts": getattr(model, "enable_pts", None),
        "enable_norm": getattr(model, "enable_norm", None),
        "enable_gs": getattr(model, "enable_gs", None),
    }
    if mode == "depth+gs":
        model.enable_pts = False
        model.enable_norm = False
        model.enable_gs = bool(use_gsplat and original["enable_gs"])
    elif mode == "depth_only":
        model.enable_pts = False
        model.enable_norm = False
        model.enable_gs = False
    else:
        model.enable_gs = bool(use_gsplat and original["enable_gs"])
    return original


def _restore_worldmirror_head_compute_mode(model, original):
    for key, value in original.items():
        if value is not None:
            setattr(model, key, value)


def _camera_debug_angles(c2w):
    """Return approximate yaw/pitch/roll in degrees from a c2w matrix."""
    R = c2w[:3, :3].float()
    forward = R @ torch.tensor([0.0, 0.0, 1.0], device=R.device)
    up = R @ torch.tensor([0.0, -1.0, 0.0], device=R.device)
    yaw = torch.atan2(forward[0], forward[2]) * 180.0 / torch.pi
    pitch = torch.asin(torch.clamp(-forward[1] / forward.norm().clamp_min(1e-8), -1.0, 1.0)) * 180.0 / torch.pi
    roll = torch.atan2(up[0], -up[1]) * 180.0 / torch.pi
    return yaw.item(), pitch.item(), roll.item()


def _angle_delta_deg(a, b):
    return ((a - b + 180.0) % 360.0) - 180.0


def _tensor_stats_line(name, value):
    if value is None:
        return f"{name}=none"
    t = value.detach().cpu().float()
    if t.numel() == 0:
        return f"{name}=empty"
    flat = t.flatten()
    sample_note = ""
    max_stats_values = 1_000_000
    if flat.numel() > max_stats_values:
        step = max(1, flat.numel() // max_stats_values)
        flat = flat[::step][:max_stats_values]
        sample_note = f", sampled={flat.numel()}/{t.numel()}"
    return (
        f"{name}: min={flat.min().item():.4f}, p10={torch.quantile(flat, 0.10).item():.4f}, "
        f"median={flat.median().item():.4f}, p90={torch.quantile(flat, 0.90).item():.4f}, "
        f"max={flat.max().item():.4f}{sample_note}"
    )


def _log_camera_table(title, poses, intrs, H, W):
    if poses is None and intrs is None:
        print(f"[V2 DEBUG] {title}: none")
        return
    poses_cpu = poses.detach().cpu().float() if poses is not None else None
    intrs_cpu = intrs.detach().cpu().float() if intrs is not None else None
    if poses_cpu is not None and poses_cpu.dim() == 4:
        poses_cpu = poses_cpu[0]
    if intrs_cpu is not None and intrs_cpu.dim() == 4:
        intrs_cpu = intrs_cpu[0]
    count = poses_cpu.shape[0] if poses_cpu is not None else intrs_cpu.shape[0]
    print(f"[V2 DEBUG] {title}: {count} cameras")
    for i in range(count):
        if poses_cpu is not None:
            yaw, pitch, roll = _camera_debug_angles(poses_cpu[i])
            t = poses_cpu[i, :3, 3]
            pose_part = (
                f"yaw={yaw:8.3f} pitch={pitch:8.3f} roll={roll:8.3f} "
                f"t=({t[0].item(): .4f},{t[1].item(): .4f},{t[2].item(): .4f})"
            )
        else:
            pose_part = "pose=none"
        if intrs_cpu is not None:
            fx = intrs_cpu[i, 0, 0].item()
            fy = intrs_cpu[i, 1, 1].item()
            cx = intrs_cpu[i, 0, 2].item()
            cy = intrs_cpu[i, 1, 2].item()
            fov_x = 2.0 * np.degrees(np.arctan(W * 0.5 / max(fx, 1e-8)))
            fov_y = 2.0 * np.degrees(np.arctan(H * 0.5 / max(fy, 1e-8)))
            intr_part = (
                f"fx={fx:8.3f} fy={fy:8.3f} cx={cx:7.2f} cy={cy:7.2f} "
                f"fov=({fov_x:6.2f},{fov_y:6.2f})"
            )
        else:
            intr_part = "intr=none"
        print(f"[V2 DEBUG]   {i:02d}: {pose_part} | {intr_part}")


def _log_per_view_points(name, points, S, H, W):
    if points is None:
        print(f"[V2 DEBUG] {name}: none")
        return
    pts = points.detach().cpu().float()
    if pts.dim() == 5:
        pts = pts[0].reshape(S, H * W, 3)
    elif pts.dim() == 3 and pts.shape[1] == S * H * W:
        pts = pts[0].reshape(S, H * W, 3)
    elif pts.dim() == 2 and pts.shape[0] == S * H * W:
        pts = pts.reshape(S, H * W, 3)
    else:
        print(f"[V2 DEBUG] {name}: unsupported shape={tuple(pts.shape)}")
        return
    print(f"[V2 DEBUG] {name}: per-view world bounds")
    for i in range(S):
        p = pts[i]
        center = p.mean(dim=0)
        pmin = p.amin(dim=0)
        pmax = p.amax(dim=0)
        print(
            f"[V2 DEBUG]   {i:02d}: center=({center[0]: .4f},{center[1]: .4f},{center[2]: .4f}) "
            f"min=({pmin[0]: .4f},{pmin[1]: .4f},{pmin[2]: .4f}) "
            f"max=({pmax[0]: .4f},{pmax[1]: .4f},{pmax[2]: .4f})"
        )


def _log_mask_coverage(name, mask):
    if mask is None:
        print(f"[V2 DEBUG] {name}: none")
        return
    m = np.asarray(mask).astype(bool)
    if m.ndim == 3:
        total_kept = int(m.sum())
        total = int(m.size)
        print(f"[V2 DEBUG] {name}: kept {total_kept}/{total} ({100.0 * total_kept / max(total, 1):.2f}%)")
        for i in range(m.shape[0]):
            kept = int(m[i].sum())
            count = int(m[i].size)
            print(f"[V2 DEBUG]   {i:02d}: kept {kept}/{count} ({100.0 * kept / max(count, 1):.2f}%)")
    else:
        kept = int(m.sum())
        count = int(m.size)
        print(f"[V2 DEBUG] {name}: kept {kept}/{count} ({100.0 * kept / max(count, 1):.2f}%)")


def _log_worldmirror_debug(predictions, views, camera_poses, camera_intrinsics, imgs_tensor):
    S, _, H, W = imgs_tensor.shape[1:]
    print("[V2 DEBUG] ================= WorldMirror V2 Debug =================")
    print(f"[V2 DEBUG] images: S={S}, H={H}, W={W}")
    _log_camera_table("input camera priors", camera_poses, camera_intrinsics, H, W)
    _log_camera_table("model predicted cameras", predictions.get("camera_poses"), predictions.get("camera_intrs"), H, W)
    if camera_poses is not None and predictions.get("camera_poses") is not None:
        pred = predictions["camera_poses"].detach().cpu().float()[0]
        inp = camera_poses.detach().cpu().float()
        print("[V2 DEBUG] predicted - input angular deltas")
        yaw_deltas = []
        roll_deltas = []
        for i in range(min(inp.shape[0], pred.shape[0])):
            iy, ip, ir = _camera_debug_angles(inp[i])
            py, pp, pr = _camera_debug_angles(pred[i])
            dyaw = _angle_delta_deg(py, iy)
            droll = _angle_delta_deg(pr, ir)
            yaw_deltas.append(dyaw)
            roll_deltas.append(droll)
            print(f"[V2 DEBUG]   {i:02d}: dyaw={dyaw: .3f}, dpitch={pp - ip: .3f}, droll={droll: .3f}")
        if roll_deltas:
            mean_abs_roll = sum(abs(v) for v in roll_deltas) / len(roll_deltas)
            mean_abs_yaw = sum(abs(v) for v in yaw_deltas) / len(yaw_deltas)
            if mean_abs_roll > 120.0 and mean_abs_yaw > 120.0:
                print(
                    "[V2 DEBUG] warning: input camera priors look axis-flipped relative to model predictions; "
                    "try pose_convention=opencv_c2w before tuning offsets."
                )
    print("[V2 DEBUG] " + _tensor_stats_line("depth", predictions.get("depth")))
    print("[V2 DEBUG] " + _tensor_stats_line("gs_depth", predictions.get("gs_depth")))
    print("[V2 DEBUG] " + _tensor_stats_line("depth_conf", predictions.get("depth_conf")))
    print("[V2 DEBUG] " + _tensor_stats_line("pts3d_conf", predictions.get("pts3d_conf")))
    print("[V2 DEBUG] " + _tensor_stats_line("gs_depth_conf", predictions.get("gs_depth_conf")))
    _log_per_view_points("pts3d", predictions.get("pts3d"), S, H, W)
    splats = predictions.get("splats")
    if isinstance(splats, dict):
        _log_per_view_points("splats.means", splats.get("means"), S, H, W)
    print("[V2 DEBUG] =========================================================")


# ─────────────────────────────────────────────────────────────────────────────
# VNCCS_LoadWorldMirrorV2Model
# ─────────────────────────────────────────────────────────────────────────────
class VNCCS_LoadWorldMirrorV2Model:
    """Download and load WorldMirror 2.0 (tencent/HY-World-2.0)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "precision": (["bf16", "fp8", "float32"], {
                    "default": "bf16",
                    "tooltip": (
                        "bf16: recommended, ~2× VRAM vs float32. "
                        "fp8: weight-only quantization via torchao, ~2× vs bf16 (requires torchao, Ampere+). "
                        "float32: full precision."
                    ),
                }),
            }
        }

    RETURN_TYPES = ("WORLDMIRROR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "VNCCS/3D"

    def load_model(self, device="cuda", precision="bf16"):
        from huggingface_hub import snapshot_download
        from hyworld2.worldrecon.hyworldmirror.models.models.worldmirror import WorldMirror

        # ── resolve local cache path ──────────────────────────────────────────
        models_base = (
            folder_paths.models_dir if FOLDER_PATHS_AVAILABLE
            else os.path.join(PROJECT_ROOT, "models")
        )
        local_dir = os.path.join(models_base, "WorldMirror-V2")
        model_dir = os.path.join(local_dir, "HY-WorldMirror-2.0")
        weights   = os.path.join(model_dir, "model.safetensors")
        config    = os.path.join(model_dir, "config.json")

        # ── download if not cached ────────────────────────────────────────────
        if os.path.exists(weights) and os.path.exists(config):
            print(f"✅ [V2] Cached model: {model_dir}")
        else:
            print(f"⬇️ [V2] Downloading → {model_dir}  (~5 GB)")
            snapshot_download(
                repo_id="tencent/HY-World-2.0",
                allow_patterns=["HY-WorldMirror-2.0/**"],
                local_dir=local_dir,
            )
            print("✅ [V2] Download complete")

        # ── load on CPU first ─────────────────────────────────────────────────
        # Always load without enable_bf16 so weights arrive as float32 and
        # .to() is the standard nn.Module version. Precision is applied before
        # the CUDA move so loading does not briefly allocate the full fp32 model
        # in VRAM.
        print(f"🔄 [V2] Loading model (device={device}, precision={precision})")
        model = WorldMirror.from_pretrained(model_dir)

        # ── bf16 ──────────────────────────────────────────────────────────────
        if precision == "bf16":
            _prepare_worldmirror_bf16(model)

        # ── fp8 weight-only via torchao ───────────────────────────────────────
        elif precision == "fp8":
            # fp8 weight-only: weights stored as e4m3fn, dequantized to bf16 for matmul.
            # Uses bf16 activations in the forward pass.
            _prepare_worldmirror_bf16(model)
            if str(device).startswith("cuda") and torch.cuda.is_available():
                # TorchAO float8 tensor subclasses do not reliably survive a
                # post-quantization CPU -> CUDA .to(). Move the already-bf16
                # model first, then quantize on the target GPU.
                _move_worldmirror(model, device)
            _quantize_bf16_linear_weights_fp8(model)

        if _first_real_device(model) != torch.device(device):
            _move_worldmirror(model, device)

        model.eval()
        print("✅ [V2] Model ready")

        return ({"model": model, "device": device, "precision": precision},)


# ─────────────────────────────────────────────────────────────────────────────
# VNCCS_WorldMirrorV2_3D
# ─────────────────────────────────────────────────────────────────────────────
class VNCCS_WorldMirrorV2_3D:
    """WorldMirror V2 — 3D reconstruction from images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":     ("WORLDMIRROR_MODEL",),
                "images":    ("IMAGE",),
                "use_gsplat": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Gaussian Splatting output. Requires gsplat>=1.5.3."
                }),
            },
            "optional": {
                "target_size": ("INT", {
                    "default": 952, "min": 252, "max": 4096, "step": 14,
                    "tooltip": "Longest side in pixels. Experimental high values are VRAM-heavy; use low-VRAM modes above 1400."
                }),
                "offload_scheme": (["none", "model_cpu_offload"], {"default": "none"}),
                "low_vram_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply the low-VRAM profile: depth_only, frame chunks=1, GS param chunks=1, transformer MLP chunks=8192."
                }),
                "head_frame_chunk_size": ("INT", {
                    "default": 2, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Frames processed at once by depth/point/normal/GS heads. Lower values reduce VRAM, especially for FP8 multi-view runs."
                }),
                "head_compute_mode": (["all", "depth+gs", "depth_only"], {
                    "default": "all",
                    "tooltip": "all computes every output head. depth+gs skips points/normals to raise the VRAM ceiling. depth_only also skips native GS."
                }),
                "gs_param_chunk_size": ("INT", {
                    "default": 1, "min": 1, "max": 24, "step": 1,
                    "tooltip": "Frames processed at once by the Gaussian parameter Conv2d head. 1 uses the least VRAM."
                }),
                "transformer_mlp_chunk_size": ("INT", {
                    "default": 0, "min": 0, "max": 262144, "step": 4096,
                    "tooltip": "Token chunk size for transformer MLPs. 0 disables. Lower values reduce VRAM at high target_size but are slower."
                }),
                "confidence_percentile": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Discard bottom N% lowest-confidence points."
                }),
                "apply_sky_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Remove sky. V2 uses its own depth_mask prediction — no ONNX required."
                }),
                "filter_edges": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove points at depth discontinuities."
                }),
                "filter_splats": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply filter masks to Gaussian splats. Off keeps panorama coverage and avoids mask-carved black holes."
                }),
                "edge_normal_threshold": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 90.0, "step": 0.5}),
                "edge_depth_threshold":  ("FLOAT", {"default": 0.03, "min": 0.001, "max": 0.5, "step": 0.001}),
                "apply_confidence_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Discard the lowest-confidence points using confidence_percentile. Official V2 defaults this off."
                }),
                "camera_conditioning": (["pose+intrinsics", "intrinsics_only", "pose_only", "none"], {
                    "default": "pose+intrinsics",
                    "tooltip": "Which input camera priors to pass into WorldMirror V2. Panorama poses are rotation-only, so testing intrinsics_only/none can reduce seam conflicts."
                }),
                "missing_camera_strategy": (["off", "stabilize_predicted_intrinsics", "reproject_stabilized_predicted"], {
                    "default": "off",
                    "tooltip": "When camera inputs are missing, derive pseudo input cameras from WorldMirror predictions. stabilize_predicted_intrinsics keeps predicted poses but uses sequence-median focal lengths. reproject_stabilized_predicted also rebuilds pts3d/splat means from depth with those stable cameras."
                }),
                "splat_camera_source": (["input_align", "input_when_available", "input_icp_scaled_depth", "predicted"], {
                    "default": "input_align",
                    "tooltip": "input_align keeps model depth/splats internally consistent, then aligns the complete predicted reconstruction to input cameras. input_when_available applies full ICP alignment. input_icp_scaled_depth is experimental and backprojects depths from input cameras. predicted matches official inference."
                }),
                "splat_color_source": (["input_image", "model_sh"], {
                    "default": "input_image",
                    "tooltip": "input_image colors every Gaussian from the source view RGB, avoiding black SH artifacts. model_sh preserves the model residual SH output."
                }),
                "adaptive_target_size": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Clamp target_size to the input resolution, matching the official Gradio flow more closely."
                }),
                "apply_model_masks": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Filter outputs using V2 native depth_mask / gs_depth_mask predictions."
                }),
                "model_mask_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Keep pixels whose native model mask is at least this value."
                }),
                "voxel_prune_splats": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Voxel-merge Gaussian splats after inference. Matches official saving flow and keeps panorama files smaller."
                }),
                "voxel_size": ("FLOAT", {
                    "default": 0.002, "min": 0.0001, "max": 0.1, "step": 0.0001,
                    "tooltip": "Voxel size used when voxel_prune_splats is enabled."
                }),
                "splat_scale_multiplier": ("FLOAT", {
                    "default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05,
                    "tooltip": "Multiply Gaussian scale before saving. Increase slightly if dense visible surfaces have pinholes."
                }),
                "splat_opacity_floor": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Minimum Gaussian opacity before saving. Raise for debugging holes caused by transparent splats."
                }),
                "debug_log": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print camera/depth/splat alignment diagnostics to the ComfyUI console."
                }),
                "camera_intrinsics": ("TENSOR", {
                    "tooltip": "Optional: intrinsics from Equirect360ToViews node."
                }),
                "camera_poses": ("TENSOR", {
                    "tooltip": "Optional: extrinsics from Equirect360ToViews node."
                }),
                "depth_prior": ("IMAGE", {
                    "tooltip": "Optional depth prior matching the input views. Enables WorldMirror cond_flags[1]."
                }),
            }
        }

    RETURN_TYPES  = (
        "PLY_DATA", "IMAGE", "IMAGE", "TENSOR", "TENSOR", "VNCCS_SPLAT",
        "IMAGE", "IMAGE",
    )
    RETURN_NAMES  = (
        "ply_data", "depth_maps", "normal_maps", "camera_poses", "camera_intrinsics", "raw_splats",
        "filter_mask", "gs_filter_mask",
    )
    FUNCTION      = "run_inference"
    CATEGORY      = "VNCCS/3D"

    def run_inference(
        self,
        model,
        images,
        use_gsplat          = True,
        target_size         = 952,
        offload_scheme      = "none",
        low_vram_mode = False,
        head_frame_chunk_size = 2,
        head_compute_mode = "all",
        gs_param_chunk_size = 1,
        transformer_mlp_chunk_size = 0,
        confidence_percentile = 10.0,
        apply_sky_mask      = False,
        filter_edges        = True,
        filter_splats       = False,
        edge_normal_threshold = 1.0,
        edge_depth_threshold  = 0.03,
        apply_confidence_mask = False,
        camera_conditioning = "pose+intrinsics",
        normalize_camera_poses_to_first = False,
        missing_camera_strategy = "off",
        splat_camera_source = "input_align",
        splat_color_source = "input_image",
        adaptive_target_size = False,
        apply_model_masks = False,
        model_mask_threshold = 0.5,
        voxel_prune_splats = True,
        voxel_size = 0.002,
        splat_scale_multiplier = 1.0,
        splat_opacity_floor = 0.0,
        debug_log = False,
        camera_intrinsics   = None,
        camera_poses        = None,
        depth_prior         = None,
    ):

        if low_vram_mode:
            head_frame_chunk_size = 1
            head_compute_mode = "depth_only"
            gs_param_chunk_size = 1
            transformer_mlp_chunk_size = 8192
        target_size    = (target_size // _PATCH_SIZE) * _PATCH_SIZE
        if adaptive_target_size:
            target_size = _adaptive_target_size_from_images(images, target_size)
        worldmirror    = model["model"]
        model_precision = model.get("precision", "unknown") if isinstance(model, dict) else "unknown"
        _set_worldmirror_head_frame_chunk_size(worldmirror, head_frame_chunk_size)
        _set_worldmirror_gs_param_chunk_size(worldmirror, gs_param_chunk_size)
        _set_worldmirror_transformer_mlp_chunk_size(worldmirror, transformer_mlp_chunk_size)
        exec_dev       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        original_dev   = _first_real_device(worldmirror)
        has_meta_params = _module_has_meta_tensors(worldmirror)

        # ── 1. Preprocess: ComfyUI IMAGE [B,H,W,C] → tensor [1,S,3,H,W] ─────
        B = images.shape[0]
        camera_poses, camera_intrinsics = _align_camera_priors_to_image_count(
            images,
            camera_poses,
            camera_intrinsics,
        )
        if normalize_camera_poses_to_first and isinstance(camera_poses, torch.Tensor):
            camera_poses = _normalize_camera_poses_to_first(camera_poses)
            print("[V2] Official prior-camera normalization enabled: poses are relative to frame 0.")
        tensor_list = []
        adjusted_intrinsics = camera_intrinsics.clone().float() if isinstance(camera_intrinsics, torch.Tensor) else None

        for i in range(B):
            img_np  = (images[i].cpu().numpy()[..., :3] * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            t, orig_w, orig_h, new_w, new_h = _resize_to_tensor(pil_img, target_size)

            if adjusted_intrinsics is not None:
                sx, sy = new_w / orig_w, new_h / orig_h
                adjusted_intrinsics[i, 0, 0] *= sx
                adjusted_intrinsics[i, 1, 1] *= sy
                adjusted_intrinsics[i, 0, 2] *= sx
                adjusted_intrinsics[i, 1, 2] *= sy

            # centre-crop height if it exceeds target_size
            if new_h > target_size:
                crop = (new_h - target_size) // 2
                t = t[:, crop:crop + target_size, :]
                if adjusted_intrinsics is not None:
                    adjusted_intrinsics[i, 1, 2] -= crop

            # centre-crop width if a future resize strategy ever exceeds target_size
            if new_w > target_size:
                crop = (new_w - target_size) // 2
                t = t[:, :, crop:crop + target_size]
                if adjusted_intrinsics is not None:
                    adjusted_intrinsics[i, 0, 2] -= crop

            tensor_list.append(t)

        imgs_tensor = torch.stack(tensor_list).unsqueeze(0).to(exec_dev)  # [1,S,3,H,W]
        camera_intrinsics = adjusted_intrinsics

        # ── 2. Build views dict + cond_flags ──────────────────────────────────
        views      = {"img": imgs_tensor}
        cond_flags = [0, 0, 0]  # [pose, depth, intrinsics]

        use_pose_prior = camera_conditioning in ("pose+intrinsics", "pose_only")
        use_intrinsics_prior = camera_conditioning in ("pose+intrinsics", "intrinsics_only")
        has_input_cameras = camera_poses is not None and camera_intrinsics is not None
        use_input_global_align = splat_camera_source == "input_align" and has_input_cameras
        use_input_icp_align = splat_camera_source == "input_when_available" and has_input_cameras
        use_input_icp_scaled_depth = splat_camera_source == "input_icp_scaled_depth" and has_input_cameras
        use_input_splat_cameras = use_input_global_align or use_input_icp_align or use_input_icp_scaled_depth

        if (use_pose_prior or use_input_splat_cameras) and camera_poses is not None:
            views["camera_poses"] = camera_poses.unsqueeze(0).to(exec_dev)
        if use_pose_prior and camera_poses is not None:
            cond_flags[0] = 1
        if (use_intrinsics_prior or use_input_splat_cameras) and camera_intrinsics is not None:
            views["camera_intrs"] = camera_intrinsics.unsqueeze(0).to(exec_dev)
        if use_intrinsics_prior and camera_intrinsics is not None:
            cond_flags[2] = 1
        if depth_prior is not None:
            depth_tensor = _resize_depth_prior(depth_prior, target_size, B).to(exec_dev)
            views["depthmap"] = depth_tensor
            cond_flags[1] = 1

        # ── 3. Offload ────────────────────────────────────────────────────────
        if has_meta_params and offload_scheme != "model_cpu_offload":
            raise RuntimeError(
                "WorldMirror V2 model is currently offloaded/meta. Set offload_scheme "
                "to model_cpu_offload, or reload the Load WorldMirror V2 Model node "
                "before running with offload_scheme=none."
            )
        if (
            offload_scheme == "model_cpu_offload"
            and (model_precision == "fp8" or _has_torchao_tensor_subclasses(worldmirror))
        ):
            print("⚠️ [V2] model_cpu_offload is not compatible with torchao fp8 weights; using GPU residency.")
            offload_scheme = "none"
        if offload_scheme == "model_cpu_offload" and exec_dev.type == "cuda":
            if has_meta_params:
                print("ℹ️ [V2] Model is already accelerate-offloaded; keeping existing offload hooks.")
            else:
                try:
                    from accelerate import cpu_offload
                    cpu_offload(worldmirror, execution_device=exec_dev)
                except Exception as e:
                    print(f"⚠️ [V2] model_cpu_offload failed ({e}), moving to GPU.")
                    _move_worldmirror(worldmirror, exec_dev)
        else:
            if original_dev != exec_dev:
                _move_worldmirror(worldmirror, exec_dev)

        # ── 4. Inference ──────────────────────────────────────────────────────
        original_heads = _apply_worldmirror_head_compute_mode(worldmirror, head_compute_mode, use_gsplat and GSPLAT_AVAILABLE)
        gs_renderer = getattr(worldmirror, "gs_renderer", None)
        original_inference_position_from = (
            getattr(gs_renderer, "inference_position_from", "gsdepth+predcamera")
            if gs_renderer is not None else None
        )
        worldmirror.enable_gs = bool(worldmirror.enable_gs and use_gsplat and GSPLAT_AVAILABLE)
        effective_splat_camera_source = "predicted"
        if worldmirror.enable_gs and gs_renderer is not None:
            gs_renderer.inference_position_from = "gsdepth+predcamera"
            if use_input_global_align:
                effective_splat_camera_source = "predicted+global_input_align"
            elif use_input_icp_align:
                effective_splat_camera_source = "predicted+input_icp"
            elif use_input_icp_scaled_depth:
                effective_splat_camera_source = "predicted_depth+input_camera_icp_scale"
            else:
                effective_splat_camera_source = "predicted"

        try:
            print(
                f"🚀 [V2] Inference: {B} images @ {target_size}px, gs={worldmirror.enable_gs}, "
                f"camera_conditioning={camera_conditioning}, splat_camera_source={splat_camera_source} "
                f"(effective={effective_splat_camera_source}), splat_color_source={splat_color_source}, "
                f"head_compute_mode={head_compute_mode}, head_frame_chunk_size={head_frame_chunk_size}, "
                f"gs_param_chunk_size={gs_param_chunk_size}, transformer_mlp_chunk_size={transformer_mlp_chunk_size}"
            )
            with torch.no_grad():
                # Determine the target sequence length
                num_images = images.shape[0] if isinstance(images, torch.Tensor) else len(images)

                # Find the mismatching sequence length (num_poses) inside views
                num_poses = None
                
                # Case A: views is a dictionary
                if isinstance(views, dict):
                    for k, v in views.items():
                        if isinstance(v, torch.Tensor):
                            if v.dim() >= 2 and v.shape[1] > 0 and v.shape[1] != num_images:
                                num_poses = v.shape[1]
                                break
                            elif v.dim() == 1 and v.shape[0] > 0 and v.shape[0] != num_images:
                                num_poses = v.shape[0]
                                break
                        elif isinstance(v, list) and len(v) > 0 and len(v) != num_images:
                            num_poses = len(v)
                            break
                            
                # Case B: views is a list or tuple
                elif isinstance(views, (list, tuple)):
                    for v in views:
                        if isinstance(v, torch.Tensor):
                            if v.dim() >= 2 and v.shape[1] > 0 and v.shape[1] != num_images:
                                num_poses = v.shape[1]
                                break
                            elif v.dim() == 1 and v.shape[0] > 0 and v.shape[0] != num_images:
                                num_poses = v.shape[0]
                                break
                        elif isinstance(v, list) and len(v) > 0 and len(v) != num_images:
                            num_poses = len(v)
                            break

                # Case C: views is a custom object
                elif hasattr(views, '__dict__'):
                    for k, v in views.__dict__.items():
                        if isinstance(v, torch.Tensor):
                            if v.dim() >= 2 and v.shape[1] > 0 and v.shape[1] != num_images:
                                num_poses = v.shape[1]
                                break
                            elif v.dim() == 1 and v.shape[0] > 0 and v.shape[0] != num_images:
                                num_poses = v.shape[0]
                                break
                        elif isinstance(v, list) and len(v) > 0 and len(v) != num_images:
                            num_poses = len(v)
                            break

                # Perform the sequence-length alignment if a mismatch is confirmed
                if num_poses is not None and num_poses != num_images and num_images > 0:
                    print(f"[WorldMirror V2 PATCH] Aligning views (type={type(views).__name__}): {num_poses} poses -> {num_images} images.")
                    indices = torch.linspace(0, num_poses - 1, num_images, dtype=torch.long)

                    # 1. Align Dictionary
                    if isinstance(views, dict):
                        for k, v in list(views.items()):
                            if k == 'images':
                                continue
                            if isinstance(v, torch.Tensor):
                                if v.dim() >= 2 and v.shape[1] == num_poses:
                                    views[k] = v[:, indices.to(v.device)]
                                elif v.dim() >= 1 and v.shape[0] == num_poses:
                                    views[k] = v[indices.to(v.device)]
                            elif isinstance(v, list) and len(v) == num_poses:
                                views[k] = [v[i] for i in indices.tolist()]

                    # 2. Align List or Tuple
                    elif isinstance(views, (list, tuple)):
                        new_views = []
                        for v in views:
                            if isinstance(v, torch.Tensor):
                                if v.dim() >= 2 and v.shape[1] == num_poses:
                                    new_views.append(v[:, indices.to(v.device)])
                                elif v.dim() >= 1 and v.shape[0] == num_poses:
                                    new_views.append(v[indices.to(v.device)])
                                else:
                                    new_views.append(v)
                            elif isinstance(v, list) and len(v) == num_poses:
                                new_views.append([v[i] for i in indices.tolist()])
                            else:
                                new_views.append(v)
                        views = type(views)(new_views)

                    # 3. Align Object attributes
                    elif hasattr(views, '__dict__'):
                        for k, v in list(views.__dict__.items()):
                            if k == 'images':
                                continue
                            if isinstance(v, torch.Tensor):
                                if v.dim() >= 2 and v.shape[1] == num_poses:
                                    setattr(views, k, v[:, indices.to(v.device)])
                                elif v.dim() >= 1 and v.shape[0] == num_poses:
                                    setattr(views, k, v[indices.to(v.device)])
                            elif isinstance(v, list) and len(v) == num_poses:
                                setattr(views, k, [v[i] for i in indices.tolist()])

                # Run the model
                predictions = worldmirror(
                    views      = views,
                    cond_flags = cond_flags,
                    is_inference = True,
                )
            print("✅ [V2] Inference complete")
        finally:
            _restore_worldmirror_head_compute_mode(worldmirror, original_heads)
            if gs_renderer is not None and original_inference_position_from is not None:
                gs_renderer.inference_position_from = original_inference_position_from
            if offload_scheme == "none" and original_dev.type == "cpu" and not _module_has_meta_tensors(worldmirror):
                _move_worldmirror(worldmirror, "cpu")
                torch.cuda.empty_cache()

        S, H, W = predictions["depth"].shape[1:4]

        model_camera_poses = predictions.get("camera_poses")
        model_camera_intrs = predictions.get("camera_intrs")
        model_pts3d = predictions.get("pts3d")
        model_pts3d_filtered = None
        if isinstance(model_camera_poses, torch.Tensor):
            model_camera_poses = model_camera_poses.detach().clone()
        if isinstance(model_camera_intrs, torch.Tensor):
            model_camera_intrs = model_camera_intrs.detach().clone()
        if isinstance(model_pts3d, torch.Tensor):
            model_pts3d = model_pts3d.detach().clone()

        input_cameras_were_missing = camera_poses is None and camera_intrinsics is None
        missing_camera_info = None
        if input_cameras_were_missing and missing_camera_strategy != "off":
            auto_poses, auto_intrs, missing_camera_info = _auto_stabilize_missing_input_cameras(
                predictions,
                missing_camera_strategy,
                image_hw=(H, W),
            )
            if auto_poses is not None and auto_intrs is not None:
                camera_poses = auto_poses
                camera_intrinsics = auto_intrs
                if missing_camera_strategy == "stabilize_predicted_intrinsics":
                    predictions = dict(predictions)
                    predictions["camera_poses"] = auto_poses.unsqueeze(0).to(
                        device=predictions["depth"].device,
                        dtype=torch.float32,
                    )
                    predictions["camera_intrs"] = auto_intrs.unsqueeze(0).to(
                        device=predictions["depth"].device,
                        dtype=torch.float32,
                    )
                elif missing_camera_strategy == "reproject_stabilized_predicted":
                    predictions = _project_predictions_from_input_cameras(
                        predictions,
                        auto_poses,
                        auto_intrs,
                        scale_value=1.0,
                    )

        if debug_log:
            _log_worldmirror_debug(predictions, views, camera_poses, camera_intrinsics, imgs_tensor)

        input_camera_icp = None
        if use_input_global_align:
            predictions, input_camera_icp = _align_predictions_to_input_cameras_global(
                predictions,
                camera_poses,
                camera_intrinsics,
                image_hw=(H, W),
            )
        elif use_input_icp_align:
            predictions, input_camera_icp = _align_predictions_to_input_cameras_icp(
                predictions,
                camera_poses,
                camera_intrinsics,
                image_hw=(H, W),
            )
        elif use_input_icp_scaled_depth:
            predictions, input_camera_icp = _reproject_predictions_from_input_cameras_with_icp_scale(
                predictions,
                camera_poses,
                camera_intrinsics,
                image_hw=(H, W),
            )

        # ── 5. Sky mask (model-native first, ONNX fallback) ──────────────────
        sky_mask_np = None

        if apply_sky_mask and V2_UTILS_AVAILABLE:
            sky_mask_np = _compute_sky_mask_from_model(predictions, H, W, S)
            if sky_mask_np is not None:
                print(f"[V2] Sky mask: model-native ({S} frames)")
            elif ONNX_AVAILABLE:
                sky_model_path = _get_skyseg_path()
                if sky_model_path:
                    try:
                        sess   = onnxruntime.InferenceSession(sky_model_path)
                        frames = []
                        for i in range(S):
                            np_img = (imgs_tensor[0, i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                            m = segment_sky(np_img, sess)
                            if m.shape[:2] != (H, W) and cv2 is not None:
                                m = cv2.resize(m, (W, H))
                            frames.append(m)
                        sky_mask_np = np.stack(frames) > 0
                        print(f"[V2] Sky mask: ONNX ({S} frames)")
                    except Exception as e:
                        print(f"⚠️ [V2] Sky segmentation failed: {e}")

        # ── 6. Geometric filter mask (V2 native) ─────────────────────────────
        pts_mask = gs_mask = None
        if filter_edges and "normals" not in predictions:
            print("ℹ️ [V2] Edge filter skipped because head_compute_mode did not compute normals.")
        if V2_UTILS_AVAILABLE and "depth" in predictions and "normals" in predictions:
            pts_mask, gs_mask = compute_filter_mask(
                predictions          = predictions,
                imgs                 = imgs_tensor,
                img_paths            = [],            # not needed: sky_mask provided directly
                H                    = H, W = W, S = S,
                apply_confidence_mask= apply_confidence_mask,
                apply_edge_mask      = filter_edges,
                apply_sky_mask       = apply_sky_mask,
                confidence_percentile= confidence_percentile,
                edge_normal_threshold= edge_normal_threshold,
                edge_depth_threshold = edge_depth_threshold,
                sky_mask             = sky_mask_np,
                use_gs_depth         = "gs_depth" in predictions,
            )

        if apply_model_masks:
            def _native_mask(key):
                value = predictions.get(key)
                if value is None:
                    return None
                t = value[0].detach().cpu().float()
                if t.dim() == 4 and t.shape[-1] == 1:
                    t = t[..., 0]
                return (t.numpy() >= model_mask_threshold)

            depth_model_mask = _native_mask("depth_mask")
            gs_depth_model_mask = _native_mask("gs_depth_mask")
            if depth_model_mask is not None:
                pts_mask = depth_model_mask if pts_mask is None else (pts_mask & depth_model_mask)
            if gs_depth_model_mask is not None:
                gs_mask = gs_depth_model_mask if gs_mask is None else (gs_mask & gs_depth_model_mask)
            elif depth_model_mask is not None and gs_mask is not None:
                gs_mask = gs_mask & depth_model_mask

        if debug_log:
            _log_mask_coverage("pts_filter_mask", pts_mask)
            _log_mask_coverage("gs_filter_mask", gs_mask)
            if not filter_splats:
                print("[V2 DEBUG] gs_filter_mask is diagnostic only; filter_splats=false, so splats are not mask-filtered.")

        # ── 7. Filter pts3d ───────────────────────────────────────────────────
        filtered_pts = None
        if "pts3d" in predictions:
            pts = predictions["pts3d"][0].reshape(-1, 3)
            if pts_mask is not None:
                flat = torch.from_numpy(pts_mask.reshape(-1)).to(pts.device)
                filtered_pts = pts[flat]
            else:
                filtered_pts = pts
        if isinstance(model_pts3d, torch.Tensor):
            model_pts3d_flat = model_pts3d[0].reshape(-1, 3)
            if pts_mask is not None and int(np.prod(pts_mask.shape)) == int(model_pts3d_flat.shape[0]):
                flat = torch.from_numpy(pts_mask.reshape(-1)).to(model_pts3d_flat.device)
                model_pts3d_filtered = model_pts3d_flat[flat]
            else:
                model_pts3d_filtered = model_pts3d_flat

        # ── 8. Filter splats with GS-specific mask ────────────────────────────
        splats = predictions.get("splats")
        if splat_color_source == "input_image":
            splats = _replace_splat_colors_from_images(splats, imgs_tensor)
        splats = _tune_splats(splats, splat_scale_multiplier, splat_opacity_floor)
        if debug_log:
            _log_splat_stats(splats)
            if input_camera_icp is not None:
                print("[V2 DEBUG] splats stats after input-camera ICP processing")
                _log_splat_stats(splats)
                if isinstance(splats, dict):
                    _log_per_view_points("splats.means after input-camera projection", splats.get("means"), S, H, W)
        splat_mask = gs_mask if gs_mask is not None else pts_mask
        splats = _apply_mask_to_splats(splats, splat_mask if filter_splats else None)
        if voxel_prune_splats:
            splats = _voxel_prune_splats_dict(splats, voxel_size)

        # ── 9. Assemble PLY_DATA ──────────────────────────────────────────────
        ply_data = {
            "pts3d":          predictions.get("pts3d"),
            "pts3d_filtered": filtered_pts,
            "depth":          predictions.get("depth"),
            "normals":        predictions.get("normals"),
            "pts3d_conf":     predictions.get("pts3d_conf"),
            "depth_conf":     predictions.get("depth_conf"),
            "depth_mask":     predictions.get("depth_mask"),
            "gs_depth_conf":  predictions.get("gs_depth_conf"),
            "gs_depth_mask":  predictions.get("gs_depth_mask"),
            "splats":         splats,
            "images":         imgs_tensor,
            "filter_mask": (
                torch.from_numpy(pts_mask.reshape(-1)).to(exec_dev)
                if pts_mask is not None else None
            ),
            "gs_filter_mask": (
                torch.from_numpy(gs_mask.reshape(-1)).to(exec_dev)
                if gs_mask is not None else None
            ),
            "camera_poses":   predictions.get("camera_poses"),
            "camera_intrs":   predictions.get("camera_intrs"),
            "model_camera_poses": model_camera_poses,
            "model_camera_intrs": model_camera_intrs,
            "model_pts3d": model_pts3d,
            "model_pts3d_filtered": model_pts3d_filtered,
            "missing_camera_info": missing_camera_info,
        }

        # ── 10. Depth / normals → ComfyUI IMAGE [S,H,W,3] ────────────────────
        depth_t  = predictions.get("depth")
        normal_t = predictions.get("normals")

        if depth_t is not None:
            d = depth_t[0]                                          # [S,H,W,1]
            d = (d - d.min()) / (d.max() - d.min() + 1e-8)
            depth_out = d.repeat(1, 1, 1, 3).cpu().float()         # [S,H,W,3]
        else:
            depth_out = torch.zeros(B, target_size, target_size, 3)

        if normal_t is not None:
            normals_out = ((normal_t[0] + 1) / 2).cpu().float()    # [S,H,W,3]
        else:
            normals_out = torch.zeros(B, target_size, target_size, 3)

        # ── 11. Camera outputs + raw_splats for downstream nodes ──────────────
        cam_poses  = predictions.get("camera_poses")
        cam_intrs  = predictions.get("camera_intrs")
        if cam_poses  is not None: cam_poses  = cam_poses.cpu().float()
        if cam_intrs  is not None: cam_intrs  = cam_intrs.cpu().float()

        predictions["images"] = imgs_tensor   # needed by SplatRefiner

        fallback_shape = (S, H, W)
        filter_mask_out = _map_to_comfy_image(pts_mask, fallback_shape, normalize=False)
        gs_filter_mask_out = _map_to_comfy_image(gs_mask, fallback_shape, normalize=False)

        return (
            ply_data, depth_out, normals_out, cam_poses, cam_intrs, predictions,
            filter_mask_out, gs_filter_mask_out,
        )


class VNCCS_WorldMirrorV2_3D_Experimental(VNCCS_WorldMirrorV2_3D):
    """Experimental WorldMirror V2 reconstruction node.

    Kept as a separate ComfyUI node id so new knobs can evolve without
    changing saved state on the stable reconstruction node.
    """

    CATEGORY = "VNCCS/3D/Experimental"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = copy.deepcopy(super().INPUT_TYPES())
        optional = inputs.setdefault("optional", {})
        optional["splat_upsample_mode"] = (["none", "depth_backproject"], {
            "default": "none",
            "tooltip": "Experimental: replace model splats with dense high-res splats from upsampled depth and high-res RGB."
        })
        optional["splat_upsample_size"] = ("INT", {
            "default": 1022, "min": 252, "max": 4096, "step": 14,
            "tooltip": "High-res splat grid size. Model inference can stay at target_size=518."
        })
        optional["splat_upsample_depth_source"] = (["gs_depth", "depth"], {
            "default": "gs_depth",
            "tooltip": "Depth tensor to upsample and backproject into dense high-res splats."
        })
        optional["splat_upsample_scale"] = ("FLOAT", {
            "default": 0.003, "min": 0.0001, "max": 0.05, "step": 0.0001,
            "tooltip": "Constant Gaussian scale for high-res backprojected splats."
        })
        optional["splat_upsample_scale_mode"] = (["constant", "depth_adaptive", "footprint_adaptive", "hybrid_adaptive"], {
            "default": "depth_adaptive",
            "tooltip": "constant uses one world-space size; depth_adaptive grows with depth; footprint_adaptive follows local 3D spacing; hybrid uses both."
        })
        optional["splat_upsample_depth_scale_strength"] = ("FLOAT", {
            "default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05,
            "tooltip": "Depth growth strength, or local footprint multiplier in footprint/hybrid modes."
        })
        optional["splat_upsample_depth_scale_max"] = ("FLOAT", {
            "default": 3.0, "min": 1.0, "max": 12.0, "step": 0.1,
            "tooltip": "Maximum depth-adaptive multiplier for splat_upsample_scale."
        })
        optional["splat_upsample_opacity"] = ("FLOAT", {
            "default": 0.9, "min": 0.01, "max": 1.0, "step": 0.01,
            "tooltip": "Constant Gaussian opacity for high-res backprojected splats."
        })
        optional["splat_upsample_voxel_prune"] = ("BOOLEAN", {
            "default": True,
            "tooltip": "Voxel-merge high-res backprojected splats before saving."
        })
        optional["splat_upsample_voxel_size"] = ("FLOAT", {
            "default": 0.0015, "min": 0.0001, "max": 0.1, "step": 0.0001,
            "tooltip": "Voxel size for high-res backprojected splat merge."
        })
        optional["splat_upsample_max_points"] = ("INT", {
            "default": 9_000_000, "min": 0, "max": 50000000, "step": 100000,
            "tooltip": "Depth-aware downsample high-res splats to this many points. 0 disables the cap."
        })
        optional["splat_upsample_cap_far_bias"] = ("FLOAT", {
            "default": 1.75, "min": 0.0, "max": 8.0, "step": 0.05,
            "tooltip": "Preserve proportionally more far-depth splats when applying splat_upsample_max_points."
        })
        return inputs

    def _splat_value_to_points(self, value):
        if isinstance(value, list):
            if not value or not isinstance(value[0], torch.Tensor):
                return None
            return value[0]
        if isinstance(value, torch.Tensor):
            return value[0] if value.dim() >= 3 and value.shape[0] == 1 else value
        return None

    def _depth_aware_indices(self, depth_values, max_points, far_bias=1.75, bins=8):
        depth = depth_values.detach().cpu().float().flatten()
        total = int(depth.numel())
        max_points = int(max_points)
        if max_points <= 0 or total <= max_points:
            return None

        finite = torch.isfinite(depth)
        if not bool(finite.any()):
            return torch.linspace(0, total - 1, max_points, dtype=torch.long)

        depth_valid = depth[finite]
        d_min = depth_valid.min()
        d_max = depth_valid.max()
        if (d_max - d_min).abs().item() < 1e-8:
            return torch.linspace(0, total - 1, max_points, dtype=torch.long)

        normalized = ((depth - d_min) / (d_max - d_min)).clamp(0.0, 1.0)
        normalized = torch.where(torch.isfinite(normalized), normalized, torch.zeros_like(normalized))
        bin_ids = torch.clamp((normalized * bins).long(), max=bins - 1)
        quotas = []
        remaining = max_points
        for bin_id in range(bins):
            count = int((bin_ids == bin_id).sum().item())
            weight = 1.0 + float(far_bias) * (bin_id / max(1, bins - 1))
            quotas.append([bin_id, count, weight, 0])

        weighted_total = sum(count * weight for _, count, weight, _ in quotas)
        if weighted_total <= 0:
            return torch.linspace(0, total - 1, max_points, dtype=torch.long)

        for item in quotas:
            _, count, weight, _ = item
            quota = min(count, int(round(max_points * (count * weight) / weighted_total)))
            item[3] = quota
            remaining -= quota

        if remaining != 0:
            order = sorted(
                range(len(quotas)),
                key=lambda i: quotas[i][2],
                reverse=remaining > 0,
            )
            while remaining != 0:
                changed = False
                for i in order:
                    if remaining == 0:
                        break
                    count = quotas[i][1]
                    quota = quotas[i][3]
                    if remaining > 0 and quota < count:
                        quotas[i][3] += 1
                        remaining -= 1
                        changed = True
                    elif remaining < 0 and quota > 0:
                        quotas[i][3] -= 1
                        remaining += 1
                        changed = True
                if not changed:
                    break

        selected = []
        for bin_id, count, _, quota in quotas:
            if count <= 0 or quota <= 0:
                continue
            ids = torch.where(bin_ids == bin_id)[0]
            if ids.numel() <= quota:
                selected.append(ids)
            else:
                take = torch.linspace(0, ids.numel() - 1, quota, dtype=torch.long)
                selected.append(ids[take])

        if not selected:
            return torch.linspace(0, total - 1, max_points, dtype=torch.long)
        idx = torch.cat(selected).sort().values
        if idx.numel() > max_points:
            take = torch.linspace(0, idx.numel() - 1, max_points, dtype=torch.long)
            idx = idx[take]
        return idx

    def _cap_splats(self, splats, max_points, depth_values=None, far_bias=1.75):
        if not max_points or max_points <= 0 or not isinstance(splats, dict):
            return splats
        current_means = self._splat_value_to_points(splats.get("means"))
        if not isinstance(current_means, torch.Tensor) or current_means.shape[0] <= max_points:
            return splats

        n = current_means.shape[0]
        if isinstance(depth_values, torch.Tensor) and depth_values.numel() == n:
            idx = self._depth_aware_indices(depth_values, max_points, far_bias=far_bias)
            cap_kind = "depth-aware"
        else:
            distance = torch.linalg.norm(current_means.detach().cpu().float(), dim=-1)
            idx = self._depth_aware_indices(distance, max_points, far_bias=far_bias)
            cap_kind = "distance-aware"
        if idx is None:
            return splats

        for key, value in list(splats.items()):
            points = self._splat_value_to_points(value)
            if isinstance(points, torch.Tensor) and points.shape[0] == n:
                splats[key] = [points[idx.to(points.device)]]
        print(f"🧪 [V2 EXP] Upsample {cap_kind} point cap: {n} -> {int(max_points)} gaussians")
        return splats

    def _prepare_highres_views(self, images, camera_intrinsics, upsample_size):
        image_tensors = []
        intrinsics = camera_intrinsics.clone().float() if isinstance(camera_intrinsics, torch.Tensor) else None
        total = int(images.shape[0])
        for i in range(total):
            img_np = (images[i].detach().cpu().numpy()[..., :3] * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            t, orig_w, orig_h, new_w, new_h = _resize_to_tensor(pil_img, upsample_size)

            if new_h > upsample_size:
                crop = (new_h - upsample_size) // 2
                t = t[:, crop:crop + upsample_size, :]
                if intrinsics is not None:
                    intrinsics[i, 1, 2] -= crop
            if new_w > upsample_size:
                crop = (new_w - upsample_size) // 2
                t = t[:, :, crop:crop + upsample_size]
                if intrinsics is not None:
                    intrinsics[i, 0, 2] -= crop

            if intrinsics is not None:
                sx, sy = new_w / orig_w, new_h / orig_h
                intrinsics[i, 0, 0] *= sx
                intrinsics[i, 1, 1] *= sy
                intrinsics[i, 0, 2] *= sx
                intrinsics[i, 1, 2] *= sy

            image_tensors.append(t)

        return torch.stack(image_tensors).unsqueeze(0), intrinsics

    def _scale_intrinsics_to_highres(self, intrinsics, source_size, target_size):
        intrs = intrinsics.detach().cpu().float()
        if intrs.dim() == 4 and intrs.shape[0] == 1:
            intrs = intrs[0]
        if intrs.dim() != 3 or intrs.shape[-2:] != (3, 3):
            return None
        scale = float(target_size) / max(float(source_size), 1.0)
        out = intrs.clone()
        out[:, 0, 0] *= scale
        out[:, 1, 1] *= scale
        out[:, 0, 2] *= scale
        out[:, 1, 2] *= scale
        return out

    def _build_upsampled_splats(
        self,
        output,
        original_images,
        camera_poses,
        camera_intrinsics,
        upsample_size,
        depth_source,
        splat_scale,
        scale_mode,
        depth_scale_strength,
        depth_scale_max,
        splat_opacity,
        voxel_prune,
        voxel_size,
        max_points,
        cap_far_bias,
    ):
        if not V2_UTILS_AVAILABLE:
            print("⚠️ [V2 EXP] splat upsample skipped: V2 utilities unavailable.")
            return output
        if camera_poses is None or camera_intrinsics is None:
            print("⚠️ [V2 EXP] splat upsample skipped: camera_poses/camera_intrinsics required.")
            return output

        raw = output[5]
        if not isinstance(raw, dict):
            return output
        requested_depth_source = depth_source
        actual_depth_source = depth_source
        depth = raw.get(depth_source)
        if depth is None and depth_source != "depth":
            depth = raw.get("depth")
            actual_depth_source = "depth"
        if not isinstance(depth, torch.Tensor):
            print(f"⚠️ [V2 EXP] splat upsample skipped: missing depth source '{requested_depth_source}'.")
            return output

        if depth.dim() == 5:
            depth_low = depth[0, ..., 0]
        elif depth.dim() == 4 and depth.shape[-1] == 1:
            depth_low = depth[..., 0]
        elif depth.dim() == 4:
            depth_low = depth[0]
        else:
            print(f"⚠️ [V2 EXP] splat upsample skipped: unsupported depth shape {tuple(depth.shape)}.")
            return output

        source_h, source_w = int(depth_low.shape[1]), int(depth_low.shape[2])
        highres_imgs, highres_intrs = self._prepare_highres_views(original_images, camera_intrinsics, upsample_size)
        S, _, H, W = highres_imgs.shape[1:]
        depth_hi = torch.nn.functional.interpolate(
            depth_low.detach().cpu().float().unsqueeze(1),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[:, 0]

        if raw.get("_splat_camera_source") == "input_align":
            poses = _extract_pose_tensor(raw.get("camera_poses"))
            raw_intrs = raw.get("camera_intrs")
            intrs = self._scale_intrinsics_to_highres(raw_intrs, source_w, W) if isinstance(raw_intrs, torch.Tensor) else None
            if poses is None or intrs is None:
                print("⚠️ [V2 EXP] input_align upsample fallback: aligned predicted cameras unavailable; using input cameras.")
                predicted_poses = raw.get("camera_poses")
                if predicted_poses is None and len(output) > 3:
                    predicted_poses = output[3]
                poses = _rescale_input_pose_translation_to_prediction(camera_poses, predicted_poses)
                intrs = highres_intrs.detach().cpu().float()
            else:
                print("[V2 EXP] input_align upsample: backprojecting depth with aligned predicted cameras.")
        else:
            predicted_poses = raw.get("camera_poses")
            if predicted_poses is None and len(output) > 3:
                predicted_poses = output[3]
            poses = _rescale_input_pose_translation_to_prediction(camera_poses, predicted_poses)
            intrs = highres_intrs.detach().cpu().float()
        if poses is None:
            print("⚠️ [V2 EXP] splat upsample skipped: invalid input camera_poses.")
            return output
        pts, _, _ = depth_to_world_coords_points(depth_hi, poses, intrs)
        pose_t = poses[:, :3, 3]
        anchor_frames = pose_t.norm(dim=1) <= 1e-5
        keep_mask = _suppress_generated_surfaces_seen_by_anchors(
            pts,
            depth_hi,
            poses,
            intrs,
            anchor_frames,
        )
        keep_flat = keep_mask.reshape(-1)
        pts_flat = pts.reshape(S * H * W, 3)[keep_flat]
        rgb_flat = highres_imgs.permute(0, 1, 3, 4, 2).reshape(S * H * W, 3)[keep_flat]
        depth_flat_values = depth_hi.reshape(S * H * W)[keep_flat]
        means = pts_flat.unsqueeze(0)
        sh = ((rgb_flat.unsqueeze(0) - 0.5) / 0.28209479177387814).float()
        depth_flat = depth_flat_values.unsqueeze(0)

        def depth_adaptive_scales():
            d_min = depth_flat.min()
            d_max = depth_flat.max()
            if (d_max - d_min).abs().item() > 1e-8:
                depth_norm = ((depth_flat - d_min) / (d_max - d_min)).clamp(0.0, 1.0)
                multiplier = (1.0 + float(depth_scale_strength) * depth_norm).clamp(
                    1.0,
                    float(depth_scale_max),
                )
            else:
                multiplier = torch.ones_like(depth_flat)
            return torch.full_like(means, float(splat_scale)) * multiplier[..., None], multiplier

        def footprint_adaptive_scales(multiplier=None):
            dx = torch.linalg.norm(pts[:, :, 1:, :] - pts[:, :, :-1, :], dim=-1)
            dy = torch.linalg.norm(pts[:, 1:, :, :] - pts[:, :-1, :, :], dim=-1)
            dx = torch.cat([dx, dx[:, :, -1:]], dim=2)
            dy = torch.cat([dy, dy[:, -1:, :]], dim=1)
            footprint = torch.maximum(dx, dy).clamp_min(float(splat_scale))
            multiplier = max(1.0, float(depth_scale_strength)) if multiplier is None else float(multiplier)
            scales_hw = (footprint * multiplier).clamp(
                float(splat_scale),
                float(splat_scale) * float(depth_scale_max),
            )
            return scales_hw.reshape(S * H * W, 1)[keep_flat].unsqueeze(0).expand_as(means).float()

        if scale_mode == "hybrid_adaptive":
            scales_depth, multiplier = depth_adaptive_scales()
            scales_footprint = footprint_adaptive_scales(multiplier=1.0)
            scales = torch.maximum(scales_depth, scales_footprint)
            print(
                "🧪 [V2 EXP] Upsample hybrid-adaptive scale: "
                f"base={float(splat_scale):.5f}, max_multiplier={float(multiplier.max().item()):.2f}, "
                f"max_scale={float(scales.max().item()):.5f}, mean_scale={float(scales.mean().item()):.5f}"
            )
        elif scale_mode == "footprint_adaptive":
            scales = footprint_adaptive_scales()
            print(
                "🧪 [V2 EXP] Upsample footprint-adaptive scale: "
                f"base={float(splat_scale):.5f}, multiplier={max(1.0, float(depth_scale_strength)):.2f}, "
                f"max_scale={float(scales.max().item()):.5f}, mean_scale={float(scales.mean().item()):.5f}"
            )
        elif scale_mode == "depth_adaptive":
            scales, multiplier = depth_adaptive_scales()
            print(
                "🧪 [V2 EXP] Upsample depth-adaptive scale: "
                f"base={float(splat_scale):.5f}, max_multiplier={float(multiplier.max().item()):.2f}"
            )
        else:
            scales = torch.full_like(means, float(splat_scale))
        opacities = torch.full((1, means.shape[1]), float(splat_opacity), dtype=torch.float32)
        quats = torch.zeros((1, means.shape[1], 4), dtype=torch.float32)
        quats[..., 0] = 1.0
        weights = torch.ones((1, means.shape[1]), dtype=torch.float32)
        if bool(anchor_frames.any()) and bool((~anchor_frames).any()):
            anchor_weight = 64.0
            frame_weights = torch.where(
                anchor_frames,
                torch.full_like(pose_t[:, 0], anchor_weight),
                torch.ones_like(pose_t[:, 0]),
            )
            weights = frame_weights[:, None].expand(S, H * W).reshape(S * H * W)[keep_flat].unsqueeze(0).float()
            print(
                "[V2 EXP] High-res anchor splat priority enabled: "
                f"anchors={int(anchor_frames.sum().item())}/{S}, weight={anchor_weight:g}"
            )

        splats = {
            "means": means,
            "scales": scales,
            "quats": quats,
            "opacities": opacities,
            "sh": sh[:, :, None, :],
            "weights": weights,
        }
        n_before = means.shape[1]
        if voxel_prune:
            splats = _voxel_prune_splats_dict(splats, voxel_size)
            pruned_means = self._splat_value_to_points(splats.get("means")) if isinstance(splats, dict) else None
            if isinstance(pruned_means, torch.Tensor):
                print(f"🧪 [V2 EXP] Upsample voxel prune: {n_before} -> {pruned_means.shape[0]} gaussians")

        cap_depth = depth_flat.flatten() if not voxel_prune else None
        splats = self._cap_splats(splats, max_points, depth_values=cap_depth, far_bias=cap_far_bias)

        ply_data = dict(output[0])
        ply_data.update({
            "splats": splats,
            "skip_scale_filter": True,
            "images": highres_imgs,
            "camera_poses": poses.unsqueeze(0),
            "camera_intrs": intrs.unsqueeze(0),
            "pts3d": None,
            "pts3d_filtered": None,
        })
        raw_new = dict(raw)
        raw_new.update({
            "splats": splats,
            "images": highres_imgs,
            "camera_poses": poses.unsqueeze(0),
            "camera_intrs": intrs.unsqueeze(0),
        })

        print(
            f"🧪 [V2 EXP] Upsampled splats: {S} views @ {H}x{W}, "
            f"{n_before} raw gaussians from {actual_depth_source}"
        )
        return (
            ply_data,
            output[1],
            output[2],
            poses.unsqueeze(0),
            intrs.unsqueeze(0),
            raw_new,
            output[6],
            output[7],
        )

    def run_inference(
        self,
        model,
        images,
        use_gsplat=True,
        splat_upsample_mode="none",
        splat_upsample_size=1022,
        splat_upsample_depth_source="gs_depth",
        splat_upsample_scale=0.003,
        splat_upsample_scale_mode="depth_adaptive",
        splat_upsample_depth_scale_strength=1.0,
        splat_upsample_depth_scale_max=3.0,
        splat_upsample_opacity=0.9,
        splat_upsample_voxel_prune=True,
        splat_upsample_voxel_size=0.0015,
        splat_upsample_max_points=9_000_000,
        splat_upsample_cap_far_bias=1.75,
        **kwargs,
    ):
        kwargs = dict(kwargs)
        kwargs["camera_poses"], kwargs["camera_intrinsics"] = _align_camera_priors_to_image_count(
            images,
            kwargs.get("camera_poses"),
            kwargs.get("camera_intrinsics"),
        )
        output = super().run_inference(model, images, use_gsplat=use_gsplat, **kwargs)
        if splat_upsample_mode == "depth_backproject":
            output = self._build_upsampled_splats(
                output,
                images,
                kwargs.get("camera_poses"),
                kwargs.get("camera_intrinsics"),
                splat_upsample_size,
                splat_upsample_depth_source,
                splat_upsample_scale,
                splat_upsample_scale_mode,
                splat_upsample_depth_scale_strength,
                splat_upsample_depth_scale_max,
                splat_upsample_opacity,
                splat_upsample_voxel_prune,
                splat_upsample_voxel_size,
                splat_upsample_max_points,
                splat_upsample_cap_far_bias,
            )
        return output


class VNCCS_WorldMirrorV2_3D_Advanced(VNCCS_WorldMirrorV2_3D_Experimental):
    """Advanced WorldMirror V2 node with all tuning and debug controls."""

    CATEGORY = "VNCCS/3D/Advanced"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = copy.deepcopy(super().INPUT_TYPES())
        inputs.get("optional", {}).pop("low_vram_mode", None)
        inputs.setdefault("optional", {})["normalize_camera_poses_to_first"] = ("BOOLEAN", {
            "default": False,
            "tooltip": "Official pipeline behavior for prior cameras: convert input poses to frame-0-relative space with inv(first_pose) @ pose before conditioning/alignment."
        })
        return inputs


class VNCCS_WorldMirrorV2_3D_Clean(VNCCS_WorldMirrorV2_3D_Advanced):
    """Clean WorldMirror V2 node for the panorama-to-dense-splat workflow."""

    CATEGORY = "VNCCS/3D"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WORLDMIRROR_MODEL",),
                "images": ("IMAGE",),
            },
            "optional": {
                "target_size": ("INT", {
                    "default": 518, "min": 252, "max": 4096, "step": 14,
                    "tooltip": "Model inference resolution. Experimental high values are VRAM-heavy; dense splats can be upsampled separately."
                }),
                "offload_scheme": (["none", "model_cpu_offload"], {
                    "default": "none",
                    "tooltip": "Move model weights to CPU between GPU use to reduce VRAM at the cost of speed."
                }),
                "low_vram_mode": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply the low-VRAM profile: depth_only, frame chunks=1, GS param chunks=1, transformer MLP chunks=8192."
                }),
                "apply_sky_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Remove sky-like regions before saving. Useful for outdoor panoramas where sky can create far/infinite splats."
                }),
                "debug_log": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print camera/depth/splat alignment diagnostics to the ComfyUI console."
                }),
                "splat_upsample_scale": ("FLOAT", {
                    "default": 0.003, "min": 0.0001, "max": 0.05, "step": 0.0001,
                    "tooltip": "Gaussian size for dense backprojected splats."
                }),
                "splat_upsample_scale_mode": (["constant", "depth_adaptive", "footprint_adaptive", "hybrid_adaptive"], {
                    "default": "depth_adaptive",
                    "tooltip": "constant uses one Gaussian size; depth_adaptive grows with depth; footprint_adaptive follows local 3D spacing; hybrid uses both."
                }),
                "splat_upsample_depth_scale_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05,
                    "tooltip": "Depth growth strength, or local footprint multiplier in footprint/hybrid modes."
                }),
                "splat_upsample_depth_scale_max": ("FLOAT", {
                    "default": 3.0, "min": 1.0, "max": 12.0, "step": 0.1,
                    "tooltip": "Maximum multiplier over splat_upsample_scale for adaptive modes."
                }),
                "splat_upsample_voxel_prune": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Voxel-merge dense splats to reduce file size."
                }),
                "splat_upsample_voxel_size": ("FLOAT", {
                    "default": 0.0015, "min": 0.0001, "max": 0.1, "step": 0.0001,
                    "tooltip": "Voxel size for dense splat compression."
                }),
                "splat_upsample_max_points": ("INT", {
                    "default": 9_000_000, "min": 0, "max": 50_000_000, "step": 100_000,
                    "tooltip": "Depth/distance-aware cap for dense splats. Lower values reduce file size; higher values preserve distant surfaces."
                }),
                "splat_upsample_cap_far_bias": ("FLOAT", {
                    "default": 1.75, "min": 0.0, "max": 8.0, "step": 0.05,
                    "tooltip": "Preserve proportionally more far splats when applying the point cap."
                }),
                "splat_camera_source": (["input_align", "input_when_available", "input_icp_scaled_depth", "predicted"], {
                    "default": "input_align",
                    "tooltip": "input_align keeps model depth/splats internally consistent, then aligns the complete predicted reconstruction to input cameras. input_when_available applies full ICP alignment. input_icp_scaled_depth is experimental and backprojects depths from input cameras. predicted matches official inference."
                }),
                "camera_intrinsics": ("TENSOR", {
                    "tooltip": "Optional: intrinsics from Equirect360ToViews or WorldStereo."
                }),
                "camera_poses": ("TENSOR", {
                    "tooltip": "Optional: camera poses from Equirect360ToViews or WorldStereo."
                }),
                "depth_prior": ("IMAGE", {
                    "tooltip": "Optional depth prior matching the input views."
                }),
            },
        }

    def run_inference(
        self,
        model,
        images,
        target_size=518,
        offload_scheme="none",
        low_vram_mode=True,
        head_frame_chunk_size=2,
        head_compute_mode="depth+gs",
        gs_param_chunk_size=1,
        transformer_mlp_chunk_size=32768,
        apply_sky_mask=False,
        splat_camera_source="input_align",
        splat_upsample_mode="depth_backproject",
        splat_upsample_size=1022,
        splat_upsample_scale=0.003,
        splat_upsample_scale_mode="depth_adaptive",
        splat_upsample_depth_scale_strength=1.0,
        splat_upsample_depth_scale_max=3.0,
        splat_upsample_voxel_prune=True,
        splat_upsample_voxel_size=0.0015,
        splat_upsample_max_points=9_000_000,
        splat_upsample_cap_far_bias=1.75,
        debug_log=False,
        camera_intrinsics=None,
        camera_poses=None,
        depth_prior=None,
    ):
        splat_upsample_size = self._infer_splat_upsample_size(images)
        return super().run_inference(
            model,
            images,
            use_gsplat=True,
            target_size=target_size,
            offload_scheme=offload_scheme,
            low_vram_mode=low_vram_mode,
            head_frame_chunk_size=head_frame_chunk_size,
            head_compute_mode=head_compute_mode,
            gs_param_chunk_size=gs_param_chunk_size,
            transformer_mlp_chunk_size=transformer_mlp_chunk_size,
            confidence_percentile=10.0,
            apply_sky_mask=apply_sky_mask,
            filter_edges=True,
            filter_splats=False,
            edge_normal_threshold=1.0,
            edge_depth_threshold=0.03,
            apply_confidence_mask=False,
            camera_conditioning="pose+intrinsics",
            splat_camera_source=splat_camera_source,
            splat_color_source="input_image",
            adaptive_target_size=False,
            apply_model_masks=False,
            model_mask_threshold=0.5,
            voxel_prune_splats=True,
            voxel_size=0.002,
            splat_scale_multiplier=1.0,
            splat_opacity_floor=0.0,
            debug_log=debug_log,
            camera_intrinsics=camera_intrinsics,
            camera_poses=camera_poses,
            depth_prior=depth_prior,
            splat_upsample_mode="depth_backproject",
            splat_upsample_size=splat_upsample_size,
            splat_upsample_depth_source="gs_depth",
            splat_upsample_scale=splat_upsample_scale,
            splat_upsample_scale_mode=splat_upsample_scale_mode,
            splat_upsample_depth_scale_strength=splat_upsample_depth_scale_strength,
            splat_upsample_depth_scale_max=splat_upsample_depth_scale_max,
            splat_upsample_opacity=0.9,
            splat_upsample_voxel_prune=splat_upsample_voxel_prune,
            splat_upsample_voxel_size=splat_upsample_voxel_size,
            splat_upsample_max_points=splat_upsample_max_points,
            splat_upsample_cap_far_bias=splat_upsample_cap_far_bias,
        )

    @staticmethod
    def _infer_splat_upsample_size(images):
        if isinstance(images, torch.Tensor) and images.dim() >= 3:
            return max(int(images.shape[1]), int(images.shape[2]))
        return 1022


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def _get_skyseg_path():
    """Return local path to skyseg.onnx, downloading if absent."""
    base = folder_paths.models_dir if FOLDER_PATHS_AVAILABLE else os.path.join(PROJECT_ROOT, "models")
    path = os.path.join(base, "skyseg.onnx")
    if not os.path.exists(path) and V2_UTILS_AVAILABLE:
        try:
            download_file_from_url(
                "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx",
                path,
            )
        except Exception as e:
            print(f"❌ [V2] skyseg.onnx download failed: {e}")
    return path if os.path.exists(path) else None


# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI registration
# ─────────────────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "VNCCS_LoadWorldMirrorV2Model":          VNCCS_LoadWorldMirrorV2Model,
    "VNCCS_WorldMirrorV2_3D":                VNCCS_WorldMirrorV2_3D_Clean,
    "VNCCS_WorldMirrorV2_3D_Advanced":       VNCCS_WorldMirrorV2_3D_Advanced,
    "VNCCS_WorldMirrorV2_3D_Experimental":   VNCCS_WorldMirrorV2_3D_Experimental,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_LoadWorldMirrorV2Model":          "🌍 Load WorldMirror V2 Model",
    "VNCCS_WorldMirrorV2_3D":                "🌍 WorldMirror V2 3D Reconstruction",
    "VNCCS_WorldMirrorV2_3D_Advanced":       "🌍 WorldMirror V2 3D Reconstruction Advanced",
    "VNCCS_WorldMirrorV2_3D_Experimental":   "🌍 WorldMirror V2 3D Reconstruction Advanced",
}
