import math

import numpy as np
import torch

try:
    import cv2
except ImportError:
    cv2 = None


SH_C0 = 0.28209479177387814


def _first_tensor(value):
    if isinstance(value, list) and value:
        value = value[0]
    return value


def _as_flat_tensor(value, dim, default=None):
    value = _first_tensor(value)
    if value is None:
        return default
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    value = value.detach().float()
    if value.dim() >= 3 and value.shape[0] == 1:
        value = value[0]
    return value.reshape(-1, dim)


def _extract_splats(ply_data):
    splats = (ply_data or {}).get("splats")
    if not isinstance(splats, dict):
        return None

    means = _as_flat_tensor(splats.get("means"), 3)
    if means is None:
        return None
    n = means.shape[0]

    scales = _as_flat_tensor(splats.get("scales"), 3, torch.full((n, 3), 0.003))
    quats = _as_flat_tensor(splats.get("quats"), 4)
    if quats is None:
        quats = torch.zeros(n, 4)
        quats[:, 0] = 1.0

    sh = _as_flat_tensor(splats.get("sh"), 3)
    if sh is not None:
        rgb = (sh * SH_C0 + 0.5).clamp(0.0, 1.0)
        colors_for_gsplat = sh[:, None, :]
    else:
        rgb = _as_flat_tensor(splats.get("colors"), 3, torch.full((n, 3), 0.5)).clamp(0.0, 1.0)
        colors_for_gsplat = ((rgb - 0.5) / SH_C0)[:, None, :]

    opacities = _as_flat_tensor(splats.get("opacities"), 1, torch.full((n, 1), 0.9)).reshape(-1)
    n = min(means.shape[0], scales.shape[0], quats.shape[0], rgb.shape[0], opacities.shape[0])
    return {
        "means": means[:n].contiguous(),
        "scales": scales[:n].clamp_min(1e-8).contiguous(),
        "quats": quats[:n].contiguous(),
        "rgb": rgb[:n].contiguous(),
        "sh": colors_for_gsplat[:n].contiguous(),
        "opacities": opacities[:n].contiguous(),
    }


def _opacity_to_alpha(opacities):
    opacities = opacities.float()
    if opacities.numel() and (float(opacities.min()) < 0.0 or float(opacities.max()) > 1.0):
        return opacities.sigmoid().clamp(0.0, 1.0)
    return opacities.clamp(0.0, 1.0)


def _build_intrinsics(width, height, fov_deg):
    focal = width / (2.0 * math.tan(math.radians(float(fov_deg)) / 2.0))
    K = torch.eye(3, dtype=torch.float32)
    K[0, 0] = focal
    K[1, 1] = focal
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0
    return K


def _make_look_at_pose(position, target):
    position = position.float()
    target = target.float()
    forward = target - position
    forward = forward / forward.norm().clamp_min(1e-8)
    up_hint = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32)
    right = torch.linalg.cross(forward, up_hint, dim=0)
    if right.norm() < 1e-6:
        up_hint = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        right = torch.linalg.cross(forward, up_hint, dim=0)
    right = right / right.norm().clamp_min(1e-8)
    up = torch.linalg.cross(right, forward, dim=0)

    pose = torch.eye(4, dtype=torch.float32)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = forward
    pose[:3, 3] = position
    return pose


def _normalize_pose_tensor(camera_poses):
    if not isinstance(camera_poses, torch.Tensor):
        return None
    poses = camera_poses.detach().cpu().float()
    if poses.dim() == 4 and poses.shape[0] == 1:
        poses = poses[0]
    if poses.dim() == 3 and poses.shape[-2:] == (4, 4):
        return poses
    return None


def _normalize_intrinsics_tensor(camera_intrinsics):
    if not isinstance(camera_intrinsics, torch.Tensor):
        return None
    intrs = camera_intrinsics.detach().cpu().float()
    if intrs.dim() == 4 and intrs.shape[0] == 1:
        intrs = intrs[0]
    if intrs.dim() == 3 and intrs.shape[-2:] == (3, 3):
        return intrs
    return None


def _camera_pitch_degrees(poses):
    poses = _normalize_pose_tensor(poses)
    if poses is None or poses.shape[0] == 0:
        return None
    forward = poses[:, :3, 2]
    norm = forward.norm(dim=-1).clamp_min(1e-8)
    pitch = torch.asin((-forward[:, 1] / norm).clamp(-1.0, 1.0))
    return pitch * (180.0 / math.pi)


def _filter_source_views_by_pitch(images, poses, intrinsics, mode, tolerance_deg, label):
    if mode == "all":
        return images, poses, intrinsics

    poses = _normalize_pose_tensor(poses)
    intrinsics = _normalize_intrinsics_tensor(intrinsics)
    if poses is None:
        print(f"[{label}] source pitch filter skipped: source_camera_poses are missing or invalid.")
        return images, poses, intrinsics

    pitch = _camera_pitch_degrees(poses)
    if pitch is None:
        return images, poses, intrinsics
    keep = pitch.abs() <= float(tolerance_deg)
    if not bool(keep.any()):
        print(
            f"[{label}] source pitch filter kept 0/{poses.shape[0]} views; "
            "keeping all source views to avoid an empty batch."
        )
        return images, poses, intrinsics

    kept = int(keep.sum().item())
    print(
        f"[{label}] source pitch filter {mode}: kept {kept}/{poses.shape[0]} "
        f"views with |pitch| <= {float(tolerance_deg):.2f} deg."
    )

    filtered_images = images
    if isinstance(images, torch.Tensor) and images.dim() >= 4 and int(images.shape[0]) == int(poses.shape[0]):
        filtered_images = images.detach().cpu().float()[keep]
    elif images is not None:
        print(f"[{label}] source image count does not match poses; images were not pitch-filtered.")

    filtered_intrs = intrinsics
    if intrinsics is not None and int(intrinsics.shape[0]) == int(poses.shape[0]):
        filtered_intrs = intrinsics[keep]
    elif intrinsics is not None:
        print(f"[{label}] source intrinsics count does not match poses; intrinsics were not pitch-filtered.")

    return filtered_images, poses[keep], filtered_intrs


def _resize_images_and_intrinsics(images, intrinsics, width, height):
    images = images.detach().cpu().float()
    if images.dim() != 4:
        raise ValueError(f"source_images must be [B,H,W,C], got {tuple(images.shape)}")
    src_h, src_w = int(images.shape[1]), int(images.shape[2])
    if src_w == int(width) and src_h == int(height):
        return images, intrinsics

    resized = torch.nn.functional.interpolate(
        images.permute(0, 3, 1, 2),
        size=(int(height), int(width)),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1).contiguous()

    if intrinsics is not None:
        intrinsics = intrinsics.clone()
        sx = float(width) / max(float(src_w), 1.0)
        sy = float(height) / max(float(src_h), 1.0)
        intrinsics[:, 0, 0] *= sx
        intrinsics[:, 1, 1] *= sy
        intrinsics[:, 0, 2] *= sx
        intrinsics[:, 1, 2] *= sy
    print(f"[SplatPatchPOC] resized source views for combined batch: {src_w}x{src_h} -> {width}x{height}")
    return resized, intrinsics


def _resize_image_batch(images, width, height):
    images = images.detach().cpu().float()
    if images.numel() and float(images.max()) > 2.0:
        images = images / 255.0
    images = images.clamp(0.0, 1.0)
    if images.dim() == 3:
        images = images.unsqueeze(-1)
    if images.dim() != 4:
        raise ValueError(f"IMAGE batch must be [B,H,W,C], got {tuple(images.shape)}")
    if images.shape[-1] == 1:
        images = images.repeat(1, 1, 1, 3)
    elif images.shape[-1] > 3:
        images = images[..., :3]
    if int(images.shape[1]) == int(height) and int(images.shape[2]) == int(width):
        return images.contiguous()
    return torch.nn.functional.interpolate(
        images.permute(0, 3, 1, 2),
        size=(int(height), int(width)),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0)


def _scale_intrinsics_to_image(intrinsics, source_width, source_height, target_width, target_height):
    if intrinsics is None:
        return None
    intrinsics = intrinsics.detach().cpu().float().clone()
    sx = float(target_width) / max(float(source_width), 1.0)
    sy = float(target_height) / max(float(source_height), 1.0)
    intrinsics[:, 0, 0] *= sx
    intrinsics[:, 1, 1] *= sy
    intrinsics[:, 0, 2] *= sx
    intrinsics[:, 1, 2] *= sy
    return intrinsics


def _infer_intrinsics_image_size(intrinsics):
    if intrinsics is None or intrinsics.numel() == 0:
        return None
    intrinsics = intrinsics.detach().cpu().float()
    cx = float(torch.median(intrinsics[:, 0, 2]).item())
    cy = float(torch.median(intrinsics[:, 1, 2]).item())
    if cx <= 0.0 or cy <= 0.0:
        return None
    return cx * 2.0, cy * 2.0


def _resize_mask_batch(mask, width, height):
    mask = mask.detach().cpu().float()
    if mask.dim() == 4:
        mask = mask[..., 0] if mask.shape[-1] == 1 else mask[..., :3].mean(dim=-1)
    if mask.dim() != 3:
        raise ValueError(f"MASK batch must be [B,H,W] or [B,H,W,C], got {tuple(mask.shape)}")
    if int(mask.shape[1]) == int(height) and int(mask.shape[2]) == int(width):
        return mask.unsqueeze(-1).contiguous().clamp(0.0, 1.0)
    return torch.nn.functional.interpolate(
        mask.unsqueeze(1),
        size=(int(height), int(width)),
        mode="nearest",
    ).permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0)


def _resize_depth_batch(depth_maps, width, height):
    depth = depth_maps.detach().cpu().float()
    if depth.dim() == 4:
        depth = depth[..., 0] if depth.shape[-1] == 1 else depth[..., :3].mean(dim=-1)
    if depth.dim() != 3:
        raise ValueError(f"depth batch must be [B,H,W] or [B,H,W,C], got {tuple(depth.shape)}")
    if int(depth.shape[1]) == int(height) and int(depth.shape[2]) == int(width):
        return depth.contiguous()
    return torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=(int(height), int(width)),
        mode="bilinear",
        align_corners=False,
    )[:, 0].contiguous()


def _image_stats(label, images):
    t = images.detach().cpu().float()
    if t.numel() == 0:
        print(f"[SplatPatchAssemble] {label}: empty")
        return
    print(
        f"[SplatPatchAssemble] {label}: shape={tuple(t.shape)}, "
        f"min={float(t.min().item()):.4f}, mean={float(t.mean().item()):.4f}, max={float(t.max().item()):.4f}"
    )
    if float(t.max().item()) < 0.02:
        print(f"⚠️ [SplatPatchAssemble] {label} is almost fully black before WorldMirror.")


def _source_observation_vectors(means, source_poses):
    if source_poses is None or source_poses.shape[0] == 0:
        return None
    means = means.detach().cpu().float()
    poses = source_poses.detach().cpu().float()
    cam_pos = poses[:, :3, 3]
    forward = poses[:, :3, 2]

    best_score = torch.full((means.shape[0],), -1e9, dtype=torch.float32)
    best_vec = torch.zeros_like(means)
    chunk = 250_000
    for start in range(0, means.shape[0], chunk):
        end = min(start + chunk, means.shape[0])
        pts = means[start:end]
        to_point = pts[:, None, :] - cam_pos[None, :, :]
        dist = torch.linalg.norm(to_point, dim=-1).clamp_min(1e-8)
        ray = to_point / dist[..., None]
        score = (ray * forward[None, :, :]).sum(dim=-1)
        src_id = torch.argmax(score, dim=1)
        row = torch.arange(end - start)
        best_score[start:end] = score[row, src_id]
        best_vec[start:end] = -ray[row, src_id]
    best_vec = best_vec / best_vec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    best_vec[best_score <= 0.0] = 0.0
    return best_vec


def _keep_large_components(mask, min_area_ratio):
    mask_np = (mask.detach().cpu().numpy() > 0.5).astype(np.uint8)
    if min_area_ratio <= 0.0 or mask_np.max() == 0:
        return torch.from_numpy(mask_np.astype(np.float32))
    min_area = max(1, int(mask_np.size * float(min_area_ratio)))
    if cv2 is not None:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
        kept = np.zeros_like(mask_np)
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
                kept[labels == label] = 1
        return torch.from_numpy(kept.astype(np.float32))

    # Lightweight fallback without scipy: keep the mask unchanged if cv2 is absent.
    return torch.from_numpy(mask_np.astype(np.float32))


def _dark_region_mask(image, alpha, threshold, max_alpha, min_area_ratio, dilate):
    img = image.detach().cpu().float()[..., :3].clamp(0.0, 1.0)
    luminance = img.mean(dim=-1)
    mask = luminance < float(threshold)
    if max_alpha < 1.0:
        mask = mask & (alpha.detach().cpu().float() <= float(max_alpha))
    mask = _keep_large_components(mask.float(), min_area_ratio)
    if dilate > 0 and cv2 is not None:
        mask_np = (mask.numpy() > 0.5).astype(np.uint8)
        kernel = np.ones((int(dilate), int(dilate)), dtype=np.uint8)
        mask = torch.from_numpy(cv2.dilate(mask_np, kernel, iterations=1).astype(np.float32))
    return mask.float().clamp(0.0, 1.0)


def _scene_motion_params(splats, offset_mode, side_offset, forward_offset, bounds_margin_fraction):
    means = splats["means"].detach().cpu().float()
    if means.shape[0] > 1000000:
        idx = torch.linspace(0, means.shape[0] - 1, 1000000).long()
        means = means[idx]
    if means.numel() == 0:
        low = torch.full((3,), -1.0, dtype=torch.float32)
        high = torch.full((3,), 1.0, dtype=torch.float32)
    else:
        low = torch.quantile(means, 0.02, dim=0)
        high = torch.quantile(means, 0.98, dim=0)
    size = (high - low).clamp_min(1e-6)
    horizontal_span = float(torch.maximum(size[0], size[2]).item())
    if not math.isfinite(horizontal_span) or horizontal_span <= 1e-6:
        horizontal_span = 1.0

    if offset_mode == "scene_fraction":
        side_world = float(side_offset) * horizontal_span
        forward_world = float(forward_offset) * horizontal_span
    else:
        side_world = float(side_offset)
        forward_world = float(forward_offset)

    margin = size * float(bounds_margin_fraction)
    clamp_low = low + margin
    clamp_high = high - margin
    bad_dims = clamp_low >= clamp_high
    if bool(bad_dims.any()):
        center = (low + high) * 0.5
        clamp_low = torch.where(bad_dims, center, clamp_low)
        clamp_high = torch.where(bad_dims, center, clamp_high)

    return {
        "low": low,
        "high": high,
        "clamp_low": clamp_low,
        "clamp_high": clamp_high,
        "horizontal_span": horizontal_span,
        "side_world": side_world,
        "forward_world": forward_world,
    }


def _clamp_position_to_scene(position, motion, clamp_to_splat_bounds):
    if not clamp_to_splat_bounds:
        return position, False
    clamped = torch.maximum(torch.minimum(position, motion["clamp_high"]), motion["clamp_low"])
    changed = bool(torch.max(torch.abs(clamped - position)).item() > 1e-6)
    return clamped, changed


def _estimate_scene_origin(source_poses, splats):
    source_poses = _normalize_pose_tensor(source_poses)
    if source_poses is not None and source_poses.shape[0] > 0:
        return torch.median(source_poses[:, :3, 3], dim=0).values.float()
    means = splats["means"].detach().cpu().float()
    if means.shape[0] > 1000000:
        idx = torch.linspace(0, means.shape[0] - 1, 1000000).long()
        means = means[idx]
    return torch.median(means, dim=0).values.float()


def _make_yaw_pitch_pose(position, yaw, pitch=0.0):
    cp = math.cos(float(pitch))
    forward = torch.tensor(
        [math.sin(float(yaw)) * cp, -math.sin(float(pitch)), math.cos(float(yaw)) * cp],
        dtype=torch.float32,
    )
    forward = forward / forward.norm().clamp_min(1e-8)
    up_hint = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32)
    right = torch.linalg.cross(forward, up_hint, dim=0)
    if right.norm() < 1e-6:
        up_hint = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        right = torch.linalg.cross(forward, up_hint, dim=0)
    right = right / right.norm().clamp_min(1e-8)
    up = torch.linalg.cross(right, forward, dim=0)

    pose = torch.eye(4, dtype=torch.float32)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = forward
    pose[:3, 3] = position.float()
    return pose


def _flip_pose_image_up(pose):
    pose = pose.clone()
    pose[:3, 1] *= -1.0
    return pose


def _generate_triage_candidate_poses(
    source_poses,
    splats,
    scan_yaw_count,
    scan_pitch_deg,
    camera_radius_fraction,
    clamp_to_splat_bounds,
    bounds_margin_fraction,
):
    motion = _scene_motion_params(splats, "scene_fraction", 0.0, 0.0, bounds_margin_fraction)
    origin = _estimate_scene_origin(source_poses, splats)
    radius = float(camera_radius_fraction) * motion["horizontal_span"]
    pitch_values = [0.0]
    scan_pitch = math.radians(float(scan_pitch_deg))
    if abs(scan_pitch) > 1e-5:
        pitch_values.extend([scan_pitch, -scan_pitch])

    poses = []
    labels = []
    total_yaw = max(4, int(scan_yaw_count))
    for i in range(total_yaw):
        yaw = 2.0 * math.pi * i / total_yaw
        forward = torch.tensor([math.sin(yaw), 0.0, math.cos(yaw)], dtype=torch.float32)
        right = torch.tensor([math.cos(yaw), 0.0, -math.sin(yaw)], dtype=torch.float32)
        offsets = (
            torch.zeros(3, dtype=torch.float32),
            right * radius,
            -right * radius,
            forward * radius,
        )
        for offset_i, offset in enumerate(offsets):
            position, clamped = _clamp_position_to_scene(origin + offset, motion, clamp_to_splat_bounds)
            for pitch in pitch_values:
                poses.append(_flip_pose_image_up(_make_yaw_pitch_pose(position, yaw, pitch)))
                labels.append((math.degrees(yaw), math.degrees(pitch), offset_i, clamped))

    print(
        "[SplatPatchPOC2] triage scan candidates: "
        f"origin=({float(origin[0]):.4f},{float(origin[1]):.4f},{float(origin[2]):.4f}), "
        f"yaw={total_yaw}, pitch_variants={len(pitch_values)}, offsets=4, "
        f"radius={radius:.4f}, total={len(poses)}"
    )
    return torch.stack(poses, dim=0), labels


def _bad_geometry_mask(alpha, coverage_threshold, speckle_kernel):
    alpha = alpha.detach().cpu().float()
    missing = alpha < float(coverage_threshold)
    if int(speckle_kernel) <= 1:
        return missing.float()
    pad = int(speckle_kernel) // 2
    a = alpha.unsqueeze(0).unsqueeze(0)
    local = torch.nn.functional.avg_pool2d(a, int(speckle_kernel), stride=1, padding=pad)[0, 0]
    speckle = (alpha > float(coverage_threshold)) & (local < max(float(coverage_threshold) * 2.0, 0.18))
    return torch.maximum(missing.float(), speckle.float()).clamp(0.0, 1.0)


def _central_weight(height, width, inner_fraction):
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, int(height)),
        torch.linspace(-1.0, 1.0, int(width)),
        indexing="ij",
    )
    rr = torch.sqrt(xx * xx + yy * yy)
    inner = max(float(inner_fraction), 1e-3)
    return (1.0 - (rr / inner).clamp(0.0, 1.0)).clamp(0.0, 1.0)


def _bad_geometry_components(bad, weight, speckle_kernel, min_area_ratio):
    bad = bad.detach().cpu().float().clamp(0.0, 1.0)
    height, width = int(bad.shape[0]), int(bad.shape[1])
    mask_np = (bad.numpy() > 0.5).astype(np.uint8)
    if mask_np.max() == 0:
        return []

    if cv2 is not None:
        kernel_size = max(3, int(speckle_kernel))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask_np = cv2.dilate(mask_np, kernel, iterations=1)

    min_area = max(16, int(mask_np.size * float(min_area_ratio) * 0.35))
    if cv2 is None:
        ys, xs = np.where(mask_np > 0)
        if ys.size < min_area:
            return []
        components = [(0, int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1), int(ys.size))]
        labels = mask_np
    else:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
        components = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= min_area:
                components.append(
                    (
                        label,
                        int(stats[label, cv2.CC_STAT_LEFT]),
                        int(stats[label, cv2.CC_STAT_TOP]),
                        int(stats[label, cv2.CC_STAT_WIDTH]),
                        int(stats[label, cv2.CC_STAT_HEIGHT]),
                        area,
                    )
                )

    results = []
    weight_sum = float(weight.sum().clamp_min(1e-6).item())
    for label, x, y, w, h, area in components:
        if cv2 is None:
            comp_mask = torch.from_numpy(labels > 0).float()
        else:
            comp_mask = torch.from_numpy(labels == label).float()
        raw_bad = bad * comp_mask
        raw_bad_ratio = float(raw_bad.mean().item())
        if raw_bad_ratio <= 0.0:
            continue
        central_bad = float((raw_bad * weight).sum().item() / weight_sum)
        ys, xs = torch.where(comp_mask > 0.5)
        if xs.numel() == 0:
            continue
        bad_values = raw_bad[ys, xs].clamp_min(1e-4)
        cx = float((xs.float() * bad_values).sum().item() / bad_values.sum().item())
        cy = float((ys.float() * bad_values).sum().item() / bad_values.sum().item())
        edge_margin = min(x, y, max(width - (x + w), 0), max(height - (y + h), 0)) / max(min(width, height), 1)
        edge_penalty = max(0.0, 0.08 - float(edge_margin)) / 0.08
        center_dx = (cx / max(width - 1, 1)) * 2.0 - 1.0
        center_dy = (cy / max(height - 1, 1)) * 2.0 - 1.0
        center_distance = math.sqrt(center_dx * center_dx + center_dy * center_dy)
        bbox_fraction = float((w * h) / max(width * height, 1))
        results.append(
            {
                "cx": cx,
                "cy": cy,
                "bbox": (x, y, w, h),
                "area_ratio": float(area / max(width * height, 1)),
                "raw_bad_ratio": raw_bad_ratio,
                "central_bad": central_bad,
                "bbox_fraction": bbox_fraction,
                "edge_penalty": edge_penalty,
                "center_distance": center_distance,
            }
        )
    return results


def _component_world_direction(pose, K, cx, cy):
    pose = pose.detach().cpu().float()
    K = K.detach().cpu().float()
    x = (float(cx) - float(K[0, 2])) / max(float(K[0, 0]), 1e-6)
    y = (float(cy) - float(K[1, 2])) / max(float(K[1, 1]), 1e-6)
    camera_dir = torch.tensor([x, y, 1.0], dtype=torch.float32)
    world_dir = pose[:3, :3] @ camera_dir
    return world_dir / world_dir.norm().clamp_min(1e-8)


def _project_points_to_view(points, pose, K, width, height):
    w2c = torch.linalg.inv(pose.detach().cpu().float())
    pts_h = torch.cat([points, torch.ones(points.shape[0], 1, dtype=torch.float32)], dim=1)
    cam = (w2c @ pts_h.T).T[:, :3]
    z = cam[:, 2]
    valid = z > 1e-5
    u = torch.round(cam[:, 0] * K[0, 0] / z.clamp_min(1e-5) + K[0, 2]).long()
    v = torch.round(cam[:, 1] * K[1, 1] / z.clamp_min(1e-5) + K[1, 2]).long()
    valid = valid & (u >= 0) & (u < int(width)) & (v >= 0) & (v < int(height))
    return u, v, z, valid


def _render_depth_from_splats(splats, pose, K, width, height, max_points=0):
    means = splats["means"].detach().cpu().float()
    if max_points > 0 and means.shape[0] > int(max_points):
        idx = torch.linspace(0, means.shape[0] - 1, int(max_points)).long()
        means = means[idx]
    depth = np.full((int(height), int(width)), np.inf, dtype=np.float32)
    chunk = 250_000
    for start in range(0, means.shape[0], chunk):
        pts = means[start:start + chunk]
        u, v, z, valid = _project_points_to_view(pts, pose, K, width, height)
        if not bool(valid.any()):
            continue
        flat = v[valid].numpy() * int(width) + u[valid].numpy()
        np.minimum.at(depth.reshape(-1), flat, z[valid].numpy().astype(np.float32))
    return torch.from_numpy(depth)


def _dilate_mask(mask, dilate):
    mask = mask.detach().cpu().float().clamp(0.0, 1.0)
    if int(dilate) <= 0 or cv2 is None:
        return mask
    kernel = np.ones((int(dilate), int(dilate)), dtype=np.uint8)
    mask_np = (mask.numpy() > 0.5).astype(np.uint8)
    return torch.from_numpy(cv2.dilate(mask_np, kernel, iterations=1).astype(np.float32))


def _mask_debug_stats(mask, threshold):
    mask = mask.detach().cpu().float()
    binary = mask >= float(threshold)
    pixels = int(binary.numel())
    count = int(binary.sum().item())
    if count <= 0:
        return {
            "pixels": pixels,
            "count": 0,
            "ratio": 0.0,
            "bbox": None,
            "min": float(mask.min().item()) if mask.numel() else 0.0,
            "mean": float(mask.mean().item()) if mask.numel() else 0.0,
            "max": float(mask.max().item()) if mask.numel() else 0.0,
        }
    ys, xs = torch.where(binary)
    bbox = (
        int(xs.min().item()),
        int(ys.min().item()),
        int(xs.max().item() - xs.min().item() + 1),
        int(ys.max().item() - ys.min().item() + 1),
    )
    return {
        "pixels": pixels,
        "count": count,
        "ratio": count / max(pixels, 1),
        "bbox": bbox,
        "min": float(mask.min().item()),
        "mean": float(mask.mean().item()),
        "max": float(mask.max().item()),
    }


def _depth_debug_stats(depth, mask=None, threshold=0.5):
    depth = depth.detach().cpu().float()
    finite = torch.isfinite(depth) & (depth > 1e-5)
    if mask is not None:
        in_region = mask.detach().cpu().float() >= float(threshold)
        finite_region = finite & in_region
        denom = int(in_region.sum().item())
    else:
        in_region = torch.ones_like(finite)
        finite_region = finite
        denom = int(finite.numel())
    vals = depth[finite_region]
    return {
        "finite_total": int(finite.sum().item()),
        "finite_region": int(finite_region.sum().item()),
        "region_pixels": denom,
        "region_ratio": int(finite_region.sum().item()) / max(denom, 1),
        "min": float(vals.min().item()) if vals.numel() else None,
        "mean": float(vals.mean().item()) if vals.numel() else None,
        "max": float(vals.max().item()) if vals.numel() else None,
    }


def _point_bounds_debug(points):
    if not isinstance(points, torch.Tensor) or points.numel() == 0:
        return None
    pts = points.detach().cpu().float()
    low = pts.min(dim=0).values
    high = pts.max(dim=0).values
    return (
        (float(low[0].item()), float(low[1].item()), float(low[2].item())),
        (float(high[0].item()), float(high[1].item()), float(high[2].item())),
    )


def _fmt_float(value, precision=4):
    if value is None:
        return "n/a"
    return f"{float(value):.{precision}f}"


def _remove_splats_projecting_to_mask(splats, pose, K, mask, mask_threshold, depth, depth_tolerance):
    means = splats["means"].detach().cpu().float()
    keep = torch.ones(means.shape[0], dtype=torch.bool)
    height, width = int(mask.shape[0]), int(mask.shape[1])
    chunk = 250_000
    stats = {
        "input": int(means.shape[0]),
        "projected": 0,
        "mask_hits": 0,
        "depth_checked": 0,
        "depth_near_hits": 0,
    }
    for start in range(0, means.shape[0], chunk):
        end = min(start + chunk, means.shape[0])
        pts = means[start:end]
        u, v, z, valid = _project_points_to_view(pts, pose, K, width, height)
        if not bool(valid.any()):
            continue
        selected = torch.where(valid)[0]
        stats["projected"] += int(selected.numel())
        in_mask = mask[v[selected], u[selected]] >= float(mask_threshold)
        stats["mask_hits"] += int(in_mask.sum().item())
        if depth is not None and float(depth_tolerance) > 0.0:
            depth_at = depth[v[selected], u[selected]]
            finite = torch.isfinite(depth_at)
            near_depth = finite & ((z[selected] - depth_at).abs() <= float(depth_tolerance))
            stats["depth_checked"] += int(finite.sum().item())
            stats["depth_near_hits"] += int((in_mask & near_depth).sum().item())
            in_mask = in_mask & near_depth
        remove_ids = selected[in_mask] + start
        keep[remove_ids] = False
    removed = int((~keep).sum().item())
    stats["removed"] = removed
    out = {}
    for key, value in splats.items():
        if isinstance(value, torch.Tensor):
            if value.dim() >= 1 and value.shape[0] == means.shape[0]:
                out[key] = value[keep].contiguous()
            elif value.dim() >= 3 and value.shape[0] == 1 and value.shape[1] == means.shape[0]:
                out[key] = value[:, keep].contiguous()
            else:
                out[key] = value
        else:
            out[key] = value
    return out, removed, stats


def _remove_splats_behind_patch_surface(
    splats,
    pose,
    K,
    mask,
    mask_threshold,
    patch_depth,
    front_margin,
    far_margin,
):
    means = splats["means"].detach().cpu().float()
    keep = torch.ones(means.shape[0], dtype=torch.bool)
    height, width = int(mask.shape[0]), int(mask.shape[1])
    patch_depth = patch_depth.detach().cpu().float()
    region = mask >= float(mask_threshold)
    chunk = 250_000
    stats = {
        "input": int(means.shape[0]),
        "projected": 0,
        "mask_hits": 0,
        "surface_checked": 0,
        "behind_hits": 0,
    }
    for start in range(0, means.shape[0], chunk):
        end = min(start + chunk, means.shape[0])
        pts = means[start:end]
        u, v, z, valid = _project_points_to_view(pts, pose, K, width, height)
        if not bool(valid.any()):
            continue
        selected = torch.where(valid)[0]
        stats["projected"] += int(selected.numel())
        in_region = region[v[selected], u[selected]]
        stats["mask_hits"] += int(in_region.sum().item())
        depth_at = patch_depth[v[selected], u[selected]]
        finite_surface = torch.isfinite(depth_at) & (depth_at > 1e-5)
        checked = in_region & finite_surface
        stats["surface_checked"] += int(checked.sum().item())
        behind = z[selected] >= (depth_at - float(front_margin))
        if float(far_margin) > 0.0:
            behind = behind & (z[selected] <= (depth_at + float(far_margin)))
        remove = checked & behind
        stats["behind_hits"] += int(remove.sum().item())
        remove_ids = selected[remove] + start
        keep[remove_ids] = False
    removed = int((~keep).sum().item())
    stats["removed"] = removed
    out = {}
    for key, value in splats.items():
        if isinstance(value, torch.Tensor):
            if value.dim() >= 1 and value.shape[0] == means.shape[0]:
                out[key] = value[keep].contiguous()
            elif value.dim() >= 3 and value.shape[0] == 1 and value.shape[1] == means.shape[0]:
                out[key] = value[:, keep].contiguous()
            else:
                out[key] = value
        else:
            out[key] = value
    return out, removed, stats


def _complete_depth_in_region(depth, mask, mask_threshold, iterations):
    depth = depth.detach().cpu().float()
    region = mask.detach().cpu().float() >= float(mask_threshold)
    valid = torch.isfinite(depth) & (depth > 1e-5) & region
    result = depth.clone()
    stats = {
        "input_valid": int(valid.sum().item()),
        "region_pixels": int(region.sum().item()),
        "filled": 0,
        "iterations": 0,
    }
    if int(iterations) <= 0 or int(valid.sum().item()) == 0:
        return result, stats

    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32)
    values = torch.where(valid, result, torch.zeros_like(result))[None, None]
    weights = valid.float()[None, None]
    fillable = region & ~valid
    for idx in range(int(iterations)):
        sums = torch.nn.functional.conv2d(values, kernel, padding=1)
        counts = torch.nn.functional.conv2d(weights, kernel, padding=1)
        can_fill = fillable & (counts[0, 0] > 0.0)
        if not bool(can_fill.any()):
            break
        result[can_fill] = sums[0, 0][can_fill] / counts[0, 0][can_fill].clamp_min(1e-6)
        valid = valid | can_fill
        values = torch.where(valid, result, torch.zeros_like(result))[None, None]
        weights = valid.float()[None, None]
        fillable = region & ~valid
        stats["iterations"] = idx + 1
    stats["filled"] = int((valid & region).sum().item()) - stats["input_valid"]
    return result, stats


def _append_patch_splats(splats, points, colors, scale, opacity):
    if points.numel() == 0:
        return splats
    n = points.shape[0]
    device = splats["means"].device if isinstance(splats.get("means"), torch.Tensor) else torch.device("cpu")
    points = points.to(device).float()
    colors = colors.to(device).float().clamp(0.0, 1.0)
    scales = torch.full((n, 3), float(scale), dtype=torch.float32, device=device)
    quats = torch.zeros((n, 4), dtype=torch.float32, device=device)
    quats[:, 0] = 1.0
    opacities = torch.full((n,), float(opacity), dtype=torch.float32, device=device)
    sh = ((colors - 0.5) / SH_C0)[:, None, :]

    out = dict(splats)
    out["means"] = torch.cat([splats["means"].to(device).float(), points], dim=0).contiguous()
    out["scales"] = torch.cat([splats["scales"].to(device).float(), scales], dim=0).contiguous()
    out["quats"] = torch.cat([splats["quats"].to(device).float(), quats], dim=0).contiguous()
    out["opacities"] = torch.cat([splats["opacities"].to(device).float(), opacities], dim=0).contiguous()
    out["sh"] = torch.cat([splats["sh"].to(device).float(), sh], dim=0).contiguous()
    out["rgb"] = torch.cat([splats["rgb"].to(device).float(), colors], dim=0).contiguous()
    return out


def _backproject_patch_pixels(image, mask, depth, pose, K, mask_threshold, max_points, pixel_stride, min_luma=0.0):
    height, width = int(image.shape[0]), int(image.shape[1])
    mask_valid = mask >= float(mask_threshold)
    depth_valid = torch.isfinite(depth) & (depth > 1e-5)
    colors_all = image[..., :3].detach().cpu().float().clamp(0.0, 1.0)
    luma = colors_all.mean(dim=-1)
    color_valid = luma >= float(min_luma)
    valid = mask_valid & depth_valid & color_valid
    valid_before_stride = int(valid.sum().item())
    if int(pixel_stride) > 1:
        stride_mask = torch.zeros_like(valid)
        stride_mask[:: int(pixel_stride), :: int(pixel_stride)] = True
        valid = valid & stride_mask
    ys, xs = torch.where(valid)
    stats = {
        "mask_pixels": int(mask_valid.sum().item()),
        "depth_valid_pixels": int(depth_valid.sum().item()),
        "color_valid_pixels": int(color_valid.sum().item()),
        "min_luma": float(min_luma),
        "valid_before_stride": valid_before_stride,
        "valid_after_stride": int(xs.numel()),
        "capped_from": None,
        "selected": 0,
    }
    if xs.numel() == 0:
        return torch.empty(0, 3), torch.empty(0, 3), stats
    if int(max_points) > 0 and xs.numel() > int(max_points):
        stats["capped_from"] = int(xs.numel())
        take = torch.linspace(0, xs.numel() - 1, int(max_points)).long()
        xs = xs[take]
        ys = ys[take]
    stats["selected"] = int(xs.numel())
    z = depth[ys, xs].float()
    x = (xs.float() - float(K[0, 2])) * z / max(float(K[0, 0]), 1e-6)
    y = (ys.float() - float(K[1, 2])) * z / max(float(K[1, 1]), 1e-6)
    cam = torch.stack([x, y, z, torch.ones_like(z)], dim=1)
    world = (pose.detach().cpu().float() @ cam.T).T[:, :3]
    colors = colors_all[ys, xs]
    stats["depth_min"] = float(z.min().item()) if z.numel() else None
    stats["depth_mean"] = float(z.mean().item()) if z.numel() else None
    stats["depth_max"] = float(z.max().item()) if z.numel() else None
    stats["world_bounds"] = _point_bounds_debug(world)
    stats["color_min"] = float(colors.min().item()) if colors.numel() else None
    stats["color_mean"] = float(colors.mean().item()) if colors.numel() else None
    stats["color_max"] = float(colors.max().item()) if colors.numel() else None
    return world, colors, stats


def _generate_repair_poses(
    source_poses,
    splats,
    view_count,
    trajectory_preset,
    side_offset,
    forward_offset,
    pattern,
    dual_side_offsets,
    offset_mode,
    clamp_to_splat_bounds,
    bounds_margin_fraction,
):
    motion = _scene_motion_params(splats, offset_mode, side_offset, forward_offset, bounds_margin_fraction)
    print(
        "[SplatPatchPOC] repair offset units: "
        f"mode={offset_mode}, side_offset={float(side_offset):.4f} -> {motion['side_world']:.4f} world, "
        f"forward_offset={float(forward_offset):.4f} -> {motion['forward_world']:.4f} world, "
        f"horizontal_span={motion['horizontal_span']:.4f}, clamp_to_splat_bounds={bool(clamp_to_splat_bounds)}"
    )
    print(
        "[SplatPatchPOC] robust splat bounds: "
        f"x=[{float(motion['low'][0]):.3f}, {float(motion['high'][0]):.3f}], "
        f"y=[{float(motion['low'][1]):.3f}, {float(motion['high'][1]):.3f}], "
        f"z=[{float(motion['low'][2]):.3f}, {float(motion['high'][2]):.3f}]"
    )

    def resample_pose_list(poses):
        if not poses:
            return None
        target = max(1, int(view_count))
        if len(poses) == target:
            return torch.stack(poses, dim=0)
        ids = torch.linspace(0, len(poses) - 1, target).round().long().tolist()
        return torch.stack([poses[int(i)].clone() for i in ids], dim=0)

    preset = str(trajectory_preset or "custom")
    if preset != "custom":
        low = motion["clamp_low"] if bool(clamp_to_splat_bounds) else motion["low"]
        high = motion["clamp_high"] if bool(clamp_to_splat_bounds) else motion["high"]
        center = (motion["low"] + motion["high"]) * 0.5
        size = (motion["high"] - motion["low"]).clamp_min(1e-6)
        span_x = float(size[0].item())
        span_z = float(size[2].item())
        radius = max(span_x, span_z) * 0.48
        if not math.isfinite(radius) or radius <= 1e-6:
            radius = motion["horizontal_span"] * 0.35
        y_mid = center[1]
        y_levels = [low[1], high[1]]
        x_levels = [low[0], high[0]]
        z_levels = [low[2], high[2]]
        poses = []

        def preset_look_at(position, target):
            return _flip_pose_image_up(_make_look_at_pose(position, target))

        if preset == "orbit_sideways":
            total = max(1, int(view_count))
            for i in range(total):
                yaw = 2.0 * math.pi * i / max(total, 1)
                radial = torch.tensor([math.cos(yaw), 0.0, math.sin(yaw)], dtype=torch.float32)
                tangent = torch.tensor([-math.sin(yaw), 0.0, math.cos(yaw)], dtype=torch.float32)
                pos = center + radial * float(radius)
                pos[1] = y_mid
                pos, _ = _clamp_position_to_scene(pos, motion, clamp_to_splat_bounds)
                poses.append(preset_look_at(pos, pos + tangent))
        elif preset == "orbit_center":
            total = max(1, int(view_count))
            for i in range(total):
                yaw = 2.0 * math.pi * i / max(total, 1)
                pos = center + torch.tensor(
                    [math.cos(yaw) * radius, 0.0, math.sin(yaw) * radius],
                    dtype=torch.float32,
                )
                pos[1] = y_mid
                pos, _ = _clamp_position_to_scene(pos, motion, clamp_to_splat_bounds)
                poses.append(preset_look_at(pos, center))
        elif preset == "room_corners_upper_lower":
            for y in y_levels:
                for x in x_levels:
                    for z in z_levels:
                        pos = torch.tensor([float(x), float(y), float(z)], dtype=torch.float32)
                        pos, _ = _clamp_position_to_scene(pos, motion, clamp_to_splat_bounds)
                        poses.append(preset_look_at(pos, center))
        elif preset == "wall_midpoints":
            wall_positions = (
                torch.tensor([low[0], y_mid, center[2]], dtype=torch.float32),
                torch.tensor([high[0], y_mid, center[2]], dtype=torch.float32),
                torch.tensor([center[0], y_mid, low[2]], dtype=torch.float32),
                torch.tensor([center[0], y_mid, high[2]], dtype=torch.float32),
            )
            wall_targets = (
                torch.tensor([high[0], y_mid, center[2]], dtype=torch.float32),
                torch.tensor([low[0], y_mid, center[2]], dtype=torch.float32),
                torch.tensor([center[0], y_mid, high[2]], dtype=torch.float32),
                torch.tensor([center[0], y_mid, low[2]], dtype=torch.float32),
            )
            for pos, target in zip(wall_positions, wall_targets):
                pos, _ = _clamp_position_to_scene(pos, motion, clamp_to_splat_bounds)
                poses.append(preset_look_at(pos, target))
                poses.append(preset_look_at(pos, center))
        elif preset == "floor_ceiling_orbit":
            total = max(2, int(view_count))
            per_level = max(1, math.ceil(total / 2))
            for y in y_levels:
                for i in range(per_level):
                    yaw = 2.0 * math.pi * i / max(per_level, 1)
                    pos = center + torch.tensor(
                        [math.cos(yaw) * radius, 0.0, math.sin(yaw) * radius],
                        dtype=torch.float32,
                    )
                    pos[1] = y
                    pos, _ = _clamp_position_to_scene(pos, motion, clamp_to_splat_bounds)
                    target = center.clone()
                    target[1] = y_mid
                    poses.append(preset_look_at(pos, target))
        elif preset == "diagonal_cross":
            corner_pairs = (
                ((low[0], y_mid, low[2]), (high[0], y_mid, high[2])),
                ((high[0], y_mid, high[2]), (low[0], y_mid, low[2])),
                ((low[0], y_mid, high[2]), (high[0], y_mid, low[2])),
                ((high[0], y_mid, low[2]), (low[0], y_mid, high[2])),
            )
            for pos_vals, target_vals in corner_pairs:
                pos = torch.tensor([float(v) for v in pos_vals], dtype=torch.float32)
                target = torch.tensor([float(v) for v in target_vals], dtype=torch.float32)
                pos, _ = _clamp_position_to_scene(pos, motion, clamp_to_splat_bounds)
                poses.append(preset_look_at(pos, target))
                poses.append(preset_look_at(pos, center))
        else:
            print(f"[SplatPatchPOC] unknown trajectory_preset={preset}; falling back to custom.")
            preset = "custom"

        if preset != "custom":
            out = resample_pose_list(poses)
            print(
                "[SplatPatchPOC] trajectory preset generated repair cameras: "
                f"preset={preset}, base_poses={len(poses)}, output={int(out.shape[0])}, "
                f"center=({float(center[0]):.4f},{float(center[1]):.4f},{float(center[2]):.4f}), "
                f"radius={float(radius):.4f}"
            )
            for i in range(min(8, int(out.shape[0]))):
                pos = out[i, :3, 3]
                forward = out[i, :3, 2]
                print(
                    f"[SplatPatchPOC] preset pose {i:02d}: "
                    f"pos=({float(pos[0]): .4f},{float(pos[1]): .4f},{float(pos[2]): .4f}), "
                    f"forward=({float(forward[0]): .4f},{float(forward[1]): .4f},{float(forward[2]): .4f})"
                )
            return out

    if source_poses is not None and source_poses.shape[0] > 0:
        total = min(int(view_count), int(source_poses.shape[0]))
        ids = torch.linspace(0, source_poses.shape[0] - 1, total).round().long().unique()
        poses = []
        clamped_count = 0
        if dual_side_offsets:
            print(
                "[SplatPatchPOC] dual_side_offsets enabled: each selected source view emits "
                "four repair cameras: +forward/+side, +forward/-side, -forward/+side, -forward/-side."
            )
        for src_order, src_i in enumerate(ids.tolist()):
            side_signs = (1.0, -1.0) if dual_side_offsets else (-1.0 if src_order % 2 else 1.0,)
            forward_signs = (1.0, -1.0) if dual_side_offsets else (1.0,)
            for forward_sign in forward_signs:
                for side_sign in side_signs:
                    out_i = len(poses)
                    base = source_poses[src_i].clone()
                    right = base[:3, 0]
                    forward = base[:3, 2]
                    translation = (
                        right * motion["side_world"] * float(side_sign)
                        + forward * motion["forward_world"] * float(forward_sign)
                    )
                    if not dual_side_offsets and pattern == "left_right_forward":
                        mode = src_order % 3
                        if mode == 1:
                            translation = right * -motion["side_world"] + forward * motion["forward_world"]
                        elif mode == 2:
                            translation = forward * (motion["forward_world"] + abs(motion["side_world"]) * 0.5)
                    unclamped_pos = base[:3, 3] + translation
                    base[:3, 3], clamped = _clamp_position_to_scene(unclamped_pos, motion, clamp_to_splat_bounds)
                    clamped_count += int(clamped)
                    if out_i < 12:
                        print(
                            f"[SplatPatchPOC] repair pose {out_i:02d} from source {src_i}: "
                            f"forward_sign={float(forward_sign): .0f}, side_sign={float(side_sign): .0f}, "
                            f"move=({float(translation[0]): .4f},{float(translation[1]): .4f},{float(translation[2]): .4f}), "
                            f"pos=({float(base[0, 3]): .4f},{float(base[1, 3]): .4f},{float(base[2, 3]): .4f}), "
                            f"clamped={clamped}"
                        )
                    poses.append(base)
        if clamped_count:
            print(f"[SplatPatchPOC] clamped {clamped_count}/{len(poses)} repair camera positions to splat bounds.")
        return torch.stack(poses, dim=0)

    means = splats["means"].detach().cpu().float()
    center = means.mean(dim=0)
    radius = torch.quantile((means - center).norm(dim=1), 0.75).clamp_min(0.25)
    poses = []
    total = int(view_count)
    for i in range(total):
        yaw = 2.0 * math.pi * i / max(total, 1)
        pos = center + torch.tensor(
            [math.cos(yaw) * radius * 0.35, 0.0, math.sin(yaw) * radius * 0.35],
            dtype=torch.float32,
        )
        pos, _ = _clamp_position_to_scene(pos, motion, clamp_to_splat_bounds)
        poses.append(_make_look_at_pose(pos, center))
    return torch.stack(poses, dim=0)


def _inpaint_frame(image, hole_mask, mode, dilate):
    image_np = (image.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    mask_np = (hole_mask.detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
    if dilate > 0 and cv2 is not None:
        kernel = np.ones((int(dilate), int(dilate)), dtype=np.uint8)
        mask_np = cv2.dilate(mask_np, kernel, iterations=1)

    if mode == "none" or mask_np.max() == 0:
        return torch.from_numpy(image_np.astype(np.float32) / 255.0)

    if cv2 is not None and mode in ("telea", "navier_stokes"):
        flag = cv2.INPAINT_TELEA if mode == "telea" else cv2.INPAINT_NS
        filled = cv2.inpaint(image_np, mask_np, 5, flag)
        return torch.from_numpy(filled.astype(np.float32) / 255.0)

    valid = mask_np == 0
    if valid.any():
        fallback = image_np.copy()
        fallback[mask_np > 0] = np.median(image_np[valid], axis=0)
    else:
        fallback = np.full_like(image_np, 127)
    return torch.from_numpy(fallback.astype(np.float32) / 255.0)


def _point_render_fallback(splats, pose, K, width, height, max_points, source_vectors=None):
    means = splats["means"].detach().cpu().float()
    rgb = splats["rgb"].detach().cpu().float()
    opacities = splats["opacities"].detach().cpu().float()
    src_vec = source_vectors.detach().cpu().float() if isinstance(source_vectors, torch.Tensor) else None
    if max_points > 0 and means.shape[0] > max_points:
        idx = torch.linspace(0, means.shape[0] - 1, int(max_points)).long()
        means, rgb, opacities = means[idx], rgb[idx], opacities[idx]
        if src_vec is not None and src_vec.shape[0] >= idx.max().item() + 1:
            src_vec = src_vec[idx]

    w2c = torch.linalg.inv(pose.detach().cpu().float())
    pts_h = torch.cat([means, torch.ones(means.shape[0], 1)], dim=1)
    cam = (w2c @ pts_h.T).T[:, :3]
    z = cam[:, 2]
    valid = z > 1e-5
    x = cam[:, 0]
    y = cam[:, 1]
    u = torch.round(x * K[0, 0] / z.clamp_min(1e-5) + K[0, 2]).long()
    v = torch.round(y * K[1, 1] / z.clamp_min(1e-5) + K[1, 2]).long()
    valid = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)

    image = np.zeros((height, width, 3), dtype=np.float32)
    alpha = np.zeros((height, width), dtype=np.float32)
    support = np.ones((height, width), dtype=np.float32)
    if bool(valid.any()):
        ids = torch.where(valid)[0]
        order = torch.argsort(z[ids], descending=True)
        ids = ids[order]
        uu = u[ids].numpy()
        vv = v[ids].numpy()
        image[vv, uu] = rgb[ids].numpy()
        alpha[vv, uu] = _opacity_to_alpha(opacities[ids]).numpy()
        if src_vec is not None:
            cam_pos = pose.detach().cpu().float()[:3, 3]
            repair_vec = cam_pos[None, :] - means[ids]
            repair_vec = repair_vec / repair_vec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            score = (repair_vec * src_vec[ids]).sum(dim=-1).clamp(-1.0, 1.0)
            unsupported = src_vec[ids].norm(dim=-1) <= 1e-6
            score = torch.where(unsupported, torch.full_like(score, -1.0), score)
            support[vv, uu] = score.numpy()
        if cv2 is not None:
            image = cv2.dilate(image, np.ones((3, 3), np.uint8), iterations=1)
            alpha = cv2.dilate(alpha, np.ones((3, 3), np.uint8), iterations=1)
    return torch.from_numpy(image).float(), torch.from_numpy(alpha).float(), torch.from_numpy(support).float()


def _render_with_gsplat(splats, pose, K, width, height, max_points):
    from gsplat.rendering import rasterization

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    means = splats["means"].to(device)
    scales = splats["scales"].to(device)
    quats = splats["quats"].to(device)
    opacities = _opacity_to_alpha(splats["opacities"]).to(device)
    sh = splats["sh"].to(device)
    rgb = splats["rgb"].to(device)
    if max_points > 0 and means.shape[0] > max_points:
        idx = torch.linspace(0, means.shape[0] - 1, int(max_points), device=device).long()
        means = means[idx]
        scales = scales[idx]
        quats = quats[idx]
        opacities = opacities[idx]
        sh = sh[idx]
        rgb = rgb[idx]

    viewmat = torch.linalg.inv(pose.to(device)).unsqueeze(0)
    K = K.to(device).unsqueeze(0)
    kwargs = dict(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        viewmats=viewmat,
        Ks=K,
        width=int(width),
        height=int(height),
        packed=False,
    )
    attempts = (
        dict(kwargs, colors=rgb),
        dict(kwargs, colors=rgb[:, None, :], sh_degree=0),
        dict(kwargs, colors=sh, sh_degree=0),
        dict(kwargs, shs=sh, sh_degree=0),
    )
    last_error = None
    for attempt in attempts:
        try:
            rendered, alpha, _ = rasterization(**attempt)
            image = rendered[0, ..., :3].detach().clamp(0.0, 1.0).cpu()
            alpha = alpha[0].detach().float().cpu()
            if alpha.dim() == 3:
                alpha = alpha[..., 0]
            if float(image.max().item()) < 0.02 and float(alpha.max().item()) > 0.02:
                raise RuntimeError("gsplat produced alpha but an almost black RGB image")
            return image, alpha.clamp(0.0, 1.0)
        except Exception as exc:
            last_error = exc
    raise last_error


class VNCCS_SplatPatchPOC:
    """Generate synthetic repair frames from shifted views of the first splat."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_data": ("PLY_DATA",),
                "source_images": ("IMAGE",),
            },
            "optional": {
                "source_camera_poses": ("TENSOR",),
                "source_camera_intrinsics": ("TENSOR",),
                "view_count": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
                "width": ("INT", {"default": 1022, "min": 252, "max": 2048, "step": 14}),
                "height": ("INT", {"default": 1022, "min": 252, "max": 2048, "step": 14}),
                "fov": ("FLOAT", {"default": 90.0, "min": 30.0, "max": 130.0, "step": 1.0}),
                "trajectory_preset": (
                    [
                        "custom",
                        "orbit_sideways",
                        "orbit_center",
                        "room_corners_upper_lower",
                        "wall_midpoints",
                        "floor_ceiling_orbit",
                        "diagonal_cross",
                    ],
                    {"default": "custom"},
                ),
                "source_pitch_filter": (["zero_pitch", "all"], {"default": "zero_pitch"}),
                "source_pitch_tolerance_deg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 45.0, "step": 0.5}),
                "offset_mode": (["scene_fraction", "world_units"], {"default": "scene_fraction"}),
                "side_offset": ("FLOAT", {"default": 0.12, "min": -2.0, "max": 2.0, "step": 0.01}),
                "dual_side_offsets": ("BOOLEAN", {"default": False}),
                "forward_offset": ("FLOAT", {"default": 0.03, "min": -2.0, "max": 2.0, "step": 0.01}),
                "pattern": (["alternate_left_right", "left_right_forward"], {"default": "left_right_forward"}),
                "clamp_to_splat_bounds": ("BOOLEAN", {"default": True}),
                "bounds_margin_fraction": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.45, "step": 0.01}),
                "mask_mode": (
                    [
                        "alpha_pixels",
                        "black_pixels",
                        "dark_regions",
                        "dark_regions_with_support_gate",
                        "source_support",
                        "alpha",
                        "alpha_or_source_support",
                        "source_support_or_dark",
                    ],
                    {"default": "alpha_pixels"},
                ),
                "coverage_threshold": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01}),
                "support_angle_deg": ("FLOAT", {"default": 75.0, "min": 0.0, "max": 180.0, "step": 1.0}),
                "min_hole_area_ratio": ("FLOAT", {"default": 0.005, "min": 0.0, "max": 0.5, "step": 0.001}),
                "dark_threshold": ("FLOAT", {"default": 0.045, "min": 0.0, "max": 0.5, "step": 0.005}),
                "dark_max_alpha": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dark_min_area_ratio": ("FLOAT", {"default": 0.0005, "min": 0.0, "max": 0.1, "step": 0.0001}),
                "dark_mask_dilate": ("INT", {"default": 3, "min": 0, "max": 51, "step": 2}),
                "inpaint_mode": (["telea", "navier_stokes", "median", "none"], {"default": "telea"}),
                "mask_dilate": ("INT", {"default": 9, "min": 0, "max": 101, "step": 2}),
                "combine_with_source": ("BOOLEAN", {"default": True}),
                "use_gsplat": ("BOOLEAN", {"default": True}),
                "max_render_points": ("INT", {"default": 1200000, "min": 0, "max": 20000000, "step": 100000}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "TENSOR", "TENSOR", "IMAGE", "TENSOR", "TENSOR", "IMAGE")
    RETURN_NAMES = (
        "repair_images",
        "hole_masks",
        "repair_camera_poses",
        "repair_camera_intrinsics",
        "combined_images",
        "combined_camera_poses",
        "combined_camera_intrinsics",
        "hole_images",
    )
    FUNCTION = "generate"
    CATEGORY = "VNCCS/3D/Experimental"

    def generate(
        self,
        ply_data,
        source_images,
        source_camera_poses=None,
        source_camera_intrinsics=None,
        view_count=8,
        width=1022,
        height=1022,
        fov=90.0,
        trajectory_preset="custom",
        source_pitch_filter="zero_pitch",
        source_pitch_tolerance_deg=5.0,
        offset_mode="scene_fraction",
        side_offset=0.12,
        dual_side_offsets=False,
        forward_offset=0.03,
        pattern="left_right_forward",
        clamp_to_splat_bounds=True,
        bounds_margin_fraction=0.08,
        mask_mode="alpha_pixels",
        coverage_threshold=0.08,
        support_angle_deg=75.0,
        min_hole_area_ratio=0.005,
        dark_threshold=0.045,
        dark_max_alpha=0.55,
        dark_min_area_ratio=0.0005,
        dark_mask_dilate=3,
        inpaint_mode="telea",
        mask_dilate=9,
        combine_with_source=True,
        use_gsplat=True,
        max_render_points=1200000,
    ):
        splats = _extract_splats(ply_data)
        if splats is None:
            raise ValueError("VNCCS_SplatPatchPOC requires Gaussian splats in PLY_DATA.")

        source_poses = _normalize_pose_tensor(source_camera_poses)
        source_intrs = _normalize_intrinsics_tensor(source_camera_intrinsics)
        source_images, source_poses, source_intrs = _filter_source_views_by_pitch(
            source_images,
            source_poses,
            source_intrs,
            source_pitch_filter,
            source_pitch_tolerance_deg,
            "SplatPatchPOC",
        )
        source_vectors = _source_observation_vectors(splats["means"], source_poses)
        support_threshold = math.cos(math.radians(float(support_angle_deg)))
        repair_poses = _generate_repair_poses(
            source_poses,
            splats,
            view_count,
            trajectory_preset,
            side_offset,
            forward_offset,
            pattern,
            dual_side_offsets,
            offset_mode,
            clamp_to_splat_bounds,
            bounds_margin_fraction,
        )
        K = _build_intrinsics(width, height, fov)
        repair_intrs = K.unsqueeze(0).repeat(repair_poses.shape[0], 1, 1)

        rendered_frames = []
        hole_frames = []
        masks = []
        gsplat_failed = False
        for i, pose in enumerate(repair_poses):
            support_image = None
            if use_gsplat and not gsplat_failed:
                try:
                    image, alpha = _render_with_gsplat(splats, pose, K, width, height, max_render_points)
                except Exception as exc:
                    print(f"⚠️ [SplatPatchPOC] gsplat render failed; using point fallback ({type(exc).__name__}: {exc})")
                    gsplat_failed = True
                    image, alpha, support_image = _point_render_fallback(
                        splats, pose, K, width, height, max_render_points, source_vectors=source_vectors
                    )
            else:
                image, alpha, support_image = _point_render_fallback(
                    splats, pose, K, width, height, max_render_points, source_vectors=source_vectors
                )

            alpha_mask = (alpha < float(coverage_threshold)).float()
            if mask_mode in (
                "source_support",
                "alpha_or_source_support",
                "source_support_or_dark",
                "dark_regions_with_support_gate",
            ):
                if support_image is None:
                    _, _, support_image = _point_render_fallback(
                        splats, pose, K, width, height, max_render_points, source_vectors=source_vectors
                    )
                support_mask = (support_image < support_threshold).float()
            else:
                support_mask = torch.zeros_like(alpha_mask)

            if mask_mode == "black_pixels":
                dark_mask = (image.detach().cpu().float()[..., :3].mean(dim=-1) < float(dark_threshold)).float()
            elif mask_mode in ("dark_regions", "source_support_or_dark", "dark_regions_with_support_gate"):
                dark_mask = _dark_region_mask(
                    image,
                    alpha,
                    dark_threshold,
                    dark_max_alpha,
                    dark_min_area_ratio,
                    dark_mask_dilate,
                )
            else:
                dark_mask = torch.zeros_like(alpha_mask)

            if mask_mode == "alpha_pixels":
                hole_mask = alpha_mask
            elif mask_mode == "source_support":
                hole_mask = support_mask
            elif mask_mode == "alpha_or_source_support":
                hole_mask = torch.maximum(alpha_mask, support_mask)
            elif mask_mode == "dark_regions":
                hole_mask = dark_mask
            elif mask_mode == "black_pixels":
                hole_mask = dark_mask
            elif mask_mode == "dark_regions_with_support_gate":
                hole_mask = dark_mask * support_mask
            elif mask_mode == "source_support_or_dark":
                hole_mask = torch.maximum(support_mask, dark_mask)
            else:
                hole_mask = alpha_mask

            if mask_mode not in ("alpha_pixels", "black_pixels"):
                hole_mask = _keep_large_components(hole_mask, min_hole_area_ratio)
            hole_frames.append(image.float().clamp(0.0, 1.0))
            repaired = _inpaint_frame(image, hole_mask, inpaint_mode, int(mask_dilate))
            rendered_frames.append(repaired)
            masks.append(hole_mask)
            if (i + 1) % 4 == 0 or i == repair_poses.shape[0] - 1:
                missing = float(hole_mask.mean().item()) * 100.0
                support_note = ""
                if support_image is not None:
                    support_note = f", min_support={float(support_image.min().item()):.3f}"
                print(
                    f"[SplatPatchPOC] repair view {i + 1}/{repair_poses.shape[0]}: "
                    f"mask={missing:.2f}% ({mask_mode}{support_note}, "
                    f"dark={float(dark_mask.mean().item()) * 100.0:.2f}%)"
                )

        repair_images = torch.stack(rendered_frames, dim=0).float().clamp(0.0, 1.0)
        hole_images = torch.stack(hole_frames, dim=0).float().clamp(0.0, 1.0)
        hole_masks = torch.stack(masks, dim=0).float().clamp(0.0, 1.0)
        print(
            "[SplatPatchPOC] hole_images stats: "
            f"shape={tuple(hole_images.shape)}, min={float(hole_images.min().item()):.4f}, "
            f"mean={float(hole_images.mean().item()):.4f}, max={float(hole_images.max().item()):.4f}"
        )
        if float(hole_images.max().item()) < 0.02:
            print("⚠️ [SplatPatchPOC] hole_images are almost fully black at the POC output.")

        if combine_with_source and source_poses is not None and source_intrs is not None:
            src_images, source_intrs = _resize_images_and_intrinsics(source_images, source_intrs, width, height)
            combined_images = torch.cat([src_images, repair_images], dim=0)
            combined_poses = torch.cat([source_poses, repair_poses], dim=0)
            if source_intrs.shape[0] == src_images.shape[0]:
                combined_intrs = torch.cat([source_intrs, repair_intrs], dim=0)
            else:
                print("[SplatPatchPOC] source intrinsics count mismatch; combined intrinsics use repair intrinsics only.")
                combined_intrs = repair_intrs
        else:
            combined_images = repair_images
            combined_poses = repair_poses
            combined_intrs = repair_intrs

        print(
            "[SplatPatchPOC] generated repair set: "
            f"repair={tuple(repair_images.shape)}, combined={tuple(combined_images.shape)}"
        )
        return (
            repair_images,
            hole_masks,
            repair_poses,
            repair_intrs,
            combined_images,
            combined_poses,
            combined_intrs,
            hole_images,
        )


class VNCCS_SplatPatchAssemble:
    """Assemble externally edited repair frames into a second-pass WorldMirror batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "patched_repair_images": ("IMAGE",),
                "repair_camera_poses": ("TENSOR",),
                "repair_camera_intrinsics": ("TENSOR",),
            },
            "optional": {
                "base_repair_images": ("IMAGE",),
                "hole_masks": ("MASK",),
                "source_images": ("IMAGE",),
                "source_camera_poses": ("TENSOR",),
                "source_camera_intrinsics": ("TENSOR",),
                "include_source": ("BOOLEAN", {"default": True}),
                "use_mask_blend": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "TENSOR", "TENSOR")
    RETURN_NAMES = ("combined_images", "combined_camera_poses", "combined_camera_intrinsics")
    FUNCTION = "assemble"
    CATEGORY = "VNCCS/3D/Experimental"

    def assemble(
        self,
        patched_repair_images,
        repair_camera_poses,
        repair_camera_intrinsics,
        base_repair_images=None,
        hole_masks=None,
        source_images=None,
        source_camera_poses=None,
        source_camera_intrinsics=None,
        include_source=True,
        use_mask_blend=True,
    ):
        patched_raw = patched_repair_images
        if not isinstance(patched_raw, torch.Tensor):
            patched_raw = torch.as_tensor(patched_raw)
        if patched_raw.dim() == 3:
            patched_height, patched_width = int(patched_raw.shape[0]), int(patched_raw.shape[1])
        else:
            patched_height, patched_width = int(patched_raw.shape[1]), int(patched_raw.shape[2])
        patched = _resize_image_batch(patched_raw, patched_width, patched_height)
        repair_poses = _normalize_pose_tensor(repair_camera_poses)
        repair_intrs = _normalize_intrinsics_tensor(repair_camera_intrinsics)
        if repair_poses is None or repair_intrs is None:
            raise ValueError("repair_camera_poses and repair_camera_intrinsics are required.")
        if patched.shape[0] != repair_poses.shape[0]:
            raise ValueError(
                "patched_repair_images count must match repair_camera_poses: "
                f"{patched.shape[0]} images vs {repair_poses.shape[0]} poses"
            )
        if patched.shape[0] != repair_intrs.shape[0]:
            raise ValueError(
                "patched_repair_images count must match repair_camera_intrinsics: "
                f"{patched.shape[0]} images vs {repair_intrs.shape[0]} intrinsics"
            )
        repair_intrs_size = _infer_intrinsics_image_size(repair_intrs)
        if repair_intrs_size is not None:
            src_w, src_h = repair_intrs_size
            dst_w, dst_h = float(patched.shape[2]), float(patched.shape[1])
            if abs(src_w - dst_w) > 1.0 or abs(src_h - dst_h) > 1.0:
                repair_intrs = _scale_intrinsics_to_image(repair_intrs, src_w, src_h, dst_w, dst_h)
                print(
                    "[SplatPatchAssemble] scaled repair intrinsics to patched image size: "
                    f"{src_w:.1f}x{src_h:.1f} -> {dst_w:.1f}x{dst_h:.1f}"
                )

        if use_mask_blend and base_repair_images is not None and hole_masks is not None:
            base = _resize_image_batch(base_repair_images, int(patched.shape[2]), int(patched.shape[1]))
            mask = _resize_mask_batch(hole_masks, int(patched.shape[2]), int(patched.shape[1]))
            count = min(int(patched.shape[0]), int(base.shape[0]), int(mask.shape[0]))
            if count != int(patched.shape[0]):
                print(
                    "[SplatPatchAssemble] mask blend count mismatch; trimming to "
                    f"{count} frames (patched={patched.shape[0]}, base={base.shape[0]}, mask={mask.shape[0]})"
                )
                patched = patched[:count]
                base = base[:count]
                mask = mask[:count]
                repair_poses = repair_poses[:count]
                repair_intrs = repair_intrs[:count]
            patched = (base * (1.0 - mask) + patched * mask).clamp(0.0, 1.0)
            print(
                "[SplatPatchAssemble] mask-blended edited repair frames: "
                f"mask_mean={float(mask.mean().item()) * 100.0:.2f}%"
            )

        if include_source and source_images is not None:
            source_poses = _normalize_pose_tensor(source_camera_poses)
            source_intrs = _normalize_intrinsics_tensor(source_camera_intrinsics)
            if source_poses is None or source_intrs is None:
                raise ValueError("source camera poses/intrinsics are required when include_source is enabled.")
            height, width = int(patched.shape[1]), int(patched.shape[2])
            source_images, source_intrs = _resize_images_and_intrinsics(source_images, source_intrs, width, height)
            if source_images.shape[0] != source_poses.shape[0] or source_images.shape[0] != source_intrs.shape[0]:
                raise ValueError(
                    "source image/camera count mismatch: "
                    f"images={source_images.shape[0]}, poses={source_poses.shape[0]}, "
                    f"intrinsics={source_intrs.shape[0]}"
                )
            images = torch.cat([source_images, patched], dim=0)
            poses = torch.cat([source_poses, repair_poses], dim=0)
            intrs = torch.cat([source_intrs, repair_intrs], dim=0)
        else:
            images = patched
            poses = repair_poses
            intrs = repair_intrs

        _image_stats("patched_repair_images", patched)
        _image_stats("combined_images", images)
        print(
            "[SplatPatchAssemble] camera tensors: "
            f"poses={tuple(poses.shape)}, intrinsics={tuple(intrs.shape)}, "
            f"pose_last={tuple(poses.shape[-2:])}, intr_last={tuple(intrs.shape[-2:])}"
        )
        print(
            "[SplatPatchAssemble] assembled second-pass batch: "
            f"images={tuple(images.shape)}, poses={tuple(poses.shape)}, intrinsics={tuple(intrs.shape)}"
        )
        return images, poses, intrs


class VNCCS_SplatPatchPOC_v2(VNCCS_SplatPatchPOC):
    pass


class VNCCS_SplatPatchPOC2:
    """Find weak geometry regions and render central repair views aimed at them."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_data": ("PLY_DATA",),
            },
            "optional": {
                "source_camera_poses": ("TENSOR",),
                "view_count": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
                "width": ("INT", {"default": 1022, "min": 252, "max": 2048, "step": 14}),
                "height": ("INT", {"default": 1022, "min": 252, "max": 2048, "step": 14}),
                "fov": ("FLOAT", {"default": 90.0, "min": 45.0, "max": 130.0, "step": 1.0}),
                "scan_yaw_count": ("INT", {"default": 24, "min": 4, "max": 96, "step": 4}),
                "scan_pitch_deg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 30.0, "step": 1.0}),
                "scan_size": ("INT", {"default": 384, "min": 128, "max": 768, "step": 64}),
                "camera_radius_fraction": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.35, "step": 0.01}),
                "central_weight_fraction": ("FLOAT", {"default": 0.82, "min": 0.25, "max": 1.25, "step": 0.01}),
                "coverage_threshold": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01}),
                "speckle_kernel": ("INT", {"default": 9, "min": 1, "max": 41, "step": 2}),
                "min_bad_ratio": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 0.95, "step": 0.01}),
                "max_missing_ratio": ("FLOAT", {"default": 0.65, "min": 0.05, "max": 1.0, "step": 0.01}),
                "min_yaw_separation_deg": ("FLOAT", {"default": 18.0, "min": 0.0, "max": 90.0, "step": 1.0}),
                "clamp_to_splat_bounds": ("BOOLEAN", {"default": True}),
                "bounds_margin_fraction": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.45, "step": 0.01}),
                "use_gsplat": ("BOOLEAN", {"default": True}),
                "max_render_points": ("INT", {"default": 1200000, "min": 0, "max": 20000000, "step": 100000}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "TENSOR", "TENSOR", "IMAGE")
    RETURN_NAMES = ("triage_images", "triage_masks", "triage_camera_poses", "triage_camera_intrinsics", "triage_raw_images")
    FUNCTION = "generate"
    CATEGORY = "VNCCS/3D/Experimental"

    def generate(
        self,
        ply_data,
        source_camera_poses=None,
        view_count=8,
        width=1022,
        height=1022,
        fov=90.0,
        scan_yaw_count=24,
        scan_pitch_deg=8.0,
        scan_size=384,
        camera_radius_fraction=0.08,
        central_weight_fraction=0.82,
        coverage_threshold=0.08,
        speckle_kernel=9,
        min_bad_ratio=0.03,
        max_missing_ratio=0.65,
        min_yaw_separation_deg=18.0,
        clamp_to_splat_bounds=True,
        bounds_margin_fraction=0.08,
        use_gsplat=True,
        max_render_points=1200000,
    ):
        splats = _extract_splats(ply_data)
        if splats is None:
            raise ValueError("VNCCS_SplatPatchPOC2 requires Gaussian splats in PLY_DATA.")

        source_poses = _normalize_pose_tensor(source_camera_poses)
        candidate_poses, labels = _generate_triage_candidate_poses(
            source_poses,
            splats,
            scan_yaw_count,
            scan_pitch_deg,
            camera_radius_fraction,
            clamp_to_splat_bounds,
            bounds_margin_fraction,
        )
        scan_size = int(scan_size)
        scan_k = _build_intrinsics(scan_size, scan_size, fov)
        weight = _central_weight(scan_size, scan_size, central_weight_fraction)

        scored = []
        gsplat_failed = False
        for i, pose in enumerate(candidate_poses):
            if use_gsplat and not gsplat_failed:
                try:
                    _, alpha = _render_with_gsplat(splats, pose, scan_k, scan_size, scan_size, max_render_points)
                except Exception as exc:
                    print(f"⚠️ [SplatPatchPOC2] gsplat scan failed; using point fallback ({type(exc).__name__}: {exc})")
                    gsplat_failed = True
                    _, alpha, _ = _point_render_fallback(splats, pose, scan_k, scan_size, scan_size, max_render_points)
            else:
                _, alpha, _ = _point_render_fallback(splats, pose, scan_k, scan_size, scan_size, max_render_points)

            bad = _bad_geometry_mask(alpha, coverage_threshold, speckle_kernel)
            missing_ratio = float((alpha < float(coverage_threshold)).float().mean().item())
            full_bad = float(bad.mean().item())
            if missing_ratio > float(max_missing_ratio):
                continue
            yaw, pitch, offset_i, clamped = labels[i]
            components = _bad_geometry_components(bad, weight, speckle_kernel, min_bad_ratio)
            for component in components:
                central_bad = component["central_bad"]
                raw_bad_ratio = component["raw_bad_ratio"]
                if central_bad < float(min_bad_ratio) and raw_bad_ratio < float(min_bad_ratio) * 0.75:
                    continue
                direction = _component_world_direction(pose, scan_k, component["cx"], component["cy"])
                score = (
                    central_bad * 2.4
                    + raw_bad_ratio * 1.0
                    + component["bbox_fraction"] * 0.35
                    - component["center_distance"] * 0.18
                    - component["edge_penalty"] * 0.70
                    - max(missing_ratio - 0.45, 0.0)
                )
                scored.append(
                    (
                        score,
                        central_bad,
                        full_bad,
                        missing_ratio,
                        i,
                        yaw,
                        pitch,
                        offset_i,
                        clamped,
                        direction,
                        component,
                    )
                )

        if not scored:
            raise ValueError(
                "Splat Patch POC 2 found no usable weak-geometry views. "
                "Lower min_bad_ratio, raise max_missing_ratio, or increase scan_yaw_count."
            )
        scored.sort(reverse=True, key=lambda item: item[0])

        selected = []
        min_sep = float(min_yaw_separation_deg)
        used_candidates = set()
        for item in scored:
            _, _, _, _, candidate_i, yaw, _, _, _, direction, _ = item
            if candidate_i in used_candidates:
                continue
            too_close = False
            for picked in selected:
                picked_yaw = picked[5]
                delta = abs(math.atan2(math.sin(math.radians(yaw - picked_yaw)), math.cos(math.radians(yaw - picked_yaw))))
                direction_delta = math.degrees(
                    math.acos(float((direction * picked[9]).sum().clamp(-1.0, 1.0).item()))
                )
                if direction_delta < max(15.0, min_sep) or (math.degrees(delta) < min_sep and direction_delta < 35.0):
                    too_close = True
                    break
            if too_close:
                continue
            selected.append(item)
            used_candidates.add(candidate_i)
            if len(selected) >= int(view_count):
                break

        final_k = _build_intrinsics(width, height, fov)
        images = []
        raw_images = []
        masks = []
        poses = []
        gsplat_failed_final = False
        for out_i, item in enumerate(selected[: int(view_count)]):
            score, central_bad, full_bad, missing_ratio, candidate_i, yaw, pitch, offset_i, clamped, _, component = item
            pose = candidate_poses[candidate_i]
            if use_gsplat and not gsplat_failed_final:
                try:
                    image, alpha = _render_with_gsplat(splats, pose, final_k, width, height, max_render_points)
                except Exception as exc:
                    print(f"⚠️ [SplatPatchPOC2] gsplat final render failed; using point fallback ({type(exc).__name__}: {exc})")
                    gsplat_failed_final = True
                    image, alpha, _ = _point_render_fallback(splats, pose, final_k, width, height, max_render_points)
            else:
                image, alpha, _ = _point_render_fallback(splats, pose, final_k, width, height, max_render_points)
            mask = _bad_geometry_mask(alpha, coverage_threshold, speckle_kernel)
            raw_images.append(image.float().clamp(0.0, 1.0))
            images.append(image.float().clamp(0.0, 1.0))
            masks.append(mask.float().clamp(0.0, 1.0))
            poses.append(pose)
            print(
                f"[SplatPatchPOC2] selected {out_i:02d}: yaw={yaw:.1f}, pitch={pitch:.1f}, "
                f"offset={offset_i}, score={score:.4f}, central_bad={central_bad * 100.0:.2f}%, "
                f"full_bad={full_bad * 100.0:.2f}%, missing={missing_ratio * 100.0:.2f}%, "
                f"hole_bbox={component['bbox']}, clamped={clamped}"
            )

        triage_images = torch.stack(images, dim=0)
        triage_raw_images = torch.stack(raw_images, dim=0)
        triage_masks = torch.stack(masks, dim=0)
        triage_poses = torch.stack(poses, dim=0)
        triage_intrs = final_k.unsqueeze(0).repeat(triage_poses.shape[0], 1, 1)
        print(
            "[SplatPatchPOC2] generated triage set: "
            f"images={tuple(triage_images.shape)}, poses={tuple(triage_poses.shape)}, "
            f"intrinsics={tuple(triage_intrs.shape)}, scanned={len(labels)}, usable={len(scored)}"
        )
        return triage_images, triage_masks, triage_poses, triage_intrs, triage_raw_images


class VNCCS_ApplySplatPatchFromEditedViews:
    """Patch an existing splat directly from edited repair views and masks."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_data": ("PLY_DATA",),
                "edited_repair_images": ("IMAGE",),
                "repair_camera_poses": ("TENSOR",),
                "repair_camera_intrinsics": ("TENSOR",),
            },
            "optional": {
                "hole_masks": ("MASK",),
                "patch_depth_maps": ("IMAGE",),
                "blend_mode": (["replace_surface", "add_only", "legacy_project_mask"], {"default": "replace_surface"}),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_dilate": ("INT", {"default": 5, "min": 0, "max": 101, "step": 2}),
                "remove_old_splats": ("BOOLEAN", {"default": True}),
                "remove_depth_tolerance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.005}),
                "replace_front_margin": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.5, "step": 0.002}),
                "replace_far_margin": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "depth_fill_iterations": ("INT", {"default": 96, "min": 0, "max": 512, "step": 8}),
                "min_patch_luma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.25, "step": 0.005}),
                "patch_splat_scale": ("FLOAT", {"default": 0.003, "min": 0.0001, "max": 0.1, "step": 0.0001}),
                "patch_opacity": ("FLOAT", {"default": 0.95, "min": 0.01, "max": 1.0, "step": 0.01}),
                "pixel_stride": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
                "max_patch_points_per_view": ("INT", {"default": 250000, "min": 0, "max": 2000000, "step": 50000}),
                "max_depth_render_points": ("INT", {"default": 0, "min": 0, "max": 20000000, "step": 100000}),
                "debug_log": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("PLY_DATA",)
    RETURN_NAMES = ("patched_ply_data",)
    FUNCTION = "apply_patch"
    CATEGORY = "VNCCS/3D/Experimental"

    def apply_patch(
        self,
        ply_data,
        edited_repair_images,
        repair_camera_poses,
        repair_camera_intrinsics,
        hole_masks=None,
        patch_depth_maps=None,
        blend_mode="replace_surface",
        mask_threshold=0.5,
        mask_dilate=5,
        remove_old_splats=True,
        remove_depth_tolerance=0.0,
        replace_front_margin=0.01,
        replace_far_margin=0.0,
        depth_fill_iterations=96,
        min_patch_luma=0.0,
        patch_splat_scale=0.003,
        patch_opacity=0.95,
        pixel_stride=1,
        max_patch_points_per_view=250000,
        max_depth_render_points=0,
        debug_log=True,
    ):
        splats = _extract_splats(ply_data)
        if splats is None:
            raise ValueError("Apply Splat Patch requires Gaussian splats in PLY_DATA.")
        poses = _normalize_pose_tensor(repair_camera_poses)
        intrs = _normalize_intrinsics_tensor(repair_camera_intrinsics)
        if poses is None or intrs is None:
            raise ValueError("repair_camera_poses and repair_camera_intrinsics are required.")

        images = _resize_image_batch(edited_repair_images, int(edited_repair_images.shape[2]), int(edited_repair_images.shape[1]))
        original_image_shape = tuple(edited_repair_images.shape) if isinstance(edited_repair_images, torch.Tensor) else None
        image_width = int(images.shape[2])
        image_height = int(images.shape[1])
        inferred_intr_size = _infer_intrinsics_image_size(intrs)
        if inferred_intr_size is not None:
            src_w, src_h = inferred_intr_size
            if abs(src_w - image_width) > 1.0 or abs(src_h - image_height) > 1.0:
                intrs = _scale_intrinsics_to_image(intrs, src_w, src_h, image_width, image_height)
                print(
                    "[ApplySplatPatch] scaled repair intrinsics to edited image size: "
                    f"{src_w:.1f}x{src_h:.1f} -> {image_width}x{image_height}"
                )
        if hole_masks is None:
            masks = torch.ones(int(images.shape[0]), int(images.shape[1]), int(images.shape[2]), dtype=torch.float32)
            print("[ApplySplatPatch] hole_masks not connected; using full-frame masks for testing.")
        else:
            masks = _resize_mask_batch(hole_masks, int(images.shape[2]), int(images.shape[1]))[..., 0]
        patch_depths = None
        if patch_depth_maps is not None:
            patch_depths = _resize_depth_batch(patch_depth_maps, int(images.shape[2]), int(images.shape[1]))
            if int(patch_depths.shape[0]) == 1 and int(images.shape[0]) > 1:
                patch_depths = patch_depths.repeat(int(images.shape[0]), 1, 1)
            print(f"[ApplySplatPatch] patch_depth_maps connected: {tuple(patch_depths.shape)}")
        count_items = [int(images.shape[0]), int(masks.shape[0]), int(poses.shape[0]), int(intrs.shape[0])]
        if patch_depths is not None:
            count_items.append(int(patch_depths.shape[0]))
        count = min(count_items)
        if count <= 0:
            raise ValueError("Apply Splat Patch received an empty image/mask/camera batch.")
        images = images[:count]
        masks = masks[:count]
        poses = poses[:count]
        intrs = intrs[:count]
        if patch_depths is not None:
            patch_depths = patch_depths[:count]

        patched_splats = splats
        total_removed = 0
        total_added = 0
        height, width = int(images.shape[1]), int(images.shape[2])
        if debug_log:
            means_bounds = _point_bounds_debug(patched_splats["means"])
            print(
                "[ApplySplatPatch][BEGIN] "
                f"input_splats={int(patched_splats['means'].shape[0])}, "
                f"images_raw={original_image_shape}, images={tuple(images.shape)}, "
                f"masks={tuple(masks.shape)}, patch_depths={tuple(patch_depths.shape) if patch_depths is not None else None}, "
                f"poses={tuple(poses.shape)}, intrinsics={tuple(intrs.shape)}, "
                f"count={count}, size={width}x{height}"
            )
            print(
                "[ApplySplatPatch][PARAMS] "
                f"blend_mode={blend_mode}, mask_threshold={float(mask_threshold):.3f}, mask_dilate={int(mask_dilate)}, "
                f"remove_old_splats={bool(remove_old_splats)}, remove_depth_tolerance={float(remove_depth_tolerance):.4f}, "
                f"replace_front_margin={float(replace_front_margin):.4f}, replace_far_margin={float(replace_far_margin):.4f}, "
                f"depth_fill_iterations={int(depth_fill_iterations)}, min_patch_luma={float(min_patch_luma):.4f}, "
                f"patch_splat_scale={float(patch_splat_scale):.5f}, patch_opacity={float(patch_opacity):.3f}, "
                f"pixel_stride={int(pixel_stride)}, max_patch_points_per_view={int(max_patch_points_per_view)}, "
                f"max_depth_render_points={int(max_depth_render_points)}"
            )
            if means_bounds is not None:
                print(
                    "[ApplySplatPatch][INPUT_BOUNDS] "
                    f"min=({_fmt_float(means_bounds[0][0])},{_fmt_float(means_bounds[0][1])},{_fmt_float(means_bounds[0][2])}), "
                    f"max=({_fmt_float(means_bounds[1][0])},{_fmt_float(means_bounds[1][1])},{_fmt_float(means_bounds[1][2])})"
                )
        for i in range(count):
            mask_before = masks[i]
            mask = _dilate_mask(masks[i], int(mask_dilate))
            before_stats = _mask_debug_stats(mask_before, mask_threshold)
            after_stats = _mask_debug_stats(mask, mask_threshold)
            pose_t = poses[i, :3, 3]
            pose_f = poses[i, :3, 2]
            K = intrs[i]
            if debug_log:
                print(
                    f"[ApplySplatPatch][VIEW {i:02d}][CAMERA] "
                    f"pos=({_fmt_float(pose_t[0])},{_fmt_float(pose_t[1])},{_fmt_float(pose_t[2])}), "
                    f"forward=({_fmt_float(pose_f[0])},{_fmt_float(pose_f[1])},{_fmt_float(pose_f[2])}), "
                    f"fx={_fmt_float(K[0, 0])}, fy={_fmt_float(K[1, 1])}, "
                    f"cx={_fmt_float(K[0, 2])}, cy={_fmt_float(K[1, 2])}"
                )
                print(
                    f"[ApplySplatPatch][VIEW {i:02d}][MASK] "
                    f"before={before_stats['count']}/{before_stats['pixels']} ({before_stats['ratio'] * 100.0:.2f}%), "
                    f"bbox={before_stats['bbox']}, value=min/mean/max "
                    f"{before_stats['min']:.3f}/{before_stats['mean']:.3f}/{before_stats['max']:.3f}; "
                    f"after_dilate={after_stats['count']}/{after_stats['pixels']} ({after_stats['ratio'] * 100.0:.2f}%), "
                    f"bbox={after_stats['bbox']}"
                )
            if float(mask.max().item()) < float(mask_threshold):
                print(f"[ApplySplatPatch] view {i:02d}: mask is empty; skipped.")
                continue
            depth = _render_depth_from_splats(
                patched_splats,
                poses[i],
                intrs[i],
                width,
                height,
                int(max_depth_render_points),
            )
            depth_stats = _depth_debug_stats(depth, mask, mask_threshold)
            if debug_log:
                print(
                    f"[ApplySplatPatch][VIEW {i:02d}][DEPTH] "
                    f"finite_total={depth_stats['finite_total']}/{width * height}, "
                    f"finite_in_mask={depth_stats['finite_region']}/{depth_stats['region_pixels']} "
                    f"({depth_stats['region_ratio'] * 100.0:.2f}%), "
                    f"depth=min/mean/max {_fmt_float(depth_stats['min'])}/"
                    f"{_fmt_float(depth_stats['mean'])}/{_fmt_float(depth_stats['max'])}"
                )

            patch_depth = patch_depths[i] if patch_depths is not None else depth
            depth_fill_stats = None
            if str(blend_mode) == "replace_surface":
                patch_depth, depth_fill_stats = _complete_depth_in_region(
                    patch_depth,
                    mask,
                    mask_threshold,
                    int(depth_fill_iterations),
                )
                if debug_log:
                    patch_depth_stats = _depth_debug_stats(patch_depth, mask, mask_threshold)
                    print(
                        f"[ApplySplatPatch][VIEW {i:02d}][DEPTH_FILL] "
                        f"source={'patch_depth_maps' if patch_depths is not None else 'rendered_splat_depth'}, "
                        f"input_valid={depth_fill_stats['input_valid']}/{depth_fill_stats['region_pixels']}, "
                        f"filled={depth_fill_stats['filled']}, iterations={depth_fill_stats['iterations']}, "
                        f"final_valid={patch_depth_stats['finite_region']}/{patch_depth_stats['region_pixels']} "
                        f"({patch_depth_stats['region_ratio'] * 100.0:.2f}%), "
                        f"depth=min/mean/max {_fmt_float(patch_depth_stats['min'])}/"
                        f"{_fmt_float(patch_depth_stats['mean'])}/{_fmt_float(patch_depth_stats['max'])}"
                    )

            if remove_old_splats and str(blend_mode) == "replace_surface":
                patched_splats, removed, remove_stats = _remove_splats_behind_patch_surface(
                    patched_splats,
                    poses[i],
                    intrs[i],
                    mask,
                    mask_threshold,
                    patch_depth,
                    replace_front_margin,
                    replace_far_margin,
                )
                total_removed += removed
            elif remove_old_splats and str(blend_mode) == "legacy_project_mask":
                patched_splats, removed, remove_stats = _remove_splats_projecting_to_mask(
                    patched_splats,
                    poses[i],
                    intrs[i],
                    mask,
                    mask_threshold,
                    depth,
                    remove_depth_tolerance,
                )
                total_removed += removed
            else:
                removed = 0
                remove_stats = {
                    "input": int(patched_splats["means"].shape[0]),
                    "projected": 0,
                    "mask_hits": 0,
                    "surface_checked": 0,
                    "behind_hits": 0,
                    "depth_checked": 0,
                    "depth_near_hits": 0,
                    "removed": 0,
                }
            if debug_log:
                print(
                    f"[ApplySplatPatch][VIEW {i:02d}][REMOVE] "
                    f"enabled={bool(remove_old_splats)}, splats_before={remove_stats['input']}, "
                    f"projected={remove_stats['projected']} "
                    f"({100.0 * remove_stats['projected'] / max(remove_stats['input'], 1):.2f}%), "
                    f"mask_hits={remove_stats['mask_hits']}, "
                    f"surface_checked={remove_stats.get('surface_checked', 0)}, "
                    f"behind_hits={remove_stats.get('behind_hits', 0)}, "
                    f"depth_checked={remove_stats.get('depth_checked', 0)}, "
                    f"depth_near_hits={remove_stats.get('depth_near_hits', 0)}, "
                    f"removed={removed}, splats_after={int(patched_splats['means'].shape[0])}"
                )

            points, colors, patch_stats = _backproject_patch_pixels(
                images[i],
                mask,
                patch_depth,
                poses[i],
                intrs[i],
                mask_threshold,
                int(max_patch_points_per_view),
                int(pixel_stride),
                float(min_patch_luma),
            )
            patched_splats = _append_patch_splats(
                patched_splats,
                points,
                colors,
                patch_splat_scale,
                patch_opacity,
            )
            total_added += int(points.shape[0])
            if debug_log:
                bounds = patch_stats.get("world_bounds")
                cap_text = (
                    f", capped_from={patch_stats['capped_from']}"
                    if patch_stats.get("capped_from") is not None else ""
                )
                if bounds is None:
                    bounds_text = "world_bounds=n/a"
                else:
                    bounds_text = (
                        "world_bounds="
                        f"min=({_fmt_float(bounds[0][0])},{_fmt_float(bounds[0][1])},{_fmt_float(bounds[0][2])}), "
                        f"max=({_fmt_float(bounds[1][0])},{_fmt_float(bounds[1][1])},{_fmt_float(bounds[1][2])})"
                    )
                print(
                    f"[ApplySplatPatch][VIEW {i:02d}][ADD] "
                    f"mask_pixels={patch_stats['mask_pixels']}, "
                    f"depth_valid_pixels={patch_stats['depth_valid_pixels']}, "
                    f"color_valid_pixels={patch_stats.get('color_valid_pixels', 0)}, "
                    f"min_luma={patch_stats.get('min_luma', 0.0):.4f}, "
                    f"valid_before_stride={patch_stats['valid_before_stride']}, "
                    f"valid_after_stride={patch_stats['valid_after_stride']}{cap_text}, "
                    f"selected={patch_stats['selected']}, "
                    f"depth=min/mean/max {_fmt_float(patch_stats.get('depth_min'))}/"
                    f"{_fmt_float(patch_stats.get('depth_mean'))}/{_fmt_float(patch_stats.get('depth_max'))}, "
                    f"color=min/mean/max {_fmt_float(patch_stats.get('color_min'))}/"
                    f"{_fmt_float(patch_stats.get('color_mean'))}/{_fmt_float(patch_stats.get('color_max'))}, "
                    f"{bounds_text}"
                )
            valid_depth_ratio = float((torch.isfinite(patch_depth) & (patch_depth > 1e-5) & (mask >= float(mask_threshold))).float().sum().item())
            valid_depth_ratio /= max(float((mask >= float(mask_threshold)).float().sum().item()), 1.0)
            print(
                f"[ApplySplatPatch] view {i:02d}: removed={removed}, added={int(points.shape[0])}, "
                f"mask={float(mask.mean().item()) * 100.0:.2f}%, valid_depth_in_mask={valid_depth_ratio * 100.0:.2f}%"
            )

        out = dict(ply_data or {})
        out["splats"] = {
            "means": patched_splats["means"].unsqueeze(0),
            "scales": patched_splats["scales"].unsqueeze(0),
            "quats": patched_splats["quats"].unsqueeze(0),
            "opacities": patched_splats["opacities"].unsqueeze(0),
            "sh": patched_splats["sh"].unsqueeze(0),
        }
        out["skip_scale_filter"] = True
        out["pts3d"] = None
        out["pts3d_filtered"] = None
        if debug_log:
            final_bounds = _point_bounds_debug(patched_splats["means"])
            if final_bounds is not None:
                print(
                    "[ApplySplatPatch][FINAL_BOUNDS] "
                    f"min=({_fmt_float(final_bounds[0][0])},{_fmt_float(final_bounds[0][1])},{_fmt_float(final_bounds[0][2])}), "
                    f"max=({_fmt_float(final_bounds[1][0])},{_fmt_float(final_bounds[1][1])},{_fmt_float(final_bounds[1][2])})"
                )
        print(
            "[ApplySplatPatch] patched PLY_DATA: "
            f"views={count}, removed={total_removed}, added={total_added}, "
            f"final={int(patched_splats['means'].shape[0])} gaussians"
        )
        return (out,)


NODE_CLASS_MAPPINGS = {
    "VNCCS_SplatPatchPOC": VNCCS_SplatPatchPOC,
    "VNCCS_SplatPatchPOC_v2": VNCCS_SplatPatchPOC_v2,
    "VNCCS_SplatPatchPOC2": VNCCS_SplatPatchPOC2,
    "VNCCS_ApplySplatPatchFromEditedViews": VNCCS_ApplySplatPatchFromEditedViews,
    "VNCCS_SplatPatchAssemble": VNCCS_SplatPatchAssemble,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_SplatPatchPOC": "🧩 Splat Patch POC",
    "VNCCS_SplatPatchPOC_v2": "🧩 Splat Patch POC v2",
    "VNCCS_SplatPatchPOC2": "🧩 Splat Patch POC 2",
    "VNCCS_ApplySplatPatchFromEditedViews": "🧩 Apply Splat Patch From Edited Views",
    "VNCCS_SplatPatchAssemble": "🧩 Assemble Splat Patches",
}
