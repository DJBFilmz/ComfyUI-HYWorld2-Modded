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


def _generate_repair_poses(source_poses, splats, view_count, side_offset, forward_offset, pattern):
    if source_poses is not None and source_poses.shape[0] > 0:
        total = min(int(view_count), int(source_poses.shape[0]))
        ids = torch.linspace(0, source_poses.shape[0] - 1, total).round().long().unique()
        poses = []
        for out_i, src_i in enumerate(ids.tolist()):
            base = source_poses[src_i].clone()
            sign = -1.0 if out_i % 2 else 1.0
            right = base[:3, 0]
            forward = base[:3, 2]
            translation = right * float(side_offset) * sign + forward * float(forward_offset)
            if pattern == "left_right_forward":
                mode = out_i % 3
                if mode == 1:
                    translation = right * -float(side_offset) + forward * float(forward_offset)
                elif mode == 2:
                    translation = forward * (float(forward_offset) + abs(float(side_offset)) * 0.5)
            base[:3, 3] = base[:3, 3] + translation
            poses.append(base)
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
                "side_offset": ("FLOAT", {"default": 0.12, "min": -2.0, "max": 2.0, "step": 0.01}),
                "forward_offset": ("FLOAT", {"default": 0.03, "min": -2.0, "max": 2.0, "step": 0.01}),
                "pattern": (["alternate_left_right", "left_right_forward"], {"default": "left_right_forward"}),
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
        side_offset=0.12,
        forward_offset=0.03,
        pattern="left_right_forward",
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
        source_vectors = _source_observation_vectors(splats["means"], source_poses)
        support_threshold = math.cos(math.radians(float(support_angle_deg)))
        repair_poses = _generate_repair_poses(
            source_poses,
            splats,
            view_count,
            side_offset,
            forward_offset,
            pattern,
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


NODE_CLASS_MAPPINGS = {
    "VNCCS_SplatPatchPOC": VNCCS_SplatPatchPOC,
    "VNCCS_SplatPatchPOC_v2": VNCCS_SplatPatchPOC_v2,
    "VNCCS_SplatPatchAssemble": VNCCS_SplatPatchAssemble,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_SplatPatchPOC": "🧩 Splat Patch POC",
    "VNCCS_SplatPatchPOC_v2": "🧩 Splat Patch POC v2",
    "VNCCS_SplatPatchAssemble": "🧩 Assemble Splat Patches",
}
