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

        # ── load in float32 ───────────────────────────────────────────────────
        # Always load without enable_bf16 so weights arrive as float32 and
        # .to() is the standard nn.Module version. We apply precision below.
        print(f"🔄 [V2] Loading model (device={device}, precision={precision})")
        model = WorldMirror.from_pretrained(model_dir)
        _move_worldmirror(model, device)

        # ── bf16 ──────────────────────────────────────────────────────────────
        if precision == "bf16":
            from hyworld2.worldrecon.pipeline import _collect_fp32_critical_modules
            crit = _collect_fp32_critical_modules(model)
            model.to(torch.bfloat16)
            for mod in crit:
                mod.to(torch.float32)

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

            model.enable_bf16 = True
            model.to = model._bf16_to

        # ── fp8 weight-only via torchao ───────────────────────────────────────
        elif precision == "fp8":
            try:
                from torchao.quantization import quantize_, float8_weight_only
            except ImportError:
                raise ImportError(
                    "torchao is required for fp8. Install with: pip install torchao"
                )
            # fp8 weight-only: weights stored as e4m3fn, dequantized to bf16 for matmul.
            # Uses bf16 activations in the forward pass.
            from hyworld2.worldrecon.pipeline import _collect_fp32_critical_modules
            crit = _collect_fp32_critical_modules(model)
            model.to(torch.bfloat16)
            for mod in crit:
                mod.to(torch.float32)

            quantize_(model, float8_weight_only())
            print(f"✅ [V2] fp8 weight quantization applied")

            # Still need the bf16 forward path for activations
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

            model.enable_bf16 = True
            model.to = model._bf16_to

        model.eval()
        print("✅ [V2] Model ready")

        return ({"model": model, "device": device},)


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
                    "default": 952, "min": 252, "max": 1400, "step": 14,
                    "tooltip": "Longest side in pixels. V2 natively supports high resolutions."
                }),
                "offload_scheme": (["none", "model_cpu_offload"], {"default": "none"}),
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
                "splat_camera_source": (["input_when_available", "predicted"], {
                    "default": "input_when_available",
                    "tooltip": "input_when_available keeps panorama GS positions locked to supplied cameras. predicted matches official inference."
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
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
    )
    RETURN_NAMES  = (
        "ply_data", "depth_maps", "normal_maps", "camera_poses", "camera_intrinsics", "raw_splats",
        "depth_conf", "pts3d_conf", "depth_mask", "gs_depth_conf", "gs_depth_mask", "filter_mask", "gs_filter_mask",
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
        confidence_percentile = 10.0,
        apply_sky_mask      = False,
        filter_edges        = True,
        filter_splats       = False,
        edge_normal_threshold = 1.0,
        edge_depth_threshold  = 0.03,
        apply_confidence_mask = False,
        camera_conditioning = "pose+intrinsics",
        splat_camera_source = "input_when_available",
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

        target_size    = (target_size // _PATCH_SIZE) * _PATCH_SIZE
        if adaptive_target_size:
            target_size = _adaptive_target_size_from_images(images, target_size)
        worldmirror    = model["model"]
        exec_dev       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        original_dev   = _first_real_device(worldmirror)
        has_meta_params = _module_has_meta_tensors(worldmirror)

        # ── 1. Preprocess: ComfyUI IMAGE [B,H,W,C] → tensor [1,S,3,H,W] ─────
        B = images.shape[0]
        tensor_list = []

        for i in range(B):
            img_np  = (images[i].cpu().numpy()[..., :3] * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            t, orig_w, orig_h, new_w, new_h = _resize_to_tensor(pil_img, target_size)

            # centre-crop height if it exceeds target_size
            if new_h > target_size:
                crop = (new_h - target_size) // 2
                t = t[:, crop:crop + target_size, :]
                if camera_intrinsics is not None:
                    camera_intrinsics = camera_intrinsics.clone()
                    camera_intrinsics[i, 1, 2] -= crop

            # scale intrinsics to match resized resolution
            if camera_intrinsics is not None:
                camera_intrinsics = camera_intrinsics.clone()
                sx, sy = new_w / orig_w, new_h / orig_h
                camera_intrinsics[i, 0, 0] *= sx
                camera_intrinsics[i, 1, 1] *= sy
                camera_intrinsics[i, 0, 2] *= sx
                camera_intrinsics[i, 1, 2] *= sy

            tensor_list.append(t)

        imgs_tensor = torch.stack(tensor_list).unsqueeze(0).to(exec_dev)  # [1,S,3,H,W]

        # ── 2. Build views dict + cond_flags ──────────────────────────────────
        views      = {"img": imgs_tensor}
        cond_flags = [0, 0, 0]  # [pose, depth, intrinsics]

        use_pose_prior = camera_conditioning in ("pose+intrinsics", "pose_only")
        use_intrinsics_prior = camera_conditioning in ("pose+intrinsics", "intrinsics_only")
        has_input_cameras = camera_poses is not None and camera_intrinsics is not None
        use_input_splat_cameras = splat_camera_source == "input_when_available" and has_input_cameras

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
        original_gs = worldmirror.enable_gs
        gs_renderer = getattr(worldmirror, "gs_renderer", None)
        original_inference_position_from = (
            getattr(gs_renderer, "inference_position_from", "gsdepth+predcamera")
            if gs_renderer is not None else None
        )
        worldmirror.enable_gs = use_gsplat and GSPLAT_AVAILABLE
        effective_splat_camera_source = "predicted"
        if worldmirror.enable_gs and gs_renderer is not None:
            gs_renderer.inference_position_from = (
                "gsdepth+gtcamera" if use_input_splat_cameras else "gsdepth+predcamera"
            )
            effective_splat_camera_source = "input" if use_input_splat_cameras else "predicted"

        try:
            print(
                f"🚀 [V2] Inference: {B} images @ {target_size}px, gs={worldmirror.enable_gs}, "
                f"camera_conditioning={camera_conditioning}, splat_camera_source={splat_camera_source} "
                f"(effective={effective_splat_camera_source}), splat_color_source={splat_color_source}"
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
            worldmirror.enable_gs = original_gs
            if gs_renderer is not None and original_inference_position_from is not None:
                gs_renderer.inference_position_from = original_inference_position_from
            if offload_scheme == "none" and original_dev.type == "cpu" and not _module_has_meta_tensors(worldmirror):
                _move_worldmirror(worldmirror, "cpu")
                torch.cuda.empty_cache()

        if debug_log:
            _log_worldmirror_debug(predictions, views, camera_poses, camera_intrinsics, imgs_tensor)

        # ── 5. Sky mask (model-native first, ONNX fallback) ──────────────────
        S, H, W = predictions["depth"].shape[1:4]
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

        # ── 8. Filter splats with GS-specific mask ────────────────────────────
        splats = predictions.get("splats")
        if splat_color_source == "input_image":
            splats = _replace_splat_colors_from_images(splats, imgs_tensor)
        splats = _tune_splats(splats, splat_scale_multiplier, splat_opacity_floor)
        if debug_log:
            _log_splat_stats(splats)
        splat_mask = gs_mask if gs_mask is not None else pts_mask
        splats = _apply_mask_to_splats(splats, splat_mask if filter_splats else None)
        if voxel_prune_splats:
            splats = _voxel_prune_splats_dict(splats, voxel_size)

        # ── 9. Assemble PLY_DATA ──────────────────────────────────────────────
        ply_data = {
            "pts3d":          predictions.get("pts3d"),
            "pts3d_filtered": filtered_pts,
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
        depth_conf_out = _map_to_comfy_image(predictions.get("depth_conf"), fallback_shape)
        pts3d_conf_out = _map_to_comfy_image(predictions.get("pts3d_conf"), fallback_shape)
        depth_mask_out = _map_to_comfy_image(predictions.get("depth_mask"), fallback_shape, normalize=False)
        gs_depth_conf_out = _map_to_comfy_image(predictions.get("gs_depth_conf"), fallback_shape)
        gs_depth_mask_out = _map_to_comfy_image(predictions.get("gs_depth_mask"), fallback_shape, normalize=False)
        filter_mask_out = _map_to_comfy_image(pts_mask, fallback_shape, normalize=False)
        gs_filter_mask_out = _map_to_comfy_image(gs_mask, fallback_shape, normalize=False)

        return (
            ply_data, depth_out, normals_out, cam_poses, cam_intrs, predictions,
            depth_conf_out, pts3d_conf_out, depth_mask_out, gs_depth_conf_out,
            gs_depth_mask_out, filter_mask_out, gs_filter_mask_out,
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
            "default": 1022, "min": 252, "max": 1400, "step": 14,
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
        depth = raw.get(depth_source)
        if depth is None and depth_source != "depth":
            depth = raw.get("depth")
        if not isinstance(depth, torch.Tensor):
            print(f"⚠️ [V2 EXP] splat upsample skipped: missing depth source '{depth_source}'.")
            return output

        highres_imgs, highres_intrs = self._prepare_highres_views(original_images, camera_intrinsics, upsample_size)
        S, _, H, W = highres_imgs.shape[1:]
        if depth.dim() == 5:
            depth_low = depth[0, ..., 0]
        elif depth.dim() == 4 and depth.shape[-1] == 1:
            depth_low = depth[..., 0]
        elif depth.dim() == 4:
            depth_low = depth[0]
        else:
            print(f"⚠️ [V2 EXP] splat upsample skipped: unsupported depth shape {tuple(depth.shape)}.")
            return output

        depth_hi = torch.nn.functional.interpolate(
            depth_low.detach().cpu().float().unsqueeze(1),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[:, 0]

        poses = camera_poses.detach().cpu().float()
        intrs = highres_intrs.detach().cpu().float()
        pts, _, _ = depth_to_world_coords_points(depth_hi, poses, intrs)
        means = pts.reshape(1, S * H * W, 3)
        sh = ((highres_imgs.permute(0, 1, 3, 4, 2).reshape(1, S * H * W, 3) - 0.5) / 0.28209479177387814).float()
        depth_flat = depth_hi.reshape(1, S * H * W)

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
            return scales_hw.reshape(1, S * H * W, 1).expand_as(means).float()

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

        splats = {
            "means": means,
            "scales": scales,
            "quats": quats,
            "opacities": opacities,
            "sh": sh[:, :, None, :],
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
            f"{n_before} raw gaussians from {depth_source}"
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
            output[8],
            output[9],
            output[10],
            output[11],
            output[12],
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
                    "default": 518, "min": 252, "max": 1400, "step": 14,
                    "tooltip": "Model inference resolution. Keep this low for multi-view panorama passes; dense splats can be upsampled separately."
                }),
                "offload_scheme": (["none", "model_cpu_offload"], {
                    "default": "none",
                    "tooltip": "Move model weights to CPU between GPU use to reduce VRAM at the cost of speed."
                }),
                "apply_sky_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Remove sky-like regions before saving. Useful for outdoor panoramas where sky can create far/infinite splats."
                }),
                "camera_conditioning": (["pose+intrinsics", "intrinsics_only", "pose_only", "none"], {
                    "default": "pose+intrinsics",
                    "tooltip": "Which input camera priors to pass into WorldMirror V2."
                }),
                "splat_camera_source": (["input_when_available", "predicted"], {
                    "default": "input_when_available",
                    "tooltip": "Use supplied panorama cameras for Gaussian positions when available."
                }),
                "splat_upsample_mode": (["depth_backproject", "none"], {
                    "default": "depth_backproject",
                    "tooltip": "Build dense high-resolution splats from model depth and high-resolution view RGB."
                }),
                "splat_upsample_size": ("INT", {
                    "default": 1022, "min": 252, "max": 1400, "step": 14,
                    "tooltip": "Dense splat grid size. This can be higher than target_size without rerunning the transformer at that resolution."
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
                "debug_log": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print camera/depth/splat alignment diagnostics to the ComfyUI console."
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
        apply_sky_mask=False,
        camera_conditioning="pose+intrinsics",
        splat_camera_source="input_when_available",
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
        return super().run_inference(
            model,
            images,
            use_gsplat=True,
            target_size=target_size,
            offload_scheme=offload_scheme,
            confidence_percentile=10.0,
            apply_sky_mask=apply_sky_mask,
            filter_edges=True,
            filter_splats=False,
            edge_normal_threshold=1.0,
            edge_depth_threshold=0.03,
            apply_confidence_mask=False,
            camera_conditioning=camera_conditioning,
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
            splat_upsample_mode=splat_upsample_mode,
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
