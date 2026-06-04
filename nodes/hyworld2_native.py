import contextlib
import gc
import json
import os
import shutil
import sys
from argparse import Namespace
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

HYWORLD2_QWENVL_MODELS = {
    "Qwen3-VL-2B-Instruct": {"repo_id": "Qwen/Qwen3-VL-2B-Instruct", "quantized": False},
    "Qwen3-VL-2B-Thinking": {"repo_id": "Qwen/Qwen3-VL-2B-Thinking", "quantized": False},
    "Qwen3-VL-2B-Instruct-FP8": {"repo_id": "Qwen/Qwen3-VL-2B-Instruct-FP8", "quantized": True},
    "Qwen3-VL-2B-Thinking-FP8": {"repo_id": "Qwen/Qwen3-VL-2B-Thinking-FP8", "quantized": True},
    "Qwen3-VL-4B-Instruct": {"repo_id": "Qwen/Qwen3-VL-4B-Instruct", "quantized": False},
    "Qwen3-VL-4B-Thinking": {"repo_id": "Qwen/Qwen3-VL-4B-Thinking", "quantized": False},
    "Qwen3-VL-4B-Instruct-FP8": {"repo_id": "Qwen/Qwen3-VL-4B-Instruct-FP8", "quantized": True},
    "Qwen3-VL-4B-Thinking-FP8": {"repo_id": "Qwen/Qwen3-VL-4B-Thinking-FP8", "quantized": True},
    "Qwen3-VL-8B-Instruct": {"repo_id": "Qwen/Qwen3-VL-8B-Instruct", "quantized": False},
    "Qwen3-VL-8B-Thinking": {"repo_id": "Qwen/Qwen3-VL-8B-Thinking", "quantized": False},
    "Qwen3-VL-8B-Instruct-FP8": {"repo_id": "Qwen/Qwen3-VL-8B-Instruct-FP8", "quantized": True},
    "Qwen3-VL-8B-Thinking-FP8": {"repo_id": "Qwen/Qwen3-VL-8B-Thinking-FP8", "quantized": True},
    "Qwen3-VL-32B-Instruct": {"repo_id": "Qwen/Qwen3-VL-32B-Instruct", "quantized": False},
    "Qwen3-VL-32B-Thinking": {"repo_id": "Qwen/Qwen3-VL-32B-Thinking", "quantized": False},
    "Qwen3-VL-32B-Instruct-FP8": {"repo_id": "Qwen/Qwen3-VL-32B-Instruct-FP8", "quantized": True},
    "Qwen3-VL-32B-Thinking-FP8": {"repo_id": "Qwen/Qwen3-VL-32B-Thinking-FP8", "quantized": True},
    "Qwen2.5-VL-3B-Instruct": {"repo_id": "Qwen/Qwen2.5-VL-3B-Instruct", "quantized": False},
    "Qwen2.5-VL-7B-Instruct": {"repo_id": "Qwen/Qwen2.5-VL-7B-Instruct", "quantized": False},
}
HYWORLD2_QWENVL_DEFAULT = "Qwen3-VL-8B-Instruct"
HYWORLD2_QWENVL_QUANTIZATION = ["None (FP16)", "8-bit (Balanced)", "4-bit (VRAM-friendly)"]
HYWORLD2_QWENVL_ATTENTION = ["auto", "sage", "flash_attention_2", "sdpa"]
HYWORLD2_QWENVL_MAX_IMAGE_EDGE = 768
HYWORLD2_SAM3_REPO_ID = "MIUProject/sam3"


def _ensure_worldgen_path():
    for path in (str(PROJECT_ROOT), str(WORLDGEN_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _output_root() -> Path:
    if folder_paths is not None:
        return Path(folder_paths.get_output_directory())
    return PROJECT_ROOT / "output"


def _qwenvl_model_names():
    return list(HYWORLD2_QWENVL_MODELS.keys())


def _qwenvl_repo_id(model_name):
    name = str(model_name or "").strip()
    if name in HYWORLD2_QWENVL_MODELS:
        return HYWORLD2_QWENVL_MODELS[name]["repo_id"]
    if "/" in name:
        return name
    raise ValueError(f"Unsupported QwenVL model: {model_name}")


def _qwenvl_is_fp8(model_name):
    name = str(model_name or "")
    info = HYWORLD2_QWENVL_MODELS.get(name, {})
    return bool(info.get("quantized")) or "-fp8" in name.lower() or "_fp8" in name.lower()


def _qwenvl_models_dir():
    if folder_paths is not None:
        try:
            llm_paths = folder_paths.get_folder_paths("LLM") if "LLM" in folder_paths.folder_names_and_paths else []
            if llm_paths:
                return Path(llm_paths[0]) / "Qwen-VL"
        except Exception:
            pass
        try:
            return Path(folder_paths.models_dir) / "LLM" / "Qwen-VL"
        except Exception:
            pass
    return PROJECT_ROOT / "models" / "LLM" / "Qwen-VL"


def _qwenvl_ensure_model(model_name):
    repo_id = _qwenvl_repo_id(model_name)
    target = _qwenvl_models_dir() / repo_id.split("/")[-1]
    if target.exists() and target.is_dir():
        if any(target.glob("*.safetensors")) or any(target.glob("*.bin")):
            print(f"[HYWorld2 QwenVL] Using local model '{model_name}' from {target}")
            return str(target)
    target.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise ImportError("QwenVL auto-download requires huggingface_hub.") from exc
    print(f"[HYWorld2 QwenVL] Downloading model '{model_name}' from {repo_id} to {target}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        ignore_patterns=["*.md", ".git*"],
    )
    print(f"[HYWorld2 QwenVL] Model ready: {target}")
    return str(target)


def _qwenvl_normalize_device(device):
    device = str(device or "auto").strip()
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device.isdigit():
        device = f"cuda:{int(device)}"
    if device == "cuda":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            return "cpu"
        try:
            idx = int(device.split(":", 1)[1]) if ":" in device else 0
        except Exception:
            idx = 0
        if idx >= torch.cuda.device_count():
            idx = 0
        return f"cuda:{idx}"
    if device == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        return "cpu"
    return device


def _qwenvl_flash_attn_available():
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            return False
        import flash_attn  # noqa: F401
    except Exception:
        return False
    return True


def _qwenvl_sage_attn_available():
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            return False
        import sageattention  # noqa: F401
    except Exception:
        return False
    return True


def _qwenvl_resolve_attention(attention_mode, force_sdpa=False):
    mode = str(attention_mode or "auto")
    if force_sdpa or mode == "sdpa":
        return "sdpa"
    if mode == "flash_attention_2":
        return "flash_attention_2" if _qwenvl_flash_attn_available() else "sdpa"
    if mode == "sage":
        # We expose the selector, but only use kernels when the installed transformers stack supports it.
        return "sdpa" if not _qwenvl_sage_attn_available() else "sdpa"
    if _qwenvl_flash_attn_available():
        return "flash_attention_2"
    return "sdpa"


def _qwenvl_quantization_config(model_name, quantization, cpu_offload=False):
    if _qwenvl_is_fp8(model_name):
        return None, None, True
    quant = str(quantization or "None (FP16)")
    if quant == "4-bit (VRAM-friendly)":
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            raise ImportError("QwenVL 4-bit quantization requires transformers BitsAndBytesConfig and bitsandbytes.") from exc
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ), None, False
    if quant == "8-bit (Balanced)":
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            raise ImportError("QwenVL 8-bit quantization requires transformers BitsAndBytesConfig and bitsandbytes.") from exc
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=bool(cpu_offload)), None, False
    return None, torch.float16 if torch.cuda.is_available() else torch.float32, False


def _qwenvl_auto_max_memory():
    if not torch.cuda.is_available():
        return None
    max_memory = {}
    for idx in range(torch.cuda.device_count()):
        total_gib = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        gpu_limit = max(1, int(max(1.0, total_gib - 3.0)))
        max_memory[idx] = f"{gpu_limit}GiB"
    max_memory["cpu"] = "64GiB"
    return max_memory


def _qwenvl_preview_image(image, max_image_edge=HYWORLD2_QWENVL_MAX_IMAGE_EDGE):
    if not isinstance(image, Image.Image):
        return image
    width, height = image.size
    max_edge = max(width, height)
    limit = max(64, int(max_image_edge or HYWORLD2_QWENVL_MAX_IMAGE_EDGE))
    if max_edge <= limit:
        return image.convert("RGB")
    scale = limit / float(max_edge)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
    resized = image.convert("RGB").resize(new_size, resampling)
    print(f"[HYWorld2 QwenVL] Resized VLM preview {width}x{height} -> {new_size[0]}x{new_size[1]}")
    return resized


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


def _hyworld2_missing_memory_prerequisites(scene):
    scene = Path(scene)
    render_root = scene / "render_results"
    required_files = [
        scene / "meta_info.json",
        render_root / "global_pcd.ply",
        render_root / "sky_mask.png",
        render_root / "full_depth_prediction.pt",
        render_root / "pano_bank" / "cameras.json",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    pano_images = sorted((render_root / "pano_bank" / "images").glob("*.png"))
    pano_depths = sorted((render_root / "pano_bank" / "depths").glob("*.png"))
    if not pano_images:
        missing.append(str(render_root / "pano_bank" / "images" / "*.png"))
    if not pano_depths:
        missing.append(str(render_root / "pano_bank" / "depths" / "*.png"))
    return missing


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


def _quat_wxyz_multiply(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def _convert_trainer_gaussian_ply_to_worldmirror_basis(ply_path):
    path = Path(ply_path)
    if not path.exists():
        return str(path)

    with open(path, "rb") as handle:
        header = b""
        vertex_count = None
        props = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Invalid PLY header in {path}")
            header += line
            text = line.decode("ascii", "replace").strip()
            if text.startswith("element vertex"):
                vertex_count = int(text.split()[-1])
            elif text.startswith("property"):
                parts = text.split()
                props.append((parts[1], parts[2]))
            elif text == "end_header":
                data_offset = handle.tell()
                break

    if vertex_count is None:
        raise ValueError(f"PLY has no vertex count: {path}")
    prop_names = [name for _, name in props]
    required_xyz = {"x", "y", "z"}
    if not required_xyz.issubset(prop_names):
        return str(path)

    type_map = {
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "uchar": "u1",
        "uint8": "u1",
        "char": "i1",
        "int": "<i4",
        "uint": "<u4",
    }
    dtype = np.dtype([(name, type_map.get(kind, "<f4")) for kind, name in props])
    vertices = np.fromfile(path, dtype=dtype, count=vertex_count, offset=data_offset).copy()

    old_x = vertices["x"].copy()
    old_y = vertices["y"].copy()
    old_z = vertices["z"].copy()
    vertices["x"] = old_x
    vertices["y"] = -old_z
    vertices["z"] = old_y

    rot_names = ["rot_0", "rot_1", "rot_2", "rot_3"]
    if set(rot_names).issubset(prop_names):
        quats = np.stack([vertices[name] for name in rot_names], axis=-1).astype(np.float32)
        norms = np.linalg.norm(quats, axis=1, keepdims=True)
        valid = norms[:, 0] > 1e-8
        quats[valid] = quats[valid] / norms[valid]
        basis_quat = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0], dtype=np.float32)
        quats[valid] = _quat_wxyz_multiply(basis_quat, quats[valid])
        for idx, name in enumerate(rot_names):
            vertices[name] = quats[:, idx]

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as handle:
        handle.write(header)
        vertices.tofile(handle)
    os.replace(tmp_path, path)
    return str(path)


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


def _coerce_worldstereo_ref_index(ref_index, max_ref_index):
    if ref_index is None:
        value = 0
    elif isinstance(ref_index, torch.Tensor):
        value = int(ref_index.flatten()[0].detach().cpu().item()) if ref_index.numel() > 0 else 0
    elif isinstance(ref_index, (list, tuple)):
        value = int(ref_index[0]) if ref_index else 0
    else:
        value = int(ref_index)
    max_ref_index = max(0, int(max_ref_index))
    if max_ref_index < 19:
        value = int(round(float(value) * (float(max_ref_index) / 19.0)))
    return max(0, min(value, max_ref_index))


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
    max_ref_index = max(0, keyframe_indices.numel() - 2)
    pipeline_kwargs["ref_index"] = _coerce_worldstereo_ref_index(pipeline_kwargs.get("ref_index"), max_ref_index)
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


def _hy_log(node, message):
    print(f"[HYWorld2 {node}] {message}")


def _load_workspace_panorama(scene):
    image_path = scene / "panorama_sr.png"
    if not image_path.exists():
        image_path = scene / "panorama.png"
    if not image_path.exists():
        raise FileNotFoundError(f"HYWorld2 Trajectories requires panorama.png in workspace: {scene}")
    return Image.open(image_path).convert("RGB")


def _parse_scene_type(text):
    lowered = str(text or "").lower()
    if "outdoor" in lowered and "indoor" not in lowered:
        return "outdoor"
    if "indoor" in lowered:
        return "indoor"
    if "outdoor" in lowered:
        return "outdoor"
    return "indoor"


def _parse_qwenvl_objects(text):
    from hyworld2.worldgen.src.json_utils import loads_repaired

    raw = str(text or "").strip()
    try:
        parsed = loads_repaired(raw)
        if isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict):
            items = parsed.get("objects") or parsed.get("items") or []
        else:
            items = []
    except Exception:
        cleaned = raw.replace("[", "").replace("]", "").replace('"', "").replace("'", "").replace("```json", "").replace("```", "")
        items = []
        for line in cleaned.replace("\n", ",").split(","):
            item = line.strip(" -\t\r")
            if item:
                items.append(item)
    result = []
    seen = set()
    for item in items:
        item = str(item).strip().replace("-", "_")
        item = " ".join(item.split())
        if not item or len(item.split()) > 8:
            continue
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _ensure_trajectory_planner_context(
    workspace,
    scene_type,
    apply_nav_traj,
    force_vlm,
    qwen_model_id,
    qwen_quantization,
    qwen_attention_mode,
    qwen_device,
    qwen_max_new_tokens,
    qwen_max_image_edge,
    qwen_keep_model_loaded,
    qwen_cpu_offload,
):
    from hyworld2.worldgen.src.vlm_utils import get_qwen_caption_format
    from hyworld2.worldgen.src.navi_utils import get_navigation_instruction

    scene = Path(workspace["scene_dir"])
    panorama = _load_workspace_panorama(scene)
    pano_tensor = _pil_list_to_image_tensor([panorama])
    qwen = HYWorld2QwenVL()
    written = {}
    requested_scene_type = str(scene_type or "auto").lower()
    if requested_scene_type not in ("auto", "indoor", "outdoor"):
        requested_scene_type = "auto"
    print(f"[HYWorld2 Trajectories] Planner context: scene={scene}")
    print(f"[HYWorld2 Trajectories] Planner context: QwenVL model={qwen_model_id}, quantization={qwen_quantization}, device={qwen_device}, cpu_offload={bool(qwen_cpu_offload)}")

    meta_path = scene / "meta_info.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            meta.update(loaded)
    if requested_scene_type in ("indoor", "outdoor"):
        if str(meta.get("scene_type", "")).lower() != requested_scene_type:
            meta["scene_type"] = requested_scene_type
            with open(meta_path, "w", encoding="utf-8") as handle:
                json.dump(meta, handle, indent=2)
            written["meta_info"] = str(meta_path)
        print(f"[HYWorld2 Trajectories] Planner context: using manual scene_type={requested_scene_type}")
    elif str(meta.get("scene_type", "unknown")).lower() not in ("indoor", "outdoor"):
        print(f"[HYWorld2 Trajectories] Planner context: classifying scene_type from 480px preview -> {meta_path}")
        scene_type_tensor = _pil_list_to_image_tensor([_qwenvl_preview_image(panorama, max_image_edge=480)])
        text = qwen._generate(
            qwen_model_id,
            get_qwen_caption_format("env_cls"),
            images=scene_type_tensor,
            device=qwen_device,
            max_new_tokens=min(int(qwen_max_new_tokens), 64),
            max_image_edge=480,
            quantization=qwen_quantization,
            attention_mode=qwen_attention_mode,
            temperature=0.2,
            top_p=0.9,
            num_beams=1,
            repetition_penalty=1.0,
            keep_model_loaded=qwen_keep_model_loaded,
            cpu_offload=qwen_cpu_offload,
            seed=1024,
        )
        meta["scene_type"] = _parse_scene_type(text)
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        written["meta_info"] = str(meta_path)
        print(f"[HYWorld2 Trajectories] Planner context: scene_type={meta['scene_type']}")
    else:
        print(f"[HYWorld2 Trajectories] Planner context: reusing scene_type={meta.get('scene_type')} from {meta_path}")
    workspace["scene_type"] = str(meta.get("scene_type", workspace.get("scene_type", "unknown"))).lower()

    objects_path = scene / "objects.json"
    if apply_nav_traj and not objects_path.exists():
        print(f"[HYWorld2 Trajectories] Planner context: extracting navigation objects -> {objects_path}")
        text = qwen._generate(
            qwen_model_id,
            get_navigation_instruction(bool(force_vlm)),
            images=pano_tensor,
            device=qwen_device,
            max_new_tokens=int(qwen_max_new_tokens),
            max_image_edge=int(qwen_max_image_edge),
            quantization=qwen_quantization,
            attention_mode=qwen_attention_mode,
            temperature=0.2,
            top_p=0.9,
            num_beams=1,
            repetition_penalty=1.1,
            keep_model_loaded=qwen_keep_model_loaded,
            cpu_offload=qwen_cpu_offload,
            seed=1024,
        )
        objects = _parse_qwenvl_objects(text)
        with open(objects_path, "w", encoding="utf-8") as handle:
            json.dump(objects, handle, indent=2)
        written["objects"] = str(objects_path)
        print(f"[HYWorld2 Trajectories] Planner context: wrote {len(objects)} navigation object(s)")
    elif apply_nav_traj:
        print(f"[HYWorld2 Trajectories] Planner context: reusing navigation objects from {objects_path}")
    if not qwen_keep_model_loaded:
        HYWorld2QwenVL._clear_cache()
    return written


def _trajectory_scene_median_depth(scene):
    path = Path(scene) / "render_results" / "full_depth_prediction.pt"
    if not path.exists():
        return 1.0
    try:
        full_depth = torch.load(path, weights_only=False, map_location="cpu")
        distance = full_depth.get("distance") if isinstance(full_depth, dict) else None
        if distance is None:
            return 1.0
        values = distance.detach().float()
        values = values[torch.isfinite(values) & (values > 0)]
        if values.numel() == 0:
            return 1.0
        return float(torch.median(values).item())
    except Exception as exc:
        print(f"[HYWorld2 Trajectories] Could not read median depth: {exc}")
        return 1.0


def _anchor_camera_candidates(scene):
    render_root = Path(scene) / "render_results"
    paths = []
    for pattern in ("wonder*/traj*/camera.json", "reconstruct*/traj*/camera.json"):
        paths.extend(render_root.glob(pattern))
    result = []
    for path in sorted(paths):
        if path.parts[-3].startswith("wonder_scan_"):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            w2cs = np.asarray(data.get("extrinsic", []), dtype=np.float64)
            if w2cs.ndim != 3 or w2cs.shape[1:] != (4, 4) or len(w2cs) == 0:
                continue
            c2ws = np.linalg.inv(w2cs)
            position = c2ws[-1, :3, 3].astype(np.float64)
            result.append({"path": path, "data": data, "c2w": c2ws[-1], "position": position})
        except Exception as exc:
            print(f"[HYWorld2 Trajectories] Skipping anchor candidate {path}: {exc}")
    return result


def _make_anchor_scan_c2ws(anchor_c2w, nframe, yaw_degrees):
    position = anchor_c2w[:3, 3].astype(np.float64)
    base_forward = anchor_c2w[:3, 2].astype(np.float64)
    base_forward[2] = 0.0
    if np.linalg.norm(base_forward) < 1e-6:
        base_forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    base_forward = base_forward / np.linalg.norm(base_forward)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    frames = max(2, int(nframe))
    angles = np.linspace(0.0, np.deg2rad(float(yaw_degrees)), frames, endpoint=False)
    c2ws = []
    for angle in angles:
        c, s = np.cos(angle), np.sin(angle)
        forward = np.array([
            base_forward[0] * c - base_forward[1] * s,
            base_forward[0] * s + base_forward[1] * c,
            0.0,
        ], dtype=np.float64)
        forward = forward / max(np.linalg.norm(forward), 1e-8)
        cam_up = -up
        right = np.cross(cam_up, forward)
        right = right / max(np.linalg.norm(right), 1e-8)
        cam_up = np.cross(forward, right)
        cam_up = cam_up / max(np.linalg.norm(cam_up), 1e-8)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, 0] = right
        c2w[:3, 1] = cam_up
        c2w[:3, 2] = forward
        c2w[:3, 3] = position
        c2ws.append(c2w)
    return np.asarray(c2ws, dtype=np.float64)


def _write_anchor_scans(scene, topk, min_distance, min_separation, yaw_degrees, nframe):
    import cv2
    from hyworld2.worldgen.src.panorama_utils import split_panorama_image

    scene = Path(scene)
    candidates = _anchor_camera_candidates(scene)
    if not candidates:
        print("[HYWorld2 Trajectories] Anchor scan: no wonder/reconstruct camera candidates found")
        return []
    median_depth = max(_trajectory_scene_median_depth(scene), 1e-6)
    print(
        "[HYWorld2 Trajectories] Anchor scan: "
        f"candidates={len(candidates)}, topk={int(topk)}, median_depth={median_depth:.4f}, "
        f"min_distance={float(min_distance)}x, min_separation={float(min_separation)}x"
    )
    min_distance_abs = float(min_distance) * median_depth
    min_separation_abs = float(min_separation) * median_depth
    candidates.sort(key=lambda item: float(np.linalg.norm(item["position"][:2])), reverse=True)
    selected = []
    for candidate in candidates:
        pos = candidate["position"]
        if np.linalg.norm(pos[:2]) < min_distance_abs:
            continue
        if any(np.linalg.norm(pos[:2] - other["position"][:2]) < min_separation_abs for other in selected):
            continue
        selected.append(candidate)
        if len(selected) >= int(topk):
            break
    if not selected:
        print("[HYWorld2 Trajectories] Anchor scan: no candidates passed distance/separation filters")
        return []

    full_img = _load_workspace_panorama(scene)
    written = []
    for index, candidate in enumerate(selected):
        data = candidate["data"]
        image_w = int(data["width"])
        image_h = int(data["height"])
        K = np.asarray(data["intrinsic"][0], dtype=np.float64)
        c2ws = _make_anchor_scan_c2ws(candidate["c2w"], nframe, yaw_degrees)
        w2cs = np.linalg.inv(c2ws)
        dets = np.linalg.det(w2cs[:, :3, :3])
        up_z = c2ws[:, 2, 1]
        if np.any(dets < 0.9) or np.any(dets > 1.1) or np.any(up_z > -0.5):
            raise RuntimeError(
                "HYWorld2 anchor scan generated invalid camera orientation "
                f"for {candidate['path']}: det_range=({float(dets.min()):.4f}, {float(dets.max()):.4f}), "
                f"up_z_range=({float(up_z.min()):.4f}, {float(up_z.max()):.4f})."
            )
        K_pano = K.copy()
        K_pano[0, :] /= image_w
        K_pano[1, :] /= image_h
        start = split_panorama_image(np.array(full_img), w2cs[0:1], np.array([K_pano]), h=image_h, w=image_w, interp=cv2.INTER_AREA)[0]
        view_dir = scene / "render_results" / f"wonder_scan_{index}"
        traj_dir = view_dir / "traj0"
        _ensure_dir(traj_dir)
        Image.fromarray(start).save(view_dir / "start_frame.png")
        camera_info = {
            "id": index,
            "type": "anchor_scan",
            "source_camera": str(candidate["path"]),
            "width": image_w,
            "height": image_h,
            "intrinsic": [K.tolist()] * len(w2cs),
            "extrinsic": w2cs.tolist(),
            "anchor_position": candidate["position"].tolist(),
            "yaw_degrees": float(yaw_degrees),
        }
        with open(traj_dir / "camera.json", "w", encoding="utf-8") as handle:
            json.dump(camera_info, handle, indent=2)
        written.append(str(traj_dir / "camera.json"))
        print(f"[HYWorld2 Trajectories] Anchor scan: wrote {traj_dir / 'camera.json'}")
    return written


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
        _hy_log("Workspace", f"Stage 1/3: resolving workspace (mode={mode}, name={workspace_name})")
        if mode in ("load_existing", "resume") and str(scene_dir).strip():
            scene = Path(scene_dir)
        else:
            root = Path(root_dir) if str(root_dir).strip() else _output_root() / "hyworld2_worldgen"
            scene = root / _sanitize_name(workspace_name, "comfy_worldgen")
        _hy_log("Workspace", f"Workspace directory: {scene}")
        _ensure_dir(scene)
        _ensure_dir(scene / "render_results")
        if panorama is not None:
            _hy_log("Workspace", "Stage 2/3: saving input panorama")
            frames = _image_tensor_to_pil_list(panorama)
            if frames:
                frames[0].save(scene / "panorama.png")
                _hy_log("Workspace", f"Saved panorama: {scene / 'panorama.png'}")
        else:
            _hy_log("Workspace", "Stage 2/3: no panorama input connected; reusing workspace files")
        meta_path = scene / "meta_info.json"
        meta = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                meta.update(loaded)
        if scene_type != "unknown" or "scene_type" not in meta:
            meta["scene_type"] = scene_type
        _hy_log("Workspace", f"Stage 3/3: writing metadata scene_type={meta.get('scene_type', 'unknown')}")
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        workspace = {
            "scene_dir": str(scene),
            "render_results_dir": str(scene / "render_results"),
            "workspace_name": workspace_name,
            "result_name": result_name,
            "scene_type": meta.get("scene_type", "unknown"),
        }
        _hy_log("Workspace", "Workspace ready")
        return (workspace, _safe_json_dumps(workspace))


class HYWorld2QwenVL:
    _model_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        device_options = ["auto", "cpu", "mps"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        return {
            "required": {
                "workspace": ("HYWORLD2_WORKSPACE",),
                "mode": (["scene_objects", "trajectory_caption", "prompt_refine"], {"default": "trajectory_caption"}),
                "model_id": (_qwenvl_model_names(), {"default": HYWORLD2_QWENVL_DEFAULT}),
                "quantization": (HYWORLD2_QWENVL_QUANTIZATION, {"default": "None (FP16)"}),
                "attention_mode": (HYWORLD2_QWENVL_ATTENTION, {"default": "auto"}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "images": ("IMAGE",),
                "trajectory_set": ("HYWORLD2_TRAJECTORY_SET",),
                "device": (device_options, {"default": "auto"}),
                "max_new_tokens": ("INT", {"default": 256, "min": 16, "max": 4096, "step": 16}),
                "max_image_edge": ("INT", {"default": HYWORLD2_QWENVL_MAX_IMAGE_EDGE, "min": 128, "max": 4096, "step": 64}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.5, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 8}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.5, "max": 2.0, "step": 0.05}),
                "cpu_offload": ("BOOLEAN", {"default": True}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
                "write_results": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("HYWORLD2_LLM_CONTEXT", "STRING")
    RETURN_NAMES = ("llm_context", "text")
    FUNCTION = "run"
    CATEGORY = "VNCCS/HYWorld2"

    @classmethod
    def _clear_cache(cls, keep_signature=None):
        for signature, bundle in list(cls._model_cache.items()):
            if keep_signature is not None and signature == keep_signature:
                continue
            try:
                model = bundle.get("model")
                if model is not None:
                    model.cpu()
            except Exception:
                pass
            cls._model_cache.pop(signature, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_bundle(self, model_id, quantization="None (FP16)", attention_mode="auto", device="auto", keep_model_loaded=True, cpu_offload=False):
        try:
            from transformers import AutoProcessor, AutoTokenizer
            try:
                from transformers import AutoModelForImageTextToText as AutoModel
            except ImportError:
                from transformers import AutoModelForVision2Seq as AutoModel
        except Exception as exc:
            raise ImportError("QwenVL requires transformers with vision-language model support. Install project requirements.") from exc

        requested_device = str(device or "auto").strip()
        selected_device = _qwenvl_normalize_device(device)
        allow_cpu_offload = bool(cpu_offload) and requested_device == "auto" and torch.cuda.is_available()
        quant_cfg, dtype, is_fp8 = _qwenvl_quantization_config(model_id, quantization, cpu_offload=allow_cpu_offload)
        force_sdpa = is_fp8 or quant_cfg is not None
        attn_impl = _qwenvl_resolve_attention(attention_mode, force_sdpa=force_sdpa)
        signature = (str(model_id), str(quantization), attn_impl, selected_device, allow_cpu_offload)
        if keep_model_loaded and signature in self._model_cache:
            return self._model_cache[signature], signature

        self._clear_cache()
        model_path = _qwenvl_ensure_model(model_id)
        load_kwargs = {
            "attn_implementation": attn_impl,
            "use_safetensors": True,
            "trust_remote_code": True,
        }
        if is_fp8:
            load_kwargs["device_map"] = None
            load_kwargs["torch_dtype"] = "auto"
        else:
            if allow_cpu_offload:
                load_kwargs["device_map"] = "auto"
                max_memory = _qwenvl_auto_max_memory()
                if max_memory is not None:
                    load_kwargs["max_memory"] = max_memory
            else:
                load_kwargs["device_map"] = selected_device if selected_device not in ("cpu", "mps") else None
            if dtype is not None:
                load_kwargs["torch_dtype"] = dtype
            if quant_cfg is not None:
                load_kwargs["quantization_config"] = quant_cfg
        print(f"[HYWorld2 QwenVL] Loading {model_id} ({quantization}, attn={attn_impl}, device={selected_device}, cpu_offload={allow_cpu_offload})")
        print(f"[HYWorld2 QwenVL] Local model path: {model_path}")
        if "max_memory" in load_kwargs:
            print(f"[HYWorld2 QwenVL] Accelerate max_memory: {load_kwargs['max_memory']}")
        if quant_cfg is not None and str(quantization) == "8-bit (Balanced)" and allow_cpu_offload:
            print("[HYWorld2 QwenVL] 8-bit CPU offload enabled (llm_int8_enable_fp32_cpu_offload=True)")
        model = AutoModel.from_pretrained(model_path, **load_kwargs).eval()
        if selected_device in ("cpu", "mps") or is_fp8:
            model = model.to(selected_device)
        if hasattr(model, "hf_device_map"):
            print(f"[HYWorld2 QwenVL] Device map: {model.hf_device_map}")
        print("[HYWorld2 QwenVL] Loading processor/tokenizer")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("[HYWorld2 QwenVL] Model ready")
        bundle = {"model": model, "processor": processor, "tokenizer": tokenizer}
        if keep_model_loaded:
            self._model_cache[signature] = bundle
        return bundle, signature

    def _generate(
        self,
        model_id,
        prompt,
        images=None,
        device="auto",
        max_new_tokens=256,
        max_image_edge=HYWORLD2_QWENVL_MAX_IMAGE_EDGE,
        quantization="None (FP16)",
        attention_mode="auto",
        temperature=0.6,
        top_p=0.9,
        num_beams=1,
        repetition_penalty=1.2,
        keep_model_loaded=True,
        cpu_offload=False,
        seed=1,
    ):
        torch.manual_seed(int(seed))
        bundle, signature = self._load_bundle(
            model_id,
            quantization=quantization,
            attention_mode=attention_mode,
            device=device,
            keep_model_loaded=keep_model_loaded,
            cpu_offload=cpu_offload,
        )
        model = bundle["model"]
        processor = bundle["processor"]
        tokenizer = bundle["tokenizer"]
        pil_images = [_qwenvl_preview_image(image, max_image_edge=max_image_edge) for image in (_image_tensor_to_pil_list(images)[:8] if images is not None else [])]
        content = []
        for image in pil_images:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=pil_images or None, return_tensors="pt")
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}
        stop_tokens = [tokenizer.eos_token_id]
        if getattr(tokenizer, "eot_id", None) is not None:
            stop_tokens.append(tokenizer.eot_id)
        generate_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "repetition_penalty": float(repetition_penalty),
            "num_beams": int(num_beams),
            "eos_token_id": stop_tokens,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if int(num_beams) == 1:
            generate_kwargs.update({"do_sample": True, "temperature": float(temperature), "top_p": float(top_p)})
        else:
            generate_kwargs["do_sample"] = False
        with torch.no_grad():
            generated = model.generate(**inputs, **generate_kwargs)
        generated = generated[:, inputs["input_ids"].shape[1]:]
        result = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        if not keep_model_loaded:
            self._clear_cache(keep_signature=None)
        else:
            self._clear_cache(keep_signature=signature)
        return result

    def run(
        self,
        workspace,
        mode,
        model_id,
        quantization,
        attention_mode,
        prompt,
        images=None,
        trajectory_set=None,
        device="auto",
        max_new_tokens=256,
        max_image_edge=HYWORLD2_QWENVL_MAX_IMAGE_EDGE,
        temperature=0.6,
        top_p=0.9,
        num_beams=1,
        repetition_penalty=1.2,
        cpu_offload=True,
        keep_model_loaded=True,
        seed=1,
        write_results=True,
    ):
        scene = Path(workspace["scene_dir"])
        _hy_log("QwenVL", f"Stage 1/3: preparing prompt (mode={mode})")
        if not prompt.strip():
            if mode == "scene_objects":
                prompt = "Analyze this panoramic scene. Return concise JSON with scene_type, objects, navigable_areas, and visual_style."
            elif mode == "trajectory_caption":
                prompt = "Describe the visible trajectory render as a concise image generation prompt. Return only the prompt text."
            else:
                raise ValueError("prompt_refine requires a non-empty prompt; fallback prompts are disabled.")
        image_count = len(_image_tensor_to_pil_list(images)) if images is not None else 0
        traj_count = len((trajectory_set or {}).get("render_list", [])) if trajectory_set else 0
        _hy_log("QwenVL", f"Stage 2/3: generating text with model={model_id}, quantization={quantization}, device={device}, cpu_offload={bool(cpu_offload)}, images={image_count}, trajectories={traj_count}")
        text = self._generate(
            model_id,
            prompt,
            images=images,
            device=device,
            max_new_tokens=max_new_tokens,
            max_image_edge=max_image_edge,
            quantization=quantization,
            attention_mode=attention_mode,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            cpu_offload=bool(cpu_offload),
            keep_model_loaded=keep_model_loaded,
            seed=seed,
        )
        context = {"mode": mode, "text": text, "model_id": model_id, "quantization": quantization, "attention_mode": attention_mode}
        if write_results:
            _hy_log("QwenVL", "Stage 3/3: writing QwenVL outputs")
            if mode == "scene_objects":
                out_path = scene / "hyworld2_qwenvl_scene.json"
                try:
                    from hyworld2.worldgen.src.json_utils import loads_repaired

                    parsed = loads_repaired(text)
                except Exception:
                    parsed = {"raw": text}
                with open(out_path, "w", encoding="utf-8") as handle:
                    json.dump(parsed, handle, indent=2)
                context["scene_objects_path"] = str(out_path)
                _hy_log("QwenVL", f"Wrote scene context: {out_path}")
            elif mode == "trajectory_caption" and trajectory_set:
                render_list = trajectory_set.get("render_list", [])
                for render_path in render_list:
                    path = Path(render_path)
                    caption_path = path.parent / "traj_caption.json"
                    with open(caption_path, "w", encoding="utf-8") as handle:
                        json.dump({"prompt": text, "source": "HYWorld2 QwenVL"}, handle, indent=2)
                context["captions_written"] = len(render_list)
                _hy_log("QwenVL", f"Wrote {len(render_list)} trajectory caption file(s)")
        else:
            _hy_log("QwenVL", "Stage 3/3: write_results disabled")
        _hy_log("QwenVL", "QwenVL node complete")
        return (context, text)


class HYWorld2Trajectories:
    @classmethod
    def INPUT_TYPES(cls):
        device_options = ["auto", "cpu", "mps"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        return {
            "required": {
                "workspace": ("HYWORLD2_WORKSPACE",),
                "mode": (["generate_and_render_official", "reuse_existing"], {"default": "generate_and_render_official"}),
            },
            "optional": {
                "skip_existing": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1024, "min": 0, "max": 2**31 - 1}),
                "scene_type": (["auto", "indoor", "outdoor"], {"default": "auto"}),
                "apply_nav_traj": ("BOOLEAN", {"default": False}),
                "qwen_model_id": (_qwenvl_model_names(), {"default": HYWORLD2_QWENVL_DEFAULT}),
                "qwen_quantization": (HYWORLD2_QWENVL_QUANTIZATION, {"default": "None (FP16)"}),
                "qwen_device": (device_options, {"default": "auto"}),
                "qwen_cpu_offload": ("BOOLEAN", {"default": True}),
                "qwen_max_image_edge": ("INT", {"default": HYWORLD2_QWENVL_MAX_IMAGE_EDGE, "min": 128, "max": 4096, "step": 64}),
                "apply_anchor_scan": ("BOOLEAN", {"default": False}),
                "anchor_scan_topk": ("INT", {"default": 2, "min": 0, "max": 32}),
            },
        }

    RETURN_TYPES = ("HYWORLD2_TRAJECTORY_SET", "STRING")
    RETURN_NAMES = ("trajectory_set", "info")
    FUNCTION = "run"
    CATEGORY = "VNCCS/HYWorld2"

    @classmethod
    def IS_CHANGED(cls, workspace, mode, **kwargs):
        scene = Path(workspace["scene_dir"])
        watched = [
            scene / "panorama.png",
            scene / "meta_info.json",
            scene / "objects.json",
            scene / "render_results" / "global_pcd.ply",
            scene / "render_results" / "full_depth_prediction.pt",
            scene / "render_results" / "sky_mask.png",
            scene / "render_results" / "pano_bank" / "cameras.json",
        ]
        state = [str(scene), str(mode)]
        for path in watched:
            if path.exists():
                stat = path.stat()
                state.append(f"{path}:{stat.st_mtime_ns}:{stat.st_size}")
            else:
                state.append(f"{path}:missing")
        return "|".join(state)

    def _sort(self, workspace, generated=False, captions_written=None, anchor_scans_written=None, logs=None):
        from hyworld2.worldgen.src.data_utils import sort_trajs

        render_root = Path(workspace["scene_dir"]) / "render_results"
        render_list = sort_trajs(str(render_root))
        data = {
            "workspace": workspace,
            "render_list": render_list,
            "count": len(render_list),
            "generated": bool(generated),
            "captions_written": captions_written or [],
            "anchor_scans_written": anchor_scans_written or [],
            "logs": logs or [],
        }
        return (data, _safe_json_dumps(data))

    def run(
        self,
        workspace,
        mode,
        skip_existing=True,
        seed=1024,
        scene_type="auto",
        fov_x=120.0,
        fov_y=90.0,
        split_view_num=3,
        splitted_resolution=480,
        nframe=21,
        distance_threshold=0.1,
        obs_iteration_limit=3,
        rotation_deg=120.0,
        rotation_up=45.0,
        up_right=60.0,
        obs_decay=2 / 3,
        contract=8.0,
        skip_exist=True,
        apply_nav_traj=False,
        wonder_topk=3,
        recon_topk=5,
        move_dist=8.0,
        radius_threshold=4.0,
        min_angle_threshold=40.0,
        traj_sim_threshold=0.7,
        traj_sim_threshold_recon=0.7,
        apply_up_route=False,
        apply_recon_iteration=False,
        eloop_dist=0.25,
        force_vlm=False,
        cellSize=0.1,
        cellHeight=0.1,
        agentHeight=0.2,
        agentRadius=0.1,
        agentMaxClimb=0.1,
        maxSlope=30.0,
        roof_height_threshold=0.1,
        sam3_path=HYWORLD2_SAM3_REPO_ID,
        local_files_only=False,
        render_processes=0,
        caption_mode="qwenvl_missing",
        qwen_model_id=HYWORLD2_QWENVL_DEFAULT,
        qwen_quantization="None (FP16)",
        qwen_attention_mode="auto",
        qwen_device="auto",
        qwen_cpu_offload=True,
        qwen_max_image_edge=HYWORLD2_QWENVL_MAX_IMAGE_EDGE,
        qwen_max_new_tokens=256,
        qwen_keep_model_loaded=True,
        qwen_frame_count=4,
        apply_anchor_scan=False,
        anchor_scan_topk=2,
        anchor_scan_min_distance=1.0,
        anchor_scan_min_separation=0.75,
        anchor_scan_yaw_degrees=360.0,
    ):
        if mode == "reuse_existing":
            scene = Path(workspace["scene_dir"])
            missing = _hyworld2_missing_memory_prerequisites(scene)
            if missing:
                raise FileNotFoundError(
                    "HYWorld2 Trajectories cannot reuse this workspace because required base geometry is missing. "
                    "Run HYWorld2 Trajectories in generate_and_render_official mode first. Missing:\n"
                    + "\n".join(f"- {path}" for path in missing)
                )
            print(f"[HYWorld2 Trajectories] Reusing existing trajectory renders from {scene / 'render_results'}")
            return self._sort(workspace, generated=False)
        if mode != "generate_and_render_official":
            raise ValueError(f"Unsupported HYWorld2 Trajectories mode: {mode}")
        skip_exist = bool(skip_existing)

        scene = Path(workspace["scene_dir"])
        render_root = scene / "render_results"
        render_root_existed = render_root.exists()
        _ensure_dir(render_root)
        logs = []
        missing_geometry = _hyworld2_missing_memory_prerequisites(scene)
        if skip_exist and missing_geometry and render_root_existed:
            print("[HYWorld2 Trajectories] Existing render_results is incomplete; forcing geometry/trajectory rebuild.")
            for path in missing_geometry:
                print(f"[HYWorld2 Trajectories] Missing prerequisite: {path}")
            skip_exist = False
        print("[HYWorld2 Trajectories] Stage 0/5: official trajectory pipeline")
        print(f"[HYWorld2 Trajectories] Workspace: {scene}")
        print(f"[HYWorld2 Trajectories] SAM3 repo/path: {sam3_path or HYWORLD2_SAM3_REPO_ID}")

        print("[HYWorld2 Trajectories] Releasing Comfy models before local QwenVL planner")
        _release_model_memory("HYWorld2 Trajectories")
        print("[HYWorld2 Trajectories] Stage 1/5: preparing local QwenVL planner context")
        planner_written = _ensure_trajectory_planner_context(
            workspace,
            scene_type=scene_type,
            apply_nav_traj=bool(apply_nav_traj),
            force_vlm=bool(force_vlm),
            qwen_model_id=qwen_model_id,
            qwen_quantization=qwen_quantization,
            qwen_attention_mode=qwen_attention_mode,
            qwen_device=qwen_device,
            qwen_max_new_tokens=int(qwen_max_new_tokens),
            qwen_max_image_edge=int(qwen_max_image_edge),
            qwen_keep_model_loaded=bool(qwen_keep_model_loaded),
            qwen_cpu_offload=bool(qwen_cpu_offload),
        )
        HYWorld2QwenVL._clear_cache()
        print("[HYWorld2 Trajectories] Stage 1/5 complete: planner context ready")
        print("[HYWorld2 Trajectories] Releasing Comfy/Qwen models before geometry generation")
        _release_model_memory("HYWorld2 Trajectories")

        print("[HYWorld2 Trajectories] Stage 2/5: generating official camera trajectories")
        from hyworld2.worldgen import traj_generate, traj_render

        generate_config = Namespace(
            target_path=str(scene),
            fov_x=float(fov_x),
            fov_y=float(fov_y),
            seed=int(seed),
            split_view_num=int(split_view_num),
            splitted_resolution=int(splitted_resolution),
            nframe=int(nframe),
            distance_threshold=float(distance_threshold),
            obs_iteration_limit=int(obs_iteration_limit),
            rotation_deg=float(rotation_deg),
            rotation_up=float(rotation_up),
            up_right=float(up_right),
            obs_decay=float(obs_decay),
            contract=float(contract),
            skip_exist=bool(skip_exist),
            apply_nav_traj=bool(apply_nav_traj),
            wonder_topk=int(wonder_topk),
            recon_topk=int(recon_topk),
            move_dist=float(move_dist),
            radius_threshold=float(radius_threshold),
            min_angle_threshold=float(min_angle_threshold),
            traj_sim_threshold=float(traj_sim_threshold),
            traj_sim_threshold_recon=float(traj_sim_threshold_recon),
            apply_up_route=bool(apply_up_route),
            apply_recon_iteration=bool(apply_recon_iteration),
            eloop_dist=float(eloop_dist),
            force_vlm=bool(force_vlm),
            cellSize=float(cellSize),
            cellHeight=float(cellHeight),
            agentHeight=float(agentHeight),
            agentRadius=float(agentRadius),
            agentMaxClimb=float(agentMaxClimb),
            maxSlope=float(maxSlope),
            roof_height_threshold=float(roof_height_threshold),
            node_rank=0,
            node_size=1,
            sam3_path=sam3_path or HYWORLD2_SAM3_REPO_ID,
            local_files_only=bool(local_files_only),
        )
        traj_generate.run_traj_generate(generate_config)
        logs.append({"stage": "traj_generate", "mode": "native_api"})
        print("[HYWorld2 Trajectories] Stage 2/5 complete: camera trajectories generated")

        anchor_scans_written = []
        print("[HYWorld2 Trajectories] Stage 3/5: optional anchor scan")
        if bool(apply_anchor_scan) and int(anchor_scan_topk) > 0:
            anchor_scans_written = _write_anchor_scans(
                scene,
                topk=int(anchor_scan_topk),
                min_distance=float(anchor_scan_min_distance),
                min_separation=float(anchor_scan_min_separation),
                yaw_degrees=float(anchor_scan_yaw_degrees),
                nframe=int(nframe),
            )
        else:
            print("[HYWorld2 Trajectories] Anchor scan disabled")
        print(f"[HYWorld2 Trajectories] Stage 3/5 complete: {len(anchor_scans_written)} scan camera file(s)")

        if int(render_processes) not in (0, 1):
            print("[HYWorld2 Trajectories] render_processes is ignored in native mode; using one in-process renderer.")
        print("[HYWorld2 Trajectories] Stage 4/5: rendering trajectories natively with 1 process")
        render_config = Namespace(
            target_path=str(scene),
            seed=int(seed),
            node_rank=0,
            node_size=1,
            llm_addr="localhost",
            llm_port=8000,
            llm_name=HYWORLD2_QWENVL_MODELS.get(qwen_model_id, {}).get("repo_id", qwen_model_id),
            caption_workers=1,
            caption_sample_count=4,
            caption_max_tokens=256,
            disable_vlm_caption=True,
        )
        traj_render.run_traj_render(render_config, rank=0, world_size=1, local_rank=0)
        logs.append({"stage": "traj_render", "mode": "native_api", "world_size": 1})
        print("[HYWorld2 Trajectories] Stage 4/5 complete: render.mp4/render_mask.mp4 generated")

        from hyworld2.worldgen.src.data_utils import sort_trajs

        render_list = sort_trajs(str(render_root))
        print(f"[HYWorld2 Trajectories] Stage 5/5: local QwenVL captions for {len(render_list)} trajectory render(s)")
        captions_written = HYWorld2WorldExpansion()._ensure_captions(
            workspace,
            render_list,
            caption_mode,
            qwen_model_id,
            qwen_device,
            int(qwen_max_new_tokens),
            qwen_quantization,
            qwen_attention_mode,
            bool(qwen_keep_model_loaded),
            bool(qwen_cpu_offload),
            int(qwen_max_image_edge),
            int(qwen_frame_count),
        )
        HYWorld2QwenVL._clear_cache()
        print(f"[HYWorld2 Trajectories] Stage 5/5 complete: wrote {len(captions_written)} caption file(s)")
        data = {
            "workspace": workspace,
            "render_list": render_list,
            "count": len(render_list),
            "generated": True,
            "planner_context_written": planner_written,
            "captions_written": captions_written,
            "anchor_scans_written": anchor_scans_written,
            "logs": logs,
        }
        return (data, _safe_json_dumps(data))


class HYWorld2MemoryBank:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workspace": ("HYWORLD2_WORKSPACE",),
                "trajectory_set": ("HYWORLD2_TRAJECTORY_SET",),
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

    RETURN_TYPES = ("HYWORLD2_MEMORY_BANK", "STRING", "IMAGE")
    RETURN_NAMES = ("memory_bank", "info", "memory_images")
    FUNCTION = "run"
    CATEGORY = "VNCCS/HYWorld2"

    def run(self, workspace, trajectory_set, mode, image_width=0, image_height=0, nframe=0, max_reference=8, align_nframe=8, downsampled_pts=2_000_000, kb_anomaly_percentile=90.0):
        _hy_log("Memory Bank", f"Stage 1/3: initializing memory bank (mode={mode})")
        _ensure_worldgen_path()
        from hyworld2.worldgen.src.retrieval_wm import PanoramaMemoryBank

        scene = Path(workspace["scene_dir"])
        traj_workspace = trajectory_set.get("workspace", {}) if isinstance(trajectory_set, dict) else {}
        traj_scene = Path(traj_workspace.get("scene_dir", scene))
        if traj_scene.resolve() != scene.resolve():
            raise ValueError(
                "HYWorld2 Memory Bank got workspace and trajectory_set from different scene directories:\n"
                f"- workspace: {scene}\n"
                f"- trajectory_set: {traj_scene}"
            )
        if int(trajectory_set.get("count", 0)) <= 0:
            raise ValueError(
                "HYWorld2 Memory Bank requires a non-empty HYWorld2 Trajectories output. "
                "Connect HYWorld2 Trajectories.trajectory_set and run generate_and_render_official first."
            )
        missing = _hyworld2_missing_memory_prerequisites(scene)
        if missing:
            raise FileNotFoundError(
                "HYWorld2 Memory Bank requires completed HYWorld2 Trajectories base geometry before initialization. "
                "Connect HYWorld2 Trajectories.trajectory_set to this node so Comfy executes trajectories before Memory Bank. "
                "Missing:\n"
                + "\n".join(f"- {path}" for path in missing)
            )
        if image_width <= 0 or image_height <= 0:
            _hy_log("Memory Bank", "Stage 2/3: resolving image size from trajectory start frame or panorama")
            from imagesize import get as image_size

            start_frames = sorted((scene / "render_results").glob("*/start_frame.png"))
            if start_frames:
                image_width, image_height = image_size(str(start_frames[0]))
                _hy_log("Memory Bank", f"Using start frame size {image_width}x{image_height}: {start_frames[0]}")
            else:
                pano = scene / "panorama.png"
                image_width, image_height = image_size(str(pano))
                _hy_log("Memory Bank", f"Using panorama size {image_width}x{image_height}: {pano}")
        else:
            _hy_log("Memory Bank", f"Stage 2/3: using explicit image size {image_width}x{image_height}")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        _hy_log("Memory Bank", f"Stage 3/3: constructing PanoramaMemoryBank on {device} (pts_num={int(downsampled_pts)})")
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
        memory_images = _pil_list_to_image_tensor(getattr(bank, "ref_frames", []))
        if memory_images.numel() == 0:
            memory_images = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        info = {
            "scene_dir": str(scene),
            "device": str(device),
            "memory_size": int(bank.mem_size),
            "results_path": bank.results_path,
            "memory_image_count": int(memory_images.shape[0]),
            "memory_frame_names_preview": list(getattr(bank, "fnames", []))[:16],
        }
        _hy_log("Memory Bank", f"Memory bank ready: memory_size={int(bank.mem_size)}, results_path={bank.results_path}")
        return (state, _safe_json_dumps(info), memory_images)


class HYWorld2WorldExpansion:
    @classmethod
    def INPUT_TYPES(cls):
        device_options = ["auto", "cpu", "mps"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        return {
            "required": {
                "workspace": ("HYWORLD2_WORKSPACE",),
                "memory_bank": ("HYWORLD2_MEMORY_BANK",),
                "trajectory_set": ("HYWORLD2_TRAJECTORY_SET",),
                "model": ("WORLDSTEREO_MODEL",),
            },
            "optional": {
                "caption_mode": (["qwenvl_missing", "qwenvl_overwrite", "existing_files_only"], {"default": "qwenvl_missing"}),
                "qwen_model_id": (_qwenvl_model_names(), {"default": HYWORLD2_QWENVL_DEFAULT}),
                "qwen_quantization": (HYWORLD2_QWENVL_QUANTIZATION, {"default": "None (FP16)"}),
                "qwen_attention_mode": (HYWORLD2_QWENVL_ATTENTION, {"default": "auto"}),
                "qwen_device": (device_options, {"default": "auto"}),
                "qwen_cpu_offload": ("BOOLEAN", {"default": True}),
                "qwen_max_image_edge": ("INT", {"default": HYWORLD2_QWENVL_MAX_IMAGE_EDGE, "min": 128, "max": 4096, "step": 64}),
                "qwen_max_new_tokens": ("INT", {"default": 192, "min": 16, "max": 2048, "step": 16}),
                "qwen_keep_model_loaded": ("BOOLEAN", {"default": True}),
                "qwen_frame_count": ("INT", {"default": 4, "min": 1, "max": 16}),
                "seed": ("INT", {"default": 1024, "min": 0, "max": 2**31 - 1}),
                "skip_existing": ("BOOLEAN", {"default": True}),
                "max_trajectories": ("INT", {"default": 0, "min": 0, "max": 100000}),
            },
        }

    RETURN_TYPES = ("HYWORLD2_MEMORY_BANK", "STRING")
    RETURN_NAMES = ("memory_bank", "info")
    FUNCTION = "run"
    CATEGORY = "VNCCS/HYWorld2"

    def _ensure_captions(
        self,
        workspace,
        render_list,
        caption_mode,
        qwen_model_id,
        qwen_device,
        qwen_max_new_tokens,
        qwen_quantization="None (FP16)",
        qwen_attention_mode="auto",
        qwen_keep_model_loaded=True,
        qwen_cpu_offload=True,
        qwen_max_image_edge=HYWORLD2_QWENVL_MAX_IMAGE_EDGE,
        qwen_frame_count=4,
    ):
        if caption_mode == "existing_files_only":
            _hy_log("World Expansion", "Caption stage: existing_files_only, not generating captions")
            return []
        qwen = HYWorld2QwenVL()
        written = []
        _hy_log("World Expansion", f"Caption stage: mode={caption_mode}, trajectories={len(render_list)}, model={qwen_model_id}, cpu_offload={bool(qwen_cpu_offload)}")
        for render_path in render_list:
            traj_dir = Path(render_path).parent
            caption_path = traj_dir / "traj_caption.json"
            if caption_path.exists() and caption_mode != "qwenvl_overwrite":
                _hy_log("World Expansion", f"Caption stage: reusing {caption_path}")
                continue
            _hy_log("World Expansion", f"Caption stage: generating caption for {render_path}")
            frames = _load_video_frames(render_path)
            sample = []
            if frames:
                sample = [frames[0]]
                if len(frames) > 2:
                    sample.append(frames[len(frames) // 2])
                if len(frames) > 1:
                    sample.append(frames[-1])
                if int(qwen_frame_count) > len(sample):
                    idx = np.linspace(0, len(frames) - 1, min(int(qwen_frame_count), len(frames)), dtype=int)
                    sample = [frames[i] for i in idx]
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
                images=_pil_list_to_image_tensor(sample[: max(1, int(qwen_frame_count))]),
                device=qwen_device,
                max_new_tokens=qwen_max_new_tokens,
                max_image_edge=int(qwen_max_image_edge),
                quantization=qwen_quantization,
                attention_mode=qwen_attention_mode,
                temperature=0.6,
                top_p=0.9,
                num_beams=1,
                repetition_penalty=1.2,
                keep_model_loaded=qwen_keep_model_loaded,
                cpu_offload=qwen_cpu_offload,
                seed=1,
            )
            if not text.strip():
                raise RuntimeError(f"QwenVL returned an empty caption for {render_path}")
            with open(caption_path, "w", encoding="utf-8") as handle:
                json.dump({"prompt": text.strip(), "source": "HYWorld2 World Expansion QwenVL"}, handle, indent=2)
            written.append(str(caption_path))
            _hy_log("World Expansion", f"Caption stage: wrote {caption_path}")
        return written

    def run(
        self,
        workspace,
        memory_bank,
        trajectory_set,
        model,
        caption_mode="qwenvl_missing",
        qwen_model_id=HYWORLD2_QWENVL_DEFAULT,
        qwen_quantization="None (FP16)",
        qwen_attention_mode="auto",
        qwen_device="auto",
        qwen_cpu_offload=True,
        qwen_max_image_edge=HYWORLD2_QWENVL_MAX_IMAGE_EDGE,
        qwen_max_new_tokens=192,
        qwen_keep_model_loaded=True,
        qwen_frame_count=4,
        seed=1024,
        skip_existing=True,
        max_trajectories=0,
    ):
        _hy_log("World Expansion", "Stage 1/6: preparing WorldStereo memory expansion")
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
        _hy_log("World Expansion", f"Stage 2/6: trajectory count={len(render_list)}, device={device}, model_type={model_type}")
        _hy_log("World Expansion", "Stage 3/6: ensuring trajectory captions")
        captions_written = self._ensure_captions(
            workspace,
            render_list,
            caption_mode,
            qwen_model_id,
            qwen_device,
            int(qwen_max_new_tokens),
            qwen_quantization,
            qwen_attention_mode,
            bool(qwen_keep_model_loaded),
            bool(qwen_cpu_offload),
            int(qwen_max_image_edge),
            int(qwen_frame_count),
        )
        _hy_log("World Expansion", f"Stage 3/6 complete: captions_written={len(captions_written)}")
        _hy_log("World Expansion", "Stage 4/6: encoding prompt cache")
        prompt_cache = _build_prompt_cache(model, workspace, render_list, model_type, device)
        _hy_log("World Expansion", f"Stage 4/6 complete: cached {len(prompt_cache)} prompt embedding set(s)")
        generator = torch.Generator(device=device).manual_seed(int(seed))
        autocast_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        completed = []
        _hy_log("World Expansion", "Stage 5/6: generating trajectory videos and updating memory")
        for render_path in render_list:
            render_parts = Path(render_path).parts
            view_id, traj_id = render_parts[-3], render_parts[-2]
            traj_dir = Path(workspace["scene_dir"]) / "render_results" / view_id / traj_id
            result_path = traj_dir / f"{model_type}_result.mp4"
            _hy_log("World Expansion", f"Trajectory {len(completed)+1}/{len(render_list)}: {view_id}/{traj_id}")
            camera_data = json.load(open(traj_dir / "camera.json", "r", encoding="utf-8"))
            tar_w2cs = torch.from_numpy(np.asarray(camera_data["extrinsic"], dtype=np.float32)).to(device)
            tar_Ks = torch.from_numpy(np.asarray(camera_data["intrinsic"], dtype=np.float32)).to(device)
            if skip_existing and result_path.exists():
                _hy_log("World Expansion", f"Trajectory {view_id}/{traj_id}: reusing existing result {result_path}")
                frames = _load_video_frames(result_path)
                update_w2cs, update_Ks = _sample_camera_tensors_to_frame_count(tar_w2cs, tar_Ks, len(frames))
                bank.update_memory(frames, update_w2cs, update_Ks, view_id=view_id, traj_id=traj_id)
                completed.append(str(result_path))
                continue
            _hy_log("World Expansion", f"Trajectory {view_id}/{traj_id}: retrieving references from memory bank")
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
                _hy_log("World Expansion", f"Trajectory {view_id}/{traj_id}: running WorldStereo generation")
                output = pipeline(**pipeline_kwargs).frames[0].float()
            frames_np = output.permute(0, 2, 3, 1).detach().cpu().clamp(0, 1).numpy()
            _export_video(frames_np, result_path, fps=16)
            gen_frames = _load_video_frames(result_path)
            update_w2cs, update_Ks = _sample_camera_tensors_to_frame_count(tar_w2cs, tar_Ks, len(gen_frames))
            bank.update_memory(gen_frames, update_w2cs, update_Ks, view_id=view_id, traj_id=traj_id)
            completed.append(str(result_path))
            _hy_log("World Expansion", f"Trajectory {view_id}/{traj_id}: wrote {result_path} and updated memory")
            del output, pipeline_kwargs, meta_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        memory_bank["bank"] = bank
        del pipeline
        _hy_log("World Expansion", "Stage 6/6: releasing model memory")
        _release_model_memory("HYWorld2 World Expansion")
        _hy_log("World Expansion", f"World expansion complete: completed={len(completed)}")
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
        _hy_log("Prepare WorldMirror Batch", "Stage 1/4: preparing WorldMirror export batch")
        bank = memory_bank["bank"]
        world_mirror_dir = Path(bank.root_path) / "render_results" / bank.results_path / "world_mirror_data"
        render_root = (Path(bank.root_path) / "render_results" / bank.results_path).resolve()
        world_mirror_resolved = world_mirror_dir.resolve()
        if world_mirror_dir.exists():
            if render_root not in world_mirror_resolved.parents:
                raise RuntimeError(f"Refusing to clear unexpected WorldMirror directory: {world_mirror_dir}")
            shutil.rmtree(world_mirror_dir)
        images_dir = _ensure_dir(world_mirror_dir / "images")
        _hy_log("Prepare WorldMirror Batch", f"WorldMirror directory: {world_mirror_dir}")
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
        _hy_log("Prepare WorldMirror Batch", f"Stage 2/4: exporting {len(entries)} reference frame(s)")
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
        _hy_log("Prepare WorldMirror Batch", "Stage 3/4: writing cameras.json and name_map.json")
        with open(world_mirror_dir / "cameras.json", "w", encoding="utf-8") as handle:
            json.dump(cameras, handle, indent=2)
        with open(world_mirror_dir / "name_map.json", "w", encoding="utf-8") as handle:
            json.dump(name_map, handle, indent=2)
        bank.world_mirror_dir = str(world_mirror_dir)
        bank.name_map = name_map
        image_tensor = _pil_list_to_image_tensor(images)
        camera_poses = torch.stack(poses).float()
        camera_intrinsics = torch.stack(intrs).float()
        batch = {
            "memory_bank": memory_bank,
            "world_mirror_dir": str(world_mirror_dir),
            "name_map": name_map,
            "images": image_tensor,
            "camera_poses": camera_poses,
            "camera_intrinsics": camera_intrinsics,
        }
        _hy_log("Prepare WorldMirror Batch", f"Stage 4/4 complete: images={len(images)}, world_mirror_dir={world_mirror_dir}")
        return (image_tensor, camera_poses, camera_intrinsics, batch, _safe_json_dumps({"frames": len(images), "world_mirror_dir": world_mirror_dir}))


class HYWorld2MemoryAlignment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "worldmirror_batch": ("HYWORLD2_WORLDMIRROR_BATCH",),
                "mode": (["consume_worldmirror_depths", "align_and_export", "bypass"], {"default": "align_and_export"}),
            },
            "optional": {
                "raw_splats": ("VNCCS_SPLAT",),
                "ply_data": ("PLY_DATA",),
                "downsampled_pts": ("INT", {"default": 2_000_000, "min": 1, "max": 50_000_000, "step": 100000}),
                "debug_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("HYWORLD2_MEMORY_BANK", "STRING", "STRING")
    RETURN_NAMES = ("memory_bank", "aligned_ply", "info")
    FUNCTION = "run"
    CATEGORY = "VNCCS/HYWorld2"

    def run(self, worldmirror_batch, mode, raw_splats=None, ply_data=None, downsampled_pts=2_000_000, debug_mode=False):
        _hy_log("Memory Alignment", f"Stage 1/4: consuming WorldMirror depths (mode={mode})")
        memory_bank = worldmirror_batch["memory_bank"]
        bank = memory_bank["bank"]
        world_mirror_dir = Path(worldmirror_batch["world_mirror_dir"])
        depth_dir = _ensure_dir(world_mirror_dir / "results" / "depth")
        depths, depth_source = _raw_worldmirror_depths_to_numpy(raw_splats)
        if not depths and mode != "bypass":
            raise ValueError(
                "HYWorld2 Memory Alignment requires connected raw_splats with metric depth: raw_splats.gs_depth or raw_splats.depth."
            )
        _hy_log("Memory Alignment", f"Writing {len(depths)} depth map(s) to {depth_dir} (source={depth_source})")
        for index, depth in enumerate(depths):
            np.save(depth_dir / f"depth_{index:04d}.npy", depth)
        if mode == "align_and_export":
            _hy_log("Memory Alignment", "Stage 2/4: running memory bank alignment")
            _ensure_single_process_dist(bank)
            bank.alignment(debug_mode=bool(debug_mode))
            _hy_log("Memory Alignment", "Stage 3/4: exporting aligned/global point clouds")
            export_dir = Path(bank.root_path) / "render_results" / bank.results_path
            _ensure_dir(export_dir)
            bank.export_pcd(str(export_dir), N_points=int(downsampled_pts))
            aligned = str(export_dir / "aligned_pcd.ply")
            bypass_source_points = 0
        elif mode == "bypass":
            _hy_log("Memory Alignment", "Stage 2/4: bypassing alignment and exporting source point clouds")
            aligned, bypass_source_points = _export_bypass_memory_bank_pcds(bank, ply_data, raw_splats, downsampled_pts)
        else:
            _hy_log("Memory Alignment", "Stage 2/4: consume depths only; alignment/export skipped")
            aligned = ""
            bypass_source_points = 0
        memory_bank["bank"] = bank
        _hy_log("Memory Alignment", f"Stage 4/4 complete: aligned_ply={aligned or '<none>'}")
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
        _hy_log("GS Data", f"Stage 1/3: preparing GS dataset (mode={mode})")
        scene = Path(workspace["scene_dir"])
        gs_dir = scene / _sanitize_name(out_name, "gs_data")
        _hy_log("GS Data", f"Scene: {scene}")
        _hy_log("GS Data", f"GS data directory: {gs_dir}")
        if mode == "validate":
            _hy_log("GS Data", "Stage 2/3: validating required files")
            required = [gs_dir / "cameras.json", gs_dir / "points.ply", gs_dir / "images"]
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise FileNotFoundError(f"HYWorld2 GS data missing required files: {missing}")
            _hy_log("GS Data", "Stage 3/3 complete: dataset is valid")
            return ({"workspace": workspace, "gs_data_dir": str(gs_dir)}, _safe_json_dumps({"valid": True, "gs_data_dir": gs_dir}))
        if mode == "repair_metadata":
            _hy_log("GS Data", "Stage 2/3: repairing metadata")
            meta = gs_dir / "meta_info.json"
            if not meta.exists():
                with open(meta, "w", encoding="utf-8") as handle:
                    json.dump({"scene_type": workspace.get("scene_type", "unknown")}, handle, indent=2)
                _hy_log("GS Data", f"Wrote {meta}")
            _hy_log("GS Data", "Stage 3/3 complete: metadata ready")
            return ({"workspace": workspace, "gs_data_dir": str(gs_dir)}, _safe_json_dumps({"repaired": True, "gs_data_dir": gs_dir}))
        _ensure_worldgen_path()
        import hyworld2.worldgen.gen_gs_data as gen_gs_data

        if not hasattr(gen_gs_data, "run_gen_gs_data"):
            raise RuntimeError("gen_gs_data.py must expose run_gen_gs_data for native node execution.")
        _hy_log("GS Data", "Stage 2/3: running gen_gs_data")
        _hy_log("GS Data", f"Options: result_name={result_name or workspace.get('result_name', 'worldstereo-memory-dmd')}, save_normal={bool(save_normal)}, split_sky={bool(split_sky)}, split_align={bool(split_align)}")
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
        _hy_log("GS Data", f"Stage 3/3 complete: output_path={gs_dir}")
        return ({"workspace": workspace, "gs_data_dir": str(gs_dir)}, _safe_json_dumps(result))


class HYWorld2Train3DGS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gs_data": ("HYWORLD2_GS_DATA",),
            },
            "optional": {
                "max_steps": ("INT", {"default": 5000, "min": 1, "max": 100000, "step": 100}),
                "save_steps": ("STRING", {"default": "4000,5000,6000,8000,10000"}),
                "eval_steps": ("STRING", {"default": "1000,2000,3000,4000,5000,6000,7000,8000,9000,10000"}),
                "ply_steps": ("STRING", {"default": "4000,5000,6000,8000,10000"}),
                "downsample_pts_num": ("INT", {"default": 1_000_000, "min": 1, "max": 50_000_000, "step": 100000}),
                "save_ply": ("BOOLEAN", {"default": True}),
                "disable_video": ("BOOLEAN", {"default": True}),
                "disable_viewer": ("BOOLEAN", {"default": True}),
                "depth_loss": ("BOOLEAN", {"default": False}),
                "normal_loss": ("BOOLEAN", {"default": False}),
                "sky_depth_from_pcd": ("BOOLEAN", {"default": False}),
                "use_scale_regularization": ("BOOLEAN", {"default": False}),
                "use_mask_gaussian": ("BOOLEAN", {"default": False}),
                "mask_export_stochastic": ("BOOLEAN", {"default": True}),
                "do_prune": ("BOOLEAN", {"default": False}),
                "prune_opacity_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}),
                "antialiased": ("BOOLEAN", {"default": False}),
                "normalize_world_space": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "TENSOR", "TENSOR", "STRING", "STRING")
    RETURN_NAMES = ("ply_path", "camera_poses", "camera_intrinsics", "train_dir", "info")
    FUNCTION = "run"
    CATEGORY = "VNCCS/HYWorld2"
    OUTPUT_NODE = True

    def run(self, gs_data, max_steps=5000, save_steps="4000,5000,6000,8000,10000", eval_steps="1000,2000,3000,4000,5000,6000,7000,8000,9000,10000", ply_steps="4000,5000,6000,8000,10000", downsample_pts_num=1_000_000, save_ply=True, disable_video=True, disable_viewer=True, depth_loss=False, normal_loss=False, sky_depth_from_pcd=False, use_scale_regularization=False, use_mask_gaussian=False, mask_export_stochastic=True, do_prune=False, prune_opacity_threshold=0.01, antialiased=False, normalize_world_space=True):
        _hy_log("Train 3DGS", "Stage 1/5: preparing trainer config")
        _ensure_worldgen_path()
        import hyworld2.worldgen.world_gs_trainer as trainer
        from gsplat.strategy import DefaultStrategy

        data_dir = Path(gs_data["gs_data_dir"])
        out_dir = data_dir.parent / "gs_results"
        _hy_log("Train 3DGS", f"Input data_dir: {data_dir}")
        _hy_log("Train 3DGS", f"Output train_dir: {out_dir}")
        _reset_dir(out_dir, "HYWorld2 train_dir")
        _ensure_scene_type_meta(data_dir)
        strategy = DefaultStrategy(verbose=True)
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
        cfg.do_prune = bool(do_prune)
        cfg.prune_opacity_threshold = float(prune_opacity_threshold)
        cfg.antialiased = bool(antialiased)
        cfg.no_normalize = not bool(normalize_world_space)
        _hy_log(
            "Train 3DGS",
            "Config: "
            f"max_steps={cfg.max_steps}, downsample_pts_num={cfg.downsample_pts_num}, save_ply={cfg.save_ply}, "
            f"depth_loss={cfg.depth_loss}, normal_loss={cfg.normal_loss}, do_prune={cfg.do_prune}, "
            f"normalize_world_space={bool(normalize_world_space)}"
        )
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
            "do_prune": bool(cfg.do_prune),
            "prune_opacity_threshold": float(cfg.prune_opacity_threshold),
            "antialiased": bool(cfg.antialiased),
            "normalize_world_space": bool(normalize_world_space),
            "lpips_net": cfg.lpips_net,
            "in_process": True,
        }
        with open(out_dir / "train_command.json", "w", encoding="utf-8") as handle:
            json.dump(command_info, handle, indent=2)
        _hy_log("Train 3DGS", f"Stage 2/5: wrote train command metadata: {out_dir / 'train_command.json'}")
        if depth_loss and not cfg.depth_loss:
            print(f"[HYWorld2 Train 3DGS] depth_loss requested but valid metric float16-packed depths are missing under {data_dir / 'depths'}; disabling depth_loss.")
        if normal_loss and not cfg.normal_loss:
            print(f"[HYWorld2 Train 3DGS] normal_loss requested but normals are missing/constant under {data_dir / 'normals'}; disabling normal_loss.")
        if sky_depth_from_pcd and not cfg.sky_depth_from_pcd:
            print("[HYWorld2 Train 3DGS] sky_depth_from_pcd requested but depth/normal inputs are not usable; disabling sky_depth_from_pcd.")
        _hy_log("Train 3DGS", "Stage 3/5: running 3DGS trainer")
        with torch.inference_mode(False), torch.enable_grad():
            trainer.main(0, 0, 1, cfg)
        _hy_log("Train 3DGS", "Stage 4/5: locating and converting latest PLY")
        ply_path = _find_latest_ply(out_dir)
        if ply_path:
            _hy_log("Train 3DGS", f"Latest PLY before basis conversion: {ply_path}")
            ply_path = _convert_trainer_gaussian_ply_to_worldmirror_basis(ply_path)
            _hy_log("Train 3DGS", f"PLY ready: {ply_path}")
        else:
            _hy_log("Train 3DGS", "No PLY file found after training")
        camera_json = out_dir / "ply" / "trainer_cameras.json"
        if not camera_json.exists():
            candidates = sorted((out_dir / "ply").glob("trainer_cameras_*.json")) if (out_dir / "ply").exists() else []
            camera_json = candidates[-1] if candidates else data_dir / "cameras.json"
        poses, intrs = _load_camera_tensors_from_json(camera_json) if camera_json.exists() else (torch.empty((0, 4, 4)), torch.empty((0, 3, 3)))
        if poses.numel() > 0:
            poses = torch.stack([_worldstereo_c2w_to_worldmirror_c2w(pose) for pose in poses]).float()
        _hy_log("Train 3DGS", f"Stage 5/5 complete: cameras={int(poses.shape[0]) if poses.ndim >= 1 else 0}, camera_json={camera_json}")
        info = {
            "ply_path": ply_path,
            "train_dir": str(out_dir),
            "camera_json": str(camera_json) if camera_json.exists() else "",
            "camera_pose_basis": "worldmirror_c2w",
            "ply_basis": "worldmirror",
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
