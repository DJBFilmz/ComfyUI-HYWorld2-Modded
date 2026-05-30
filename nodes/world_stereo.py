"""
WorldStereo ComfyUI nodes — camera-guided video generation.

Nodes:
  - VNCCS_LoadWorldStereoModel   — download and load WorldStereo + MoGe models
  - VNCCS_CameraTrajectoryBuilder — build camera trajectory tensors
  - VNCCS_WorldStereoGenerate    — run WorldStereo inference (Task 5 stub)
"""

import os
import sys
import json
import math
import hashlib
import importlib
from contextlib import contextmanager
import numpy as np
import torch

# Get the absolute path of ComfyUI-HYWorld2 and its worldstereo subdirectory
node_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
worldstereo_dir = os.path.join(node_dir, "worldstereo")

# Inject the paths to the front of Python's search path to allow importing camera_utils
for path in [node_dir, worldstereo_dir, os.path.join(worldstereo_dir, "src")]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)
# ─── PATH RESOLUTION PATCH END ───

# ── nodes/ -> repo root ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── worldstereo camera utils ───────────────────────────────────────────────────
_WORLDSTEREO_PATH = os.path.join(PROJECT_ROOT, "worldstereo")
if _WORLDSTEREO_PATH not in sys.path:
    sys.path.insert(0, _WORLDSTEREO_PATH)


def _prepend_package_search_path(package_name: str, search_path: str):
    """Allow WorldStereo's local namespace packages to coexist with ComfyUI modules."""
    if not os.path.isdir(search_path):
        return

    module = sys.modules.get(package_name)
    if module is None:
        import types
        module = types.ModuleType(package_name)
        sys.modules[package_name] = module

    search_path = os.path.abspath(search_path)
    existing_paths = list(getattr(module, "__path__", []) or [])
    normalized = {os.path.normcase(os.path.abspath(p)) for p in existing_paths}
    if os.path.normcase(search_path) not in normalized:
        existing_paths.insert(0, search_path)

    module.__path__ = existing_paths
    module.__package__ = package_name
    spec = getattr(module, "__spec__", None)
    if spec is not None:
        spec.submodule_search_locations = existing_paths


def _import_worldstereo_class(base_dir: str = PROJECT_ROOT):
    worldstereo_path = _prepare_worldstereo_import_paths(base_dir)
    try:
        from models.worldstereo_wrapper import WorldStereo
        return WorldStereo
    except ModuleNotFoundError as e:
        if e.name not in ("models.worldstereo_wrapper", "models"):
            raise

    wrapper_path = os.path.join(worldstereo_path, "models", "worldstereo_wrapper.py")
    if not os.path.exists(wrapper_path):
        raise FileNotFoundError(f"worldstereo_wrapper.py not found at {wrapper_path!r}")

    spec = importlib.util.spec_from_file_location("models.worldstereo_wrapper", wrapper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {wrapper_path!r}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["models.worldstereo_wrapper"] = module
    spec.loader.exec_module(module)
    return module.WorldStereo


def _prepare_worldstereo_import_paths(base_dir: str = PROJECT_ROOT) -> str:
    worldstereo_path = None
    for name in os.listdir(base_dir):
        if name.lower() == "worldstereo":
            worldstereo_path = os.path.join(base_dir, name)
            break
    if worldstereo_path is None:
        worldstereo_path = os.path.join(base_dir, "worldstereo")

    world_models_path = os.path.join(worldstereo_path, "models")
    world_src_path = os.path.join(worldstereo_path, "src")

    for path in (base_dir, worldstereo_path, world_src_path):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)

    _prepend_package_search_path("models", world_models_path)
    _prepend_package_search_path("src", world_src_path)
    importlib.invalidate_caches()
    return worldstereo_path


_prepare_worldstereo_import_paths()


def _fallback_camera_backward_forward(c2w, distance):
    c2w[:3, 3:4] = (c2w @ np.array([0, 0, distance, 1.0], dtype=np.float32).reshape(4, 1))[:3]
    return c2w


def _fallback_camera_left_right(c2w, distance):
    c2w[:3, 3:4] = (c2w @ np.array([distance, 0, 0, 1.0], dtype=np.float32).reshape(4, 1))[:3]
    return c2w


def _camera_up_down(c2w, distance):
    c2w[:3, 3:4] = (c2w @ np.array([0, distance, 0, 1.0], dtype=np.float32).reshape(4, 1))[:3]
    return c2w


def _fallback_native_camera_rotation(c2w, medium_depth, phi, theta):
    R_elevation = np.array([[1, 0, 0, 0],
                            [0, np.cos(theta), -np.sin(theta), 0],
                            [0, np.sin(theta), np.cos(theta), 0],
                            [0, 0, 0, 1]], dtype=np.float32)
    R_azimuth = np.array([[np.cos(phi), 0, np.sin(phi), 0],
                          [0, 1, 0, 0],
                          [-np.sin(phi), 0, np.cos(phi), 0],
                          [0, 0, 0, 1]], dtype=np.float32)
    dummy_c2w = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, -medium_depth],
                          [0, 0, 0, 1]], dtype=np.float32)
    dummy_c2w = R_azimuth @ R_elevation @ dummy_c2w
    dummy_c2w[:3, 3] += np.array([0, 0, medium_depth], dtype=np.float32)
    return c2w @ dummy_c2w


def _fallback_axis_angle_to_matrix(axis, angle):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=np.float32)


def _fallback_camera_rotation(c2w, medium_depth, phi, theta):
    z0 = c2w[2, 3]
    if z0 != 0 and phi != 0:
        axis_origin = c2w @ np.array([0, 0, medium_depth, 1], dtype=np.float32).reshape(4, 1)
        axis_origin = axis_origin[:3, 0]
        axis_origin[2] = 0
        R = np.eye(4, dtype=np.float32)
        R[:3, :3] = _fallback_axis_angle_to_matrix(np.array([0, 0, 1], dtype=np.float32), -phi)
        T1 = np.eye(4, dtype=np.float32)
        T2 = np.eye(4, dtype=np.float32)
        T1[:3, 3] = -axis_origin
        T2[:3, 3] = axis_origin
        return T2 @ R @ T1 @ c2w
    return _fallback_native_camera_rotation(c2w, medium_depth, phi, theta)


def _fallback_interpolate_poses(poses, M):
    if poses.shape[0] == M:
        return poses
    indices = np.linspace(0, poses.shape[0] - 1, M)
    nearest = np.clip(np.round(indices).astype(int), 0, poses.shape[0] - 1)
    return poses[nearest]


try:
    # 1. Try importing via the default package path
    from src.camera_utils import (
        camera_backward_forward,
        camera_left_right,
        camera_rotation,
        native_camera_rotation,
        interpolate_poses,
    )
    CAMERA_UTILS_AVAILABLE = True
except Exception as first_import_error:
    try:
        # 2. Fallback: Import directly to bypass the contested "src" namespace
        from camera_utils import (
            camera_backward_forward,
            camera_left_right,
            camera_rotation,
            native_camera_rotation,
            interpolate_poses,
        )
        CAMERA_UTILS_AVAILABLE = True
    except Exception as e:
        print("\n" + "="*80)
        print(f"[WorldStereo DEBUG] src.camera_utils import error: {first_import_error}")
        print(f"[WorldStereo DEBUG] Real camera_utils import error: {e}")
        import traceback
        traceback.print_exc()
        print("="*80 + "\n")
        camera_backward_forward = _fallback_camera_backward_forward
        camera_left_right = _fallback_camera_left_right
        camera_rotation = _fallback_camera_rotation
        native_camera_rotation = _fallback_native_camera_rotation
        interpolate_poses = _fallback_interpolate_poses
        CAMERA_UTILS_AVAILABLE = True
        CAMERA_UTILS_FALLBACK = True
else:
    CAMERA_UTILS_FALLBACK = False

try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    folder_paths = None
    FOLDER_PATHS_AVAILABLE = False

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

# ── pytorch3d (optional, needed for circular preset) ─────────────────────────
try:
    from pytorch3d.renderer.cameras import look_at_rotation
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False


@contextmanager
def _temporary_worldstereo_runtime_patches(patch_diffusers=False):
    """Limit WorldStereo compatibility monkey-patches to the code block that needs them."""
    import torch.distributed as dist

    env_keys = ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    original_env = {key: os.environ.get(key) for key in env_keys}
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    class DummyProcessGroup:
        def __init__(self):
            self.group_name = "dummy_group"

        def __getattr__(self, name):
            def _fallback_func(*args, **kwargs):
                return args[0] if args else None
            return _fallback_func

    def _dummy_dist_func(*args, **kwargs):
        return args[0] if args else None

    def _dummy_all_gather(tensor_list, tensor, *args, **kwargs):
        if tensor_list:
            tensor_list[0].copy_(tensor)
        return tensor_list

    _dummy_pg = DummyProcessGroup()
    attr_patchers = {
        "is_initialized": lambda: True,
        "get_rank": lambda *args, **kwargs: 0,
        "get_world_size": lambda *args, **kwargs: 1,
        "barrier": lambda *args, **kwargs: None,
        "new_group": lambda *args, **kwargs: _dummy_pg,
        "all_reduce": _dummy_dist_func,
        "broadcast": _dummy_dist_func,
        "all_gather": _dummy_all_gather,
        "reduce_scatter": lambda output, input_list, *args, **kwargs: output,
        "get_backend": lambda *args, **kwargs: "gloo",
        "init_process_group": lambda *args, **kwargs: None,
    }

    patched_attrs = []
    for mod_name, mod in list(sys.modules.items()):
        if mod_name == "torch.distributed" or mod_name.startswith("torch.distributed."):
            if mod is None:
                continue
            for attr, patched in attr_patchers.items():
                if hasattr(mod, attr):
                    patched_attrs.append((mod, attr, getattr(mod, attr)))
                    try:
                        setattr(mod, attr, patched)
                    except Exception:
                        patched_attrs.pop()

    if dist not in [item[0] for item in patched_attrs]:
        for attr, patched in attr_patchers.items():
            if hasattr(dist, attr):
                patched_attrs.append((dist, attr, getattr(dist, attr)))
                setattr(dist, attr, patched)

    original_signature_keys = None
    pipeline_utils = None
    if patch_diffusers:
        import diffusers.pipelines.pipeline_utils as pipeline_utils
        original_signature_keys = pipeline_utils.DiffusionPipeline._get_signature_keys

        @staticmethod
        def patched_get_signature_keys(obj):
            expected_modules, optional_parameters = original_signature_keys(obj)

            if isinstance(expected_modules, list):
                expected_modules = [x for x in expected_modules if x != "device"]
            elif isinstance(expected_modules, set):
                expected_modules = expected_modules - {"device"}

            if isinstance(optional_parameters, list):
                if "device" not in optional_parameters:
                    optional_parameters.append("device")
            elif isinstance(optional_parameters, set):
                optional_parameters = optional_parameters | {"device"}

            return expected_modules, optional_parameters

        pipeline_utils.DiffusionPipeline._get_signature_keys = patched_get_signature_keys

    try:
        yield
    finally:
        if pipeline_utils is not None:
            pipeline_utils.DiffusionPipeline._get_signature_keys = original_signature_keys

        for mod, attr, original in reversed(patched_attrs):
            try:
                setattr(mod, attr, original)
            except Exception:
                pass

        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _build_intrinsics(fov_deg: float, width: int, height: int) -> torch.Tensor:
    """Build a [3, 3] camera intrinsics matrix from field-of-view and image size."""
    fx = fy = (width / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    cx = width / 2.0
    cy = height / 2.0
    K = torch.tensor([
        [fx,  0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)
    return K


def _c2w_to_w2c(c2ws: torch.Tensor) -> torch.Tensor:
    """Batch-invert [N, 4, 4] camera-to-world matrices to world-to-camera."""
    return torch.linalg.inv(c2ws)


def _opencv_look_at_c2w(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    z_axis = target - cam_pos
    z_axis = z_axis / max(np.linalg.norm(z_axis), 1e-8)

    world_down = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(world_down, z_axis))) > 0.99:
        world_down = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    x_axis = np.cross(world_down, z_axis)
    x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-8)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-8)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = x_axis
    c2w[:3, 1] = y_axis
    c2w[:3, 2] = z_axis
    c2w[:3, 3] = cam_pos
    return c2w


def _build_trajectory(
    preset: str,
    num_frames: int,
    radius: float,
    speed: float,
    elevation_deg: float,
    fov_deg: float,
    width: int,
    height: int,
    median_depth: float = 1.0,
    custom_json: str = "",
) -> tuple:
    """
    Build camera trajectory tensors.

    Returns:
        c2ws  : torch.Tensor [N, 4, 4] camera-to-world matrices
        intrs : torch.Tensor [N, 3, 3] intrinsics (same for every frame)
    """
    c2w_start = np.eye(4, dtype=np.float32)

    # ── per-preset trajectory construction ───────────────────────────────────
    if preset == "circular":
        if not PYTORCH3D_AVAILABLE:
            raise RuntimeError(
                "circular preset requires pytorch3d. "
                "Install it or choose a different preset."
            )
        look_at_point = np.array([0, 0, median_depth], dtype=np.float32)
        angles = np.linspace(0, 2 * math.pi, num_frames + 1)[1:]
        rx = radius * median_depth
        ry = radius * median_depth
        c2ws_np = []
        for angle in angles:
            cam_pos = np.array(
                [rx * np.sin(angle), ry * np.cos(angle) - ry, 0],
                dtype=np.float32,
            )
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, 3] = cam_pos
            R_new = look_at_rotation(
                cam_pos,
                at=(look_at_point.tolist(),),
                up=((0, 1, 0),),
                device="cpu",
            ).numpy()[0]
            c2w[:3, :3] = R_new
            c2w = c2w_start @ c2w
            c2ws_np.append(c2w)

    elif preset == "forward":
        if not CAMERA_UTILS_AVAILABLE:
            raise RuntimeError(
                "forward preset requires worldstereo camera_utils. "
                "Ensure worldstereo/ is present in the repo root."
            )
        c2ws_np = []
        for j in range(1, num_frames + 1):
            c2w = c2w_start.copy()
            c2w = camera_backward_forward(c2w, speed * j)
            c2ws_np.append(c2w)

    elif preset == "zoom_in":
        if not CAMERA_UTILS_AVAILABLE:
            raise RuntimeError(
                "zoom_in preset requires worldstereo camera_utils. "
                "Ensure worldstereo/ is present in the repo root."
            )
        c2ws_np = []
        for j in range(1, num_frames + 1):
            c2w = c2w_start.copy()
            c2w = camera_backward_forward(c2w, radius * j / num_frames)
            c2ws_np.append(c2w)

    elif preset == "zoom_out":
        if not CAMERA_UTILS_AVAILABLE:
            raise RuntimeError(
                "zoom_out preset requires worldstereo camera_utils. "
                "Ensure worldstereo/ is present in the repo root."
            )
        c2ws_np = []
        for j in range(1, num_frames + 1):
            c2w = c2w_start.copy()
            c2w = camera_backward_forward(c2w, -radius * j / num_frames)
            c2ws_np.append(c2w)

    elif preset == "stereo_orbit":
        c2ws_np = []
        target = np.array([0.0, 0.0, median_depth], dtype=np.float32)
        orbit_radius = max(float(speed), 1e-4)
        angles = np.linspace(0.0, 2.0 * math.pi, num_frames, endpoint=False, dtype=np.float32)
        for angle in angles:
            cam_pos = np.array(
                [orbit_radius * math.cos(float(angle)), orbit_radius * math.sin(float(angle)), 0.0],
                dtype=np.float32,
            )
            c2ws_np.append(_opencv_look_at_c2w(cam_pos, target))

    elif preset == "left_right":
        if not CAMERA_UTILS_AVAILABLE:
            raise RuntimeError(
                "left_right preset requires worldstereo camera_utils. "
                "Ensure worldstereo/ is present in the repo root."
            )
        c2ws_np = []
        offsets = np.linspace(-radius, radius, num_frames, dtype=np.float32)
        for offset in offsets:
            c2w = camera_left_right(c2w_start.copy(), float(offset))
            c2ws_np.append(c2w)

    elif preset == "up_down":
        c2ws_np = []
        offsets = np.linspace(-radius, radius, num_frames, dtype=np.float32)
        for offset in offsets:
            c2w = _camera_up_down(c2w_start.copy(), float(offset))
            c2ws_np.append(c2w)

    elif preset == "aerial":
        if not CAMERA_UTILS_AVAILABLE:
            raise RuntimeError(
                "aerial preset requires worldstereo camera_utils. "
                "Ensure worldstereo/ is present in the repo root."
            )
        c2ws_np = []
        phi_total = math.radians(elevation_deg)
        theta_total = math.radians(elevation_deg * 0.5)  # half elevation for theta
        n_theta = max(1, num_frames // 2)
        n_phi = num_frames - n_theta
        for j in range(1, n_theta + 1):
            theta_j = theta_total * j / n_theta
            c2w = camera_rotation(c2w_start.copy(), median_depth, 0, theta_j)
            c2ws_np.append(c2w)
        c2w_mid = c2ws_np[-1].copy() if c2ws_np else c2w_start.copy()
        for j in range(1, n_phi + 1):
            phi_j = phi_total * j / n_phi
            c2w = camera_rotation(c2w_mid.copy(), median_depth, phi_j, 0)
            c2ws_np.append(c2w)

    elif preset == "custom":
        data = json.loads(custom_json)
        c2ws_np = [np.array(m, dtype=np.float32) for m in data]
        if len(c2ws_np) == 0:
            raise ValueError("custom_json contains no matrices.")
        for i, mat in enumerate(c2ws_np):
            if mat.shape != (4, 4):
                raise ValueError(f"custom_json matrix {i} has shape {mat.shape}, expected (4, 4).")

    else:
        raise ValueError(f"Unknown preset: {preset!r}")

    # ── convert to torch ──────────────────────────────────────────────────────
    c2ws = torch.from_numpy(np.stack(c2ws_np)).float()  # [N, 4, 4]

    # ── build intrinsics (broadcast to all frames) ────────────────────────────
    K = _build_intrinsics(fov_deg, width, height)                # [3, 3]
    intrs = K.unsqueeze(0).expand(c2ws.shape[0], -1, -1).clone()  # [N, 3, 3]

    return c2ws, intrs


def _get_models_base() -> str:
    return (
        folder_paths.models_dir if FOLDER_PATHS_AVAILABLE
        else os.path.join(PROJECT_ROOT, "models")
    )


WORLDSTEREO_TURBO_LORAS = {
    "none": None,
    "CausVid 14B rank32 v2": {
        "repo_id": "Kijai/WanVideo_comfy",
        "filename": "Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors",
        "adapter_name": "causvid_14b_rank32_v2",
    },
    "AccVid I2V 480P 14B rank32": {
        "repo_id": "Kijai/WanVideo_comfy",
        "filename": "Wan21_AccVid_I2V_480P_14B_lora_rank32_fp16.safetensors",
        "adapter_name": "accvid_i2v_480p_14b_rank32",
    },
}


def _download_worldstereo_turbo_lora(lora_name: str):
    spec = WORLDSTEREO_TURBO_LORAS.get(lora_name)
    if spec is None:
        return None, None

    from huggingface_hub import hf_hub_download

    lora_dir = os.path.join(_get_models_base(), "loras", "wan")
    os.makedirs(lora_dir, exist_ok=True)
    lora_path = os.path.join(lora_dir, spec["filename"])

    if not os.path.exists(lora_path):
        print(f"[WorldStereo] Downloading turbo LoRA: {lora_name} ...")
        lora_path = hf_hub_download(
            repo_id=spec["repo_id"],
            filename=spec["filename"],
            local_dir=lora_dir,
        )
    else:
        print(f"[WorldStereo] Turbo LoRA cached: {lora_path}")

    return lora_path, spec


def _call_lora_method(method_name: str, calls):
    last_error = None
    for call in calls:
        try:
            return call()
        except TypeError as e:
            last_error = e
    if last_error is not None:
        raise last_error
    raise AttributeError(method_name)


def _apply_worldstereo_turbo_lora(pipeline, lora_name: str, lora_strength: float):
    lora_path, spec = _download_worldstereo_turbo_lora(lora_name)
    if spec is None:
        return None

    adapter_name = spec["adapter_name"]
    lora_strength = float(lora_strength)
    half_dtype = _worldstereo_half_dtype()

    # LoRA injection needs regular module weights; run it before fp8 quant/freeze.
    for name in ("text_encoder", "image_encoder", "transformer", "vae"):
        _move_module_to_half(getattr(pipeline, name, None), half_dtype)

    try:
        _call_lora_method(
            "load_lora_weights",
            (
                lambda: pipeline.load_lora_weights(
                    os.path.dirname(lora_path),
                    weight_name=os.path.basename(lora_path),
                    adapter_name=adapter_name,
                ),
                lambda: pipeline.load_lora_weights(lora_path, adapter_name=adapter_name),
            ),
        )
        _call_lora_method(
            "set_adapters",
            (
                lambda: pipeline.set_adapters([adapter_name], adapter_weights=[lora_strength]),
                lambda: pipeline.set_adapters([adapter_name], weights=[lora_strength]),
                lambda: pipeline.set_adapters(adapter_name, adapter_weights=lora_strength),
            ),
        )
        _call_lora_method(
            "fuse_lora",
            (
                lambda: pipeline.fuse_lora(adapter_names=[adapter_name], lora_scale=1.0),
                lambda: pipeline.fuse_lora(lora_scale=1.0),
                lambda: pipeline.fuse_lora(),
            ),
        )
        if hasattr(pipeline, "unload_lora_weights"):
            try:
                pipeline.unload_lora_weights()
            except Exception as e:
                print(f"[WorldStereo] Turbo LoRA adapter cleanup skipped ({type(e).__name__}: {e})")
    except Exception as e:
        raise RuntimeError(
            f"Failed to apply WorldStereo turbo LoRA {lora_name!r} from {lora_path}: {e}"
        ) from e

    print(f"[WorldStereo] Turbo LoRA applied and fused: {lora_name} (strength={lora_strength})")
    return lora_path


def _worldstereo_half_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.bfloat16


def _move_module_to_half(module, dtype: torch.dtype):
    if module is not None and hasattr(module, "to"):
        module.to(dtype=dtype)
    return module


def _apply_worldstereo_precision(pipeline, precision: str):
    """Apply non-fp32 precision to every heavy WorldStereo module."""
    half_dtype = _worldstereo_half_dtype()
    module_names = ("text_encoder", "image_encoder", "transformer", "vae")

    for name in module_names:
        _move_module_to_half(getattr(pipeline, name, None), half_dtype)

    if precision == "bf16":
        print(f"[WorldStereo] bf16/fp16 runtime dtype applied: {half_dtype}")
        return "bf16"

    if precision != "fp8":
        raise ValueError(f"Unsupported precision: {precision!r}")

    try:
        from optimum.quanto import freeze, qfloat8_e4m3fn, quantize
    except ImportError:
        print("[WorldStereo] optimum-quanto not installed; falling back to bf16/fp16 weights")
        return "bf16"

    quantized = []
    skipped = []
    for name in module_names:
        module = getattr(pipeline, name, None)
        if module is None:
            continue
        try:
            quantize(module, weights=qfloat8_e4m3fn)
            freeze(module)
            quantized.append(name)
        except Exception as e:
            skipped.append(f"{name} ({type(e).__name__}: {e})")
            _move_module_to_half(module, half_dtype)

    if quantized:
        print(f"[WorldStereo] fp8 weight quantization applied: {', '.join(quantized)}")
    if skipped:
        print(f"[WorldStereo] fp8 unavailable for: {', '.join(skipped)}; kept at {half_dtype}")
    return "fp8" if quantized else "bf16"


def _apply_worldstereo_transformer_export_precision(pipeline, precision: str):
    half_dtype = _worldstereo_half_dtype()
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        raise RuntimeError("WorldStereo pipeline has no transformer to export.")

    _move_module_to_half(transformer, half_dtype)
    if precision == "bf16":
        return "bf16"
    if precision != "fp8":
        raise ValueError(f"Unsupported export precision: {precision!r}")

    try:
        from optimum.quanto import freeze, qfloat8_e4m3fn, quantize
    except ImportError as e:
        raise ImportError("optimum-quanto required for fp8 export: pip install optimum-quanto") from e

    quantize(transformer, weights=qfloat8_e4m3fn)
    freeze(transformer)
    return "fp8"


def _tensor_state_dict_for_safetensors(module):
    state = {}
    for key, value in module.state_dict().items():
        if isinstance(value, torch.Tensor):
            state[key] = value.detach().cpu().contiguous()
    return state


def _resolve_worldstereo_export_paths(export_path: str, model_type: str, precision: str, turbo_lora: str):
    export_path = os.path.abspath(os.path.expanduser(export_path.strip()))
    if export_path.lower().endswith(".safetensors"):
        output_file = export_path
    else:
        suffix = turbo_lora.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        suffix = suffix if suffix and suffix != "none" else "clean"
        output_file = os.path.join(export_path, f"{model_type}-{suffix}-{precision}.safetensors")
    metadata_file = os.path.splitext(output_file)[0] + ".json"
    return output_file, metadata_file


def _download_hf_repo_missing(repo_id: str, local_dir: str, label: str, allow_patterns=None):
    from fnmatch import fnmatch
    from huggingface_hub import HfApi, hf_hub_download

    def _allowed(path: str) -> bool:
        if allow_patterns is None:
            return True
        return any(fnmatch(path, pattern) for pattern in allow_patterns)

    api = HfApi()
    try:
        entries = list(api.list_repo_tree(repo_id, recursive=True))
        remote_files = [
            (entry.path, getattr(entry, "size", None))
            for entry in entries
            if getattr(entry, "path", None) and _allowed(entry.path) and getattr(entry, "size", None) is not None
        ]
    except Exception:
        remote_files = [(path, None) for path in api.list_repo_files(repo_id) if _allowed(path)]

    missing = []
    mismatched = []
    for rel_path, remote_size in remote_files:
        local_path = os.path.join(local_dir, *rel_path.split("/"))
        if not os.path.exists(local_path):
            missing.append((rel_path, remote_size))
        elif remote_size is not None and os.path.getsize(local_path) != remote_size:
            mismatched.append((rel_path, remote_size, os.path.getsize(local_path)))

    if not missing and not mismatched:
        print(f"[WorldStereo] {label} cached: {local_dir}")
        return

    os.makedirs(local_dir, exist_ok=True)
    total_size = sum(size or 0 for _, size in missing) + sum(size or 0 for _, size, _ in mismatched)
    size_msg = f", {total_size / (1024 ** 3):.2f} GB" if total_size else ""
    print(
        f"[WorldStereo] Resuming {label}: "
        f"{len(missing)} missing, {len(mismatched)} incomplete/corrupt files{size_msg}"
    )

    for rel_path, remote_size, local_size in mismatched:
        local_path = os.path.join(local_dir, *rel_path.split("/"))
        print(
            f"[WorldStereo] Re-fetching incomplete file: {rel_path} "
            f"({local_size} / {remote_size} bytes)"
        )
        try:
            os.remove(local_path)
        except OSError:
            pass

    files_to_fetch = [(path, size) for path, size in missing] + [(path, size) for path, size, _ in mismatched]
    try:
        from tqdm.auto import tqdm
        iterator = tqdm(files_to_fetch, desc=f"[WorldStereo] {label}", unit="file")
    except Exception:
        iterator = files_to_fetch

    for index, (rel_path, _) in enumerate(iterator, start=1):
        print(f"[WorldStereo] [{index}/{len(files_to_fetch)}] downloading {rel_path}")
        hf_hub_download(
            repo_id=repo_id,
            filename=rel_path,
            local_dir=local_dir,
        )

    print(f"[WorldStereo] {label} cached: {local_dir}")


def _download_worldstereo_components(model_type: str, include_moge: bool = True) -> tuple:
    """
    Download all required model components. Returns (transformer_dir, base_model_dir, moge_dir).
    """
    from huggingface_hub import snapshot_download

    base = _get_models_base()

    # 1. WorldStereo transformer weights
    transformer_dir = os.path.join(base, "WorldStereo", model_type)
    transformer_weights = os.path.join(transformer_dir, "model.safetensors")
    if not os.path.exists(transformer_weights):
        print(f"[WorldStereo] Downloading transformer ({model_type}) ...")
        tmp_dir = os.path.join(base, "WorldStereo", "_tmp")
        snapshot_download(
            repo_id="hanshanxue/WorldStereo",  
            allow_patterns=[f"{model_type}/**"],
            local_dir=tmp_dir,
        )
        nested = os.path.join(tmp_dir, model_type)
        if os.path.isdir(nested):
            import shutil
            os.makedirs(transformer_dir, exist_ok=True)
            for f in os.listdir(nested):
                shutil.move(os.path.join(nested, f), transformer_dir)
            shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"[WorldStereo] Transformer cached: {transformer_dir}")
    else:
        print(f"[WorldStereo] Transformer cached: {transformer_dir}")

    # 2. Wan2.1 base model (VAE, T5, CLIP)
    base_model_dir = os.path.join(base, "Wan2.1-I2V-14B-480P")
    _download_hf_repo_missing(
        repo_id="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        local_dir=base_model_dir,
        label="Wan2.1-I2V-14B-480P base model",
    )

    # 3. MoGe depth estimator
    moge_dir = os.path.join(base, "MoGe")
    if include_moge:
        moge_config = os.path.join(moge_dir, "config.json")
        if not os.path.exists(moge_config):
            print(f"[WorldStereo] Downloading MoGe depth estimator ...")
            snapshot_download(
                repo_id="Ruicheng/moge-2-vitl-normal",
                local_dir=moge_dir,
            )
            print(f"[WorldStereo] MoGe cached: {moge_dir}")
        else:
            print(f"[WorldStereo] MoGe cached: {moge_dir}")

    # 4. Patch transformer config.json to use local base_model path
    config_path = os.path.join(transformer_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        if cfg.get("base_model") != base_model_dir:
            cfg["base_model"] = base_model_dir
            with open(config_path, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=4)
            print(f"[WorldStereo] config.json patched -> base_model={base_model_dir}")

    return transformer_dir, base_model_dir, moge_dir


# ─────────────────────────────────────────────────────────────────────────────
# VNCCS_CameraTrajectoryBuilder
# ─────────────────────────────────────────────────────────────────────────────

class VNCCS_CameraTrajectoryBuilder:
    PRESETS = ["circular", "stereo_orbit", "forward", "zoom_in", "zoom_out", "left_right", "up_down", "aerial", "custom"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "preset": (cls.PRESETS, {"default": "stereo_orbit"}),
                "num_frames": ("INT", {"default": 25, "min": 4, "max": 81}),
                "radius": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Orbit radius, travel distance, or lateral/up-down span depending on preset.",
                    },
                ),
                "speed": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.001,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Per-frame translation for forward preset or parallax radius for stereo_orbit.",
                    },
                ),
                "elevation_deg": (
                    "FLOAT",
                    {
                        "default": 15.0,
                        "min": -90.0,
                        "max": 90.0,
                        "step": 1.0,
                        "tooltip": "Camera elevation for aerial preset.",
                    },
                ),
                "median_depth": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "Estimated scene depth — orbit center distance.",
                    },
                ),
                "fov_deg": (
                    "FLOAT",
                    {"default": 70.0, "min": 10.0, "max": 150.0, "step": 1.0},
                ),
                "image_width": (
                    "INT",
                    {"default": 768, "min": 64, "max": 2048, "step": 64},
                ),
                "image_height": (
                    "INT",
                    {"default": 480, "min": 64, "max": 2048, "step": 64},
                ),
                "custom_json": (
                    "STRING",
                    {
                        "default": "[]",
                        "multiline": True,
                        "tooltip": "JSON list of N 4x4 C2W matrices. Used when preset=custom.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("CAMERA_TRAJECTORY",)
    RETURN_NAMES = ("trajectory",)
    FUNCTION = "build"
    CATEGORY = "VNCCS/Video"

    def build(
        self,
        preset="circular",
        num_frames=25,
        radius=1.0,
        speed=0.05,
        elevation_deg=15.0,
        median_depth=1.0,
        fov_deg=70.0,
        image_width=768,
        image_height=480,
        custom_json="[]",
    ):
        c2ws, intrs = _build_trajectory(
            preset,
            num_frames,
            radius,
            speed,
            elevation_deg,
            fov_deg,
            image_width,
            image_height,
            median_depth,
            custom_json,
        )
        trajectory = {
            "c2ws": c2ws,
            "intrs": intrs,
            "width": image_width,
            "height": image_height,
            "preset": preset,
        }
        print(
            f"[Trajectory] preset={preset}, frames={c2ws.shape[0]}, "
            f"size={image_width}x{image_height}"
        )
        return (trajectory,)


class VNCCS_LoadWorldStereoModel:
    """Download and load the WorldStereo pipeline + MoGe depth estimator."""

    MODEL_TYPES = ["worldstereo-camera", "worldstereo-memory", "worldstereo-memory-dmd"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "model_type": (cls.MODEL_TYPES, {
                    "default": "worldstereo-camera",
                    "tooltip": (
                        "worldstereo-camera: 10.9 GB transformer, feasible on 16 GB VRAM with offloading. "
                        "worldstereo-memory: ~22 GB, requires 24+ GB VRAM. "
                        "worldstereo-memory-dmd: 34.9 GB distilled, requires 40+ GB VRAM."
                    ),
                }),
                "precision": (["fp8", "bf16"], {
                    "default": "fp8",
                    "tooltip": (
                        "fp8: quantize all supported heavy modules via optimum-quanto. "
                        "bf16: half precision fallback, never fp32."
                    ),
                }),
                "turbo_lora": (list(WORLDSTEREO_TURBO_LORAS.keys()), {
                    "default": "none",
                    "tooltip": (
                        "Optional Wan2.1 14B acceleration LoRA. It is downloaded, loaded, fused, "
                        "then the pipeline is quantized/offloaded."
                    ),
                }),
                "lora_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Scale for the selected turbo LoRA before fusing.",
                    },
                ),
                "offload_mode": (["sequential_cpu_offload", "model_cpu_offload", "none"], {
                    "default": "sequential_cpu_offload",
                    "tooltip": (
                        "sequential_cpu_offload: layer-by-layer, slower but less VRAM. "
                        "model_cpu_offload: move components to CPU between steps. Faster, but requires more VRAM. "
                        "none: all components stay on GPU."
                    ),
                }),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("WORLDSTEREO_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "VNCCS/Video"

    def load_model(
        self,
        model_type="worldstereo-camera",
        precision="fp8",
        turbo_lora="none",
        lora_strength=1.0,
        offload_mode="sequential_cpu_offload",
        device="cuda",
    ):
        import sys
        import os

        # Resolve and patch WorldStereo import paths. ComfyUI may already own a
        # top-level "models" module, so make it a namespace package if needed.
        current_dir = os.path.dirname(os.path.dirname(__file__))
        _prepare_worldstereo_import_paths(current_dir)

        # ── Call download helper to resolve and prepare directory paths ───────
        transformer_dir, base_model_dir, moge_dir = _download_worldstereo_components(model_type)

        # Normalize Windows backslashes to forward slashes to prevent split('/') failures inside WorldStereo
        transformer_dir = transformer_dir.replace("\\", "/")
        base_model_dir = base_model_dir.replace("\\", "/")
        moge_dir = moge_dir.replace("\\", "/")

        with _temporary_worldstereo_runtime_patches(patch_diffusers=True):
            WorldStereo = _import_worldstereo_class(current_dir)

            # Resolve MoGeModel class import based on your installed MoGe library version
            try:
                from moge.model.v2 import MoGeModel
            except ImportError:
                try:
                    from moge.model.v1 import MoGeModel
                except ImportError:
                    from moge.model import MoGeModel

            # ── Load WorldStereo pipeline ─────────────────────────────────────
            print(f"[WorldStereo] Loading pipeline (model_type={model_type}, precision={precision}) ...")
            parent_dir = os.path.dirname(transformer_dir)
            model_device = "cpu" if device == "cuda" and offload_mode != "none" else device
            worldstereo = WorldStereo.from_pretrained(
                parent_dir,
                subfolder=model_type,
                device=device,
                model_device=model_device,
            )
        pipeline = worldstereo.pipeline

        turbo_lora_path = _apply_worldstereo_turbo_lora(pipeline, turbo_lora, lora_strength)
        precision = _apply_worldstereo_precision(pipeline, precision)

        # ── Apply offloading ──────────────────────────────────────────────────
        if device == "cuda":
            if offload_mode == "model_cpu_offload":
                pipeline.enable_model_cpu_offload()
                print("[WorldStereo] model_cpu_offload enabled")
            elif offload_mode == "sequential_cpu_offload":
                pipeline.enable_sequential_cpu_offload()
                print("[WorldStereo] sequential_cpu_offload enabled")
            elif offload_mode == "none":
                print("[WorldStereo Warning] CPU offload disabled; this is likely too large for normal VRAM.")

        # ── Load MoGe on CPU ─────────────────────────────────────────────────
        print("[WorldStereo] Loading MoGe depth estimator ...")
        
        # Point directly to the model.pt file inside the cached directory
        actual_moge_path = os.path.join(moge_dir, "model.pt")
        if not os.path.exists(actual_moge_path):
            actual_moge_path = moge_dir  # Fallback if structure varies
            
        moge_model = MoGeModel.from_pretrained(actual_moge_path).eval()
        try:
            moge_model.to(dtype=_worldstereo_half_dtype())
        except Exception as e:
            print(f"[WorldStereo] MoGe half precision cast skipped ({type(e).__name__}: {e})")
        print("[WorldStereo] MoGe loaded (CPU)")

        print("[WorldStereo] Pipeline ready")
        return ({
            "worldstereo": worldstereo,
            "pipeline":    pipeline,
            "moge":        moge_model,
            "device":      device,
            "model_type":  model_type,
            "precision":   precision,
            "offload_mode": offload_mode,
            "turbo_lora":  turbo_lora,
            "lora_strength": float(lora_strength),
            "turbo_lora_path": turbo_lora_path,
        },)

def _prepare_pipeline_inputs(
    image_pil,
    c2ws: torch.Tensor,
    intrs: torch.Tensor,
    moge_model,
    device: str,
    width: int,
    height: int,
) -> dict:
    """
    Build render_video, render_mask, camera_embedding from a single image + trajectory.
    Replicates WorldStereo's load_single_view_data() for arbitrary inputs.
    """
    import torchvision.transforms as T
    from src.pointcloud import get_points3d_and_colors, point_rendering
    from models.camera import get_camera_embedding

    N = c2ws.shape[0]

    # 1. Image tensor in [-1, 1] for pipeline
    img_tensor = T.ToTensor()(image_pil) * 2.0 - 1.0   # [3, H, W], range [-1, 1]
    
    # Create [1, 3, H, W] PyTorch tensor in [0, 1] for pointcloud functions
    img_tensor_01 = (img_tensor + 1.0).mul(0.5).unsqueeze(0).float() # [1, 3, H, W]

    # 2. Depth via MoGe
    torch_device = torch.device(device)
    moge_model = moge_model.to(torch_device)
    infer_dtype = _worldstereo_half_dtype()
    with torch.no_grad(), torch.autocast(device_type=torch_device.type, dtype=infer_dtype, enabled=torch_device.type in ("cuda", "cpu")):
        depth_output = moge_model.infer(
            img_tensor.unsqueeze(0).to(torch_device, dtype=infer_dtype)
        )
    # MoGe returns dict; extract depth as [1, 1, H, W] PyTorch tensor
    depth_raw = depth_output["depth"]
    if isinstance(depth_raw, torch.Tensor):
        depth_tensor = depth_raw.float()
        if depth_tensor.dim() == 2:
            depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)
        elif depth_tensor.dim() == 3:
            depth_tensor = depth_tensor.unsqueeze(0)
    else:
        depth_tensor = torch.from_numpy(depth_raw).float().unsqueeze(0).unsqueeze(0)
    
    moge_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. W2C matrices as PyTorch Tensors
    w2cs_t = _c2w_to_w2c(c2ws).float()   # [N, 4, 4]
    intrs_t = intrs.float()              # [N, 3, 3]
    ref_w2c = w2cs_t[0:1]                # [1, 4, 4] reference view
    ref_K   = intrs_t[0:1]               # [1, 3, 3] reference view (preserves batch dim)

    # 4. 3D point cloud from reference view
    # Keep on CPU initially — get_points3d_and_colors will move it to target device
    points3d, colors = get_points3d_and_colors(
        K=ref_K,
        w2cs=ref_w2c,
        depth=depth_tensor.cpu(),
        image=img_tensor_01,
        device=device,
    )

    # 5. Render point cloud from all N target views
    render_result = point_rendering(
        K=intrs_t,
        w2cs=w2cs_t,
        points=points3d,
        colors=colors,
        device=device,
        h=height,
        w=width,
    )
    # point_rendering returns (render_rgbs, render_masks) OR (render_rgbs, render_masks, render_depth)
    if isinstance(render_result, (tuple, list)):
        render_rgbs_raw, render_masks_raw = render_result[0], render_result[1]
    else:
        raise RuntimeError(f"Unexpected point_rendering return type: {type(render_result)}")

    # render_rgbs_raw: [N, 3, H, W] or [N, H, W, 3] — normalise to [-1, 1] tensor
    if isinstance(render_rgbs_raw, np.ndarray):
        render_rgbs_t = torch.from_numpy(render_rgbs_raw).float()
    else:
        render_rgbs_t = render_rgbs_raw.float()

    if render_rgbs_t.dim() == 4 and render_rgbs_t.shape[-1] == 3:
        render_rgbs_t = render_rgbs_t.permute(0, 3, 1, 2)  # [N, H, W, 3] → [N, 3, H, W]

    if render_rgbs_t.max() <= 1.5:   # [0, 1] range → convert to [-1, 1]
        render_rgbs_t = render_rgbs_t * 2.0 - 1.0
    render_rgbs_t[0] = img_tensor    # first frame = original image

    # render_masks_raw: [N, 1, H, W] or [N, H, W, 1] or [N, H, W]
    if isinstance(render_masks_raw, np.ndarray):
        render_masks_t = torch.from_numpy(render_masks_raw).float()
    else:
        render_masks_t = render_masks_raw.float()

    if render_masks_t.dim() == 3:
        render_masks_t = render_masks_t.unsqueeze(1)  # [N, H, W] → [N, 1, H, W]
    elif render_masks_t.dim() == 4 and render_masks_t.shape[-1] == 1:
        render_masks_t = render_masks_t.permute(0, 3, 1, 2)  # [N, H, W, 1] → [N, 1, H, W]

    # Reshape to [1, C, N, H, W] (batch=1)
    conditioning_dtype = _worldstereo_half_dtype()
    render_video = render_rgbs_t.unsqueeze(0).permute(0, 2, 1, 3, 4).to(torch_device, dtype=conditioning_dtype)   # [1, 3, N, H, W]
    render_mask  = render_masks_t.unsqueeze(0).permute(0, 2, 1, 3, 4).to(torch_device, dtype=conditioning_dtype)  # [1, 1, N, H, W]

    # 6. Camera embedding [1, 6, N, H, W]
    camera_emb = get_camera_embedding(
        intrinsic=intrs.to(torch_device),  # [N, 3, 3]
        extrinsic=c2ws.to(torch_device),   # [N, 4, 4] C2W (is_w2c=False)
        f=N, h=height, w=width,
        normalize=True,
        is_w2c=False,
    ).to(dtype=conditioning_dtype)

    return {
        "image":            image_pil,
        "render_video":     render_video,
        "render_mask":      render_mask,
        "camera_embedding": camera_emb,
        "extrinsics":       _c2w_to_w2c(c2ws).float().to(torch_device),  # [N, 4, 4] W2C
        "intrinsics":       intrs.to(torch_device),   # [N, 3, 3]
        "height":           height,
        "width":            width,
        "num_frames":       N,
    }


def _get_pipeline_execution_device(pipeline, fallback_device: str) -> torch.device:
    execution_device = getattr(pipeline, "_execution_device", None)
    if callable(execution_device):
        execution_device = execution_device()
    if execution_device is None:
        execution_device = fallback_device
    return torch.device(execution_device)


def _free_pipeline_offload_hooks(pipeline, context: str):
    if not hasattr(pipeline, "maybe_free_model_hooks"):
        return
    try:
        pipeline.maybe_free_model_hooks()
    except Exception as e:
        print(f"[WorldStereo] Offload hook cleanup skipped after {context} ({type(e).__name__}: {e})")


def _encode_prompt_cache(
    pipeline,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
    model_type: str,
    device: str,
):
    do_cfg = guidance_scale > 1.0 and "dmd" not in model_type
    execution_device = _get_pipeline_execution_device(pipeline, device)

    with torch.no_grad(), torch.autocast(
        execution_device.type,
        dtype=_worldstereo_half_dtype(),
        enabled=execution_device.type in ("cuda", "cpu"),
    ):
        embeds = pipeline.encode_prompt(
            prompt=prompt if prompt else "",
            negative_prompt=negative_prompt if negative_prompt else None,
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=1,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            max_sequence_length=512,
            device=execution_device,
        )
    _free_pipeline_offload_hooks(pipeline, "prompt encode")
    return embeds


def _image_cache_key(image_pil) -> str:
    hasher = hashlib.sha1()
    hasher.update(str(image_pil.size).encode("ascii"))
    hasher.update(image_pil.mode.encode("ascii"))
    hasher.update(image_pil.tobytes())
    return hasher.hexdigest()


def _encode_image_cache(pipeline, image_pil, device: str):
    execution_device = _get_pipeline_execution_device(pipeline, device)
    with torch.no_grad(), torch.autocast(
        execution_device.type,
        dtype=_worldstereo_half_dtype(),
        enabled=execution_device.type in ("cuda", "cpu"),
    ):
        image_embeds = pipeline.encode_image(image_pil, execution_device)
    _free_pipeline_offload_hooks(pipeline, "image encode")
    return image_embeds


def _move_pipeline_inputs(pipeline_inputs: dict, device: str | torch.device):
    moved = {}
    for key, value in pipeline_inputs.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _crop_generated_edges(frames: torch.Tensor, intrs: torch.Tensor, crop_percent: float):
    crop_percent = float(crop_percent)
    if crop_percent <= 0.0:
        return frames, intrs

    _, height, width, _ = frames.shape
    crop_x = int(round(width * crop_percent / 100.0))
    crop_y = int(round(height * crop_percent / 100.0))
    max_crop_x = max(0, (width - 16) // 2)
    max_crop_y = max(0, (height - 16) // 2)
    crop_x = min(crop_x, max_crop_x)
    crop_y = min(crop_y, max_crop_y)

    if crop_x == 0 and crop_y == 0:
        return frames, intrs

    cropped = frames[:, crop_y:height - crop_y, crop_x:width - crop_x, :]
    adjusted_intrs = intrs.clone()
    adjusted_intrs[:, 0, 2] -= crop_x
    adjusted_intrs[:, 1, 2] -= crop_y
    return cropped.contiguous(), adjusted_intrs


def _drop_duplicate_first_frame(
    frames: torch.Tensor,
    poses: torch.Tensor,
    intrs: torch.Tensor,
    image_pil,
):
    if frames.shape[0] <= 1:
        return frames, poses, intrs, False

    ref_np = np.asarray(image_pil.convert("RGB"), dtype=np.float32) / 255.0
    ref = torch.from_numpy(ref_np).to(dtype=frames.dtype, device=frames.device)
    if ref.shape != frames[0].shape:
        return frames, poses, intrs, False

    diff = (frames[0] - ref).abs()
    mean_diff = float(diff.mean().item())
    max_diff = float(diff.max().item())
    if mean_diff <= 0.006 and max_diff <= 0.04:
        print(
            "[WorldStereo] First frame matches source slice; "
            f"dropping duplicate (mean diff={mean_diff:.5f}, max diff={max_diff:.5f})"
        )
        return frames[1:].contiguous(), poses[1:].contiguous(), intrs[1:].contiguous(), True

    print(
        "[WorldStereo] First frame kept; source comparison differs "
        f"(mean diff={mean_diff:.5f}, max diff={max_diff:.5f})"
    )
    return frames, poses, intrs, False


@contextmanager
def _pipeline_conditioning_cache(pipeline, enabled: bool):
    old_enabled = getattr(pipeline, "_vnccs_cache_latent_condition", False)
    old_cache = getattr(pipeline, "_vnccs_latent_condition_cache", None)

    if enabled:
        pipeline._vnccs_cache_latent_condition = True
        pipeline._vnccs_latent_condition_cache = {}
    try:
        yield
    finally:
        if enabled:
            pipeline._vnccs_latent_condition_cache = old_cache
            pipeline._vnccs_cache_latent_condition = old_enabled


class VNCCS_WorldStereoGenerate:
    """
    WorldStereo camera-guided video generation from a single image or batch of sliced views.
    Outputs video_frames + camera_poses + camera_intrinsics for VNCCS_WorldMirrorV2_3D.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":      ("WORLDSTEREO_MODEL",),
                "image":      ("IMAGE",),
                "trajectory": ("CAMERA_TRAJECTORY",),
            },
            "optional": {
                "prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Text prompt to guide the inpainting/generation of occluded parts.",
                }),
                "num_inference_steps": ("INT", {
                    "default": 0, "min": 0, "max": 100,
                    "tooltip": "0 = auto (4 for turbo LoRA or memory-dmd, 20 for others).",
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 5.0, "min": 1.0, "max": 20.0, "step": 0.5,
                    "tooltip": "Forced to 1.0 automatically when a turbo LoRA is loaded.",
                }),
                "seed": ("INT", {
                    "default": -1, "min": -1, "max": 2**31 - 1,
                    "tooltip": "-1 = random.",
                }),
                "negative_prompt": ("STRING", {"default": ""}),
                "cache_conditioning": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cache T5 prompt embeds, CLIP image embeds, and VAE latent conditioning across slices.",
                }),
                "crop_generated_edges": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Crop generated borders from every output frame and adjust intrinsics.",
                }),
                "edge_crop_percent": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 30.0,
                        "step": 0.5,
                        "tooltip": "Percent to crop from each side when crop_generated_edges is enabled.",
                    },
                ),
                "base_camera_poses": ("TENSOR",),
                "base_camera_intrinsics": ("TENSOR",),
            },
        }

    RETURN_TYPES  = ("IMAGE",         "TENSOR",         "TENSOR")
    RETURN_NAMES  = ("video_frames",  "camera_poses",   "camera_intrinsics")
    FUNCTION      = "generate"
    CATEGORY = "VNCCS/Video"

    def generate(
        self,
        model,
        image,
        trajectory,
        prompt="",
        num_inference_steps=0,
        guidance_scale=5.0,
        seed=-1,
        negative_prompt="",
        cache_conditioning=True,
        crop_generated_edges=False,
        edge_crop_percent=8.0,
        base_camera_poses=None,
        base_camera_intrinsics=None,
    ):
        pipeline   = model["pipeline"]
        moge_model = model["moge"]
        device     = model["device"]
        model_type = model["model_type"]
        turbo_lora = model.get("turbo_lora", "none")
        has_turbo_lora = turbo_lora not in (None, "", "none")

        c2ws  = trajectory["c2ws"]    # [N, 4, 4]
        intrs = trajectory["intrs"]   # [N, 3, 3]
        W     = trajectory["width"]
        H     = trajectory["height"]
        N     = c2ws.shape[0]
        trajectory_preset = trajectory.get("preset", "")

        if num_inference_steps == 0:
            num_inference_steps = 4 if has_turbo_lora or "dmd" in model_type else 20
        if has_turbo_lora and guidance_scale != 1.0:
            print(
                f"[WorldStereo] Turbo LoRA '{turbo_lora}' loaded; "
                f"forcing guidance_scale {guidance_scale} -> 1.0"
            )
            guidance_scale = 1.0
        if trajectory_preset == "stereo_orbit" and not crop_generated_edges:
            crop_generated_edges = True
            edge_crop_percent = max(float(edge_crop_percent), 6.0)
            print(
                "[WorldStereo] stereo_orbit trajectory detected; "
                f"auto edge crop enabled ({edge_crop_percent:.1f}%)"
            )

        # Extract the total slice count from the input image batch
        B = image.shape[0] if isinstance(image, torch.Tensor) else len(image)
        print(f"[WorldStereo Batch Loop] Slices found: {B}. Initializing sequence generation loop...")

        # ── Safety and Format Checks for Poses and Intrinsics ──
        # Check if the user swapped camera_poses and intrinsics connections
        if base_camera_poses is not None and base_camera_intrinsics is not None:
            # Poses are typically [B, 4, 4] or [B, 3, 4], intrinsics are [B, 3, 3]
            # If poses has last dim 3, but intrinsics has last dim 4, they are swapped
            if base_camera_poses.shape[-1] == 3 and base_camera_intrinsics.shape[-1] == 4:
                print("[WorldStereo Warning] Swapped inputs detected! base_camera_poses and base_camera_intrinsics appear to be reversed. Swapping them back automatically.")
                base_camera_poses, base_camera_intrinsics = base_camera_intrinsics, base_camera_poses

        # If base_camera_poses was provided but looks like intrinsics [B, 3, 3]
        if base_camera_poses is not None and base_camera_poses.shape[-2] == 3 and base_camera_poses.shape[-1] == 3:
            print("[WorldStereo Warning] base_camera_poses has shape [3, 3], which looks like intrinsics. Treating it as base_camera_intrinsics instead.")
            if base_camera_intrinsics is None:
                base_camera_intrinsics = base_camera_poses
            base_camera_poses = None

        # Helper function to pad [3, 4] poses to [4, 4]
        def pad_pose_to_4x4(pose: torch.Tensor) -> torch.Tensor:
            if pose.shape[-2] == 3 and pose.shape[-1] == 4:
                if len(pose.shape) == 2:  # [3, 4]
                    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=pose.dtype, device=pose.device)
                    return torch.cat([pose, bottom], dim=0)  # [4, 4]
                elif len(pose.shape) == 3:  # [B, 3, 4]
                    B_dim = pose.shape[0]
                    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=pose.dtype, device=pose.device)
                    bottom = bottom.unsqueeze(0).repeat(B_dim, 1, 1)  # [B, 1, 4]
                    return torch.cat([pose, bottom], dim=1)  # [B, 4, 4]
            return pose

        # Pad base camera poses if they exist
        if base_camera_poses is not None:
            base_camera_poses = pad_pose_to_4x4(base_camera_poses)

        all_video_frames = []
        all_camera_poses = []
        all_camera_intrinsics = []
        cached_prompt_embeds = None
        cached_negative_prompt_embeds = None
        image_embeds_cache = {}
        latent_condition_cache = {}

        if cache_conditioning:
            print("[WorldStereo] Conditioning cache enabled: T5 prompt, CLIP image, VAE latent condition")
            cached_prompt_embeds, cached_negative_prompt_embeds = _encode_prompt_cache(
                pipeline=pipeline,
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                model_type=model_type,
                device=device,
            )
            cached_prompt_embeds = cached_prompt_embeds.to("cpu")
            if cached_negative_prompt_embeds is not None:
                cached_negative_prompt_embeds = cached_negative_prompt_embeds.to("cpu")

        # Iterate through each panoramic view slice
        for b in range(B):
            print(f"[WorldStereo Batch Loop] Processing view slice {b + 1}/{B}...")
            
            # Extract slice image and convert to PIL
            img_slice = image[b] if isinstance(image, torch.Tensor) else image[b]
            img_np  = (img_slice.cpu().numpy()[..., :3] * 255).astype(np.uint8)
            img_pil = PILImage.fromarray(img_np).resize((W, H), PILImage.Resampling.BICUBIC)

            # Determine the base camera pose for this specific slice
            if base_camera_poses is not None:
                if len(base_camera_poses.shape) == 3:  # Shape [B, 4, 4]
                    idx_pose = b if b < base_camera_poses.shape[0] else 0
                    base_pose = base_camera_poses[idx_pose]
                else:  # Fallback to single [4, 4] matrix
                    base_pose = base_camera_poses
            else:
                # Identity matrix fallback if unconnected
                base_pose = torch.eye(4, dtype=torch.float32)

            # Determine the base intrinsics for this specific slice
            if base_camera_intrinsics is not None:
                if len(base_camera_intrinsics.shape) == 3:  # Shape [B, 3, 3]
                    idx_intr = b if b < base_camera_intrinsics.shape[0] else 0
                    base_K = base_camera_intrinsics[idx_intr]
                else:  # Fallback to single [3, 3] matrix
                    base_K = base_camera_intrinsics
            else:
                # Fallback to standard intrinsics from the trajectory
                base_K = intrs[0]

            # Compose absolute trajectory camera-to-world (c2w) matrices: T_abs = T_base * T_trajectory
            base_pose_dev = base_pose.to(c2ws.device, dtype=c2ws.dtype)
            # Use explicit batch expansion and bmm to avoid batch broadcasting issues in PyTorch
            base_pose_expanded = base_pose_dev.unsqueeze(0).expand(N, -1, -1)  # [N, 4, 4]
            c2ws_abs = torch.bmm(base_pose_expanded, c2ws)                     # [N, 4, 4]

            # Expand base intrinsics across all frames: [N, 3, 3]
            base_K_dev = base_K.to(intrs.device, dtype=intrs.dtype)
            intrs_abs = base_K_dev.unsqueeze(0).expand(N, -1, -1).clone()

            # ── Preprocess Pipeline Inputs for the Slice ──
            print(f"[WorldStereo] Slice {b + 1} processing: depth estimation + point rendering...")
            with _temporary_worldstereo_runtime_patches():
                pipeline_inputs = _prepare_pipeline_inputs(
                    image_pil=img_pil,
                    c2ws=c2ws_abs,
                    intrs=intrs_abs,
                    moge_model=moge_model,
                    device=device,
                    width=W,
                    height=H,
                )
            if device == "cuda":
                try:
                    moge_model.to("cpu")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    pipeline_inputs = _move_pipeline_inputs(pipeline_inputs, "cpu")
                except Exception as e:
                    print(f"[WorldStereo] MoGe CPU offload skipped ({type(e).__name__}: {e})")

            cached_image_embeds = None
            if cache_conditioning:
                image_key = _image_cache_key(img_pil)
                cached_image_embeds = image_embeds_cache.get(image_key)
                if cached_image_embeds is None:
                    cached_image_embeds = _encode_image_cache(pipeline, img_pil, device).to("cpu")
                    image_embeds_cache[image_key] = cached_image_embeds
                    print(f"[WorldStereo] Slice {b + 1} image embedding cached")
                else:
                    print(f"[WorldStereo] Slice {b + 1} image embedding reused")

            # ── Generator Setup ──
            generator = None
            if seed >= 0:
                generator = torch.Generator(device=device).manual_seed(seed + b)

            # ── Diffusion Generation ──
            print(f"[WorldStereo] Slice {b + 1} executing inference for {N} camera frames...")
            autocast_device = torch.device(device).type
            with torch.autocast(autocast_device, dtype=_worldstereo_half_dtype()):
                if device == "cuda":
                    pipeline_inputs = _move_pipeline_inputs(pipeline_inputs, device)
                pipeline_kwargs = {
                    **pipeline_inputs,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "generator": generator,
                    "output_type": "pt",
                }
                if cache_conditioning:
                    pipeline_kwargs.update({
                        "prompt": None,
                        "negative_prompt": None,
                        "prompt_embeds": cached_prompt_embeds.to(device),
                        "negative_prompt_embeds": cached_negative_prompt_embeds.to(device) if cached_negative_prompt_embeds is not None else None,
                        "image_embeds": cached_image_embeds.to(device),
                    })
                else:
                    pipeline_kwargs.update({
                        "prompt": prompt if prompt else "",
                        "negative_prompt": negative_prompt if negative_prompt else None,
                    })
                with _pipeline_conditioning_cache(pipeline, cache_conditioning):
                    if cache_conditioning:
                        pipeline._vnccs_latent_condition_cache = latent_condition_cache
                    output = pipeline(**pipeline_kwargs)
                del pipeline_kwargs
                del pipeline_inputs

            # Clean cache immediately to keep low VRAM overhead on sequential slices
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ── Extract Generated Frames ──
            frames = output.frames[0].float().cpu().clamp(0.0, 1.0)  # [N, 3, H, W]
            slice_video_frames = frames.permute(0, 2, 3, 1)          # [N, H, W, 3]
            slice_poses = c2ws_abs.cpu().float()
            slice_intrs = intrs_abs.cpu().float()
            slice_video_frames, slice_poses, slice_intrs, _ = _drop_duplicate_first_frame(
                slice_video_frames,
                slice_poses,
                slice_intrs,
                img_pil,
            )
            if crop_generated_edges:
                slice_video_frames, slice_intrs = _crop_generated_edges(
                    slice_video_frames,
                    slice_intrs,
                    edge_crop_percent,
                )

            all_video_frames.append(slice_video_frames)
            all_camera_poses.append(slice_poses)
            all_camera_intrinsics.append(slice_intrs)

        # Concatenate sequences of all processed slices into unified tensors
        video_frames     = torch.cat(all_video_frames, dim=0)       # [B * N, H, W, 3]
        camera_poses_out = torch.cat(all_camera_poses, dim=0)       # [B * N, 4, 4]
        camera_intrs_out = torch.cat(all_camera_intrinsics, dim=0)  # [B * N, 3, 3]

        print(f"[WorldStereo Batch Loop] Complete. Total of {video_frames.shape[0]} frames prepared for 3D fusion.")

        return video_frames, camera_poses_out, camera_intrs_out


class VNCCS_ExportWorldStereoSingleModel:
    """Experimental exporter for single-file WorldStereo transformer checkpoints."""

    MODEL_TYPES = VNCCS_LoadWorldStereoModel.MODEL_TYPES

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "export_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Absolute file path ending in .safetensors, or a directory. Can be outside ComfyUI.",
                    },
                ),
            },
            "optional": {
                "model_type": (cls.MODEL_TYPES, {"default": "worldstereo-camera"}),
                "precision": (["bf16", "fp8"], {
                    "default": "bf16",
                    "tooltip": "bf16 is reload-safe. fp8 is experimental and requires a future fp8-aware single loader.",
                }),
                "turbo_lora": (list(WORLDSTEREO_TURBO_LORAS.keys()), {"default": "none"}),
                "lora_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "device": (["cpu", "cuda"], {
                    "default": "cpu",
                    "tooltip": "cpu avoids VRAM pressure but needs a lot of RAM. cuda may be faster but risky.",
                }),
                "overwrite": ("BOOLEAN", {"default": False}),
                "write_metadata_json": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("export_info",)
    FUNCTION = "export_model"
    CATEGORY = "VNCCS/Video"

    def export_model(
        self,
        export_path,
        model_type="worldstereo-camera",
        precision="bf16",
        turbo_lora="none",
        lora_strength=1.0,
        device="cpu",
        overwrite=False,
        write_metadata_json=True,
    ):
        import gc
        from safetensors.torch import save_file

        if not export_path or not export_path.strip():
            raise ValueError("export_path is required and may point outside the ComfyUI output folder.")
        if device == "cuda" and not torch.cuda.is_available():
            print("[WorldStereo Export] CUDA requested but unavailable; using CPU")
            device = "cpu"

        output_file, metadata_file = _resolve_worldstereo_export_paths(
            export_path,
            model_type,
            precision,
            turbo_lora,
        )
        if os.path.exists(output_file) and not overwrite:
            raise FileExistsError(f"Export file already exists: {output_file}")
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        transformer_dir, base_model_dir, _ = _download_worldstereo_components(model_type, include_moge=False)
        transformer_dir = transformer_dir.replace("\\", "/")

        current_dir = os.path.dirname(os.path.dirname(__file__))
        with _temporary_worldstereo_runtime_patches(patch_diffusers=True):
            WorldStereo = _import_worldstereo_class(current_dir)
            print(
                f"[WorldStereo Export] Loading {model_type} for single-file export "
                f"(precision={precision}, turbo_lora={turbo_lora}) ..."
            )
            worldstereo = WorldStereo.from_pretrained(
                os.path.dirname(transformer_dir),
                subfolder=model_type,
                device=device,
                model_device=device,
                transformer_only=True,
            )

        pipeline = worldstereo.pipeline
        turbo_lora_path = _apply_worldstereo_turbo_lora(pipeline, turbo_lora, lora_strength)
        actual_precision = _apply_worldstereo_transformer_export_precision(pipeline, precision)

        transformer = pipeline.transformer
        state = _tensor_state_dict_for_safetensors(transformer)
        total_params = sum(t.numel() for t in state.values())
        total_bytes = sum(t.numel() * t.element_size() for t in state.values())
        dtype_summary = {}
        for tensor in state.values():
            dtype_summary[str(tensor.dtype)] = dtype_summary.get(str(tensor.dtype), 0) + tensor.numel()

        metadata = {
            "format": "hyworld2_worldstereo_single_transformer_v1",
            "model_type": model_type,
            "precision": actual_precision,
            "turbo_lora": turbo_lora,
            "lora_strength": str(float(lora_strength)),
            "turbo_lora_path": turbo_lora_path or "",
            "base_model_dir": base_model_dir,
            "worldstereo_transformer_dir": transformer_dir,
            "num_tensors": str(len(state)),
            "num_params": str(total_params),
            "tensor_bytes": str(total_bytes),
            "dtype_summary": json.dumps(dtype_summary, sort_keys=True),
            "note": "Contains WorldStereo transformer only: base WAN transformer plus WorldStereo weights. T5/CLIP/VAE are not included.",
        }

        print(f"[WorldStereo Export] Saving single transformer checkpoint: {output_file}")
        save_file(state, output_file, metadata=metadata)
        if write_metadata_json:
            with open(metadata_file, "w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2)

        del state
        del pipeline
        del worldstereo
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        info = (
            f"exported={output_file}\n"
            f"metadata={metadata_file if write_metadata_json else ''}\n"
            f"precision={actual_precision}\n"
            f"turbo_lora={turbo_lora}\n"
            f"tensors={metadata['num_tensors']}\n"
            f"params={metadata['num_params']}\n"
            f"bytes={metadata['tensor_bytes']}\n"
            f"dtypes={metadata['dtype_summary']}"
        )
        print(f"[WorldStereo Export] Done:\n{info}")
        return (info,)


# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI registration
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "VNCCS_LoadWorldStereoModel":    VNCCS_LoadWorldStereoModel,
    "VNCCS_CameraTrajectoryBuilder": VNCCS_CameraTrajectoryBuilder,
    "VNCCS_WorldStereoGenerate":     VNCCS_WorldStereoGenerate,
    "VNCCS_ExportWorldStereoSingleModel": VNCCS_ExportWorldStereoSingleModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_LoadWorldStereoModel":    "Load WorldStereo Model",
    "VNCCS_CameraTrajectoryBuilder": "Camera Trajectory Builder",
    "VNCCS_WorldStereoGenerate":     "WorldStereo Generate",
    "VNCCS_ExportWorldStereoSingleModel": "Export WorldStereo Single Model",
}
