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
except ImportError:
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
    except ImportError as e:
        print("\n" + "="*80)
        print(f"[WorldStereo DEBUG] Real camera_utils import error: {e}")
        import traceback
        traceback.print_exc()
        print("="*80 + "\n")
        CAMERA_UTILS_AVAILABLE = False

try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_AVAILABLE = False

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
            c2w = camera_backward_forward(c2w, -speed * j)  # negative = forward
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
            c2w = camera_backward_forward(c2w, -radius * j / num_frames)
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
            c2w = camera_backward_forward(c2w, radius * j / num_frames)
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


def _download_worldstereo_components(model_type: str) -> tuple:
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
    wan_vae = os.path.join(base_model_dir, "vae", "diffusion_pytorch_model.safetensors")
    if not os.path.exists(wan_vae):
        print(f"[WorldStereo] Downloading Wan2.1-I2V-14B-480P base model (~40 GB) ...")
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            local_dir=base_model_dir,
        )
        print(f"[WorldStereo] Base model cached: {base_model_dir}")
    else:
        print(f"[WorldStereo] Base model cached: {base_model_dir}")

    # 3. MoGe depth estimator
    moge_dir = os.path.join(base, "MoGe")
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
    PRESETS = ["circular", "forward", "zoom_in", "zoom_out", "aerial", "custom"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "preset": (cls.PRESETS, {"default": "circular"}),
                "num_frames": ("INT", {"default": 25, "min": 4, "max": 81}),
                "radius": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Orbit radius (circular) or travel distance (zoom).",
                    },
                ),
                "speed": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.001,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Per-frame translation for forward preset.",
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
                "precision": (["bf16", "fp8", "fp4"], {
                    "default": "bf16",
                    "tooltip": (
                        "bf16: recommended. "
                        "fp8: transformer weight-only via optimum-quanto. "
                        "fp4: transformer weight-only via optimum-quanto (lower quality)."
                    ),
                }),
                "offload_mode": (["model_cpu_offload", "sequential_cpu_offload", "none"], {
                    "default": "model_cpu_offload",
                    "tooltip": (
                        "model_cpu_offload: move components to CPU between steps. Recommended for 16 GB VRAM. "
                        "sequential_cpu_offload: layer-by-layer, slower but less VRAM. "
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
        precision="bf16",
        offload_mode="model_cpu_offload",
        device="cuda",
    ):
        import sys
        import os

        # Resolve the parent custom node folder (ComfyUI-HYWorld2)
        current_dir = os.path.dirname(os.path.dirname(__file__))

        # Locate the "worldstereo" folder case-insensitively
        worldstereo_path = None
        for name in os.listdir(current_dir):
            if name.lower() == "worldstereo":
                worldstereo_path = os.path.join(current_dir, name)
                break

        if worldstereo_path:
            # 1. Add worldstereo root to sys.path
            if worldstereo_path not in sys.path:
                sys.path.insert(0, worldstereo_path)

            # 2. Append worldstereo's models directory to ComfyUI's 'models' search paths.
            world_models_path = os.path.join(worldstereo_path, "models")
            if "models" in sys.modules:
                models_module = sys.modules["models"]
                if hasattr(models_module, "__path__"):
                    if world_models_path not in models_module.__path__:
                        models_module.__path__.append(world_models_path)

            # 3. Append worldstereo's src directory to the active 'src' search paths to prevent collisions
            world_src_path = os.path.join(worldstereo_path, "src")
            if "src" in sys.modules:
                src_module = sys.modules["src"]
                if hasattr(src_module, "__path__"):
                    if isinstance(src_module.__path__, list):
                        if world_src_path not in src_module.__path__:
                            src_module.__path__.append(world_src_path)
            else:
                # If 'src' is not yet imported, import it now (will resolve to worldstereo/src)
                try:
                    import src
                    if hasattr(src, "__path__") and isinstance(src.__path__, list):
                        if world_src_path not in src.__path__:
                            src.__path__.append(world_src_path)
                except ImportError:
                    pass

        # ── Call download helper to resolve and prepare directory paths ───────
        transformer_dir, base_model_dir, moge_dir = _download_worldstereo_components(model_type)

        # Normalize Windows backslashes to forward slashes to prevent split('/') failures inside WorldStereo
        transformer_dir = transformer_dir.replace("\\", "/")
        base_model_dir = base_model_dir.replace("\\", "/")
        moge_dir = moge_dir.replace("\\", "/")

        # ── Monkey-patch torch.distributed to bypass Windows Gloo network bugs ──
        import torch.distributed as dist
        
        # Set environment variables as a secondary fallback defense
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

        # Dummy Process Group class to satisfy PyTorch's DeviceMesh
        class DummyProcessGroup:
            def __init__(self):
                self.group_name = "dummy_group"
                
            def __getattr__(self, name):
                def _fallback_func(*args, **kwargs):
                    return args[0] if args else None
                return _fallback_func

        # Generic no-op function that returns the first argument (the tensor) if present
        def _dummy_dist_func(*args, **kwargs):
            return args[0] if args else None

        _dummy_pg = DummyProcessGroup()

        dist.is_initialized = lambda: True
        dist.get_rank = lambda *args, **kwargs: 0
        dist.get_world_size = lambda *args, **kwargs: 1
        dist.barrier = lambda *args, **kwargs: None
        dist.new_group = lambda *args, **kwargs: _dummy_pg
        dist.all_reduce = _dummy_dist_func
        dist.broadcast = _dummy_dist_func
        dist.all_gather = lambda tensor_list, tensor, *args, **kwargs: tensor_list
        dist.reduce_scatter = lambda output, input_list, *args, **kwargs: output
        dist.get_backend = lambda *args, **kwargs: "gloo"
        dist.init_process_group = lambda *args, **kwargs: None

        # Deep-patch all loaded torch.distributed submodules to prevent binding leaks
        import sys
        for mod_name, mod in list(sys.modules.items()):
            if mod_name.startswith("torch.distributed") and mod:
                for attr in ["init_process_group", "barrier", "broadcast", "all_reduce", "all_gather", "reduce_scatter", "new_group", "get_backend"]:
                    if hasattr(mod, attr):
                        try:
                            if attr == "new_group":
                                setattr(mod, attr, lambda *args, **kwargs: _dummy_pg)
                            elif attr == "get_backend":
                                setattr(mod, attr, lambda *args, **kwargs: "gloo")
                            elif attr in ["all_reduce", "broadcast"]:
                                setattr(mod, attr, _dummy_dist_func)
                            else:
                                setattr(mod, attr, lambda *args, **kwargs: None)
                        except Exception:
                            pass
                if hasattr(mod, "is_initialized"):
                    try:
                        mod.is_initialized = lambda: True
                    except Exception:
                        pass
                if hasattr(mod, "get_rank"):
                    try:
                        mod.get_rank = lambda *args, **kwargs: 0
                    except Exception:
                        pass
                if hasattr(mod, "get_world_size"):
                    try:
                        mod.get_world_size = lambda *args, **kwargs: 1
                    except Exception:
                        pass
                if hasattr(mod, "get_backend"):
                    try:
                        mod.get_backend = lambda *args, **kwargs: "gloo"
                    except Exception:
                        pass

        # ── Monkey-patch Diffusers to bypass WorldStereo's required 'device' parameter bug ──
        import diffusers.pipelines.pipeline_utils
        _orig_get_signature_keys = diffusers.pipelines.pipeline_utils.DiffusionPipeline._get_signature_keys

        @staticmethod
        def patched_get_signature_keys(obj):
            expected_modules, optional_parameters = _orig_get_signature_keys(obj)
            
            # Safely handle expected_modules whether it is a list or a set
            if isinstance(expected_modules, list):
                expected_modules = [x for x in expected_modules if x != "device"]
            elif isinstance(expected_modules, set):
                expected_modules = expected_modules - {"device"}
                
            # Safely handle optional_parameters whether it is a list or a set
            if isinstance(optional_parameters, list):
                if "device" not in optional_parameters:
                    optional_parameters.append("device")
            elif isinstance(optional_parameters, set):
                optional_parameters = optional_parameters | {"device"}
                
            return expected_modules, optional_parameters

        diffusers.pipelines.pipeline_utils.DiffusionPipeline._get_signature_keys = patched_get_signature_keys

        from models.worldstereo_wrapper import WorldStereo

        # Resolve MoGeModel class import based on your installed MoGe library version
        try:
            from moge.model.v2 import MoGeModel
        except ImportError:
            try:
                from moge.model.v1 import MoGeModel
            except ImportError:
                from moge.model import MoGeModel  

        # ── Load WorldStereo pipeline ─────────────────────────────────────────
        print(f"[WorldStereo] Loading pipeline (model_type={model_type}, precision={precision}) ...")
        parent_dir = os.path.dirname(transformer_dir)
        worldstereo = WorldStereo.from_pretrained(
            parent_dir,
            subfolder=model_type,
            device=device
        )
        pipeline = worldstereo.pipeline

        # ── Apply precision to transformer ────────────────────────────────────
        if precision == "bf16":
            pipeline.transformer.to(torch.bfloat16)
            if hasattr(pipeline, "vae"):
                pipeline.vae.to(torch.bfloat16)

        elif precision == "fp8":
            try:
                from optimum.quanto import quantize, freeze, qfloat8_e4m3fn
                pipeline.transformer.to(torch.bfloat16)
                quantize(pipeline.transformer, weights=qfloat8_e4m3fn)
                freeze(pipeline.transformer)
                print("[WorldStereo] fp8 weight quantization applied")
            except ImportError:
                raise ImportError("optimum-quanto required for fp8: pip install optimum-quanto")

        elif precision == "fp4":
            try:
                from optimum.quanto import quantize, freeze, qint4
                pipeline.transformer.to(torch.bfloat16)
                quantize(pipeline.transformer, weights=qint4)
                freeze(pipeline.transformer)
                print("[WorldStereo] fp4 (qint4) weight quantization applied")
            except ImportError:
                raise ImportError("optimum-quanto required for fp4: pip install optimum-quanto")

        # ── Apply offloading ──────────────────────────────────────────────────
        if device == "cuda":
            if offload_mode == "model_cpu_offload":
                pipeline.enable_model_cpu_offload()
                print("[WorldStereo] model_cpu_offload enabled")
            elif offload_mode == "sequential_cpu_offload":
                pipeline.enable_sequential_cpu_offload()
                print("[WorldStereo] sequential_cpu_offload enabled")

        # ── Load MoGe on CPU ─────────────────────────────────────────────────
        print("[WorldStereo] Loading MoGe depth estimator ...")
        
        # Point directly to the model.pt file inside the cached directory
        actual_moge_path = os.path.join(moge_dir, "model.pt")
        if not os.path.exists(actual_moge_path):
            actual_moge_path = moge_dir  # Fallback if structure varies
            
        moge_model = MoGeModel.from_pretrained(actual_moge_path).eval()
        print("[WorldStereo] MoGe loaded (CPU)")

        print("[WorldStereo] Pipeline ready")
        return ({
            "worldstereo": worldstereo,
            "pipeline":    pipeline,
            "moge":        moge_model,
            "device":      device,
            "model_type":  model_type,
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
    img_tensor_01 = T.ToTensor()(image_pil).unsqueeze(0).float() # [1, 3, H, W]

    # 2. Depth via MoGe
    torch_device = torch.device(device)
    moge_model = moge_model.to(torch_device)
    with torch.no_grad():
        depth_output = moge_model.infer(
            img_tensor.unsqueeze(0).to(torch_device)
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
    render_video = render_rgbs_t.unsqueeze(0).permute(0, 2, 1, 3, 4).to(torch_device)   # [1, 3, N, H, W]
    render_mask  = render_masks_t.unsqueeze(0).permute(0, 2, 1, 3, 4).to(torch_device)  # [1, 1, N, H, W]

    # 6. Camera embedding [1, 6, N, H, W]
    camera_emb = get_camera_embedding(
        intrinsic=intrs.to(torch_device),  # [N, 3, 3]
        extrinsic=c2ws.to(torch_device),   # [N, 4, 4] C2W (is_w2c=False)
        f=N, h=height, w=width,
        normalize=True,
        is_w2c=False,
    )

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
                    "tooltip": "0 = auto (4 for memory-dmd, 20 for others).",
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 5.0, "min": 1.0, "max": 20.0, "step": 0.5,
                }),
                "seed": ("INT", {
                    "default": -1, "min": -1, "max": 2**31 - 1,
                    "tooltip": "-1 = random.",
                }),
                "negative_prompt": ("STRING", {"default": ""}),
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
        base_camera_poses=None,
        base_camera_intrinsics=None,
    ):
        pipeline   = model["pipeline"]
        moge_model = model["moge"]
        device     = model["device"]
        model_type = model["model_type"]

        c2ws  = trajectory["c2ws"]    # [N, 4, 4]
        intrs = trajectory["intrs"]   # [N, 3, 3]
        W     = trajectory["width"]
        H     = trajectory["height"]
        N     = c2ws.shape[0]

        if num_inference_steps == 0:
            num_inference_steps = 4 if "dmd" in model_type else 20

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
            pipeline_inputs = _prepare_pipeline_inputs(
                image_pil=img_pil,
                c2ws=c2ws_abs,
                intrs=intrs_abs,
                moge_model=moge_model,
                device=device,
                width=W,
                height=H,
            )

            # ── Generator Setup ──
            generator = None
            if seed >= 0:
                generator = torch.Generator(device=device).manual_seed(seed + b)

            # ── Diffusion Generation ──
            print(f"[WorldStereo] Slice {b + 1} executing inference for {N} camera frames...")
            with torch.autocast(device, dtype=torch.bfloat16):
                output = pipeline(
                    **pipeline_inputs,
                    prompt=prompt if prompt else "",
                    negative_prompt=negative_prompt if negative_prompt else None,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    output_type="pt",
                )

            # Clean cache immediately to keep low VRAM overhead on sequential slices
            torch.cuda.empty_cache()

            # ── Extract Generated Frames ──
            frames = output.frames[0].float().cpu().clamp(0.0, 1.0)  # [N, 3, H, W]
            slice_video_frames = frames.permute(0, 2, 3, 1)          # [N, H, W, 3]

            all_video_frames.append(slice_video_frames)
            all_camera_poses.append(c2ws_abs.cpu().float())
            all_camera_intrinsics.append(intrs_abs.cpu().float())

        # Concatenate sequences of all processed slices into unified tensors
        video_frames     = torch.cat(all_video_frames, dim=0)       # [B * N, H, W, 3]
        camera_poses_out = torch.cat(all_camera_poses, dim=0)       # [B * N, 4, 4]
        camera_intrs_out = torch.cat(all_camera_intrinsics, dim=0)  # [B * N, 3, 3]

        print(f"[WorldStereo Batch Loop] Complete. Total of {video_frames.shape[0]} frames prepared for 3D fusion.")

        return video_frames, camera_poses_out, camera_intrs_out


# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI registration
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "VNCCS_LoadWorldStereoModel":    VNCCS_LoadWorldStereoModel,
    "VNCCS_CameraTrajectoryBuilder": VNCCS_CameraTrajectoryBuilder,
    "VNCCS_WorldStereoGenerate":     VNCCS_WorldStereoGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_LoadWorldStereoModel":    "Load WorldStereo Model",
    "VNCCS_CameraTrajectoryBuilder": "Camera Trajectory Builder",
    "VNCCS_WorldStereoGenerate":     "WorldStereo Generate",
}
