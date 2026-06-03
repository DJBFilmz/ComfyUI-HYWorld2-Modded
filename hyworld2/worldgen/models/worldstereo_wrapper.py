"""
WorldStereo unified inference class.

Bundles all sub-models (transformer, text/image encoders, VAE) and the
matching inference pipeline under a single diffusers-style interface::

    worldstereo = WorldStereo.from_pretrained(
        "/path/to/checkpoint_root",
        device=device,
    )
    output = worldstereo(**pipeline_inputs)

Hugging Face format expects ``config.json`` plus ``model.safetensors``
in the same directory.

The config must include a ``model_type`` field with one of the
supported values:

* ``worldstereo-camera``      – keyframe + camera control
* ``worldstereo-memory``      – keyframe + camera control + GGM + SSM
* ``worldstereo-memory-dmd``  – DMD (distribution matching distillation) mode
"""

from __future__ import annotations

import gc
import inspect
import json
import os
import types
from contextlib import contextmanager
from typing import Any

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DIFFUSERS_VERBOSITY"] = "error"

import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import load as load_safetensors_bytes
from safetensors.torch import load_file as load_safetensors
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from .attention import WanAttnProcessorSP
from .dmd_scheduler import FlowGeneratorScheduler
from .pipelines.pipeline_dmd_keyframe import RefKFDMDGeneratorPipeline
from .pipelines.pipeline_pcd_keyframe import KFPCDControllerPipeline
from .pipelines.pipeline_ref_keyframe import KFPCDControllerRefPipeline
from .worldstereo import WorldStereoModel, WorldStereoRefSModel
try:
    from ..src.general_utils import rank0_log
except ImportError:
    from src.general_utils import rank0_log

# ── suppress noisy third-party logs ───────────────────────────────────
import logging
import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DIFFUSERS_VERBOSITY"] = "error"

# transformers / diffusers print a wall of "Some weights were not
# initialized / unexpected keys" on every load.  We already inspect
# load_state_dict results ourselves in worldstereo_wrapper.py, so
# silence their own reporting.
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("diffusers.modeling_utils").setLevel(logging.ERROR)

# huggingface_hub HTTP request logs (newer versions use httpx as the HTTP client)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("filelock").setLevel(logging.ERROR)

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

from diffusers.utils import logging as diffusers_logging
diffusers_logging.set_verbosity_error()

# torch.compile / inductor verbose output
logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
logging.getLogger("torch._inductor").setLevel(logging.WARNING)

# misc deprecation / user warnings from HF internals
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="diffusers")
# ──────────────────────────────────────────────────────────────────────

SUPPORTED_MODEL_TYPES = ("worldstereo-camera", "worldstereo-memory", "worldstereo-memory-dmd")
_HYWORLD2_SINGLE_FORMAT = "hyworld2_worldstereo_single_transformer_v1"


class _Int4Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool, group_size: int, device=None):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        padded_in = ((self.in_features + self.group_size - 1) // self.group_size) * self.group_size
        self.register_buffer("weight_packed", torch.empty((self.out_features, padded_in // 2), dtype=torch.uint8, device=device))
        self.register_buffer("weight_scale", torch.empty((self.out_features, padded_in // self.group_size), dtype=torch.float16, device=device))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(self.out_features, device=device))
        else:
            self.register_parameter("bias", None)

    def _dequantize_weight(self, dtype: torch.dtype, device) -> torch.Tensor:
        packed = self.weight_packed.to(device=device)
        unpacked = torch.empty((packed.shape[0], packed.shape[1] * 2), dtype=torch.uint8, device=device)
        unpacked[:, 0::2] = packed & 0x0F
        unpacked[:, 1::2] = packed >> 4
        unpacked = unpacked[:, : self.in_features].to(torch.float32) - 8.0
        scales = self.weight_scale.to(device=device, dtype=torch.float32)
        scales = scales.repeat_interleave(self.group_size, dim=1)[:, : self.in_features]
        return (unpacked * scales).to(dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bias = self.bias.to(device=input.device, dtype=input.dtype) if self.bias is not None else None
        return F.linear(input, self._dequantize_weight(input.dtype, input.device), bias)


def _load_safetensors_cpu(path: str) -> dict[str, torch.Tensor]:
    try:
        return load_safetensors(path, device="cpu")
    except OSError as exc:
        if "1455" not in str(exc) and "paging file" not in str(exc).lower():
            raise
        file_size_gb = os.path.getsize(path) / (1024 ** 3)
        if file_size_gb > 8:
            raise OSError(
                f"Safetensors mmap failed for a huge checkpoint ({file_size_gb:.2f} GB): {path}. "
                "Use a recent ComfyUI build with model mmap support."
            ) from exc
        with open(path, "rb") as f:
            return load_safetensors_bytes(f.read())


def _read_safetensors_metadata(path: str) -> dict[str, str]:
    with safe_open(path, framework="pt", device="cpu") as f:
        return dict(f.metadata() or {})


@contextmanager
def _empty_weights_context():
    try:
        from accelerate import init_empty_weights
    except Exception:
        with torch.device("meta"):
            yield
    else:
        with init_empty_weights():
            yield


def _assign_tensor_to_module(module: torch.nn.Module, key: str, tensor: torch.Tensor) -> None:
    module_path, _, tensor_name = key.rpartition(".")
    parent = module.get_submodule(module_path) if module_path else module
    if tensor_name in parent._parameters:
        old_param = parent._parameters[tensor_name]
        requires_grad = old_param.requires_grad if old_param is not None else False
        parent._parameters[tensor_name] = torch.nn.Parameter(tensor, requires_grad=requires_grad)
        return
    if tensor_name in parent._buffers:
        parent._buffers[tensor_name] = tensor
        return
    raise KeyError(f"{key!r} is not a parameter or buffer in {parent.__class__.__name__}")


def _replace_submodule(module: torch.nn.Module, module_path: str, replacement: torch.nn.Module) -> None:
    parent_path, _, child_name = module_path.rpartition(".")
    parent = module.get_submodule(parent_path) if parent_path else module
    setattr(parent, child_name, replacement)


def _floating_load_dtype(tensor: torch.Tensor, target_dtype: torch.dtype) -> torch.dtype | None:
    if not tensor.is_floating_point() or str(tensor.dtype).startswith("torch.float8"):
        return None
    return target_dtype


def _prepare_int4_linears(module: torch.nn.Module, path: str, *, device) -> int:
    with safe_open(path, framework="pt", device="cpu") as f:
        packed_keys = sorted(key for key in f.keys() if key.endswith(".weight_packed"))
        group_size = int((f.metadata() or {}).get("int4_group_size", "128"))
    replaced = 0
    for packed_key in packed_keys:
        module_name = packed_key[: -len(".weight_packed")]
        original = module.get_submodule(module_name)
        if not isinstance(original, torch.nn.Linear):
            raise TypeError(f"int4 key {packed_key!r} targets {original.__class__.__name__}, not Linear")
        replacement = _Int4Linear(original.in_features, original.out_features, original.bias is not None, group_size, device=device)
        if original.bias is not None:
            replacement.bias.requires_grad = original.bias.requires_grad
        _replace_submodule(module, module_name, replacement)
        replaced += 1
    return replaced


def _stream_safetensors_to_module(module: torch.nn.Module, path: str, *, device, dtype: torch.dtype) -> tuple[list[str], list[str]]:
    expected_keys = set(module.state_dict().keys())
    with safe_open(path, framework="pt", device="cpu") as f:
        checkpoint_keys = set(f.keys())
        load_keys = sorted(checkpoint_keys & expected_keys)
        missing_keys = sorted(expected_keys - checkpoint_keys)
        unexpected_keys = sorted(checkpoint_keys - expected_keys)
        for key in load_keys:
            tensor = f.get_tensor(key)
            target_dtype = _floating_load_dtype(tensor, dtype)
            tensor = tensor.to(device=device) if target_dtype is None else tensor.to(device=device, dtype=target_dtype)
            _assign_tensor_to_module(module, key, tensor)
    return missing_keys, unexpected_keys


def _stream_int4_safetensors_to_module(module: torch.nn.Module, path: str, *, device, dtype: torch.dtype) -> tuple[list[str], list[str]]:
    expected_keys = set(module.state_dict().keys())
    with safe_open(path, framework="pt", device="cpu") as f:
        checkpoint_keys = set(f.keys())
        shape_keys = {key for key in checkpoint_keys if key.endswith(".weight_shape")}
        effective_keys = checkpoint_keys - shape_keys
        load_keys = sorted(effective_keys & expected_keys)
        missing_keys = sorted(expected_keys - effective_keys)
        unexpected_keys = sorted(effective_keys - expected_keys)
        for key in load_keys:
            tensor = f.get_tensor(key)
            if key.endswith(".weight_packed"):
                tensor = tensor.to(device=device)
            else:
                target_dtype = _floating_load_dtype(tensor, dtype)
                tensor = tensor.to(device=device) if target_dtype is None else tensor.to(device=device, dtype=target_dtype)
            _assign_tensor_to_module(module, key, tensor)
    return missing_keys, unexpected_keys


def _constructor_kwargs_from_config(model_cls, config: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(model_cls.__init__)
    valid = {
        name
        for name, param in signature.parameters.items()
        if name != "self" and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    }
    kwargs = {key: value for key, value in config.items() if key in valid}
    kwargs.update(extra)
    return kwargs


def _get_half_dtype() -> torch.dtype:
    """Select the best half-precision dtype based on current GPU capability: bf16 > fp16 > fp32."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    elif torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        return torch.float16
    else:
        return torch.float32


class WorldStereo:
    """Diffusers-style wrapper that owns every sub-model and its pipeline."""

    def __init__(self, pipeline: Any, cfg: Any) -> None:
        self.pipeline = pipeline
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        subfolder: str = "",
        local_files_only: bool = False,
        sp_world_size: int = 1,
        fsdp: bool = False,
        device_mesh=None,
        device: torch.device | None = None,
    ) -> "WorldStereo":
        """
        Build a WorldStereo instance from Hugging Face format
        (``config.json`` + ``model.safetensors``).

        Args:
            repo_id: Model directory or HF repo ID.
            subfolder: Subfolder within the HF repo or local directory. This is equivalent to the `model_type` (e.g., 'worldstereo-camera').
            local_files_only: If True, avoid downloading the file and return the path to the local cached file if it exists.
            sp_world_size: Sequence-Parallel degree (1 = disabled).
            fsdp: Wrap models with PyTorch FSDP.  Requires ``device_mesh``.
            device_mesh: ``DeviceMesh`` with dims ``("rep", "shard")``.
            device: Target CUDA device.
        """
        if os.path.isdir(repo_id):
            json_cfg_path = os.path.join(repo_id, subfolder, "config.json")
            safetensors_path = os.path.join(repo_id, subfolder, "model.safetensors")

            if not os.path.exists(json_cfg_path):
                raise FileNotFoundError(f"config.json not found under {json_cfg_path!r}")
            if not os.path.exists(safetensors_path):
                raise FileNotFoundError(f"model.safetensors not found at {safetensors_path!r}")
        else:
            from huggingface_hub import hf_hub_download
            json_cfg_path = hf_hub_download(
                repo_id=repo_id,
                filename="config.json",
                subfolder=subfolder if subfolder else None,
                local_files_only=local_files_only,
            )
            safetensors_path = hf_hub_download(
                repo_id=repo_id,
                filename="model.safetensors",
                subfolder=subfolder if subfolder else None,
                local_files_only=local_files_only,
            )

        cfg = OmegaConf.create(cls._load_hf_config(json_cfg_path))
        model_weights_path = safetensors_path

        model_type = subfolder
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type {model_type!r}. "
                f"Expected one of {SUPPORTED_MODEL_TYPES}."
            )

        transformer = cls._load_transformer(
            cfg,
            model_type,
            model_weights_path,
            sp_world_size=sp_world_size,
            fsdp=fsdp,
            device_mesh=device_mesh,
            device=device,
        )

        text_encoder, image_clip, vae = cls._load_aux(
            cfg, device=device, device_mesh=device_mesh, fsdp=fsdp, local_files_only=local_files_only
        )
        image_processor = CLIPImageProcessor.from_pretrained(
            cfg.base_model, do_rescale=False, subfolder="image_processor", local_files_only=local_files_only
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, subfolder="tokenizer", local_files_only=local_files_only)

        pipeline = cls._build_pipeline(
            model_type,
            cfg,
            transformer=transformer,
            text_encoder=text_encoder,
            image_clip=image_clip,
            image_processor=image_processor,
            tokenizer=tokenizer,
            vae=vae,
            device=device,
            local_files_only=local_files_only,
        )

        rank0_log(f"WorldStereo ({model_type}) ready.")
        return cls(pipeline=pipeline, cfg=cfg)

    @classmethod
    def from_single_transformer(
        cls,
        repo_id: str,
        single_model_path: str,
        *,
        subfolder: str = "",
        local_files_only: bool = False,
        sp_world_size: int = 1,
        fsdp: bool = False,
        device_mesh=None,
        device: torch.device | None = None,
        model_device: torch.device | str | None = None,
    ) -> "WorldStereo":
        """Build the official worldgen pipeline from a HYWorld2 single-transformer checkpoint."""
        if fsdp:
            raise NotImplementedError("from_single_transformer does not support FSDP yet.")
        if not os.path.exists(single_model_path):
            raise FileNotFoundError(f"Single transformer checkpoint not found: {single_model_path}")

        if os.path.isdir(repo_id):
            json_cfg_path = os.path.join(repo_id, subfolder, "config.json")
        else:
            from huggingface_hub import hf_hub_download
            json_cfg_path = hf_hub_download(
                repo_id=repo_id,
                filename="config.json",
                subfolder=subfolder if subfolder else None,
                local_files_only=local_files_only,
            )
        if not os.path.exists(json_cfg_path):
            raise FileNotFoundError(f"config.json not found under {json_cfg_path!r}")

        cfg = OmegaConf.create(cls._load_hf_config(json_cfg_path))
        model_type = subfolder
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type {model_type!r}. "
                f"Expected one of {SUPPORTED_MODEL_TYPES}."
            )

        model_device = model_device if model_device is not None else device
        transformer = cls._load_single_transformer(
            cfg,
            model_type,
            single_model_path,
            sp_world_size=sp_world_size,
            device=model_device,
        )

        text_encoder, image_clip, vae = cls._load_aux(
            cfg, device=model_device, device_mesh=device_mesh, fsdp=False, local_files_only=local_files_only
        )
        image_processor = CLIPImageProcessor.from_pretrained(
            cfg.base_model, do_rescale=False, subfolder="image_processor", local_files_only=local_files_only
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, subfolder="tokenizer", local_files_only=local_files_only)

        pipeline = cls._build_pipeline(
            model_type,
            cfg,
            transformer=transformer,
            text_encoder=text_encoder,
            image_clip=image_clip,
            image_processor=image_processor,
            tokenizer=tokenizer,
            vae=vae,
            device=device,
            local_files_only=local_files_only,
        )

        rank0_log(f"WorldStereo single transformer ({model_type}) ready.")
        return cls(pipeline=pipeline, cfg=cfg)

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward all arguments to the underlying inference pipeline."""
        return self.pipeline(*args, **kwargs)

    def to(self, device: torch.device) -> "WorldStereo":
        self.pipeline = self.pipeline.to(device)
        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_hf_config(config_json_path: str) -> dict[str, Any]:
        with open(config_json_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        required_keys = ["base_model", "controlnet_cfg"]
        missing = [k for k in required_keys if k not in cfg]
        if missing:
            raise ValueError(
                f"config.json missing required keys: {missing}. "
                "Please use the conversion script to export a valid HF package."
            )

        return cfg

    @staticmethod
    def _load_transformer(
        cfg,
        model_type: str,
        weights_path: str,
        *,
        sp_world_size: int,
        fsdp: bool,
        device_mesh,
        device,
    ):

        half_dtype = _get_half_dtype()
        rank0_log(f"Loading transformer ({model_type})… dtype={half_dtype}")

        if model_type == "worldstereo-camera":
            transformer = WorldStereoModel.from_pretrained(
                cfg.base_model,
                subfolder="transformer",
                controlnet_cfg=cfg.controlnet_cfg,
                torch_dtype=half_dtype,
            )
        else:
            transformer = WorldStereoRefSModel.from_pretrained(
                cfg.base_model,
                subfolder="transformer",
                controlnet_cfg=cfg.controlnet_cfg,
                torch_dtype=half_dtype,
            )

        rank0_log("Building ControlNet…")
        transformer.build_controlnet(load_uni3c=False, freeze_backbone=cfg.freeze_backbone)

        if sp_world_size > 1:
            transformer.sp_size = sp_world_size
            for layer in transformer.controlnet.controlnet_blocks:
                layer.self_attn.processor.sp_size = sp_world_size
            for block in transformer.blocks:
                if model_type == "worldstereo-camera":
                    block.attn1.set_processor(WanAttnProcessorSP(sp_size=sp_world_size))
                else:
                    block.attn1.processor.sp_size = sp_world_size

        rank0_log(f"Loading HF safetensors weights from {weights_path}…")
        weights = load_safetensors(weights_path, device="cpu")

        result = transformer.load_state_dict(weights, strict=False)

        def _summarize_keys(keys: list[str], label: str) -> None:
            if not keys:
                return
            from collections import Counter
            # Count unloaded parameters
            total_params = sum(
                transformer.state_dict()[k].numel()
                for k in keys
                if k in transformer.state_dict()
            )
            # Count occurrence frequency of each field (split by ".") across all keys, take top-2
            field_counter: Counter[str] = Counter()
            for k in keys:
                parts = k.split(".")
                # Skip pure numeric indices (e.g. blocks.0) and common prefixes/suffixes
                field_counter.update(p for p in parts if not p.isdigit())
            top_fields = [f for f, _ in field_counter.most_common(2)]
            # Filter representative keys using top-2 fields (prefer keys that contain both fields)
            repr_keys = sorted([k for k in keys if all(f in k.split(".") for f in top_fields)])
            if not repr_keys:
                repr_keys = sorted(keys)
            sample_keys = repr_keys[:3]
            rank0_log(
                f"{label}: {len(keys)} keys ({total_params / 1e6:.1f}M params), "
                f"top fields: {top_fields}. "
                f"Representative: {sample_keys}"
                + (f" … and {len(keys) - len(sample_keys)} more" if len(keys) > len(sample_keys) else "")
            )
            rank0_log(f"These are frozen backbone weights initialized by the base video model ({cfg.base_model}).")

        _summarize_keys(result.unexpected_keys, "Unexpected keys")
        _summarize_keys(result.missing_keys, "Missing keys")

        if fsdp:
            fsdp_kwargs = dict(
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=half_dtype,
                    reduce_dtype=torch.float32,
                ),
                mesh=device_mesh["rep", "shard"],
                reshard_after_forward=True,
            )
            transformer = transformer.to(half_dtype)
            for layer in transformer.blocks:
                fully_shard(layer, **fsdp_kwargs)
            for layer in transformer.controlnet.controlnet_blocks:
                fully_shard(layer, **fsdp_kwargs)
            fully_shard(transformer, **fsdp_kwargs)
            rank0_log("FSDP wrapping done for transformer.")
        else:
            transformer = transformer.to(device=device)

        gc.collect()
        torch.cuda.empty_cache()
        return transformer.eval()

    @staticmethod
    def _load_single_transformer(
        cfg,
        model_type: str,
        weights_path: str,
        *,
        sp_world_size: int,
        device,
    ):
        half_dtype = _get_half_dtype()
        transformer_config_path = os.path.join(cfg.base_model, "transformer", "config.json")
        if not os.path.exists(transformer_config_path):
            raise FileNotFoundError(
                f"Base transformer config is required for single checkpoint loading: {transformer_config_path}"
            )
        with open(transformer_config_path, "r", encoding="utf-8") as f:
            transformer_config = json.load(f)

        model_cls = WorldStereoModel if model_type == "worldstereo-camera" else WorldStereoRefSModel
        init_kwargs = _constructor_kwargs_from_config(
            model_cls,
            transformer_config,
            {
                "controlnet_cfg": cfg.controlnet_cfg,
                "base_model": cfg.base_model,
            },
        )

        def _configure_sequence_parallel(transformer):
            if sp_world_size <= 1:
                return
            transformer.sp_size = sp_world_size
            for layer in transformer.controlnet.controlnet_blocks:
                layer.self_attn.processor.sp_size = sp_world_size
            for block in transformer.blocks:
                if model_type == "worldstereo-camera":
                    block.attn1.set_processor(WanAttnProcessorSP(sp_size=sp_world_size))
                else:
                    block.attn1.processor.sp_size = sp_world_size

        try:
            metadata = _read_safetensors_metadata(weights_path)
        except Exception as exc:
            metadata = {}
            rank0_log(f"Could not read single checkpoint metadata ({type(exc).__name__}: {exc}).", "WARNING")

        if metadata.get("format") == _HYWORLD2_SINGLE_FORMAT:
            rank0_log(f"Building empty transformer for single checkpoint ({model_type})... dtype={half_dtype}")
            with _empty_weights_context():
                transformer = model_cls(**init_kwargs)
                transformer.build_controlnet(load_uni3c=False, freeze_backbone=cfg.freeze_backbone)
            if metadata.get("precision") == "int4":
                replaced = _prepare_int4_linears(transformer, weights_path, device=device)
                rank0_log(f"Prepared int4 Linear modules: {replaced}")
            _configure_sequence_parallel(transformer)
            if metadata.get("precision") == "int4":
                missing_keys, unexpected_keys = _stream_int4_safetensors_to_module(
                    transformer,
                    weights_path,
                    device=device,
                    dtype=half_dtype,
                )
            else:
                missing_keys, unexpected_keys = _stream_safetensors_to_module(
                    transformer,
                    weights_path,
                    device=device,
                    dtype=half_dtype,
                )
            if missing_keys:
                raise RuntimeError(
                    f"Single checkpoint is missing {len(missing_keys)} tensor(s); first missing key: {missing_keys[0]}"
                )
            if unexpected_keys:
                rank0_log(f"Single checkpoint unexpected keys: {len(unexpected_keys)}", "WARNING")
            gc.collect()
            torch.cuda.empty_cache()
            return transformer.eval()

        rank0_log(f"Building transformer from single checkpoint ({model_type})... dtype={half_dtype}")
        transformer = model_cls(**init_kwargs)
        transformer.build_controlnet(load_uni3c=False, freeze_backbone=cfg.freeze_backbone)
        _configure_sequence_parallel(transformer)
        weights = _load_safetensors_cpu(weights_path)
        result = transformer.load_state_dict(weights, strict=False)
        if result.missing_keys:
            rank0_log(f"Single checkpoint missing keys: {len(result.missing_keys)}", "WARNING")
        if result.unexpected_keys:
            rank0_log(f"Single checkpoint unexpected keys: {len(result.unexpected_keys)}", "WARNING")
        del weights
        transformer = transformer.to(dtype=half_dtype, device=device)
        gc.collect()
        torch.cuda.empty_cache()
        return transformer.eval()

    @staticmethod
    def _load_aux(cfg, *, device, device_mesh, fsdp: bool, local_files_only: bool = False):
        import transformers as _tr
        from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling

        # ---- text encoder ----
        rank0_log("Loading TextEncoder (UMT5)…")
        text_encoder = UMT5EncoderModel.from_pretrained(
            cfg.base_model, subfolder="text_encoder", torch_dtype=torch.float32, local_files_only=local_files_only
        ).eval()
        if _tr.__version__ >= "5.0.0":
            rank0_log("Patching text_encoder.encoder.embed_tokens for transformers>=5.0.0", "WARNING")
            text_encoder.encoder.embed_tokens = text_encoder.shared
        # torch.compile and accelerate CPU-offload hooks do not compose reliably:
        # Dynamo tries to trace the hook's parameter moves and fails inside UMT5.

        # ---- image encoder ----
        rank0_log("Loading ImageEncoder (CLIP)…")
        image_clip = CLIPVisionModel.from_pretrained(
            cfg.base_model, subfolder="image_encoder", torch_dtype=torch.float32, local_files_only=local_files_only
        ).eval()
        if _tr.__version__ >= "5.0.0":
            rank0_log("Patching CLIP vision forward for transformers>=5.0.0", "WARNING")
            clip_vision = getattr(image_clip, "vision_model", image_clip)

            def _clip_vision_forward(self, pixel_values=None, interpolate_pos_encoding=False, **kwargs):
                if pixel_values is None:
                    raise ValueError("pixel_values is required")
                hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
                hidden_states = self.pre_layrnorm(hidden_states)
                encoder_outputs = self.encoder(inputs_embeds=hidden_states, **kwargs)
                pooled_output = self.post_layernorm(encoder_outputs.last_hidden_state[:, 0, :])
                return BaseModelOutputWithPooling(
                    last_hidden_state=encoder_outputs.last_hidden_state,
                    pooler_output=pooled_output,
                    hidden_states=encoder_outputs.hidden_states,
                )

            def _clip_encoder_forward(self, inputs_embeds, attention_mask=None, **kwargs):
                hidden_states = inputs_embeds
                encoder_states = ()
                for layer in self.layers:
                    encoder_states = encoder_states + (hidden_states,)
                    hidden_states = layer(hidden_states, attention_mask, **kwargs)
                encoder_states = encoder_states + (hidden_states,)
                return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states)

            clip_vision.forward = types.MethodType(_clip_vision_forward, clip_vision)
            clip_vision.encoder.forward = types.MethodType(_clip_encoder_forward, clip_vision.encoder)

        # ---- VAE ----
        vae_dtype = _get_half_dtype()
        rank0_log(f"Loading 3D-VAE… dtype={vae_dtype}")
        vae = AutoencoderKLWan.from_pretrained(
            cfg.base_model, subfolder="vae", torch_dtype=vae_dtype, local_files_only=local_files_only
        ).eval()
        vae = torch.compile(vae)

        if fsdp:
            fsdp_kwargs = dict(
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.float32, reduce_dtype=torch.float32,
                ),
                mesh=device_mesh["rep", "shard"],
                reshard_after_forward=True,
            )
            for layer in text_encoder.encoder.block:
                fully_shard(layer, **fsdp_kwargs)
            fully_shard(text_encoder, **fsdp_kwargs)
            rank0_log("FSDP wrapping done for T5.")

            clip_vision = getattr(image_clip, "vision_model", image_clip)
            for layer in clip_vision.encoder.layers:
                fully_shard(layer, **fsdp_kwargs)
            fully_shard(image_clip, **fsdp_kwargs)
            rank0_log("FSDP wrapping done for CLIP.")

            gc.collect()
            torch.cuda.empty_cache()
        else:
            text_encoder = text_encoder.to(device=device)
            image_clip = image_clip.to(device=device)

        vae = vae.to(device=device)
        return text_encoder, image_clip, vae

    @staticmethod
    def _build_pipeline(
        model_type: str,
        cfg,
        *,
        transformer,
        text_encoder,
        image_clip,
        image_processor,
        tokenizer,
        vae,
        device,
        local_files_only: bool = False,
    ):
        common = dict(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            image_encoder=image_clip,
            image_processor=image_processor,
            transformer=transformer,
            vae=vae,
        )
        if model_type == "worldstereo-camera":
            scheduler = UniPCMultistepScheduler.from_pretrained(
                cfg.base_model, subfolder="scheduler", local_files_only=local_files_only
            )
            return KFPCDControllerPipeline(**common, scheduler=scheduler)

        if model_type == "worldstereo-memory":
            scheduler = UniPCMultistepScheduler.from_pretrained(
                cfg.base_model, subfolder="scheduler", local_files_only=local_files_only
            )
            return KFPCDControllerRefPipeline(**common, scheduler=scheduler)

        if model_type == "worldstereo-memory-dmd":
            scheduler = FlowGeneratorScheduler(
                start_timesteps=cfg.dmd_start_steps,
                num_train_timesteps=cfg.dmd_end_steps,
                shift=cfg.gen_shift,
                use_timestep_transform=True,
                dmd_steps=cfg.dmd_steps,
                rank=dist.get_rank(),
            )
            return RefKFDMDGeneratorPipeline(
                **common,
                scheduler=scheduler,
                device=device,
                vae_compile=False,
                vae_compile_mode="max-autotune",
            )

        raise ValueError(f"Unknown model_type: {model_type!r}")
