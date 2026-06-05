import math

import torch


PANO_POSITIVE_PREFIX = (
    "Create a **ERP** panoramic expansion of the provided image. "
    "Preserve the original style, lighting, and fine details seamlessly "
    "throughout the extended areas, extend according to: "
)
PANO_POSITIVE_SUFFIX = " 8k UHD, masterpiece, razor-sharp details."

PANO_NEGATIVE_PROMPT = (
    "low resolution, low quality, blurry, chaotic background, distorted structure, "
    "blurred texture, object fusion, messy composition, overexposed highlights, "
    "AI artifacts, distorted face, oversized objects, oversized buildings, "
    "close-up shot, compressed perspective, wrong proportions, cars, vehicles, "
    "tree leaves at the top edge of the image"
)

QWEN_PANO_LLAMA_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter or modify the image. Generate a new image "
    "that meets the user's requirements while maintaining consistency with the original input where appropriate."
    "<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024


def build_positive_prompt(prompt: str, use_official_prompt: bool = True) -> str:
    prompt = (prompt or "").strip()
    if not use_official_prompt:
        return prompt
    return (PANO_POSITIVE_PREFIX + prompt + PANO_POSITIVE_SUFFIX).strip()


def build_negative_prompt(negative_prompt: str = "") -> str:
    negative = PANO_NEGATIVE_PROMPT
    negative_prompt = (negative_prompt or "").strip()
    if negative_prompt:
        negative = f"{negative} {negative_prompt}"
    return negative.strip()


def crop_border_image(image: torch.Tensor, crop_border: float) -> torch.Tensor:
    if crop_border <= 0:
        return image
    if crop_border >= 0.49:
        raise ValueError("crop_border must be lower than 0.49")

    _, h, w, _ = image.shape
    crop_x = int(round(w * crop_border))
    crop_y = int(round(h * crop_border))
    if crop_x <= 0 and crop_y <= 0:
        return image
    return image[:, crop_y : h - crop_y, crop_x : w - crop_x, :].contiguous()


def resize_to_pixel_budget(image: torch.Tensor, pixel_budget: int, *, multiple_of: int | None = None) -> torch.Tensor:
    import comfy.utils

    samples = image.movedim(-1, 1)
    _, _, h, w = samples.shape
    scale_by = math.sqrt(pixel_budget / (w * h))
    width = round(w * scale_by)
    height = round(h * scale_by)
    if multiple_of:
        width = max(multiple_of, round(width / multiple_of) * multiple_of)
        height = max(multiple_of, round(height / multiple_of) * multiple_of)
    resized = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
    return resized.movedim(1, -1)[:, :, :, :3].contiguous()


def encode_qwen_pano_conditioning(
    *,
    clip,
    vae,
    image: torch.Tensor,
    positive_prompt: str,
    negative_prompt: str,
    ref_latents_method: str = "index",
):
    import node_helpers

    condition_image = resize_to_pixel_budget(image, CONDITION_IMAGE_SIZE)
    vae_image = resize_to_pixel_budget(image, VAE_IMAGE_SIZE, multiple_of=8)
    reference_latent = vae.encode(vae_image)

    image_prompt = "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"

    positive_tokens = clip.tokenize(
        image_prompt + positive_prompt,
        images=[condition_image],
        llama_template=QWEN_PANO_LLAMA_TEMPLATE,
    )
    positive = clip.encode_from_tokens_scheduled(positive_tokens)
    positive = node_helpers.conditioning_set_values(
        positive,
        {
            "reference_latents": [reference_latent],
            "reference_latents_method": ref_latents_method,
        },
        append=True,
    )

    negative_tokens = clip.tokenize(
        image_prompt + negative_prompt,
        images=[condition_image],
        llama_template=QWEN_PANO_LLAMA_TEMPLATE,
    )
    negative = clip.encode_from_tokens_scheduled(negative_tokens)
    negative = node_helpers.conditioning_set_values(
        negative,
        {
            "reference_latents": [reference_latent],
            "reference_latents_method": ref_latents_method,
        },
        append=True,
    )

    return positive, negative


def make_empty_qwen_latent(width: int, height: int, batch_size: int, device=None):
    import comfy.model_management

    if device is None:
        device = comfy.model_management.intermediate_device()
    return {
        "samples": torch.zeros([batch_size, 16, 1, height // 8, width // 8], device=device)
    }


def sample_latent(
    *,
    model,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    positive,
    negative,
    latent: dict,
    denoise: float = 1.0,
):
    import comfy.sample
    import comfy.utils
    import latent_preview

    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model,
        latent_image,
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )

    batch_inds = latent["batch_index"] if "batch_index" in latent else None
    noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)
    noise_mask = latent.get("noise_mask", None)
    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    samples = comfy.sample.sample(
        model,
        noise,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=denoise,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )

    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    return out


def circular_blend_edges_tensor(image: torch.Tensor, blend_width: int = 32, crop_right_edge: bool = True) -> torch.Tensor:
    if blend_width <= 0:
        if image.ndim == 5 and image.shape[1] == 1:
            return image[:, 0]
        return image
    if image.ndim == 5:
        if image.shape[1] != 1:
            raise ValueError(f"Expected a single-frame IMAGE/VIDEO tensor, got {tuple(image.shape)}")
        image = image[:, 0]
    if image.ndim != 4:
        raise ValueError(f"Expected IMAGE tensor [B,H,W,C], got {tuple(image.shape)}")

    _, _, width, _ = image.shape
    if blend_width >= width:
        raise ValueError("blend_width must be smaller than image width")

    out = image.clone()
    weights = torch.linspace(
        0.0,
        1.0 - (1.0 / blend_width),
        blend_width,
        dtype=out.dtype,
        device=out.device,
    ).view(1, 1, blend_width, 1)

    left = out[:, :, :blend_width, :].clone()
    right = out[:, :, width - blend_width : width, :].clone()
    out[:, :, :blend_width, :] = right * (1.0 - weights) + left * weights

    if crop_right_edge:
        out = out[:, :, : width - blend_width, :]
    return out.contiguous()
