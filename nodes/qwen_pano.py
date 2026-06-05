import torch

from hyworld2.panogen.qwen_pano_comfy import (
    build_negative_prompt,
    build_positive_prompt,
    circular_blend_edges_tensor,
    crop_border_image,
    encode_qwen_pano_conditioning,
    make_empty_qwen_latent,
    sample_latent,
)


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


def _ensure_image_batch(image):
    if image.ndim == 3:
        return image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"Expected IMAGE tensor [B,H,W,C], got shape {tuple(image.shape)}")
    return image


def _crop_border(image, crop_border):
    if crop_border <= 0:
        return image
    if crop_border >= 0.49:
        raise ValueError("crop_border must be lower than 0.49")

    _, h, w, _ = image.shape
    crop_x = int(round(w * crop_border))
    crop_y = int(round(h * crop_border))
    if crop_x <= 0 and crop_y <= 0:
        return image
    return image[:, crop_y : h - crop_y, crop_x : w - crop_x, :]


class HYWorld2QwenPanoEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scene_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "a clean bright indoor scene",
                    },
                ),
                "width": ("INT", {"default": 1952, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 960, "min": 64, "max": 8192, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "canvas_value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "crop_border": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.49, "step": 0.001}),
                "use_official_prompt": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("reference_image", "wide_canvas_image", "positive_prompt", "negative_prompt")
    FUNCTION = "encode"
    CATEGORY = "VNCCS/HY-World/Qwen Pano"

    def encode(
        self,
        image,
        scene_prompt,
        width,
        height,
        batch_size,
        canvas_value,
        crop_border,
        use_official_prompt,
        negative_prompt="",
    ):
        image = _ensure_image_batch(image)
        reference = _crop_border(image, crop_border).contiguous()

        if reference.shape[0] != batch_size:
            if reference.shape[0] == 1:
                reference = reference.repeat(batch_size, 1, 1, 1)
            else:
                batch_size = reference.shape[0]

        dtype = reference.dtype
        device = reference.device
        canvas = torch.full((batch_size, height, width, 3), float(canvas_value), dtype=dtype, device=device)

        scene_prompt = (scene_prompt or "").strip()
        if use_official_prompt:
            positive = (PANO_POSITIVE_PREFIX + scene_prompt + PANO_POSITIVE_SUFFIX).strip()
        else:
            positive = scene_prompt

        negative = PANO_NEGATIVE_PROMPT
        if negative_prompt:
            negative = (negative + " " + negative_prompt.strip()).strip()

        return (reference, canvas, positive, negative)


class HYWorld2QwenPanoSeamBlend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blend_width": ("INT", {"default": 32, "min": 0, "max": 512, "step": 1}),
                "crop_right_edge": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "blend"
    CATEGORY = "VNCCS/HY-World/Qwen Pano"

    def blend(self, image, blend_width, crop_right_edge):
        image = _ensure_image_batch(image)
        if blend_width <= 0:
            return (image,)

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
        return (out.contiguous(),)


class HYWorld2QwenPanoPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "a clean bright indoor scene",
                    },
                ),
                "use_official_prompt": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "build"
    CATEGORY = "VNCCS/HY-World/Qwen Pano"

    def build(self, scene_prompt, use_official_prompt, negative_prompt=""):
        scene_prompt = (scene_prompt or "").strip()
        if use_official_prompt:
            positive = (PANO_POSITIVE_PREFIX + scene_prompt + PANO_POSITIVE_SUFFIX).strip()
        else:
            positive = scene_prompt

        negative = PANO_NEGATIVE_PROMPT
        if negative_prompt:
            negative = (negative + " " + negative_prompt.strip()).strip()
        return (positive, negative)


class HYWorld2QwenPanoGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers

        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "a clean bright indoor scene",
                    },
                ),
                "width": ("INT", {"default": 1952, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 960, "min": 64, "max": 8192, "step": 8}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 40, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blend_width": ("INT", {"default": 32, "min": 0, "max": 512, "step": 1}),
                "crop_border": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.49, "step": 0.001}),
                "ref_latents_method": (["index", "offset", "uxo", "index_timestep_zero"], {"default": "index"}),
                "use_official_prompt": ("BOOLEAN", {"default": True}),
                "crop_right_edge": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "STRING", "STRING")
    RETURN_NAMES = ("image", "latent", "positive_prompt", "negative_prompt")
    FUNCTION = "generate"
    CATEGORY = "VNCCS/HY-World/Qwen Pano"

    def generate(
        self,
        model,
        clip,
        vae,
        image,
        prompt,
        width,
        height,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        blend_width,
        crop_border,
        ref_latents_method,
        use_official_prompt,
        crop_right_edge,
        negative_prompt="",
    ):
        image = _ensure_image_batch(image)
        image = crop_border_image(image[:, :, :, :3], crop_border)
        batch_size = image.shape[0]

        positive_prompt = build_positive_prompt(prompt, use_official_prompt=use_official_prompt)
        full_negative_prompt = build_negative_prompt(negative_prompt)
        positive, negative = encode_qwen_pano_conditioning(
            clip=clip,
            vae=vae,
            image=image,
            positive_prompt=positive_prompt,
            negative_prompt=full_negative_prompt,
            ref_latents_method=ref_latents_method,
        )

        latent = make_empty_qwen_latent(width, height, batch_size)
        samples = sample_latent(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent=latent,
            denoise=denoise,
        )

        decoded = vae.decode(samples["samples"])
        decoded = circular_blend_edges_tensor(decoded, blend_width=blend_width, crop_right_edge=crop_right_edge)
        return (decoded, samples, positive_prompt, full_negative_prompt)


NODE_CLASS_MAPPINGS = {
    "HYWorld2QwenPanoEncoder": HYWorld2QwenPanoEncoder,
    "HYWorld2QwenPanoSeamBlend": HYWorld2QwenPanoSeamBlend,
    "HYWorld2QwenPanoPrompt": HYWorld2QwenPanoPrompt,
    "HYWorld2QwenPanoGenerate": HYWorld2QwenPanoGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HYWorld2QwenPanoEncoder": "HY-World Qwen Pano Encoder",
    "HYWorld2QwenPanoSeamBlend": "HY-World Qwen Pano Seam Blend",
    "HYWorld2QwenPanoPrompt": "HY-World Qwen Pano Prompt",
    "HYWorld2QwenPanoGenerate": "HY-World Qwen Pano Generate",
}
