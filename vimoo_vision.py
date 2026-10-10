"""Vimoo SigLIP vision encoding with profile-specific preprocessing.

The reference runs `SiglipImageProcessor` (torchvision backend: uint8 tensor,
bicubic antialias resize to 384x384, rescale 1/255, normalize 0.5/0.5) and the
so400m tower in fp16 on weights rounded fp32 -> bf16 -> fp16. The processor is
constructed with Vimoo's constants and the torchvision backend explicitly; ComfyUI's stock `encode_image` (uint8
re-quantization, fp32 compute) measurably shifts the tokens.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps
import torch
from types import MethodType
from functools import partial
from .vimoo_precision import add_method_patches
from . import vimoo_attention  # noqa: F401 - registers the selected backends
from comfy.ldm.modules.attention import get_attention_function
import comfy.clip_vision
import comfy.model_management

SIGLIP_SIZE = 384

_processors = {}


def validate_source_image_decode(value):
    if value not in (None, "pil"):
        raise ValueError(f"Unsupported Vimoo image decoder: {value!r}")
    return value


def load_pil_image(path):
    """Read an oriented RGB frame without video decoding."""
    with Image.open(path) as image:
        pixels = np.asarray(ImageOps.exif_transpose(image).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(pixels)[None]


def _get_processor(backend="torchvision"):
    if backend not in ("torchvision", "pil"):
        raise ValueError(f"Unknown SigLIP image processor: {backend!r}")
    if backend not in _processors:
        from transformers import SiglipImageProcessor, SiglipImageProcessorFast

        processor = SiglipImageProcessor if backend == "pil" else SiglipImageProcessorFast
        _processors[backend] = processor(
            size={"height": SIGLIP_SIZE, "width": SIGLIP_SIZE},
            resample=3,  # bicubic
            do_rescale=True,
            rescale_factor=1 / 255.0,
            do_normalize=True,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
        )
    return _processors[backend]


def _vision_attention(module, x, mask=None, optimized_attention=None, attention_backend=None):
    return type(module).forward(module, x, mask, attention_backend)


def _vision_patcher(clip_vision, attention_backend="comfy"):
    if attention_backend not in ("comfy", "precise"):
        raise ValueError(f"Unsupported Vimoo encoder attention backend: {attention_backend!r}")
    patcher = clip_vision.patcher.clone()
    for name, module in clip_vision.model.vision_model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            patcher.add_object_patch(f"vision_model.{name}.eps", 1e-6)
    if attention_backend == "precise":
        attention = get_attention_function("vimoo_siglip")
        methods = {
            f"vision_model.encoder.layers.{i}.self_attn.forward": MethodType(
                partial(_vision_attention, attention_backend=attention), layer.self_attn
            )
            for i, layer in enumerate(clip_vision.model.vision_model.encoder.layers)
        }
        add_method_patches(patcher, "vimoo_vision_attention", methods)
    return patcher


def encode_vimoo_vision(
    clip_vision, image_0_1: torch.Tensor, attention_backend="comfy", processor_backend="torchvision"
) -> comfy.clip_vision.Output:
    """image_0_1: [B,H,W,3] float in 0..1 (ComfyUI IMAGE layout)."""
    patcher = _vision_patcher(clip_vision, attention_backend)
    comfy.model_management.load_model_gpu(patcher)
    pixels = _get_processor(processor_backend)(
        images=[np.clip(image.cpu().numpy() * 255.0, 0, 255).round().astype(np.uint8) for image in image_0_1],
        return_tensors="pt",
    )["pixel_values"]

    out = clip_vision.model(
        pixel_values=pixels.to(clip_vision.load_device).to(torch.float16), intermediate_output="all"
    )
    output = comfy.clip_vision.Output()
    states = out[0].bfloat16() if processor_backend == "pil" else out[0]
    output.last_hidden_state = states.to(comfy.model_management.intermediate_device())
    return output
