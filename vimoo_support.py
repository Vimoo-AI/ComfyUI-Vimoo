"""Vimoo model configuration and ComfyUI execution adapter."""

from __future__ import annotations

import torch

import comfy.conds
import comfy.latent_formats
import comfy.model_base
import comfy.supported_models_base
import comfy.utils

from .vimoo_model import VimooModel
from .vimoo_vae import VimooLatentFormat


class VimooPrecisionCondition(comfy.conds.CONDRegular):
    def concat(self, others):
        # Preserve model-specific precision through BaseModel conditioning.
        return {"states": super().concat(others)}


class Vimoo(comfy.supported_models_base.BASE):
    unet_config = {"image_model": "vimoo"}
    unet_extra_config = {}
    latent_format = VimooLatentFormat
    sampling_settings = {"shift": 12.0}
    memory_usage_factor = 4.0
    # bf16 first: it is the checkpoint's native precision and the default compute
    # dtype when a quantized file leaves no single weight dtype to follow.
    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]
    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        return VimooBaseModel(self, comfy.model_base.ModelType.FLOW, device=device)


class VimooBaseModel(comfy.model_base.BaseModel):
    def __init__(self, model_config, model_type=comfy.model_base.ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=VimooModel)

    def _apply_model(
        self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kwargs
    ):
        options = transformer_options or {}
        if not options.get("vimoo_standard_t2v"):
            return super()._apply_model(x, t, c_concat, c_crossattn, control, options, **kwargs)
        # Standard T2V concatenates BF16 state with FP32 zero conditioning,
        # while Qwen stays BF16 and ByT5's precision condition stays FP32.
        xc = self.model_sampling.calculate_input(t, x.bfloat16())
        if c_concat is not None:
            xc = torch.cat((xc, c_concat.to(device=x.device, dtype=torch.float32)), dim=1)
        else:
            xc = xc.float()
        context = None if c_crossattn is None else c_crossattn.to(device=x.device, dtype=torch.bfloat16)
        extra_conds = {
            name: value.to(x.device) if torch.is_tensor(value) else value for name, value in kwargs.items()
        }
        timestep = self.model_sampling.timestep(t).to(torch.int64)
        options = options.copy()
        options["prefetch_dynamic_vbars"] = (
            self.current_patcher is not None and self.current_patcher.is_dynamic()
        )
        model_output = self.diffusion_model(
            xc, timestep, context=context, control=control, transformer_options=options, **extra_conds
        )
        return self.model_sampling.calculate_denoised(t, model_output.float(), x)

    def process_timestep(self, timestep, **kwargs):
        return timestep.trunc()

    def concat_cond(self, **kwargs):
        noise = kwargs.get("noise", None)
        extra_channels = self.diffusion_model.img_in.proj.weight.shape[1] - noise.shape[1] - 1
        image = kwargs.get("concat_latent_image", None)
        device = kwargs["device"]

        if image is None:
            image = torch.zeros(
                (noise.shape[0], extra_channels) + noise.shape[2:],
                dtype=noise.dtype,
                layout=noise.layout,
                device=noise.device,
            )
        else:
            image = comfy.utils.common_upscale(
                image.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center"
            )
            image = self.process_latent_in(image[:, :extra_channels, : noise.shape[2]])
            if image.shape[2] < noise.shape[2]:
                image = torch.nn.functional.pad(image, (0, 0, 0, 0, 0, noise.shape[2] - image.shape[2]))
            image = comfy.utils.resize_to_batch_size(image, noise.shape[0])

        mask = kwargs.get("concat_mask", kwargs.get("denoise_mask", None))
        if mask is None:
            # Only one mask channel is consumed.  Allocating the full latent
            # shape and slicing it transiently wastes 15/16 of this buffer.
            mask = noise.new_zeros((noise.shape[0], 1, *noise.shape[2:]))
        else:
            mask = 1.0 - mask
            mask = comfy.utils.common_upscale(
                mask.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center"
            )
            mask = mask[:, :, : noise.shape[2]]
            if mask.shape[2] < noise.shape[2]:
                mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, noise.shape[2] - mask.shape[2]))
            mask = comfy.utils.resize_to_batch_size(mask, noise.shape[0])

        return torch.cat((image, mask), dim=1)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        patcher = getattr(self, "current_patcher", None)
        model_options = getattr(patcher, "model_options", {})
        native_masks = model_options.get("vimoo_text_encoder_precision_profile") == "lite_compact"

        def mask_condition(value):
            if native_masks:
                if value.dtype != torch.bool:
                    raise ValueError("Native Patchify conditioning masks must retain BOOL precision")
                # BaseModel casts plain BOOL tensors to the model compute dtype.
                # Use the existing precision container through native batching.
                return VimooPrecisionCondition(value)
            return comfy.conds.CONDRegular(value)

        for name in ("observed_first_frame_timesteps", "vimoo_anchor_first_frame"):
            if kwargs.get(name) is not None:
                condition = (
                    VimooPrecisionCondition
                    if name == "observed_first_frame_timesteps"
                    else comfy.conds.CONDRegular
                )
                out[name] = condition(kwargs[name])
        fps = kwargs.get("fps")
        if fps is not None:
            out["fps"] = comfy.conds.CONDConstant(fps)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            out["attention_mask"] = mask_condition(attention_mask)

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out["c_crossattn"] = comfy.conds.CONDRegular(cross_attn)

        conditioning_byt5small = kwargs.get("conditioning_byt5small", None)
        if conditioning_byt5small is not None:
            out["txt_byt5"] = VimooPrecisionCondition(conditioning_byt5small)
        byt5_mask = kwargs.get("vimoo_byt5_mask")
        if byt5_mask is not None:
            out["byt5_mask"] = mask_condition(byt5_mask)

        clip_vision_output = kwargs.get("clip_vision_output", None)
        if clip_vision_output is not None:
            out["clip_fea"] = comfy.conds.CONDRegular(clip_vision_output.last_hidden_state)
        if kwargs.get("vimoo_vision_mask") is not None:
            out["vision_mask"] = mask_condition(kwargs["vimoo_vision_mask"])

        if self.diffusion_model.uses_cpu_frequencies():
            text_length = cross_attn.shape[1]
            if conditioning_byt5small is not None:
                text_length += conditioning_byt5small.shape[1]
            if self.diffusion_model.vision_in is not None:
                text_length += (
                    729 if clip_vision_output is None else clip_vision_output.last_hidden_state.shape[1]
                )
            image_rope, frequencies = self.diffusion_model.prepare_math_inputs(
                kwargs["noise"].shape, text_length, kwargs["device"], fps=fps
            )
            out["precise_rope"] = VimooPrecisionCondition(image_rope)
            out["precomputed_frequencies"] = VimooPrecisionCondition(frequencies)

        return out
