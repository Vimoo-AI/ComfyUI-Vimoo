"""Distilled LoRA loading and recipe-bound stochastic T2V sampling."""

import torch
import folder_paths
import comfy.model_sampling
import comfy.sample
import comfy.samplers
import comfy.utils

from .vimoo_lora import load_distilled_lora
from .vimoo_sampler import validate_distilled_sigmas, video_noise, sample_vimoo_distilled


_IMAGE_CONDITION_KEYS = ("clip_vision_output", "concat_latent_image", "concat_mask")


class DistilledModelSampling(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
    def __init__(self, model_config=None, *, sigmas):
        super().__init__(model_config)
        self.register_buffer("sample_sigmas", sigmas.detach().clone())
        self.register_buffer("sample_timesteps", (sigmas * 1000).float())

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        # Distilled T2V starts from raw CUDA BF16 noise even when its first
        # post-shift sigma is below one. The sampler rejects image latents and
        # masks, so the empty latent contributes nothing.
        return noise

    def timestep(self, sigma):
        # ComfyUI evaluates the model with FP32 sigmas. Recover the timesteps
        # from the connected FP64 schedule to preserve Diffusers' rounding.
        grid = self.sample_sigmas.to(device=sigma.device, dtype=sigma.dtype)
        index = (sigma[..., None] - grid).abs().argmin(dim=-1)
        return self.sample_timesteps.to(sigma.device)[index]

    def calculate_denoised(self, sigma, model_output, model_input):
        # Diffusers reconstructs sigma from the FP32 timestep before FP64 x0 recovery.
        sigma = comfy.model_sampling.reshape_sigma(self.timestep(sigma).double() / 1000, model_input.ndim)
        return (model_input.double() - sigma * model_output.double()).to(torch.bfloat16)


class VimooDistilledLoRALoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "Vimoo"

    def load(self, model, lora_name):
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        return (load_distilled_lora(model, path),)


def sample_distilled(model, positive, latent_image, sigmas, cfg, seed, renoise_seed=1001237):
    recipe = model.model_options.get("vimoo_distilled")
    if not recipe:
        raise ValueError("Load the matching Distilled LoRA before stochastic T2V sampling")
    if recipe.get("task") != "t2v":
        raise ValueError(
            "stochastic_flowmatch requires a T2V adapter; I2V Distilled and Lite TI2V Distilled use their Euler sampler"
        )
    if renoise_seed is None or renoise_seed == -1:
        renoise_seed = seed
    if type(renoise_seed) is not int or not 0 <= renoise_seed <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("ReNoise seed must be -1 or an integer in [0, 2**64 - 1]")
    validate_distilled_sigmas(sigmas, recipe)
    if recipe.get("cfg", 1.0) != cfg or recipe.get("guidance_rescale", 0.0) != 0.0:
        raise ValueError("Distilled guidance must match the LoRA sidecar recipe")
    if model.patches:
        raise ValueError("Do not stack weight patches on the Distilled profile")
    if latent_image.get("noise_mask") is not None or latent_image.get("batch_index") is not None:
        raise ValueError("Distilled requires fresh latents without masks or batch indices")
    for _, extra in positive:
        if "vimoo_byt5_mask" not in extra or "conditioning_byt5small" not in extra:
            raise ValueError("Use Vimoo Text Encode to supply glyph conditioning and its mask")
        if any(extra.get(key) is not None for key in _IMAGE_CONDITION_KEYS):
            raise ValueError("T2V Distilled does not accept image conditioning")
    if not positive:
        raise ValueError("Use Vimoo Text Encode to supply positive conditioning")
    latent = latent_image["samples"]
    if latent.shape[0] != 1 or torch.count_nonzero(latent):
        raise ValueError("Use an empty batch-one Vimoo latent for Distilled")
    device = model.load_device
    if device.type != "cuda" or model.model.get_dtype() != torch.bfloat16:
        raise ValueError("Distilled requires the validated CUDA BF16 base")
    result = model.clone()
    sampling = DistilledModelSampling(model.model.model_config, sigmas=sigmas)
    result.add_object_patch("model_sampling", sampling)
    # Distilled sampling preserves fractional timesteps and runs BF16 without autocast.
    result.add_object_patch("process_timestep", lambda timestep, **kwargs: timestep)
    sigmas = sigmas.detach().to(device="cpu", dtype=torch.float64)
    # Distilled T2V draws on the CUDA generator, unlike the ordinary sampler,
    # which mirrors Diffusers' CPU randn_tensor.  Same seed, different stream: two runs
    # that differ only in sampler_name do NOT start from the same noise.
    noise = video_noise(latent.shape, torch.Generator(device=device).manual_seed(seed), device)
    progress = comfy.utils.ProgressBar(len(sigmas) - 1)
    options = {"renoise_seed": renoise_seed, "coefficients": sigmas, "recipe": recipe}
    sampler = comfy.samplers.KSAMPLER(sample_vimoo_distilled, extra_options=options)
    samples = comfy.sample.sample_custom(
        result,
        noise,
        cfg,
        sampler,
        sigmas.float(),
        positive,
        [],
        latent_image=latent,
        seed=seed,
        callback=lambda i, *args: progress.update_absolute(i + 1, len(sigmas) - 1),
    )
    return ({**latent_image, "samples": samples},)


NODE_CLASS_MAPPINGS = {
    "VimooDistilledLoRALoader": VimooDistilledLoRALoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VimooDistilledLoRALoader": "Vimoo Distilled LoRA Loader",
}
