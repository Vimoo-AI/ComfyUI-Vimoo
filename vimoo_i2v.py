"""Full-size pure I2V Euler, using native Comfy execution."""

from __future__ import annotations

import copy
import math
import torch
import comfy.patcher_extension
import comfy.model_management
import comfy.samplers
import comfy.utils
import node_helpers
from comfy_api.torch_helpers import set_torch_compile_wrapper

from .vimoo_sampler import _find_ref_latent, _make_autocast_wrapper
from .vimoo_vae import VimooFullI2VLatentFormat


def _configure_math_compilation(model):
    options = model.model_options["transformer_options"]
    options["vimoo_i2v_fa2"] = model.model_options.get("vimoo_attention_backend") == "fa2"
    if model.model_options.get("vimoo_compile_scope") != "math_helpers":
        return
    dit = model.get_model_object("diffusion_model")
    keys = [
        f"diffusion_model.{name}_math" for name in ("qk_norm", "modulate", "rope_pair", "gate", "fa2_pack")
    ]
    for index in range(len(dit.double_blocks)):
        keys.extend(
            f"diffusion_model.double_blocks.{index}.{stream}_mod.lin.math" for stream in ("img", "txt")
        )
    set_torch_compile_wrapper(model=model, keys=keys, **model.model_options["torch_compile_kwargs"])
    # The compile wrapper is entered at APPLY_MODEL, after sampler precompute.
    # Compile only expansion math; native weight casting and DynamicVRAM faults
    # stay outside the full graph and retain their asynchronous buffer lifetime.
    options["vimoo_i2v_compiled_modulation"] = True


def i2v_sigmas(steps, flow_shift):
    if type(steps) is not int or steps < 1 or not math.isfinite(flow_shift) or flow_shift <= 0:
        raise ValueError("steps must be positive and flow_shift positive and finite")
    grid = torch.linspace(1, 0, steps + 1, dtype=torch.float32)
    return flow_shift * grid / (1 + (flow_shift - 1) * grid)


def i2v_guidance(cond, uncond, cfg, rescale):
    # Reference CFG and unbiased per-sample std operate in transformer dtype.
    if uncond is None or cfg <= 1:
        return cond
    guided = uncond + cfg * (cond - uncond)
    if rescale > 0:
        dims = tuple(range(1, guided.ndim))
        scaled = guided * (cond.std(dim=dims, keepdim=True) / guided.std(dim=dims, keepdim=True))
        guided = rescale * scaled + (1 - rescale) * guided
    return guided


def i2v_initial_state(shape, seed, device):
    generator = torch.Generator("cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.bfloat16).to(device)


def sample_i2v_euler(
    model,
    initial,
    sigmas,
    extra_args=None,
    callback=None,
    disable=False,
    *,
    reference=None,
    noise=None,
    manual_timesteps=None,
    **kwargs,
):
    """The supplied model returns flow, not Comfy's usual x0 prediction."""
    x = initial
    extra_args = extra_args or {}
    # Keep scheduler scalar coefficients in CPU FP32.
    coefficients = sigmas.detach().cpu().float()
    for i in range(len(sigmas) - 1):
        flow = model(x.float(), sigmas[i : i + 1].expand(x.shape[0]), **extra_args)
        if manual_timesteps is not None:
            following = manual_timesteps[i + 1] if i + 1 < len(manual_timesteps) else 0.0
            delta = (float(following) - float(manual_timesteps[i])) / 1000.0
            x = (x.float() + delta * flow.float()).bfloat16()
        else:
            x = x.float() + flow.float() * (coefficients[i + 1] - coefficients[i])
        if reference is not None:
            if manual_timesteps is not None:
                sigma_next = float(following) / 1000.0
                pinned = (
                    (1.0 - sigma_next) * reference.detach().float()
                    + sigma_next * noise[:, :, :1].detach().float()
                ).bfloat16()
                x = torch.cat((pinned, x[:, :, 1:]), dim=2)
            else:
                sigma_next = coefficients[i + 1].to(x)
                x[:, :, :1] = (1 - sigma_next) * reference.to(x) + sigma_next * noise[:, :, :1].to(x)
        if callback is not None:
            callback({"x": x, "i": i, "denoised": x, "sigma": sigmas[i], "sigma_hat": sigmas[i]})
    return x


class I2VFlowGuider(comfy.samplers.CFGGuider):
    """Retain the native lifecycle without a lossy flow -> x0 -> flow round-trip."""

    def __init__(self, model, rescale):
        super().__init__(model)
        self.rescale = rescale

    def predict_noise(self, x, timestep, model_options=None, seed=None):
        negative = self.conds.get("negative") if self.cfg > 1 else None
        cond, uncond = comfy.samplers.calc_cond_batch(
            self.inner_model, [self.conds.get("positive"), negative], x, timestep, model_options or {}
        )
        return i2v_guidance(
            cond.bfloat16(), None if negative is None else uncond.bfloat16(), self.cfg, self.rescale
        ).float()


def sample_full_i2v(
    model,
    positive,
    negative,
    latent_image,
    sigmas,
    cfg=3.0,
    guidance_rescale=0.5,
    seed=42,
    fps=24.0,
    add_fps=True,
    recipe=None,
):
    """Run full I2V through the native lifecycle with an explicit SIGMAS schedule."""
    distilled = recipe is not None and recipe.get("profile") == "i2v_distilled"
    if (model.model_options.get("vimoo_distilled") and not distilled) or getattr(
        model.model.diffusion_model, "hms", False
    ):
        raise ValueError("vimoo_i2v_euler requires Vimoo-1.0 I2V, not a Lite or Distilled model")
    if model.model.get_dtype() != torch.bfloat16:
        raise ValueError("Vimoo-1.0 I2V reference sampling requires BF16 compute (BF16 or INT8 weights)")
    if (
        not math.isfinite(cfg)
        or not 1 <= cfg <= 30
        or not math.isfinite(guidance_rescale)
        or not 0 <= guidance_rescale <= 1
    ):
        raise ValueError("cfg must be in [1,30] and guidance_rescale in [0,1]")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    if any(
        model.model_options.get(name)
        for name in ("sampler_cfg_function", "sampler_pre_cfg_function", "sampler_post_cfg_function")
    ):
        raise ValueError("Vimoo-1.0 I2V flow sampling cannot compose x0-space CFG patches")
    manual_timesteps = None
    if distilled:
        from .vimoo_sampler import validate_distilled_sigmas

        validate_distilled_sigmas(sigmas, recipe)
        if cfg != 1.0 or guidance_rescale != 0.0 or not add_fps:
            raise ValueError("I2V Distilled requires CFG1/rescale0 and temporal RoPE interpolation")
        manual_timesteps = recipe["custom_timesteps"]
    sigmas = sigmas.detach().to(device="cpu", dtype=torch.float32)
    if (
        sigmas.ndim != 1
        or len(sigmas) < 2
        or not torch.isfinite(sigmas).all()
        or sigmas[0] != 1
        or sigmas[-1] != 0
        or not (sigmas[:-1] > sigmas[1:]).all()
    ):
        raise ValueError("Vimoo-1.0 I2V sigmas must decrease from 1 to 0; use vimoo_i2v_euler scheduler")
    latent = latent_image["samples"]
    if latent_image.get("noise_mask") is not None or latent_image.get("batch_index") is not None:
        raise ValueError(
            "Vimoo-1.0 I2V reference sampling does not support masks or batch-index noise remapping"
        )
    model = model.clone()
    sampling = copy.copy(model.model.model_sampling)
    sampling.calculate_denoised = lambda sigma, model_output, model_input: model_output.float()
    if distilled:
        grid = sigmas.clone()
        timesteps = torch.tensor([*manual_timesteps, 0.0], dtype=torch.float32)

        def model_timestep(sigma):
            index = (sigma[..., None] - grid.to(sigma)).abs().argmin(-1)
            return timesteps.to(sigma.device)[index]

        sampling.timestep = model_timestep
    model.add_object_patch("model_sampling", sampling)
    if not distilled:
        model.add_object_patch("latent_format", VimooFullI2VLatentFormat())
    model.add_object_patch("process_timestep", lambda timestep, **kwargs: timestep)
    model.add_object_patch("diffusion_model.pe_embedder.ref_fps", 16.0 if add_fps else None)
    model.add_object_patch("diffusion_model.pe_embedder.use_reciprocal_temporal_scale", True)
    model.add_object_patch("diffusion_model._math_inputs_cache_key", None)
    model.model_options.setdefault("transformer_options", {})["vimoo_full_i2v"] = True
    _configure_math_compilation(model)

    if not distilled:
        comfy.patcher_extension.add_wrapper(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            _make_autocast_wrapper(),
            model.model_options,
            is_model_options=True,
        )
    positive = node_helpers.conditioning_set_values(positive, {"fps": fps})
    negative = node_helpers.conditioning_set_values(negative if negative is not None else [], {"fps": fps})
    # Full-size I2V feeds its ByT5 mapper FP32 even when no quoted
    # glyphs are present. The shared encoder's empty placeholder is BF16;
    # leaving it that way changes the mapper's LayerNorm/autocast boundary.
    for conditioning in (positive, negative):
        for _, values in conditioning:
            glyph = values.get("conditioning_byt5small")
            if glyph is not None and not distilled:
                values["conditioning_byt5small"] = glyph.float()
    reference = _find_ref_latent(positive)
    if distilled and reference is None:
        raise ValueError("I2V Distilled requires first-frame replacement conditioning")
    progress = comfy.utils.ProgressBar(len(sigmas) - 1)

    def solver(guider, ignored_initial, active_sigmas, **kwargs):
        device = ignored_initial.device
        initial = i2v_initial_state(tuple(latent.shape), seed, device)
        ref = (
            None if reference is None else model.model.latent_format.process_in(reference.to(device).float())
        )
        if distilled:
            # Normalize Wan latents in BF16, then return an FP32 reference frame.
            # First-frame replacement consumes that FP32 tensor before the
            # single BF16 rounding after each Euler update.
            ref = ref.float()
        return sample_i2v_euler(
            guider,
            initial,
            active_sigmas,
            reference=ref,
            noise=initial,
            manual_timesteps=manual_timesteps,
            **kwargs,
        )

    guider = I2VFlowGuider(model, guidance_rescale)
    guider.set_conds(positive, negative if negative is not None else [])
    guider.set_cfg(cfg)
    samples = guider.sample(
        torch.zeros_like(latent),
        torch.zeros_like(latent),
        comfy.samplers.KSAMPLER(solver),
        sigmas,
        seed=seed,
        callback=lambda i, denoised, x, total: progress.update_absolute(i + 1, total),
    )
    return (
        {
            "samples": samples.to(
                device=comfy.model_management.intermediate_device(),
                dtype=comfy.model_management.intermediate_dtype(),
            )
        },
    )
