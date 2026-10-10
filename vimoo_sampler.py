"""Vimoo flow-matching UniPC, standard T2V sampling and recipe-bound distilled schedules."""

from __future__ import annotations

import copy
import math
import numbers
from typing import NamedTuple

import numpy as np
import torch
import comfy.patcher_extension
import comfy.sample
import comfy.samplers
import comfy.utils
import node_helpers

SOLVER_ORDER = 2


class DistilledSchedule(NamedTuple):
    task: str
    flow_shift: float
    timesteps: tuple[float, ...]
    dtype: torch.dtype


# Fixed post-shift grids of the supplied distilled adapters, keyed by adapter profile.
DISTILLED_SCHEDULES = {
    "t2v_distilled": DistilledSchedule("t2v", 12.0, (999, 988, 974, 951, 923, 880, 808, 622), torch.float64),
    "i2v_distilled": DistilledSchedule(
        "i2v",
        6.0,
        (
            1000.0,
            988.2352905273438,
            975.0,
            960.0,
            911.2781982421875,
            837.73583984375,
            713.924072265625,
            461.5384521484375,
        ),
        torch.float64,
    ),
    "lite_ti2v_distilled": DistilledSchedule(
        "ti2v",
        12.0,
        (1000, 998.3, 996.5, 994.7, 991.0, 983.9, 973.0, 952.4, 923.1, 878.0, 800.0, 631.6),
        torch.float32,
    ),
}


def _alpha_sigma(sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``_sigma_to_alpha_sigma_t`` for ``use_flow_sigmas=True``."""
    return 1.0 - sigma, sigma


def _lambda(alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return torch.log(alpha) - torch.log(sigma)


def _uni_p_bh(x, order, step_index, sig, model_outputs):
    """Predictor: ``multistep_uni_p_bh_update`` (predict_x0 branch)."""
    m0 = model_outputs[-1]
    sigma_t, sigma_s0 = sig[step_index + 1], sig[step_index]
    alpha_t, sigma_t = _alpha_sigma(sigma_t)
    alpha_s0, sigma_s0 = _alpha_sigma(sigma_s0)

    h = _lambda(alpha_t, sigma_t) - _lambda(alpha_s0, sigma_s0)

    h_phi_1 = B_h = torch.expm1(-h)
    D1s = None
    if order == 2:
        alpha_si, sigma_si = _alpha_sigma(sig[step_index - 1])
        rk = (_lambda(alpha_si, sigma_si) - _lambda(alpha_s0, sigma_s0)) / h
        D1s = torch.stack([(model_outputs[-2] - m0) / rk], dim=1)
        rhos_p = torch.ones(1, dtype=x.dtype, device=x.device) * 0.5

    x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
    if D1s is not None:
        pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
    else:
        pred_res = 0
    x_t = x_t_ - alpha_t * B_h * pred_res
    return x_t.to(x.dtype)


def _uni_c_bh(model_t, last_sample, x, order, step_index, sig, model_outputs):
    """Corrector: ``multistep_uni_c_bh_update`` (predict_x0 branch)."""
    m0 = model_outputs[-1]
    sigma_t, sigma_s0 = sig[step_index], sig[step_index - 1]
    alpha_t, sigma_t = _alpha_sigma(sigma_t)
    alpha_s0, sigma_s0 = _alpha_sigma(sigma_s0)

    h = _lambda(alpha_t, sigma_t) - _lambda(alpha_s0, sigma_s0)

    rks, D1s = [], []
    for i in range(1, order):
        si = step_index - (i + 1)
        mi = model_outputs[-(i + 1)]
        alpha_si, sigma_si = _alpha_sigma(sig[si])
        rk = (_lambda(alpha_si, sigma_si) - _lambda(alpha_s0, sigma_s0)) / h
        rks.append(rk)
        D1s.append((mi - m0) / rk)
    rks.append(torch.ones((), dtype=sig.dtype, device=sig.device))
    rks = torch.stack(rks).to(x.device)

    h_phi_1 = B_h = torch.expm1(-h)
    h_phi_k = h_phi_1 / (-h) - 1

    R, b = [], []
    factorial_i = 1
    for i in range(1, order + 1):
        R.append(torch.pow(rks, i - 1))
        b.append(h_phi_k * factorial_i / B_h)
        factorial_i *= i + 1
        h_phi_k = h_phi_k / (-h) - 1 / factorial_i
    R = torch.stack(R)
    b = torch.stack(b).to(x.device)

    if len(D1s) > 0:
        D1s = torch.stack(D1s, dim=1)
    else:
        D1s = None

    if order == 1:
        rhos_c = torch.ones(1, dtype=x.dtype, device=x.device) * 0.5
    else:
        rhos_c = torch.linalg.solve(R, b).to(x.device).to(x.dtype)

    x_t_ = sigma_t / sigma_s0 * last_sample - alpha_t * h_phi_1 * m0
    if D1s is not None:
        corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
    else:
        corr_res = 0
    D1_t = model_t - m0
    x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
    return x_t.to(x.dtype)


def sample_vimoo_uni_pc(model, noise, sigmas, extra_args=None, callback=None, disable=False, **kw):
    extra_args = {} if extra_args is None else extra_args
    total = len(sigmas) - 1
    if total == 0:
        return noise

    # Retain CPU FP32 scalar coefficients and BF16 solver state.
    sig = sigmas.detach().float().cpu()

    x = noise.to(torch.bfloat16)
    model_outputs = [None] * SOLVER_ORDER
    last_sample = None
    lower_order_nums = 0
    this_order = 1

    for i in range(total):
        # batch-shaped sigma like the stock k-diffusion samplers (0-dim breaks
        # calc_cond_batch's `torch.cat([timestep] * batch_chunks)`)
        # Comfy accumulates conditional x0 predictions in the input dtype.
        # Keep that API boundary FP32; only the solver state is BF16.
        prediction = model(x.float(), sigmas[i : i + 1].expand(x.shape[0]), **extra_args)
        x0_pred = x - sig[i] * prediction.bfloat16()

        # Corrector uses the order chosen at the previous step and the
        # pre-shift history (m0 = previous x0 prediction).
        if i > 0 and last_sample is not None:
            x = _uni_c_bh(x0_pred, last_sample, x, this_order, i, sig, model_outputs)

        model_outputs = [model_outputs[-1], x0_pred]

        this_order = min(min(SOLVER_ORDER, total - i), lower_order_nums + 1)
        last_sample = x
        x = _uni_p_bh(x, this_order, i, sig, model_outputs)

        if lower_order_nums < SOLVER_ORDER:
            lower_order_nums += 1

        if callback is not None:
            callback(
                {"x": x, "i": i, "denoised": model_outputs[-1], "sigma": sigmas[i], "sigma_hat": sigmas[i]}
            )

    return x


def recipe_sigmas(recipe):
    """Resolve the adapter's explicit post-shift sampling grid."""
    steps = recipe.get("num_inference_steps")
    grid = recipe.get("custom_timesteps")
    shift = recipe.get("flow_shift")
    if type(steps) is not int or steps < 1 or not isinstance(grid, list) or len(grid) != steps:
        raise ValueError("LoRA recipe requires a matching timestep count")
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1000 for value in grid
    ):
        raise ValueError("LoRA recipe timesteps must be finite numbers in (0, 1000]")
    if type(shift) not in (int, float) or not math.isfinite(shift) or shift <= 0:
        raise ValueError("LoRA recipe flow_shift must be positive and finite")
    if recipe.get("timestep_space") != "post_shift":
        raise ValueError("LoRA recipe requires timestep_space=post_shift")
    if any(left <= right for left, right in zip(grid, grid[1:])):
        raise ValueError("LoRA recipe timesteps must descend strictly")
    if recipe.get("task") == "i2v" and type(recipe.get("i2v_first_frame_replace")) is not bool:
        raise ValueError("I2V LoRA recipe requires i2v_first_frame_replace")
    values = [value / 1000.0 for value in grid]
    return torch.tensor([*values, 0.0], dtype=torch.float64)


def validate_distilled_sigmas(sigmas, recipe):
    if sigmas.dtype != torch.float64:
        raise ValueError("Distilled sigmas require FP64 precision; use the matching Vimoo Scheduler preset")
    if not isinstance(recipe, dict):
        raise ValueError("Distilled sampling requires the matching LoRA sidecar recipe")
    expected = recipe_sigmas(recipe).to(sigmas.device)
    if sigmas.shape != expected.shape or not torch.allclose(sigmas, expected, rtol=0.0, atol=1e-12):
        raise ValueError("Connected sigmas must match the LoRA sidecar sampling recipe")


def video_noise(shape, generator, device):
    b, c, f, h, w = shape
    return (
        torch.randn((b, f, c, h, w), generator=generator, device=device, dtype=torch.bfloat16)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )


def sample_vimoo_distilled(
    model,
    noise,
    sigmas,
    extra_args=None,
    callback=None,
    disable=False,
    renoise_seed=1001237,
    coefficients=None,
    recipe=None,
    **kw,
):
    coefficients = sigmas if coefficients is None else coefficients
    validate_distilled_sigmas(coefficients, recipe)
    if not torch.equal(sigmas, coefficients.to(sigmas)):
        raise ValueError("Model sigmas must match the connected scheduler coefficients")
    # Read coefficients from the connected scheduler, retaining its
    # FP64 scalars instead of reconstructing a separate schedule in the sampler.
    coefficients = coefficients.tolist()
    x = noise.to(torch.bfloat16)
    if renoise_seed is None or renoise_seed == -1:
        renoise_seed = (extra_args or {}).get("seed", recipe.get("seed"))
    if type(renoise_seed) is not int or not 0 <= renoise_seed <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("Stochastic sampling requires a valid seed or explicit ReNoise seed")
    generator = torch.Generator(device=x.device).manual_seed(renoise_seed)
    for i in range(len(coefficients) - 1):
        denoised = model(x, sigmas[i : i + 1].expand(x.shape[0]), **(extra_args or {})).to(torch.bfloat16)
        following = coefficients[i + 1]
        x = (
            denoised
            if following == 0
            else (1.0 - following) * denoised + following * video_noise(x.shape, generator, x.device)
        )
        if callback:
            callback({"x": x, "i": i, "denoised": denoised, "sigma": sigmas[i], "sigma_hat": sigmas[i]})
    return x


def vimoo_unipc_sigmas(
    flow_shift: float, num_inference_steps: int, num_train_timesteps: int = 1000
) -> torch.Tensor:
    """UniPC initialization shift followed by the inference-grid shift."""
    if not math.isfinite(flow_shift) or flow_shift <= 0:
        raise ValueError("flow_shift must be finite and greater than zero.")
    if num_inference_steps < 1:
        raise ValueError("steps must be at least 1.")
    training = np.linspace(1.0, 1.0 / num_train_timesteps, num_train_timesteps)[::-1].copy()
    training_sigmas = torch.from_numpy(1.0 - training).float()
    training_sigmas = flow_shift * training_sigmas / (1.0 + (flow_shift - 1.0) * training_sigmas)
    sigmas = np.linspace(training_sigmas[0].item(), training_sigmas[-1].item(), num_inference_steps + 1)[:-1]
    sigmas = flow_shift * sigmas / (1.0 + (flow_shift - 1.0) * sigmas)
    return torch.from_numpy(np.concatenate([sigmas, [0.0]]).astype(np.float32))


def _combine_vimoo_flow(cond, uncond, scale, guidance_rescale, direct_rescale=False):
    # Diffusers disables CFG (including rescale) at scale 1.
    if uncond is None or math.isclose(scale, 1.0):
        return cond
    guided = uncond + scale * (cond - uncond)
    if guidance_rescale > 0.0:
        dims = tuple(range(1, cond.ndim))
        std_cond = cond.std(dim=dims, keepdim=True)
        std_cfg = guided.std(dim=dims, keepdim=True)
        if direct_rescale:
            rescaled = guided * (std_cond / std_cfg)
            return guidance_rescale * rescaled + (1.0 - guidance_rescale) * guided
        valid = torch.isfinite(std_cond) & torch.isfinite(std_cfg) & (std_cfg != 0)
        denominator = torch.where(valid, std_cfg, torch.ones_like(std_cfg))
        ratio = std_cond / denominator
        valid = valid & torch.isfinite(ratio)
        ratio = torch.where(valid, ratio, torch.ones_like(std_cfg))
        candidate = guided * ratio
        finite = torch.isfinite(candidate).all(dim=dims, keepdim=True)
        rescaled = torch.where(valid & finite, candidate, guided)
        guided = guidance_rescale * rescaled + (1.0 - guidance_rescale) * guided
    return guided


def _make_vimoo_cfg_function(guidance_rescale):
    def cfg_fn(args):
        x = args["input"]
        return _combine_vimoo_flow(
            x - args["cond_denoised"],
            None if args["uncond_denoised"] is None else x - args["uncond_denoised"],
            args["cond_scale"],
            guidance_rescale,
        )

    return cfg_fn


def _make_vimoo_rescale_function(guidance_rescale):
    def post_cfg(args):
        # An existing CFG owns guidance. Rescale its result without discarding
        # it; downstream post-CFG callbacks still see normal x0 predictions.
        if math.isclose(args["cond_scale"], 1.0):
            return args["denoised"]
        x = args["input"]
        guided = x - args["denoised"]
        cond = x - args["cond_denoised"]
        dims = tuple(range(1, guided.ndim))
        std = guided.std(dim=dims, keepdim=True)
        ratio = cond.std(dim=dims, keepdim=True) / std
        ratio = torch.where(torch.isfinite(ratio), ratio, torch.ones_like(ratio))
        rescaled = guidance_rescale * guided * ratio + (1.0 - guidance_rescale) * guided
        return x - rescaled

    return post_cfg


def _find_ref_latent(positive):
    for entry in positive:
        if len(entry) > 1 and entry[1].get("vimoo_ref_latent") is not None:
            return entry[1]["vimoo_ref_latent"]
    return None


def _prepare_vimoo_noise(latent, seed, batch_index=None, source_t2v=False):
    """Direct CPU BF16 BCFHW for standard T2V; retain distilled's BFCHW stream."""

    def draw(batch_size):
        shape = (batch_size, *latent.shape[1:])
        if source_t2v:
            return torch.randn(shape, generator=generator, device="cpu", dtype=torch.bfloat16)
        return video_noise(shape, generator, "cpu")

    generator = torch.Generator("cpu").manual_seed(seed)
    if batch_index is None:
        return draw(latent.shape[0]).float()
    if len(batch_index) != latent.shape[0] or any(
        not isinstance(index, numbers.Integral) or index < 0 for index in batch_index
    ):
        raise ValueError("batch_index must contain one non-negative integer per latent batch item.")
    unique, inverse = np.unique(batch_index, return_inverse=True)
    selected = set(unique)
    noises = []
    for index in range(int(unique[-1]) + 1):
        noise = draw(1)
        if index in selected:
            noises.append(noise)
    return torch.cat([noises[index] for index in inverse]).float()


def _make_autocast_wrapper():
    """Run the DiT under BF16 autocast.

    The fork's `VimooLoopDenoiser` wraps every transformer call in
    `torch.autocast("cuda", bf16)` for bf16 weights: fp32-policy ops (norms,
    softmax) then round differently than plain bf16 execution, which visibly
    moves the prediction. Wrap comfy's diffusion-model call the same way.
    """

    def wrapper(executor, *args, **kwargs):
        x = args[0] if args else kwargs.get("x")
        if torch.is_tensor(x) and x.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return executor(*args, **kwargs)
        return executor(*args, **kwargs)

    # Lets the modulation-schedule precompute run under the same policy.
    wrapper.vimoo_autocast_dtype = torch.bfloat16
    return wrapper


class VimooT2VFlowGuider(comfy.samplers.CFGGuider):
    """Preserve BF16 flow through CFG, without a lossy x0 round-trip."""

    def __init__(self, model, rescale):
        super().__init__(model)
        self.rescale = rescale

    def predict_noise(self, x, timestep, model_options=None, seed=None):
        negative = self.conds.get("negative") if self.cfg != 1.0 else None
        uncond, cond = comfy.samplers.calc_cond_batch(
            self.inner_model, [negative, self.conds.get("positive")], x, timestep, model_options or {}
        )
        return _combine_vimoo_flow(
            cond.bfloat16(),
            None if negative is None else uncond.bfloat16(),
            self.cfg,
            self.rescale,
            direct_rescale=True,
        ).float()


def sample_standard(
    model, positive, negative, latent_image, sigmas, *, sampler_name, cfg, guidance_rescale, seed, denoise
):
    """Run standard T2V flow UniPC or a built-in ComfyUI sampler."""
    latent = latent_image["samples"]
    sigmas = sigmas.detach().to(device="cpu", dtype=torch.float32)
    if (
        sigmas.ndim != 1
        or len(sigmas) < 2
        or not torch.isfinite(sigmas).all()
        or sigmas[-1] != 0
        or not (sigmas[:-1] > sigmas[1:]).all()
    ):
        raise ValueError("sigmas must be a finite, decreasing sequence ending in zero")
    steps = len(sigmas) - 1
    if denoise < 1.0:
        skip = int(round(steps * (1.0 - denoise)))
        if skip >= steps:
            return (latent_image.copy(),)
        sigmas = sigmas[skip:]
    if sampler_name == "vimoo_uni_pc" and sigmas[0] >= 1:
        raise ValueError(
            "vimoo_uni_pc requires nonterminal sigmas strictly between 0 and 1; "
            "use Vimoo Scheduler. Connected sigmas are not adjusted."
        )

    standard_t2v = sampler_name == "vimoo_uni_pc"
    noise = _prepare_vimoo_noise(latent, seed, latent_image.get("batch_index"), source_t2v=standard_t2v)

    device = model.load_device
    ref_latent = _find_ref_latent(positive)
    if ref_latent is not None:
        raise ValueError(
            "Vimoo-1.0 I2V first-frame conditioning requires vimoo_i2v_euler, not the T2V sampler."
        )
    if standard_t2v and latent_image.get("noise_mask") is not None:
        raise ValueError("Standard T2V flow UniPC does not support x0-space inpainting masks.")
    model = model.clone()
    if standard_t2v:
        if model.model.get_dtype() != torch.bfloat16:
            raise ValueError("Standard T2V sampling requires BF16 compute.")
        if any(
            model.model_options.get(name)
            for name in ("sampler_cfg_function", "sampler_pre_cfg_function", "sampler_post_cfg_function")
        ):
            raise ValueError("Standard T2V flow sampling cannot compose x0-space CFG patches.")
        sampling = copy.copy(model.model.model_sampling)
        sampling.calculate_denoised = lambda sigma, model_output, model_input: model_output.float()
        model.add_object_patch("model_sampling", sampling)
        model.model_options.setdefault("transformer_options", {})["vimoo_standard_t2v"] = True
        model.add_object_patch("diffusion_model.precise_math", True)
        model.add_object_patch("diffusion_model.precise_inputs", True)
        positive = node_helpers.conditioning_set_values(positive, {})
        negative = node_helpers.conditioning_set_values(negative, {}) if negative is not None else None
        for conditioning in (positive, negative):
            if conditioning is not None:
                for _, values in conditioning:
                    glyph = values.get("conditioning_byt5small")
                    if glyph is not None:
                        values["conditioning_byt5small"] = glyph.float()
    elif "sampler_cfg_function" in model.model_options:
        if guidance_rescale > 0.0:
            # Install before inherited post-CFG transforms; preserve their
            # order and their final say over the denoised prediction.
            model.model_options["sampler_post_cfg_function"] = [
                _make_vimoo_rescale_function(guidance_rescale),
                *model.model_options.get("sampler_post_cfg_function", []),
            ]
    elif guidance_rescale > 0.0:
        model.set_model_sampler_cfg_function(_make_vimoo_cfg_function(guidance_rescale))
    if device.type == "cuda" and model.model.get_dtype() == torch.bfloat16:
        comfy.patcher_extension.add_wrapper(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            _make_autocast_wrapper(),
            model.model_options,
            is_model_options=True,
        )

    # Full denoising starts from unscaled noise. Partial denoising must
    # retain ComfyUI's sigma-scaled noise to preserve the input latent.
    noise_in = noise / sigmas[0] if denoise == 1.0 else noise

    progress = comfy.utils.ProgressBar(len(sigmas) - 1)

    def callback(i, denoised, x, total_steps):
        progress.update_absolute(i + 1, total_steps)

    if standard_t2v:

        def solver(guider, initialized, active_sigmas, **kwargs):
            initial = (
                noise.to(device=initialized.device, dtype=torch.bfloat16) if denoise == 1.0 else initialized
            )
            return sample_vimoo_uni_pc(guider, initial, active_sigmas, **kwargs)

        sampler = comfy.samplers.KSAMPLER(solver)
    elif sampler_name in comfy.samplers.SAMPLER_NAMES:
        sampler = comfy.samplers.sampler_object(sampler_name)
    else:
        raise ValueError(f"Unknown Vimoo sampler: {sampler_name!r}")
    if standard_t2v:
        guider = VimooT2VFlowGuider(model, guidance_rescale)
        guider.set_conds(positive, negative if negative is not None else [])
        guider.set_cfg(cfg)
        samples = guider.sample(noise_in, latent, sampler, sigmas, seed=seed, callback=callback)
    else:
        samples = comfy.sample.sample_custom(
            model,
            noise_in,
            cfg,
            sampler,
            sigmas,
            positive,
            negative if negative is not None else [],
            latent_image=latent,
            noise_mask=latent_image.get("noise_mask"),
            seed=seed,
            callback=callback,
        )
    if latent_image.get("noise_mask") is not None:
        # The solver rounds x0 to BF16. Restore protected raw
        # latents exactly instead of letting that rounding alter them.
        mask = comfy.utils.reshape_mask(latent_image["noise_mask"], samples.shape).to(samples.device)
        samples = torch.where(mask == 0, latent.to(samples), samples)
    output = latent_image.copy()
    output["samples"] = samples
    return (output,)
