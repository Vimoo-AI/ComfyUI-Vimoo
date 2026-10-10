# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Vimoo-1.0-Lite TI2V Euler and observed-frame guidance, using native Comfy execution."""

import copy
import math
from types import SimpleNamespace

import torch
import comfy.model_management
import comfy.patcher_extension
import comfy.samplers
import comfy.utils
import node_helpers

from .vimoo_i2v import i2v_initial_state
from .vimoo_sampler import _make_autocast_wrapper

LITE_TI2V_DISTILLED_NOISE_SEED = 1234

LITE_TI2V_DEFAULTS = {
    "format": "vimoo-lite-ti2v-defaults-v1",
    "profile": "lite_ti2v_anchor",
    "positive_prompt": "A pig is reading a book",
    "negative_prompt": (
        "oversaturated colors, overexposed, static, blurry details, subtitles, "
        "style, artwork, painting, still image, grayish overall tone, worst quality, "
        "low quality, JPEG artifacts, ugly, mutilated, extra fingers, poorly drawn "
        "hands, poorly drawn face, deformed, disfigured, malformed limbs, fused "
        "fingers, motionless scene, cluttered background, three legs, many people "
        "in the background, walking backwards"
    ),
    "attention_backend": {"diffusers": "native", "comfyui": "pytorch"},
    "common": {
        "num_inference_steps": 50,
        "flow_shift": 8.0,
        "schedule_std": 2.5,
        "guidance_rescale": 0.85,
        "max_sequence_length": 1400,
        "height": 720,
        "width": 1280,
        "num_frames": 121,
        "fps": 24,
        "seed": 1234,
    },
    "t2v": {"guidance_scale": 4.0, "reference_guidance_scale": 1.0},
    "i2v": {
        "guidance_scale": 5.0,
        "reference_guidance_scale": 3.0,
        "i2v_anchor_tau": 0.001,
    },
}


def validate_guidance_values(
    *,
    guidance_scale=None,
    guidance_rescale=None,
    reference_guidance_scale=None,
    i2v_anchor_tau=None,
):
    """Validate the shared control domain; omitted values are left to defaults."""
    values = {
        "guidance_scale": guidance_scale,
        "guidance_rescale": guidance_rescale,
        "reference_guidance_scale": reference_guidance_scale,
        "i2v_anchor_tau": i2v_anchor_tau,
    }
    for name, value in values.items():
        if value is None:
            continue
        unit_interval = name == "i2v_anchor_tau"
        try:
            valid = math.isfinite(value) and value >= 0
            if unit_interval:
                valid = valid and value <= 1
        except (TypeError, ValueError):
            valid = False
        if not valid:
            domain = "in [0, 1]" if unit_interval else "non-negative"
            raise ValueError(f"{name} must be finite and {domain}")


def guidance_branches(text_scale, reference_scale, observed):
    """Ordered image-conditioning branches consumed by text encoding and flow prediction."""
    if observed and reference_scale != 1:
        if reference_scale == max(text_scale, 1):
            return ("unreferenced", "positive")
        return ("unreferenced", "negative", "positive")
    return ("negative", "positive") if text_scale > 1 else ("positive",)


PROFILE = LITE_TI2V_DEFAULTS["profile"]
IMAGE_KEYS = {"concat_latent_image", "concat_mask", "clip_vision_output", "vimoo_ref_latent"}


def ti2v_guidance_branches(cfg, reference_cfg, observed):
    """Ordered branches shared by paired text encoding and flow prediction."""
    return guidance_branches(cfg, reference_cfg, observed)


def ti2v_sigmas(
    steps,
    flow_shift=LITE_TI2V_DEFAULTS["common"]["flow_shift"],
    schedule_std=LITE_TI2V_DEFAULTS["common"]["schedule_std"],
):
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be positive")
    if any(not math.isfinite(v) or v <= 0 for v in (flow_shift, schedule_std)):
        raise ValueError("flow_shift and schedule_std must be positive and finite")
    quantiles = torch.linspace(1, 0, steps + 1, dtype=torch.float64)
    sigmas = torch.sigmoid(math.log(flow_shift) + schedule_std * torch.special.ndtri(quantiles)).float()
    sigmas[0], sigmas[-1] = 1.0, 0.0
    return sigmas


def ti2v_guidance(predictions, cfg, reference_cfg, rescale, observed):
    # The mixed-precision head returns FP32 flow; CFG/rescale consume
    # it directly. Only the scheduler's updated state rounds back to BF16.
    predictions = [prediction.float() for prediction in predictions]
    positive = predictions[-1]
    if len(predictions) == 3:
        unconditional, reference, _ = predictions
        guided = (
            unconditional
            + reference_cfg * (reference - unconditional)
            + max(cfg, 1.0) * (positive - reference)
        )
    elif len(predictions) == 2:
        guided = predictions[0] + cfg * (positive - predictions[0])
    else:
        guided = positive
    if len(predictions) > 1 and rescale > 0:
        region = slice(1, None) if observed else slice(None)
        source, target = guided[:, :, region], positive[:, :, region]
        dims = tuple(range(1, source.ndim))
        scaled = source * (target.std(dim=dims, keepdim=True) / source.std(dim=dims, keepdim=True))
        guided = guided.clone()
        guided[:, :, region] = rescale * scaled + (1 - rescale) * source
    if observed:
        guided = guided.clone()
        guided[:, :, :1] = 0
    return guided


def anchor_frame(reference, noise, tau):
    # make_noisy_latents uses FP32 coefficients and casts once afterward.
    timestep = torch.full((reference.shape[0],), tau * 1000, dtype=torch.float32, device=reference.device)
    ratio = (timestep / 1000).view(-1, *([1] * (reference.ndim - 1)))
    return ((1 - ratio) * reference + ratio * noise).bfloat16()


def sample_ti2v_euler(
    model,
    initial,
    sigmas,
    extra_args=None,
    callback=None,
    disable=False,
    *,
    reference=None,
    tau=LITE_TI2V_DEFAULTS["i2v"]["i2v_anchor_tau"],
    **kwargs,
):
    x = initial.bfloat16().clone()
    anchor = None if reference is None else anchor_frame(reference, x[:, :, :1], tau)
    coefficients = sigmas.detach().cpu().float()
    if anchor is not None:
        x[:, :, :1] = anchor
    for index in range(len(sigmas) - 1):
        flow = model(x.float(), sigmas[index : index + 1].expand(x.shape[0]), **(extra_args or {}))
        x = (x.float() + flow.float() * (coefficients[index + 1] - coefficients[index])).bfloat16()
        if anchor is not None:
            x[:, :, :1] = anchor
        if callback is not None:
            callback({"x": x, "i": index, "denoised": x, "sigma": sigmas[index], "sigma_hat": sigmas[index]})
    if reference is not None:
        x[:, :, :1] = reference.to(x)
    return x


class TI2VFlowGuider(comfy.samplers.CFGGuider):
    def __init__(self, model, cfg, reference_cfg, rescale, observed):
        super().__init__(model)
        self.cfg, self.reference_cfg, self.rescale, self.observed = cfg, reference_cfg, rescale, observed

    def predict_noise(self, x, timestep, model_options=None, seed=None):
        names = ti2v_guidance_branches(self.cfg, self.reference_cfg, self.observed)
        predictions = comfy.samplers.calc_cond_batch(
            self.inner_model, [self.conds[name] for name in names], x, timestep, model_options or {}
        )
        return ti2v_guidance(predictions, self.cfg, self.reference_cfg, self.rescale, self.observed).float()


def sample_lite_ti2v(
    model,
    positive,
    negative,
    latent_image,
    sigmas,
    cfg,
    guidance_rescale,
    seed,
    fps=LITE_TI2V_DEFAULTS["common"]["fps"],
    reference_guidance_scale=None,
    i2v_anchor_tau=None,
):
    distilled = model.model_options.get("vimoo_distilled")
    if model.model_options.get("vimoo_sampling_profile") != PROFILE or (
        distilled and distilled.get("profile") != "lite_ti2v_distilled"
    ):
        raise ValueError(
            "vimoo_lite_ti2v_euler requires the Vimoo-1.0-Lite TI2V base without a Distilled adapter"
        )
    if model.model.get_dtype() != torch.bfloat16:
        raise ValueError("Lite TI2V reference sampling requires BF16 compute")
    if latent_image["samples"].shape[2] < 2:
        raise ValueError("Lite TI2V requires video length greater than one (4n+1 frames)")
    observed = any(values.get("concat_latent_image") is not None for _, values in positive)
    task_recipe = distilled["recipes"]["i2v" if observed else "t2v"] if distilled else {}
    if distilled and (
        task_recipe.get("initial_noise_mode") != "cpu_channel_first"
        or task_recipe.get("initial_noise_seed") != LITE_TI2V_DISTILLED_NOISE_SEED
    ):
        raise ValueError(
            "Vimoo-1.0-Lite TI2V Distilled requires fixed CPU BF16 BCFHW noise generated with seed 1234; update the adapter sidecar"
        )
    if reference_guidance_scale is None:
        reference_guidance_scale = task_recipe.get(
            "reference_guidance_scale",
            1.0 if distilled else LITE_TI2V_DEFAULTS["i2v"]["reference_guidance_scale"],
        )
    if i2v_anchor_tau is None:
        i2v_anchor_tau = task_recipe.get("i2v_anchor_tau", LITE_TI2V_DEFAULTS["i2v"]["i2v_anchor_tau"])
    validate_guidance_values(
        guidance_scale=cfg,
        guidance_rescale=guidance_rescale,
        reference_guidance_scale=reference_guidance_scale,
        i2v_anchor_tau=i2v_anchor_tau,
    )
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    if any(
        model.model_options.get(name)
        for name in ("sampler_cfg_function", "sampler_pre_cfg_function", "sampler_post_cfg_function")
    ):
        raise ValueError("Lite TI2V flow sampling cannot compose x0-space CFG patches")
    if latent_image.get("noise_mask") is not None or latent_image.get("batch_index") is not None:
        raise ValueError("Lite TI2V does not support masks or batch-index noise remapping")
    references = [values.get("concat_latent_image") for _, values in positive]
    observed = any(value is not None for value in references)
    if observed and not all(value is not None for value in references):
        raise ValueError("Lite TI2V conditioning must use one consistent task")
    if distilled:
        from .vimoo_sampler import recipe_sigmas

        task = "i2v" if observed else "t2v"
        expected = recipe_sigmas({**distilled["recipes"][task], "task": task}).float()
        if sigmas.shape != expected.shape or not torch.equal(sigmas.cpu(), expected):
            raise ValueError("Vimoo-1.0-Lite TI2V Distilled requires its full manual post-shift grid")
        if cfg != 1 or reference_guidance_scale != 1 or guidance_rescale != 0 or i2v_anchor_tau != 0.001:
            raise ValueError(
                "Vimoo-1.0-Lite TI2V Distilled requires text/image CFG1, rescale0 and anchor tau0.001"
            )
    sigmas = sigmas.detach().cpu().float()
    if (
        sigmas.ndim != 1
        or len(sigmas) < 2
        or not torch.isfinite(sigmas).all()
        or sigmas[0] != 1
        or sigmas[-1] != 0
        or not (sigmas[:-1] > sigmas[1:]).all()
    ):
        raise ValueError("Lite TI2V sigmas must decrease from 1 to 0; use its lognorm scheduler")
    model = model.clone()
    sampling = copy.copy(model.model.model_sampling)
    sampling.calculate_denoised = lambda sigma, output, inputs: output.float()
    model.add_object_patch("model_sampling", sampling)
    model.add_object_patch("process_timestep", lambda timestep, **kwargs: timestep)
    options = model.model_options.setdefault("transformer_options", {})
    options["vimoo_lite_ti2v_anchor"] = True
    comfy.patcher_extension.add_wrapper(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        _make_autocast_wrapper(),
        model.model_options,
        is_model_options=True,
    )
    latent = latent_image["samples"]
    noise_seed = LITE_TI2V_DISTILLED_NOISE_SEED if distilled else seed
    initial = i2v_initial_state(tuple(latent.shape), noise_seed, "cpu")
    reference = None if not observed else model.model.latent_format.process_in(references[0][:, :, :1].cpu())
    positive = node_helpers.conditioning_set_values(positive, {"fps": fps})
    negative = node_helpers.conditioning_set_values(negative or [], {"fps": fps})
    conditions = {"positive": positive, "negative": negative}
    if observed:
        unreferenced = []
        for tensor, values in negative:
            tensor, values = values.get("vimoo_unreferenced_conditioning", (tensor, values))
            unreferenced.append(
                [tensor, {key: value for key, value in values.items() if key not in IMAGE_KEYS}]
            )
        unreferenced = node_helpers.conditioning_set_values(unreferenced, {"fps": fps})
        vision = positive[0][1].get("clip_vision_output")
        if vision is not None:
            states = vision.last_hidden_state
            vision_mask_dtype = torch.bool if distilled else torch.int64
            # Supply the same key set for native CFG packing; the zero branch
            # has explicit masked vision tokens and zero concat conditioning.
            conditions = {
                name: node_helpers.conditioning_set_values(
                    values, {"vimoo_vision_mask": torch.ones(states.shape[:2], dtype=vision_mask_dtype)}
                )
                for name, values in conditions.items()
            }
            unreferenced = node_helpers.conditioning_set_values(
                unreferenced,
                {
                    "clip_vision_output": SimpleNamespace(last_hidden_state=torch.zeros_like(states)),
                    "vimoo_vision_mask": torch.zeros(states.shape[:2], dtype=vision_mask_dtype),
                },
            )
        frame = anchor_frame(reference, initial[:, :, :1], i2v_anchor_tau)
        shared = {
            "observed_first_frame_timesteps": torch.tensor([i2v_anchor_tau * 1000], dtype=torch.float32),
            "vimoo_anchor_first_frame": frame,
        }
        conditions = {
            name: node_helpers.conditioning_set_values(values, shared) for name, values in conditions.items()
        }
        conditions["unreferenced"] = node_helpers.conditioning_set_values(
            unreferenced,
            {
                "observed_first_frame_timesteps": torch.tensor([1000.0]),
                "vimoo_anchor_first_frame": initial[:, :, :1],
            },
        )
    guider = TI2VFlowGuider(
        model, cfg, reference_guidance_scale if observed else 1, guidance_rescale, observed
    )
    guider.inner_set_conds(conditions)
    progress = comfy.utils.ProgressBar(len(sigmas) - 1)

    def solver(active_guider, ignored_initial, active_sigmas, **kwargs):
        return sample_ti2v_euler(
            active_guider,
            initial.to(ignored_initial.device),
            active_sigmas,
            reference=None if reference is None else reference.to(ignored_initial.device),
            tau=i2v_anchor_tau,
            **kwargs,
        )

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
