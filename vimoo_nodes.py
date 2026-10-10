"""ComfyUI loaders, conditioning nodes and samplers for Vimoo V1."""

from __future__ import annotations

import contextlib
import math
import os
import json
import re
from pathlib import Path
import numpy as np
import torch
from PIL import Image

import folder_paths
import comfy.model_detection
import comfy.model_management
import comfy.model_patcher
import comfy.patcher_extension
import comfy.samplers
import comfy.sd
import comfy.supported_models_base
import comfy.utils
from comfy.cli_args import args
from comfy.text_encoders.hunyuan_video import llama_detect
import node_helpers

from .vimoo_encoder import VimooTokenizer, VimooTextEncoder
from .vimoo_support import Vimoo
from .vimoo_vae import VimooVAELoader, VimooReferenceVAELoader, validate_vae_numerical_profile
from .vimoo_text import patch_vimoo_text
from .vimoo_precision import configure_anchor_linears
from .vimoo_sampler import DISTILLED_SCHEDULES, sample_standard, vimoo_unipc_sigmas
from .vimoo_vision import encode_vimoo_vision
from .vimoo_distilled_nodes import sample_distilled
from .vimoo_checkpoint import checkpoint_provenance
from .vimoo_checkpoint import normalize_transformer_state, validate_modulation_state
from .vimoo_vision import validate_source_image_decode
from .vimoo_ti2v import LITE_TI2V_DEFAULTS, ti2v_guidance_branches, validate_guidance_values

# The built-in negative prompt.
VIMOO_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, overall gray, worst quality, low quality, JPEG compression "
    "residue, ugly, incomplete, extra fingers, extra arms, extra hands, extra limbs, "
    "missing limbs, fused fingers, fused limbs, poorly drawn hands, poorly drawn faces, "
    "deformed, disfigured, misshapen limbs, twisted body, distorted limbs, contorted "
    "pose, broken anatomy, blurry face, blurry body, motion blur, ghosting, afterimage, "
    "melting face, warped features, duplicated person, cloned character, repeated "
    "subject, two identical people, extra person, person appearing out of nowhere, "
    "identical faces on different people, same face repeated, morphing body, changing "
    "identity, shapeshifting subject, object transforming, inconsistent character, "
    "many people in the background, three legs, still picture, messy background, "
    "walking backwards, distorted, artifacts, oversmooth, painting, cartoon"
)


def _modulation_schedule_is_safe(model_patcher, diffusion_model, transformer_options):
    if not hasattr(diffusion_model, "precompute_modulation_schedule"):
        return False
    if transformer_options.get("patches") or transformer_options.get("patches_replace"):
        return False
    modulation_names = (".img_mod", ".txt_mod", ".adaLN_modulation")
    patch_names = tuple(model_patcher.patches) + tuple(model_patcher.object_patches)
    return not any(any(name in patch for name in modulation_names) for patch in patch_names)


def _precompute_autocast_dtype(transformer_options):
    """The autocast dtype of the DIFFUSION_MODEL wrapper the DiT will run under, if any."""
    for wrapper in comfy.patcher_extension.get_all_wrappers(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options
    ):
        dtype = getattr(wrapper, "vimoo_autocast_dtype", None)
        if dtype is not None:
            return dtype
    return None


def _precompute_modulation_sample_wrapper(
    executor, guider, sigmas, extra_args, callback, noise, *args, **kwargs
):
    """Build execution-scoped static tables after model load and release them after sampling.

    The modulation tables replace per-step ``time_in`` / ``*_mod`` / ``adaLN_modulation``
    evaluations by exact timestep, so they are computed under the same autocast
    policy as the DiT forward (the Vimoo Sampler's DIFFUSION_MODEL wrapper).
    """
    model_options = extra_args.get("model_options")
    if model_options is None:
        return executor(guider, sigmas, extra_args, callback, noise, *args, **kwargs)
    transformer_options = model_options.setdefault("transformer_options", {})
    diffusion_model = getattr(guider.inner_model, "diffusion_model", None)
    if diffusion_model is None:
        return executor(guider, sigmas, extra_args, callback, noise, *args, **kwargs)
    transformer_options["vimoo_static_conditioning_cache"] = {}
    transformer_options["vimoo_attention_mask_cache"] = {}
    transformer_options["vimoo_attention_metadata"] = {}
    profile = model_options.get("vimoo_distilled", {}).get("profile") or model_options.get(
        "vimoo_sampling_profile"
    )
    reference_baseline = bool(getattr(diffusion_model, "precise_math", False)) and profile in (
        "standard_t2v",
        "standard_i2v",
        "t2v_distilled",
        "i2v_distilled",
        "lite_ti2v_distilled",
    )
    if (
        not reference_baseline
        and not transformer_options.get("vimoo_lite_ti2v_anchor")
        and not transformer_options.get("vimoo_i2v_compiled_modulation")
        and not transformer_options.get("vimoo_standard_t2v")
        and _modulation_schedule_is_safe(guider.model_patcher, diffusion_model, transformer_options)
    ):
        timesteps = guider.inner_model.model_sampling.timestep(sigmas[:-1]).float()
        timesteps = guider.inner_model.process_timestep(timesteps, x=noise)
        precompute_options = transformer_options.copy()
        precompute_options["prefetch_dynamic_vbars"] = guider.model_patcher.is_dynamic()
        cfg1 = guider.cfg == 1.0 and not model_options.get("disable_cfg1_optimization", False)
        autocast_dtype = _precompute_autocast_dtype(transformer_options)
        autocast = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if autocast_dtype is not None and torch.is_tensor(noise) and noise.is_cuda
            else contextlib.nullcontext()
        )
        with autocast:
            transformer_options["vimoo_modulation_schedule"] = diffusion_model.precompute_modulation_schedule(
                timesteps, precompute_options, batch_sizes=(1,) if cfg1 else (1, 2)
            )
    try:
        return executor(guider, sigmas, extra_args, callback, noise, *args, **kwargs)
    finally:
        transformer_options.pop("vimoo_modulation_schedule", None)
        transformer_options.pop("vimoo_static_conditioning_cache", None)
        transformer_options.pop("vimoo_attention_mask_cache", None)
        transformer_options.pop("vimoo_attention_metadata", None)


def _split_double_block_qkv_state_dict(state_dict):
    """Expose released fused QKV tensors as three loadable parameter views.

    Model detection intentionally runs before this migration because ComfyUI's
    public checkpoint format remains unchanged.  ``chunk`` does not duplicate
    tensor storage, which keeps the block-offload loader within its RAM budget.
    """
    pattern = re.compile(
        r"^(double_blocks\.\d+\.(?:img|txt)_attn)\.qkv\.(weight|bias|weight_scale|comfy_quant)$"
    )
    replacements = {}
    fused_keys = []
    for key, value in state_dict.items():
        match = pattern.fullmatch(key)
        if match is None:
            continue
        prefix, suffix = match.groups()
        if suffix != "comfy_quant" and (value.ndim == 0 or value.shape[0] % 3):
            raise ValueError(f"Invalid fused Vimoo QKV tensor: {key}")
        if suffix == "comfy_quant":
            if json.loads(bytes(value.tolist())) != {"format": "int8_tensorwise"}:
                raise ValueError("Only per-channel INT8 packed QKV can load as split views")
            scale = state_dict.get(prefix + ".qkv.weight_scale")
            if scale is None or scale.shape != (state_dict[prefix + ".qkv.weight"].shape[0], 1):
                raise ValueError("Packed INT8 QKV requires per-output-channel scales")
        pieces = (value,) * 3 if suffix == "comfy_quant" else value.chunk(3, dim=0)
        for part, tensor in zip("qkv", pieces):
            split_key = f"{prefix}.to_{part}.{suffix}"
            if split_key in state_dict or split_key in replacements:
                raise ValueError(f"Conflicting Vimoo QKV tensor: {split_key}")
            replacements[split_key] = tensor
        fused_keys.append(key)
    for key in fused_keys:
        state_dict.pop(key)
    state_dict.update(replacements)
    return state_dict


def _prepare_qkv_layout(state, metadata, layout, depth):
    """Select immutable checkpoint storage; packing itself is an offline step."""
    if layout not in ("auto", "split", "packed"):
        raise ValueError(f"Unknown QKV layout: {layout}")
    stored = (metadata or {}).get("vimoo_qkv_layout", "split")
    packed = (stored if layout == "auto" else layout) == "packed"
    if not packed:
        _split_double_block_qkv_state_dict(state)
        return False
    for index in range(depth):
        for stream in ("img", "txt"):
            prefix = f"double_blocks.{index}.{stream}_attn."
            if prefix + "qkv.weight" not in state or any(key.startswith(prefix + "to_") for key in state):
                raise ValueError("qkv_layout=packed requires an offline packed checkpoint")
    return True


def _validate_video_size(width, height, length):
    if width < 16 or height < 16 or width % 16 or height % 16:
        raise ValueError(
            "Vimoo width and height must be positive multiples of 16; dimensions are not rounded."
        )
    if length < 1 or (length - 1) % 4:
        raise ValueError("Vimoo length must be 4n+1 (1, 5, 9, ...); frame counts are not rounded.")


def _validate_video_inputs(node, width, height, length):
    """Reject literal sizes before loading dependencies; linked sizes are checked at runtime."""
    try:
        _validate_video_size(
            16 if width is None else width,
            16 if height is None else height,
            5 if length is None else length,
        )
    except ValueError as error:
        return str(error)
    # Custom validation replaces ComfyUI's bounds checks for these fields.
    required = node.INPUT_TYPES()["required"]
    for name, value in (("width", width), ("height", height), ("length", length)):
        if value is None:
            continue
        limits = required[name][1]
        if "min" in limits and value < limits["min"]:
            return f"Vimoo {name} must be at least {limits['min']}."
        if "max" in limits and value > limits["max"]:
            return f"Vimoo {name} must be at most {limits['max']}."
    return True


def _detect_vimoo_config(sd, metadata, model_name, qkv_layout):
    """Validate the checkpoint architecture before selecting its checkpoint QKV views."""
    unet_config = comfy.model_detection.detect_unet_config(sd, "")
    if unet_config is None:
        raise ValueError(f"Unable to detect the Vimoo transformer configuration in {model_name!r}.")
    required = ("byt5_in.fc1.weight", "byt5_in.fc2.weight", "byt5_in.fc3.weight")
    missing = [key for key in required if key not in sd]
    if missing:
        raise ValueError(f"{model_name!r} is not a Vimoo transformer; missing keys: {missing}")
    fc1 = sd["byt5_in.fc1.weight"]
    fc2 = sd["byt5_in.fc2.weight"]
    fc3 = sd["byt5_in.fc3.weight"]

    unet_config["image_model"] = "vimoo"
    unet_config["byt5_dims"] = [fc1.shape[1], fc1.shape[0], fc2.shape[0], fc3.shape[0]]
    unet_config["rope_spatial_interpolation_ref_area"] = 1600.0
    unet_config["rope_spatial_interpolation_strength"] = 1.0
    unet_config["qk_norm_eps"] = 1e-6
    hms_keys = {"img_in.patch_merge.weight", "final_layer.patch_split.weight"}
    hms = bool(hms_keys.intersection(sd))
    if hms and not hms_keys.issubset(sd):
        raise ValueError(
            "Incomplete hidden_merge_split transformer; both patch_merge and patch_split weights are required"
        )
    if hms and not {"img_in.patch_merge.bias", "final_layer.patch_split.bias"}.issubset(sd):
        raise ValueError(
            "Incomplete hidden_merge_split transformer; patch_merge and patch_split biases are required"
        )
    source_config = _apply_source_metadata(sd, metadata, unet_config, hms)
    _configure_source_policy(sd, source_config, unet_config, hms)
    if qkv_layout == "auto" and source_config.get("sampling_profile") == "lite_ti2v_anchor":
        qkv_layout = "packed"
    unet_config["packed_qkv"] = _prepare_qkv_layout(sd, metadata, qkv_layout, unet_config["depth"])
    return unet_config, source_config, hms, qkv_layout


def _apply_source_metadata(sd, metadata, unet_config, hms):
    """Match the declared transformer architecture against the checkpoint tensors."""
    source_config = {}
    if metadata and "vimoo_config" in metadata:
        source_config = json.loads(metadata["vimoo_config"])
        if metadata.get("vimoo_conversion_version") != "1":
            raise ValueError("Unsupported Vimoo checkpoint conversion version")
        # Each checkpoint family specifies intermediate precision and operation ordering.
        # Keep its numerical profile aligned with the detected architecture so loading
        # cannot silently switch the arithmetic used for its reference outputs.
        expected_profile = "lite" if hms else "standard"
        if "numerical_profile" in source_config and source_config["numerical_profile"] != expected_profile:
            raise ValueError(f"Vimoo numerical_profile must be {expected_profile!r} for this architecture")
        if (
            source_config.get("text_projection") != "single_refiner"
            or source_config.get("qk_norm_type") != "rms"
            or not source_config.get("qk_norm")
            or source_config.get("is_reshape_temporal_channels", False)
        ):
            raise ValueError("Unsupported Vimoo architecture in checkpoint metadata")
        if hms:
            hidden = unet_config["hidden_size"]
            if (
                source_config.get("patch_size") != [1, 2, 2]
                or unet_config["patch_size"] != [1, 1, 1]
                or list(sd["img_in.patch_merge.weight"].shape) != [hidden, 4 * hidden]
                or list(sd["final_layer.patch_split.weight"].shape) != [4 * hidden, hidden]
            ):
                raise ValueError("hidden_merge_split metadata/tensor mismatch for patch projections")
            unet_config["patch_size"] = [1, 2, 2]
        for native, original in (
            ("hidden_size", "hidden_size"),
            ("depth", "mm_double_blocks_depth"),
            ("out_channels", "out_channels"),
            ("patch_size", "patch_size"),
            ("context_in_dim", "text_states_dim"),
        ):
            if unet_config[native] != source_config[original]:
                raise ValueError(f"Vimoo metadata/tensor mismatch for {original}")
        expected_input = source_config["in_channels"] * (2 if source_config.get("concat_condition") else 1)
        expected_input += int(bool(source_config.get("concat_condition")))
        if unet_config["in_channels"] != expected_input:
            raise ValueError("Vimoo metadata/tensor mismatch for in_channels")
        unet_config.update(
            num_heads=source_config["heads_num"],
            axes_dim=source_config["rope_dim_list"],
            theta=source_config["rope_theta"],
            mlp_ratio=source_config["mlp_width_ratio"],
            rope_spatial_interpolation_ref_area=source_config.get(
                "rope_spatial_interpolation_ref_area", 1600.0
            ),
            rope_spatial_interpolation_strength=source_config.get("rope_spatial_interpolation_strength", 1.0),
        )
    return source_config


def _configure_source_policy(sd, source_config, unet_config, hms):
    """Validate factorized modulation and set the checkpoint patch and precision policies."""
    activation = source_config.get("mlp_act_type", "gelu_tanh")
    if activation not in ("gelu", "gelu_tanh"):
        raise ValueError(f"Unsupported Vimoo MLP activation in checkpoint metadata: {activation!r}")
    unet_config["mlp_act_type"] = activation
    rank = source_config.get("block_modulation_rank")
    bottleneck = "mod_projection.weight" in sd
    if bool(rank) != bottleneck:
        raise ValueError(
            "Vimoo modulation metadata/tensor mismatch; a factorized checkpoint needs both "
            "block_modulation_rank in its config and a mod_projection weight"
        )
    if rank:
        hidden = unet_config["hidden_size"]
        precision = source_config.get("block_modulation_precision") or "fp32"
        validate_modulation_state(sd, hidden, unet_config["depth"], rank, precision)
        unet_config["block_modulation_rank"] = int(rank)
        unet_config["block_modulation_precision"] = precision
    architecture = source_config.get("patch_architecture", "direct")
    if hms != (architecture == "hidden_merge_split"):
        raise ValueError(
            "Vimoo patch architecture metadata/tensor mismatch; hidden_merge_split requires its model config"
        )
    if hms and source_config.get("sampling_profile") != "lite_ti2v_anchor":
        raise ValueError("hidden_merge_split requires the current Lite TI2V sampling profile")
    unet_config.update(
        use_text_refiner_attention_mask=source_config.get(
            "use_text_refiner_attention_mask", source_config.get("use_attention_mask", True)
        ),
        rope_spatial_interpolation_ramp=source_config.get("rope_spatial_interpolation_ramp", "yarn"),
        patch_architecture=architecture,
        precision_policy=source_config.get("precision_policy", "bf16"),
        rope_temporal_interpolation_ref_fps=source_config.get("rope_temporal_interpolation_ref_fps"),
        rope_temporal_interpolation_mode=source_config.get("rope_temporal_interpolation_mode", "divide"),
        attention_projection_layout=source_config.get("attention_projection_layout", "strided"),
    )


def _cast_source_io_weights(sd, unet_config):
    """Retain FP32 checkpoint weights at the input, output and timestep boundaries."""
    if unet_config["precision_policy"] == "fp32_io_time":
        prefixes = (
            "img_in.",
            "final_layer.linear.",
            "final_layer.patch_split.",
            "time_in.",
            "txt_in.t_embedder.",
        )
        for key in sd:
            if key.startswith(prefixes):
                sd[key] = sd[key].float()


def _configure_anchor_precision(model, source_config):
    """Install the Lite TI2V linear, positional and normalization policies."""
    if source_config.get("sampling_profile") == "lite_ti2v_anchor":
        configure_anchor_linears(model.diffusion_model)
        model.diffusion_model.final_layer.adaLN_modulation.to(dtype=torch.bfloat16)
        model.diffusion_model.pe_embedder.compute_on_device = True
        model.diffusion_model.byt5_in.layernorm.eps = 1e-6
        if model.diffusion_model.vision_in is not None:
            for layer in (0, 4):
                model.diffusion_model.vision_in.proj[layer].eps = 1e-6


class VimooModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
                "dtype": (["bf16", "fp16", "fp32", "auto"], {"default": "bf16"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "Vimoo"

    def load(
        self,
        model_name,
        dtype,
        qkv_layout="auto",
        disable_dynamic=False,
    ):
        path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
        if "vimoo_config" not in (metadata or {}) and Path(str(path) + ".vimoo.json").is_file():
            provenance = checkpoint_provenance(path)
            if provenance.get("checkpoint_layout") == "vimoo_low_rank":
                config = provenance.get("config")
                if (
                    not isinstance(config, dict)
                    or config.get("block_modulation_rank") != 1536
                    or config.get("block_modulation_precision") != "bf16_expand_fp32_bias"
                    or config.get("patch_size") != [1, 1, 1]
                ):
                    raise ValueError(
                        "Checkpoint with shared low-rank timestep modulation requires its complete architecture config"
                    )
                # All large parameters keep their original storage; no model copy,
                # QKV concatenation, or base-modulation reconstruction is needed.
                sd = normalize_transformer_state(sd.items(), config, packed_qkv=False)
                metadata = {
                    **(metadata or {}),
                    "vimoo_config": json.dumps(config),
                    "vimoo_conversion_version": "1",
                    "vimoo_qkv_layout": "split",
                }
        # Native quantized checkpoints (comfy-quants int8_tensorwise / fp8 / nvfp4)
        # carry per-layer ``comfy_quant`` markers or legacy metadata; normalize
        # them the same way ComfyUI's own diffusion-model loader does.
        sd, metadata = comfy.utils.convert_old_quants(sd, "", metadata=metadata)
        unet_config, source_config, hms, qkv_layout = _detect_vimoo_config(
            sd, metadata, model_name, qkv_layout
        )
        model_config = Vimoo(unet_config)
        # Same dtype policy as comfy.sd.load_diffusion_model_state_dict: a
        # quantized file has no single weight dtype, so ``auto`` falls back to
        # the first dtype the model config supports (bf16 for Vimoo); an
        # explicit widget choice is honored either way.
        model_config.quant_config = comfy.utils.detect_layer_quantization(sd, "")
        if model_config.quant_config is not None:
            checkpoint_provenance(path)
        supported_dtypes = list(model_config.supported_inference_dtypes)
        weight_dtype = None if model_config.quant_config is not None else comfy.utils.weight_dtype(sd)

        if dtype == "auto":
            unet_dtype = comfy.model_management.unet_dtype(
                model_params=comfy.utils.calculate_parameters(sd),
                supported_dtypes=supported_dtypes,
                weight_dtype=weight_dtype,
            )
        else:
            unet_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]

        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()
        manual_cast_dtype = comfy.model_management.unet_manual_cast(
            None if model_config.quant_config is not None else unet_dtype, load_device, supported_dtypes
        )
        model_config.set_inference_dtype(unet_dtype, manual_cast_dtype, device=load_device)

        _cast_source_io_weights(sd, unet_config)
        model = model_config.get_model(sd, "")
        _configure_anchor_precision(model, source_config)
        patcher_class = (
            comfy.model_patcher.ModelPatcher if disable_dynamic else comfy.model_patcher.CoreModelPatcher
        )
        model_patcher = patcher_class(model, load_device=load_device, offload_device=offload_device)
        if not comfy.model_management.is_device_cpu(offload_device):
            model.to(offload_device)
        expected_shapes = {
            key: tuple(value.shape) for key, value in model.diffusion_model.state_dict().items()
        }
        # DynamicVRAM and mixed-precision linears create their weights only
        # while loading. Validate their declared dimensions before that step.
        for name, layer in model.diffusion_model.named_modules():
            if hasattr(layer, "in_features") and hasattr(layer, "out_features"):
                expected_shapes[f"{name}.weight"] = (layer.out_features, layer.in_features)
                if layer.bias is not None or getattr(layer, "comfy_need_lazy_init_bias", False):
                    expected_shapes[f"{name}.bias"] = (layer.out_features,)
        quant_keys = set()
        for key in sd:
            if key.endswith(".comfy_quant") and key.removesuffix("comfy_quant") + "weight" in expected_shapes:
                config = json.loads(bytes(sd[key].tolist()))
                parameters = comfy.ops.QUANT_ALGOS[config["format"]]["parameters"]
                quant_keys.add(key)
                quant_keys.update(key.removesuffix("comfy_quant") + name for name in parameters)
        missing = expected_shapes.keys() - sd.keys()
        if missing:
            raise ValueError(
                f"Incomplete Vimoo transformer {model_name!r}; missing weights: {sorted(missing)}"
            )
        unexpected = sd.keys() - (expected_shapes.keys() | quant_keys)
        if unexpected:
            raise ValueError(f"Unexpected Vimoo transformer weights: {sorted(unexpected)}")
        for key, shape in expected_shapes.items():
            if tuple(sd[key].shape) != shape:
                raise ValueError(f"Vimoo transformer shape mismatch for {key!r}")
        # A regional-compile delegate is non-dynamic, but it is still safe to
        # adopt the immutable safetensors storage directly.  Copy-loading a
        # full-size BF16 checkpoint momentarily keeps both the mmap and a full
        # CPU clone resident, increasing peak CPU memory.
        model.load_model_weights(
            sd,
            "",
            assign=model_patcher.is_dynamic() or disable_dynamic,
        )
        # Match ComfyUI's core loaders: DynamicVRAM clones need a factory that
        # can reconstruct a non-dynamic delegate for torch.compile.
        model_patcher.cached_patcher_init = (
            self.load,
            (model_name, dtype, qkv_layout),
            0,
        )
        model_patcher.model_options["vimoo_source_file"] = path
        profile = source_config.get("sampling_profile")
        if profile is not None:
            supported = (profile == "lite_ti2v_anchor" and hms) or (
                profile in ("standard_t2v", "standard_i2v", "t2v_distilled", "i2v_distilled")
                and not hms
                and unet_config.get("block_modulation_precision") == "bf16_expand_fp32_bias"
            )
            if not supported:
                raise ValueError(f"Unsupported Vimoo sampling profile: {profile!r}")
            model_patcher.model_options["vimoo_sampling_profile"] = profile
            if profile in ("standard_t2v", "standard_i2v", "t2v_distilled", "i2v_distilled"):
                model_patcher.model_options.setdefault("transformer_options", {})[
                    "vimoo_symmetric_self_attention"
                ] = True
        if source_config.get("source_image_decode") is not None:
            model_patcher.model_options["vimoo_source_image_decode"] = validate_source_image_decode(
                source_config["source_image_decode"]
            )
        if source_config.get("image_preprocessing") is not None:
            model_patcher.model_options["vimoo_image_preprocessing"] = source_config["image_preprocessing"]
        if source_config.get("text_encoder_precision_profile") is not None:
            model_patcher.model_options["vimoo_text_encoder_precision_profile"] = source_config[
                "text_encoder_precision_profile"
            ]
        if source_config.get("vae_numerical_profile") is not None:
            model_patcher.model_options["vimoo_vae_numerical_profile"] = validate_vae_numerical_profile(
                source_config["vae_numerical_profile"]
            )
        model_patcher.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
            "vimoo_modulation_schedule",
            _precompute_modulation_sample_wrapper,
        )
        return (model_patcher,)


class VimooTextEncoderLoader:
    DESCRIPTION = (
        "Loads Qwen for scene/motion text and ByT5 for text rendered in the video. "
        "Text Encode uses these shared weights to encode each prompt. "
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_name": (folder_paths.get_filename_list("text_encoders"),),
                "byt5_name": (folder_paths.get_filename_list("text_encoders"),),
            },
            "optional": {
                "attention_backend": (
                    ["comfy", "precise"],
                    {
                        "default": "comfy",
                        "tooltip": "Qwen text attention only. Video-model attention is selected in Performance Options.",
                    },
                ),
                "precision_profile": (
                    ["bf16_transposed_ffn", "standard_t2v", "standard_i2v", "lite_ti2v_anchor"],
                    {"default": "bf16_transposed_ffn"},
                ),
                "full_task": ("STRING", {"forceInput": True}),
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load"
    CATEGORY = "Vimoo"

    def load(
        self,
        qwen_name,
        byt5_name,
        model_options=None,
        disable_dynamic=False,
        attention_backend="comfy",
        precision_profile="bf16_transposed_ffn",
        full_task=None,
        model=None,
    ):
        if full_task is not None:
            precision_profile = "standard_t2v" if full_task == "t2v" else "bf16_transposed_ffn"
        if model is not None:
            precision_profile = model.model_options.get(
                "vimoo_text_encoder_precision_profile", precision_profile
            )
        if precision_profile not in (
            "bf16_transposed_ffn",
            "standard_t2v",
            "standard_i2v",
            "lite_ti2v_anchor",
            "vimoo_bf16",
            "lite_compact",
        ):
            raise ValueError("Unknown Vimoo text precision profile")
        paths = [
            folder_paths.get_full_path_or_raise("text_encoders", qwen_name),
            folder_paths.get_full_path_or_raise("text_encoders", byt5_name),
        ]
        state_dicts = []
        parameters = 0
        required_keys = [
            "model.layers.0.self_attn.k_proj.weight",
            "encoder.block.0.layer.0.SelfAttention.o.weight",
        ]
        for path, required_key in zip(paths, required_keys):
            state_dict, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
            state_dict, _ = comfy.utils.convert_old_quants(state_dict, model_prefix="", metadata=metadata)
            if required_key not in state_dict:
                raise ValueError(f"Wrong Vimoo text encoder checkpoint {path!r}; missing {required_key!r}.")
            parameters += comfy.utils.calculate_parameters(state_dict)
            state_dicts.append(state_dict)

        target = comfy.supported_models_base.ClipTarget(VimooTokenizer, VimooTextEncoder)
        target.params = llama_detect(state_dicts[0])
        if precision_profile in ("standard_i2v", "lite_ti2v_anchor", "lite_compact"):
            target.params["dtype_llama"] = torch.float16
        clip = comfy.sd.CLIP(
            target=target,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            parameters=parameters,
            state_dict=state_dicts,
            model_options=model_options or {},
            disable_dynamic=disable_dynamic,
        )
        clip.patcher.cached_patcher_init = (
            _reload_vimoo_text,
            (qwen_name, byt5_name, model_options, attention_backend, precision_profile),
        )
        patch_vimoo_text(clip, attention_backend, precision_profile)
        clip.cond_stage_model.vimoo_precision_profile = precision_profile
        return (clip,)


def _reload_vimoo_text(
    qwen_name,
    byt5_name,
    model_options,
    attention_backend="comfy",
    precision_profile="bf16_transposed_ffn",
    disable_dynamic=False,
):
    (clip,) = VimooTextEncoderLoader().load(
        qwen_name, byt5_name, model_options, disable_dynamic, attention_backend, precision_profile
    )
    return clip.patcher


class VimooEmptyLatentVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1280, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 720, "min": 16, "max": 8192, "step": 16}),
                "length": ("INT", {"default": 121, "min": 5, "max": 1000, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 1}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "Vimoo"

    @classmethod
    def VALIDATE_INPUTS(cls, width=None, height=None, length=None):
        return _validate_video_inputs(cls, width, height, length)

    def generate(self, width, height, length, batch_size=1):
        _validate_video_size(width, height, length)
        latent_frames = (length - 1) // 4 + 1
        latent = torch.zeros([batch_size, 48, latent_frames, height // 16, width // 16])
        return ({"samples": latent},)


class VimooImageToVideo:
    DESCRIPTION = (
        "Adds reference-image VAE latents and SigLIP visual features to text conditioning. "
        "Optional negative conditioning is used by ordinary CFG. "
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "vae": (
                    "VAE",
                    {"tooltip": "Reference-image encoding: image → latent."},
                ),
                "clip_vision": (
                    "CLIP_VISION",
                    {"tooltip": "SigLIP reference-image features, separate from the text encoder."},
                ),
                "image": ("IMAGE",),
                "width": ("INT", {"default": 1280, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 720, "min": 16, "max": 8192, "step": 16}),
                "length": ("INT", {"default": 121, "min": 5, "max": 1000, "step": 4}),
                "first_frame_replace": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "negative": (
                    "CONDITIONING",
                    {
                        "tooltip": "Unwanted-content conditioning for ordinary CFG. Distilled CFG 1 leaves it unconnected."
                    },
                ),
                "attention_backend": (
                    ["comfy", "precise"],
                    {
                        "default": "comfy",
                        "tooltip": "SigLIP vision attention only.",
                    },
                ),
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "Vimoo"

    @classmethod
    def VALIDATE_INPUTS(cls, width=None, height=None, length=None):
        return _validate_video_inputs(cls, width, height, length)

    @staticmethod
    def _cover_crop(
        image: Image.Image, height: int, width: int, floor_center=False, rounded_center=False
    ) -> Image.Image:
        rgb = image.convert("RGB")
        if rgb.height == height and rgb.width == width:
            return rgb
        scale = max(width / rgb.width, height / rgb.height)
        resized = rgb.resize(
            (int(round(rgb.width * scale)), int(round(rgb.height * scale))),
            Image.LANCZOS,
        )
        left = (resized.width - width) // 2 if floor_center else (resized.width - width) / 2
        top = (resized.height - height) // 2 if floor_center else (resized.height - height) / 2
        if rounded_center:
            left, top = int(round(left)), int(round(top))
        return resized.crop((left, top, left + width, top + height))

    def encode(
        self,
        positive,
        vae,
        clip_vision,
        image,
        width,
        height,
        length,
        first_frame_replace=False,
        attention_backend="comfy",
        model=None,
        negative=None,
    ):
        _validate_video_size(width, height, length)
        pil = Image.fromarray(np.clip(255.0 * image[0].cpu().numpy(), 0, 255).astype(np.uint8))
        preprocessing = None if model is None else model.model_options.get("vimoo_image_preprocessing")
        if preprocessing not in (None, "standard_i2v", "lanczos_floor_bf16", "lite_compact"):
            raise ValueError(f"Unsupported Vimoo image preprocessing: {preprocessing!r}")
        cropped = self._cover_crop(pil, height, width, floor_center=preprocessing == "lanczos_floor_bf16")
        pixels = torch.from_numpy(np.asarray(cropped, dtype=np.float32) / 255.0)[
            None, :, :, :
        ]  # [1,H,W,3] in 0..1

        latent_frames = (length - 1) // 4 + 1
        if preprocessing == "lanczos_floor_bf16":
            normalized = torch.from_numpy(np.asarray(cropped).copy()).bfloat16()[None] / 127.5 - 1.0
            encoded = vae.encode_normalized(normalized).to(torch.float32)
        elif preprocessing == "standard_i2v":
            vae_cropped = self._cover_crop(pil, height, width, rounded_center=True)
            normalized = torch.from_numpy(np.asarray(vae_cropped).copy()).float()[None]
            normalized = normalized.div(255.0).sub(0.5).div(0.5)
            encoded = vae.encode_normalized(normalized).to(torch.float32)
        elif preprocessing == "lite_compact":
            vae_cropped = self._cover_crop(pil, height, width, rounded_center=True)
            normalized = torch.from_numpy(np.asarray(vae_cropped).copy()).float()
            normalized = normalized.div(255.0).sub(0.5).div(0.5).bfloat16()[None]
            encoded = vae.encode_normalized(normalized).to(torch.float32)
        else:
            encoded = vae.encode(pixels).to(torch.float32)  # raw (unnormalized) latent

        # Pass the 1-frame reference only: VimooBaseModel.concat_cond normalizes
        # it (Wan22 mean/std) and ZERO-pads the remaining latent frames. Padding
        # here with raw-space zeros would normalize to -mean/std, not zero.
        # concat_mask convention: 0 = conditioned frame (model_base flips to Vimoo's 1)
        concat_mask = torch.ones([1, 1, latent_frames, encoded.shape[-2], encoded.shape[-1]])
        concat_mask[:, :, : encoded.shape[2]] = 0.0

        if preprocessing is None:
            clip_vision_output = encode_vimoo_vision(clip_vision, pixels, attention_backend)
        else:
            clip_vision_output = encode_vimoo_vision(
                clip_vision, pixels, attention_backend, processor_backend="pil"
            )
        values = {
            "concat_latent_image": encoded,
            "concat_mask": concat_mask,
            "clip_vision_output": clip_vision_output,
            "vimoo_ref_latent": None,
        }
        if preprocessing in ("standard_i2v", "lanczos_floor_bf16", "lite_compact"):
            # These conditioning profiles mark every supplied SigLIP token valid
            # with the same integer mask dtype as its text conditioning.
            vision = clip_vision_output.last_hidden_state
            values["vimoo_vision_mask"] = torch.ones(
                vision.shape[:2],
                dtype=torch.bool if preprocessing == "lite_compact" else torch.int64,
                device=vision.device,
            )
        new_negative = None if negative is None else node_helpers.conditioning_set_values(negative, values)
        if first_frame_replace:
            values["vimoo_ref_latent"] = encoded[:, :, :1].clone()
        new_positive = node_helpers.conditioning_set_values(positive, values)

        latent = torch.zeros([1, 48, latent_frames, encoded.shape[-2], encoded.shape[-1]])
        return (new_positive, new_negative, {"samples": latent})


class VimooScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scheduler": (["vimoo", "vimoo_i2v_euler", "vimoo_lite_ti2v_euler", *DISTILLED_SCHEDULES],),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "flow_shift": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "Vimoo"

    @classmethod
    def VALIDATE_INPUTS(cls, scheduler, steps, flow_shift):
        try:
            cls().get_sigmas(scheduler, steps, flow_shift)
        except ValueError as error:
            return str(error)
        return True

    def get_sigmas(self, scheduler, steps, flow_shift):
        schedule = DISTILLED_SCHEDULES.get(scheduler)
        if schedule is not None:
            if steps != len(schedule.timesteps) or flow_shift != schedule.flow_shift:
                raise ValueError(
                    f"{scheduler} requires steps={len(schedule.timesteps)} and flow_shift={schedule.flow_shift:g}"
                )
            return (
                torch.tensor([*(value / 1000.0 for value in schedule.timesteps), 0.0], dtype=schedule.dtype),
            )
        if scheduler == "vimoo_lite_ti2v_euler":
            from .vimoo_ti2v import ti2v_sigmas

            return (ti2v_sigmas(steps, flow_shift),)
        if scheduler == "vimoo_i2v_euler":
            from .vimoo_i2v import i2v_sigmas

            return (i2v_sigmas(steps, flow_shift),)
        if scheduler != "vimoo":
            raise ValueError(f"Unknown Vimoo scheduler: {scheduler!r}")
        return (vimoo_unipc_sigmas(flow_shift, steps),)


class VimooSampler:
    FIXED_NOISE_PROFILE = None
    DESCRIPTION = (
        "Positive describes wanted content; negative describes content to avoid. "
        "Vimoo-1.0-Lite TI2V I2V can need negative conditioning for reference CFG even at text CFG 1. "
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "sigmas": ("SIGMAS",),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 4.5,
                        "min": 0.0,
                        "step": 0.1,
                        "tooltip": "Text guidance strength. Lite reference-image guidance has its own scale.",
                    },
                ),
                "guidance_rescale": ("FLOAT", {"default": 1.2, "min": 0.0, "step": 0.05}),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Initial sampling noise seed.",
                    },
                ),
                "sampler_name": (
                    "COMBO",
                    {
                        "options": [
                            "vimoo_uni_pc",
                            "vimoo_i2v_euler",
                            "vimoo_lite_ti2v_euler",
                            "stochastic_flowmatch",
                            *comfy.samplers.SAMPLER_NAMES,
                        ],
                        "default": "vimoo_uni_pc",
                    },
                ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "renoise_seed": (
                    "INT",
                    {
                        "default": 1001237,
                        "min": -1,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "stochastic_flowmatch step noise only: 1001237 by default; -1 follows seed. Other samplers ignore it.",
                    },
                ),
            },
            "optional": {
                "negative": (
                    "CONDITIONING",
                    {
                        "lazy": True,
                        "tooltip": "Required by ordinary text/reference CFG branches. Distilled CFG 1 uses positive only.",
                    },
                ),
                "fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 0.01,
                        "max": 1000.0,
                        "tooltip": "Generation frame rate used by temporal RoPE; connect the same value to CreateVideo.",
                    },
                ),
                "add_fps": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Vimoo-1.0 I2V: interpolate temporal RoPE from the trained 16 fps time base.",
                    },
                ),
                "reference_guidance_scale": (
                    "FLOAT",
                    {
                        "default": LITE_TI2V_DEFAULTS["i2v"]["reference_guidance_scale"],
                        "min": 0.0,
                        "tooltip": "Vimoo-1.0-Lite TI2V I2V reference guidance. 1 keeps the reference image without extra reference CFG; T2V ignores it. Vimoo-1.0-Lite TI2V Distilled requires 1.",
                    },
                ),
                "i2v_anchor_tau": (
                    "FLOAT",
                    {
                        "default": LITE_TI2V_DEFAULTS["i2v"]["i2v_anchor_tau"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "Vimoo"

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        sampler_name=None,
        cfg=None,
        guidance_rescale=None,
        denoise=None,
        reference_guidance_scale=None,
        i2v_anchor_tau=None,
    ):
        if denoise is not None and (not math.isfinite(denoise) or not 0.0 <= denoise <= 1.0):
            return "denoise must be between 0 and 1."
        if sampler_name == "vimoo_lite_ti2v_euler":
            try:
                validate_guidance_values(
                    guidance_scale=cfg,
                    guidance_rescale=guidance_rescale,
                    reference_guidance_scale=reference_guidance_scale,
                    i2v_anchor_tau=i2v_anchor_tau,
                )
            except ValueError as error:
                return str(error)
            if denoise is not None and denoise != 1.0:
                return "vimoo_lite_ti2v_euler requires denoise=1."
            return True
        if cfg is not None and (not math.isfinite(cfg) or not 0.0 <= cfg <= 30.0):
            return "cfg must be between 0 and 30."
        if guidance_rescale is not None and (
            not math.isfinite(guidance_rescale) or not 0.0 <= guidance_rescale <= 2.0
        ):
            return "guidance_rescale must be between 0 and 2."
        if sampler_name == "stochastic_flowmatch" and (
            (cfg is not None and cfg != 1.0)
            or (guidance_rescale is not None and guidance_rescale != 0.0)
            or (denoise is not None and denoise != 1.0)
        ):
            return (
                "stochastic_flowmatch requires cfg=1, guidance_rescale=0 and denoise=1; "
                "update the visible sampler controls. Parameters are not overridden."
            )
        if sampler_name == "vimoo_i2v_euler" and (
            (cfg is not None and cfg < 1.0)
            or (guidance_rescale is not None and guidance_rescale > 1.0)
            or (denoise is not None and denoise != 1.0)
        ):
            return f"{sampler_name} requires cfg in [1,30], guidance_rescale in [0,1] and denoise=1."
        return True

    @staticmethod
    def _needs_negative(model, positive, cfg, reference_guidance_scale):
        options = {} if model is None else model.model_options
        if options.get("vimoo_sampling_profile") == LITE_TI2V_DEFAULTS["profile"]:
            observed = any(extra.get("concat_latent_image") is not None for _, extra in positive or [])
            if reference_guidance_scale is None:
                reference_guidance_scale = (
                    1.0
                    if options.get("vimoo_distilled")
                    else LITE_TI2V_DEFAULTS["i2v"]["reference_guidance_scale"]
                )
            return len(ti2v_guidance_branches(cfg, reference_guidance_scale, observed)) > 1
        return cfg != 1.0 or options.get("disable_cfg1_optimization", False)

    def check_lazy_status(self, cfg, model=None, positive=None, reference_guidance_scale=None, **kwargs):
        if "negative" in kwargs and kwargs["negative"] is None:
            if self._needs_negative(model, positive, cfg, reference_guidance_scale):
                return ["negative"]
        return []

    def sample(
        self,
        model,
        positive,
        latent_image,
        sigmas,
        cfg,
        guidance_rescale,
        seed,
        sampler_name,
        denoise,
        renoise_seed=1001237,
        fps=24.0,
        add_fps=True,
        reference_guidance_scale=None,
        i2v_anchor_tau=None,
        negative=None,
    ):
        distilled = model.model_options.get("vimoo_distilled")
        if distilled and distilled.get("profile") == "lite_ti2v_distilled" and not self.FIXED_NOISE_PROFILE:
            raise ValueError(
                "Use Vimoo Lite Distilled Sampler for the fixed noise recipe of Vimoo-1.0-Lite TI2V Distilled"
            )
        if any(value is None for value in (sampler_name, cfg, guidance_rescale, denoise)):
            raise ValueError("Sampler inputs must resolve before execution")
        valid = self.VALIDATE_INPUTS(
            sampler_name, cfg, guidance_rescale, denoise, reference_guidance_scale, i2v_anchor_tau
        )
        if valid is not True:
            raise ValueError(valid)
        if negative is None and self._needs_negative(model, positive, cfg, reference_guidance_scale):
            raise ValueError("Connect negative conditioning when text or reference guidance requires it")
        profile = model.model_options.get("vimoo_sampling_profile")
        source_sampler = {"standard_t2v": "vimoo_uni_pc", "standard_i2v": "vimoo_i2v_euler"}.get(profile)
        if source_sampler is not None and sampler_name != source_sampler:
            raise ValueError(f"{profile} requires sampler_name={source_sampler}")
        if sampler_name == "vimoo_lite_ti2v_euler":
            from .vimoo_ti2v import sample_lite_ti2v

            return sample_lite_ti2v(
                model,
                positive,
                negative,
                latent_image,
                sigmas,
                cfg,
                guidance_rescale,
                seed,
                fps,
                reference_guidance_scale,
                i2v_anchor_tau,
            )
        if model.model_options.get("vimoo_sampling_profile") == "lite_ti2v_anchor":
            raise ValueError(
                "The Vimoo-1.0-Lite TI2V base requires vimoo_lite_ti2v_euler and its lognorm schedule"
            )
        if sampler_name == "vimoo_i2v_euler":
            from .vimoo_i2v import sample_full_i2v

            return sample_full_i2v(
                model,
                positive,
                negative,
                latent_image,
                sigmas,
                cfg=cfg,
                guidance_rescale=guidance_rescale,
                seed=seed,
                fps=fps,
                add_fps=add_fps,
                recipe=model.model_options.get("vimoo_distilled"),
            )
        if fps is not None:
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError("fps must be positive and finite")
            positive = node_helpers.conditioning_set_values(positive, {"fps": fps})
            if negative is not None:
                negative = node_helpers.conditioning_set_values(negative, {"fps": fps})
        if sampler_name == "stochastic_flowmatch":
            return sample_distilled(model, positive, latent_image, sigmas, cfg, seed, renoise_seed)
        if model.model_options.get("vimoo_distilled"):
            raise ValueError("Distilled LoRA requires sampler_name=stochastic_flowmatch in Vimoo Sampler")
        return sample_standard(
            model,
            positive,
            negative,
            latent_image,
            sigmas,
            sampler_name=sampler_name,
            cfg=cfg,
            guidance_rescale=guidance_rescale,
            seed=seed,
            denoise=denoise,
        )


class VimooTextEncode:
    DESCRIPTION = (
        "Encodes one prompt. Connect separate instances to positive and negative for ordinary CFG; "
        "distilled CFG 1 uses one positive instance. Vimoo-1.0-Lite TI2V uses Paired Text Encode for its CFG batch. "
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "Positive: wanted scene/motion. Negative: unwanted content. Its connection determines the role.",
                    },
                ),
                "clip": ("CLIP",),
                "glyph_conditioning": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "ByT5 conditioning for text rendered inside the video. Usually on for positive, off for negative.",
                    },
                ),
            },
            "optional": {
                "max_sequence_length": (
                    "INT",
                    {
                        "default": 1000,
                        "min": 1,
                        "tooltip": "Text limit: 1000 for Vimoo-1.0 models, 1400 for Vimoo-1.0-Lite models. A connected schedule supplies this value.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "Vimoo"

    def encode(self, clip, text, glyph_conditioning=True, max_sequence_length=1000):
        tokens = clip.tokenize(
            text, glyph_conditioning=glyph_conditioning, max_sequence_length=max_sequence_length
        )
        if getattr(clip.cond_stage_model, "vimoo_precision_profile", None) in (
            LITE_TI2V_DEFAULTS["profile"],
            "lite_compact",
        ):
            tokens = {"qwen25_7b": tokens["qwen25_7b"], "vimoo_cfg_prompts": [tokens]}
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        required = {"attention_mask", "conditioning_byt5small", "vimoo_byt5_mask"}
        if any(not required.issubset(extra) for _, extra in conditioning):
            raise ValueError(
                "CLIP encoding lost required Vimoo Qwen/glyph conditioning. "
                "If using Set CLIP Hooks, disable schedule_clip. "
                "Incomplete conditioning cannot be used for sampling."
            )
        return (conditioning,)


_PROBED = {}


def encoder_available(name):
    """True when this machine can actually open ``name`` and encode a frame.

    Merely finding the codec is not enough: an ffmpeg build can carry NVENC while the
    device refuses a session ("OpenEncodeSessionEx failed: unsupported device"), which
    is what happens on some cloud instances.  The probe result is cached.
    """
    if name in _PROBED:
        return _PROBED[name]
    result = False
    try:
        import av

        with av.open(os.devnull, mode="w", format="mp4") as container:
            stream = container.add_stream(name, rate=24)
            stream.width, stream.height, stream.pix_fmt = 256, 256, "yuv420p"
            container.mux(
                stream.encode(av.VideoFrame.from_ndarray(np.zeros((256, 256, 3), np.uint8), format="rgb24"))
            )
            container.mux(stream.encode())
        result = True
    except Exception:
        result = False
    _PROBED[name] = result
    return result


def frame_bytes(images):
    """``images`` ([N, H, W, C] float in 0..1) as host uint8, converted on the GPU if it fits."""
    images = images.detach()
    if not images.is_cuda and torch.cuda.is_available():
        try:
            return (
                images.to("cuda", non_blocking=True)
                .clamp(0, 1)
                .mul(255)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    return images.clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()


def write_video(images, path, fps, codec, quality, preset="fast", metadata=None):
    """Encode ``images`` ([N, H, W, C] float in 0..1) to ``path``; returns the codec used."""
    import av

    frames = frame_bytes(images)
    height, width = frames.shape[1:3]
    options = {"movflags": "use_metadata_tags"} if metadata else None
    with av.open(path, mode="w", options=options) as container:
        if metadata:
            container.metadata.update({key: json.dumps(value) for key, value in metadata.items()})
        stream = container.add_stream(codec, rate=int(round(fps)))
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        # NVENC takes a constant-quantiser target; libx264 takes a CRF.
        stream.options = (
            {"cq": str(quality), "preset": "p4"}
            if "nvenc" in codec
            else {"crf": str(quality), "preset": preset}
        )
        for frame in frames:
            container.mux(
                stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24"))
            )
        container.mux(stream.encode())
    return codec


class VimooSaveVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "vimoo_t2v"}),
                "fps": (
                    "FLOAT",
                    {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0, "forceInput": True},
                ),
                "encoder": (
                    ["auto", "h264_nvenc", "libx264"],
                    {
                        "default": "auto",
                        "tooltip": "auto prefers NVENC and falls back to libx264.",
                    },
                ),
                "quality": (
                    "INT",
                    {
                        "default": 19,
                        "min": 0,
                        "max": 51,
                        "tooltip": "NVENC constant quantiser, or CRF for libx264. Lower is better quality.",
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Vimoo"
    DESCRIPTION = "Write the frames straight to an MP4, using NVENC when the ffmpeg build has it."

    def save(
        self, images, filename_prefix, fps, encoder, quality, preset="fast", prompt=None, extra_pnginfo=None
    ):
        if encoder == "auto":
            encoder = "h264_nvenc" if encoder_available("h264_nvenc") else "libx264"
        elif not encoder_available(encoder):
            raise ValueError(f"This ffmpeg build cannot encode with {encoder!r}")
        directory, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), images.shape[2], images.shape[1]
        )
        file = f"{name}_{counter:05}_.mp4"
        path = os.path.join(directory, file)
        metadata = None
        if not args.disable_metadata:
            metadata = dict(extra_pnginfo or {})
            if prompt is not None:
                metadata["prompt"] = prompt
        try:
            used = write_video(images, path, fps, encoder, quality, preset, metadata=metadata)
        except Exception:
            if encoder == "libx264":
                raise
            used = write_video(images, path, fps, "libx264", quality, preset, metadata=metadata)
        return {
            "ui": {
                "images": [{"filename": file, "subfolder": subfolder, "type": "output"}],
                "animated": (True,),
                "encoder": [used],
            }
        }


NODE_CLASS_MAPPINGS = {
    "VimooVAELoader": VimooVAELoader,
    "VimooReferenceVAELoader": VimooReferenceVAELoader,
    "VimooSaveVideo": VimooSaveVideo,
    "VimooModelLoader": VimooModelLoader,
    "VimooTextEncoderLoader": VimooTextEncoderLoader,
    "VimooEmptyLatentVideo": VimooEmptyLatentVideo,
    "VimooImageToVideo": VimooImageToVideo,
    "VimooSampler": VimooSampler,
    "VimooScheduler": VimooScheduler,
    "VimooTextEncode": VimooTextEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VimooVAELoader": "Vimoo Output VAE",
    "VimooReferenceVAELoader": "Vimoo Reference VAE",
    "VimooSaveVideo": "Vimoo Save Video",
    "VimooModelLoader": "Vimoo Model Loader",
    "VimooTextEncoderLoader": "Vimoo Text Encoder Loader",
    "VimooEmptyLatentVideo": "Vimoo Empty Latent Video",
    "VimooImageToVideo": "Vimoo Image To Video",
    "VimooSampler": "Vimoo Sampler",
    "VimooScheduler": "Vimoo Scheduler",
    "VimooTextEncode": "Vimoo Text Encode",
}
