"""Strict Vimoo Distilled LoRA mapping onto native ComfyUI model patches."""

from __future__ import annotations

from functools import partial

import json
import math
from pathlib import Path
import re

import torch
from safetensors import safe_open

from .vimoo_checkpoint import checkpoint_provenance, sha256, valid_sha256
from .vimoo_vae import validate_vae_numerical_profile
from .vimoo_vision import validate_source_image_decode
from .vimoo_sampler import DISTILLED_SCHEDULES, recipe_sigmas


def map_module(module, shapes):
    module = normalize_diffusers_module(module)
    match = re.fullmatch(r"double_blocks\.(\d+)\.attn\.(to_([qkv])|add_([qkv])_proj)", module)
    if match:
        block, _, image_part, text_part = match.groups()
        stream, part = ("img", image_part) if image_part else ("txt", text_part)
        key = f"diffusion_model.double_blocks.{block}.{stream}_attn.to_{part}.weight"
        return key, tuple(shapes[key])
    if not re.fullmatch(
        r"double_blocks\.\d+\.(attn\.(to_out\.0|to_add_out)|(img|txt)_mlp\.fc[12])",
        module,
    ):
        raise ValueError(f"Unsupported Vimoo adapter target: {module}")
    name = module.replace("attn.to_out.0", "img_attn.proj").replace("attn.to_add_out", "txt_attn.proj")
    name = name.replace("_mlp.fc1", "_mlp.0").replace("_mlp.fc2", "_mlp.2")
    key = "diffusion_model." + name + ".weight"
    return key, tuple(shapes[key])


def normalize_diffusers_module(module):
    """Normalize released Diffusers LoRA target names to the native mapping."""
    module = module.replace(".attn.to_out.0.projection", ".attn.to_out.0")
    module = module.replace(".attn.to_add_out.projection", ".attn.to_add_out")
    for source, target in (
        (".img_mlp.net.0.proj", ".img_mlp.fc1"),
        (".img_mlp.net.2", ".img_mlp.fc2"),
        (".txt_mlp.net.0.proj", ".txt_mlp.fc1"),
        (".txt_mlp.net.2", ".txt_mlp.fc2"),
        (".img_mlp.in_proj.projection", ".img_mlp.fc1"),
        (".img_mlp.out_proj.projection", ".img_mlp.fc2"),
        (".txt_mlp.in_proj.projection", ".txt_mlp.fc1"),
        (".txt_mlp.out_proj.projection", ".txt_mlp.fc2"),
    ):
        module = module.replace(source, target)
    return module


def build_adapters(state, metadata, shapes, *, expected_modules=480, expected_rank=128, expected_alpha=64):
    config = json.loads(metadata["lora_adapter_metadata"])
    if (
        config.get("transformer.r") != expected_rank
        or config.get("transformer.lora_alpha") != expected_alpha
        or any(
            config.get(f"transformer.{option}")
            for option in (
                "use_dora",
                "use_rslora",
                "rank_pattern",
                "alpha_pattern",
                "modules_to_save",
                "lora_bias",
                "fan_in_fan_out",
            )
        )
        or config.get("transformer.bias", "none") != "none"
    ):
        raise ValueError(
            f"Expected Vimoo LoRA rank={expected_rank}, alpha={expected_alpha} without unsupported PEFT options"
        )
    modules = {}
    for key, tensor in state.items():
        match = re.fullmatch(r"transformer\.(.+)\.lora_([AB])\.weight", key)
        # Validate file storage before applying a profile's runtime allocation dtype.
        if (
            not match
            or tensor.dtype not in (torch.bfloat16, torch.float32)
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError(f"Invalid Vimoo LoRA tensor: {key}")
        module, side = match.groups()
        modules.setdefault(module, {})[side] = tensor
    if expected_modules is not None and len(modules) != expected_modules:
        raise ValueError("Incomplete Distilled LoRA module inventory")
    adapters = {}
    for module, pair in modules.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"Missing LoRA A/B pair: {module}")
        target, shape = map_module(module, shapes)
        a, b = pair["A"], pair["B"]
        if (
            a.ndim != 2
            or b.ndim != 2
            or a.shape[0] != expected_rank
            or b.shape[1] != expected_rank
            or (b.shape[0], a.shape[1]) != shape
        ):
            raise ValueError(f"LoRA shape mismatch: {module}")
        adapters[target.removesuffix(".weight")] = (a, b)
    return adapters


def _is_linear(module):
    # ComfyUI's mixed-precision (quantized) Linear derives from nn.Module, not nn.Linear.
    return isinstance(module, (torch.nn.Linear, LoRALinear, PackedLoRALinear)) or (
        type(module).__name__ == "Linear" and hasattr(module, "weight")
    )


def base_matches(base_path, expected_sha256, *, task=None):
    """Match the public checkpoint or its explicitly bound public base."""
    if not valid_sha256(expected_sha256):
        return False
    sidecar = Path(str(base_path) + ".vimoo.json")
    if sidecar.exists():
        try:
            binding = checkpoint_provenance(base_path, task=task)
        except ValueError:
            return False
        return expected_sha256 in (binding["sha256"], (binding.get("base") or {}).get("sha256"))
    return sha256(base_path) == expected_sha256


def load_distilled_lora(model, path):
    if model.model_options.get("vimoo_distilled") or model.patches:
        raise ValueError("Load Distilled onto an unpatched Vimoo base; adapters cannot be stacked")
    base_path = model.model_options.get("vimoo_source_file")
    if not base_path:
        raise ValueError("Use Vimoo Model Loader before Vimoo Distilled LoRA Loader")
    recipe = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
    if (
        recipe.get("format") != "vimoo-distilled-lora-v1"
        or recipe.get("profile") not in DISTILLED_SCHEDULES
        or recipe.get("task") != DISTILLED_SCHEDULES[recipe["profile"]].task
    ):
        raise ValueError("Unsupported Vimoo LoRA sidecar")
    if not base_matches(base_path, recipe.get("base_sha256"), task=recipe["task"]):
        raise ValueError(
            "LoRA requires its exact public task base (or a derivative bound to that public file in a "
            "<file>.vimoo.json binding sidecar); base checksum mismatch"
        )
    if sha256(path) != recipe.get("adapter_sha256"):
        raise ValueError("LoRA checksum mismatch")
    with safe_open(path, framework="pt", device="cpu") as f:
        state = {k: f.get_tensor(k) for k in f.keys()}
        metadata = f.metadata()
    lite_compact_runtime = recipe["profile"] == "lite_ti2v_distilled"
    if lite_compact_runtime:
        if (
            recipe["task"] != "ti2v"
            or recipe.get("rank") != 128
            or recipe.get("alpha") != 64
            or recipe.get("effective_scale") != 0.5
            or recipe.get("adapter_storage_dtype") != "float32"
            or recipe.get("adapter_compute_dtype") != "bfloat16"
            or recipe.get("text_encoder_precision_profile") != "lite_compact"
            or recipe.get("image_preprocessing") != "lite_compact"
            or recipe.get("source_image_decode") != "pil"
            or recipe.get("vae_numerical_profile") != "wan_fp32_norm"
            or any(value.dtype != torch.float32 for value in state.values())
        ):
            raise ValueError(
                "Vimoo-1.0-Lite TI2V Distilled requires rank128/alpha64 FP32 adapter masters, BF16 compute and its matching conditioning/VAE profiles"
            )
    if recipe["profile"] in ("t2v_distilled", "i2v_distilled"):
        if model.model_options.get("vimoo_sampling_profile") != recipe["profile"]:
            raise ValueError("Distilled LoRA task/profile differs from the loaded base Transformer")
        if recipe["profile"] != f"{recipe['task']}_distilled":
            raise ValueError("Distilled LoRA profile/task mismatch")
        if (
            recipe.get("adapter_format") != "diffusers_lora"
            or recipe.get("rank") != 128
            or recipe.get("alpha") != 64
            or recipe.get("effective_scale") != 0.5
            or recipe.get("adapter_storage_dtype") != "bfloat16"
            or recipe.get("adapter_compute_dtype") != "bfloat16"
            or not metadata
            or metadata.get("format") != "pt"
            or "lora_adapter_metadata" not in metadata
            or any(value.dtype != torch.bfloat16 for value in state.values())
        ):
            raise ValueError("Distilled profiles require their bound rank128/alpha64 BF16 Diffusers adapter")

    # A resident sibling clone may have installed LoRALinear object patches.
    # The patcher's view resolves their original Linear modules from its backup.
    shapes = {
        name + ".weight": tuple(model.get_model_object(name).weight.shape)
        for name, module in model.model.named_modules()
        if _is_linear(module)
    }
    for key, shape in tuple(shapes.items()):
        if re.fullmatch(r"diffusion_model\.double_blocks\.\d+\.(img|txt)_attn\.qkv\.weight", key):
            for part in "qkv":
                shapes[key.replace(".qkv.weight", f".to_{part}.weight")] = (shape[0] // 3, shape[1])
    scale = 0.5
    if recipe["task"] == "ti2v":
        rank, alpha, inventory = recipe.get("rank"), recipe.get("alpha"), recipe.get("target_modules")
        if (
            type(rank) is not int
            or rank < 1
            or type(alpha) not in (int, float)
            or not math.isfinite(alpha)
            or not isinstance(inventory, list)
            or not inventory
            or len(set(inventory)) != len(inventory)
        ):
            raise ValueError("TI2V sidecar requires rank, alpha and an exact target module inventory")
        for task in ("t2v", "i2v"):
            task_recipe = recipe.get("recipes", {}).get(task)
            if not isinstance(task_recipe, dict):
                raise ValueError(f"TI2V adapter is missing its {task} recipe")
            recipe_sigmas({**task_recipe, "task": task})
            if (
                type(task_recipe.get("max_sequence_length")) is not int
                or task_recipe["max_sequence_length"] < 1
            ):
                raise ValueError("TI2V recipe requires max_sequence_length")
            if task_recipe.get("cfg", 1.0) != 1.0 or task_recipe.get("guidance_rescale", 0.0) != 0.0:
                raise ValueError("TI2V distilled requires CFG 1 and guidance rescale 0")
        adapters = build_adapters(
            state, metadata, shapes, expected_modules=len(inventory), expected_rank=rank, expected_alpha=alpha
        )
        expected = {map_module(name, shapes)[0].removesuffix(".weight") for name in inventory}
        if set(adapters) != expected:
            raise ValueError("TI2V adapter module inventory mismatch")
        scale = alpha / rank
    else:
        adapters = build_adapters(state, metadata, shapes)
    # Keep the validated Patchify FP32 masters intact. The TI2V sampler's
    # BF16 autocast evaluates each adapter GEMM in BF16 after casting
    # its hidden input to the stored weight dtype.
    adapters = _group_packed_adapters(adapters, shapes)
    result = model.clone()
    # Keep the low-rank branches separate, as in the original BF16 PEFT run.
    # Merging B@A into BF16 weights introduces a different rounding at every
    # layer. Object patches are activated/restored by Comfy's ModelPatcher.
    from comfy.patcher_extension import CallbacksMP

    adapter_size = 0
    for name, values in adapters.items():
        if name in result.object_patches or name + ".forward" in result.object_patches:
            raise ValueError("Distilled cannot replace an existing module patch: " + name)
        module = result.get_model_object(name)
        result.add_object_patch(name, _wrap_adapter(module, values, scale, before_bias=lite_compact_runtime))
        pairs = values if len(values) == 3 else (values,)
        adapter_size += sum(a.nbytes + b.nbytes for a, b in pairs)
    result.add_callback_with_key(
        CallbacksMP.ON_CLONE,
        "vimoo_distilled_lora",
        partial(_rebind_lora_modules, adapters=adapters, scale=scale, before_bias=lite_compact_runtime),
    )
    # The adapter modules are installed as reversible object patches, so they
    # are not visible to module_size() until model load.  Include them in the
    # patcher's accounting up front.
    result.size = model.model_size() + adapter_size
    result.model_options["vimoo_distilled"] = recipe
    if recipe.get("source_image_decode") is not None:
        result.model_options["vimoo_source_image_decode"] = validate_source_image_decode(
            recipe["source_image_decode"]
        )
    if recipe.get("text_encoder_precision_profile") is not None:
        if not lite_compact_runtime:
            raise ValueError("Native Patchify text conditioning must bind its distilled profile")
        result.model_options["vimoo_text_encoder_precision_profile"] = recipe[
            "text_encoder_precision_profile"
        ]
    if recipe.get("image_preprocessing") is not None:
        if not lite_compact_runtime:
            raise ValueError("Native Patchify image conditioning must bind its distilled profile")
        result.model_options["vimoo_image_preprocessing"] = recipe["image_preprocessing"]
    if recipe.get("vae_numerical_profile") is not None:
        result.model_options["vimoo_vae_numerical_profile"] = validate_vae_numerical_profile(
            recipe["vae_numerical_profile"]
        )
    if lite_compact_runtime:
        options = result.model_options.setdefault("transformer_options", {})
        options["vimoo_symmetric_self_attention"] = True
        options["vimoo_flash_padded_capacity"] = True
    from .vimoo_performance import reapply_release_performance

    return reapply_release_performance(result)


def _group_packed_adapters(adapters, shapes):
    result = dict(adapters)
    for key in shapes:
        if not re.fullmatch(r"diffusion_model\.double_blocks\.\d+\.(img|txt)_attn\.qkv\.weight", key):
            continue
        prefix = key.removesuffix("qkv.weight")
        names = [prefix + "to_" + part for part in "qkv"]
        present = [name in result for name in names]
        if any(present):
            if not all(present):
                raise ValueError("Incomplete packed QKV adapter triplet")
            result[prefix + "qkv"] = tuple(result.pop(name) for name in names)
    return result


def _wrap_adapter(base, values, scale=0.5, *, before_bias=False):
    return (
        PackedLoRALinear(base, values, scale=scale)
        if len(values) == 3
        else LoRALinear(base, *values, scale=scale, before_bias=before_bias)
    )


def _adapter_parameter(base, value):
    # Replacing object patches on a resident clone can skip Comfy's weight
    # loading pass. Match the current base device without rounding FP32 masters.
    # Meta and lazy-casting weights have no resident storage to follow.
    device = base.weight.device
    if isinstance(device, torch.device) and device.type != "meta":
        value = value.to(device=device)
    return torch.nn.Parameter(value, requires_grad=False)


class PackedLoRALinear(torch.nn.Module):
    """One packed base projection, with unmerged Q/K/V LoRA branches.

    The base may be a native ComfyUI quantized Linear. Keeping its forward
    intact preserves activation quantization and CUDA INT8 GEMM dispatch.
    """

    def __init__(self, base, pairs, scale=0.5):
        super().__init__()
        if len(pairs) != 3:
            raise ValueError("Packed QKV requires three LoRA pairs")
        self.base = base
        self.scale = scale
        self.down = torch.nn.ModuleList()
        self.up = torch.nn.ModuleList()
        for a, b in pairs:
            down = type(base)(a.shape[1], a.shape[0], bias=False, device="meta", dtype=a.dtype)
            up = type(base)(b.shape[1], b.shape[0], bias=False, device="meta", dtype=b.dtype)
            down.weight = _adapter_parameter(base, a)
            up.weight = _adapter_parameter(base, b)
            self.down.append(down)
            self.up.append(up)

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x, *args, **kwargs):
        result = self.base(x, *args, **kwargs)
        return torch.cat(
            tuple(
                (value + up(down(x.to(down.weight.dtype))) * self.scale).to(value.dtype)
                for value, down, up in zip(result.chunk(3, -1), self.down, self.up)
            ),
            -1,
        )


class LoRALinear(torch.nn.Module):
    """Keep the base and scaled B(A(x)) separate, with Comfy-managed weights."""

    def __init__(self, base, a, b, scale=0.5, *, before_bias=False):
        super().__init__()
        self.base = base
        self.scale = scale
        self.before_bias = before_bias and getattr(base, "vimoo_separate_bias", False)
        self.down = type(base)(a.shape[1], a.shape[0], bias=False, device="meta", dtype=a.dtype)
        self.up = type(base)(b.shape[1], b.shape[0], bias=False, device="meta", dtype=b.dtype)
        self.down.weight = _adapter_parameter(base, a)
        self.up.weight = _adapter_parameter(base, b)

    def forward(self, x, *args, **kwargs):
        if self.before_bias:
            return self.base(x, *args, vimoo_lora_delta=self.vimoo_lora_delta, **kwargs)
        result = self.base(x, *args, **kwargs)
        # Autocast controls adapter GEMMs without rounding their stored
        # masters. Restore the base dtype after the addition so an FP32
        # adapter cannot promote the next BF16 block stream.
        # Both projections return owned buffers. Reuse them for scaling/addition
        # instead of allocating another full MLP-width activation.
        return result.add_(self.vimoo_lora_delta(x))

    @property
    def weight(self):
        """Expose the base format to Comfy's fused activation dispatch helper."""
        return self.base.weight

    def vimoo_lora_delta(self, x):
        return self.up(self.down(x.to(self.down.weight.dtype))).mul_(self.scale)


def _rebind_lora_modules(source, clone, adapters, scale=0.5, *, before_bias=False):
    if source.model is clone.model:
        return
    for path, values in adapters.items():
        # A later object patch owns its slot; cloning must preserve that choice.
        if isinstance(clone.object_patches.get(path), (LoRALinear, PackedLoRALinear)):
            base = clone.object_patches_backup.get(path)
            if base is None:
                base = clone.model.get_submodule(path)
            clone.add_object_patch(path, _wrap_adapter(base, values, scale, before_bias=before_bias))
