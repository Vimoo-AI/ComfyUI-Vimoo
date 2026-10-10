"""Release attention and compilation switches with internal numerical diagnostics."""

from __future__ import annotations

from functools import partial
import logging

import torch

import comfy.ldm.modules.attention
from comfy.patcher_extension import WrappersMP
from comfy_api.torch_helpers import set_torch_compile_wrapper

from . import vimoo_attention
from .vimoo_performance_config import ATTENTION_BACKENDS, PerformanceConfig


def _memory_usage_factor_wrapper(executor, model, *args, factor, **kwargs):
    """Apply the scoped factor before ComfyUI estimates sampling memory, then restore it."""
    original = model.model.memory_usage_factor
    model.model.memory_usage_factor = factor
    try:
        return executor(model, *args, **kwargs)
    finally:
        model.model.memory_usage_factor = original


def _validate(model, opts):
    """Reject impossible settings before cloning, and return the FA2-retained block set."""
    opts.validate()
    vimoo_attention._validate_sage_pv(opts.sage_pv_dtype, opts.sage_pv_accum_dtype)
    for name in ("sage_fa2_first_blocks", "sage_fa2_last_blocks"):
        count = getattr(opts, name)
        if type(count) is not int or count < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")
    if not (opts.sage_fa2_first_blocks or opts.sage_fa2_last_blocks):
        return frozenset()
    if (opts.attention, opts.sage_scope, opts.sage_fallback) != ("sage", "joint", "fa2"):
        raise ValueError(
            "FA2 block retention requires attention=sage, sage_scope=joint and sage_fallback=fa2."
        )
    depth = len(model.get_model_object("diffusion_model").double_blocks)
    if opts.sage_fa2_first_blocks > depth or opts.sage_fa2_last_blocks > depth:
        raise ValueError(f"sage_fa2 block counts cannot exceed the model's {depth} DiT blocks.")
    options = model.model_options["transformer_options"]
    if options.get("patches") or options.get("patches_replace"):
        raise ValueError("FA2 block retention does not support attention patches or block replacements.")
    return frozenset(range(opts.sage_fa2_first_blocks)) | frozenset(
        range(depth - opts.sage_fa2_last_blocks, depth)
    )


def _apply_memory_options(patched, opts):
    """Residency and layout: the activation estimate, block prefetch and packed QKV."""
    factor = float(opts.memory_usage_factor)
    patched.add_object_patch("memory_usage_factor", factor)
    patched.remove_wrappers_with_key(WrappersMP.PREPARE_SAMPLING, "vimoo_memory_usage_factor")
    patched.add_wrapper_with_key(
        WrappersMP.PREPARE_SAMPLING,
        "vimoo_memory_usage_factor",
        partial(_memory_usage_factor_wrapper, factor=factor),
    )
    options = patched.model_options["transformer_options"]
    # BaseModel owns prefetch_dynamic_vbars; keep the user choice separate.
    options["vimoo_block_prefetch"] = opts.block_prefetch
    options.pop("vimoo_allocation_graph", None)
    options.pop("vimoo_fused_ops", None)
    if opts.fused_qkv:
        options["vimoo_fused_qkv"] = True
    else:
        options.pop("vimoo_fused_qkv", None)


def _configured_sage_attention(model, opts):
    """Configure SageAttention 2 precision, image/text routing and exact fallback."""
    vimoo_attention._load_sage_kernel(model.load_device, pv_dtype=opts.sage_pv_dtype)
    if opts.sage_fallback in ("fa2", "fa4"):
        vimoo_attention._load_flash_kernels(opts.sage_fallback, model.load_device)
    return partial(
        vimoo_attention.attention_sage2,
        image_only=bool(opts.sage_image_only),
        exact_kernel=opts.sage_exact_kernel,
        scope=opts.sage_scope,
        fallback_backend=opts.sage_fallback,
        qk_quant_gran=opts.sage_qk_quant_gran,
        pv_dtype=opts.sage_pv_dtype,
        pv_accum_dtype=opts.sage_pv_accum_dtype,
        smooth_k=bool(opts.sage_smooth_k),
        smooth_v=vimoo_attention._sage_smooth_v(
            opts.sage_pv_dtype, opts.sage_pv_accum_dtype, opts.sage_smooth_v
        ),
    )


def _apply_attention(patched, model, opts, retained_blocks):
    """Select the attention backend and publish what its quadrant split needs."""
    options = patched.model_options["transformer_options"]
    patched.model_options["vimoo_attention_backend"] = opts.attention
    options.pop("vimoo_sage_fa2_blocks", None)
    options.pop("vimoo_sage_fa2_override", None)
    if retained_blocks:
        options["vimoo_sage_fa2_blocks"] = retained_blocks
        options["vimoo_sage_fa2_override"] = vimoo_attention._attention_fa2_override
    options["vimoo_drop_text_padding"] = opts.text_padding == "drop"
    if opts.attention == "sage" and opts.sage_image_only:
        # Tells the DiT to publish the conditioning length the split needs.
        options["vimoo_sage_image_only"] = True
    else:
        options.pop("vimoo_sage_image_only", None)
    if opts.attention == "sage":
        patched.set_model_optimized_attention(_configured_sage_attention(model, opts))
    elif opts.attention != "pytorch":
        vimoo_attention._load_flash_kernels(opts.attention, model.load_device)
        patched.set_model_optimized_attention(
            partial(vimoo_attention.attention_flash, backend=opts.attention)
        )
    else:
        patched.set_model_optimized_attention(comfy.ldm.modules.attention.get_attention_function("pytorch"))


def _apply_math_and_compile(patched, opts):
    """Choose precise or device math, then compile whichever callables it exposes."""
    regional = opts.compile == "regional"
    from .vimoo_compile import configure_regional_casting

    configure_regional_casting(patched, regional)
    patched.model_options["vimoo_compile_scope"] = opts.compile
    # Helper compilation enables precise arithmetic; the independent
    # precise_math option also supports eager execution.
    compile_math = opts.compile == "math_helpers"
    patched.add_object_patch("diffusion_model.precise_math", compile_math or bool(opts.precise_math))
    patched.add_object_patch(
        "diffusion_model.precise_inputs", regional or compile_math or bool(opts.precise_math)
    )
    options = patched.model_options["transformer_options"]
    if regional:
        dit = patched.get_model_object("diffusion_model")
        options["vimoo_math"] = (dit.qk_norm_math, dit.rope_math, dit.modulate_math)
        options["vimoo_math_graph"] = True
    else:
        options.pop("vimoo_math", None)
        options.pop("vimoo_math_graph", None)

    if not (regional or compile_math):
        patched.remove_wrappers_with_key(WrappersMP.APPLY_MODEL, "torch.compile")
        patched.model_options.pop("torch_compile_kwargs", None)
        return
    if compile_math:
        keys = [f"diffusion_model.{name}_math" for name in ("qk_norm", "rope", "modulate")]
    else:
        dit = patched.get_model_object("diffusion_model")
        keys = [f"diffusion_model.double_blocks.{index}" for index in range(len(dit.double_blocks))]
    if not keys:
        raise ValueError("The loaded model has no Vimoo repeated blocks to compile.")
    compile_settings = {"mode": None if opts.compile_mode == "default" else opts.compile_mode}
    if patched.load_device.type == "cuda" and torch.cuda.memory.get_allocator_backend() == "cudaMallocAsync":
        # CUDA graph trees query allocator pool state unsupported by cudaMallocAsync.
        # Keep this model's selected Inductor optimizations without global config changes.
        compiler_options = torch._inductor.list_mode_options(opts.compile_mode).copy()
        compiler_options["triton.cudagraphs"] = False
        compile_settings = {"mode": None, "options": compiler_options}
        logging.info(
            "Vimoo %s compilation uses Inductor without CUDA graphs with cudaMallocAsync.", opts.compile_mode
        )
    set_torch_compile_wrapper(
        model=patched,
        backend="inductor",
        **compile_settings,
        fullgraph=compile_math,
        dynamic=compile_math if opts.compile_dynamic == "auto" else opts.compile_dynamic == "dynamic",
        keys=keys,
    )
    logging.info("Enabled Vimoo %s compilation for %d callables.", opts.compile, len(keys))


class VimooPerformanceOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "attention": (list(ATTENTION_BACKENDS), {"default": PerformanceConfig().attention}),
                "compile": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Vimoo"
    DESCRIPTION = "Select video attention and enable compilation using the current release settings."

    def patch(self, model, attention, compile):
        return (apply_release_performance(model, attention, compile),)


def apply_performance(model, config: PerformanceConfig):
    """Apply validated model settings without depending on a node interface."""
    retained_blocks = _validate(model, config)
    patched = model.clone(disable_dynamic=False)
    _apply_memory_options(patched, config)
    _apply_attention(patched, model, config, retained_blocks)
    _apply_math_and_compile(patched, config)
    return patched


def release_performance_config(model, attention, compile):
    """Resolve public controls and the model's current release profile once."""
    from .vimoo_release import RELEASE_RECIPES, release_profile

    if type(compile) is not bool:
        raise ValueError("compile must be a Boolean")
    settings = dict(RELEASE_RECIPES[release_profile(model)]["performance"])
    settings.update(attention=attention, compile="regional" if compile else "disabled")
    return PerformanceConfig(**settings)


def apply_release_performance(model, attention, compile):
    """Shared entry for node execution and adapter profile changes."""
    config = release_performance_config(model, attention, compile)
    patched = apply_performance(model, config)
    patched.model_options["vimoo_release_performance"] = {"attention": attention, "compile": compile}
    return patched


NODE_CLASS_MAPPINGS = {"VimooPerformanceOptions": VimooPerformanceOptions}
NODE_DISPLAY_NAME_MAPPINGS = {"VimooPerformanceOptions": "Vimoo Performance Options"}


def reapply_release_performance(model):
    """Bind adapter-specific defaults while retaining the chosen performance controls."""
    controls = model.model_options.get("vimoo_release_performance")
    if controls is None:
        return model
    return apply_release_performance(model, **controls)
