"""Regional tensor graphs with native ComfyUI weight lifetimes outside Dynamo."""

from types import FunctionType, MethodType
from functools import lru_cache

import torch
import comfy.ops
from comfy.patcher_extension import PatcherInjection


@torch.compiler.disable
def acquire_weights(*args, **kwargs):
    context = comfy.ops.CastBiasWeightContext(*args, **kwargs)
    weight, bias = context.__enter__()
    return weight, bias, context


@torch.compiler.disable
def release_weights(context):
    context.__exit__(None, None, None)


def _cast(module, value):
    owner = module if isinstance(module, torch.nn.Linear) or module.weight is not None else None
    weight, bias, context = acquire_weights(owner, value, offloadable=True)
    try:
        if isinstance(module, torch.nn.Linear):
            return torch.nn.functional.linear(value, weight, bias)
        if isinstance(module, torch.nn.RMSNorm):
            return torch.nn.functional.rms_norm(value, module.normalized_shape, weight, module.eps)
        return torch.nn.functional.layer_norm(value, module.normalized_shape, weight, bias, module.eps)
    finally:
        release_weights(context)


@lru_cache(maxsize=128)
def _shape_function(function, signature):
    # Dynamo caches graph-break continuations by code object. Different native
    # operator shapes must not consume one shared continuation's variant limit.
    name = f"{function.__name__}_{signature}"
    specialized = FunctionType(
        function.__code__.replace(co_name=name),
        function.__globals__,
        name,
        function.__defaults__,
        function.__closure__,
    )
    specialized.__kwdefaults__ = function.__kwdefaults__
    return specialized


def specialize_cast(function, module):
    shape = (
        (module.in_features, module.out_features)
        if isinstance(module, torch.nn.Linear)
        else module.normalized_shape
    )
    signature = (type(module).__name__, tuple(shape), getattr(module, "vimoo_upcast_bias", False))
    return _shape_function(function, signature)


_CASTING_KEY = "vimoo.regional_casting"


def _eject_casting(patcher):
    for module, name, original, installed in reversed(patcher.model.__dict__.pop(_CASTING_KEY, [])):
        if module.__dict__.get(name) is installed:
            if original is None:
                delattr(module, name)
            else:
                setattr(module, name, original)


def _inject_casting(patcher):
    # Native object patches are installed before injection and restored after
    # ejection. Keep module references, never adapter-internal patch paths.
    originals = []
    patcher.model.__dict__[_CASTING_KEY] = originals
    native_functions = {
        operation.forward_comfy_cast_weights
        for operation in (
            comfy.ops.disable_weight_init.Linear,
            comfy.ops.disable_weight_init.RMSNorm,
            comfy.ops.disable_weight_init.LayerNorm,
        )
    }
    try:
        for module in patcher.model.diffusion_model.double_blocks.modules():
            if not hasattr(module, "forward_comfy_cast_weights"):
                continue
            function = module.forward_comfy_cast_weights.__func__
            replacement = (
                specialize_cast(_cast, module)
                if function in native_functions
                else torch.compiler.disable(function)
            )
            method = MethodType(replacement, module)
            originals.append(
                (
                    module,
                    "forward_comfy_cast_weights",
                    module.__dict__.get("forward_comfy_cast_weights"),
                    method,
                )
            )
            module.forward_comfy_cast_weights = method
            # The native dispatch entry also resumes after run_every_op().
            # Specializing only its cast callee leaves all shapes and dtypes
            # competing for Linear.forward's shared continuation cache.
            forward = MethodType(specialize_cast(module.forward.__func__, module), module)
            originals.append((module, "forward", module.__dict__.get("forward"), forward))
            module.forward = forward
    except Exception:
        _eject_casting(patcher)
        raise


def configure_regional_casting(patcher, enabled):
    """Scope casting methods to native model load/unload, including adapters."""
    if enabled:
        patcher.set_injections(_CASTING_KEY, [PatcherInjection(_inject_casting, _eject_casting)])
    else:
        patcher.remove_injections(_CASTING_KEY)
