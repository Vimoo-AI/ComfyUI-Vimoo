"""Explicit attention choices registered with ComfyUI's backend API."""

import importlib
import importlib.metadata
import math
import logging
import threading
import weakref

_sage_fallback_routes = set()
_flash_metadata_cache = {}

import torch
import comfy.ldm.modules.attention
from torch.nn.attention import SDPBackend, sdpa_kernel
from comfy.ldm.modules.attention import attention_pytorch, register_attention_function, wrap_attn


@wrap_attn
def qwen_attention(
    q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs
):
    # This backend is selected explicitly by the Vimoo text loader.
    causal = kwargs.pop("is_causal", False)
    if not causal:
        return attention_pytorch(
            q,
            k,
            v,
            heads,
            mask=mask,
            attn_precision=attn_precision,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            **kwargs,
        )
    return _ordered_sdpa(
        q,
        k,
        v,
        heads,
        mask,
        skip_reshape,
        skip_output_reshape,
        [
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ],
        set_priority=True,
        is_causal=True,
        **kwargs,
    )


def _ordered_sdpa(
    q,
    k,
    v,
    heads,
    mask,
    skip_reshape,
    skip_output_reshape,
    backend,
    scale=None,
    set_priority=False,
    is_causal=False,
    **kwargs,
):
    if not skip_reshape:
        dim = q.shape[-1] // heads
        q, k, v = [x.view(x.shape[0], x.shape[1], -1, dim).transpose(1, 2) for x in (q, k, v)]
    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
    with sdpa_kernel(backend, set_priority=set_priority):
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            scale=scale,
            is_causal=is_causal,
            enable_gqa=kwargs.get("enable_gqa", False),
        )
    return out if skip_output_reshape else out.transpose(1, 2).reshape(q.shape[0], -1, heads * q.shape[-1])


@wrap_attn
def byt5_attention(
    q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs
):
    dim = q.shape[-1] if skip_reshape else q.shape[-1] // heads
    scale = kwargs.pop("scale", 1.0)
    return _ordered_sdpa(
        q,
        k / math.sqrt(dim),
        v,
        heads,
        mask,
        skip_reshape,
        skip_output_reshape,
        SDPBackend.MATH,
        scale=scale,
        **kwargs,
    )


@wrap_attn
def byt5_cpu_attention(
    q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs
):
    """The Lite TI2V FP32 CPU encoder uses ordinary SDPA dispatch."""
    dim = q.shape[-1] if skip_reshape else q.shape[-1] // heads
    scale = kwargs.pop("scale", 1.0)
    return _ordered_sdpa(
        q,
        k / math.sqrt(dim),
        v,
        heads,
        mask,
        skip_reshape,
        skip_output_reshape,
        [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH],
        scale=scale,
        **kwargs,
    )


@wrap_attn
def siglip_attention(
    q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs
):
    return _ordered_sdpa(
        q, k, v, heads, mask, skip_reshape, skip_output_reshape, SDPBackend.CUDNN_ATTENTION, **kwargs
    )


register_attention_function("vimoo_qwen", qwen_attention)
register_attention_function("vimoo_byt5", byt5_attention)
register_attention_function("vimoo_byt5_cpu", byt5_cpu_attention)
register_attention_function("vimoo_siglip", siglip_attention)


# --- SageAttention 2 exact image/text attention merge ---------------------------------------------------------------


def _flash_attention_advice():
    """Say which FlashAttention this GPU needs, and spot the import that hides it.

    flash-attn-4 installs a namespace package that holds only ``flash_attn/cute``.  A regular
    ``flash_attn`` package from FlashAttention 2 wins the import, so ``flash_attn.cute``
    disappears even though FA4 is installed.
    """
    capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA device"
    if capability is None:
        which = "FlashAttention 4 (requirements-fa4.txt) on an NVIDIA GPU"
    elif capability >= (9, 0):
        which = f"FlashAttention 4 (requirements-fa4.txt); {name} is sm_{capability[0]}{capability[1]}"
    else:
        which = (
            f"FlashAttention 2 (requirements-flash-attn.txt); {name} is sm_{capability[0]}{capability[1]}, "
            "below the sm_90 that FlashAttention 4's cute kernels need"
        )
    installed = shadowed = ""
    try:
        from importlib.metadata import version

        installed = version("flash-attn-4")
    except Exception:
        pass
    if installed:
        try:
            import flash_attn

            shadowed = (
                f" flash-attn-4 {installed} is installed but 'import flash_attn' resolves to "
                f"{getattr(flash_attn, '__file__', '?')}, a regular package that hides it; "
                "install both into the same site-packages, or remove the FlashAttention 2 copy."
            )
        except Exception:
            shadowed = f" flash-attn-4 {installed} is installed but did not import."
    return (
        f"The exact attention quadrants need {which}.{shadowed} "
        "Select sage_exact_kernel=fp32 to accept this plugin's much slower FP32 "
        "implementation instead."
    )


# CuTe's wrapper specializes on tensor lengths and autocast; keep the entire
# exact branch outside the regional graph, not only its inner kernel call.
@torch.compiler.disable
def exact_partial_attention(q, k, v, scale, chunk=8192, kernel="fa4"):
    """Attention of all queries against one key range, with its log-sum-exp.

    Returns the output [B, H, Lq, D] and the natural-log log-sum-exp [B, H, Lq], which is
    what lets two key ranges be merged (SageAttention returns the same quantity).  Torch
    SDPA cannot serve here: it does not return the log-sum-exp.

    ``kernel="fa4"`` is the production path: FlashAttention 4's cute interface, fused and
    BF16-exact.  ``kernel="fp32"`` computes softmax explicitly in FP32 with PyTorch, kept for
    tests, for CPU, and for checking the fused kernel; it materialises the score block and
    is far slower, so it never runs unless it is asked for by name.
    """
    if kernel == "fa4":
        try:
            from flash_attn.cute.interface import flash_attn_func
        except ImportError as error:
            raise RuntimeError(_flash_attention_advice()) from error
        out, lse = flash_attn_func(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            softmax_scale=scale,
            causal=False,
            return_lse=True,
        )
        return out.transpose(1, 2), lse
    if kernel != "fp32":
        raise ValueError(f"Unknown exact attention kernel: {kernel!r}")
    kf, vf = k.float(), v.float()
    outs, lses = [], []
    for start in range(0, q.shape[2], chunk):
        scores = torch.matmul(q[:, :, start : start + chunk].float(), kf.transpose(-1, -2)) * scale
        peak = scores.amax(dim=-1, keepdim=True)
        probs = torch.exp(scores - peak)
        total = probs.sum(dim=-1, keepdim=True)
        outs.append(torch.matmul(probs, vf) / total)
        lses.append((peak + torch.log(total)).squeeze(-1))
    return torch.cat(outs, dim=2), torch.cat(lses, dim=2)


def merge_lse(out_a, lse_a, out_b, lse_b):
    """Combine two normalised attention outputs over disjoint key sets using their log-sum-exps.

    The two weights sum to one, so a single ``lerp`` in the inputs' own dtype replaces
    two multiplies and an add in fp32.
    """
    lse = torch.logaddexp(lse_a.float(), lse_b.float())
    weight_b = torch.exp(lse_b.float() - lse).unsqueeze(-1).to(out_a.dtype)
    return torch.lerp(out_a, out_b.to(out_a.dtype), weight_b)


def image_only_attention(q, k, v, img_len, scale, kernel, exact_kernel="fa4"):
    """Run only image-to-image attention on ``kernel``; keep the three text quadrants exact.

    ``q``/``k``/``v`` are [B, H, L, D] with the image tokens first.  ``kernel(q, k, v)``
    must return the attention output and its natural-log log-sum-exp so the image rows can
    be merged with their exact half.

    This is what keeps rendered text legible and object identity stable: the conditioning
    stream never passes through a quantized kernel.
    """
    out_img, lse_img = kernel(q[:, :, :img_len], k[:, :, :img_len], v[:, :, :img_len])
    out_txt, lse_txt = exact_partial_attention(
        q[:, :, :img_len], k[:, :, img_len:], v[:, :, img_len:], scale, kernel=exact_kernel
    )
    image_rows = merge_lse(out_img, lse_img, out_txt, lse_txt)
    text_rows = exact_partial_attention(q[:, :, img_len:], k, v, scale, kernel=exact_kernel)[0]
    return torch.cat((image_rows, text_rows), dim=2).to(q.dtype)


_FLASH_HUB = {
    "fa2": ("kernels-community/flash-attn2", "c3f3c93a4a39e3d2cf49c13eac49229bc726ea13"),
    "fa3": ("kernels-community/flash-attn3", "acb0f42e6901d3e9effa8a83bd32bc2bf0cfdfc2"),
}

_FLASH_KERNELS = {}

_FLASH_LOAD_LOCK = threading.Lock()

_SAGE_KERNELS = {}

_SAGE_LOAD_LOCK = threading.Lock()

_SAGE_PV_ACCUM_DTYPES = {
    "fp16": ("fp16", "fp16+fp32", "fp32"),
    "fp8": ("fp32", "fp32+fp32", "fp32+fp16"),
}


def _attention_fa2_override(_original_function, *args, **kwargs):
    """Match Comfy's override ABI, which prepends the original attention."""
    return attention_flash(*args, backend="fa2", **kwargs)


def _load_flash_kernels(backend, device=None):
    if backend not in ("fa2", "fa3", "fa4"):
        raise ValueError(f"Unknown FlashAttention backend: {backend!r}")
    if (device is not None and torch.device(device).type != "cuda") or not torch.cuda.is_available():
        raise RuntimeError(f"Vimoo {backend.upper()} requires an NVIDIA CUDA GPU.")
    major = torch.cuda.get_device_capability(device)[0]
    if backend == "fa3" and major != 9:
        raise RuntimeError("Vimoo FA3 requires an NVIDIA Hopper GPU (compute capability 9.x).")
    if backend in ("fa2", "fa4") and major not in (8, 9, 10, 11, 12):
        raise RuntimeError(f"Vimoo {backend.upper()} requires compute capability 8.x through 12.x.")
    # Check the selected GPU even when another device already loaded the kernel.
    with _FLASH_LOAD_LOCK:
        native_kernel = False
        if backend == "fa2":
            if ("fa2", "native") in _FLASH_KERNELS:
                native_kernel = True
            elif "fa2" not in _FLASH_KERNELS:
                try:
                    importlib.metadata.version("flash-attn")
                except importlib.metadata.PackageNotFoundError:
                    pass
                else:
                    # Share one FA2 binary across ordinary and distilled calls.
                    native_kernel = True
        cache_key = (backend, "native") if native_kernel and backend == "fa2" else backend
        if cache_key not in _FLASH_KERNELS:
            try:
                if native_kernel and backend == "fa2":
                    # Reuse an installed native FA2 for every model profile.
                    module = importlib.import_module("flash_attn")
                    if not all(
                        callable(getattr(module, name, None))
                        for name in ("flash_attn_varlen_func", "flash_attn_func")
                    ):
                        raise ImportError("The flash_attn namespace does not provide the native FA2 API")
                elif backend == "fa4":
                    try:
                        module = importlib.import_module("flash_attn.cute")
                    except ModuleNotFoundError as error:
                        if error.name != "flash_attn.cute":
                            raise
                        # flash-attn-4 intentionally installs a namespace-only
                        # subpackage. A co-installed regular flash_attn package
                        # otherwise hides it from Python's package search path.
                        package = importlib.import_module("flash_attn")
                        root = str(importlib.metadata.distribution("flash-attn-4").locate_file("flash_attn"))
                        if root not in package.__path__:
                            package.__path__.append(root)
                        module = importlib.import_module("flash_attn.cute")
                else:
                    import kernels

                    repo, revision = _FLASH_HUB[backend]
                    module = kernels.get_kernel(repo, revision=revision)
            except ImportError as error:
                requirement = (
                    "requirements-native-fa2.txt"
                    if native_kernel and backend == "fa2"
                    else "requirements-fa4.txt"
                    if backend == "fa4"
                    else "requirements-flash-attn.txt"
                )
                raise RuntimeError(
                    f"Vimoo {backend.upper()} could not load its dependencies. Install {requirement} "
                    "in the ComfyUI environment and check the chained dependency error."
                ) from error
            _FLASH_KERNELS[cache_key] = (module.flash_attn_varlen_func, module.flash_attn_func)
            source = (
                "native flash-attn"
                if native_kernel
                else "flash_attn.cute"
                if backend == "fa4"
                else f"Hub {_FLASH_HUB[backend][0]}@{_FLASH_HUB[backend][1]}"
            )
            logging.info("Vimoo %s kernel loaded: source=%s", backend.upper(), source)
        return _FLASH_KERNELS[cache_key]


def _validate_sage_pv(pv_dtype, pv_accum_dtype="fp32"):
    if pv_dtype not in _SAGE_PV_ACCUM_DTYPES:
        raise ValueError(f"Unknown SageAttention PV dtype: {pv_dtype!r}")
    if pv_accum_dtype not in _SAGE_PV_ACCUM_DTYPES[pv_dtype]:
        raise ValueError(
            f"SageAttention PV {pv_dtype} does not support accumulation {pv_accum_dtype!r}; "
            f"choose one of {_SAGE_PV_ACCUM_DTYPES[pv_dtype]}."
        )


def _sage_smooth_v(pv_dtype, pv_accum_dtype, requested):
    return bool(requested and (pv_dtype, pv_accum_dtype) in (("fp16", "fp16"), ("fp8", "fp32")))


def _is_hopper(device=None):
    try:
        return torch.cuda.get_device_capability(device) == (9, 0)
    except (RuntimeError, ValueError, AssertionError):
        return False


def _sage_fp8_with_sm90(core, sm89_entry):
    """Prefer SageAttention's Hopper kernel for FP8-PV with FP32 accumulation.

    The SM90 entry point only implements ``fp32+fp32`` accumulation and has no V smoothing, so
    both ``fp32`` and ``fp32+fp32`` map to it; every other request keeps the SM89 entry point,
    which on Hopper is several times slower than the SM90 kernel.
    """
    sm90_entry = core.sageattn_qk_int8_pv_fp8_cuda_sm90

    def entry(*args, pv_accum_dtype="fp32", smooth_v=False, **kwargs):
        if pv_accum_dtype in ("fp32", "fp32+fp32") and not smooth_v:
            return sm90_entry(*args, pv_accum_dtype="fp32+fp32", **kwargs)
        return sm89_entry(*args, pv_accum_dtype=pv_accum_dtype, smooth_v=smooth_v, **kwargs)

    return entry


def _load_sage_kernel(device=None, pv_dtype="fp16"):
    """Load the explicitly selected INT8-QK / FP16- or FP8-PV CUDA entry point."""
    _validate_sage_pv(pv_dtype)
    if (device is not None and torch.device(device).type != "cuda") or not torch.cuda.is_available():
        raise RuntimeError("Vimoo SageAttention requires an NVIDIA CUDA GPU.")
    # Only Hopper has a dedicated FP8 kernel in SageAttention 2.2.0; Ampere, Ada and Blackwell share the SM89 entry point.
    hopper = pv_dtype == "fp8" and _is_hopper(device)
    cache_key = (pv_dtype, hopper)
    with _SAGE_LOAD_LOCK:
        if cache_key not in _SAGE_KERNELS:
            try:
                core = importlib.import_module("sageattention.core")
                if pv_dtype == "fp16":
                    if not core.SM80_ENABLED:
                        raise ImportError("sageattention._qattn_sm80 is not installed")
                    kernel = core.sageattn_qk_int8_pv_fp16_cuda
                else:
                    if not core.SM89_ENABLED:
                        raise ImportError("sageattention._qattn_sm89 is not installed")
                    kernel = core.sageattn_qk_int8_pv_fp8_cuda
                    if (
                        hopper
                        and getattr(core, "SM90_ENABLED", False)
                        and hasattr(core, "sageattn_qk_int8_pv_fp8_cuda_sm90")
                    ):
                        kernel = _sage_fp8_with_sm90(core, kernel)
            except (ImportError, AttributeError) as error:
                raise RuntimeError(
                    f"Vimoo SageAttention could not load sageattention 2.2.0 PV {pv_dtype}. "
                    "Install a build containing the CUDA QAttn extensions for the selected GPU."
                ) from error
            _SAGE_KERNELS[cache_key] = kernel
        return _SAGE_KERNELS[cache_key]


def _key_padding_mask(mask: torch.Tensor, batch: int, key_length: int):
    """Convert ComfyUI's additive key-padding mask to a boolean [B, K] mask."""
    if mask.ndim == 2 and mask.shape[0] != 1:
        return None  # ComfyUI interprets 2-D masks as [query, key], not [batch, key].
    while mask.ndim > 2 and mask.shape[-2] == 1:
        mask = mask.squeeze(-2)
    if mask.ndim != 2 or mask.shape[-1] != key_length:
        return None
    if mask.shape[0] == 1 and batch > 1:
        mask = mask.expand(batch, -1)
    if mask.shape[0] != batch:
        return None
    if mask.dtype == torch.bool:
        return mask if torch.all(mask.any(dim=-1)) else None
    if not torch.is_floating_point(mask):
        return None
    # Varlen can only express binary padding, not finite attention biases.
    padding = torch.isneginf(mask) | (mask == torch.finfo(mask.dtype).min)
    if not torch.all((mask == 0) | padding):
        return None
    valid = ~padding
    return valid if torch.all(valid.any(dim=-1)) else None


def _prepare_flash_metadata(
    valid_keys, batch, query_length, key_length, device, *, symmetric=False, padded_capacity=False
):
    lengths = valid_keys.sum(dim=-1, dtype=torch.int32)
    compiling = torch.compiler.is_compiling()
    if not compiling and torch.any(lengths == 0):
        raise ValueError("FlashAttention cannot attend over an empty key sequence.")
    cu_q = torch.arange(0, (batch + 1) * query_length, query_length, dtype=torch.int32, device=device)
    cu_k = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    cu_k[1:] = torch.cumsum(lengths, dim=0, dtype=torch.int32)
    if symmetric:
        if query_length != key_length:
            raise ValueError("Symmetric FlashAttention packing requires matching self-attention lengths")
        cu_q = cu_k
    if padded_capacity:
        if not symmetric:
            raise ValueError("Packed attention requires symmetric self-attention")
        # Keep B*L allocated rows even when cu_seqlens contains fewer valid tokens.
        # Its trailing gather indices point to one appended zero row; retaining
        # that capacity and original maximum length preserves kernel dispatch.
        indices = valid_keys.flatten().nonzero(as_tuple=False).flatten()
        indices = torch.nn.functional.pad(
            indices, (0, batch * key_length - indices.numel()), value=batch * key_length
        )
        return {
            "cu_q": cu_q,
            "cu_k": cu_k,
            "max_key_length": key_length,
            "prefix_lengths": None,
            "indices": indices,
            "padded_capacity": True,
        }
    if compiling:
        return {
            "cu_q": cu_q,
            "cu_k": cu_k,
            "max_key_length": key_length,
            "prefix_lengths": None,
            "indices": valid_keys.flatten().nonzero(as_tuple=False).flatten(),
        }
    prefix = torch.arange(key_length, device=valid_keys.device)[None] < lengths[:, None]
    prefix_lengths = None
    indices = None
    if torch.equal(valid_keys, prefix):
        prefix_lengths = tuple(int(length.item()) for length in lengths)
    else:
        indices = valid_keys.flatten().nonzero(as_tuple=False).flatten()
    return {
        "cu_q": cu_q,
        "cu_k": cu_k,
        "max_key_length": int(lengths.max().item()),
        "prefix_lengths": prefix_lengths,
        "indices": indices,
    }


def _pack_valid_tensor(value, metadata):
    """Pack one sequence using launch metadata shared by repeated blocks."""
    prefix_lengths = metadata["prefix_lengths"]
    if prefix_lengths is not None:
        if value.shape[0] == 1:
            length = prefix_lengths[0]
            return value[0, :length]
        return torch.cat(tuple(value[index, :length] for index, length in enumerate(prefix_lengths)), dim=0)
    indices = metadata["indices"]
    if metadata.get("padded_capacity"):
        flattened = value.flatten(0, 1)
        flattened = torch.cat((flattened, flattened.new_zeros((1, *flattened.shape[1:]))), dim=0)
        return flattened.index_select(0, indices)
    return value.flatten(0, 1)[indices]


def _pack_valid_kv(k, v, metadata):
    """Pack key padding using launch metadata shared by all repeated blocks."""
    return _pack_valid_tensor(k, metadata), _pack_valid_tensor(v, metadata)


def _unpack_valid_tensor(value, valid_keys, metadata):
    batch, length = valid_keys.shape
    if metadata.get("padded_capacity"):
        # Every unused row targets one extra sentinel, which is discarded.
        # Keep gather/scatter shapes fixed when prompt padding lengths change.
        padded = value.new_zeros((batch * length + 1, *value.shape[1:]))
        return padded.index_copy_(0, metadata["indices"], value)[:-1].unflatten(0, (batch, length))
    indices = valid_keys.flatten().nonzero(as_tuple=False).flatten()
    padded = value.new_zeros((batch * length, *value.shape[1:]))
    return padded.index_copy_(0, indices, value).unflatten(0, (batch, length))


@torch.compiler.disable
def _cached_flash_metadata(mask, valid_keys, batch, query_length, key_length, device, options):
    """Keep mask identities and mutable launch caches outside Dynamo guards."""
    symmetric = bool(options.get("vimoo_symmetric_self_attention"))
    padded_capacity = bool(options.get("vimoo_flash_padded_capacity"))
    cache = options.get("vimoo_attention_metadata")
    key = (
        mask.data_ptr(),
        tuple(mask.shape),
        mask.dtype,
        batch,
        query_length,
        key_length,
        device,
        symmetric,
        padded_capacity,
    )
    if cache is None:
        # Compiled blocks must not receive a dictionary of cached tensors.
        # Dynamo otherwise aliases returned metadata to pointer-keyed entries
        # in the input dictionary and guards keys from previous sampler runs.
        key = (id(mask), *key[1:])
        entry = _flash_metadata_cache.get(key)
        if entry is not None and entry[0]() is mask:
            return entry[1]
    else:
        entry = cache.get(key)
        if entry is not None:
            return entry[1]
    metadata = _prepare_flash_metadata(
        valid_keys,
        batch,
        query_length,
        key_length,
        device,
        symmetric=symmetric,
        padded_capacity=padded_capacity,
    )
    if cache is not None:
        # Retain the mask so allocator pointer reuse cannot select stale offsets.
        cache[key] = (mask, metadata)
    else:

        def release(reference):
            current = _flash_metadata_cache.get(key)
            if current is not None and current[0] is reference:
                _flash_metadata_cache.pop(key, None)

        # Metadata owns its tensors, but never prolongs the mask/model lifetime.
        _flash_metadata_cache[key] = (weakref.ref(mask, release), metadata)
    return metadata


@torch.compiler.disable
def _call_flash_kernel(backend, device, varlen, **kwargs):
    """Keep third-party kernels opaque to regional compilation."""
    kernels = _load_flash_kernels(backend, device)
    if backend == "fa4":
        # CuTe uses None for an unbounded window; FA2/FA3 use -1.
        kwargs["window_size"] = (None, None)
    return kernels[0 if varlen else 1](**kwargs)


def attention_flash(
    q,
    k,
    v,
    heads,
    mask=None,
    attn_precision=None,
    skip_reshape=False,
    skip_output_reshape=False,
    backend="fa3",
    **kwargs,
):
    """ComfyUI attention ABI for FA2/FA3/FA4, with exact padding-mask handling."""
    if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
        return comfy.ldm.modules.attention.attention_pytorch(
            q,
            k,
            v,
            heads,
            mask=mask,
            attn_precision=attn_precision,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            **kwargs,
        )
    if kwargs.get("enable_gqa", False):
        raise ValueError("Vimoo FlashAttention does not support grouped-query attention.")

    if skip_reshape:
        batch, actual_heads, query_length, head_dim = q.shape
        if actual_heads != heads:
            raise ValueError(f"FlashAttention received {actual_heads} heads, expected {heads}.")
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
    else:
        batch, query_length, inner_dim = q.shape
        if inner_dim % heads:
            raise ValueError(f"FlashAttention input width {inner_dim} is not divisible by {heads} heads.")
        head_dim = inner_dim // heads
        q = q.view(batch, query_length, heads, head_dim).contiguous()
        k = k.view(batch, k.shape[1], heads, head_dim).contiguous()
        v = v.view(batch, v.shape[1], heads, head_dim).contiguous()

    key_length = k.shape[1]
    valid_keys = None if mask is None else _key_padding_mask(mask, batch, key_length)
    if mask is not None and valid_keys is None:
        # FlashAttention varlen represents key padding, not arbitrary pairwise masks. The
        # small token-refiner path may supply a square mask, so preserve its
        # semantics with SDPA instead of silently producing a different result.
        return comfy.ldm.modules.attention.attention_pytorch(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            heads,
            mask=mask,
            attn_precision=attn_precision,
            skip_reshape=True,
            skip_output_reshape=skip_output_reshape,
            **kwargs,
        )

    device = q.device
    scale = kwargs.get("scale", None)
    options = kwargs.get("transformer_options", {})
    symmetric = bool(options.get("vimoo_symmetric_self_attention"))
    if mask is None:
        # Mask-free calls do not need varlen packing.
        out = _call_flash_kernel(
            backend,
            device,
            False,
            q=q,
            k=k,
            v=v,
            softmax_scale=None if scale is None else float(scale),
            causal=False,
            window_size=(-1, -1),
        )
        if isinstance(out, tuple):
            out = out[0]
        if skip_output_reshape:
            return out.transpose(1, 2)
        return out.reshape(batch, query_length, heads * head_dim)

    q_packed = q.flatten(0, 1)
    valid_keys = valid_keys.to(device=device)
    metadata = _cached_flash_metadata(mask, valid_keys, batch, query_length, key_length, device, options)
    k_packed, v_packed = _pack_valid_kv(k, v, metadata)
    if symmetric:
        q_packed = _pack_valid_tensor(q, metadata)

    out = _call_flash_kernel(
        backend,
        device,
        True,
        q=q_packed,
        k=k_packed,
        v=v_packed,
        cu_seqlens_q=metadata["cu_q"],
        cu_seqlens_k=metadata["cu_k"],
        max_seqlen_q=(metadata["max_key_length"] if symmetric else query_length),
        max_seqlen_k=(key_length if torch.compiler.is_compiling() else metadata["max_key_length"]),
        softmax_scale=None if scale is None else float(scale),
        causal=False,
        window_size=(-1, -1),
    )
    if isinstance(out, tuple):
        out = out[0]
    if symmetric:
        out = _unpack_valid_tensor(out, valid_keys, metadata)
    else:
        out = out.unflatten(0, (batch, query_length))

    if skip_output_reshape:
        return out.transpose(1, 2)
    return out.reshape(batch, query_length, heads * head_dim)


def _sage_fallback(backend, q, k, v, heads, **kwargs):
    logging.debug(
        "Vimoo SageAttention exact fallback: backend=%s, kind=%s, shape=%s, mask=%s",
        backend,
        kwargs.get("transformer_options", {}).get("vimoo_attention_kind", "joint"),
        tuple(q.shape),
        kwargs.get("mask") is not None,
    )
    if backend in ("fa2", "fa4"):
        return attention_flash(q, k, v, heads, backend=backend, **kwargs)
    return comfy.ldm.modules.attention.attention_pytorch(q, k, v, heads, **kwargs)


@torch.compiler.disable
def _call_sage_kernel(device, pv_dtype="fp16", **kwargs):
    """Keep the third-party quantization and CUDA kernels outside compiled graphs."""
    return _load_sage_kernel(device, pv_dtype=pv_dtype)(**kwargs)


def attention_sage2(
    q,
    k,
    v,
    heads,
    mask=None,
    attn_precision=None,
    skip_reshape=False,
    skip_output_reshape=False,
    scope="joint",
    fallback_backend="fa4",
    qk_quant_gran="per_thread",
    pv_accum_dtype="fp32",
    smooth_k=True,
    smooth_v=False,
    pv_dtype="fp16",
    image_only=True,
    exact_kernel="fa4",
    **kwargs,
):
    """ComfyUI ABI for parameterized SageAttention2 with exact fallback semantics.

    Vimoo's double-stream block uses one joint image/text softmax, so ``joint``
    includes both self- and cross-attention terms.  ``text_refiner`` selects the
    two small text-only self-attention blocks.  Binary key padding is represented
    by slicing K/V per batch; arbitrary pairwise masks use the configured exact
    fallback instead of being silently ignored by SageAttention's ``**kwargs``.

    ``image_only`` (the default) sends only the image-to-image quadrant of the joint
    softmax to the kernel and computes the three quadrants that touch the conditioning
    stream exactly, merging the two halves by log-sum-exp.  That is what keeps rendered
    text legible and object identity stable; set it to False to put the whole joint
    softmax on the kernel, which is faster and measurably less faithful.
    """
    _validate_sage_pv(pv_dtype, pv_accum_dtype)
    options = kwargs.get("transformer_options", {})
    attention_kind = kwargs.get("vimoo_attention_kind", options.get("vimoo_attention_kind", "joint"))
    selected = scope == "all" or scope == attention_kind

    fallback_kwargs = dict(
        mask=mask,
        attn_precision=attn_precision,
        skip_reshape=skip_reshape,
        skip_output_reshape=skip_output_reshape,
        **kwargs,
    )
    if not selected or not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
        return _sage_fallback(fallback_backend, q, k, v, heads, **fallback_kwargs)
    if kwargs.get("enable_gqa", False):
        return _sage_fallback(fallback_backend, q, k, v, heads, **fallback_kwargs)

    # Head-major views first: the image-only split works in that layout, so it never
    # pays for the NHD copy, which is about 0.7 GB per tensor at 720p.
    if skip_reshape:
        batch, actual_heads, query_length, head_dim = q.shape
        if actual_heads != heads:
            return _sage_fallback(fallback_backend, q, k, v, heads, **fallback_kwargs)
        q_hnd, k_hnd, v_hnd = q, k, v
    else:
        batch, query_length, inner_dim = q.shape
        if inner_dim % heads:
            return _sage_fallback(fallback_backend, q, k, v, heads, **fallback_kwargs)
        head_dim = inner_dim // heads
        q_hnd = q.view(batch, query_length, heads, head_dim).transpose(1, 2)
        k_hnd = k.view(batch, k.shape[1], heads, head_dim).transpose(1, 2)
        v_hnd = v.view(batch, v.shape[1], heads, head_dim).transpose(1, 2)

    key_length = k_hnd.shape[2]
    valid_keys = None if mask is None else _key_padding_mask(mask, batch, key_length)
    if mask is not None and valid_keys is None:
        return _sage_fallback(fallback_backend, q, k, v, heads, **fallback_kwargs)

    kernel_kwargs = dict(
        pv_dtype=pv_dtype,
        tensor_layout="NHD",
        is_causal=False,
        qk_quant_gran=qk_quant_gran,
        sm_scale=kwargs.get("scale", None),
        pv_accum_dtype=pv_accum_dtype,
        smooth_k=smooth_k,
        smooth_v=_sage_smooth_v(pv_dtype, pv_accum_dtype, smooth_v),
    )
    if image_only and attention_kind == "joint":
        txt_len = options.get("vimoo_sage_text_length", 0)
        img_len = key_length - txt_len
        if txt_len and img_len > 0 and query_length == key_length:
            scale = kwargs.get("scale") or head_dim**-0.5
            split_kwargs = {
                **kernel_kwargs,
                "tensor_layout": "HND",
                "sm_scale": scale,
                "return_lse": True,
            }

            def kernel(q_img, k_img, v_img):
                return _call_sage_kernel(q_img.device, q=q_img, k=k_img, v=v_img, **split_kwargs)

            rows = []
            for index in range(batch):
                keys = slice(None)
                text_keys = txt_len
                if valid_keys is not None and not bool(torch.all(valid_keys[index])):
                    # Image keys are never padded; drop only the padded text keys, which
                    # keeps [image..., text...] order so the split still holds.
                    keep = valid_keys[index].to(q_hnd.device).clone()
                    keep[:img_len] = True
                    text_keys = int(keep[img_len:].sum())
                    keys = keep
                if text_keys == 0:
                    rows.append(
                        kernel(
                            q_hnd[index : index + 1],
                            k_hnd[index : index + 1, :, :img_len],
                            v_hnd[index : index + 1, :, :img_len],
                        )[0].to(q_hnd.dtype)
                    )
                    continue
                rows.append(
                    image_only_attention(
                        q_hnd[index : index + 1],
                        k_hnd[index : index + 1][:, :, keys],
                        v_hnd[index : index + 1][:, :, keys],
                        img_len,
                        scale,
                        kernel,
                        exact_kernel=exact_kernel,
                    )
                )
            split = torch.cat(rows, dim=0)
            if skip_output_reshape:
                return split
            return split.transpose(1, 2).reshape(batch, query_length, heads * head_dim)
    q_nhd = q_hnd.transpose(1, 2).contiguous()
    k_nhd = k_hnd.transpose(1, 2).contiguous()
    v_nhd = v_hnd.transpose(1, 2).contiguous()
    if valid_keys is None or bool(torch.all(valid_keys)):
        out = _call_sage_kernel(q_nhd.device, q=q_nhd, k=k_nhd, v=v_nhd, **kernel_kwargs)
    else:
        valid_keys = valid_keys.to(device=q_nhd.device)
        outputs = []
        for index in range(batch):
            selection = valid_keys[index]
            outputs.append(
                _call_sage_kernel(
                    q_nhd.device,
                    q=q_nhd[index : index + 1],
                    k=k_nhd[index : index + 1, selection],
                    v=v_nhd[index : index + 1, selection],
                    **kernel_kwargs,
                )
            )
        out = torch.cat(outputs, dim=0)

    if skip_output_reshape:
        return out.transpose(1, 2)
    return out.reshape(batch, query_length, heads * head_dim)
