"""Vimoo text math installed through the CLIP model patcher."""

import math
from types import MethodType
from functools import partial

import torch
import comfy.ops
from comfy.text_encoders import llama, t5
from .vimoo_precision import add_method_patches
from . import vimoo_attention  # noqa: F401 - registers the selected backends
from comfy.ldm.modules.attention import get_attention_function


def _qwen_norm(module, x):
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + module.eps)
    return value.to(x.dtype) * comfy.ops.cast_to_input(module.weight, x)


def _qwen_frequencies(module, position_ids, device, compute_dtype=torch.bfloat16):
    dim = module.config.head_dim
    # The checkpoint rotary buffer is rounded to BF16.
    inv = (
        (1.0 / (module.config.rope_theta ** (torch.arange(0, dim, 2, device=device).float() / dim)))
        .to(compute_dtype)
        .float()
    )
    freqs = (inv[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = (
        emb.cos().to(compute_dtype).unsqueeze(1),
        emb.sin().to(compute_dtype).unsqueeze(1),
    )
    half = dim // 2
    return cos, sin[..., :half], -sin[..., half:]


def _qwen_forward(module, *args, compute_dtype=torch.bfloat16, **kwargs):
    if kwargs.get("embeds") is not None:
        kwargs["embeds"] = kwargs["embeds"].to(compute_dtype)
    kwargs["dtype"] = compute_dtype
    return type(module).forward(module, *args, **kwargs)


def _is_causal_mask(mask):
    if mask is None:
        return True
    if mask.shape[-2] != mask.shape[-1]:
        return False
    if mask.dtype == torch.bool:
        expected = torch.ones_like(mask).tril()
    else:
        expected = torch.full_like(mask, torch.finfo(mask.dtype).min / 4).triu(1)
    return torch.equal(mask, expected)


def _qwen_attention(
    module,
    hidden_states,
    attention_mask=None,
    freqs_cis=None,
    optimized_attention=None,
    past_key_value=None,
    sliding_window=None,
    attention_backend=None,
):
    if past_key_value is not None or sliding_window is not None:
        return type(module).forward(
            module,
            hidden_states,
            attention_mask,
            freqs_cis,
            optimized_attention,
            past_key_value,
            sliding_window,
        )
    batch, length, _ = hidden_states.shape
    q = module.q_proj(hidden_states).view(batch, length, module.num_heads, module.head_dim).transpose(1, 2)
    k = module.k_proj(hidden_states).view(batch, length, module.num_kv_heads, module.head_dim).transpose(1, 2)
    v = module.v_proj(hidden_states).view(batch, length, module.num_kv_heads, module.head_dim).transpose(1, 2)
    cos = freqs_cis[0].to(q.dtype)
    sin = torch.cat((freqs_cis[1], -freqs_cis[2]), dim=-1).to(q.dtype)

    def rotate(x):
        # Qwen rounds both BF16 products before addition.
        half = x.shape[-1] // 2
        return x * cos + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sin

    q, k = rotate(q), rotate(k)
    attention = optimized_attention if attention_backend is None else attention_backend
    attention_args = {}
    if attention_backend is not None and _is_causal_mask(attention_mask):
        attention_mask = None
        attention_args["is_causal"] = True
    out = attention(
        q,
        k,
        v,
        module.num_heads,
        mask=attention_mask,
        skip_reshape=True,
        enable_gqa=module.num_heads != module.num_kv_heads,
        **attention_args,
    )
    return module.o_proj(out), None


def _glyph_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3.0))))


def _glyph_layout(weight):
    return weight.T.contiguous().T


def _byt5_attention(
    module,
    x,
    mask=None,
    past_bias=None,
    optimized_attention=None,
    attention_backend=None,
):
    return type(module).forward(
        module,
        x,
        mask,
        past_bias,
        optimized_attention if attention_backend is None else attention_backend,
    )


def _byt5_eager_attention(module, x, mask=None, past_bias=None, optimized_attention=None):
    """Preserve CPU T5 eager operation ordering through ComfyUI projections."""
    batch, length, _ = x.shape
    q, k, v = [
        projection(x).view(batch, length, module.num_heads, -1).transpose(1, 2)
        for projection in (module.q, module.k, module.v)
    ]
    scores = torch.matmul(q, k.transpose(3, 2))
    if past_bias is None:
        if module.relative_attention_bias is None:
            past_bias = torch.zeros((1, module.num_heads, length, length), device=x.device, dtype=x.dtype)
        else:
            past_bias = module.compute_bias(length, length, x.device, x.dtype)
        if mask is not None:
            past_bias = past_bias + mask
    # The first layer returns bias with the mask already added, as in HF T5.
    scores += past_bias
    weights = torch.nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
    output = torch.matmul(weights, v).transpose(1, 2).contiguous()
    return module.o(output.view(batch, length, -1)), past_bias


def _cpu_byt5_forward(module, tokens):
    previous = module.execution_device
    module.execution_device = torch.device("cpu")
    try:
        return type(module).forward(module, tokens)
    finally:
        module.execution_device = previous


def patch_vimoo_text(clip, attention_backend="comfy", precision_profile="bf16_transposed_ffn"):
    if attention_backend not in ("comfy", "precise"):
        raise ValueError(f"Unsupported Vimoo encoder attention backend: {attention_backend!r}")
    qwen_attention = get_attention_function("vimoo_qwen") if attention_backend == "precise" else None
    encoder = clip.cond_stage_model
    patcher = clip.patcher
    qwen = encoder.qwen25_7b.transformer.model
    prefix = "qwen25_7b.transformer.model"
    cpu_glyph_profile = precision_profile in ("lite_ti2v_anchor", "lite_compact")
    compute_dtype = (
        torch.float16 if cpu_glyph_profile or precision_profile == "standard_i2v" else torch.bfloat16
    )
    methods = {
        prefix + ".forward": MethodType(partial(_qwen_forward, compute_dtype=compute_dtype), qwen),
        prefix + ".compute_freqs_cis": MethodType(
            partial(_qwen_frequencies, compute_dtype=compute_dtype), qwen
        ),
    }
    if cpu_glyph_profile:
        methods["byt5_small.forward"] = MethodType(_cpu_byt5_forward, encoder.byt5_small)
    for name, module in qwen.named_modules():
        if isinstance(module, llama.RMSNorm):
            methods[f"{prefix}.{name}.forward"] = MethodType(_qwen_norm, module)
        elif isinstance(module, llama.Attention):
            methods[f"{prefix}.{name}.forward"] = MethodType(
                partial(_qwen_attention, attention_backend=qwen_attention), module
            )
    for name, module in encoder.byt5_small.named_modules():
        if isinstance(module, t5.T5Attention) and attention_backend == "precise":
            if precision_profile == "lite_compact":
                methods[f"byt5_small.{name}.forward"] = MethodType(_byt5_eager_attention, module)
            else:
                byt5_attention = "vimoo_byt5_cpu" if cpu_glyph_profile else "vimoo_byt5"
                methods[f"byt5_small.{name}.forward"] = MethodType(
                    partial(
                        _byt5_attention,
                        attention_backend=get_attention_function(byt5_attention),
                    ),
                    module,
                )
        if isinstance(module, t5.T5DenseGatedActDense):
            prefix = f"byt5_small.{name}"
            patcher.add_object_patch(prefix + ".act", _glyph_gelu)
            if precision_profile != "lite_compact":
                for linear_name in ("wi_0", "wi_1", "wo"):
                    patcher.add_weight_wrapper(f"{prefix}.{linear_name}.weight", _glyph_layout)

    add_method_patches(patcher, "vimoo_text_math", methods)
