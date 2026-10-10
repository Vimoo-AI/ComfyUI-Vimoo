"""Reference arithmetic on ComfyUI-owned Vimoo blocks and casting operations."""

import math
from contextlib import nullcontext
from functools import partial
from types import MethodType

import torch
from comfy.ldm.flux import layers
from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.modules.attention import optimized_attention
from comfy.patcher_extension import CallbacksMP

from .vimoo_patchify import BottleneckModulation, FP32Linear
from .vimoo_compile import acquire_weights, release_weights, specialize_cast


def _rebind_methods(source, clone, functions):
    if source.model is clone.model:
        return
    for path, function in functions.items():
        method = clone.object_patches.get(path)
        if isinstance(method, MethodType) and method.__func__ is function:
            owner = clone.model.get_submodule(path.rpartition(".")[0])
            clone.add_object_patch(path, MethodType(function, owner))


def add_method_patches(patcher, key, methods):
    for path, method in methods.items():
        patcher.add_object_patch(path, method)
    patcher.add_callback_with_key(
        CallbacksMP.ON_CLONE,
        key,
        partial(_rebind_methods, functions={path: method.__func__ for path, method in methods.items()}),
    )


def _anchor_linear_forward(module, value, *, vimoo_lora_delta=None):
    """Linear projections add bias separately from the matrix product."""
    context = (
        torch.autocast(device_type=value.device.type, enabled=False)
        if isinstance(module, FP32Linear)
        else nullcontext()
    )
    with context:
        if isinstance(module, FP32Linear):
            value = value.float()
        else:
            # Linear projections consume the parameter dtype. Cast
            # the activation before Comfy stages weights, avoiding a needless
            # BF16 -> FP32 -> BF16 weight copy for FP32 normalized activations.
            value = value.to(module.weight.dtype)
        weight, bias, cast_context = acquire_weights(module, value, offloadable=True)
        try:
            output = torch.nn.functional.linear(value, weight, None)
            if vimoo_lora_delta is not None:
                # Add the LoRA delta to the matrix product before adding bias,
                # retaining the base + delta rounding boundary.
                output = output + vimoo_lora_delta(value).to(output.dtype)
            if bias is None:
                return output
            if module.vimoo_upcast_bias:
                return output.float() + bias.float()
            return output + bias.to(output.dtype)
        finally:
            release_weights(cast_context)


def configure_anchor_linears(model):
    """Configure one profile instance; retain ComfyUI casting and state keys."""
    prefixes = ("img_in.", "time_in.", "txt_in.", "vision_in.", "byt5_in.", "final_layer.")
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or name.endswith("qkv"):
            continue
        decoder_projection = name.startswith("double_blocks.") and any(
            part in name for part in ("_mod.", "_attn.proj", "_mlp.")
        )
        if name.startswith(prefixes) or name == "mod_projection" or decoder_projection:
            module.vimoo_separate_bias = True
            module.vimoo_upcast_bias = decoder_projection and name.endswith(
                ("img_attn.proj", "img_mlp.2", "txt_mlp.2")
            )
            module.forward = MethodType(specialize_cast(_anchor_linear_forward, module), module)


def timestep_embedding(
    t, dim, max_period=10000, time_factor=1000.0, frequencies=None, frequency_float64=False
):
    half = dim // 2
    if frequency_float64:
        # Build frequencies and trigonometry on the timestep's device
        # in FP64 instead of using the CPU FP32 conditioning table.
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float64, device=t.device) / half
        )
        phases = (t * time_factor)[:, None].to(torch.float64) * frequencies[None]
        result = torch.cat((phases.cos(), phases.sin()), dim=-1)
        return torch.cat((result, torch.zeros_like(result[:, :1])), dim=-1) if dim % 2 else result
    if frequencies is None:
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
    phases = (t * time_factor)[:, None].float() * frequencies[None]
    result = torch.cat((phases.cos(), phases.sin()), dim=-1)
    return torch.cat((result, torch.zeros_like(result[:, :1])), dim=-1) if dim % 2 else result


def precise_qk_norm(q, k, qw, kw, eps):
    packed = torch.stack((q, k))
    value = packed.float()
    value = (value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + eps)).to(packed.dtype)
    q, k = value.unbind(0)
    return q * qw, k * kw


def anchor_qk_norm(q, k, qw, kw, eps):
    """Apply FP32 tree reduction and scaling before one BF16 rounding."""
    packed = torch.stack((q, k))
    value = packed.float()
    dim = value.shape[-1]
    if dim % 4 == 0 and (dim // 4) & (dim // 4 - 1) == 0:
        squares = value.unflatten(-1, (dim // 4, 4)).square()
        variance = squares[..., 0] + squares[..., 1]
        variance = (variance + squares[..., 2]) + squares[..., 3]
        while variance.shape[-1] > 1:
            variance = variance[..., ::2] + variance[..., 1::2]
        variance = variance / dim
    else:
        variance = value.square().mean(-1, keepdim=True)
    q, k = (value * torch.rsqrt(variance + eps)).unbind(0)
    return (q * qw.float()).to(packed.dtype), (k * kw.float()).to(packed.dtype)


def precise_rope(x, cos, sin):
    if cos.ndim == 2:
        shape = (1, x.shape[1], 1, x.shape[-1])
        cos = cos.to(device=x.device, dtype=torch.float32).view(*shape)
        sin = sin.to(device=x.device, dtype=torch.float32).view(*shape)
    value = x.float()
    real, imag = value.reshape(*value.shape[:-1], -1, 2).unbind(-1)
    rotated = torch.stack((-imag, real), dim=-1).flatten(-2)
    return (value * cos + rotated * sin).to(x.dtype)


def precise_modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def precise_rope_pair(q, k, cos, sin):
    return precise_rope(q, cos, sin), precise_rope(k, cos, sin)


def precise_gate(x, gate):
    return x * gate.unsqueeze(1)


def precise_fa2_pack(iq, ik, iv, tq, tk, tv, text_mask):
    packed = torch.stack(
        (torch.cat((iq, tq), dim=1), torch.cat((ik, tk), dim=1), torch.cat((iv, tv), dim=1)),
        dim=2,
    )
    return packed, torch.nn.functional.pad(text_mask, (iq.shape[1], 0), value=True)


def precise_refiner_block(block, x, c, mask, options, gate):
    source = options.get("vimoo_lite_ti2v_anchor", False)
    if source:
        activated = block.adaLN_modulation[0](c).to(x.dtype)
        mod1, mod2 = block.adaLN_modulation[1](activated).chunk(2, dim=1)
    else:
        mod1, mod2 = block.adaLN_modulation(c).chunk(2, dim=1)
    qkv = block.self_attn.qkv(block.norm1(x))
    q, k, v = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, block.heads, -1).permute(2, 0, 3, 1, 4)
    attn = optimized_attention(
        q, k, v, block.heads, mask=mask, skip_reshape=True, transformer_options=options
    )
    if source:
        x = (x.float() + block.self_attn.proj(attn).float() * mod1.float().unsqueeze(1)).to(x.dtype)
        return (x.float() + block.mlp(block.norm2(x)).float() * mod2.float().unsqueeze(1)).to(x.dtype)
    x = x + gate(block.self_attn.proj(attn), mod1)
    return x + gate(block.mlp(block.norm2(x)), mod2)


def _modulation_vector(value):
    return value.squeeze(1) if value.ndim == 3 else value


def _squeeze_modulation(modulation):
    return layers.ModulationOut(
        _modulation_vector(modulation.shift),
        _modulation_vector(modulation.scale),
        _modulation_vector(modulation.gate),
    )


def _pack_modulations(img_mod1, img_mod2, txt_mod1, txt_mod2):
    """Store one block's four AdaLN results in a compact tensor table."""
    values = []
    for modulation in (img_mod1, img_mod2, txt_mod1, txt_mod2):
        values.extend(
            _modulation_vector(value) for value in (modulation.shift, modulation.scale, modulation.gate)
        )
    return torch.stack(values)


def _unpack_modulations(values):
    values = values.unbind(0)
    outputs = []
    for offset in range(0, 12, 3):
        outputs.append(layers.ModulationOut(*values[offset : offset + 3]))
    return outputs


def qk_norm(q, k, norm, function=None):
    if function is not None:
        eps = norm.query_norm.eps
        if eps is None:
            eps = torch.finfo(q.dtype).eps
        qw, _, query_context = acquire_weights(norm.query_norm, q, offloadable=True)
        try:
            kw, _, key_context = acquire_weights(norm.key_norm, k, offloadable=True)
            try:
                q, k = function(q, k, qw, kw, eps)
            finally:
                release_weights(key_context)
        finally:
            release_weights(query_context)
        return q, k
    # Use native RMSNorm for weight casting/offloading and scale-before-rounding.
    return norm.query_norm(q), norm.key_norm(k)


def apply_rope(x, pe, function=None):
    if isinstance(pe, tuple):
        # A precomputed rotary table is already cos/sin, and apply_rope1 cannot read
        # that form; the rotation itself is the same either way.
        return (function or precise_rope)(x, *pe)
    if function is not None:
        length, dim = x.shape[1], x.shape[-1]
        cos = pe[0, 0, ..., 0, 0].repeat_interleave(2, dim=-1).view(1, length, 1, dim)
        sin = pe[0, 0, ..., 1, 0].repeat_interleave(2, dim=-1).view(1, length, 1, dim)
        return function(x, cos, sin)
    return apply_rope1(x.transpose(1, 2), pe).transpose(1, 2)


def modulate(x, modulation, dims=None, function=None):
    if dims is not None:
        return layers.apply_mod(x, 1 + modulation.scale, modulation.shift, dims)
    shift = _modulation_vector(modulation.shift)
    scale = _modulation_vector(modulation.scale)
    if function is not None:
        return function(x, shift, scale)
    # The published compiled modulation computes in FP32 and rounds once.
    # Use Comfy's addcmul helper without silently enabling torch.compile.
    dtype = torch.promote_types(x.dtype, scale.dtype)
    return precise_modulate(x.float(), shift.float(), scale.float()).to(dtype)


def apply_gate(x, gate, dims=None, function=None):
    if dims is not None:
        return layers.apply_mod(x, gate, None, dims)
    if function is not None:
        return function(x, _modulation_vector(gate))
    return x * _modulation_vector(gate).unsqueeze(1)


def observed_modulate(x, regular, observed, mask):
    """Select the F0 time inside FP32 pointwise math, before the BF16 rounding."""

    def expression(mod):
        shift = _modulation_vector(mod.shift).float().unsqueeze(1)
        scale = _modulation_vector(mod.scale).float().unsqueeze(1)
        return x.float() * (1 + scale) + shift

    return torch.where(mask, expression(observed), expression(regular)).to(x.dtype)


def modulate_fp32(x, modulation):
    scale = _modulation_vector(modulation.scale).float().unsqueeze(1)
    shift = _modulation_vector(modulation.shift).float().unsqueeze(1)
    return (x.float() * (1 + scale) + shift).to(x.dtype)


def observed_gate(x, regular, observed, mask):
    gate = torch.where(
        mask, _modulation_vector(observed.gate).unsqueeze(1), _modulation_vector(regular.gate).unsqueeze(1)
    )
    return x * gate


def source_gate_add(residual, value, regular, observed=None, mask=None):
    gate = _modulation_vector(regular.gate).unsqueeze(1)
    if observed is not None:
        gate = torch.where(mask, _modulation_vector(observed.gate).unsqueeze(1), gate)
    return (residual.float() + value.float() * gate.float()).to(residual.dtype)


def _fused_qkv(attn, x):
    """Consume one resident packed projection, including native INT8 dispatch."""
    if not isinstance(attn, PackedSelfAttention):
        raise ValueError("QKV fusion requires qkv_layout=packed in Vimoo Model Loader")
    return attn.qkv(x).chunk(3, dim=-1)


@torch.compiler.disable
def _post_attention_graph_boundary(image_attn, text_attn):
    """Keep output projections separate from residual/MLP regional graphs."""
    return image_attn, text_attn


class SplitSelfAttention(torch.nn.Module):
    """ComfyUI attention storage with Diffusers-shaped Q/K/V modules.

    Released Vimoo ComfyUI checkpoints keep the three projections in one
    safetensors entry.  The loader exposes three views of that entry to these
    modules, preserving the file format and resident bytes while giving
    regional compilation the same three-GEMM graph as the reference model.
    """

    def __init__(
        self, dim, num_heads=8, qkv_bias=False, proj_bias=True, dtype=None, device=None, operations=None
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.to_q = operations.Linear(dim, dim, bias=qkv_bias, dtype=dtype, device=device)
        self.to_k = operations.Linear(dim, dim, bias=qkv_bias, dtype=dtype, device=device)
        self.to_v = operations.Linear(dim, dim, bias=qkv_bias, dtype=dtype, device=device)
        self.norm = layers.QKNorm(head_dim, dtype=dtype, device=device, operations=operations)
        self.proj = operations.Linear(dim, dim, bias=proj_bias, dtype=dtype, device=device)

    def qkv(self, x):
        # The stock Flux attention-patch ABI expects one concatenated projection.
        return torch.cat((self.to_q(x), self.to_k(x), self.to_v(x)), dim=-1)


class PackedSelfAttention(torch.nn.Module):
    """Keep checkpoint QKV storage intact through casting, offload and GEMM."""

    def __init__(
        self, dim, num_heads=8, qkv_bias=False, proj_bias=True, dtype=None, device=None, operations=None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = operations.Linear(dim, 3 * dim, bias=qkv_bias, dtype=dtype, device=device)
        self.norm = layers.QKNorm(dim // num_heads, dtype=dtype, device=device, operations=operations)
        self.proj = operations.Linear(dim, dim, bias=proj_bias, dtype=dtype, device=device)


class DoubleStreamBlock(layers.DoubleStreamBlock):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio,
        qkv_bias=False,
        dtype=None,
        device=None,
        operations=None,
        packed_qkv=False,
        modulation_rank=None,
        modulation_precision="fp32",
        mlp_act_type="gelu_tanh",
    ):
        torch.nn.Module.__init__(self)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        # The stock forward used for attention patches reads this flag.
        self.modulation = True
        attention = getattr(layers, "ComfyAttention", None)
        self.comfy_attention = attention() if attention is not None else None
        # modulation_rank: the checkpoint carries the factorized form instead, so each
        # stream expands a shared rank-r code rather than projecting vec itself.
        self.modulation_rank = modulation_rank
        self.mixed_precision_modulation = (
            modulation_rank is not None and modulation_precision == "bf16_expand_fp32_bias"
        )
        settings = dict(dtype=dtype, device=device, operations=operations)
        for stream in ("img", "txt"):
            if modulation_rank is not None:
                setattr(
                    self,
                    f"{stream}_mod",
                    BottleneckModulation(
                        hidden_size,
                        modulation_rank,
                        precision=modulation_precision,
                        output_dtype=torch.float32
                        if stream == "txt" and self.mixed_precision_modulation
                        else dtype,
                        **settings,
                    ),
                )
            else:
                setattr(self, f"{stream}_mod", layers.Modulation(hidden_size, double=True, **settings))
            setattr(
                self,
                f"{stream}_norm1",
                operations.LayerNorm(
                    hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
                ),
            )
            attention = PackedSelfAttention if packed_qkv else SplitSelfAttention
            setattr(self, f"{stream}_attn", attention(hidden_size, num_heads, qkv_bias, **settings))
            setattr(
                self,
                f"{stream}_norm2",
                operations.LayerNorm(
                    hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
                ),
            )
            mlp = layers.build_mlp(hidden_size, int(hidden_size * mlp_ratio), **settings)
            if mlp_act_type == "gelu":
                # build_mlp defaults to tanh GELU; this checkpoint requires exact erf GELU.
                mlp[1] = torch.nn.GELU()
            elif mlp_act_type != "gelu_tanh":
                raise ValueError(f"Unsupported Vimoo MLP activation: {mlp_act_type!r}")
            setattr(self, f"{stream}_mlp", mlp)

    def precompute_modulation(self, vec):
        """Evaluate the small, timestep-only AdaLN branches outside the main block loop.

        ``vec`` is the low-rank code, not the timestep vector, when the block was built
        with a modulation rank; ``VimooModel.modulation_code`` decides which.
        """
        return _pack_modulations(*self.img_mod(vec), *self.txt_mod(vec))

    def body_modules(self):
        """Dynamic-VRAM roots for the block after AdaLN has been precomputed."""
        return tuple(
            getattr(self, f"{stream}_{name}")
            for stream in ("img", "txt")
            for name in ("norm1", "attn", "norm2", "mlp")
        )

    def _resolve_modulations(self, vec, observed_vec, precomputed_modulation, img_dtype, txt_dtype):
        """Project global and first-frame codes in their original shared GEMM."""
        observed_mods = None
        if precomputed_modulation is not None:
            img_mod1, img_mod2, txt_mod1, txt_mod2 = _unpack_modulations(precomputed_modulation)
        elif observed_vec is not None:
            # Project the stacked global/F0 modulation codes
            # in one Linear; preserve that GEMM's row count and rounding.
            joined = self.img_mod(torch.cat((vec, observed_vec), dim=0))
            batch = vec.shape[0]
            regular, observed_mods = [], []
            for modulation in joined:
                regular.append(
                    layers.ModulationOut(
                        *(value[:batch] for value in (modulation.shift, modulation.scale, modulation.gate))
                    )
                )
                observed_mods.append(
                    layers.ModulationOut(
                        *(value[batch:] for value in (modulation.shift, modulation.scale, modulation.gate))
                    )
                )
            img_mod1, img_mod2 = regular
            txt_mod1, txt_mod2 = self.txt_mod(vec)
        else:
            img_mod1, img_mod2 = self.img_mod(vec)
            txt_mod1, txt_mod2 = self.txt_mod(vec)
        if self.mixed_precision_modulation:
            # A packed cache promotes image and text groups to FP32. Restore each
            # stream's activation dtype without rounding the FP32 text modulation.
            def cast(modulation, dtype):
                return layers.ModulationOut(
                    *(value.to(dtype) for value in (modulation.shift, modulation.scale, modulation.gate))
                )

            img_mod1, img_mod2 = cast(img_mod1, img_dtype), cast(img_mod2, img_dtype)
            txt_mod1, txt_mod2 = cast(txt_mod1, txt_dtype), cast(txt_mod2, txt_dtype)
            if observed_mods is not None:
                observed_mods = [cast(modulation, img_dtype) for modulation in observed_mods]
        return img_mod1, img_mod2, txt_mod1, txt_mod2, observed_mods

    @staticmethod
    def _pack_joint_qkv(image_qkv, text_qkv, attn_mask, fa2_pack_math, use_fa2_pack):
        """Pack image-first QKV with explicit allocation and autograd boundaries."""
        iq, ik, iv = image_qkv
        tq, tk, tv = text_qkv
        # Use one image-first BSHD QKV allocation. Besides
        # avoiding three large allocations, the out= boundary keeps the full
        # QK-norm + RoPE expression together before joint-stream packing.
        batch, image_length, heads, head_dim = iq.shape
        total_length = image_length + tq.shape[1]
        if fa2_pack_math is not None and use_fa2_pack:
            if attn_mask is None:
                text_mask = torch.ones(tq.shape[:2], dtype=torch.bool, device=tq.device)
            else:
                text_mask = attn_mask[:, 0, image_length:]
                if text_mask.dtype != torch.bool:
                    text_mask = text_mask == 0
            packed, key_mask = fa2_pack_math(iq, ik, iv, tq, tk, tv, text_mask)
            q, k, v = packed.unbind(2)
            attn_mask = key_mask.unsqueeze(1)
        elif torch.is_grad_enabled():
            packed = torch.stack(
                tuple(torch.cat(pair, dim=1) for pair in ((iq, tq), (ik, tk), (iv, tv))),
                dim=0,
            )
            q, k, v = packed.unbind(0)
        else:
            # Every element is overwritten by the three cat(out=...) calls.
            packed = iq.new_empty((3, batch, total_length, heads, head_dim))
            torch.cat((iq, tq), dim=1, out=packed[0])
            torch.cat((ik, tk), dim=1, out=packed[1])
            torch.cat((iv, tv), dim=1, out=packed[2])
            q, k, v = packed.unbind(0)
        return q, k, v, attn_mask

    @staticmethod
    def _add_residual(
        value,
        update,
        modulation,
        dims,
        gate_math,
        source_residual,
        observed_modulation=None,
        observed_mask=None,
    ):
        """Apply the same gated residual policy after attention and after the MLP."""
        if source_residual:
            return source_gate_add(value, update, modulation, observed_modulation, observed_mask)
        gated = (
            observed_gate(update, modulation, observed_modulation, observed_mask)
            if observed_modulation is not None
            else apply_gate(update, modulation.gate, dims, gate_math)
        )
        return value + gated

    def forward(
        self,
        img,
        txt,
        vec,
        pe,
        attn_mask=None,
        modulation_dims_img=None,
        modulation_dims_txt=None,
        transformer_options=None,
        precomputed_modulation=None,
    ):
        options = transformer_options or {}
        norm_math, rope_math, mod_math = options.get("vimoo_math", (None, None, None))
        gate_math, rope_pair_math, fa2_pack_math = options.get("vimoo_i2v_math", (None, None, None))
        fused_qkv = options.get("vimoo_fused_qkv", False)
        # Preserve the stock patch ABI (text-first QKV and positional tables).
        # Patches deliberately change the computation, so use the stock block.
        patches = options.get("patches", {})
        if patches.get("attn1_patch") or patches.get("attn1_output_patch"):
            if options.get("vimoo_observed_vec") is not None:
                raise ValueError("Observed-frame Lite TI2V does not support stock attention patches")
            return super().forward(
                img, txt, vec, pe, attn_mask, modulation_dims_img, modulation_dims_txt, options
            )
        observed_vec = options.get("vimoo_observed_vec")
        img_mod1, img_mod2, txt_mod1, txt_mod2, observed_mods = self._resolve_modulations(
            vec, observed_vec, precomputed_modulation, img.dtype, txt.dtype
        )
        observed_mask = options.get("vimoo_observed_mask")
        source_residual = options.get("vimoo_lite_ti2v_anchor", False)
        if source_residual:
            norm_math = anchor_qk_norm
        if options.get("vimoo_math_graph"):
            img_mod1, img_mod2 = (
                _squeeze_modulation(img_mod1),
                _squeeze_modulation(img_mod2),
            )
            txt_mod1, txt_mod2 = (
                _squeeze_modulation(txt_mod1),
                _squeeze_modulation(txt_mod2),
            )
        streams = []
        for value, norm, attn, mod, dims in (
            (img, self.img_norm1, self.img_attn, img_mod1, modulation_dims_img),
            (txt, self.txt_norm1, self.txt_attn, txt_mod1, modulation_dims_txt),
        ):
            if value is img and observed_mods is not None:
                projected = observed_modulate(norm(value), mod, observed_mods[0], observed_mask)
            elif source_residual:
                projected = modulate_fp32(norm(value), mod)
            else:
                projected = modulate(norm(value), mod, dims, mod_math)
            if fused_qkv or isinstance(attn, PackedSelfAttention):
                q, k, v = _fused_qkv(attn, projected)
            else:
                q, k, v = attn.to_q(projected), attn.to_k(projected), attn.to_v(projected)
            q, k, v = [
                tensor.view(value.shape[0], value.shape[1], self.num_heads, -1) for tensor in (q, k, v)
            ]
            q, k = qk_norm(q, k, attn.norm, norm_math)
            streams.append((q, k, v))
        text_length = txt.shape[1]
        image_pe = pe if isinstance(pe, tuple) else pe[:, :, text_length:]
        iq, ik, iv = streams[0]
        tq, tk, tv = streams[1]
        if rope_pair_math is not None and isinstance(image_pe, tuple):
            iq, ik = rope_pair_math(iq, ik, *image_pe)
        else:
            iq, ik = apply_rope(iq, image_pe, rope_math), apply_rope(ik, image_pe, rope_math)
        q, k, v, attn_mask = self._pack_joint_qkv(
            (iq, ik, iv), (tq, tk, tv), attn_mask, fa2_pack_math, options.get("vimoo_i2v_fa2")
        )
        # Packed QKV now owns the attention inputs. Release the original
        # streams and RoPE intermediates before allocating attention/MLP outputs.
        del streams, projected, iq, ik, iv, tq, tk, tv
        attn = optimized_attention(
            q.flatten(2),
            k.flatten(2),
            v.flatten(2),
            self.num_heads,
            mask=attn_mask,
            transformer_options=options,
            vimoo_attention_kind="joint",
        )
        del q, k, v
        image_attn, text_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]
        if (
            options.get("vimoo_attention_projection_layout") == "contiguous"
            or options.get("vimoo_full_i2v")
            or source_residual
        ):
            # Make both stream slices contiguous before projection.
            # Strided CFG batches otherwise select different BF16 GEMM kernels.
            image_attn, text_attn = image_attn.contiguous(), text_attn.contiguous()
        image_attn = self.img_attn.proj(image_attn)
        text_attn = self.txt_attn.proj(text_attn)
        del attn
        if source_residual and attn_mask is not None:
            valid_text = attn_mask[..., img.shape[1] :]
            if valid_text.dtype != torch.bool:
                valid_text = valid_text == 0
            text_attn = text_attn * valid_text.transpose(-1, -2).to(text_attn.dtype)
        if options.get("vimoo_math_graph"):
            image_attn, text_attn = _post_attention_graph_boundary(image_attn, text_attn)
        img = self._add_residual(
            img,
            image_attn,
            img_mod1,
            modulation_dims_img,
            gate_math,
            source_residual,
            None if observed_mods is None else observed_mods[0],
            observed_mask,
        )
        txt = self._add_residual(txt, text_attn, txt_mod1, modulation_dims_txt, gate_math, source_residual)
        del image_attn, text_attn

        if observed_mods is not None:
            image_mlp_input = observed_modulate(
                self.img_norm2(img), img_mod2, observed_mods[1], observed_mask
            )
        elif source_residual:
            image_mlp_input = modulate_fp32(self.img_norm2(img), img_mod2)
        else:
            image_mlp_input = modulate(self.img_norm2(img), img_mod2, modulation_dims_img, mod_math)
        image_mlp = self.img_mlp(image_mlp_input)
        del image_mlp_input
        img = self._add_residual(
            img,
            image_mlp,
            img_mod2,
            modulation_dims_img,
            gate_math,
            source_residual,
            None if observed_mods is None else observed_mods[1],
            observed_mask,
        )
        del image_mlp

        text_mlp_input = (
            modulate_fp32(self.txt_norm2(txt), txt_mod2)
            if source_residual
            else modulate(self.txt_norm2(txt), txt_mod2, modulation_dims_txt, mod_math)
        )
        text_mlp = self.txt_mlp(text_mlp_input)
        txt = self._add_residual(txt, text_mlp, txt_mod2, modulation_dims_txt, gate_math, source_residual)
        return img, txt


class LastLayer(layers.LastLayer):
    def __init__(self, hidden_size, patch_size, out_channels, dtype=None, device=None, operations=None):
        torch.nn.Module.__init__(self)
        self.norm_final = operations.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )
        self.linear = operations.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True, dtype=dtype, device=device
        )
        self.adaLN_modulation = torch.nn.Sequential(
            torch.nn.SiLU(),
            operations.Linear(hidden_size, 2 * hidden_size, bias=True, dtype=dtype, device=device),
        )

    def precompute_modulation(self, vec):
        return torch.stack(self.adaLN_modulation(vec).chunk(2, dim=-1))

    def forward(self, x, vec, modulation_dims=None, modulate_function=None, precomputed_modulation=None):
        if precomputed_modulation is None:
            shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        else:
            shift, scale = precomputed_modulation.unbind(0)
        if shift.ndim == 2:
            shift, scale = shift[:, None], scale[:, None]
        return self.linear(
            modulate(
                self.norm_final(x),
                layers.ModulationOut(shift, scale, None),
                modulation_dims,
                modulate_function,
            )
        )
