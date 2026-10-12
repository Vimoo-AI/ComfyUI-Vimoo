"""Vimoo's FP32 Wan VAE using ComfyUI loading, decoding and offloading."""

import math
import copy
import logging

from types import MethodType
from functools import partial

import torch
from einops import rearrange
import folder_paths
import comfy.latent_formats
import comfy.model_management
import comfy.ops
import comfy.sd
import comfy.utils
from comfy.ldm.wan.vae import AttentionBlock as WanAttentionBlock, CausalConv3d, RMS_norm
from comfy.ldm.wan.vae2_2 import (
    CACHE_T as WAN22_CACHE_T,
    ResidualBlock as Wan22ResidualBlock,
    Up_ResidualBlock as Wan22UpResidualBlock,
    WanVAE as Wan22VAE,
    count_conv3d,
    patchify,
    unpatchify,
)
from .vimoo_precision import add_method_patches


class VimooLatentFormat(comfy.latent_formats.Wan22):
    def process_in(self, latent):
        mean = self.latents_mean.to(device=latent.device, dtype=torch.bfloat16)
        inverse_std = (1.0 / self.latents_std.to(latent.device)).bfloat16()
        return (latent.bfloat16() - mean) * inverse_std

    def process_out(self, latent):
        mean = self.latents_mean.to(device=latent.device, dtype=torch.bfloat16)
        inverse_std = (1.0 / self.latents_std.to(latent.device)).bfloat16()
        return latent.bfloat16() / inverse_std + mean


class VimooFullI2VLatentFormat(VimooLatentFormat):
    """Euler leaves FP32 latents; only the normalization statistics round to BF16."""

    def process_out(self, latent):
        mean = self.latents_mean.to(device=latent.device, dtype=torch.bfloat16)
        inverse_std = (1.0 / self.latents_std.to(latent.device)).bfloat16()
        return latent.float() / inverse_std + mean


def _rms_forward(module, x):
    normalized = torch.nn.functional.normalize(x, dim=1 if module.channel_first else -1)
    gamma = comfy.ops.cast_to_input(module.gamma, normalized)
    # Preserve two BF16-rounded multiplications without retaining
    # three full-sized elementwise temporaries. Wan's norms normally have no
    # bias, so also avoid materializing the previous tensor + scalar-zero copy.
    normalized.mul_(module.scale)
    normalized.mul_(gamma)
    if module.bias is not None:
        normalized.add_(comfy.ops.cast_to_input(module.bias, normalized))
    return normalized


def validate_vae_numerical_profile(profile):
    if profile not in (None, "wan_fp32_norm"):
        raise ValueError(f"Unknown Vimoo VAE numerical profile: {profile!r}")
    return profile


def _rms_forward_fp32_affine(module, x):
    """Wan arithmetic: BF16 normalization/scaling, then FP32 affine."""
    # Partial VAE loading can leave plain RMS_norm parameters on the CPU.
    # Stage the small affine tensors without rounding their FP32 masters.
    # Casting gamma to the activation dtype changes both SiLU and the next
    # autocast convolution input, even for a BF16-representable checkpoint.
    normalized = torch.nn.functional.normalize(x, dim=1 if module.channel_first else -1)
    gamma = comfy.model_management.cast_to(module.gamma, dtype=torch.float32, device=x.device)
    bias = (
        comfy.model_management.cast_to(module.bias, dtype=torch.float32, device=x.device)
        if module.bias is not None
        else 0.0
    )
    return normalized * module.scale * gamma + bias


def _attention_forward_sdpa(module, x, feat_cache=None, feat_idx=None, final=False):
    """Preserve SDPA dispatch and packed-QKV layout on ComfyUI modules."""
    del feat_cache, feat_idx, final
    identity = x
    batch, channels, frames, height, width = x.shape
    x = rearrange(x, "b c t h w -> (b t) c h w")
    x = module.norm(x)
    q, k, v = (
        module.to_qkv(x)
        .reshape(batch * frames, 1, channels * 3, -1)
        .permute(0, 1, 3, 2)
        .contiguous()
        .reshape(batch * frames, 1, height * width, channels * 3)
        .chunk(3, dim=-1)
    )
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = x.squeeze(1).permute(0, 2, 1).reshape(batch * frames, channels, height, width)
    x = module.proj(x)
    return rearrange(x, "(b t) c h w -> b c t h w", t=frames) + identity


def _attention_forward_native(module, x, feat_cache=None, feat_idx=None, final=False):
    if x.shape[-2:] == (1, 1):
        return _attention_forward_sdpa(module, x)
    return WanAttentionBlock.forward(module, x, feat_cache=feat_cache, feat_idx=feat_idx, final=final)


def _silu_forward(module, x):
    del module
    return torch.nn.functional.silu(x, inplace=True)


def _residual_forward_inplace(module, x, feat_cache=None, feat_idx=[0], explicit_numerical=False):
    """Wan2.2 residual math, retaining dtype promotion when requested."""
    old_x = x
    for layer in module.residual:
        if isinstance(layer, CausalConv3d) and feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -WAN22_CACHE_T:, :, :].clone()
            if cache_x.shape[2] < WAN22_CACHE_T and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            x = layer(x, cache_list=feat_cache, cache_idx=idx)
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = layer(x)
    shortcut = module.shortcut(old_x)
    return x + shortcut if explicit_numerical else x.add_(shortcut)


def _add_dup_up_shortcut_(output, source, shortcut, first_chunk=False):
    """Add DupUp3D's deterministic expansion without building the full shortcut."""
    factor_t, factor_s = shortcut.factor_t, shortcut.factor_s
    crop_t = factor_t - 1 if first_chunk else 0
    expected_shape = (
        source.shape[0],
        shortcut.out_channels,
        source.shape[2] * factor_t - crop_t,
        source.shape[3] * factor_s,
        source.shape[4] * factor_s,
    )
    if tuple(output.shape) != expected_shape:
        raise ValueError(
            f"Wan2.2 shortcut shape mismatch: main={tuple(output.shape)}, expected={expected_shape}"
        )

    identity_channels = shortcut.in_channels == shortcut.out_channels and shortcut.repeats == shortcut.factor
    for offset_t in range(factor_t):
        output_t = (offset_t - crop_t) % factor_t
        input_t = (output_t + crop_t) // factor_t
        for offset_h in range(factor_s):
            for offset_w in range(factor_s):
                offset = (offset_t * factor_s + offset_h) * factor_s + offset_w
                target = output[:, :, output_t::factor_t, offset_h::factor_s, offset_w::factor_s]
                if target.shape[2] == 0:
                    continue
                if identity_channels:
                    selected = source
                else:
                    channels = torch.div(
                        torch.arange(shortcut.out_channels, device=source.device) * shortcut.factor + offset,
                        shortcut.repeats,
                        rounding_mode="floor",
                    )
                    selected = source.index_select(1, channels)
                target.add_(selected[:, :, input_t : input_t + target.shape[2]])
    return output


def _up_residual_forward_inplace(
    module, x, feat_cache=None, feat_idx=[0], first_chunk=False, explicit_numerical=False
):
    """Wan2.2 up block without a full-sized materialized DupUp3D branch."""
    x_main = x
    for layer in module.upsamples:
        x_main = layer(x_main, feat_cache, feat_idx)
    if module.avg_shortcut is None:
        return x_main
    if explicit_numerical:
        x_main = x_main.to(torch.promote_types(x_main.dtype, x.dtype))
    return _add_dup_up_shortcut_(x_main, x, module.avg_shortcut, first_chunk)


def _wan_decode_output_shape(module, shape):
    del module
    return (shape[0], 3, max(0, shape[2] * 4 - 3), shape[3] * 16, shape[4] * 16)


def _wan_encode_with_device(module, x, device=None):
    if device is not None:
        x = x.to(device)
    return Wan22VAE.encode(module, x)


def _wan_decode_streaming(module, z, output_buffer=None):
    """Decode Wan2.2 chunks once, avoiding quadratic output concatenation.

    ComfyUI supplies a CPU output buffer for the normal VAEDecode path. Writing
    each completed temporal chunk into it keeps the full decoded clip out of
    device memory. Direct callers retain the native tensor return contract and
    use one final concatenation instead of one concatenation per latent frame.
    """
    expected_shape = _wan_decode_output_shape(module, z.shape)
    if output_buffer is not None and tuple(output_buffer.shape) != expected_shape:
        raise ValueError(
            f"Wan2.2 output buffer shape mismatch: got {tuple(output_buffer.shape)}, "
            f"expected {expected_shape}"
        )

    feat_map = [None] * count_conv3d(module.decoder)
    x = module.conv2(z)
    packed_chunks = []
    output_t = 0
    for index in range(z.shape[2]):
        conv_idx = [0]
        packed = module.decoder(
            x[:, :, index : index + 1, :, :],
            feat_cache=feat_map,
            feat_idx=conv_idx,
            first_chunk=index == 0,
        )
        if output_buffer is None:
            packed_chunks.append(packed)
            continue

        decoded = unpatchify(packed, patch_size=2)
        frames = decoded.shape[2]
        output_buffer[:, :, output_t : output_t + frames].copy_(decoded)
        output_t += frames

    if output_buffer is not None:
        if output_t != output_buffer.shape[2]:
            raise RuntimeError(
                f"Wan2.2 decoder produced {output_t} frames for a {output_buffer.shape[2]}-frame buffer"
            )
        return output_buffer
    return unpatchify(torch.cat(packed_chunks, dim=2), patch_size=2)


_BF16_MEMORY_HEADROOM = 1.15


def _bf16_autocast_memory_estimate(shape, dtype, *, estimate):
    del dtype
    return math.ceil(estimate(shape, torch.bfloat16) * _BF16_MEMORY_HEADROOM)


# Bound the im2col allocation responsible for PyTorch issue #166790. Other
# tensors and library workspaces still use Comfy's normal OOM recovery.
_IM2COL_WORKSPACE_LIMIT = 2 * 1024**3


def _conv_columns_bytes(module, x):
    output = [
        (size - dilation * (kernel - 1) - 1) // stride + 1
        for size, kernel, stride, dilation in zip(
            x.shape[2:], module.kernel_size, module.stride, module.dilation
        )
    ]
    # SlowDilated3d reuses this matrix between batch entries.
    return x.element_size() * x.shape[1] * math.prod(module.kernel_size) * math.prod(output)


def _causal_conv_forward(
    module,
    x,
    cache_x=None,
    cache_list=None,
    cache_idx=None,
    spatial_padding=None,
    im2col_bounded=False,
    im2col_workspace_limit=None,
):
    if cache_list is not None:
        cache_x = cache_list[cache_idx]
        cache_list[cache_idx] = None
    temporal_padding = module._padding
    if cache_x is not None and temporal_padding > 0:
        x = torch.cat((cache_x.to(x.device), x), dim=2)
        temporal_padding -= cache_x.shape[2]
    h, w = spatial_padding if spatial_padding is not None else (0, 0)
    x = torch.nn.functional.pad(x, (w, w, h, h, temporal_padding, 0))
    # Materialize the VAE compute dtype so Comfy's version-specific Conv3d
    # workspace workaround also sees the actual autocast weight dtype.
    if torch.is_autocast_enabled(x.device.type):
        x = x.to(torch.get_autocast_dtype(x.device.type))
    comfy.ops.run_every_op()
    with comfy.ops.CastBiasWeightContext(module, x, offloadable=True) as (weight, bias):
        if im2col_bounded and x.is_cuda and math.prod(module.kernel_size) != 1:
            if (
                x.dtype != torch.bfloat16
                or weight.dtype != torch.bfloat16
                or module.groups != 1
                or not x.is_contiguous()
                or not weight.is_contiguous()
            ):
                raise ValueError("Bounded reference Conv3d requires contiguous NCDHW BF16 and groups=1")
            columns_bytes = _conv_columns_bytes(module, x)
            columns_limit = (
                _IM2COL_WORKSPACE_LIMIT if im2col_workspace_limit is None else im2col_workspace_limit
            )
            if columns_bytes > columns_limit:
                raise comfy.model_management.OOM_EXCEPTION(
                    f"Vimoo reference Conv3d needs {columns_bytes} im2col bytes; "
                    f"limit is {columns_limit}. Use tiled VAE processing or increase the loader's workspace budget."
                )
            # Select the same im2col/GEMM kernel used by Torch 2.9's NCDHW
            # BF16 dispatch directly. Newer Torch/cuDNN dispatch rules must
            # not switch this bounded path to a different convolution.
            return torch.ops.aten.slow_conv_dilated3d.default(
                x, weight, module.kernel_size, bias, module.stride, (0, 0, 0), module.dilation
            )
        # Native dispatch keeps ComfyUI's cuDNN Conv3d workaround; Torch 2.9's
        # own BF16 Conv3d dispatch is about 2x slower and uses several times the memory.
        return module._conv_forward(x, weight, bias)


def _blend_tiles(previous, current, axis, extent=32):
    extent = min(previous.shape[axis], current.shape[axis], extent)
    if extent == 0:
        return current
    # Blend packed pixels in BF16, before unpatchify/clamp. Dtype-matched
    # vector weights preserve two rounded multiplications,
    # while replacing 32 Python launches with one elementwise expression.
    shape = [1] * current.ndim
    shape[axis] = extent
    current_slice = current.narrow(axis, 0, extent)
    previous_slice = previous.narrow(axis, previous.shape[axis] - extent, extent)
    weight = torch.tensor(
        [index / extent for index in range(extent)], device=current.device, dtype=torch.float32
    ).reshape(shape)
    inverse_weight = torch.tensor(
        [1 - index / extent for index in range(extent)],
        device=current.device,
        dtype=torch.float32,
    ).reshape(shape)
    previous_weighted = (previous_slice.float() * inverse_weight).to(current.dtype)
    current_weighted = (current_slice.float() * weight).to(current.dtype)
    current_slice.copy_(previous_weighted + current_weighted)
    return current


def _spatial_tiled_decode(
    model,
    samples,
    vae_options=None,
    tile_height=256,
    tile_width=256,
    stride_height=192,
    stride_width=192,
):
    """Spatial tiling using Wan decoder modules."""
    vae_options = {} if vae_options is None else vae_options
    latent_tile_height, latent_tile_width = tile_height // 16, tile_width // 16
    latent_stride_height, latent_stride_width = stride_height // 16, stride_width // 16
    packed_stride_height, packed_stride_width = stride_height // 2, stride_width // 2
    blend_height = (tile_height - stride_height) // 2
    blend_width = (tile_width - stride_width) // 2
    previous_row = None
    results = []
    for top in range(0, samples.shape[3], latent_stride_height):
        row = []
        for left in range(0, samples.shape[4], latent_stride_width):
            tile = samples[
                :,
                :,
                :,
                top : top + latent_tile_height,
                left : left + latent_tile_width,
            ]
            # Let the native VAE own temporal caching and decode options.
            decoded = model.decode(tile, **vae_options)
            row.append(patchify(decoded, patch_size=2))
        result_row = []
        for j, tile in enumerate(row):
            if previous_row is not None and blend_height:
                tile = _blend_tiles(previous_row[j], tile, 3, blend_height)
            if j and blend_width:
                tile = _blend_tiles(row[j - 1], tile, 4, blend_width)
            result_row.append(tile[:, :, :, :packed_stride_height, :packed_stride_width])
        results.append(torch.cat(result_row, dim=4).to(device="cpu", copy=True))
        previous_row = row
    packed = torch.cat(results, dim=3)[:, :, :, : samples.shape[3] * 8, : samples.shape[4] * 8]
    return unpatchify(packed, patch_size=2).clamp(-1, 1)


def _im2col_memory_estimate(shape, dtype, *, estimate, columns_limit=_IM2COL_WORKSPACE_LIMIT):
    return estimate(shape, dtype) + columns_limit


def _source_decode_memory_coefficients(decoder):
    """Count causal caches and the largest chunk from the native decoder geometry."""
    spatial_scale = 1
    temporal_scale = 1
    cache_channels_area = 0
    peak_channels_area = 0

    def account(module):
        nonlocal cache_channels_area, peak_channels_area
        for child in module.modules():
            if isinstance(child, CausalConv3d):
                area = spatial_scale**2
                peak_channels_area = max(
                    peak_channels_area, max(child.in_channels, child.out_channels) * area
                )
                # Pointwise shortcut convolutions do not receive feat_cache.
                if child.kernel_size[0] > 1:
                    cache_channels_area += child.in_channels * area

    account(decoder.conv1)
    account(decoder.middle)
    for block in decoder.upsamples:
        account(block)
        if block.avg_shortcut is not None:
            spatial_scale *= block.avg_shortcut.factor_s
            temporal_scale *= block.avg_shortcut.factor_t
    account(decoder.head)
    return cache_channels_area, peak_channels_area, temporal_scale


def _source_decode_memory_estimate(shape, dtype, *, estimate, coefficients):
    # The native FP32 estimate covers convolution workspace. Explicit affine
    # promotion, non-inplace residuals and temporal caches need extra capacity.
    # Reserve eight largest chunk tensors for norm/SiLU, residual/dup, padding
    # and convolution staging; streaming keeps the full RGB buffer on CPU.
    cache_channels_area, peak_channels_area, temporal_scale = coefficients
    batch_pixels = shape[0] * shape[-2] * shape[-1]
    cache_frames = min(WAN22_CACHE_T, shape[2])
    chunk_frames = 1 if shape[2] == 1 else temporal_scale
    additional_elements = batch_pixels * (
        cache_channels_area * cache_frames + 8 * peak_channels_area * chunk_frames
    )
    return estimate(shape, torch.float32) + additional_elements * torch.float32.itemsize


class VimooVAE(comfy.sd.VAE):
    def __init__(
        self,
        sd,
        metadata=None,
        decode_mode="auto",
        convolution_backend="comfy",
        tile_height=256,
        tile_width=256,
        stride_height=192,
        stride_width=192,
        im2col_workspace_limit_gib=2.0,
        numerical_profile=None,
    ):
        self.numerical_profile = validate_vae_numerical_profile(numerical_profile)
        if decode_mode not in ("auto", "tiled"):
            raise ValueError(f"Unknown Vimoo VAE decode mode: {decode_mode}")
        if convolution_backend not in ("comfy", "im2col_bounded"):
            raise ValueError(f"Unknown Vimoo VAE convolution backend: {convolution_backend}")
        tile_values = (tile_height, tile_width, stride_height, stride_width)
        if any(value < 16 or value % 16 for value in tile_values):
            raise ValueError("Vimoo VAE tile sizes and strides must be positive multiples of 16")
        if stride_height > tile_height or stride_width > tile_width:
            raise ValueError("Vimoo VAE tile strides cannot exceed their tile sizes")
        if not math.isfinite(im2col_workspace_limit_gib) or im2col_workspace_limit_gib <= 0:
            raise ValueError("Vimoo reference convolution workspace budget must be positive and finite")
        self.im2col_workspace_limit = int(im2col_workspace_limit_gib * 1024**3)
        super().__init__(sd=sd, dtype=torch.float32, metadata=metadata)
        self.convolution_backend = convolution_backend
        if self.device.type == "cuda" and self.numerical_profile is None:
            # The legacy norm uses BF16 activations throughout. Source FP32
            # affine/SiLU intermediates keep native FP32 memory estimates so
            # Comfy can evict staged models before allocating those tensors.
            for name in ("memory_used_encode", "memory_used_decode"):
                estimate = getattr(self, name)
                setattr(self, name, partial(_bf16_autocast_memory_estimate, estimate=estimate))
        elif self.device.type == "cuda" and self.numerical_profile == "wan_fp32_norm":
            self.memory_used_decode = partial(
                _source_decode_memory_estimate,
                estimate=self.memory_used_decode,
                coefficients=_source_decode_memory_coefficients(self.first_stage_model.decoder),
            )
        if convolution_backend == "im2col_bounded":
            if self.device.type == "cuda" and torch.version.hip is not None:
                raise RuntimeError("im2col_bounded requires an NVIDIA CUDA GPU; select comfy on ROCm")
            # Reserve the bounded columns buffer alongside native activation
            # estimates. Native allocation failures still trigger tiled recovery.
            for name in ("memory_used_encode", "memory_used_decode"):
                estimate = getattr(self, name)
                setattr(
                    self,
                    name,
                    partial(
                        _im2col_memory_estimate,
                        estimate=estimate,
                        columns_limit=self.im2col_workspace_limit,
                    ),
                )
        self.decode_mode = decode_mode
        self.tile_height = tile_height
        self.tile_width = tile_width
        self.stride_height = stride_height
        self.stride_width = stride_width
        self.last_decode_mode = None
        self.throw_exception_if_invalid()
        if self.latent_channels != 48:
            raise ValueError("Vimoo requires the Wan2.2 48-channel VAE")
        methods = {}
        for name, module in self.first_stage_model.named_modules():
            if isinstance(module, RMS_norm):
                rms_forward = _rms_forward
                if self.numerical_profile == "wan_fp32_norm":
                    if module.gamma.dtype != torch.float32 or (
                        module.bias is not None and module.bias.dtype != torch.float32
                    ):
                        raise ValueError("wan_fp32_norm requires FP32 VAE affine parameters")
                    rms_forward = _rms_forward_fp32_affine
                methods[name + ".forward"] = MethodType(rms_forward, module)
            elif isinstance(module, WanAttentionBlock):
                attention_forward = (
                    _attention_forward_sdpa
                    if self.numerical_profile == "wan_fp32_norm"
                    else _attention_forward_native
                )
                methods[name + ".forward"] = MethodType(attention_forward, module)
            elif isinstance(module, torch.nn.SiLU):
                methods[name + ".forward"] = MethodType(_silu_forward, module)
            elif isinstance(module, (Wan22ResidualBlock, Wan22UpResidualBlock)):
                forward = (
                    _residual_forward_inplace
                    if isinstance(module, Wan22ResidualBlock)
                    else _up_residual_forward_inplace
                )
                methods[name + ".forward"] = MethodType(
                    partial(forward, explicit_numerical=self.numerical_profile == "wan_fp32_norm"), module
                )
            elif isinstance(module, CausalConv3d):
                forward = partial(
                    _causal_conv_forward,
                    spatial_padding=module.padding[1:],
                    im2col_bounded=convolution_backend == "im2col_bounded",
                    im2col_workspace_limit=self.im2col_workspace_limit,
                )
                methods[name + ".forward"] = MethodType(forward, module)
                self.patcher.add_object_patch(name + ".padding", (0, 0, 0))
        methods["encode"] = MethodType(_wan_encode_with_device, self.first_stage_model)
        methods["decode"] = MethodType(_wan_decode_streaming, self.first_stage_model)
        add_method_patches(self.patcher, "vimoo_vae_math", methods)
        # ComfyUI checks this before applying model patches, so the capability
        # and shape method live directly on this plugin-owned model instance.
        self.first_stage_model.comfy_has_chunked_io = True
        # The shape helper needs no model reference; keep release independent of cyclic GC.
        self.first_stage_model.decode_output_shape = partial(_wan_decode_output_shape, None)

    def encode(self, pixel_samples):
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            return super().encode(pixel_samples)

    def encode_normalized(self, pixel_samples):
        """Encode already normalized pixels through the native VAE lifecycle."""
        view = copy.copy(self)
        view.process_input = lambda value: value
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            return super(VimooVAE, view).encode(pixel_samples)

    def decode(self, samples, vae_options=None):
        vae_options = {} if vae_options is None else vae_options
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            if self.decode_mode == "tiled" and (
                samples.shape[-2] > self.tile_height // 16 or samples.shape[-1] > self.tile_width // 16
            ):
                self.last_decode_mode = "reference_tiled"
                logging.info(
                    "Vimoo VAE decode: requested=%s actual=reference_tiled backend=%s "
                    "tile_pixels=%sx%s stride_pixels=%sx%s (height x width)",
                    self.decode_mode,
                    self.convolution_backend,
                    self.tile_height,
                    self.tile_width,
                    self.stride_height,
                    self.stride_width,
                )
                bounded = self._tile_bounded_shape(
                    samples.shape, self.tile_width // 16, self.tile_height // 16, None
                )
                batch_size = self.prepare_decode(bounded)
                outputs = []
                with comfy.model_management.cuda_device_context(self.device):
                    for start in range(0, samples.shape[0], batch_size):
                        decoded = _spatial_tiled_decode(
                            self.first_stage_model,
                            samples[start : start + batch_size].to(device=self.device, dtype=self.vae_dtype),
                            vae_options,
                            self.tile_height,
                            self.tile_width,
                            self.stride_height,
                            self.stride_width,
                        )
                        decoded = decoded.to(
                            device=self.output_device, dtype=self.vae_output_dtype(), copy=True
                        )
                        outputs.append(self.process_output(decoded).movedim(1, -1))
                return torch.cat(outputs)
            self.last_decode_mode = "untiled"
            logging.info(
                "Vimoo VAE decode: requested=%s attempting=untiled backend=%s",
                self.decode_mode,
                self.convolution_backend,
            )
            decoded = super().decode(samples, vae_options)
            logging.info("Vimoo VAE decode completed: actual=%s", self.last_decode_mode)
            return decoded

    def decode_tiled_3d(self, samples, tile_t=999, tile_x=32, tile_y=32, overlap=(1, 8, 8)):
        self.last_decode_mode = "comfy_tiled"
        if self.convolution_backend == "im2col_bounded":
            tile_x, tile_y = min(tile_x, 16), min(tile_y, 16)
            overlap = (overlap[0], min(overlap[1], tile_x // 4), min(overlap[2], tile_y // 4))
        logging.info(
            "Vimoo VAE decode: actual=comfy_tiled backend=%s tile_latent_txy=%s overlap_latent_txy=%s",
            self.convolution_backend,
            (tile_t, tile_x, tile_y),
            overlap,
        )
        return super().decode_tiled_3d(samples, tile_t, tile_x, tile_y, overlap)

    def encode_tiled_3d(self, samples, tile_t=9999, tile_x=512, tile_y=512, overlap=(1, 64, 64)):
        if self.convolution_backend == "im2col_bounded":
            tile_x, tile_y = min(tile_x, 256), min(tile_y, 256)
            overlap = (overlap[0], min(overlap[1], tile_x // 4), min(overlap[2], tile_y // 4))
        return super().encode_tiled_3d(samples, tile_t, tile_x, tile_y, overlap)

    def encode_tiled(self, pixel_samples, **kwargs):
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            return super().encode_tiled(pixel_samples, **kwargs)

    def decode_tiled(self, samples, **kwargs):
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            return super().decode_tiled(samples, **kwargs)


class VimooVAELoader:
    DESCRIPTION = (
        "Loads the VAE for reference-image encoding and/or output-video decoding, according to its connections. "
        "Separate reference/output loaders keep their settings independent while using the same weight file. "
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"),),
                "tiled_decode": ("BOOLEAN", {"default": False}),
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "Vimoo"

    def load_vae(self, vae_name, tiled_decode=False, model=None):
        return self._load_role(vae_name, model, "output", tiled_decode)

    @staticmethod
    def _load_role(vae_name, model, role, tiled_decode=None):
        from .vimoo_release import RELEASE_RECIPES, release_profile

        if model is None:
            raise ValueError("Connect the current Vimoo model to its VAE loader")
        options = dict(RELEASE_RECIPES[release_profile(model)]["vae"][role])
        if tiled_decode is not None:
            options["decode_mode"] = "tiled" if tiled_decode else "auto"
        options["numerical_profile"] = model.model_options.get("vimoo_vae_numerical_profile")
        return (_load_vimoo_vae(vae_name, **options),)


class VimooReferenceVAELoader(VimooVAELoader):
    DESCRIPTION = "Reference image encoding with the model's release numerical settings."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"),),
                "model": ("MODEL",),
            }
        }

    FUNCTION = "load_reference_vae"

    def load_reference_vae(self, vae_name, model):
        return self._load_role(vae_name, model, "reference")


def _load_vimoo_vae(
    vae_name,
    decode_mode,
    convolution_backend,
    tile_height,
    tile_width,
    stride_height,
    stride_width,
    im2col_workspace_limit_gib,
    numerical_profile,
):
    numerical_profile = validate_vae_numerical_profile(numerical_profile)
    path = folder_paths.get_full_path_or_raise("vae", vae_name)
    state, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    vae = VimooVAE(
        state,
        metadata,
        decode_mode,
        convolution_backend,
        tile_height,
        tile_width,
        stride_height,
        stride_width,
        im2col_workspace_limit_gib,
        numerical_profile,
    )
    vae.patcher.cached_patcher_init = (
        _reload_vimoo_vae,
        (
            vae_name,
            decode_mode,
            convolution_backend,
            tile_height,
            tile_width,
            stride_height,
            stride_width,
            im2col_workspace_limit_gib,
            numerical_profile,
        ),
    )
    return vae


def _reload_vimoo_vae(
    vae_name,
    decode_mode,
    convolution_backend="comfy",
    tile_height=256,
    tile_width=256,
    stride_height=192,
    stride_width=192,
    im2col_workspace_limit_gib=2.0,
    numerical_profile=None,
    disable_dynamic=False,
):
    vae = _load_vimoo_vae(
        vae_name,
        decode_mode,
        convolution_backend,
        tile_height,
        tile_width,
        stride_height,
        stride_width,
        im2col_workspace_limit_gib,
        numerical_profile,
    )
    if disable_dynamic:
        return vae.patcher.clone(disable_dynamic=True, model_override=vae.patcher.get_clone_model_override())
    return vae.patcher
