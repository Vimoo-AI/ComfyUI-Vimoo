"""hidden_merge_split patch projections and mixed-precision timestep embeddings."""

import math

import torch
from torch import nn

import comfy.ops
from comfy.ldm.flux.layers import MLPEmbedder, ModulationOut
from .vimoo_compile import acquire_weights, release_weights


class FP32Linear(comfy.ops.manual_cast.Linear):
    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return super().forward(x.float())


class _PreserveModulationDtype:
    """Move modulation parameters through module hooks without rounding their storage."""

    def _apply(self, fn, recurse=True):
        def preserve(value):
            moved = fn(value)
            if moved.dtype != value.dtype:
                return value.to(device=moved.device)
            return moved

        return super()._apply(preserve, recurse=recurse)


class FP32ModulationProjection(_PreserveModulationDtype, FP32Linear):
    pass


def bf16_modulation_expansion(code, weight, bias):
    expanded = torch.nn.functional.linear(code.to(torch.bfloat16), weight, None)
    return expanded.float() + bias


class BF16ModulationExpansion(_PreserveModulationDtype, comfy.ops.manual_cast.Linear):
    """BF16 expansion GEMM followed by an unfused FP32 bias addition."""

    def __init__(self, in_features, out_features, device=None):
        super().__init__(in_features, out_features, dtype=torch.bfloat16, device=device)
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias.float(), requires_grad=False)
        self.weight_comfy_model_dtype = torch.bfloat16
        self.bias_comfy_model_dtype = torch.float32
        self.math = bf16_modulation_expansion

    def forward(self, code):
        comfy.ops.run_every_op()
        with torch.autocast(device_type=code.device.type, enabled=False):
            # The native cast context supplies ModelPatcher/DynamicVRAM hooks and
            # releases asynchronous offload buffers after both operations.
            weight, bias, cast_context = acquire_weights(
                self, code, dtype=torch.bfloat16, bias_dtype=torch.float32, offloadable=True
            )
            try:
                return self.math(code, weight, bias)
            finally:
                release_weights(cast_context)


class BottleneckModulation(nn.Module):
    """A modulation Linear that expands a shared low-rank timestep code.

    A shared rank-r code expands through one ``[6*hidden, r]`` Linear per stream.
    Ordinary Lite keeps its FP32 expansion; full-size checkpoints with shared low-rank timestep modulation use
    a BF16 GEMM followed by an FP32 bias addition. The model projects ``vec`` once
    after SiLU, so this module only expands and casts the six modulation groups.
    """

    def __init__(
        self, hidden_size, rank, dtype=None, device=None, operations=None, precision="fp32", output_dtype=None
    ):
        super().__init__()
        self.is_double = True
        self.multiplier = 6
        self.output_dtype = dtype if output_dtype is None else output_dtype
        if precision == "bf16_expand_fp32_bias":
            self.lin = BF16ModulationExpansion(rank, self.multiplier * hidden_size, device=device)
        elif precision == "fp32":
            self.lin = FP32Linear(rank, self.multiplier * hidden_size, dtype=torch.float32, device=device)
        else:
            raise ValueError(f"Unsupported block modulation precision: {precision!r}")

    def forward(self, code):
        if code.ndim == 2:
            code = code[:, None, :]
        out = self.lin(code).to(self.output_dtype).chunk(self.multiplier, dim=-1)
        return ModulationOut(*out[:3]), ModulationOut(*out[3:])


class FP32Conv3d(comfy.ops.manual_cast.Conv3d):
    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return super().forward(x.float())


class TimestepEmbedder(MLPEmbedder):
    def __init__(self, hidden_size, fp32_projection=False, dtype=None, device=None, operations=None):
        self.fp32_projection = fp32_projection
        if fp32_projection:
            nn.Module.__init__(self)
            self.in_layer = FP32Linear(256, hidden_size, dtype=torch.float32, device=device)
            self.silu = nn.SiLU()
            self.out_layer = FP32Linear(hidden_size, hidden_size, dtype=torch.float32, device=device)
        else:
            super().__init__(256, hidden_size, dtype=dtype, device=device, operations=operations)

    def forward(self, frequencies, output_dtype):
        if self.fp32_projection:
            with torch.autocast(device_type=frequencies.device.type, enabled=False):
                return super().forward(frequencies.float()).to(output_dtype)
        return super().forward(frequencies.to(output_dtype))


class HMSPatchEmbed(nn.Module):
    def __init__(self, in_channels, hidden_size, device=None):
        super().__init__()
        settings = dict(dtype=torch.float32, device=device)
        self.proj = FP32Conv3d(in_channels, hidden_size, kernel_size=1, stride=1, **settings)
        self.patch_merge = FP32Linear(4 * hidden_size, hidden_size, **settings)

    def forward(self, x):
        embedded = self.proj(x)
        batch, hidden, frames, height, width = embedded.shape
        tokens = (
            embedded.reshape(batch, hidden, frames, height // 2, 2, width // 2, 2)
            .permute(0, 2, 3, 5, 4, 6, 1)
            .reshape(batch, -1, 4 * hidden)
        )
        return self.patch_merge(tokens).to(x.dtype)


class HMSLastLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        out_channels,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        settings = dict(dtype=torch.float32, device=device)
        self.patch_split = FP32Linear(hidden_size, 4 * hidden_size, **settings)
        self.linear = FP32Linear(hidden_size, out_channels, **settings)
        self.norm_final = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), operations.Linear(hidden_size, 2 * hidden_size, **settings)
        )

    def precompute_modulation(self, vec):
        # Convert vec to FP32 before SiLU, but leaves this modulation
        # projection under the surrounding autocast policy.
        return torch.stack(self.adaLN_modulation(vec.float()).to(vec.dtype).chunk(2, dim=-1))

    def forward(self, x, vec, modulate_function=None, precomputed_modulation=None, preserve_fp32=False):
        if preserve_fp32:
            # SiLU consumes the FP32 timestep vector, then its BF16
            # modulation Linear precedes the FP32 patch/output projection.
            activated = self.adaLN_modulation[0](vec.float()).to(torch.bfloat16)
            with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
                modulation = torch.stack(self.adaLN_modulation[1](activated).chunk(2, dim=-1))
            with torch.autocast(device_type=x.device.type, enabled=False):
                return self.forward(x.float(), vec.float(), modulate_function, modulation)
        if precomputed_modulation is None:
            shift, scale = self.precompute_modulation(vec).unbind(0)
        else:
            shift, scale = precomputed_modulation.unbind(0)
        batch, length, hidden = x.shape
        delta = self.patch_split(x)
        states = x.unsqueeze(2) + delta.reshape(batch, length, 4, hidden)
        states = self.norm_final(states)
        states = (
            states.float() * (1.0 + scale.float().reshape(batch, 1, 1, hidden))
            + shift.float().reshape(batch, 1, 1, hidden)
        ).to(states.dtype)
        values = self.linear(states.reshape(batch, length * 4, hidden))
        return (
            values.reshape(batch, length, 4, self.out_channels)
            .transpose(2, 3)
            .reshape(batch, length, 4 * self.out_channels)
            .to(x.dtype)
        )


class VimooRoPE(nn.Module):
    def __init__(self, axes_dim, theta, ref_area, strength, ref_fps, ramp="yarn", temporal_mode="divide"):
        super().__init__()
        if ref_fps is not None and (not math.isfinite(ref_fps) or ref_fps <= 0):
            raise ValueError("rope_temporal_interpolation_ref_fps must be positive and finite")
        if ramp not in ("yarn", "none", "", None):
            raise ValueError(f"Unsupported Vimoo rotary interpolation ramp: {ramp}")
        if temporal_mode not in ("divide", "multiply_reciprocal"):
            raise ValueError(f"Unsupported Vimoo temporal interpolation mode: {temporal_mode!r}")
        self.ramp = ramp
        self.axes_dim = axes_dim
        self.theta = theta
        self.ref_area = ref_area
        self.strength = strength
        self.ref_fps = ref_fps
        self.use_reciprocal_temporal_scale = temporal_mode == "multiply_reciprocal"
        self.compute_on_device = False

    def forward(self, ids, latent_shape, fps=None):
        spatial_scale = 1.0
        if self.ramp == "yarn" and self.ref_area is not None:
            spatial_scale = max(1.0, math.sqrt(latent_shape[-2] * latent_shape[-1] / self.ref_area))
        spatial_scale = 1.0 + (spatial_scale - 1.0) * self.strength
        temporal_scale = 1.0
        if self.ref_fps is not None:
            if fps is None:
                raise ValueError(
                    "Vimoo temporal RoPE requires fps; connect the generation frame rate to Vimoo Sampler."
                )
            if not math.isfinite(float(fps)) or float(fps) <= 0:
                raise ValueError("fps must be positive and finite")
            temporal_scale = 1.0 + (float(fps) / self.ref_fps - 1.0) * self.strength
            if temporal_scale <= 0:
                raise ValueError("Vimoo temporal RoPE scale must be positive")
        # Standard profiles use CPU tables. Lite TI2V computes FP32 positions,
        # frequencies and trigonometry on the active model device.
        positions = ids.float() if self.compute_on_device else ids.cpu().float()
        if self.compute_on_device:
            temporal_scale = torch.as_tensor(1.0, device=positions.device, dtype=torch.float32)
            if self.ref_fps is not None:
                temporal_scale = torch.as_tensor(fps, device=positions.device, dtype=torch.float32).reshape(
                    ()
                )
                temporal_scale = temporal_scale / self.ref_fps
            temporal_scale = 1.0 + (temporal_scale - 1.0) * self.strength
        embeddings = []
        for axis, dim in enumerate(self.axes_dim):
            freqs = 1.0 / (
                self.theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=positions.device) / dim)
            )
            pos = positions[..., axis]
            if axis == 0:
                # Preserve the checkpoint's reference arithmetic: standard uses reciprocal
                # multiplication and Lite uses division. Rounding the reciprocal first can
                # change the FP32 rotary table, so keep this choice when refactoring.
                pos = (
                    pos * (1.0 / temporal_scale)
                    if self.use_reciprocal_temporal_scale
                    else pos / temporal_scale
                )
            elif self.ramp == "yarn":
                mix = torch.clamp(
                    (torch.linspace(0.0, 1.0, dim // 2, device=positions.device) - 0.25) / 0.5, 0.0, 1.0
                )
                freqs = freqs / (1.0 + (spatial_scale - 1.0) * mix)
            angles = pos.unsqueeze(-1) * freqs
            embedding = torch.stack((angles.cos(), -angles.sin(), angles.sin(), angles.cos()), dim=-1)
            embeddings.append(embedding.reshape(*pos.shape, dim // 2, 2, 2))
        return torch.cat(embeddings, dim=-3).unsqueeze(1).to(ids.device)
