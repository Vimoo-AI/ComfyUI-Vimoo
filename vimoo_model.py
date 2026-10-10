# Vimoo video DiT implemented with ComfyUI Flux blocks and native operations.

import math
import torch
import comfy.model_prefetch
import comfy.patcher_extension
import comfy.ldm.flux.layers
import comfy.ldm.modules.diffusionmodules.mmdit
from comfy.ldm.hunyuan_video.model import (
    ByT5Mapper,
    IndividualTokenRefiner as NativeIndividualTokenRefiner,
)

from dataclasses import dataclass
from einops import repeat

from torch import Tensor, nn

from comfy.ldm.flux.layers import (
    MLPEmbedder,
)

import comfy.ldm.common_dit

from .vimoo_precision import (
    DoubleStreamBlock,
    LastLayer,
    timestep_embedding,
    precise_qk_norm,
    precise_rope,
    precise_modulate,
    precise_rope_pair,
    precise_gate,
    precise_fa2_pack,
    precise_refiner_block,
)
from .vimoo_patchify import FP32Linear, HMSPatchEmbed, HMSLastLayer, VimooRoPE, TimestepEmbedder


@dataclass
class VimooParams:
    in_channels: int
    out_channels: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    axes_dim: list
    theta: int
    patch_size: list
    qkv_bias: bool
    use_cond_type_embedding: bool
    vision_in_dim: int
    use_text_refiner_attention_mask: bool = True
    rope_spatial_interpolation_ramp: str | None = "yarn"
    byt5_dims: list | None = None
    rope_spatial_interpolation_ref_area: float | None = None
    rope_spatial_interpolation_strength: float = 1.0
    qk_norm_eps: float | None = None
    patch_architecture: str = "direct"
    block_modulation_rank: int | None = None
    block_modulation_precision: str = "fp32"
    mlp_act_type: str = "gelu_tanh"
    precision_policy: str = "bf16"
    rope_temporal_interpolation_ref_fps: float | None = None
    rope_temporal_interpolation_mode: str = "divide"
    attention_projection_layout: str = "strided"
    packed_qkv: bool = False


class IndividualTokenRefiner(NativeIndividualTokenRefiner):
    def forward(self, x, c, mask, transformer_options=None):
        transformer_options = transformer_options or {}
        refiner_options = dict(transformer_options)
        refiner_options["vimoo_attention_kind"] = "text_refiner"
        m = None
        if mask is not None:
            m = (mask[:, None, None, :] == 0).clone()
            m[..., 0] = True
            if not x.is_cuda:
                m = torch.zeros_like(m, dtype=x.dtype).masked_fill(~m, -torch.finfo(x.dtype).max)

        for block in self.blocks:
            if transformer_options.get("vimoo_lite_ti2v_anchor"):
                x = precise_refiner_block(block, x, c, m, refiner_options, precise_gate)
            elif "vimoo_i2v_math" in transformer_options:
                x = precise_refiner_block(
                    block, x, c, m, refiner_options, transformer_options["vimoo_i2v_math"][0]
                )
            else:
                x = block(x, c, m, transformer_options=refiner_options)
        return x


class TokenRefiner(nn.Module):
    def __init__(
        self,
        text_dim,
        hidden_size,
        heads,
        num_blocks,
        dtype=None,
        device=None,
        operations=None,
        fp32_time=False,
    ):
        super().__init__()

        self.input_embedder = operations.Linear(text_dim, hidden_size, bias=True, dtype=dtype, device=device)
        self.t_embedder = TimestepEmbedder(
            hidden_size, fp32_projection=fp32_time, dtype=dtype, device=device, operations=operations
        )
        self.c_embedder = MLPEmbedder(
            text_dim, hidden_size, dtype=dtype, device=device, operations=operations
        )
        self.individual_token_refiner = IndividualTokenRefiner(
            hidden_size, heads, num_blocks, dtype=dtype, device=device, operations=operations
        )

    def forward(
        self,
        x,
        timesteps,
        mask,
        transformer_options=None,
        precomputed_time=None,
    ):
        transformer_options = transformer_options or {}
        t = precomputed_time
        if t is None:
            t = self.t_embedder(
                timestep_embedding(
                    timesteps,
                    256,
                    time_factor=1.0,
                    frequencies=transformer_options.get("vimoo_frequencies"),
                    frequency_float64=transformer_options.get("vimoo_lite_ti2v_anchor", False),
                ),
                output_dtype=(
                    torch.float32 if transformer_options.get("vimoo_lite_ti2v_anchor") else x.dtype
                ),
            )

        static_cache = transformer_options.get("vimoo_static_conditioning_cache")
        uuids = transformer_options.get("uuids")
        cache_key = None
        static = None
        if static_cache is not None and uuids is not None:
            cache_key = (
                "qwen",
                tuple(uuids),
                tuple(transformer_options.get("cond_or_uncond", ())),
                tuple(x.shape),
                x.dtype,
                x.device,
            )
            static = static_cache.get(cache_key)
        if static is None:
            if mask is not None:
                valid = (mask == 0).to(x.dtype).unsqueeze(-1)
                c = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
            elif x.dtype == torch.float16:
                c = x.float().sum(dim=1) / x.shape[1]
            else:
                c = x.sum(dim=1) / x.shape[1]
            static = self.c_embedder(c.to(x.dtype)), self.input_embedder(x)
            if cache_key is not None:
                static_cache[cache_key] = static

        c = t + static[0]
        x = static[1]
        x = self.individual_token_refiner(x, c, mask, transformer_options=transformer_options)
        return x


class VimooModel(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, image_model=None, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.precise_math = False
        self.precise_inputs = False
        self.qk_norm_math = precise_qk_norm
        self.rope_math = precise_rope
        self.modulate_math = precise_modulate
        self.rope_pair_math = precise_rope_pair
        self.gate_math = precise_gate
        self.fa2_pack_math = precise_fa2_pack
        self.dtype = dtype
        self._math_inputs_cache_key = None
        self._math_inputs_cache_value = None
        operation_settings = {"operations": operations, "device": device, "dtype": dtype}

        # ComfyUI's Hunyuan detector also emits options for unrelated variants.
        for name, expected in {
            "vec_in_dim": None,
            "depth_single_blocks": 0,
            "guidance_embed": False,
            "byt5": True,
            "meanflow": False,
        }.items():
            if kwargs.pop(name, expected) != expected:
                raise ValueError(f"Unsupported Vimoo architecture: {name}")
        kwargs.pop("meanflow_sum", None)
        params = VimooParams(**kwargs)
        self.params = params
        self.patch_architecture = params.patch_architecture
        self.precision_policy = params.precision_policy
        self.hms = params.patch_architecture == "hidden_merge_split"
        if params.patch_architecture not in ("direct", "hidden_merge_split"):
            raise ValueError(f"Unsupported Vimoo patch architecture: {params.patch_architecture}")
        if self.hms and list(params.patch_size) != [1, 2, 2]:
            raise ValueError("hidden_merge_split requires patch_size=[1, 2, 2]")
        expected_policy = "fp32_io_time" if self.hms else "bf16"
        if params.precision_policy != expected_policy:
            raise ValueError(f"{params.patch_architecture} requires precision_policy={expected_policy}")
        self.patch_size = params.patch_size
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        self.use_cond_type_embedding = params.use_cond_type_embedding
        self.vision_in_dim = params.vision_in_dim
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = VimooRoPE(
            params.axes_dim,
            params.theta,
            params.rope_spatial_interpolation_ref_area,
            params.rope_spatial_interpolation_strength,
            params.rope_temporal_interpolation_ref_fps,
            params.rope_spatial_interpolation_ramp,
            params.rope_temporal_interpolation_mode,
        )
        if params.attention_projection_layout not in ("strided", "contiguous"):
            raise ValueError(
                f"Unsupported Vimoo attention projection layout: {params.attention_projection_layout!r}"
            )

        if self.hms:
            self.img_in = HMSPatchEmbed(
                self.in_channels,
                self.hidden_size,
                device=device,
            )
        else:
            self.img_in = comfy.ldm.modules.diffusionmodules.mmdit.PatchEmbed(
                None,
                self.patch_size,
                self.in_channels,
                self.hidden_size,
                dynamic_img_pad=False,
                conv3d=len(self.patch_size) == 3,
                dtype=dtype,
                device=device,
                operations=operations,
            )
        self.time_in = TimestepEmbedder(
            self.hidden_size,
            fp32_projection=params.precision_policy == "fp32_io_time",
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.txt_in = TokenRefiner(
            params.context_in_dim,
            self.hidden_size,
            self.num_heads,
            2,
            dtype=dtype,
            device=device,
            operations=operations,
            fp32_time=params.precision_policy == "fp32_io_time",
        )

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    modulation_rank=params.block_modulation_rank,
                    modulation_precision=params.block_modulation_precision,
                    mlp_act_type=params.mlp_act_type,
                    packed_qkv=params.packed_qkv,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
                for _ in range(params.depth)
            ]
        )

        if params.qk_norm_eps is not None:
            for block in self.double_blocks:
                block.img_attn.norm.query_norm.eps = params.qk_norm_eps
                block.img_attn.norm.key_norm.eps = params.qk_norm_eps
                block.txt_attn.norm.query_norm.eps = params.qk_norm_eps
                block.txt_attn.norm.key_norm.eps = params.qk_norm_eps

        byt5_dims = params.byt5_dims or [1472, 2048, 2048, self.hidden_size]
        self.byt5_in = ByT5Mapper(
            in_dim=byt5_dims[0],
            out_dim=byt5_dims[2],
            hidden_dim=byt5_dims[1],
            out_dim1=byt5_dims[3],
            dtype=dtype,
            device=device,
            operations=operations,
        )
        if self.hms:
            self.final_layer = HMSLastLayer(
                self.hidden_size,
                self.out_channels,
                device=device,
                operations=operations,
            )
        else:
            self.final_layer = LastLayer(
                self.hidden_size,
                self.patch_size[-1],
                self.out_channels,
                dtype=dtype,
                device=device,
                operations=operations,
            )

        # Optional conditioning modules used by video model variants.
        if self.vision_in_dim is not None:
            from comfy.ldm.wan.model import MLPProj

            self.vision_in = MLPProj(
                in_dim=self.vision_in_dim, out_dim=self.hidden_size, operation_settings=operation_settings
            )
        else:
            self.vision_in = None

        if self.use_cond_type_embedding:
            # 0: text_encoder feature 1: byt5 feature 2: vision_encoder feature
            self.cond_type_embedding = operations.Embedding(3, self.hidden_size, dtype=dtype, device=device)
        else:
            self.cond_type_embedding = None

        # One projection feeds every block's factorized modulation Linear; upstream runs
        # it in FP32 with autocast disabled, and so does FP32Linear.
        projection = FP32Linear
        if params.block_modulation_precision == "bf16_expand_fp32_bias":
            from .vimoo_patchify import FP32ModulationProjection

            projection = FP32ModulationProjection
        self.mod_projection = (
            projection(self.hidden_size, params.block_modulation_rank, dtype=torch.float32, device=device)
            if params.block_modulation_rank is not None
            else None
        )

    def get_dynamic_vram__units(self):
        """Stage each repeated transformer block as one DynamicVRAM unit."""
        return (list(self.double_blocks),)

    def prepare_math_inputs(self, shape, text_length, device, fps=None):
        # Image RoPE is independent of the text length. Keep a bounded cache so
        # positive/negative conditioning shares both the host work and H2D copy.
        spatial = tuple(math.ceil(size / patch) * patch for size, patch in zip(shape[2:], self.patch_size))
        cache_key = (
            tuple(shape[2:]),
            spatial,
            str(torch.device(device)),
            None if fps is None else float(fps),
        )
        if cache_key == self._math_inputs_cache_key:
            return self._math_inputs_cache_value
        layout = torch.empty(
            (1, 1, 1, 1, 1),
            dtype=torch.bfloat16,
            device=device if self.pe_embedder.compute_on_device else "cpu",
        ).expand(1, 1, *spatial)
        image_ids = self.img_ids(layout)
        pe = self.pe_embedder(image_ids, shape[2:], fps)
        image_pe = pe[0, 0]
        image_rope = torch.stack(
            (
                image_pe[..., 0, 0].repeat_interleave(2, dim=-1),
                image_pe[..., 1, 0].repeat_interleave(2, dim=-1),
            ),
            dim=0,
        ).unsqueeze(0)
        frequencies = self.cpu_frequency_table()
        value = (image_rope.to(device), frequencies.unsqueeze(0).to(device))
        self._math_inputs_cache_key = cache_key
        self._math_inputs_cache_value = value
        return value

    def modulation_code(self, vec):
        """What the per-block modulation Linears consume: vec, or its low-rank code."""
        if self.mod_projection is None:
            return vec
        with torch.autocast(device_type=vec.device.type, enabled=False):
            if self.params.block_modulation_precision == "bf16_expand_fp32_bias":
                return self.mod_projection(torch.nn.functional.silu(vec).float())
            return self.mod_projection(torch.nn.functional.silu(vec.float()))

    @staticmethod
    def cpu_frequency_table():
        """CPU sinusoidal frequency table passed as conditioning."""
        return torch.exp(-math.log(10000) * torch.arange(128, dtype=torch.float32) / 128)

    def uses_cpu_frequencies(self):
        """Whether VimooBaseModel.extra_conds attaches the CPU frequency table."""
        return self.hms or self.precise_math or self.precise_inputs

    @torch.no_grad()
    def precompute_modulation_schedule(self, timesteps, transformer_options=None, batch_sizes=(1, 2)):
        """Evaluate all timestep-only AdaLN branches once per sampling schedule.

        Must run under the same autocast policy as the DiT forward (see
        ``_precompute_modulation_sample_wrapper``); the tables are looked up
        by exact timestep, so any arithmetic difference would be visible.
        """
        options = transformer_options or {}
        device = timesteps.device
        roots = [self.time_in, self.txt_in.t_embedder]
        if self.mod_projection is not None:
            roots.append(self.mod_projection)
        modulation_root_start = len(roots)
        roots.extend((block.img_mod, block.txt_mod) for block in self.double_blocks)
        roots.append(self.final_layer.adaLN_modulation)
        queue = comfy.model_prefetch.make_prefetch_queue(roots, device, options)
        comfy.model_prefetch.prefetch_queue_pop(queue, device, roots[0])
        frequencies = options.get("vimoo_frequencies")
        if frequencies is None and self.uses_cpu_frequencies():
            frequencies = self.cpu_frequency_table().to(device)
        vectors = {}
        text_vectors = {}
        for batch in batch_sizes:
            vectors[batch] = tuple(
                self.time_in(
                    timestep_embedding(
                        timestep.reshape(1).expand(batch),
                        256,
                        time_factor=1.0,
                        frequencies=frequencies,
                    ),
                    output_dtype=self.dtype,
                )
                for timestep in timesteps
            )

        comfy.model_prefetch.prefetch_queue_pop(queue, device, roots[1])
        for batch in batch_sizes:
            text_vectors[batch] = torch.stack(
                tuple(
                    self.txt_in.t_embedder(
                        timestep_embedding(
                            timestep.reshape(1).expand(batch),
                            256,
                            time_factor=1.0,
                            frequencies=frequencies,
                        ),
                        output_dtype=self.dtype,
                    )
                    for timestep in timesteps
                )
            )

        # Stage the shared projection as its own queue root. Calling it while
        # an expansion root is prefetched can reuse that root's transient cast
        # buffer when its weights do not fit in VRAM. Compute each timestep code
        # once before any block expansion is staged, retaining GEMM row counts.
        if self.mod_projection is not None:
            comfy.model_prefetch.prefetch_queue_pop(queue, device, self.mod_projection)
        block_vectors = {
            batch: tuple(self.modulation_code(vec) for vec in vectors[batch]) for batch in batch_sizes
        }
        block_tables = {batch: [] for batch in batch_sizes}
        for index, block in enumerate(self.double_blocks):
            root = roots[index + modulation_root_start]
            comfy.model_prefetch.prefetch_queue_pop(queue, device, root)
            for batch in batch_sizes:
                block_tables[batch].append(
                    torch.stack(tuple(block.precompute_modulation(code) for code in block_vectors[batch]))
                )

        comfy.model_prefetch.prefetch_queue_pop(queue, device, roots[-1])
        final_tables = {
            batch: torch.stack(tuple(self.final_layer.precompute_modulation(vec) for vec in vectors[batch]))
            for batch in batch_sizes
        }
        comfy.model_prefetch.prefetch_queue_pop(queue, device, None)
        return {
            "timesteps": timesteps.detach().clone(),
            "text_time": text_vectors,
            "blocks": block_tables,
            "final": final_tables,
        }

    def _prepare_time_modulation(
        self, img, timesteps, initial_shape, observed_first_frame_timesteps, transformer_options
    ):
        """Select scheduled AdaLN tables or build the global and observed-frame codes."""
        # Block / final-layer AdaLN tables may come from the per-sampling runtime
        # schedule precomputed by the sampler wrapper.  It is keyed by the exact
        # model-side timestep and falls back to the full path on a mismatch.
        modulation_schedule = transformer_options.get("vimoo_modulation_schedule")
        if observed_first_frame_timesteps is not None:
            modulation_schedule = None
        modulation_index = None
        block_modulations = final_modulation = None
        if modulation_schedule is not None and img.shape[0] in modulation_schedule["blocks"]:
            scheduled_timesteps = modulation_schedule["timesteps"].to(timesteps)
            modulation_index = (scheduled_timesteps - timesteps[0]).abs().argmin()
            if bool(torch.all(timesteps == scheduled_timesteps[modulation_index]).item()):
                block_modulations = [
                    table[modulation_index] for table in modulation_schedule["blocks"][img.shape[0]]
                ]
                final_modulation = modulation_schedule["final"][img.shape[0]][modulation_index]
            else:
                modulation_schedule = None
        else:
            modulation_schedule = None
        if block_modulations is None:
            vec = self.time_in(
                timestep_embedding(
                    timesteps,
                    256,
                    time_factor=1.0,
                    frequencies=transformer_options.get("vimoo_frequencies"),
                    frequency_float64=transformer_options.get("vimoo_lite_ti2v_anchor", False),
                ),
                output_dtype=(
                    torch.float32 if transformer_options.get("vimoo_lite_ti2v_anchor") else img.dtype
                ),
            )
        else:
            # vec is only consumed by the modulation Linears.
            vec = None

        # The blocks take the low-rank code when the checkpoint is factorized; the final
        # layer is not factorized and keeps vec either way.
        block_vec = None
        if observed_first_frame_timesteps is None and vec is not None:
            block_vec = self.modulation_code(vec)
        if observed_first_frame_timesteps is not None:
            observed_vec = self.time_in(
                timestep_embedding(
                    observed_first_frame_timesteps.reshape(-1),
                    256,
                    time_factor=1.0,
                    frequencies=transformer_options.get("vimoo_frequencies"),
                    frequency_float64=transformer_options.get("vimoo_lite_ti2v_anchor", False),
                ),
                output_dtype=(
                    torch.float32 if transformer_options.get("vimoo_lite_ti2v_anchor") else img.dtype
                ),
            )
            block_vec, observed_code = self.modulation_code(torch.cat((vec, observed_vec), dim=0)).chunk(
                2, dim=0
            )
            transformer_options["vimoo_observed_vec"] = observed_code
            first_frame_tokens = (initial_shape[-2] // self.patch_size[-2]) * (
                initial_shape[-1] // self.patch_size[-1]
            )
            transformer_options["vimoo_observed_mask"] = (
                torch.arange(img.shape[1], device=img.device)[None, :, None] < first_frame_tokens
            )
        text_time = (
            modulation_schedule["text_time"][img.shape[0]][modulation_index]
            if modulation_schedule is not None
            else None
        )
        return vec, block_vec, block_modulations, final_modulation, text_time

    def _append_glyph_tokens(self, txt, txt_ids, txt_mask, txt_byt5, byt5_mask, transformer_options):
        """Reuse the glyph projection and append its tokens with the configured mask order."""
        if self.byt5_in is not None and txt_byt5 is not None:
            static_cache = transformer_options.get("vimoo_static_conditioning_cache")
            uuids = transformer_options.get("uuids")
            cache_key = None
            mapped_byt5 = None
            if static_cache is not None and uuids is not None:
                cache_key = (
                    "byt5",
                    tuple(uuids),
                    tuple(transformer_options.get("cond_or_uncond", ())),
                    tuple(txt_byt5.shape),
                    txt_byt5.dtype,
                    txt_byt5.device,
                )
                mapped_byt5 = static_cache.get(cache_key)
            if mapped_byt5 is None:
                mapped_byt5 = self.byt5_in(txt_byt5)
                if self.cond_type_embedding is not None:
                    cond_emb = self.cond_type_embedding(
                        torch.ones_like(mapped_byt5[:, :, 0], device=mapped_byt5.device, dtype=torch.long),
                        out_dtype=mapped_byt5.dtype,
                    )
                    mapped_byt5 = mapped_byt5 + cond_emb.to(mapped_byt5.dtype)
                if cache_key is not None:
                    static_cache[cache_key] = mapped_byt5
            txt_byt5 = mapped_byt5
            if self.cond_type_embedding is not None:
                txt = torch.cat((txt_byt5, txt), dim=1)
            else:
                txt = torch.cat((txt, txt_byt5), dim=1)
            if txt_mask is not None:
                glyph_mask = txt_mask.new_zeros((txt_mask.shape[0], txt_byt5.shape[1]))
                if byt5_mask is not None:
                    glyph_mask = (byt5_mask.to(txt_mask) - 1) * torch.finfo(txt_mask.dtype).max
                txt_mask = torch.cat(
                    (glyph_mask, txt_mask)
                    if self.cond_type_embedding is not None
                    else (txt_mask, glyph_mask),
                    dim=1,
                )
                if byt5_mask is not None:
                    valid = txt_mask == 0
                    order = torch.argsort(valid.int(), dim=1, descending=True, stable=True)
                    txt_mask = torch.gather(txt_mask, 1, order)
                    txt = torch.gather(txt, 1, order.unsqueeze(-1).expand_as(txt))
                    txt = txt * (txt_mask == 0).unsqueeze(-1).to(txt.dtype)
            if txt_ids is not None:
                txt_byt5_ids = torch.zeros(
                    (txt_ids.shape[0], txt_byt5.shape[1], txt_ids.shape[-1]),
                    device=txt_ids.device,
                    dtype=txt_ids.dtype,
                )
                txt_ids = torch.cat((txt_ids, txt_byt5_ids), dim=1)
        return txt, txt_ids, txt_mask

    def _append_vision_tokens(self, img, txt, txt_ids, txt_mask, clip_fea, vision_mask, transformer_options):
        """Reuse vision projections while preserving masked tokens and their positions."""
        if self.vision_in is not None:
            supplied_vision_mask = vision_mask
            static_cache = transformer_options.get("vimoo_static_conditioning_cache")
            uuids = transformer_options.get("uuids")
            cache_key = None
            txt_vision_states = None
            if static_cache is not None and uuids is not None:
                cache_key = (
                    "vision",
                    tuple(uuids),
                    tuple(transformer_options.get("cond_or_uncond", ())),
                    None if clip_fea is None else tuple(clip_fea.shape),
                    img.shape[0],
                    img.dtype,
                    img.device,
                )
                txt_vision_states = static_cache.get(cache_key)
            if txt_vision_states is None:
                if clip_fea is None:
                    vision_dtype = (
                        torch.float32 if transformer_options.get("vimoo_standard_t2v") else img.dtype
                    )
                    zero_vision = torch.zeros(
                        (img.shape[0], 729, self.vision_in_dim), device=img.device, dtype=vision_dtype
                    )
                    txt_vision_states = self.vision_in(zero_vision) * 0.0
                else:
                    txt_vision_states = self.vision_in(clip_fea)
                if transformer_options.get("vimoo_lite_ti2v_anchor") and supplied_vision_mask is not None:
                    txt_vision_states = txt_vision_states * supplied_vision_mask.to(
                        txt_vision_states
                    ).unsqueeze(-1)
                if self.cond_type_embedding is not None:
                    cond_emb = self.cond_type_embedding(
                        2
                        * torch.ones_like(
                            txt_vision_states[:, :, 0],
                            dtype=torch.long,
                            device=txt_vision_states.device,
                        ),
                        out_dtype=txt_vision_states.dtype,
                    )
                    txt_vision_states = txt_vision_states + cond_emb
                if cache_key is not None:
                    static_cache[cache_key] = txt_vision_states
            txt = torch.cat((txt_vision_states, txt), dim=1)
            if txt_mask is not None:
                vision_mask = txt_mask.new_zeros((txt_mask.shape[0], txt_vision_states.shape[1]))
                if clip_fea is None:
                    vision_mask.fill_(-torch.finfo(txt_mask.dtype).max)
                elif supplied_vision_mask is not None:
                    vision_mask = (supplied_vision_mask.to(txt_mask) - 1) * torch.finfo(txt_mask.dtype).max
                txt_mask = torch.cat((vision_mask, txt_mask), dim=1)
                order = torch.argsort((txt_mask == 0).int(), dim=1, descending=True, stable=True)
                txt_mask = torch.gather(txt_mask, 1, order)
                txt = torch.gather(txt, 1, order.unsqueeze(-1).expand_as(txt))
            if txt_ids is not None:
                extra_txt_ids = torch.zeros(
                    (txt_ids.shape[0], txt_vision_states.shape[1], txt_ids.shape[-1]),
                    device=txt_ids.device,
                    dtype=txt_ids.dtype,
                )
                txt_ids = torch.cat((txt_ids, extra_txt_ids), dim=1)
        return txt, txt_ids, txt_mask

    @staticmethod
    def _joint_attention_masks(img, txt, txt_mask, blocks_replace, transformer_options):
        """Cache the native image-first mask and Comfy's replacement-block mask."""
        mask_cache = transformer_options.get("vimoo_attention_mask_cache")
        uuids = transformer_options.get("uuids")
        mask_cache_key = None
        cached_masks = None
        if mask_cache is not None and uuids is not None:
            mask_cache_key = (
                "joint_attention",
                tuple(uuids),
                tuple(transformer_options.get("cond_or_uncond", ())),
                img.shape[0],
                img.shape[1],
                txt.shape[1],
                img.dtype,
                img.device,
                bool(blocks_replace),
            )
            cached_masks = mask_cache.get(mask_cache_key)

        if cached_masks is not None:
            attn_mask, patch_attn_mask = cached_masks
        else:
            patch_attn_mask = None
            if txt_mask is not None:
                # Blocks take the image-first boolean key mask once (additive on
                # CPU) instead of converting it in every block. Replaced blocks
                # still receive Comfy's text-first additive ABI and block_wrap
                # converts it back.
                valid = torch.cat(
                    (
                        torch.ones((img.shape[0], img.shape[1]), dtype=torch.bool, device=img.device),
                        txt_mask == 0,
                    ),
                    dim=1,
                ).unsqueeze(1)
                attn_mask = valid
                if not img.is_cuda:
                    attn_mask = torch.zeros_like(valid, dtype=img.dtype).masked_fill(
                        ~valid, -torch.finfo(img.dtype).max
                    )
                if blocks_replace:
                    patch_attn_mask = torch.zeros(
                        (img.shape[0], 1, txt.shape[1] + img.shape[1]), dtype=img.dtype, device=img.device
                    )
                    patch_attn_mask[:, 0, : txt.shape[1]] = txt_mask
            else:
                attn_mask = None
            if mask_cache_key is not None:
                mask_cache[mask_cache_key] = (attn_mask, patch_attn_mask)
        return attn_mask, patch_attn_mask

    def _block_options(self, transformer_options, attn_mask, text_length, retained_fa2_blocks):
        """Keep per-block indices out of the compiled attention options."""
        direct_block_options = transformer_options
        if not transformer_options.get("patches"):
            # The compiled block only consumes these stable options.  Keeping
            # the loop's changing block_index out of the regional graph lets
            # all 40 identical blocks reuse one Dynamo specialization.
            direct_block_options = {
                key: transformer_options[key]
                for key in (
                    "optimized_attention_override",
                    "vimoo_math",
                    "vimoo_math_graph",
                    "vimoo_full_i2v",
                    "vimoo_attention_projection_layout",
                    "vimoo_i2v_math",
                    "vimoo_i2v_fa2",
                    "vimoo_sage_image_only",
                    "vimoo_fused_qkv",
                    "vimoo_observed_vec",
                    "vimoo_observed_mask",
                    "vimoo_lite_ti2v_anchor",
                    "vimoo_symmetric_self_attention",
                    "vimoo_flash_padded_capacity",
                )
                if key in transformer_options
            }
            if attn_mask is not None:
                direct_block_options["vimoo_attention_metadata"] = transformer_options.get(
                    "vimoo_attention_metadata", {}
                )
        fa2_block_options = None
        if transformer_options.get("vimoo_math_graph"):
            if direct_block_options is transformer_options:
                direct_block_options = direct_block_options.copy()
            # Cache tensors passed as graph inputs leak pointer-keyed alias
            # guards even when their lookup is in a disabled helper. Regional
            # attention uses the helper's weak mask cache outside these inputs.
            direct_block_options.pop("vimoo_attention_metadata", None)
        if retained_fa2_blocks:
            fa2_block_options = direct_block_options.copy()
            fa2_block_options["optimized_attention_override"] = transformer_options["vimoo_sage_fa2_override"]
        transformer_options["total_blocks"] = len(self.double_blocks)
        transformer_options["block_type"] = "double"
        if transformer_options.get("vimoo_sage_image_only"):
            direct_block_options["vimoo_sage_text_length"] = text_length
        return direct_block_options, fa2_block_options

    def _forward_blocks(
        self,
        img,
        txt,
        block_vec,
        pe,
        attn_mask,
        patch_attn_mask,
        blocks_replace,
        block_modulations,
        control,
        transformer_options,
    ):
        """Run the double blocks with scoped prefetch, backend and replacement state."""
        retained_fa2_blocks = transformer_options.get("vimoo_sage_fa2_blocks", ())
        # With precomputed modulation the modulation Linears are not touched,
        # so Dynamic VRAM only stages the block bodies.
        block_roots = (
            [block.body_modules() for block in self.double_blocks]
            if block_modulations is not None
            else list(self.double_blocks)
        )
        prefetch_queue = (
            comfy.model_prefetch.make_prefetch_queue(block_roots, img.device, transformer_options)
            if transformer_options.get("vimoo_block_prefetch", True)
            else None
        )
        direct_block_options, fa2_block_options = self._block_options(
            transformer_options, attn_mask, txt.shape[1], retained_fa2_blocks
        )
        for i, block in enumerate(self.double_blocks):
            transformer_options["block_index"] = i
            block_modulation = None if block_modulations is None else block_modulations[i]
            comfy.model_prefetch.prefetch_queue_pop(
                prefetch_queue,
                img.device,
                block_roots[i],
            )
            if ("double_block", i) in blocks_replace:

                def block_wrap(args):
                    block_mask = args["attention_mask"]
                    if block_mask is not None:
                        text_length = args["txt"].shape[1]
                        block_mask = (
                            torch.cat((block_mask[..., text_length:], block_mask[..., :text_length]), dim=-1)
                            == 0
                        )
                        if not args["img"].is_cuda:
                            block_mask = torch.zeros_like(block_mask, dtype=args["img"].dtype).masked_fill(
                                ~block_mask, -torch.finfo(args["img"].dtype).max
                            )
                    out = {}
                    out["img"], out["txt"] = block(
                        img=args["img"],
                        txt=args["txt"],
                        vec=args["vec"],
                        pe=args["pe"],
                        attn_mask=block_mask,
                        transformer_options=args["transformer_options"],
                        precomputed_modulation=block_modulation,
                    )
                    return out

                out = blocks_replace[("double_block", i)](
                    {
                        "img": img,
                        "txt": txt,
                        "vec": block_vec,
                        "pe": pe,
                        "attention_mask": patch_attn_mask,
                        "transformer_options": transformer_options,
                    },
                    {"original_block": block_wrap},
                )
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(
                    img=img,
                    txt=txt,
                    vec=block_vec,
                    pe=pe,
                    attn_mask=attn_mask,
                    transformer_options=(
                        fa2_block_options if i in retained_fa2_blocks else direct_block_options
                    ),
                    precomputed_modulation=block_modulation,
                )

            if control is not None:  # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add
        if prefetch_queue is not None:
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, img.device, None)
        return img, txt

    def _unpatchify(self, img, initial_shape, transformer_options):
        """Restore the video layout and retain the autocast rounding boundary."""
        shape = initial_shape[-len(self.patch_size) :]
        for i in range(len(shape)):
            shape[i] = shape[i] // self.patch_size[i]
        img = img.reshape([img.shape[0]] + shape + [self.out_channels] + self.patch_size)
        img = img.permute(0, 4, 1, 5, 2, 6, 3, 7)
        img = img.reshape(initial_shape[0], self.out_channels, *initial_shape[2:])
        if transformer_options.get("vimoo_lite_ti2v_anchor") and torch.is_autocast_enabled(img.device.type):
            # Source unpatchify rounds under autocast. Its wrapper widens
            # only after the padding crop, preserving the dense flow layout.
            img = img.to(torch.get_autocast_dtype(img.device.type))
        return img

    def forward_orig(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        txt_mask: Tensor,
        timesteps: Tensor,
        txt_byt5=None,
        clip_fea=None,
        control=None,
        transformer_options=None,
        byt5_mask=None,
        precise_rope=None,
        precomputed_frequencies=None,
        latent_shape=None,
        fps=None,
        observed_first_frame_timesteps=None,
        vision_mask=None,
    ) -> Tensor:
        transformer_options = transformer_options or {}
        transformer_options = transformer_options.copy()
        if self.params.attention_projection_layout == "contiguous":
            transformer_options["vimoo_attention_projection_layout"] = "contiguous"
        if precomputed_frequencies is not None:
            transformer_options["vimoo_frequencies"] = precomputed_frequencies["states"][0].to(
                device=img.device
            )
        if self.precise_math:
            transformer_options["vimoo_math"] = (self.qk_norm_math, self.rope_math, self.modulate_math)
        if transformer_options.get("vimoo_full_i2v"):
            transformer_options["vimoo_i2v_math"] = (self.gate_math, self.rope_pair_math, self.fa2_pack_math)
        patches_replace = transformer_options.get("patches_replace", {})
        retained_fa2_blocks = transformer_options.get("vimoo_sage_fa2_blocks", ())
        if retained_fa2_blocks and (transformer_options.get("patches") or patches_replace):
            raise ValueError("FA2 block retention does not support attention patches or block replacements.")

        initial_shape = list(img.shape)
        # running on sequences img
        img = self.img_in(img)
        vec, block_vec, block_modulations, final_modulation, text_time = self._prepare_time_modulation(
            img, timesteps, initial_shape, observed_first_frame_timesteps, transformer_options
        )

        blocks_replace = patches_replace.get("dit", {})
        if txt_mask is not None:
            txt_mask = (txt_mask.to(img.dtype) - 1) * torch.finfo(img.dtype).max

        refiner_mask = txt_mask if self.params.use_text_refiner_attention_mask else None
        txt = self.txt_in(
            txt,
            timesteps,
            refiner_mask,
            transformer_options=transformer_options,
            precomputed_time=text_time,
        )
        if txt_mask is None:
            # Glyph padding and absent vision tokens still need a joint mask
            # when all Qwen tokens are valid or no Qwen mask was supplied.
            txt_mask = img.new_zeros(txt.shape[:2])

        if self.cond_type_embedding is not None:
            cond_emb = self.cond_type_embedding(
                torch.zeros_like(txt[:, :, 0], device=txt.device, dtype=torch.long), out_dtype=txt.dtype
            )
            txt = txt + cond_emb.to(txt.dtype)

        txt, txt_ids, txt_mask = self._append_glyph_tokens(
            txt, txt_ids, txt_mask, txt_byt5, byt5_mask, transformer_options
        )
        txt, txt_ids, txt_mask = self._append_vision_tokens(
            img, txt, txt_ids, txt_mask, clip_fea, vision_mask, transformer_options
        )

        if transformer_options.get("vimoo_lite_ti2v_anchor"):
            # Pack both streams into one FP32 tensor before the Transformer blocks.
            # Keep this promotion even though ComfyUI blocks carry two tensors.
            img, txt = img.float(), txt.float()

        if (
            txt_mask is not None
            and transformer_options.get("vimoo_drop_text_padding")
            and txt.shape[0] == 1
            and not blocks_replace
            and not transformer_options.get("patches")
        ):
            # Padded text tokens are masked as keys and never reach the decoded
            # image stream. Removing them makes the joint attention mask-free.
            valid = txt_mask[0] == 0
            if not bool(valid.all()):
                order = torch.argsort(valid.int(), descending=True, stable=True)
                keep = order[: int(valid.sum())]
                txt = txt[:, keep]
                if txt_ids is not None:
                    txt_ids = txt_ids[:, keep]
            txt_mask = None

        if (
            precise_rope is not None
            and not patches_replace
            and not transformer_options.get("patches")
            and precise_rope["states"].shape[-2] == img.shape[1]
        ):
            states = precise_rope["states"].to(device=img.device)
            pe = (states[0, 0], states[0, 1])
        else:
            if txt_ids is None or img_ids is None:
                raise ValueError("Vimoo positional IDs are required without a matching reference RoPE")
            ids = torch.cat((txt_ids, img_ids), dim=1)
            pe = self.pe_embedder(ids, latent_shape, fps)

        attn_mask, patch_attn_mask = self._joint_attention_masks(
            img, txt, txt_mask, blocks_replace, transformer_options
        )

        img, txt = self._forward_blocks(
            img,
            txt,
            block_vec,
            pe,
            attn_mask,
            patch_attn_mask,
            blocks_replace,
            block_modulations,
            control,
            transformer_options,
        )

        img = self.final_layer(
            img,
            vec,
            modulate_function=self.modulate_math if self.precise_math else None,
            precomputed_modulation=final_modulation,
            **(
                {"preserve_fp32": bool(transformer_options.get("vimoo_lite_ti2v_anchor"))} if self.hms else {}
            ),
        )

        return self._unpatchify(img, initial_shape, transformer_options)

    def img_ids(self, x):
        bs, c, t, h, w = x.shape
        patch_size = self.patch_size
        axes = [
            torch.arange(size // patch, dtype=torch.float32, device=x.device) * patch
            for size, patch in zip((t, h, w), patch_size)
        ]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        return repeat(grid, "t h w c -> b (t h w) c", b=bs)

    def forward(
        self,
        x,
        timestep,
        context,
        y=None,
        txt_byt5=None,
        clip_fea=None,
        attention_mask=None,
        control=None,
        transformer_options=None,
        **kwargs,
    ):
        transformer_options = transformer_options or {}
        if isinstance(attention_mask, dict):
            attention_mask = attention_mask["states"].to(device=x.device)
        for name in ("byt5_mask", "vision_mask"):
            if isinstance(kwargs.get(name), dict):
                kwargs[name] = kwargs[name]["states"].to(device=x.device)
        executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options
            ),
        )
        return executor.execute(
            x,
            timestep,
            context,
            txt_byt5,
            clip_fea,
            attention_mask,
            control,
            transformer_options,
            **kwargs,
        )

    def _forward(
        self,
        x,
        timestep,
        context,
        txt_byt5=None,
        clip_fea=None,
        attention_mask=None,
        control=None,
        transformer_options=None,
        **kwargs,
    ):
        if isinstance(txt_byt5, dict):
            txt_byt5 = txt_byt5["states"].to(device=x.device)
            if not torch.is_autocast_enabled(x.device.type):
                txt_byt5 = txt_byt5.to(x.dtype)
        original_shape = x.shape[2:]
        observed_time = kwargs.get("observed_first_frame_timesteps")
        if isinstance(observed_time, dict):
            observed_time = observed_time["states"].to(device=x.device)
        first_frame = kwargs.get("vimoo_anchor_first_frame")
        if first_frame is not None:
            x = x.clone()
            x[:, : self.out_channels, :1] = first_frame.to(x)
        x = comfy.ldm.common_dit.pad_to_patch_size(
            x, self.patch_size, padding_mode="constant" if self.hms else "circular"
        )
        transformer_options = transformer_options or {}
        precise_rope = kwargs.get("precise_rope")
        reference_states = precise_rope.get("states") if isinstance(precise_rope, dict) else None
        image_length = math.prod(size // patch for size, patch in zip(x.shape[2:], self.patch_size))
        fast_precise_rope = (
            reference_states is not None
            and not transformer_options.get("patches_replace")
            and not transformer_options.get("patches")
            and reference_states.shape[-2] == image_length
        )
        img_ids = None if fast_precise_rope else self.img_ids(x)
        txt_ids = (
            None
            if fast_precise_rope
            else torch.zeros((x.shape[0], context.shape[1], 3), device=x.device, dtype=x.dtype)
        )
        out = self.forward_orig(
            x,
            img_ids,
            context,
            txt_ids,
            attention_mask,
            timestep,
            txt_byt5=txt_byt5,
            clip_fea=clip_fea,
            control=control,
            transformer_options=transformer_options,
            byt5_mask=kwargs.get("byt5_mask"),
            precise_rope=precise_rope,
            precomputed_frequencies=kwargs.get("precomputed_frequencies"),
            latent_shape=original_shape,
            fps=kwargs.get("fps"),
            observed_first_frame_timesteps=observed_time,
            vision_mask=kwargs.get("vision_mask"),
        )
        out = out[(slice(None), slice(None), *(slice(size) for size in original_shape))]
        if transformer_options.get("vimoo_lite_ti2v_anchor"):
            out = out.float()
        return out
