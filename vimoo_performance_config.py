"""Typed internal performance settings, independent of ComfyUI widget schemas."""

from dataclasses import dataclass, field, fields


ATTENTION_BACKENDS = ("pytorch", "fa2", "fa3", "fa4", "sage")


@dataclass(frozen=True)
class PerformanceConfig:
    """Complete effective settings for a model; public controls are resolved separately."""

    attention: str = field(
        default="pytorch",
        metadata={"choices": ATTENTION_BACKENDS, "noun": "attention backend"},
    )
    compile: str = field(
        default="math_helpers",
        metadata={"choices": ("math_helpers", "regional", "disabled"), "noun": "compile scope"},
    )
    compile_mode: str = field(
        default="default",
        metadata={
            "choices": ("default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"),
            "noun": "compile mode",
        },
    )
    block_prefetch: bool = True
    compile_dynamic: str = field(
        default="auto", metadata={"choices": ("auto", "static", "dynamic"), "noun": "compile shape policy"}
    )
    text_padding: str = field(
        default="keep", metadata={"choices": ("keep", "drop"), "noun": "text_padding policy"}
    )
    fused_qkv: bool = False
    memory_usage_factor: float = 4.0
    sage_scope: str = field(
        default="joint", metadata={"choices": ("joint", "text_refiner", "all"), "noun": "SageAttention scope"}
    )
    sage_fallback: str = field(
        default="pytorch", metadata={"choices": ("fa4", "fa2", "pytorch"), "noun": "SageAttention fallback"}
    )
    sage_qk_quant_gran: str = field(
        default="per_thread",
        metadata={"choices": ("per_warp", "per_thread"), "noun": "SageAttention QK quantization granularity"},
    )
    # SM89's single-level FP8 PV accumulation loses large V outliers; the
    # two-level mode matches Hopper's SM90 kernel, which serves it there.
    sage_pv_accum_dtype: str = field(
        default="fp32+fp32",
        metadata={"choices": ("fp16", "fp16+fp32", "fp32", "fp32+fp32", "fp32+fp16"), "noun": None},
    )
    sage_smooth_k: bool = True
    sage_smooth_v: bool = False
    sage_pv_dtype: str = field(default="fp8", metadata={"choices": ("fp8", "fp16"), "noun": None})
    sage_fa2_first_blocks: int = 0
    sage_fa2_last_blocks: int = 0
    sage_image_only: bool = True
    sage_exact_kernel: str = field(
        default="fp32", metadata={"choices": ("fa4", "fp32"), "noun": "SageAttention exact kernel"}
    )
    precise_math: bool = False

    def validate(self):
        """Validate internal selectors without publishing extra node controls."""
        for option in fields(self):
            value = getattr(self, option.name)
            noun = option.metadata.get("noun")
            if noun is not None and value not in option.metadata["choices"]:
                raise ValueError(f"Unknown {noun}: {value!r}")
            if option.type is bool and type(value) is not bool:
                raise ValueError(f"{option.name} must be a Boolean")
