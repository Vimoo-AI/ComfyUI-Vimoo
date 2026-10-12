"""Release recipes; performance entries override the typed shared defaults."""

RELEASE_RECIPES = {
    "i2v": {
        "vae": {
            "reference": {
                "decode_mode": "auto",
                "convolution_backend": "im2col_bounded",
                "tile_height": 576,
                "tile_width": 512,
                "stride_height": 512,
                "stride_width": 448,
                "im2col_workspace_limit_gib": 2.0,
            },
            "output": {
                "decode_mode": "tiled",
                "convolution_backend": "comfy",
                "tile_height": 576,
                "tile_width": 512,
                "stride_height": 512,
                "stride_width": 448,
                "im2col_workspace_limit_gib": 2.0,
            },
        },
        "sampling": {
            "VimooTextEncoderLoader": {
                "qwen_name": "vimoo_v1_qwen2_5_vl_7b_bf16.safetensors",
                "byt5_name": "vimoo_v1_glyph_byt5_fp32.safetensors",
                "attention_backend": "precise",
                "precision_profile": "bf16_transposed_ffn",
            },
            "VimooTI2VSchedule": {
                "steps": 50,
                "flow_shift": 12.0,
                "t2v_cfg": 4.5,
                "i2v_cfg": 3.0,
                "t2v_rescale": 1.2,
                "i2v_rescale": 0.5,
                "first_frame_replace": True,
                "recipe": "full",
                "full_i2v_flow_shift": 6.0,
            },
            "VimooTI2VConditioning": {
                "width": 1280,
                "height": 720,
                "length": 121,
                "attention_backend": "precise",
            },
            "VimooSampler": {
                "seed": 1234,
                "denoise": 1,
                "renoise_seed": 1001237,
                "add_fps": True,
                "reference_guidance_scale": 3.0,
                "i2v_anchor_tau": 0.001,
            },
        },
        "performance": {
            "compile": "disabled",
            "precise_math": True,
        },
    },
    "i2v_distilled": {
        "vae": {
            "reference": {
                "decode_mode": "auto",
                "convolution_backend": "comfy",
                "tile_height": 256,
                "tile_width": 256,
                "stride_height": 192,
                "stride_width": 192,
                "im2col_workspace_limit_gib": 2.0,
            },
            "output": {
                "decode_mode": "auto",
                "convolution_backend": "comfy",
                "tile_height": 256,
                "tile_width": 256,
                "stride_height": 192,
                "stride_width": 192,
                "im2col_workspace_limit_gib": 2.0,
            },
        },
        "sampling": {
            "VimooTextEncoderLoader": {
                "qwen_name": "vimoo_v1_qwen2_5_vl_7b_bf16.safetensors",
                "byt5_name": "vimoo_v1_glyph_byt5_fp32.safetensors",
                "attention_backend": "precise",
                "precision_profile": "bf16_transposed_ffn",
            },
            "VimooScheduler": {"scheduler": "i2v_distilled", "steps": 8, "flow_shift": 6.0},
            "VimooImageToVideo": {
                "width": 1280,
                "height": 720,
                "length": 121,
                "first_frame_replace": True,
                "attention_backend": "precise",
            },
            "VimooSampler": {
                "cfg": 1.0,
                "guidance_rescale": 0.0,
                "seed": 1234,
                "sampler_name": "vimoo_i2v_euler",
                "denoise": 1.0,
                "renoise_seed": 1001237,
                "fps": 24.0,
                "add_fps": True,
                "reference_guidance_scale": 1.0,
                "i2v_anchor_tau": 0.001,
            },
        },
        "performance": {
            "compile": "disabled",
            "precise_math": True,
        },
    },
    "t2v": {
        "vae": {
            "reference": {
                "decode_mode": "auto",
                "convolution_backend": "im2col_bounded",
                "tile_height": 576,
                "tile_width": 512,
                "stride_height": 512,
                "stride_width": 448,
                "im2col_workspace_limit_gib": 2.0,
            },
            "output": {
                "decode_mode": "tiled",
                "convolution_backend": "comfy",
                "tile_height": 576,
                "tile_width": 512,
                "stride_height": 512,
                "stride_width": 448,
                "im2col_workspace_limit_gib": 2.0,
            },
        },
        "sampling": {
            "VimooTextEncoderLoader": {
                "qwen_name": "vimoo_v1_qwen2_5_vl_7b_bf16.safetensors",
                "byt5_name": "vimoo_v1_glyph_byt5_fp32.safetensors",
                "attention_backend": "precise",
                "precision_profile": "bf16_transposed_ffn",
            },
            "VimooTI2VSchedule": {
                "steps": 50,
                "flow_shift": 12.0,
                "t2v_cfg": 4.5,
                "i2v_cfg": 3.0,
                "t2v_rescale": 1.2,
                "i2v_rescale": 0.5,
                "first_frame_replace": True,
                "recipe": "full",
                "full_i2v_flow_shift": 6.0,
            },
            "VimooTI2VConditioning": {
                "width": 1280,
                "height": 720,
                "length": 121,
                "attention_backend": "precise",
            },
            "VimooSampler": {
                "seed": 1234,
                "denoise": 1,
                "renoise_seed": 1001237,
                "add_fps": True,
                "reference_guidance_scale": 3.0,
                "i2v_anchor_tau": 0.001,
            },
        },
        "performance": {
            "compile": "disabled",
            "precise_math": True,
        },
    },
    "t2v_distilled": {
        "vae": {
            "reference": {
                "decode_mode": "auto",
                "convolution_backend": "comfy",
                "tile_height": 256,
                "tile_width": 256,
                "stride_height": 192,
                "stride_width": 192,
                "im2col_workspace_limit_gib": 2.0,
            },
            "output": {
                "decode_mode": "auto",
                "convolution_backend": "comfy",
                "tile_height": 256,
                "tile_width": 256,
                "stride_height": 192,
                "stride_width": 192,
                "im2col_workspace_limit_gib": 2.0,
            },
        },
        "sampling": {
            "VimooTextEncoderLoader": {
                "qwen_name": "vimoo_v1_qwen2_5_vl_7b_bf16.safetensors",
                "byt5_name": "vimoo_v1_glyph_byt5_fp32.safetensors",
                "attention_backend": "precise",
                "precision_profile": "bf16_transposed_ffn",
            },
            "VimooScheduler": {"scheduler": "t2v_distilled", "steps": 8, "flow_shift": 12.0},
            "VimooSampler": {
                "cfg": 1.0,
                "guidance_rescale": 0.0,
                "seed": 1234,
                "sampler_name": "stochastic_flowmatch",
                "denoise": 1.0,
                "renoise_seed": 1001237,
                "fps": 24.0,
                "add_fps": True,
                "reference_guidance_scale": 1.0,
                "i2v_anchor_tau": 0.001,
            },
        },
        "performance": {
            "compile": "disabled",
            "precise_math": True,
        },
    },
    "ti2v_lite": {
        "vae": {
            "reference": {
                "decode_mode": "auto",
                "convolution_backend": "im2col_bounded",
                "tile_height": 256,
                "tile_width": 256,
                "stride_height": 192,
                "stride_width": 192,
                "im2col_workspace_limit_gib": 2.0,
            },
            "output": {
                "decode_mode": "tiled",
                "convolution_backend": "im2col_bounded",
                "tile_height": 256,
                "tile_width": 256,
                "stride_height": 192,
                "stride_width": 192,
                "im2col_workspace_limit_gib": 2.0,
            },
        },
        "sampling": {
            "VimooTextEncoderLoader": {
                "qwen_name": "vimoo_v1_qwen2_5_vl_7b_bf16.safetensors",
                "byt5_name": "vimoo_v1_glyph_byt5_fp32.safetensors",
                "attention_backend": "precise",
                "precision_profile": "lite_ti2v_anchor",
            },
            "VimooTI2VSchedule": {
                "steps": 50,
                "flow_shift": 8.0,
                "t2v_cfg": 4.0,
                "i2v_cfg": 5.0,
                "t2v_rescale": 0.85,
                "i2v_rescale": 0.85,
                "first_frame_replace": False,
                "recipe": "lite",
                "full_i2v_flow_shift": 6.0,
            },
            "VimooTI2VConditioning": {
                "width": 1280,
                "height": 720,
                "length": 121,
                "attention_backend": "precise",
            },
            "VimooSampler": {
                "seed": 1234,
                "denoise": 1.0,
                "renoise_seed": 1001237,
                "add_fps": True,
                "i2v_anchor_tau": 0.001,
            },
        },
        "performance": {
            "compile": "disabled",
        },
    },
    "ti2v_lite_distilled": {
        "vae": {
            "reference": {
                "decode_mode": "auto",
                "convolution_backend": "comfy",
                "tile_height": 256,
                "tile_width": 256,
                "stride_height": 192,
                "stride_width": 192,
                "im2col_workspace_limit_gib": 2.0,
            },
            "output": {
                "decode_mode": "tiled",
                "convolution_backend": "comfy",
                "tile_height": 512,
                "tile_width": 512,
                "stride_height": 448,
                "stride_width": 448,
                "im2col_workspace_limit_gib": 2.0,
            },
        },
        "sampling": {
            "VimooTextEncoderLoader": {
                "qwen_name": "vimoo_v1_qwen2_5_vl_7b_bf16.safetensors",
                "byt5_name": "vimoo_v1_glyph_byt5_fp32.safetensors",
                "attention_backend": "precise",
                "precision_profile": "lite_ti2v_anchor",
            },
            "VimooTI2VSchedule": {
                "steps": 12,
                "flow_shift": 12.0,
                "t2v_cfg": 1.0,
                "i2v_cfg": 1.0,
                "t2v_rescale": 0.0,
                "i2v_rescale": 0.0,
                "first_frame_replace": False,
                "recipe": "lite",
                "full_i2v_flow_shift": 6.0,
            },
            "VimooTI2VConditioning": {
                "width": 1280,
                "height": 720,
                "length": 121,
                "attention_backend": "precise",
            },
            "VimooSampler": {
                "cfg": 1.0,
                "guidance_rescale": 0.0,
                "seed": 1234,
                "sampler_name": "vimoo_lite_ti2v_euler",
                "denoise": 1.0,
                "renoise_seed": 1001237,
                "fps": 24.0,
                "add_fps": True,
                "reference_guidance_scale": 1.0,
                "i2v_anchor_tau": 0.001,
            },
        },
        "performance": {
            "compile": "disabled",
            "precise_math": True,
            "text_padding": "drop",
            "fused_qkv": True,
            "sage_image_only": False,
        },
    },
}


def release_profile(model):
    distilled = model.model_options.get("vimoo_distilled")
    if distilled:
        name = distilled["profile"]
        return "ti2v_lite_distilled" if name == "lite_ti2v_distilled" else name
    name = model.model_options.get("vimoo_sampling_profile")
    profiles = {
        "standard_t2v": "t2v",
        "standard_i2v": "i2v",
        "lite_ti2v_anchor": "ti2v_lite",
        "t2v_distilled": "t2v_distilled",
        "i2v_distilled": "i2v_distilled",
    }
    if name not in profiles:
        raise ValueError(f"Model has no current Vimoo release profile: {name!r}")
    return profiles[name]
