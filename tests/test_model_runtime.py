"""Small Vimoo transformer checks using actual ComfyUI operations on CPU."""

import sys
import json
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from test_nodes_runtime import nodes, spec

model_module = sys.modules[f"{spec.name}.vimoo_model"]
lora_module = sys.modules[f"{spec.name}.vimoo_lora"]


def _tiny_kwargs(cond_type=True, hidden_size=24):
    return dict(
        in_channels=2,
        out_channels=2,
        vec_in_dim=None,
        context_in_dim=6,
        hidden_size=hidden_size,
        mlp_ratio=4.0,
        num_heads=2 if hidden_size == 24 else 1,
        depth=1,
        depth_single_blocks=0,
        axes_dim=[4, 4, 4] if hidden_size == 24 else [16, 56, 56],
        theta=256,
        patch_size=[1, 2, 2],
        qkv_bias=True,
        guidance_embed=False,
        byt5=True,
        meanflow=False,
        use_cond_type_embedding=cond_type,
        vision_in_dim=6,
        meanflow_sum=False,
        byt5_dims=[6, 8, 8, hidden_size],
        dtype=torch.float32,
        device=torch.device("cpu"),
        operations=nodes.comfy.ops.manual_cast,
    )


def tiny_vimoo(cond_type=True, hidden_size=24):
    model = model_module.VimooModel(**_tiny_kwargs(cond_type, hidden_size))
    with torch.no_grad():
        generator = torch.Generator("cpu").manual_seed(42)
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
    return model


def packed_tiny_copy(model):
    packed = model_module.VimooModel(**{**_tiny_kwargs(), "packed_qkv": True}).eval()
    state = dict(model.state_dict())
    for stream in ("img", "txt"):
        prefix = f"double_blocks.0.{stream}_attn."
        for suffix in ("weight", "bias"):
            state[prefix + "qkv." + suffix] = torch.cat(
                tuple(state.pop(prefix + f"to_{p}." + suffix) for p in "qkv")
            )
    packed.load_state_dict(state, strict=True)
    return packed


class ModelRuntimeTests(unittest.TestCase):
    def test_patchify_bool_masks_reach_native_forward_through_framework_conditioning(self):
        support = sys.modules[f"{spec.name}.vimoo_support"]
        sampling_class = type(
            "NativeFlowSampling",
            (nodes.comfy.model_sampling.ModelSamplingDiscreteFlow, nodes.comfy.model_sampling.CONST),
            {},
        )

        class ReachedNativeForward(RuntimeError):
            pass

        for profile, dtype in (("lite_compact", torch.bool), (None, torch.bool), (None, torch.int64)):
            with self.subTest(profile=profile, dtype=dtype):
                model = tiny_vimoo().eval()
                adapter = support.VimooBaseModel.__new__(support.VimooBaseModel)
                torch.nn.Module.__init__(adapter)
                adapter.diffusion_model = model
                adapter.manual_cast_dtype = torch.bfloat16
                adapter.current_patcher = None
                patcher = nodes.comfy.model_patcher.ModelPatcher(
                    adapter, torch.device("cpu"), torch.device("cpu")
                )
                patcher.model_options["vimoo_text_encoder_precision_profile"] = profile
                patcher.pre_run()
                self.assertIs(adapter.current_patcher, patcher)
                adapter.model_sampling = sampling_class(SimpleNamespace(sampling_settings={"shift": 1.0}))
                adapter.concat_cond = lambda **kwargs: None
                adapter.encode_adm = lambda **kwargs: None
                masks = {
                    "attention_mask": torch.tensor([[1, 0, 1]], dtype=dtype),
                    "vimoo_byt5_mask": torch.tensor([[1, 0]], dtype=dtype),
                    "vimoo_vision_mask": torch.tensor([[1, 0]], dtype=dtype),
                }
                conditions = adapter.extra_conds(**masks)
                inputs = {
                    name: condition.process_cond(batch_size=1).concat([])
                    for name, condition in conditions.items()
                }
                seen = {}
                previous = sys.getprofile()

                def observe(frame, event, result):
                    if previous is not None:
                        previous(frame, event, result)
                    if event == "call" and frame.f_code is model_module.VimooModel._forward.__code__:
                        seen["attention_mask"] = frame.f_locals["attention_mask"]
                        seen.update(
                            {name: frame.f_locals["kwargs"][name] for name in ("byt5_mask", "vision_mask")}
                        )
                        raise ReachedNativeForward

                sys.setprofile(observe)
                try:
                    with torch.inference_mode(), self.assertRaises(ReachedNativeForward):
                        adapter._apply_model(
                            torch.zeros(1, 2, 1, 2, 2),
                            torch.tensor([0.5]),
                            c_crossattn=torch.zeros(1, 3, 6),
                            transformer_options={},
                            **inputs,
                        )
                finally:
                    sys.setprofile(previous)
                    patcher.cleanup()
                    self.assertIsNone(adapter.current_patcher)
                expected_dtype = (
                    dtype if profile == "lite_compact" or dtype == torch.int64 else torch.bfloat16
                )
                for source, target in (
                    ("attention_mask", "attention_mask"),
                    ("vimoo_byt5_mask", "byt5_mask"),
                    ("vimoo_vision_mask", "vision_mask"),
                ):
                    self.assertEqual(seen[target].dtype, expected_dtype)
                    self.assertEqual(seen[target].tolist(), masks[source].tolist())
                    if profile == "lite_compact":
                        self.assertIsInstance(conditions[target], support.VimooPrecisionCondition)

    def test_patchify_mask_contract_rejects_a_legacy_precision_profile(self):
        support = sys.modules[f"{spec.name}.vimoo_support"]
        adapter = support.VimooBaseModel.__new__(support.VimooBaseModel)
        torch.nn.Module.__init__(adapter)
        adapter.diffusion_model = tiny_vimoo().eval()
        adapter.current_patcher = SimpleNamespace(
            model_options={"vimoo_text_encoder_precision_profile": "lite_compact"}
        )
        adapter.concat_cond = lambda **kwargs: None
        adapter.encode_adm = lambda **kwargs: None
        for name in ("attention_mask", "vimoo_byt5_mask", "vimoo_vision_mask"):
            with self.subTest(mask=name), self.assertRaisesRegex(ValueError, "BOOL"):
                adapter.extra_conds(**{name: torch.ones(1, 2, dtype=torch.int64)})

    def test_patchify_flash_capacity_reaches_actual_double_block_without_affecting_legacy(self):
        model = tiny_vimoo().eval()

        class ObservedBlock(RuntimeError):
            pass

        seen = []

        def observe(module, args, kwargs):
            seen.append(dict(kwargs["transformer_options"]))
            raise ObservedBlock

        handle = model.double_blocks[0].register_forward_pre_hook(observe, with_kwargs=True)
        try:
            for padded_capacity in (None, True):
                options = {
                    "vimoo_symmetric_self_attention": True,
                }
                if padded_capacity is not None:
                    options["vimoo_flash_padded_capacity"] = padded_capacity
                with torch.inference_mode(), self.assertRaises(ObservedBlock):
                    model(
                        torch.randn(1, 2, 1, 2, 2),
                        torch.tensor([500.0]),
                        torch.randn(1, 3, 6),
                        transformer_options=options,
                    )
        finally:
            handle.remove()
        self.assertNotIn("vimoo_flash_padded_capacity", seen[0])
        self.assertTrue(seen[1]["vimoo_flash_padded_capacity"])
        for options in seen:
            self.assertTrue(options["vimoo_symmetric_self_attention"])

    def test_sage_image_only_keeps_regional_compile_options_stable(self):
        model = tiny_vimoo().eval()
        model.double_blocks = torch.nn.ModuleList([model.double_blocks[0]] * 5)
        seen = []

        def record(module, args, kwargs):
            options = kwargs["transformer_options"]
            self.assertNotIn("vimoo_attention_metadata", options)
            seen.append(options["vimoo_sage_text_length"])

        handle = model.double_blocks[0].register_forward_pre_hook(record, with_kwargs=True)
        try:
            with torch.inference_mode():
                model(
                    torch.randn(1, 2, 1, 2, 2),
                    torch.tensor([500.0]),
                    torch.randn(1, 3, 6),
                    transformer_options={
                        "vimoo_sage_image_only": True,
                        "vimoo_block_prefetch": False,
                        "vimoo_drop_text_padding": True,
                        "vimoo_attention_metadata": {"old_mask": object()},
                    },
                )
        finally:
            handle.remove()
        self.assertEqual(len(seen), 5)
        self.assertEqual(seen, [3] * 5)

    def test_sage_retained_blocks_route_with_two_stable_options_and_shared_cache(self):
        model = tiny_vimoo().eval()
        model.double_blocks = torch.nn.ModuleList([model.double_blocks[0]] * 5)
        x, context = torch.randn(1, 2, 1, 2, 2), torch.randn(1, 3, 6)
        timestep = torch.tensor([500.0])
        for retained in (
            frozenset({0, 1}),
            frozenset({3, 4}),
            frozenset({0, 1, 4}),
            frozenset(range(5)),
        ):
            seen, routes, metadata = [], [], {}

            def sage(original, *args, **kwargs):
                routes.append("sage")
                return original(*args, **kwargs)

            def fa2(original, *args, **kwargs):
                routes.append("fa2")
                return original(*args, **kwargs)

            options = {
                "optimized_attention_override": sage,
                "vimoo_sage_fa2_blocks": retained,
                "vimoo_sage_fa2_override": fa2,
                "vimoo_attention_metadata": metadata,
                "vimoo_block_prefetch": False,
            }

            def record(module, args, kwargs):
                seen.append(kwargs["transformer_options"])

            handle = model.double_blocks[0].register_forward_pre_hook(record, with_kwargs=True)
            try:
                with torch.inference_mode():
                    actual = model(x, timestep, context, transformer_options=options)
            finally:
                handle.remove()
            self.assertTrue(torch.isfinite(actual).all())
            self.assertEqual(len(seen), 5)
            for index, block_options in enumerate(seen):
                expected = fa2 if index in retained else sage
                self.assertIs(block_options["optimized_attention_override"], expected)
                self.assertIs(block_options["vimoo_attention_metadata"], metadata)
            self.assertEqual(
                routes,
                ["sage", "sage"] + ["fa2" if index in retained else "sage" for index in range(5)],
            )

    def test_regional_options_exclude_eager_tensor_cache_and_preserve_patches(self):
        model = tiny_vimoo()
        cache = {"old_mask": torch.ones(5)}
        patches = {"custom": [object()]}
        options = {
            "vimoo_math_graph": True,
            "vimoo_attention_metadata": cache,
            "patches": patches,
            "vimoo_sage_fa2_override": object(),
        }
        direct, retained = model._block_options(options, torch.ones(1, 1, 5), 5, {0})
        self.assertNotIn("vimoo_attention_metadata", direct)
        self.assertNotIn("vimoo_attention_metadata", retained)
        self.assertIs(direct["patches"], patches)
        self.assertIs(options["vimoo_attention_metadata"], cache)

    def test_sage_retained_blocks_reject_patches_added_after_configuration(self):
        model = tiny_vimoo().eval()
        x, context = torch.randn(1, 2, 1, 2, 2), torch.randn(1, 3, 6)
        for extra in (
            {"patches": {"attn1_patch": [lambda *args: args]}},
            {"patches_replace": {"dit": {("double_block", 0): object()}}},
        ):
            options = {
                "vimoo_sage_fa2_blocks": frozenset({0}),
                "vimoo_sage_fa2_override": lambda original, *args, **kwargs: original(*args, **kwargs),
                **extra,
            }
            with self.subTest(extra=extra), torch.inference_mode():
                with self.assertRaisesRegex(ValueError, "patch"):
                    model(x, torch.tensor([500.0]), context, transformer_options=options)

    def test_block_prefetch_switch_survives_core_capability_flag(self):
        model = tiny_vimoo()
        x, context = torch.randn(1, 2, 1, 2, 2), torch.randn(1, 3, 6)
        timestep = torch.tensor([500.0])
        outputs = []
        # BaseModel sets the native capability flag immediately before forward.
        # The plugin's user choice must still suppress the actual queue.
        for enabled in (False, True):
            options = {"prefetch_dynamic_vbars": True, "vimoo_block_prefetch": enabled}
            with (
                torch.inference_mode(),
                mock.patch.object(
                    model_module.comfy.model_prefetch, "make_prefetch_queue", return_value=None
                ) as make_queue,
            ):
                outputs.append(model(x, timestep, context, transformer_options=options))
            self.assertEqual(make_queue.call_count, int(enabled))
            self.assertEqual(options["vimoo_block_prefetch"], enabled)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)

    def test_split_qkv_supports_native_attention_patches(self):
        model = tiny_vimoo()
        seen = []

        def attention_patch(q, k, v, **kwargs):
            seen.append(q.shape)
            return {"q": q, "k": k, "v": v}

        x = torch.randn(1, 2, 1, 2, 2)
        with torch.inference_mode():
            output = model(
                x,
                torch.tensor([500.0]),
                torch.randn(1, 3, 6),
                transformer_options={"patches": {"attn1_patch": [attention_patch]}},
            )
        self.assertEqual(len(seen), 1)
        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_packed_qkv_option_is_checked_and_numerically_equivalent(self):
        model = tiny_vimoo().eval()
        packed = packed_tiny_copy(model)
        generator = torch.Generator().manual_seed(123)
        x = torch.randn(1, 2, 1, 2, 2, generator=generator)
        context = torch.randn(1, 3, 6, generator=generator)
        timestep = torch.tensor([500.0])
        with torch.inference_mode():
            expected = model(x, timestep, context)
            actual = packed(
                x,
                timestep,
                context,
                transformer_options={"vimoo_fused_qkv": True},
            )
            with self.assertRaisesRegex(ValueError, "qkv_layout=packed"):
                model(
                    x,
                    timestep,
                    context,
                    transformer_options={"vimoo_fused_qkv": True},
                )
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_fused_qkv_keeps_runtime_lora_unmerged(self):
        model = tiny_vimoo().eval()
        packed = packed_tiny_copy(model)
        generator = torch.Generator().manual_seed(77)
        for stream in ("img", "txt"):
            attn = getattr(model.double_blocks[0], stream + "_attn")
            packed_attn = getattr(packed.double_blocks[0], stream + "_attn")
            pairs = []
            for part in "qkv":
                base = getattr(attn, f"to_{part}")
                a = torch.randn(3, base.in_features, generator=generator) * 0.05
                b = torch.randn(base.out_features, 3, generator=generator) * 0.05
                setattr(attn, f"to_{part}", lora_module.LoRALinear(base, a, b))
                pairs.append((a, b))
            packed_attn.qkv = lora_module.PackedLoRALinear(packed_attn.qkv, pairs)
        x = torch.randn(1, 2, 1, 2, 2, generator=generator)
        context = torch.randn(1, 3, 6, generator=generator)
        with torch.inference_mode():
            expected = model(x, torch.tensor([500.0]), context)
            actual = packed(
                x,
                torch.tensor([500.0]),
                context,
            )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_block_replacement_preserves_original_block_call(self):
        model = tiny_vimoo()
        x, context = torch.randn(1, 2, 1, 2, 2), torch.randn(1, 3, 6)
        timestep = torch.tensor([500.0])
        replacement = mock.Mock(side_effect=lambda args, extra: extra["original_block"](args))
        with torch.inference_mode():
            expected = model(x, timestep, context)
            actual = model(
                x,
                timestep,
                context,
                transformer_options={
                    "patches_replace": {"dit": {("double_block", 0): replacement}},
                },
            )
        replacement.assert_called_once()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_released_fused_qkv_is_split_without_copying_storage(self):
        weight = torch.arange(48, dtype=torch.float32).reshape(12, 4)
        bias = torch.arange(12, dtype=torch.float32)
        state = {
            "double_blocks.0.img_attn.qkv.weight": weight,
            "double_blocks.0.img_attn.qkv.bias": bias,
        }
        nodes._split_double_block_qkv_state_dict(state)
        self.assertNotIn("double_blocks.0.img_attn.qkv.weight", state)
        for index, part in enumerate("qkv"):
            split_weight = state[f"double_blocks.0.img_attn.to_{part}.weight"]
            split_bias = state[f"double_blocks.0.img_attn.to_{part}.bias"]
            torch.testing.assert_close(split_weight, weight[index * 4 : (index + 1) * 4], rtol=0, atol=0)
            torch.testing.assert_close(split_bias, bias[index * 4 : (index + 1) * 4], rtol=0, atol=0)
            self.assertEqual(split_weight.untyped_storage().data_ptr(), weight.untyped_storage().data_ptr())
            self.assertEqual(split_bias.untyped_storage().data_ptr(), bias.untyped_storage().data_ptr())

    def test_odd_spatial_dimensions_are_padded_and_output_is_cropped(self):
        model = tiny_vimoo()
        x = torch.randn(1, 2, 1, 3, 5)
        context = torch.randn(1, 3, 6)
        timestep = torch.tensor([500.0])
        actual = model(x, timestep, context)
        padded = nodes.comfy.ldm.common_dit.pad_to_patch_size(x, model.patch_size)
        expected = model(padded, timestep, context)[:, :, :, :3, :5]
        self.assertEqual(actual.shape, x.shape)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_masked_text_is_ignored_with_batched_glyph_and_vision_tokens(self):
        for cond_type in [False, True]:
            with self.subTest(cond_type=cond_type):
                model = tiny_vimoo(cond_type)
                x = torch.randn(2, 2, 1, 2, 2)
                context = torch.randn(2, 3, 6)
                mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
                glyphs = torch.randn(2, 2, 6)
                vision = torch.randn(2, 2, 6)
                timestep = torch.tensor([500.0, 500.0])
                output = model(x, timestep, context, txt_byt5=glyphs, clip_fea=vision, attention_mask=mask)
                changed = torch.where(mask[..., None].bool(), context, torch.full_like(context, 100))
                repeated = model(
                    x, timestep, changed, txt_byt5=glyphs, clip_fea=vision, attention_mask=mask.float()
                )
                self.assertTrue(torch.isfinite(output).all())
                torch.testing.assert_close(output, repeated, rtol=1e-5, atol=1e-6)

    def test_padding_and_empty_vision_tokens_keep_the_conditioning_sequence(self):
        model = tiny_vimoo().eval()
        precision = sys.modules[f"{spec.name}.vimoo_precision"]
        original_attention = precision.optimized_attention
        seen = []

        def record(q, k, v, heads, **kwargs):
            seen.append((q.shape[1], kwargs.get("mask")))
            return original_attention(q, k, v, heads, **kwargs)

        with (
            torch.inference_mode(),
            mock.patch.object(precision, "optimized_attention", side_effect=record),
        ):
            model(
                torch.randn(1, 2, 1, 2, 2),
                torch.tensor([500.0]),
                torch.randn(1, 3, 6),
                txt_byt5=torch.randn(1, 2, 6),
                attention_mask=torch.tensor([[1, 1, 0]]),
                byt5_mask=torch.tensor([[1, 0]]),
            )
        length, mask = seen[0]
        # One image + three Qwen + two ByT5 + 729 empty vision tokens.
        self.assertEqual(length, 735)
        self.assertEqual(mask.shape, (1, 1, 735))
        self.assertEqual((mask == 0).sum().item(), 4)

    def test_t2v_caches_unused_vision_projection_without_removing_tokens(self):
        model = tiny_vimoo().eval()
        options = {
            "vimoo_static_conditioning_cache": {},
            "uuids": ["positive"],
            "cond_or_uncond": [0],
        }
        with (
            torch.inference_mode(),
            mock.patch.object(model.vision_in, "forward", wraps=model.vision_in.forward) as vision,
        ):
            x = torch.randn(1, 2, 1, 2, 2)
            context = torch.randn(1, 3, 6)
            first = model(x, torch.tensor([500.0]), context, transformer_options=options)
            second = model(x, torch.tensor([500.0]), context, transformer_options=options)
        self.assertEqual(vision.call_count, 1)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_attention_mask_tensor_is_reused_across_sampling_steps(self):
        model = tiny_vimoo().eval()
        precision = sys.modules[f"{spec.name}.vimoo_precision"]
        original_attention = precision.optimized_attention
        block_masks = []

        def record(q, k, v, heads, **kwargs):
            mask = kwargs.get("mask")
            if mask is not None and mask.shape[-1] == 735:
                block_masks.append(mask)
            return original_attention(q, k, v, heads, **kwargs)

        options = {
            "vimoo_attention_mask_cache": {},
            "uuids": ["positive"],
            "cond_or_uncond": [0],
        }
        x = torch.randn(1, 2, 1, 2, 2)
        context = torch.randn(1, 3, 6)
        glyphs = torch.randn(1, 2, 6)
        mask = torch.tensor([[1, 1, 0]])
        glyph_mask = torch.tensor([[1, 0]])
        outputs = []
        with (
            torch.inference_mode(),
            mock.patch.object(precision, "optimized_attention", side_effect=record),
        ):
            for timestep in (500.0, 250.0):
                outputs.append(
                    model(
                        x,
                        torch.tensor([timestep]),
                        context,
                        txt_byt5=glyphs,
                        attention_mask=mask,
                        byt5_mask=glyph_mask,
                        transformer_options=options,
                    )
                )
        self.assertEqual(len(block_masks), 2)
        self.assertIs(block_masks[0], block_masks[1])
        self.assertEqual(block_masks[0].data_ptr(), block_masks[1].data_ptr())
        self.assertTrue(torch.isfinite(outputs[0]).all())
        self.assertTrue(torch.isfinite(outputs[1]).all())

    def test_concat_condition_allocates_the_absent_mask_as_one_channel(self):
        support = sys.modules[f"{spec.name}.vimoo_support"]
        adapter = support.VimooBaseModel.__new__(support.VimooBaseModel)
        torch.nn.Module.__init__(adapter)
        adapter.diffusion_model = mock.Mock()
        adapter.diffusion_model.img_in.proj.weight = torch.empty(8, 5, 1, 1, 1)
        noise = torch.randn(2, 2, 3, 4, 5)
        with mock.patch.object(
            support.torch,
            "zeros_like",
            side_effect=AssertionError("full latent-shaped mask allocated"),
        ):
            result = adapter.concat_cond(noise=noise, device=noise.device)
        self.assertEqual(result.shape, (2, 3, 3, 4, 5))
        self.assertEqual(result.dtype, noise.dtype)
        self.assertEqual(result.device, noise.device)
        self.assertTrue(torch.count_nonzero(result) == 0)

    def test_precise_rope_cache_also_skips_runtime_position_ids(self):
        model = tiny_vimoo().eval()
        rope, frequencies = model.prepare_math_inputs((1, 2, 1, 2, 2), 3, "cpu")
        cached_rope, cached_frequencies = model.prepare_math_inputs((1, 2, 1, 2, 2), 999, "cpu")
        self.assertIs(rope, cached_rope)
        self.assertIs(frequencies, cached_frequencies)
        model.precise_math = True
        with (
            torch.inference_mode(),
            mock.patch.object(model, "img_ids", side_effect=AssertionError("ids rebuilt")),
        ):
            output = model(
                torch.randn(1, 2, 1, 2, 2),
                torch.tensor([500.0]),
                torch.randn(1, 3, 6),
                precise_rope={"states": rope},
                precomputed_frequencies={"states": frequencies},
            )
        self.assertTrue(torch.isfinite(output).all())

    def test_precomputed_modulation_is_bit_exact_and_skips_modulation_modules(self):
        model = tiny_vimoo().eval()
        x = torch.randn(1, 2, 1, 2, 2)
        context = torch.randn(1, 3, 6)
        timestep = torch.tensor([500.0])
        with torch.inference_mode():
            expected = model(x, timestep, context)
            schedule = model.precompute_modulation_schedule(timestep, batch_sizes=(1,))
            with (
                mock.patch.object(model.time_in, "forward", side_effect=AssertionError("time called")),
                mock.patch.object(
                    model.txt_in.t_embedder,
                    "forward",
                    side_effect=AssertionError("text time called"),
                ),
                mock.patch.object(
                    model.double_blocks[0].img_mod,
                    "forward",
                    side_effect=AssertionError("image modulation called"),
                ),
                mock.patch.object(
                    model.double_blocks[0].txt_mod,
                    "forward",
                    side_effect=AssertionError("text modulation called"),
                ),
                mock.patch.object(
                    model.final_layer.adaLN_modulation,
                    "forward",
                    side_effect=AssertionError("final modulation called"),
                ),
            ):
                actual = model(
                    x,
                    timestep,
                    context,
                    transformer_options={"vimoo_modulation_schedule": schedule},
                )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_all_valid_or_absent_qwen_mask_preserves_glyph_and_vision_masking(self):
        model = tiny_vimoo().eval()
        support = sys.modules[f"{spec.name}.vimoo_support"]
        adapter = support.VimooBaseModel.__new__(support.VimooBaseModel)
        torch.nn.Module.__init__(adapter)
        adapter.diffusion_model = model
        adapter.concat_cond = lambda **kwargs: None
        adapter.encode_adm = lambda **kwargs: None
        mask = torch.ones(1, 3)
        conditioning = adapter.extra_conds(attention_mask=mask)
        self.assertIs(conditioning["attention_mask"].cond, mask)
        generator = torch.Generator().manual_seed(42)
        x = torch.randn(1, 2, 1, 2, 2, generator=generator)
        context = torch.randn(1, 3, 6, generator=generator)
        glyphs = torch.randn(1, 2, 6, generator=generator)
        replacements = torch.randn(1, 2, 6, generator=generator) * 20
        timestep = torch.tensor([500.0])
        for glyph_mask in (torch.zeros(1, 2), torch.tensor([[1, 0]])):
            changed = torch.where(glyph_mask[..., None].bool(), glyphs, replacements)
            for vision in (None, torch.randn(1, 2, 6, generator=generator)):
                with (
                    self.subTest(glyph_mask=glyph_mask.tolist(), vision=vision is not None),
                    torch.inference_mode(),
                ):

                    def predict(glyph, qwen_mask):
                        return model(
                            x,
                            timestep,
                            context,
                            txt_byt5=glyph,
                            byt5_mask=glyph_mask,
                            clip_fea=vision,
                            attention_mask=qwen_mask,
                        )

                    expected = predict(glyphs, mask)
                    for qwen_mask in (mask, None):
                        torch.testing.assert_close(predict(glyphs, qwen_mask), expected, rtol=0, atol=0)
                        torch.testing.assert_close(predict(changed, qwen_mask), expected, rtol=0, atol=0)

    def test_condition_embedding_uses_comfy_cast_operations(self):
        embedding = tiny_vimoo().cond_type_embedding
        self.assertIsInstance(embedding, nodes.comfy.ops.manual_cast.Embedding)
        output = embedding(torch.tensor([0, 1, 2]), out_dtype=torch.bfloat16)
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_loader_rejects_incomplete_weights_and_loads_complete_checkpoint(self):
        source = tiny_vimoo(hidden_size=128)
        weights = dict(source.state_dict())
        # Exercise the on-disk ComfyUI format rather than the split runtime
        # representation created by tiny_vimoo.
        for stream in ("img", "txt"):
            prefix = f"double_blocks.0.{stream}_attn"
            for suffix in ("weight", "bias"):
                pieces = [weights.pop(f"{prefix}.to_{part}.{suffix}") for part in "qkv"]
                weights[f"{prefix}.qkv.{suffix}"] = torch.cat(pieces, dim=0)
        original_get_model = nodes.Vimoo.get_model

        def get_model(config, *args, **kwargs):
            model = original_get_model(config, *args, **kwargs)
            if lazy:
                self.assertIsNone(model.diffusion_model.time_in.in_layer.weight)
            return model

        cases = (
            ("complete", None),
            ("missing", "Incomplete Vimoo transformer"),
            ("unexpected", "Unexpected Vimoo transformer weights"),
            ("shape", "shape mismatch"),
        )
        for lazy, case, error in ((lazy, *case) for lazy in (False, True) for case in cases):
            with self.subTest(lazy=lazy, case=case):
                state = dict(weights)
                if case == "missing":
                    state.pop("time_in.in_layer.bias")
                elif case == "unexpected":
                    state["unused.weight"] = torch.zeros(1)
                elif case == "shape":
                    state["time_in.in_layer.weight"] = state["time_in.in_layer.weight"][:1]
                with (
                    mock.patch.object(nodes.comfy.memory_management, "aimdo_enabled", lazy),
                    mock.patch.object(nodes.Vimoo, "get_model", autospec=True, side_effect=get_model),
                    mock.patch.object(
                        nodes.folder_paths, "get_full_path_or_raise", return_value="tiny.safetensors"
                    ),
                    mock.patch.object(
                        nodes.comfy.utils, "load_torch_file", side_effect=lambda *a, **kw: (dict(state), {})
                    ),
                ):
                    if case == "complete":
                        patcher = nodes.VimooModelLoader().load(
                            "tiny.safetensors", "fp32", disable_dynamic=True
                        )[0]
                        torch.testing.assert_close(
                            patcher.model.diffusion_model.cond_type_embedding.weight,
                            source.cond_type_embedding.weight,
                            rtol=0,
                            atol=0,
                        )
                        self.assertEqual(
                            patcher.model.diffusion_model.time_in.in_layer.weight.data_ptr(),
                            source.time_in.in_layer.weight.data_ptr(),
                        )
                        self.assertTrue(
                            patcher.get_wrappers(
                                nodes.comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
                                "vimoo_modulation_schedule",
                            )
                        )
                        factory, arguments, index = patcher.cached_patcher_init
                        rebuilt = factory(*arguments, disable_dynamic=True)[index]
                        self.assertIsNot(rebuilt.model, patcher.model)
                        torch.testing.assert_close(
                            rebuilt.model.diffusion_model.time_in.in_layer.weight,
                            source.time_in.in_layer.weight,
                            rtol=0,
                            atol=0,
                        )
                    else:
                        with self.assertRaisesRegex(ValueError, error):
                            nodes.VimooModelLoader().load("tiny.safetensors", "fp32", disable_dynamic=True)

    def test_default_loader_accepts_deferred_weights(self):
        source = tiny_vimoo(hidden_size=128)
        weights = dict(source.state_dict())
        original_get_model = nodes.Vimoo.get_model

        def get_model(config, *args, **kwargs):
            model = original_get_model(config, *args, **kwargs)
            self.assertIsNone(model.diffusion_model.time_in.in_layer.weight)
            self.assertIsNone(model.diffusion_model.time_in.in_layer.bias)
            return model

        with (
            mock.patch.object(nodes.comfy.memory_management, "aimdo_enabled", True),
            mock.patch.object(nodes.Vimoo, "get_model", autospec=True, side_effect=get_model),
            mock.patch.object(nodes.folder_paths, "get_full_path_or_raise", return_value="tiny.safetensors"),
            mock.patch.object(
                nodes.comfy.utils, "load_torch_file", side_effect=lambda *a, **kw: (dict(weights), {})
            ),
        ):
            patcher = nodes.VimooModelLoader().load("tiny.safetensors", "fp32")[0]
        actual = patcher.model.diffusion_model
        torch.testing.assert_close(actual.time_in.in_layer.weight, source.time_in.in_layer.weight)
        torch.testing.assert_close(actual.time_in.in_layer.bias, source.time_in.in_layer.bias)

    def test_loader_accepts_current_int8_weights_and_rejects_missing_scales(self):
        weights = dict(tiny_vimoo(hidden_size=128).state_dict())
        prefix = "time_in.in_layer."
        quantized = torch.ones_like(weights[prefix + "weight"], dtype=torch.int8)
        weights[prefix + "weight"] = quantized
        weights[prefix + "weight_scale"] = torch.full((128, 1), 0.5)
        weights[prefix + "comfy_quant"] = torch.tensor(
            list(json.dumps({"format": "int8_tensorwise"}).encode()), dtype=torch.uint8
        )
        for complete in (True, False):
            with self.subTest(complete=complete):
                state = dict(weights)
                if not complete:
                    state.pop(prefix + "weight_scale")
                with (
                    mock.patch.object(
                        nodes.folder_paths, "get_full_path_or_raise", return_value="tiny.safetensors"
                    ),
                    mock.patch.object(
                        nodes.comfy.utils, "load_torch_file", side_effect=lambda *a, **kw: (dict(state), {})
                    ),
                    mock.patch.object(nodes, "checkpoint_provenance", return_value={}),
                ):
                    if complete:
                        patcher = nodes.VimooModelLoader().load("tiny.safetensors", "fp32")[0]
                        layer = patcher.model.diffusion_model.time_in.in_layer
                        torch.testing.assert_close(layer.weight.dequantize(), quantized.float() * 0.5)
                    else:
                        with self.assertRaisesRegex(ValueError, "Missing INT8 weight scale"):
                            nodes.VimooModelLoader().load("tiny.safetensors", "fp32")

    def test_loader_reads_config_and_rejects_inconsistent_metadata(self):
        source = tiny_vimoo(hidden_size=128)
        config = {
            "text_projection": "single_refiner",
            "qk_norm_type": "rms",
            "qk_norm": True,
            "hidden_size": 128,
            "mm_double_blocks_depth": 1,
            "out_channels": 2,
            "patch_size": [1, 2, 2],
            "text_states_dim": 6,
            "in_channels": 2,
            "concat_condition": False,
            "heads_num": 1,
            "rope_dim_list": [16, 56, 56],
            "rope_theta": 256,
            "mlp_width_ratio": 4.0,
            "rope_spatial_interpolation_ref_area": 100.0,
            "rope_spatial_interpolation_strength": 0.5,
        }

        def load():
            metadata = {"vimoo_conversion_version": "1", "vimoo_config": json.dumps(config)}
            with (
                mock.patch.object(
                    nodes.folder_paths, "get_full_path_or_raise", return_value="tiny.safetensors"
                ),
                mock.patch.object(
                    nodes.comfy.utils, "load_torch_file", return_value=(source.state_dict(), metadata)
                ),
            ):
                return nodes.VimooModelLoader().load("tiny.safetensors", "fp32", disable_dynamic=True)[0]

        model = load().model.diffusion_model
        self.assertEqual(model.params.rope_spatial_interpolation_ref_area, 100.0)
        config["hidden_size"] = 256
        with self.assertRaisesRegex(ValueError, "metadata/tensor mismatch"):
            load()

    def test_partial_block_patches_keep_the_mask_layout(self):
        # Direct blocks take the image-first mask; a replaced block sees Comfy's
        # text-first additive ABI.  Mixing the two must not change the output.
        model = model_module.VimooModel(**{**_tiny_kwargs(), "depth": 2}).eval()
        with torch.no_grad():
            generator = torch.Generator("cpu").manual_seed(7)
            for parameter in model.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
        x = torch.randn(2, 2, 1, 4, 4)
        timesteps = torch.tensor([500.0, 500.0])
        context = torch.randn(2, 3, 6)
        extra = dict(
            txt_byt5=torch.randn(2, 2, 6),
            attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
            byt5_mask=torch.tensor([[1, 0], [1, 1]]),
        )
        seen = []

        def passthrough(args, extra_options):
            seen.append(args["attention_mask"])
            return extra_options["original_block"](args)

        with torch.inference_mode():
            expected = model(x, timesteps, context, **extra)
            for index in (0, 1):
                options = {"patches_replace": {"dit": {("double_block", index): passthrough}}}
                actual = model(x, timesteps, context, transformer_options=options, **extra)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual = model(x, timesteps, context, transformer_options={"patches": {"x": []}}, **extra)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        mask = seen[0]
        # Text first (3 Qwen + 2 ByT5 + 729 vision tokens), then 4 unmasked image tokens.
        self.assertEqual(mask.shape, (2, 1, 734 + 4))
        self.assertTrue(torch.all(mask[..., -4:] == 0))
        # T2V masks the 729 empty vision tokens plus one Qwen and one ByT5 pad in item 0.
        self.assertEqual((mask[0] != 0).sum().item(), 729 + 2)
        self.assertEqual((mask[1] != 0).sum().item(), 729)

    def test_runtime_modulation_schedule_skips_patched_models(self):
        patcher = mock.Mock(patches={}, object_patches={})
        full = tiny_vimoo()
        self.assertTrue(nodes._modulation_schedule_is_safe(patcher, full, {}))
        self.assertFalse(nodes._modulation_schedule_is_safe(patcher, full, {"patches": {"x": []}}))
        self.assertFalse(nodes._modulation_schedule_is_safe(patcher, full, {"patches_replace": {"dit": {}}}))
        patcher.patches = {"diffusion_model.double_blocks.0.txt_mod.lin.weight": []}
        self.assertFalse(nodes._modulation_schedule_is_safe(patcher, full, {}))
        # The precompute mirrors the sampler's DIFFUSION_MODEL autocast wrapper.
        samplers = sys.modules[f"{spec.name}.vimoo_sampler"]
        options = {}
        self.assertIsNone(nodes._precompute_autocast_dtype(options))
        nodes.comfy.patcher_extension.add_wrapper(
            nodes.comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            samplers._make_autocast_wrapper(),
            options,
        )
        self.assertEqual(nodes._precompute_autocast_dtype(options), torch.bfloat16)

    def test_dropping_padded_text_tokens_matches_masked_attention_for_the_video(self):
        model = tiny_vimoo()
        x = torch.randn(1, 2, 1, 4, 4)
        context = torch.randn(1, 5, 6)
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        timestep = torch.tensor([500.0])
        with torch.no_grad():
            kept = model(x, timestep, context, attention_mask=mask)
            dropped = model(
                x,
                timestep,
                context,
                attention_mask=mask,
                transformer_options={"vimoo_drop_text_padding": True},
            )
        torch.testing.assert_close(dropped, kept, rtol=1e-4, atol=1e-5)

    def test_vision_helper_encodes_each_batch_item(self):
        vision = sys.modules[f"{spec.name}.vimoo_vision"]
        clip = mock.Mock()
        clip.load_device = torch.device("cpu")
        clip.model.vision_model = torch.nn.Module()
        clip.model.side_effect = lambda pixel_values, **kwargs: (pixel_values.flatten(2).transpose(1, 2),)
        image = torch.stack([torch.zeros(16, 16, 3), torch.ones(16, 16, 3)])
        with mock.patch.object(nodes.comfy.model_management, "load_model_gpu"):
            result = vision.encode_vimoo_vision(clip, image)
        self.assertEqual(result.last_hidden_state.shape[0], 2)
        self.assertFalse(torch.equal(result.last_hidden_state[0], result.last_hidden_state[1]))


class FactorizedModulationTests(unittest.TestCase):
    """The upstream patchify checkpoints ship modulation as a shared low-rank code."""

    def build(self, rank, **overrides):
        kwargs = {**_tiny_kwargs(), "block_modulation_rank": rank, **overrides}
        model = model_module.VimooModel(**kwargs).eval()
        with torch.no_grad():
            generator = torch.Generator("cpu").manual_seed(7)
            for parameter in model.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
        return model

    def test_factorized_model_builds_rank_shaped_modulation(self):
        hidden = _tiny_kwargs()["hidden_size"]
        model = self.build(4)
        self.assertEqual(list(model.mod_projection.weight.shape), [4, hidden])
        self.assertIs(model.mod_projection.weight.dtype, torch.float32)
        for block in model.double_blocks:
            for stream in ("img", "txt"):
                self.assertEqual(list(getattr(block, f"{stream}_mod").lin.weight.shape), [6 * hidden, 4])
        # The dense form is what it replaces, so it must not also be built.
        dense = model_module.VimooModel(**_tiny_kwargs())
        self.assertEqual(list(dense.double_blocks[0].img_mod.lin.weight.shape), [6 * hidden, hidden])
        self.assertIsNone(dense.mod_projection)

    def test_low_rank_modulation_matches_its_dense_materialization(self):
        """W_low @ P is the dense Linear the factorization stands for."""
        hidden = _tiny_kwargs()["hidden_size"]
        model = self.build(4)
        dense = model_module.VimooModel(**_tiny_kwargs()).eval()
        projection = model.mod_projection
        with torch.no_grad():
            for source, target in zip(model.double_blocks, dense.double_blocks):
                for stream in ("img", "txt"):
                    low = getattr(source, f"{stream}_mod").lin
                    full = getattr(target, f"{stream}_mod").lin
                    full.weight.copy_(low.weight.float() @ projection.weight.float())
                    full.bias.copy_(low.bias.float() + low.weight.float() @ projection.bias.float())
        vec = torch.randn(2, hidden, generator=torch.Generator("cpu").manual_seed(3))
        with torch.no_grad():
            got = model.double_blocks[0].precompute_modulation(model.modulation_code(vec))
            want = dense.double_blocks[0].precompute_modulation(dense.modulation_code(vec))
        self.assertEqual(got.shape, want.shape)
        self.assertLess((got - want).abs().max().item(), 1e-4)

    def test_factorized_model_still_precomputes_a_runtime_schedule(self):
        model = self.build(4)
        timesteps = torch.tensor([1000.0, 500.0])
        schedule = model.precompute_modulation_schedule(timesteps, batch_sizes=(1,))
        self.assertEqual(len(schedule["blocks"][1]), len(model.double_blocks))
        self.assertEqual(schedule["blocks"][1][0].shape[0], timesteps.shape[0])

    def test_schedule_stages_shared_projection_before_borrowing_expansion_weights(self):
        """A nested transfer must not reuse the active expansion's borrowed buffer."""
        model = self.build(4, depth=3)
        timesteps = torch.tensor([999.0, 988.0, 974.0, 951.0, 923.0, 880.0, 808.0, 622.0])
        expected = model.precompute_modulation_schedule(timesteps, batch_sizes=(1, 2))
        active_root = None
        projection_rows = []

        def make_queue(roots, device, options):
            return iter([*roots, None]) if options.get("prefetch_dynamic_vbars") else None

        def pop_queue(queue, device, root):
            nonlocal active_root
            if queue is not None:
                self.assertIs(next(queue), root)
                active_root = root

        def check_projection(module, inputs):
            if enabled:
                # CPU cannot fault GPU weights: enforce the borrowed-buffer lifetime
                # contract at the actual projection call instead of staging a transfer.
                self.assertIs(active_root, module, "projection would overwrite a borrowed expansion buffer")
            projection_rows.append(inputs[0].shape[0])

        hook = model.mod_projection.register_forward_pre_hook(check_projection)
        try:
            for enabled in (False, True):
                with self.subTest(prefetch=enabled):
                    projection_rows.clear()
                    active_root = None
                    with (
                        mock.patch.object(
                            model_module.comfy.model_prefetch, "make_prefetch_queue", make_queue
                        ),
                        mock.patch.object(model_module.comfy.model_prefetch, "prefetch_queue_pop", pop_queue),
                    ):
                        actual = model.precompute_modulation_schedule(
                            timesteps, {"prefetch_dynamic_vbars": enabled}, batch_sizes=(1, 2)
                        )
                    self.assertIsNone(active_root)
                    self.assertEqual(projection_rows, [1] * len(timesteps) + [2] * len(timesteps))
                    for batch in (1, 2):
                        for want, got in zip(expected["blocks"][batch], actual["blocks"][batch]):
                            torch.testing.assert_close(got, want, rtol=0, atol=0)
                        for key in ("text_time", "final"):
                            torch.testing.assert_close(
                                actual[key][batch], expected[key][batch], rtol=0, atol=0
                            )
        finally:
            hook.remove()

    def test_schedule_matches_direct_forward_for_dense_and_factorized_precision_profiles(self):
        timesteps = torch.tensor([999.0, 988.0, 974.0, 951.0, 923.0, 880.0, 808.0, 622.0])
        for rank, precision in ((None, "fp32"), (4, "fp32"), (4, "bf16_expand_fp32_bias")):
            with self.subTest(rank=rank, precision=precision):
                model = self.build(rank, depth=3, dtype=torch.bfloat16, block_modulation_precision=precision)
                with torch.inference_mode():
                    schedule = model.precompute_modulation_schedule(timesteps, batch_sizes=(1, 2))
                    for batch in (1, 2):
                        generator = torch.Generator("cpu").manual_seed(13)
                        x = torch.randn(batch, 2, 1, 2, 2, generator=generator, dtype=torch.bfloat16)
                        context = torch.randn(batch, 3, 6, generator=generator, dtype=torch.bfloat16)
                        for timestep in timesteps:
                            inputs = (x, timestep.expand(batch), context)
                            expected = model(*inputs, transformer_options={"vimoo_block_prefetch": False})
                            actual = model(
                                *inputs,
                                transformer_options={
                                    "vimoo_block_prefetch": False,
                                    "vimoo_modulation_schedule": schedule,
                                },
                            )
                            self.assertTrue(torch.isfinite(actual).all())
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class CheckpointDrivenMathTests(unittest.TestCase):
    """Two places where the checkpoint, not ComfyUI's default, decides the arithmetic."""

    def test_mlp_activation_follows_the_checkpoint(self):
        import torch.nn as nn

        default = model_module.VimooModel(**_tiny_kwargs())
        exact = model_module.VimooModel(**{**_tiny_kwargs(), "mlp_act_type": "gelu"})
        self.assertEqual(default.double_blocks[0].img_mlp[1].approximate, "tanh")
        self.assertIsInstance(exact.double_blocks[0].img_mlp[1], nn.GELU)
        self.assertEqual(exact.double_blocks[0].img_mlp[1].approximate, "none")
        with self.assertRaisesRegex(ValueError, "MLP activation"):
            model_module.VimooModel(**{**_tiny_kwargs(), "mlp_act_type": "silu"})

    def test_precomputed_rotary_table_works_without_the_precise_ops(self):
        """A reference table is cos/sin; apply_rope1 cannot read that, so the tuple
        form has to work whether or not the reference math is selected."""
        import sys

        precision = sys.modules[f"{spec.name}.vimoo_precision"]
        torch.manual_seed(0)
        batch, length, heads, dim = 1, 6, 2, 8
        x = torch.randn(batch, length, heads, dim)
        pe = torch.randn(1, 1, length, dim // 2, 2, 2)
        cos = pe[0, 0, ..., 0, 0].repeat_interleave(2, dim=-1).view(1, length, 1, dim)
        sin = pe[0, 0, ..., 1, 0].repeat_interleave(2, dim=-1).view(1, length, 1, dim)
        reference = precision.apply_rope(x, (cos, sin), precision.precise_rope)
        plain = precision.apply_rope(x, (cos, sin))
        self.assertTrue(torch.equal(reference, plain))


if __name__ == "__main__":
    unittest.main()
