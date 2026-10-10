"""Attention ABI and compilation regressions using CPU-sized inputs."""

import importlib.util
from dataclasses import fields
import sys
import types
import unittest
from unittest.mock import patch

import torch
from comfy.cli_args import args

args.cpu = True
import comfy.model_patcher
from comfy.patcher_extension import WrappersMP
from test_nodes_runtime import spec

performance = sys.modules[f"{spec.name}.vimoo_performance"]
# The attention implementations live next door; the node only wires them up.
attention_module = sys.modules[f"{spec.name}.vimoo_attention"]


class PerformanceTests(unittest.TestCase):
    def test_every_enum_widget_rejects_a_value_off_its_own_list(self):
        import re

        for name, field in performance.PerformanceConfig.__dataclass_fields__.items():
            noun = field.metadata.get("noun")
            if noun is None:
                continue
            accepted = field.metadata["choices"]
            with self.subTest(widget=name):
                self.assertNotIn("not-a-choice", accepted)
                arguments = {
                    "attention": "pytorch",
                    "compile": "disabled",
                    "compile_mode": "default",
                    name: "not-a-choice",
                }
                with self.assertRaisesRegex(ValueError, re.escape(f"Unknown {noun}")):
                    performance.apply_performance(
                        self.make_patcher(), performance.PerformanceConfig(**arguments)
                    )

    def test_sage_defaults_keep_exact_math_without_requiring_flash_attention(self):
        config = performance.PerformanceConfig()
        self.assertEqual(config.sage_scope, "joint")
        self.assertEqual(config.sage_fallback, "pytorch")
        self.assertEqual(config.sage_qk_quant_gran, "per_thread")
        self.assertEqual(config.sage_pv_accum_dtype, "fp32")
        self.assertTrue(config.sage_smooth_k)
        self.assertFalse(config.sage_smooth_v)

    def test_sage_image_only_keeps_the_text_quadrants_off_the_kernel(self):
        """The default split sends only image-to-image keys to the INT8 kernel."""
        torch.manual_seed(0)
        batch, heads, head_dim, img_len, txt_len = 1, 2, 32, 12, 5
        length = img_len + txt_len
        q, k, v = (torch.randn(batch, length, heads * head_dim, dtype=torch.float16) for _ in range(3))
        options = {"vimoo_sage_text_length": txt_len, "vimoo_attention_kind": "joint"}
        seen = []

        def kernel(device, **kwargs):
            seen.append(tuple(kwargs[name].shape for name in ("q", "k", "v")))
            out = kwargs["q"].float()
            lse = torch.zeros(out.shape[:-1], dtype=torch.float32)
            return out.to(kwargs["q"].dtype), lse

        with (
            patch.object(torch.Tensor, "is_cuda", property(lambda self: True)),
            patch.object(attention_module, "_call_sage_kernel", side_effect=kernel),
        ):
            out = attention_module.attention_sage2(
                q, k, v, heads, transformer_options=options, exact_kernel="fp32"
            )
        self.assertEqual(out.shape, q.shape)
        self.assertEqual(len(seen), 1)
        q_shape, k_shape, v_shape = seen[0]
        self.assertEqual(q_shape, (batch, heads, img_len, head_dim))
        self.assertEqual(k_shape, (batch, heads, img_len, head_dim))
        self.assertEqual(v_shape, (batch, heads, img_len, head_dim))

    def test_sage_image_only_drops_padded_text_keys_before_the_split(self):
        """A key mask must not push the joint call back onto the whole-softmax kernel."""
        torch.manual_seed(0)
        batch, heads, head_dim, img_len, txt_len = 1, 2, 32, 10, 6
        length = img_len + txt_len
        q, k, v = (torch.randn(batch, length, heads * head_dim, dtype=torch.float16) for _ in range(3))
        mask = torch.zeros(batch, length, dtype=torch.float16)
        mask[:, -2:] = torch.finfo(torch.float16).min  # two padded text keys
        options = {"vimoo_sage_text_length": txt_len, "vimoo_attention_kind": "joint"}
        seen = []

        def kernel(device, **kwargs):
            seen.append(tuple(kwargs[name].shape for name in ("q", "k")))
            out = kwargs["q"].float()
            return out.to(kwargs["q"].dtype), torch.zeros(out.shape[:-1], dtype=torch.float32)

        with (
            patch.object(torch.Tensor, "is_cuda", property(lambda self: True)),
            patch.object(attention_module, "_call_sage_kernel", side_effect=kernel),
        ):
            out = attention_module.attention_sage2(
                q, k, v, heads, mask=mask, transformer_options=options, exact_kernel="fp32"
            )
        self.assertEqual(out.shape, q.shape)
        # One kernel call, over the image keys only; the four padded/text keys never reach it.
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][1], (batch, heads, img_len, head_dim))

    def test_sage_without_image_only_sends_the_whole_joint_softmax(self):
        torch.manual_seed(0)
        batch, heads, head_dim, length = 1, 2, 32, 17
        q, k, v = (torch.randn(batch, length, heads * head_dim, dtype=torch.float16) for _ in range(3))
        options = {"vimoo_sage_text_length": 5, "vimoo_attention_kind": "joint"}
        seen = []

        def kernel(device, **kwargs):
            seen.append(kwargs["k"].shape)
            return kwargs["q"]

        with (
            patch.object(torch.Tensor, "is_cuda", property(lambda self: True)),
            patch.object(attention_module, "_call_sage_kernel", side_effect=kernel),
        ):
            attention_module.attention_sage2(q, k, v, heads, transformer_options=options, image_only=False)
        self.assertEqual(seen, [(batch, length, heads, head_dim)])

    def test_sage_pv_config_has_compatible_defaults_and_choices(self):
        options = {option.name: option for option in fields(performance.PerformanceConfig)}
        self.assertEqual(options["sage_pv_dtype"].metadata["choices"], ("fp8", "fp16"))
        self.assertEqual(performance.PerformanceConfig().sage_pv_dtype, "fp8")
        self.assertIn("fa2", options["sage_fallback"].metadata["choices"])
        self.assertEqual(
            set(options["sage_pv_accum_dtype"].metadata["choices"]),
            {"fp16", "fp16+fp32", "fp32", "fp32+fp32", "fp32+fp16"},
        )

    def test_sage_fa2_retention_defaults_to_zero_blocks(self):
        config = performance.PerformanceConfig()
        self.assertEqual(config.sage_fa2_first_blocks, 0)
        self.assertEqual(config.sage_fa2_last_blocks, 0)

    def test_sage_fa2_retention_resolves_actual_depth_union_without_source_mutation(self):
        for depth, first, last, expected in (
            (7, 2, 1, {0, 1, 6}),
            (3, 2, 2, {0, 1, 2}),
            (4, 0, 2, {2, 3}),
            (2, 2, 0, {0, 1}),
        ):
            with self.subTest(depth=depth, first=first, last=last):
                source = self.make_patcher()
                dit = source.get_model_object("diffusion_model")
                dit.double_blocks = torch.nn.ModuleList([dit.double_blocks[0]] * depth)
                with (
                    patch.object(attention_module, "_load_sage_kernel"),
                    patch.object(attention_module, "_load_flash_kernels"),
                ):
                    result = performance.apply_performance(
                        source,
                        performance.PerformanceConfig(
                            "sage",
                            "disabled",
                            "default",
                            sage_fallback="fa2",
                            sage_fa2_first_blocks=first,
                            sage_fa2_last_blocks=last,
                        ),
                    )
                options = result.model_options["transformer_options"]
                self.assertEqual(options["vimoo_sage_fa2_blocks"], frozenset(expected))
                self.assertIsInstance(options["vimoo_sage_fa2_blocks"], frozenset)
                self.assertTrue(callable(options["vimoo_sage_fa2_override"]))
                self.assertNotIn("vimoo_sage_fa2_blocks", source.model_options["transformer_options"])
                self.assertNotIn("vimoo_sage_fa2_override", source.model_options["transformer_options"])

    def test_sage_fa2_retention_invalid_counts_or_modes_fail_before_clone(self):
        source = self.make_patcher()
        for name in ("sage_fa2_first_blocks", "sage_fa2_last_blocks"):
            for value in (-1, 2, True, 1.0, "1", None):
                with self.subTest(name=name, value=value), patch.object(source, "clone") as clone:
                    with self.assertRaisesRegex(ValueError, "sage_fa2|FA2"):
                        performance.apply_performance(
                            source,
                            performance.PerformanceConfig(
                                "sage", "disabled", "default", sage_fallback="fa2", **{name: value}
                            ),
                        )
                    clone.assert_not_called()
        for attention, scope, fallback in (
            ("pytorch", "joint", "fa2"),
            ("fa2", "joint", "fa2"),
            ("sage", "all", "fa2"),
            ("sage", "text_refiner", "fa2"),
            ("sage", "joint", "fa4"),
            ("sage", "joint", "pytorch"),
        ):
            with (
                self.subTest(attention=attention, scope=scope, fallback=fallback),
                patch.object(source, "clone") as clone,
            ):
                with self.assertRaisesRegex(ValueError, "sage_fa2|FA2"):
                    performance.apply_performance(
                        source,
                        performance.PerformanceConfig(
                            attention,
                            "disabled",
                            "default",
                            sage_scope=scope,
                            sage_fallback=fallback,
                            sage_fa2_first_blocks=1,
                        ),
                    )
                clone.assert_not_called()

    def test_sage_fa2_retention_rejects_existing_attention_or_block_patches(self):
        for extra in (
            {"patches": {"attn1_patch": [lambda *args: args]}},
            {"patches_replace": {"dit": {("double_block", 0): object()}}},
        ):
            source = self.make_patcher()
            source.model_options["transformer_options"].update(extra)
            with self.subTest(extra=extra), patch.object(source, "clone") as clone:
                with self.assertRaisesRegex(ValueError, "patch"):
                    performance.apply_performance(
                        source,
                        performance.PerformanceConfig(
                            "sage", "disabled", "default", sage_fallback="fa2", sage_fa2_last_blocks=1
                        ),
                    )
                clone.assert_not_called()

    def test_sage_fa2_retained_override_accepts_comfy_original_function_argument(self):
        q, k, v, mask, metadata = object(), object(), object(), object(), {}
        with (
            patch.object(attention_module, "_load_sage_kernel"),
            patch.object(attention_module, "_load_flash_kernels"),
            patch.object(attention_module, "attention_flash", return_value="result") as flash,
        ):
            result = performance.apply_performance(
                self.make_patcher(),
                performance.PerformanceConfig(
                    "sage", "disabled", "default", sage_fallback="fa2", sage_fa2_first_blocks=1
                ),
            )
            override = result.model_options["transformer_options"]["vimoo_sage_fa2_override"]
            original = unittest.mock.Mock(side_effect=AssertionError("must not call the previous attention"))
            self.assertEqual(
                override(
                    original,
                    q,
                    k,
                    v,
                    2,
                    mask=mask,
                    scale=0.25,
                    skip_reshape=True,
                    skip_output_reshape=True,
                    transformer_options=metadata,
                ),
                "result",
            )
            flash.assert_called_once_with(
                q,
                k,
                v,
                2,
                backend="fa2",
                mask=mask,
                scale=0.25,
                skip_reshape=True,
                skip_output_reshape=True,
                transformer_options=metadata,
            )
            original.assert_not_called()

    def test_sage_fa2_zero_defaults_and_reconfiguration_clear_stale_policy(self):
        source = self.make_patcher()
        with (
            patch.object(attention_module, "_load_sage_kernel"),
            patch.object(attention_module, "_load_flash_kernels"),
        ):
            enabled = performance.apply_performance(
                source,
                performance.PerformanceConfig(
                    "sage", "disabled", "default", sage_fallback="fa2", sage_fa2_first_blocks=1
                ),
            )
            for attention in ("sage", "pytorch", "fa2"):
                fresh = performance.apply_performance(
                    source, performance.PerformanceConfig(attention, "disabled", "default")
                )
                cleared = performance.apply_performance(
                    enabled, performance.PerformanceConfig(attention, "disabled", "default")
                )
                for result in (fresh, cleared):
                    options = result.model_options["transformer_options"]
                    self.assertNotIn("vimoo_sage_fa2_blocks", options)
                    self.assertNotIn("vimoo_sage_fa2_override", options)
            self.assertEqual(
                enabled.model_options["transformer_options"]["vimoo_sage_fa2_blocks"], frozenset({0})
            )

    def test_sage_kernel_cache_keeps_pv_precisions_separate(self):
        fp16, fp8 = object(), object()
        core = types.SimpleNamespace(
            SM80_ENABLED=True,
            SM89_ENABLED=True,
            sageattn_qk_int8_pv_fp16_cuda=fp16,
            sageattn_qk_int8_pv_fp8_cuda=fp8,
        )
        with (
            patch.dict(attention_module._SAGE_KERNELS, {}, clear=True),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(attention_module.importlib, "import_module", return_value=core),
        ):
            for pv_dtype, kernel in (("fp8", fp8), ("fp16", fp16), ("fp8", fp8), ("fp16", fp16)):
                self.assertIs(attention_module._load_sage_kernel("cuda:0", pv_dtype=pv_dtype), kernel)
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                attention_module._load_sage_kernel("cpu", pv_dtype="fp8")

    def test_sage_fp8_uses_hopper_kernel_only_for_fp32_accumulation(self):
        calls = []

        def sm89(*args, **kwargs):
            calls.append(("sm89", kwargs))

        def sm90(*args, **kwargs):
            calls.append(("sm90", kwargs))

        core = types.SimpleNamespace(
            SM80_ENABLED=True,
            SM89_ENABLED=True,
            SM90_ENABLED=True,
            sageattn_qk_int8_pv_fp16_cuda=object(),
            sageattn_qk_int8_pv_fp8_cuda=sm89,
            sageattn_qk_int8_pv_fp8_cuda_sm90=sm90,
        )
        # Blackwell (SM100/SM103 datacenter, SM120/SM121 RTX 50) has no dedicated Sage 2.2.0 kernel:
        # upstream's own dispatcher uses the generic FP8 entry point for SM120 and rejects SM100/SM103.
        for capability, expected in (
            ((9, 0), "sm90"),
            ((8, 9), "sm89"),
            ((10, 0), "sm89"),
            ((10, 3), "sm89"),
            ((12, 0), "sm89"),
            ((12, 1), "sm89"),
        ):
            with self.subTest(capability=capability):
                calls.clear()
                with (
                    patch.dict(attention_module._SAGE_KERNELS, {}, clear=True),
                    patch.object(torch.cuda, "is_available", return_value=True),
                    patch.object(torch.cuda, "get_device_capability", return_value=capability),
                    patch.object(attention_module.importlib, "import_module", return_value=core),
                ):
                    kernel = attention_module._load_sage_kernel("cuda:0", pv_dtype="fp8")
                    kernel(None, pv_accum_dtype="fp32", smooth_v=False)
                    self.assertEqual(calls[-1][0], expected)
                    if expected == "sm90":
                        self.assertEqual(calls[-1][1]["pv_accum_dtype"], "fp32+fp32")
                        self.assertNotIn("smooth_v", calls[-1][1])
                        kernel(None, pv_accum_dtype="fp32+fp16", smooth_v=False)
                        self.assertEqual(calls[-1][0], "sm89")
                        kernel(None, pv_accum_dtype="fp32", smooth_v=True)
                        self.assertEqual(calls[-1][0], "sm89")

    def test_sage_missing_selected_extension_does_not_use_other_precision(self):
        for pv_dtype, enabled in (("fp16", "SM80_ENABLED"), ("fp8", "SM89_ENABLED")):
            with self.subTest(pv_dtype=pv_dtype):
                core = types.SimpleNamespace(
                    SM80_ENABLED=True,
                    SM89_ENABLED=True,
                    sageattn_qk_int8_pv_fp16_cuda=object(),
                    sageattn_qk_int8_pv_fp8_cuda=object(),
                )
                setattr(core, enabled, False)
                with (
                    patch.dict(attention_module._SAGE_KERNELS, {}, clear=True),
                    patch.object(torch.cuda, "is_available", return_value=True),
                    patch.object(attention_module.importlib, "import_module", return_value=core),
                ):
                    with self.assertRaisesRegex(RuntimeError, "SageAttention"):
                        attention_module._load_sage_kernel("cuda:0", pv_dtype=pv_dtype)

    def test_sage_kernel_bridge_consumes_precision_selector(self):
        self.assertTrue(getattr(attention_module._call_sage_kernel, "_torchdynamo_disable", False))
        seen = []

        def kernel(**kwargs):
            seen.append(kwargs)
            return "output"

        with patch.object(attention_module, "_load_sage_kernel", return_value=kernel) as loader:
            self.assertEqual(
                attention_module._call_sage_kernel("cuda:0", pv_dtype="fp8", q="query"), "output"
            )
        loader.assert_called_once_with("cuda:0", pv_dtype="fp8")
        self.assertEqual(seen, [{"q": "query"}])

    def test_sage_invalid_precision_combinations_fail_before_tensor_work_or_fallback(self):
        q = torch.randn(1, 3, 8, dtype=torch.bfloat16)
        invalid = (
            ("fp16", "fp32+fp32"),
            ("fp16", "fp32+fp16"),
            ("fp8", "fp16"),
            ("fp8", "fp16+fp32"),
            ("fp4", "fp32"),
            ("fp8", "invalid"),
        )
        for pv_dtype, accum in invalid:
            with self.subTest(pv_dtype=pv_dtype, accum=accum):
                with (
                    patch.object(attention_module, "_call_sage_kernel") as kernel,
                    patch.object(attention_module, "_sage_fallback") as fallback,
                ):
                    with self.assertRaisesRegex(ValueError, "PV"):
                        attention_module.attention_sage2(q, q, q, 2, pv_dtype=pv_dtype, pv_accum_dtype=accum)
                kernel.assert_not_called()
                fallback.assert_not_called()

    def test_sage_all_valid_accumulations_forward_selector_and_effective_smoothing(self):
        q = torch.randn(1, 3, 8, dtype=torch.bfloat16)
        for pv_dtype, accumulations in (
            ("fp16", ("fp16", "fp16+fp32", "fp32")),
            ("fp8", ("fp32", "fp32+fp32", "fp32+fp16")),
        ):
            for accum in accumulations:
                for smooth_v in (False, True):
                    for mask in (None, torch.tensor([[[True, False, True]]])):
                        with self.subTest(pv_dtype=pv_dtype, accum=accum, smooth_v=smooth_v, mask=mask):
                            with (
                                patch.object(torch.Tensor, "is_cuda", property(lambda self: True)),
                                patch.object(
                                    attention_module,
                                    "_call_sage_kernel",
                                    side_effect=lambda device, **kw: kw["q"],
                                ) as kernel,
                            ):
                                actual = attention_module.attention_sage2(
                                    q,
                                    q,
                                    q,
                                    2,
                                    mask=mask,
                                    pv_dtype=pv_dtype,
                                    pv_accum_dtype=accum,
                                    smooth_v=smooth_v,
                                )
                            torch.testing.assert_close(actual, q)
                            self.assertIn("pv_dtype", kernel.call_args.kwargs)
                            self.assertEqual(kernel.call_args.kwargs["pv_dtype"], pv_dtype)
                            self.assertEqual(kernel.call_args.kwargs["pv_accum_dtype"], accum)
                            self.assertEqual(
                                kernel.call_args.kwargs["smooth_v"],
                                smooth_v and ((pv_dtype, accum) in (("fp16", "fp16"), ("fp8", "fp32"))),
                            )

    def test_sage_fp8_options_load_fa2_and_reject_invalid_before_clone(self):
        source = self.make_patcher()
        with (
            patch.object(attention_module, "_load_sage_kernel") as sage_loader,
            patch.object(attention_module, "_load_flash_kernels") as flash_loader,
        ):
            result = performance.apply_performance(
                source,
                performance.PerformanceConfig(
                    "sage",
                    "disabled",
                    "default",
                    sage_pv_dtype="fp8",
                    sage_pv_accum_dtype="fp32",
                    sage_smooth_v=True,
                    sage_fallback="fa2",
                ),
            )
        sage_loader.assert_called_once_with(source.load_device, pv_dtype="fp8")
        flash_loader.assert_called_once_with("fa2", source.load_device)
        configured = (
            result.model_options["transformer_options"]["optimized_attention_override"]
            .__closure__[0]
            .cell_contents
        )
        self.assertEqual(configured.keywords["pv_dtype"], "fp8")
        self.assertTrue(configured.keywords["smooth_v"])
        self.assertEqual(configured.keywords["fallback_backend"], "fa2")
        for pv_dtype, accum in (("fp8", "fp16"), ("fp16", "fp32+fp32"), ("bad", "fp32")):
            with patch.object(source, "clone") as clone:
                with self.assertRaisesRegex(ValueError, "PV"):
                    performance.apply_performance(
                        source,
                        performance.PerformanceConfig(
                            "sage", "disabled", "default", sage_pv_dtype=pv_dtype, sage_pv_accum_dtype=accum
                        ),
                    )
            clone.assert_not_called()

    def test_sage_fa2_fallback_preserves_scope_and_mask_arguments(self):
        q = torch.randn(1, 3, 8, dtype=torch.bfloat16)
        mask = torch.zeros(3, 3)
        for scope, selected_mask in (("text_refiner", None), ("joint", mask)):
            with (
                patch.object(torch.Tensor, "is_cuda", property(lambda self: True)),
                patch.object(attention_module, "_call_sage_kernel") as sage,
                patch.object(attention_module, "attention_flash", return_value=q) as flash,
            ):
                attention_module.attention_sage2(
                    q,
                    q,
                    q,
                    2,
                    scope=scope,
                    fallback_backend="fa2",
                    mask=selected_mask,
                    skip_output_reshape=True,
                    scale=0.25,
                )
            sage.assert_not_called()
            flash.assert_called_once()
            self.assertEqual(flash.call_args.kwargs["backend"], "fa2")
            self.assertIs(flash.call_args.kwargs["mask"], selected_mask)
            self.assertTrue(flash.call_args.kwargs["skip_output_reshape"])
            self.assertEqual(flash.call_args.kwargs["scale"], 0.25)

    def test_prefix_varlen_packing_uses_views_and_arbitrary_masks_still_gather(self):
        k = torch.arange(2 * 5 * 2).reshape(2, 5, 2)
        v = k + 100
        valid = torch.tensor([[True, True, True, False, False]])
        metadata = attention_module._prepare_flash_metadata(valid, 1, 5, 5, torch.device("cpu"))
        one_k, one_v = attention_module._pack_valid_kv(k[:1], v[:1], metadata)
        torch.testing.assert_close(one_k, k[0, :3])
        torch.testing.assert_close(one_v, v[0, :3])
        self.assertEqual(one_k.untyped_storage().data_ptr(), k.untyped_storage().data_ptr())

        arbitrary = torch.tensor([[True, False, True, False, False]])
        metadata = attention_module._prepare_flash_metadata(arbitrary, 1, 5, 5, torch.device("cpu"))
        packed_k, packed_v = attention_module._pack_valid_kv(k[:1], v[:1], metadata)
        torch.testing.assert_close(packed_k, k[0, [0, 2]])
        torch.testing.assert_close(packed_v, v[0, [0, 2]])

    def test_flash_launch_metadata_is_reused_by_repeated_blocks(self):
        torch.manual_seed(3)
        q, k, v = [torch.randn(1, 5, 8, dtype=torch.bfloat16) for _ in range(3)]
        mask = torch.tensor([[[True, True, True, False, False]]])
        options = {"vimoo_attention_metadata": {}}
        kernel = lambda *args, **kwargs: kwargs["q"]
        with (
            patch.object(torch.Tensor, "is_cuda", property(lambda self: True)),
            patch.object(attention_module, "_call_flash_kernel", side_effect=kernel),
            patch.object(
                attention_module,
                "_prepare_flash_metadata",
                wraps=attention_module._prepare_flash_metadata,
            ) as prepare,
        ):
            for _ in range(2):
                attention_module.attention_flash(
                    q, k, v, 2, mask=mask, transformer_options=options, backend="fa4"
                )
        self.assertEqual(prepare.call_count, 1)
        # The cache outlives a single forward, so it has to own the mask it was
        # keyed on: a freed address would otherwise be reused by a different mask
        # of the same shape and return its varlen offsets.
        (entry,) = options["vimoo_attention_metadata"].values()
        self.assertIs(entry[0], mask)

    def test_conditioning_profiles_pack_symmetric_queries_and_scatter_padding_without_changing_legacy(self):
        q, k, v = [torch.randn(2, 5, 8, dtype=torch.bfloat16) for _ in range(3)]
        mask = torch.tensor([[[True, True, True, False, False]], [[True, False, True, False, False]]])
        options = {
            "vimoo_attention_metadata": {},
            "vimoo_symmetric_self_attention": True,
        }
        with (
            patch.object(torch.Tensor, "is_cuda", property(lambda self: True)),
            patch.object(
                attention_module, "_call_flash_kernel", side_effect=lambda *args, **kw: kw["q"]
            ) as launch,
        ):
            output = attention_module.attention_flash(
                q, k, v, 2, mask=mask, transformer_options=options, backend="fa2"
            )
            call = launch.call_args.kwargs
            torch.testing.assert_close(call["cu_seqlens_q"], torch.tensor([0, 3, 5], dtype=torch.int32))
            torch.testing.assert_close(call["cu_seqlens_q"], call["cu_seqlens_k"])
            self.assertEqual(call["max_seqlen_q"], 3)
            self.assertEqual(call["q"].shape[0], 5)
            self.assertNotIn("native_kernel", call)
            expected = q.masked_fill(~mask.transpose(1, 2), 0)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            attention_module.attention_flash(q, k, v, 2, mask=mask, backend="fa2")
            legacy = launch.call_args.kwargs
            torch.testing.assert_close(legacy["cu_seqlens_q"], torch.tensor([0, 5, 10], dtype=torch.int32))
            self.assertEqual(legacy["q"].shape[0], 10)
            self.assertNotIn("native_kernel", legacy)

    def test_lite_compact_flash_retains_allocated_capacity_and_max_lengths(self):
        q, k, v = [torch.randn(2, 5, 8, dtype=torch.bfloat16) for _ in range(3)]
        mask = torch.tensor([[[True, True, False, False, False]], [[True, False, True, False, False]]])
        options = {
            "vimoo_attention_metadata": {},
            "vimoo_symmetric_self_attention": True,
            "vimoo_flash_padded_capacity": True,
        }
        with (
            patch.object(torch.Tensor, "is_cuda", property(lambda self: True)),
            patch.object(
                attention_module, "_call_flash_kernel", side_effect=lambda *args, **kw: kw["q"]
            ) as launch,
        ):
            output = attention_module.attention_flash(
                q, k, v, 2, mask=mask, transformer_options=options, backend="fa4"
            )
        call = launch.call_args.kwargs
        self.assertEqual(call["max_seqlen_q"], 5)
        self.assertEqual(call["max_seqlen_k"], 5)
        self.assertEqual(call["q"].shape, (10, 2, 4))
        self.assertEqual(call["q"].stride(), (8, 4, 1))
        torch.testing.assert_close(call["cu_seqlens_q"], torch.tensor([0, 2, 4], dtype=torch.int32))
        torch.testing.assert_close(call["cu_seqlens_q"], call["cu_seqlens_k"])
        # Append one zero row, repeats its sentinel gather index to
        # retain the original capacity, and discards sentinel rows on unpack.
        indices = torch.tensor([0, 1, 5, 7, 10, 10, 10, 10, 10, 10])[:, None, None].expand(10, 2, 4)
        for key, original in zip(("q", "k", "v"), (q, k, v)):
            values = original.reshape(10, 2, 4)
            source = torch.cat((values, values.new_zeros(1, 2, 4))).gather(0, indices)
            torch.testing.assert_close(call[key], source, rtol=0, atol=0)
        torch.testing.assert_close(output, q.masked_fill(~mask.transpose(1, 2), 0), rtol=0, atol=0)

    def test_allocated_capacity_requires_symmetric_packing(self):
        with self.assertRaisesRegex(ValueError, "symmetric"):
            attention_module._prepare_flash_metadata(
                torch.tensor([[True, False]]), 1, 2, 2, torch.device("cpu"), padded_capacity=True
            )

    def test_flash_metadata_cache_survives_compile_graph_break(self):
        q, k, v = [torch.randn(1, 5, 8, dtype=torch.bfloat16) for _ in range(3)]
        mask = torch.tensor([[[True, True, True, False, False]]])
        metadata_cache = {}
        with (
            patch.object(torch.Tensor, "is_cuda", property(lambda self: True)),
            patch.object(attention_module, "_call_flash_kernel", side_effect=lambda *args, **kw: kw["q"]),
        ):
            output = attention_module.attention_flash(
                q,
                k,
                v,
                2,
                mask=mask,
                transformer_options={"vimoo_attention_metadata": metadata_cache},
                backend="fa3",
            )
        self.assertEqual(output.shape, q.shape)
        self.assertEqual(len(metadata_cache), 1)
        self.assertTrue(getattr(attention_module._cached_flash_metadata, "_torchdynamo_disable", False))

    def test_flash_packing_and_version_specific_signatures(self):
        self.assertTrue(getattr(attention_module._call_flash_kernel, "_torchdynamo_disable", False))
        self.assertTrue(getattr(attention_module.exact_partial_attention, "_torchdynamo_disable", False))

        def kernel(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale=None,
            causal=False,
            window_size=(-1, -1),
        ):
            self.assertEqual(window_size, (None, None) if backend == "fa4" else (-1, -1))
            self.assertEqual(cu_seqlens_q.dtype, torch.int32)
            self.assertEqual(cu_seqlens_k.dtype, torch.int32)
            outputs = []
            for index in range(len(cu_seqlens_q) - 1):
                qi = q[cu_seqlens_q[index] : cu_seqlens_q[index + 1]].transpose(0, 1)[None]
                ki = k[cu_seqlens_k[index] : cu_seqlens_k[index + 1]].transpose(0, 1)[None]
                vi = v[cu_seqlens_k[index] : cu_seqlens_k[index + 1]].transpose(0, 1)[None]
                result = torch.nn.functional.scaled_dot_product_attention(qi, ki, vi, scale=softmax_scale)
                outputs.append(result[0].transpose(0, 1))
            return torch.cat(outputs)

        def fixed_kernel(q, k, v, softmax_scale=None, causal=False, window_size=(-1, -1)):
            self.assertEqual(window_size, (None, None) if backend == "fa4" else (-1, -1))
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=softmax_scale, is_causal=causal
            ).transpose(1, 2)

        torch.manual_seed(4)
        q, k, v = [torch.randn(2, 3, 8, dtype=torch.bfloat16) for _ in range(3)]
        masks = [
            None,
            torch.tensor([[[0.0, -torch.inf, 0.0]]]),
            torch.tensor([[[0.0, -2.0, 0.0]]]),
            torch.zeros(1, 1, 3, dtype=torch.bool),
        ]
        for backend in ("fa2", "fa3", "fa4"):
            for mask in masks:
                for heads_first in (False, True):
                    with self.subTest(backend=backend, mask=mask, heads_first=heads_first):
                        values = (
                            [t.reshape(2, 3, 2, 4).transpose(1, 2) for t in (q, k, v)]
                            if heads_first
                            else (q, k, v)
                        )
                        expected = performance.comfy.ldm.modules.attention.attention_pytorch(
                            *values, 2, mask=mask, skip_reshape=heads_first
                        )
                        with (
                            patch.object(torch.Tensor, "is_cuda", property(lambda self: True)),
                            patch.object(
                                attention_module, "_load_flash_kernels", return_value=(kernel, fixed_kernel)
                            ),
                        ):
                            actual = attention_module.attention_flash(
                                *values, 2, mask=mask, skip_reshape=heads_first, backend=backend
                            )
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_key_padding_preserves_bias_and_pairwise_mask_semantics(self):
        for mask in (
            torch.tensor([[0.0, -2.0, 0.0]]),
            torch.zeros(2, 3),
            torch.zeros(1, 1, 3, dtype=torch.bool),
            torch.full((1, 1, 3), -torch.inf),
        ):
            with self.subTest(mask=mask):
                self.assertIsNone(attention_module._key_padding_mask(mask, 2, 3))
        padding = torch.tensor([[[0.0, -torch.inf, 0.0]]])
        torch.testing.assert_close(
            attention_module._key_padding_mask(padding, 2, 3),
            torch.tensor([[True, False, True], [True, False, True]]),
        )

    def test_cpu_fallback_matches_sdpa_with_bias_and_layouts(self):
        torch.manual_seed(42)
        q, k, v = [torch.randn(2, 3, 8) for _ in range(3)]
        mask = torch.tensor([[[[0.0, -2.0, 0.0]]], [[[0.0, 0.0, -torch.inf]]]])
        for heads_first in (False, True):
            values = [t.reshape(2, 3, 2, 4).transpose(1, 2) for t in (q, k, v)] if heads_first else (q, k, v)
            for keep_heads in (False, True):
                with self.subTest(heads_first=heads_first, keep_heads=keep_heads):
                    actual = attention_module.attention_flash(
                        *values,
                        2,
                        mask=mask,
                        skip_reshape=heads_first,
                        skip_output_reshape=keep_heads,
                        scale=0.25,
                    )
                    expected = performance.comfy.ldm.modules.attention.attention_pytorch(
                        *values,
                        2,
                        mask=mask,
                        skip_reshape=heads_first,
                        skip_output_reshape=keep_heads,
                        scale=0.25,
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_sage_padding_is_applied_to_kv_and_parameters_reach_kernel(self):
        torch.manual_seed(8)
        q, k, v = [torch.randn(1, 3, 8, dtype=torch.bfloat16) for _ in range(3)]
        mask = torch.tensor([[[0.0, -torch.inf, 0.0]]], dtype=torch.bfloat16)
        calls = []

        def kernel(device, **kwargs):
            calls.append(kwargs)
            return torch.nn.functional.scaled_dot_product_attention(
                kwargs["q"].transpose(1, 2),
                kwargs["k"].transpose(1, 2),
                kwargs["v"].transpose(1, 2),
                scale=kwargs["sm_scale"],
            ).transpose(1, 2)

        expected = performance.comfy.ldm.modules.attention.attention_pytorch(q, k, v, 2, mask=mask)
        with (
            patch.object(torch.Tensor, "is_cuda", property(lambda self: True)),
            patch.object(attention_module, "_call_sage_kernel", side_effect=kernel),
        ):
            actual = attention_module.attention_sage2(
                q,
                k,
                v,
                2,
                mask=mask,
                qk_quant_gran="per_thread",
                pv_accum_dtype="fp16+fp32",
                smooth_k=True,
                smooth_v=False,
            )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["k"].shape[1], 2)
        self.assertEqual(calls[0]["tensor_layout"], "NHD")
        self.assertEqual(calls[0]["qk_quant_gran"], "per_thread")
        self.assertEqual(calls[0]["pv_accum_dtype"], "fp16+fp32")
        self.assertTrue(calls[0]["smooth_k"])
        self.assertFalse(calls[0]["smooth_v"])

    def test_sage_scope_and_arbitrary_mask_use_exact_fallback(self):
        q, k, v = [torch.randn(1, 3, 8, dtype=torch.bfloat16) for _ in range(3)]
        expected = torch.randn_like(q)
        pairwise_mask = torch.zeros(3, 3)
        with (
            patch.object(torch.Tensor, "is_cuda", property(lambda self: True)),
            patch.object(attention_module, "_call_sage_kernel") as kernel,
            patch.object(attention_module, "_sage_fallback", return_value=expected) as fallback,
        ):
            actual = attention_module.attention_sage2(
                q, k, v, 2, scope="text_refiner", vimoo_attention_kind="joint"
            )
            self.assertIs(actual, expected)
            actual = attention_module.attention_sage2(q, k, v, 2, mask=pairwise_mask)
            self.assertIs(actual, expected)
        kernel.assert_not_called()
        self.assertEqual(fallback.call_count, 2)

    def test_cached_kernel_still_checks_selected_gpu(self):
        with (
            patch.dict(attention_module._FLASH_KERNELS, {"fa3": (object(), object())}, clear=True),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)) as capability,
        ):
            with self.assertRaisesRegex(RuntimeError, "Hopper"):
                attention_module._load_flash_kernels("fa3", torch.device("cuda:1"))
            capability.assert_called_once_with(torch.device("cuda:1"))

    def test_kernel_download_does_not_patch_shared_libraries(self):
        kernel = object()
        fixed_kernel = object()
        from unittest.mock import Mock

        fake = types.ModuleType("kernels")
        fake.get_kernel = Mock(
            return_value=types.SimpleNamespace(
                flash_attn_varlen_func=kernel,
                flash_attn_func=fixed_kernel,
            )
        )
        with (
            patch.dict(attention_module._FLASH_KERNELS, {}, clear=True),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
            patch.dict(sys.modules, {"kernels": fake}),
            patch.object(
                attention_module.importlib.metadata,
                "version",
                side_effect=attention_module.importlib.metadata.PackageNotFoundError("flash-attn"),
            ),
        ):
            for backend in ("fa2", "fa3"):
                self.assertEqual(attention_module._load_flash_kernels(backend), (kernel, fixed_kernel))
                self.assertEqual(attention_module._load_flash_kernels(backend), (kernel, fixed_kernel))
                repo, revision = attention_module._FLASH_HUB[backend]
                fake.get_kernel.assert_called_with(repo, revision=revision)
            self.assertEqual(fake.get_kernel.call_count, 2)

    def test_installed_native_fa2_is_shared_by_all_fa2_calls(self):
        kernel, fixed_kernel = lambda **kwargs: None, lambda **kwargs: None
        module = types.SimpleNamespace(flash_attn_varlen_func=kernel, flash_attn_func=fixed_kernel)
        fake = types.ModuleType("kernels")
        fake.get_kernel = unittest.mock.Mock(side_effect=AssertionError("Hub FA2 must not also load"))
        with (
            patch.dict(attention_module._FLASH_KERNELS, {}, clear=True),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
            patch.object(attention_module.importlib.metadata, "version", return_value="2.8.3") as version,
            patch.object(attention_module.importlib, "import_module", return_value=module) as importer,
            patch.dict(sys.modules, {"kernels": fake}),
        ):
            self.assertEqual(attention_module._load_flash_kernels("fa2"), (kernel, fixed_kernel))
            self.assertEqual(attention_module._load_flash_kernels("fa2"), (kernel, fixed_kernel))
            self.assertEqual(attention_module._load_flash_kernels("fa2"), (kernel, fixed_kernel))
            version.assert_called_once_with("flash-attn")
            importer.assert_called_once_with("flash_attn")
            fake.get_kernel.assert_not_called()

    def test_fa2_hub_is_used_when_native_distribution_is_absent(self):
        fake = types.ModuleType("kernels")
        kernel, fixed_kernel = object(), object()
        fake.get_kernel = unittest.mock.Mock(
            return_value=types.SimpleNamespace(flash_attn_varlen_func=kernel, flash_attn_func=fixed_kernel)
        )
        with (
            patch.dict(attention_module._FLASH_KERNELS, {}, clear=True),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
            patch.object(
                attention_module.importlib.metadata,
                "version",
                side_effect=attention_module.importlib.metadata.PackageNotFoundError("flash-attn"),
            ),
            patch.dict(sys.modules, {"kernels": fake}),
        ):
            self.assertEqual(attention_module._load_flash_kernels("fa2"), (kernel, fixed_kernel))
            self.assertEqual(attention_module._load_flash_kernels("fa2"), (kernel, fixed_kernel))
            repo, revision = attention_module._FLASH_HUB["fa2"]
            fake.get_kernel.assert_called_once_with(repo, revision=revision)

    def test_fa2_hub_cache_is_shared_without_importing_native_namespace(self):
        kernels = (object(), object())
        with (
            patch.dict(attention_module._FLASH_KERNELS, {"fa2": kernels}, clear=True),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
            patch.object(attention_module.importlib, "import_module") as importer,
            patch.object(attention_module.importlib.metadata, "version") as version,
        ):
            self.assertEqual(attention_module._load_flash_kernels("fa2"), kernels)
            importer.assert_not_called()
            version.assert_not_called()

    def test_native_fa2_namespace_reports_its_required_dependency(self):
        with (
            patch.dict(attention_module._FLASH_KERNELS, {}, clear=True),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
            patch.object(attention_module.importlib, "import_module", return_value=types.SimpleNamespace()),
            patch.object(attention_module.importlib.metadata, "version", return_value="2.8.3"),
        ):
            with self.assertRaisesRegex(RuntimeError, "requirements-native-fa2.txt") as error:
                attention_module._load_flash_kernels("fa2")
            self.assertIsInstance(error.exception.__cause__, ImportError)

    def make_patcher(self):
        class Block(torch.nn.Module):
            def forward(self, x, transformer_options):
                return x + transformer_options["offset"]

        model = torch.nn.Module()
        model.memory_usage_factor = 4.0
        model.diffusion_model = torch.nn.Module()
        model.diffusion_model.double_blocks = torch.nn.ModuleList([Block()])
        model.diffusion_model.qk_norm_math = lambda q, k: (q, k)
        model.diffusion_model.rope_math = lambda x, cos, sin: x
        model.diffusion_model.modulate_math = lambda x, shift, scale: x
        return comfy.model_patcher.ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))

    def test_regional_compile_rechecks_changed_transformer_options(self):
        original_compile = torch.compile
        compiled = []

        def compile_small_model(model, **kwargs):
            # Keep this a real Dynamo compilation without requiring a C++ toolchain.
            kwargs["backend"] = "eager"
            compiled.append(original_compile(model, **kwargs))
            return compiled[-1]

        with patch.object(torch, "compile", side_effect=compile_small_model):
            performance.apply_performance(
                self.make_patcher(), performance.PerformanceConfig("pytorch", "regional", "default")
            )
        x = torch.ones(1)
        torch.testing.assert_close(compiled[0](x, {"offset": 2}), torch.tensor([3.0]))
        torch.testing.assert_close(compiled[0](x, {"offset": 7}), torch.tensor([8.0]))

    def test_regional_compile_keeps_dynamic_patcher_and_enables_block_prefetch(self):
        patcher = self.make_patcher()
        with (
            patch.object(patcher, "is_dynamic", return_value=True),
            patch.object(patcher, "clone", wraps=patcher.clone) as clone_fn,
            patch.object(attention_module, "_load_flash_kernels"),
            patch.object(torch, "compile", side_effect=lambda model, **kwargs: model),
        ):
            result = performance.apply_performance(
                patcher, performance.PerformanceConfig("fa3", "regional", "default")
            )
        clone_fn.assert_called_once_with(disable_dynamic=False)
        self.assertTrue(result.model_options["transformer_options"]["vimoo_block_prefetch"])
        regional_math = result.model_options["transformer_options"]["vimoo_math"]
        self.assertIs(regional_math[0], result.model.diffusion_model.qk_norm_math)
        self.assertIs(regional_math[1], result.model.diffusion_model.rope_math)
        self.assertIs(regional_math[2], result.model.diffusion_model.modulate_math)
        self.assertTrue(result.model_options["transformer_options"]["vimoo_math_graph"])
        self.assertTrue(result.get_model_object("diffusion_model.precise_inputs"))

    def test_precise_math_uses_official_scoped_compile_wrapper(self):
        from comfy_api.torch_helpers.torch_compile import TORCH_COMPILE_KWARGS

        patcher = self.make_patcher()
        dit = patcher.model.diffusion_model
        dit.precise_math = False
        for name in ("qk_norm", "rope", "modulate"):
            setattr(dit, name + "_math", lambda x: x + 1)
        with (
            patch.object(torch, "compile", side_effect=lambda model, **kwargs: model) as compile_fn,
            patch.object(patcher, "clone", wraps=patcher.clone) as clone_fn,
        ):
            result = performance.apply_performance(
                patcher, performance.PerformanceConfig("pytorch", "math_helpers", "default")
            )
        clone_fn.assert_called_once_with(disable_dynamic=False)
        self.assertEqual(compile_fn.call_count, 3)
        self.assertTrue(all(call.kwargs["fullgraph"] for call in compile_fn.call_args_list))
        self.assertTrue(result.get_wrappers(WrappersMP.APPLY_MODEL, "torch.compile"))
        self.assertTrue(result.model_options[TORCH_COMPILE_KWARGS]["dynamic"])
        self.assertTrue(result.get_model_object("diffusion_model.precise_math"))
        self.assertTrue(result.get_model_object("diffusion_model.precise_inputs"))
        self.assertFalse(dit.precise_math)
        # Legacy workflows that only saved compile=disabled keep device math.
        eager = performance.apply_performance(
            result, performance.PerformanceConfig("pytorch", "disabled", "default")
        )
        self.assertFalse(eager.get_model_object("diffusion_model.precise_math"))
        self.assertFalse(eager.get_model_object("diffusion_model.precise_inputs"))
        self.assertFalse(eager.get_wrappers(WrappersMP.APPLY_MODEL, "torch.compile"))
        reference_eager = performance.apply_performance(
            result, performance.PerformanceConfig("pytorch", "disabled", "default", precise_math=True)
        )
        self.assertTrue(reference_eager.get_model_object("diffusion_model.precise_math"))
        self.assertTrue(reference_eager.get_model_object("diffusion_model.precise_inputs"))
        self.assertFalse(reference_eager.get_wrappers(WrappersMP.APPLY_MODEL, "torch.compile"))

    def test_full_i2v_math_helpers_compile_specialized_operations(self):
        i2v = sys.modules[f"{spec.name}.vimoo_i2v"]
        source = self.make_patcher()
        dit = source.model.diffusion_model
        for name in ("rope_pair", "gate", "fa2_pack"):
            setattr(dit, name + "_math", lambda *args: args)
        for block in dit.double_blocks:
            for stream in ("img", "txt"):
                projection = sys.modules[f"{spec.name}.vimoo_patchify"].BF16ModulationExpansion(4, 6)
                setattr(block, stream + "_mod", torch.nn.Module())
                getattr(block, stream + "_mod").lin = projection
        expected = [dit.qk_norm_math, dit.modulate_math, dit.rope_pair_math, dit.gate_math, dit.fa2_pack_math]
        expected.extend(
            module.lin.math for block in dit.double_blocks for module in (block.img_mod, block.txt_mod)
        )

        with patch.object(torch, "compile", side_effect=lambda model, **kwargs: model) as compile_fn:
            configured = performance.apply_performance(
                source, performance.PerformanceConfig("pytorch", "math_helpers", "default")
            )
            compile_fn.reset_mock()
            i2v._configure_math_compilation(configured)

        self.assertCountEqual(
            [id(call.kwargs["model"]) for call in compile_fn.call_args_list],
            [id(module) for module in expected],
        )
        self.assertTrue(all(call.kwargs["fullgraph"] for call in compile_fn.call_args_list))
        self.assertTrue(all(call.kwargs["dynamic"] for call in compile_fn.call_args_list))
        self.assertTrue(configured.model_options["transformer_options"]["vimoo_i2v_compiled_modulation"])
        self.assertNotIn("vimoo_i2v_compiled_modulation", source.model_options["transformer_options"])

    def test_block_prefetch_is_independent_of_compilation_and_preserves_source(self):
        for compile in ("regional", "math_helpers", "disabled"):
            for enabled in (False, True):
                with self.subTest(compile=compile, enabled=enabled):
                    source = self.make_patcher()
                    with patch.object(torch, "compile", side_effect=lambda model, **kwargs: model):
                        result = performance.apply_performance(
                            source,
                            performance.PerformanceConfig(
                                "pytorch", compile, "default", block_prefetch=enabled
                            ),
                        )
                    self.assertEqual(
                        result.model_options["transformer_options"]["vimoo_block_prefetch"], enabled
                    )
                    self.assertNotIn("vimoo_block_prefetch", source.model_options["transformer_options"])

    def test_sage_selection_does_not_load_flash_attention_by_default(self):
        source = self.make_patcher()
        with (
            patch.object(attention_module, "_load_sage_kernel") as sage_loader,
            patch.object(
                attention_module,
                "_load_flash_kernels",
                side_effect=AssertionError("unexpected FA dependency"),
            ),
        ):
            result = performance.apply_performance(
                source, performance.PerformanceConfig("sage", "disabled", "default")
            )
        sage_loader.assert_called_once_with(source.load_device, pv_dtype="fp8")
        override = result.model_options["transformer_options"]["optimized_attention_override"]
        configured = override.__closure__[0].cell_contents
        self.assertEqual(configured.keywords["exact_kernel"], "fp32")
        self.assertEqual(configured.keywords["fallback_backend"], "pytorch")
        self.assertTrue(configured.keywords["image_only"])
        self.assertTrue(result.model_options["transformer_options"]["vimoo_sage_image_only"])
        self.assertNotIn("vimoo_sage_image_only", source.model_options["transformer_options"])

    def test_sage_options_load_selected_kernel_and_fallback(self):
        source = self.make_patcher()
        with (
            patch.object(attention_module, "_load_sage_kernel") as sage_loader,
            patch.object(attention_module, "_load_flash_kernels") as flash_loader,
        ):
            result = performance.apply_performance(
                source,
                performance.PerformanceConfig(
                    "sage",
                    "disabled",
                    "default",
                    sage_scope="all",
                    sage_fallback="fa4",
                    sage_qk_quant_gran="per_thread",
                    sage_pv_dtype="fp16",
                    sage_pv_accum_dtype="fp16+fp32",
                    sage_smooth_k=True,
                    sage_smooth_v=True,
                ),
            )
        sage_loader.assert_called_once_with(source.load_device, pv_dtype="fp16")
        flash_loader.assert_called_once_with("fa4", source.load_device)
        override = result.model_options["transformer_options"]["optimized_attention_override"]
        configured = override.__closure__[0].cell_contents
        self.assertEqual(configured.keywords["scope"], "all")
        self.assertEqual(configured.keywords["qk_quant_gran"], "per_thread")
        self.assertEqual(configured.keywords["pv_accum_dtype"], "fp16+fp32")
        self.assertFalse(configured.keywords["smooth_v"])

    def test_retained_flags_and_memory_factor_are_opt_in(self):
        source = self.make_patcher()
        result = performance.apply_performance(
            source,
            performance.PerformanceConfig(
                "pytorch", "disabled", "default", fused_qkv=True, memory_usage_factor=2.25
            ),
        )
        options = result.model_options["transformer_options"]
        self.assertTrue(options["vimoo_fused_qkv"])
        self.assertNotIn("vimoo_fused_qkv", source.model_options["transformer_options"])
        self.assertEqual(result.get_model_object("memory_usage_factor"), 2.25)
        wrapper = result.get_wrappers(WrappersMP.PREPARE_SAMPLING, "vimoo_memory_usage_factor")[0]
        seen = []
        self.assertEqual(
            wrapper(lambda current: seen.append(current.model.memory_usage_factor), result),
            None,
        )
        self.assertEqual(seen, [2.25])
        self.assertEqual(result.model.memory_usage_factor, 4.0)

        disabled = performance.apply_performance(
            result, performance.PerformanceConfig("pytorch", "disabled", "default")
        )
        disabled_options = disabled.model_options["transformer_options"]
        for key in (
            "vimoo_fused_qkv",
            "vimoo_fused_ops",
        ):
            self.assertNotIn(key, disabled_options)
        self.assertEqual(disabled.get_model_object("memory_usage_factor"), 4.0)
        self.assertEqual(
            len(disabled.get_wrappers(WrappersMP.PREPARE_SAMPLING, "vimoo_memory_usage_factor")),
            1,
        )

    def test_disabling_compile_removes_inherited_wrapper_only_on_clone(self):
        patcher = self.make_patcher()
        patcher.add_wrapper_with_key(WrappersMP.APPLY_MODEL, "torch.compile", lambda: None)
        patcher.model_options["torch_compile_kwargs"] = {"backend": "inductor"}
        patched = performance.apply_performance(
            patcher, performance.PerformanceConfig("pytorch", "disabled", "default")
        )
        self.assertNotIn("torch_compile_kwargs", patched.model_options)
        self.assertIn("torch_compile_kwargs", patcher.model_options)
        self.assertFalse(patched.get_wrappers(WrappersMP.APPLY_MODEL, "torch.compile"))
        self.assertTrue(patcher.get_wrappers(WrappersMP.APPLY_MODEL, "torch.compile"))

    def test_compile_shape_policy_and_modes_reach_scoped_wrapper(self):
        for scope in ("math_helpers", "regional"):
            for policy in ("auto", "static", "dynamic"):
                for mode in ("default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"):
                    with self.subTest(scope=scope, policy=policy, mode=mode):
                        with patch.object(performance, "set_torch_compile_wrapper") as compile_fn:
                            performance.apply_performance(
                                self.make_patcher(),
                                performance.PerformanceConfig("pytorch", scope, mode, compile_dynamic=policy),
                            )
                        options = compile_fn.call_args.kwargs
                        self.assertEqual(options["backend"], "inductor")
                        self.assertEqual(options["mode"], None if mode == "default" else mode)
                        self.assertEqual(options["fullgraph"], scope == "math_helpers")
                        self.assertEqual(
                            options["dynamic"],
                            scope == "math_helpers" if policy == "auto" else policy == "dynamic",
                        )
                        self.assertEqual(len(options["keys"]), 3 if scope == "math_helpers" else 1)

    def test_async_allocator_keeps_mode_optimizations_without_cuda_graphs(self):
        for mode in ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"):
            with self.subTest(mode=mode):
                source = self.make_patcher()
                source.load_device = torch.device("cuda")
                with (
                    patch.object(
                        performance.torch.cuda.memory, "get_allocator_backend", return_value="cudaMallocAsync"
                    ),
                    patch.object(performance, "set_torch_compile_wrapper") as compile_fn,
                ):
                    performance.apply_performance(
                        source, performance.PerformanceConfig("pytorch", "regional", mode)
                    )
                actual = compile_fn.call_args.kwargs
                self.assertIsNone(actual["mode"])
                expected = torch._inductor.list_mode_options(mode).copy()
                expected["triton.cudagraphs"] = False
                self.assertEqual(actual["options"], expected)
                self.assertFalse(actual["options"]["triton.cudagraphs"])
                if mode.startswith("max-autotune"):
                    self.assertTrue(actual["options"]["max_autotune"])

    def test_unknown_attention_does_not_silently_select_pytorch(self):
        with self.assertRaisesRegex(ValueError, "Unknown attention"):
            performance.apply_performance(
                self.make_patcher(), performance.PerformanceConfig("unknown", "disabled", "default")
            )


exact_partial_attention = attention_module.exact_partial_attention
merge_lse = attention_module.merge_lse


class SageAttentionMath(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.q, self.k, self.v = (torch.randn(1, 2, 96, 16) for _ in range(3))
        self.k[..., 3] *= 20  # an outlier channel, as in real K
        self.ref = torch.nn.functional.scaled_dot_product_attention(self.q, self.k, self.v)

    def test_lse_merge_of_key_ranges_matches_full_attention(self):
        scale = 16**-0.5
        out_a, lse_a = exact_partial_attention(
            self.q, self.k[:, :, :64], self.v[:, :, :64], scale, chunk=32, kernel="fp32"
        )
        out_b, lse_b = exact_partial_attention(
            self.q, self.k[:, :, 64:], self.v[:, :, 64:], scale, chunk=32, kernel="fp32"
        )
        torch.testing.assert_close(merge_lse(out_a, lse_a, out_b, lse_b), self.ref, atol=1e-5, rtol=1e-5)

    def test_sage_image_only_matches_full_attention_with_exact_kernel(self):
        img_len, scale = 64, 16**-0.5
        q, k, v = (t.half() for t in (self.q, self.k, self.v))

        def kernel(device, **kwargs):
            out, lse = exact_partial_attention(
                kwargs["q"], kwargs["k"], kwargs["v"], kwargs["sm_scale"], kernel="fp32"
            )
            return out.to(kwargs["q"].dtype), lse

        for mask in (None, torch.cat((torch.ones(92), torch.zeros(4))).bool().reshape(1, 1, 96)):
            with (
                self.subTest(masked=mask is not None),
                patch.object(torch.Tensor, "is_cuda", property(lambda self: True)),
                patch.object(attention_module, "_call_sage_kernel", side_effect=kernel) as sage,
            ):
                out = attention_module.attention_sage2(
                    q,
                    k,
                    v,
                    2,
                    mask=mask,
                    skip_reshape=True,
                    skip_output_reshape=True,
                    pv_dtype="fp8",
                    exact_kernel="fp32",
                    transformer_options={"vimoo_sage_text_length": k.shape[2] - img_len},
                )
            reference = torch.nn.functional.scaled_dot_product_attention(
                q.float(), k.float(), v.float(), attn_mask=mask, scale=scale
            )
            torch.testing.assert_close(out.float(), reference, atol=2e-3, rtol=2e-3)
            sage.assert_called_once()
            self.assertEqual(sage.call_args.kwargs["k"].shape[2], img_len)

    @unittest.skipUnless(
        torch.cuda.is_available() and importlib.util.find_spec("sageattention") is not None,
        "the Sage kernel needs its optional CUDA package",
    )
    def test_sage_routes_text_rows_through_exact_math(self):
        """Image-only Sage keeps text rows on exact attention."""
        # fp32 here so the test does not depend on FlashAttention 4 being installed.
        img_len = 64
        q, k, v = (t.cuda().bfloat16() for t in (self.q, self.k, self.v))
        out = attention_module.attention_sage2(
            q,
            k,
            v,
            2,
            skip_reshape=True,
            skip_output_reshape=True,
            pv_dtype="fp8",
            exact_kernel="fp32",
            transformer_options={"vimoo_sage_text_length": k.shape[2] - img_len},
        )
        reference = torch.nn.functional.scaled_dot_product_attention(q[:, :, img_len:], k, v, scale=16**-0.5)
        torch.testing.assert_close(out[:, :, img_len:], reference, atol=2e-2, rtol=2e-2)

    def test_text_rows_equal_full_attention(self):
        img_len, scale = 64, 16**-0.5
        q_txt = self.q[:, :, img_len:]
        rows = exact_partial_attention(q_txt, self.k, self.v, scale, chunk=16, kernel="fp32")[0]
        reference = torch.nn.functional.scaled_dot_product_attention(q_txt, self.k, self.v, scale=scale)
        torch.testing.assert_close(rows, reference, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
