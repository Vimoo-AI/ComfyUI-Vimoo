"""Numerical boundaries and Comfy model lifecycle used by native Vimoo."""

import copy
import gc
from functools import partial
import sys
import unittest
from unittest import mock
from types import MethodType
import weakref

import torch
from test_nodes_runtime import nodes, samplers, spec

precision = sys.modules[f"{spec.name}.vimoo_precision"]
vae_module = sys.modules[f"{spec.name}.vimoo_vae"]


class PrecisionTests(unittest.TestCase):
    def test_double_stream_releases_attention_temporaries_before_image_mlp(self):
        from test_model_runtime import tiny_vimoo

        model = tiny_vimoo().eval()
        block = model.double_blocks[0]
        x = torch.randn(1, 2, 2, 4, 4)
        timestep = torch.tensor([500.0])
        context = torch.randn(1, 3, 6)
        original_attention = precision.optimized_attention
        original_norm = precision.qk_norm
        references = []
        observed = []

        def record_norm(*args, **kwargs):
            q, k = original_norm(*args, **kwargs)
            references.extend((weakref.ref(q), weakref.ref(k)))
            return q, k

        def record_attention(q, k, v, *args, **kwargs):
            for tensor in (q, k, v):
                references.append(weakref.ref(tensor if tensor._base is None else tensor._base))
            result = original_attention(q, k, v, *args, **kwargs)
            references.append(weakref.ref(result))
            return result

        def check_released(module, args):
            observed.append(True)
            self.assertTrue(references)
            self.assertTrue(all(reference() is None for reference in references))

        with torch.inference_mode():
            expected = model(x, timestep, context)
            handle = block.img_mlp.register_forward_pre_hook(check_released)
            try:
                with (
                    mock.patch.object(precision, "qk_norm", new=record_norm),
                    mock.patch.object(precision, "optimized_attention", new=record_attention),
                ):
                    actual = model(x, timestep, context)
            finally:
                handle.remove()
        self.assertEqual(observed, [True])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_vae_conv_keeps_native_staging_and_separate_bias_operation(self):
        from comfy.ldm.wan.vae import CausalConv3d

        conv = CausalConv3d(2, 3, 3, padding=1)
        conv.padding = (0, 0, 0)
        generator = torch.Generator().manual_seed(112)
        with torch.no_grad():
            conv.weight.copy_(torch.randn(conv.weight.shape, generator=generator) * 0.1)
            conv.bias.copy_(torch.randn(conv.bias.shape, generator=generator) * 0.1)
        x = torch.randn(1, 2, 1, 5, 7, generator=generator).bfloat16()
        padded = torch.nn.functional.pad(x, (1, 1, 1, 1, 2, 0))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            expected = torch.nn.functional.conv3d(padded, conv.weight, conv.bias)
            after_bias = torch.nn.functional.conv3d(padded, conv.weight, None)
            after_bias = after_bias + conv.bias.bfloat16().view(1, -1, 1, 1, 1)
        self.assertFalse(torch.equal(expected, after_bias))
        with (
            mock.patch.object(conv, "_conv_forward", side_effect=AssertionError("legacy dispatch")),
            mock.patch.object(
                vae_module.comfy.ops,
                "CastBiasWeightContext",
                wraps=vae_module.comfy.ops.CastBiasWeightContext,
            ) as staging,
            torch.autocast("cpu", dtype=torch.bfloat16),
        ):
            actual = vae_module._causal_conv_forward(conv, x, spatial_padding=(1, 1), explicit_numerical=True)
        staging.assert_called_once()
        self.assertTrue(staging.call_args.kwargs["offloadable"])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(conv.weight.dtype, torch.float32)

    def test_cfg_combines_bf16_flow_before_converting_to_x0(self):
        generator = torch.Generator().manual_seed(18)
        cond, uncond = [torch.randn(2, 48, 2, 2, 2, generator=generator).bfloat16() for _ in range(2)]
        guided = uncond + 4.5 * (cond - uncond)
        rescaled = guided * (
            cond.std(dim=(1, 2, 3, 4), keepdim=True) / guided.std(dim=(1, 2, 3, 4), keepdim=True)
        )
        expected = 1.2 * rescaled + (1.0 - 1.2) * guided
        actual = samplers._combine_vimoo_flow(cond, uncond, 4.5, 1.2, direct_rescale=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        wrong_guided = uncond.float() + 4.5 * (cond.float() - uncond.float())
        wrong_rescaled = wrong_guided * (
            cond.float().std(dim=(1, 2, 3, 4), keepdim=True)
            / wrong_guided.std(dim=(1, 2, 3, 4), keepdim=True)
        )
        wrong = 1.2 * wrong_rescaled + (1.0 - 1.2) * wrong_guided
        self.assertFalse(torch.equal(actual, wrong.bfloat16()))

    def test_cfg_one_preserves_conditional_flow_without_rescale_rounding(self):
        cond = torch.randn(2, 48, 2, 2, 2).bfloat16()
        uncond = torch.randn_like(cond) * 100
        result = samplers._combine_vimoo_flow(cond, uncond, 1.0, 0.7)
        torch.testing.assert_close(result, cond, rtol=0, atol=0)

    def test_modulation_preserves_single_rounding_with_native_addcmul(self):
        from comfy.ldm.flux.layers import ModulationOut

        x = torch.tensor([[[2.0]]], dtype=torch.bfloat16)
        delta = torch.full_like(x, 1 / 256)
        result = precision.modulate(x, ModulationOut(delta, delta, None))
        self.assertEqual(result.item(), 2.015625)
        self.assertEqual(result.dtype, torch.bfloat16)

    def test_precise_gate_accepts_comfy_and_diffusers_modulation_shapes(self):
        x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        gate = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        expected = x * gate.unsqueeze(1)
        torch.testing.assert_close(precision.apply_gate(x, gate), expected)
        torch.testing.assert_close(precision.apply_gate(x, gate.unsqueeze(1)), expected)

    def test_vae_conv_uses_official_bf16_workspace_guard_under_autocast(self):
        from comfy.ldm.wan.vae import CausalConv3d

        conv = CausalConv3d(2, 2, 3, padding=1)
        conv.weight.data.zero_()
        conv.bias.data.zero_()
        with (
            mock.patch.object(nodes.comfy.ops, "NVIDIA_MEMORY_CONV_BUG_WORKAROUND", True),
            mock.patch.object(
                torch, "cudnn_convolution", return_value=torch.zeros(1, 2, 1, 3, 3, dtype=torch.bfloat16)
            ) as native,
            torch.autocast("cpu", dtype=torch.bfloat16),
        ):
            vae_module._causal_conv_forward(conv, torch.ones(1, 2, 1, 3, 3))
        native.assert_called_once()
        self.assertEqual(native.call_args.args[1].dtype, torch.bfloat16)

    def test_vae_rms_inplace_affine_is_bit_exact(self):
        from comfy.ldm.wan.vae import RMS_norm

        generator = torch.Generator().manual_seed(29)
        for dtype in (torch.float32, torch.bfloat16):
            for use_bias in (False, True):
                with self.subTest(dtype=dtype, use_bias=use_bias):
                    module = RMS_norm(8, images=False, bias=use_bias)
                    module.gamma.data.copy_(torch.randn(8, 1, 1, 1, generator=generator))
                    if use_bias:
                        module.bias.data.copy_(torch.randn(8, 1, 1, 1, generator=generator))
                    x = torch.randn(2, 8, 3, 4, 5, generator=generator).to(dtype)
                    normalized = torch.nn.functional.normalize(x, dim=1)
                    gamma = nodes.comfy.ops.cast_to_input(module.gamma, normalized)
                    bias = nodes.comfy.ops.cast_to_input(module.bias, normalized) if use_bias else 0
                    expected = normalized * module.scale * gamma + bias
                    actual = vae_module._rms_forward(module, x)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_vae_silu_reuses_its_input_storage(self):
        x = torch.randn(2, 3, 4)
        expected = torch.nn.functional.silu(x.clone())
        pointer = x.data_ptr()
        actual = vae_module._silu_forward(torch.nn.SiLU(), x)
        self.assertEqual(actual.data_ptr(), pointer)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_vae_fp32_affine_keeps_silu_and_convolution_cast_order(self):
        from comfy.ldm.wan.vae import RMS_norm

        generator = torch.Generator().manual_seed(29)
        for use_bias in (False, True):
            with self.subTest(use_bias=use_bias):
                module = RMS_norm(8, images=False, bias=use_bias)
                module.gamma.data.copy_(torch.randn(8, 1, 1, 1, generator=generator))
                if use_bias:
                    module.bias.data.copy_(torch.randn(8, 1, 1, 1, generator=generator))
                x = torch.randn(2, 8, 3, 4, 5, generator=generator).bfloat16()
                source = torch.nn.functional.normalize(x, dim=1) * module.scale * module.gamma
                source = source + (module.bias if use_bias else 0.0)
                actual = vae_module._rms_forward_fp32_affine(module, x)
                self.assertEqual(actual.dtype, torch.float32)
                torch.testing.assert_close(actual, source, rtol=0, atol=0)
                expected_next_input = torch.nn.functional.silu(source).bfloat16()
                actual_next_input = vae_module._silu_forward(torch.nn.SiLU(), actual).bfloat16()
                torch.testing.assert_close(actual_next_input, expected_next_input, rtol=0, atol=0)
                legacy = vae_module._rms_forward(module, x)
                self.assertEqual(legacy.dtype, torch.bfloat16)
                self.assertFalse(torch.equal(torch.nn.functional.silu(legacy), expected_next_input))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for offloaded affine parameters")
    def test_vae_fp32_affine_accepts_cpu_parameters_with_cuda_activations(self):
        from comfy.ldm.wan.vae import RMS_norm

        for use_bias in (False, True):
            with self.subTest(use_bias=use_bias):
                module = RMS_norm(8, images=False, bias=use_bias)
                module.gamma.data.normal_()
                if use_bias:
                    module.bias.data.normal_()
                x = torch.randn(1, 8, 2, 3, 5, device="cuda", dtype=torch.bfloat16)
                expected = torch.nn.functional.normalize(x, dim=1) * module.scale * module.gamma.to(x.device)
                expected = expected + (module.bias.to(x.device) if use_bias else 0.0)
                actual = vae_module._rms_forward_fp32_affine(module, x)
                self.assertEqual(actual.dtype, torch.float32)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(module.gamma.device.type, "cpu")
                self.assertEqual(module.gamma.dtype, torch.float32)
                if use_bias:
                    self.assertEqual(module.bias.device.type, "cpu")
                    self.assertEqual(module.bias.dtype, torch.float32)

    def test_vae_attention_preserves_packed_qkv_and_direct_sdpa(self):
        from comfy.ldm.wan.vae import AttentionBlock

        module = AttentionBlock(8)
        module.norm.forward = MethodType(vae_module._rms_forward_fp32_affine, module.norm)
        with torch.no_grad():
            generator = torch.Generator().manual_seed(123)
            for parameter in module.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.17)
        sdpa = torch.nn.functional.scaled_dot_product_attention
        observed = []

        def source_sdpa(q, k, v):
            observed.append((q, k, v))
            return sdpa(q, k, v)

        for dtype, height, width in (
            (torch.float32, 3, 4),
            (torch.bfloat16, 3, 4),
            (torch.float32, 1, 1),
            (torch.bfloat16, 1, 1),
        ):
            x = torch.randn(1, 8, 2, height, width).to(dtype)
            with (
                torch.no_grad(),
                torch.autocast("cpu", dtype=torch.bfloat16),
                mock.patch.object(
                    torch.nn.functional, "scaled_dot_product_attention", side_effect=source_sdpa
                ),
                mock.patch.object(
                    module, "optimized_attention", side_effect=AssertionError("attention uses direct SDPA")
                ),
            ):
                result = vae_module._attention_forward_sdpa(module, x)
            self.assertEqual(result.dtype, dtype)
            self.assertEqual(tuple(result.shape), tuple(x.shape))
            q, k, v = observed[-1]
            self.assertEqual(tuple(q.shape), (2, 1, height * width, 8))
            self.assertEqual(q.dtype, torch.bfloat16)
            self.assertEqual(q.stride()[-2:], (24, 1))
            self.assertEqual(
                {value.untyped_storage().data_ptr() for value in (q, k, v)}, {q.untyped_storage().data_ptr()}
            )
            self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(len(observed), 4)

    def test_vae_native_attention_only_changes_single_position_inputs(self):
        from comfy.ldm.wan.vae import AttentionBlock

        module = AttentionBlock(8)
        # ComfyUI skips parameter initialization because production loads weights.
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.fill_(0.1)
        single = torch.randn(1, 8, 2, 1, 1)
        with mock.patch.object(module, "optimized_attention", side_effect=AssertionError("unaligned kernel")):
            result = vae_module._attention_forward_native(module, single)
        self.assertEqual(result.shape, single.shape)
        self.assertTrue(torch.isfinite(result).all())

        sample = torch.randn(1, 8, 2, 3, 4)
        with mock.patch.object(AttentionBlock, "forward", return_value=sample) as native:
            result = vae_module._attention_forward_native(module, sample)
        self.assertIs(result, sample)
        native.assert_called_once_with(module, sample, feat_cache=None, feat_idx=None, final=False)

    def test_vae_model_binding_survives_native_reload_clone_and_offload(self):
        from types import SimpleNamespace
        from comfy.ldm.wan.vae import AttentionBlock, RMS_norm
        from comfy.ldm.wan.vae2_2 import ResidualBlock, Up_ResidualBlock

        def initialize(owner, *args, **kwargs):
            del args, kwargs
            model = torch.nn.Module()
            model.norm = RMS_norm(8, images=False)
            model.norm.gamma.data.fill_(1.003)
            model.activation = torch.nn.SiLU()
            model.block = ResidualBlock(8, 8)
            model.up = Up_ResidualBlock(8, 8, dropout=0.0, mult=1, up_flag=True)
            model.attention = AttentionBlock(8)
            with torch.no_grad():
                for parameter in model.block.parameters():
                    parameter.fill_(0.03)
                for parameter in model.up.parameters():
                    parameter.fill_(0.03)
            owner.first_stage_model = model
            owner.device = owner.output_device = torch.device("cpu")
            owner.vae_dtype = torch.float32
            owner.latent_channels = 48
            owner.memory_used_encode = owner.memory_used_decode = lambda shape, dtype: 1
            owner.patcher = nodes.comfy.model_patcher.ModelPatcher(
                model, torch.device("cpu"), torch.device("cpu")
            )

        model = SimpleNamespace(
            model_options={
                "vimoo_vae_numerical_profile": "wan_fp32_norm",
                "vimoo_sampling_profile": "standard_i2v",
            }
        )
        with (
            mock.patch.object(nodes.comfy.sd.VAE, "__init__", new=initialize),
            mock.patch.object(vae_module.folder_paths, "get_full_path_or_raise", return_value="vae"),
            mock.patch.object(nodes.comfy.utils, "load_torch_file", return_value=({}, {})),
        ):
            loader = vae_module.VimooVAELoader()
            legacy = vae_module._load_vimoo_vae("vae", "auto", "comfy", 256, 256, 192, 192, 2.0, None)
            source = loader.load_vae("vae", model=model)[0]
            factory, args = source.patcher.cached_patcher_init
            self.assertEqual(args[-1], "wan_fp32_norm")
            self.assertNotIn(model, args)
            reloaded = factory(*args)
        self.assertIsNone(legacy.numerical_profile)
        self.assertIs(legacy.patcher.object_patches["norm.forward"].__func__, vae_module._rms_forward)
        self.assertIn("block.forward", legacy.patcher.object_patches)
        self.assertIn("up.forward", legacy.patcher.object_patches)
        for name in ("block.forward", "up.forward"):
            self.assertTrue(source.patcher.object_patches[name].__func__.keywords["explicit_numerical"])
        self.assertIs(
            legacy.patcher.object_patches["attention.forward"].__func__, vae_module._attention_forward_native
        )
        self.assertIs(
            source.patcher.object_patches["attention.forward"].__func__, vae_module._attention_forward_sdpa
        )
        self.assertIs(
            source.patcher.object_patches["norm.forward"].__func__, vae_module._rms_forward_fp32_affine
        )
        fresh = copy.deepcopy(source.first_stage_model)
        fresh_patcher = nodes.comfy.model_patcher.ModelPatcher(
            fresh, torch.device("cpu"), torch.device("cpu")
        )
        clone = source.patcher.clone(model_override=fresh_patcher.get_clone_model_override())
        for patcher in (source.patcher, clone, reloaded):
            x = torch.randn(1, 8, 2, 3, 4).bfloat16()
            expected = torch.nn.functional.normalize(x, dim=1) * 8**0.5 * patcher.model.norm.gamma + 0.0
            patcher.patch_model(device_to=torch.device("cpu"))
            try:
                self.assertEqual(patcher.model.norm.gamma.dtype, torch.float32)
                torch.testing.assert_close(patcher.model.norm(x), expected, rtol=0, atol=0)
                self.assertIs(patcher.model.norm.forward.__self__, patcher.model.norm)
                with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                    block = patcher.model.block
                    expected = block.residual(x.float()) + block.shortcut(x.float())
                    actual = block(x.float())
                    self.assertEqual(actual.dtype, torch.float32)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    up = patcher.model.up
                    expected = x.float()
                    for layer in up.upsamples:
                        expected = layer(expected)
                    expected = expected + up.avg_shortcut(x.float())
                    actual = up(x.float())
                    self.assertEqual(actual.dtype, torch.float32)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            finally:
                patcher.unpatch_model(device_to=torch.device("cpu"))
            self.assertEqual(patcher.model.norm.gamma.dtype, torch.float32)
        with self.assertRaisesRegex(ValueError, "Unknown Vimoo VAE numerical profile"):
            vae_module.VimooVAE({}, numerical_profile="unknown")

    def test_bf16_autocast_memory_estimate_keeps_headroom(self):
        seen = []

        def estimate(shape, dtype):
            seen.append((shape, dtype))
            return 10_000 * nodes.comfy.model_management.dtype_size(dtype)

        actual = vae_module._bf16_autocast_memory_estimate(
            (1, 48, 31, 45, 80), torch.float32, estimate=estimate
        )
        self.assertEqual(seen, [((1, 48, 31, 45, 80), torch.bfloat16)])
        self.assertEqual(actual, 23_000)

    def test_vae_memory_reservation_reaches_native_framework_before_decode(self):
        from comfy.ldm.wan.vae2_2 import Decoder3d

        with torch.device("meta"):
            decoder = Decoder3d(dim=256, z_dim=48, temperal_upsample=[True, True, False])
        coefficients = vae_module._source_decode_memory_coefficients(decoder)
        native_estimate = lambda shape, dtype: 8000 * shape[-2] * shape[-1] * 256 * dtype.itemsize
        reserve = partial(
            vae_module._source_decode_memory_estimate, estimate=native_estimate, coefficients=coefficients
        )
        shape = (1, 48, 31, 45, 80)
        # The accepted 720p source-profile diagnostic peaked at 40.657 GB.
        # Its old 29.491 GB reservation let other models remain resident.
        self.assertGreater(reserve(shape, torch.float32), 40_656_820_224)
        self.assertEqual(reserve((1, 48, 31, 45, 40), torch.float32), reserve(shape, torch.float32) // 2)
        self.assertEqual(reserve((1, 48, 2, 45, 80), torch.float32), reserve(shape, torch.float32))
        self.assertLess(reserve((1, 48, 1, 45, 80), torch.float32), reserve(shape, torch.float32))

        vae = object.__new__(vae_module.VimooVAE)
        vae.device = vae.output_device = torch.device("cpu")
        vae.decode_mode = "auto"
        vae.convolution_backend = "comfy"
        vae.latent_dim = 3
        vae.disable_offload = False
        vae.vae_dtype = torch.float32
        vae.vae_output_dtype = lambda: torch.float32
        vae.memory_used_decode = reserve
        vae.throw_exception_if_invalid = mock.Mock()
        vae.patcher = mock.Mock()
        vae.patcher.get_free_memory.return_value = reserve(shape, torch.float32)
        vae.process_output = lambda value: value
        vae.first_stage_model = mock.Mock()
        vae.first_stage_model.comfy_has_chunked_io = True
        vae.first_stage_model.decode_output_shape.return_value = (1, 3, 1, 2, 2)
        vae.first_stage_model.decode.side_effect = lambda samples, output_buffer: output_buffer.zero_()
        with mock.patch.object(nodes.comfy.model_management, "load_models_gpu") as load:
            decoded = vae.decode(torch.zeros(shape))
        load.assert_called_once_with(
            [vae.patcher], memory_required=reserve(shape, torch.float32), force_full_load=False
        )
        self.assertEqual(vae.last_decode_mode, "untiled")
        self.assertEqual(decoded.shape, (1, 1, 2, 2, 3))

    def test_chunked_vae_encode_accepts_comfy_device_argument(self):
        model = mock.Mock()
        pixels = torch.ones(1, 3, 1, 2, 2)
        with mock.patch.object(vae_module.Wan22VAE, "encode", return_value="encoded") as native:
            result = vae_module._wan_encode_with_device(model, pixels, device=torch.device("cpu"))
        self.assertEqual(result, "encoded")
        native.assert_called_once_with(model, pixels)

    def test_both_backends_use_explicit_padding_and_im2col_adds_a_bound(self):
        from comfy.ldm.wan.vae import CausalConv3d, RMS_norm
        from comfy.ldm.wan.vae2_2 import ResidualBlock

        def initialize(owner, *args, **kwargs):
            del args, kwargs
            model = torch.nn.Module()
            model.norm = RMS_norm(2, images=False)
            model.activation = torch.nn.SiLU()
            model.conv = CausalConv3d(2, 2, 3, padding=1)
            model.block = ResidualBlock(2, 2)
            owner.first_stage_model = model
            owner.device = owner.output_device = torch.device("cpu")
            owner.vae_dtype = torch.float32
            owner.latent_channels = 48
            owner.memory_used_encode = owner.memory_used_decode = lambda shape, dtype: 1
            owner.patcher = nodes.comfy.model_patcher.ModelPatcher(
                model, torch.device("cpu"), torch.device("cpu")
            )

        with mock.patch.object(nodes.comfy.sd.VAE, "__init__", new=initialize):
            comfy_vae = vae_module.VimooVAE({})
            reference_vae = vae_module.VimooVAE({}, convolution_backend="im2col_bounded")

        for vae in (comfy_vae, reference_vae):
            self.assertIn("conv.forward", vae.patcher.object_patches)
            self.assertIn("conv.padding", vae.patcher.object_patches)
            self.assertIn("norm.forward", vae.patcher.object_patches)
            self.assertIn("activation.forward", vae.patcher.object_patches)
            self.assertIn("block.forward", vae.patcher.object_patches)
            self.assertIn("encode", vae.patcher.object_patches)
            self.assertIn("decode", vae.patcher.object_patches)
            self.assertTrue(vae.first_stage_model.comfy_has_chunked_io)
        self.assertFalse(comfy_vae.patcher.object_patches["conv.forward"].__func__.keywords["im2col_bounded"])
        self.assertTrue(
            reference_vae.patcher.object_patches["conv.forward"].__func__.keywords["im2col_bounded"]
        )

    def test_vae_shape_callback_does_not_retain_released_model(self):
        def initialize(owner, *args, **kwargs):
            del args, kwargs
            model = torch.nn.Module()
            owner.first_stage_model = model
            owner.device = owner.output_device = torch.device("cpu")
            owner.vae_dtype = torch.float32
            owner.latent_channels = 48
            owner.memory_used_encode = owner.memory_used_decode = lambda shape, dtype: 1
            owner.patcher = nodes.comfy.model_patcher.ModelPatcher(
                model, torch.device("cpu"), torch.device("cpu")
            )

        was_enabled = gc.isenabled()
        gc.disable()
        try:
            with mock.patch.object(nodes.comfy.sd.VAE, "__init__", new=initialize):
                vae = vae_module.VimooVAE({})
            self.assertEqual(
                vae.first_stage_model.decode_output_shape((1, 48, 2, 16, 16)),
                (1, 3, 5, 256, 256),
            )
            model_reference = weakref.ref(vae.first_stage_model)
            del vae
            self.assertIsNone(model_reference())
        finally:
            if was_enabled:
                gc.enable()
            gc.collect()

    def test_im2col_conv_rejects_large_columns_before_kernel_allocation(self):
        from comfy.ldm.wan.vae import CausalConv3d

        conv = CausalConv3d(2, 2, 3, padding=1)
        with (
            mock.patch.object(torch.Tensor, "is_cuda", property(lambda self: True)),
            mock.patch.object(vae_module, "_IM2COL_WORKSPACE_LIMIT", 1),
            mock.patch.object(torch.ops.aten.slow_conv_dilated3d, "default") as kernel,
            torch.autocast("cpu", dtype=torch.bfloat16),
        ):
            with self.assertRaises(nodes.comfy.model_management.OOM_EXCEPTION):
                vae_module._causal_conv_forward(conv, torch.ones(1, 2, 1, 3, 3), im2col_bounded=True)
        kernel.assert_not_called()

    def test_im2col_conv_custom_budget_is_local_and_inclusive(self):
        from comfy.ldm.wan.vae import CausalConv3d

        conv = CausalConv3d(2, 2, 3, padding=1)
        with torch.no_grad():
            conv.weight.fill_(1)
            conv.bias.zero_()
        inputs = torch.ones(1, 2, 1, 3, 3)
        expected = (
            torch.tensor([[8, 12, 8], [12, 18, 12], [8, 12, 8]], dtype=torch.bfloat16)
            .reshape(1, 1, 1, 3, 3)
            .expand(1, 2, 1, 3, 3)
        )
        # The padded convolution requires exactly 972 bytes of BF16 columns.
        # An explicit ceiling permits that boundary without relaxing defaults.
        with (
            mock.patch.object(torch.Tensor, "is_cuda", property(lambda self: True)),
            mock.patch.object(vae_module, "_IM2COL_WORKSPACE_LIMIT", 971),
            mock.patch.object(torch.ops.aten.slow_conv_dilated3d, "default", return_value=expected) as kernel,
            mock.patch.object(torch.nn.functional, "conv3d") as dispatch,
            torch.autocast("cpu", dtype=torch.bfloat16),
        ):
            result = vae_module._causal_conv_forward(
                conv, inputs, spatial_padding=(1, 1), im2col_bounded=True, im2col_workspace_limit=972
            )
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            kernel.assert_called_once()
            padded, weight, size, bias, stride, padding, dilation = kernel.call_args.args
            self.assertEqual(padded.dtype, torch.bfloat16)
            self.assertEqual(weight.dtype, torch.bfloat16)
            self.assertEqual(bias.dtype, torch.bfloat16)
            self.assertEqual((size, stride, padding, dilation), ((3, 3, 3), (1, 1, 1), (0, 0, 0), (1, 1, 1)))
            with self.assertRaises(nodes.comfy.model_management.OOM_EXCEPTION):
                vae_module._causal_conv_forward(conv, inputs, spatial_padding=(1, 1), im2col_bounded=True)
            kernel.assert_called_once()
            dispatch.assert_not_called()

    def test_im2col_backend_does_not_require_exact_torch_or_cudnn_versions(self):
        def initialize(owner, *args, **kwargs):
            owner.first_stage_model = torch.nn.Module()
            owner.device = torch.device("cuda")
            owner.output_device = torch.device("cpu")
            owner.vae_dtype = torch.float32
            owner.latent_channels = 48
            owner.memory_used_encode = owner.memory_used_decode = lambda shape, dtype: 1
            owner.patcher = nodes.comfy.model_patcher.ModelPatcher(
                owner.first_stage_model, torch.device("cpu"), torch.device("cpu")
            )

        for version in ("2.9.0+cu128", "2.10.0+cu128", "2.11.0+cu128"):
            with (
                self.subTest(version=version),
                mock.patch.object(nodes.comfy.sd.VAE, "__init__", new=initialize),
                mock.patch.object(torch, "__version__", version),
                mock.patch.object(torch.version, "hip", None),
                mock.patch.object(
                    torch.backends.cudnn, "version", side_effect=AssertionError("No cuDNN dispatch gate")
                ),
            ):
                vae = vae_module.VimooVAE({}, convolution_backend="im2col_bounded")
                self.assertEqual(vae.convolution_backend, "im2col_bounded")

        with (
            mock.patch.object(nodes.comfy.sd.VAE, "__init__", new=initialize),
            mock.patch.object(torch.version, "hip", "7.0"),
        ):
            with self.assertRaisesRegex(RuntimeError, "NVIDIA CUDA GPU"):
                vae_module.VimooVAE({}, convolution_backend="im2col_bounded")

    def test_im2col_workspace_budget_reuses_buffer_between_batch_entries(self):
        from comfy.ldm.wan.vae import CausalConv3d

        conv = CausalConv3d(2, 4, 3, padding=0)
        # Output volume is 1*3*3; columns are BF16 Cin*27*volume.
        for batch in (1, 3):
            self.assertEqual(
                vae_module._conv_columns_bytes(conv, torch.empty(batch, 2, 3, 5, 5, dtype=torch.bfloat16)),
                2 * 2 * 27 * 9,
            )

    def test_qk_norm_honors_comfy_weight_functions(self):
        from comfy.ldm.flux.layers import QKNorm

        norm = QKNorm(4, dtype=torch.float32, device="cpu", operations=nodes.comfy.ops.manual_cast)
        norm.query_norm.weight.data.fill_(1)
        norm.key_norm.weight.data.fill_(1)
        q, k = torch.randn(2, 2, 3, 4), torch.randn(2, 2, 3, 4)
        before = precision.qk_norm(q, k, norm)
        norm.query_norm.weight_function = [lambda weight: weight * 2]
        after = precision.qk_norm(q, k, norm)
        torch.testing.assert_close(after[0], before[0] * 2, rtol=0, atol=0)
        torch.testing.assert_close(after[1], before[1], rtol=0, atol=0)

    def test_vae_causal_adapter_is_per_instance_and_consumes_cache(self):
        from comfy.ldm.wan.vae import CausalConv3d

        original = CausalConv3d.forward
        conv = CausalConv3d(2, 2, 3, padding=1)
        conv.weight.data.normal_(0, 0.1)
        conv.bias.data.zero_()
        conv.forward = MethodType(vae_module._causal_conv_forward, conv)
        x, cache = torch.randn(1, 2, 1, 3, 3), torch.randn(1, 2, 2, 3, 3)
        cache_list = [cache]
        with mock.patch.object(conv, "_conv_forward", wraps=conv._conv_forward) as native:
            actual = conv(x, cache_list=cache_list, cache_idx=0)
        native.assert_called_once()
        expected = torch.nn.functional.conv3d(
            torch.nn.functional.pad(torch.cat((cache, x), dim=2), (1, 1, 1, 1, 0, 0)), conv.weight, conv.bias
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertIsNone(cache_list[0])
        self.assertIs(CausalConv3d.forward, original)

    def test_dup_up_shortcut_add_matches_materialized_update(self):
        from comfy.ldm.wan.vae2_2 import DupUp3D

        generator = torch.Generator().manual_seed(41)
        for in_channels, out_channels, factor_t in ((4, 4, 2), (4, 2, 1)):
            shortcut = DupUp3D(in_channels, out_channels, factor_t=factor_t, factor_s=2)
            source = torch.randn(1, in_channels, 3, 2, 3, generator=generator, dtype=torch.bfloat16)
            for first_chunk in (False, True):
                with self.subTest(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    factor_t=factor_t,
                    first_chunk=first_chunk,
                ):
                    expanded = shortcut(source, first_chunk)
                    main = torch.randn(expanded.shape, generator=generator).bfloat16()
                    expected = main + expanded
                    actual = vae_module._add_dup_up_shortcut_(main.clone(), source, shortcut, first_chunk)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_wan_decode_streams_into_output_buffer_and_matches_direct_result(self):
        class Decoder(torch.nn.Module):
            def forward(self, x, feat_cache, feat_idx, first_chunk=False):
                del feat_cache, feat_idx
                frames = 1 if first_chunk else 4
                value = 1 if first_chunk else 2
                return x.new_full((x.shape[0], 12, frames, x.shape[3] * 8, x.shape[4] * 8), value)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv2 = torch.nn.Identity()
                self.decoder = Decoder()

        model = Model()
        latent = torch.zeros(1, 1, 3, 1, 2)
        direct = vae_module._wan_decode_streaming(model, latent)
        self.assertEqual(tuple(direct.shape), (1, 3, 9, 16, 32))
        output = torch.empty_like(direct)
        returned = vae_module._wan_decode_streaming(model, latent, output_buffer=output)
        self.assertIs(returned, output)
        torch.testing.assert_close(output, direct, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "output buffer shape mismatch"):
            vae_module._wan_decode_streaming(model, latent, output_buffer=output[:, :, :-1])

    def test_small_wan22_decoder_is_bit_exact_with_all_activation_patches(self):
        from comfy.ldm.wan.vae import CausalConv3d, RMS_norm
        from comfy.ldm.wan.vae2_2 import ResidualBlock, Up_ResidualBlock, WanVAE

        generator = torch.Generator().manual_seed(53)
        reference = WanVAE(dim=2, dec_dim=4, z_dim=4)
        for parameter in reference.parameters():
            parameter.data.copy_(
                torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype) * 0.03
            )
        optimized = copy.deepcopy(reference)
        for module in reference.modules():
            if isinstance(module, CausalConv3d):
                spatial_padding = module.padding[1:]
                module.forward = MethodType(
                    partial(vae_module._causal_conv_forward, spatial_padding=spatial_padding),
                    module,
                )
                module.padding = (0, 0, 0)
        for _, module in optimized.named_modules():
            if isinstance(module, RMS_norm):
                module.forward = MethodType(vae_module._rms_forward, module)
            elif isinstance(module, torch.nn.SiLU):
                module.forward = MethodType(vae_module._silu_forward, module)
            elif isinstance(module, ResidualBlock):
                module.forward = MethodType(vae_module._residual_forward_inplace, module)
            elif isinstance(module, Up_ResidualBlock):
                module.forward = MethodType(vae_module._up_residual_forward_inplace, module)
            elif isinstance(module, CausalConv3d):
                spatial_padding = module.padding[1:]
                module.forward = MethodType(
                    partial(vae_module._causal_conv_forward, spatial_padding=spatial_padding),
                    module,
                )
                module.padding = (0, 0, 0)
        optimized.decode = MethodType(vae_module._wan_decode_streaming, optimized)

        latent = torch.randn(1, 4, 2, 2, 2, generator=generator)
        with torch.no_grad():
            expected = reference.decode(latent)
            actual = optimized.decode(latent)
            streamed = torch.empty_like(expected)
            optimized.decode(latent, output_buffer=streamed)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(streamed, expected, rtol=0, atol=0)

    def test_vision_epsilon_uses_patcher_clone_and_survives_reload(self):
        from types import SimpleNamespace

        vision = sys.modules[f"{spec.name}.vimoo_vision"]
        model = torch.nn.Module()
        model.vision_model = torch.nn.Sequential(torch.nn.LayerNorm(4), torch.nn.Linear(4, 4))
        base = nodes.comfy.model_patcher.ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
        clip = SimpleNamespace(model=model, patcher=base)
        patcher = vision._vision_patcher(clip)
        self.assertIsNot(patcher, base)
        self.assertFalse(base.object_patches)
        original_forward = model.vision_model[1].forward.__func__
        for _ in range(2):
            self.assertEqual(model.vision_model[0].eps, 1e-5)
            try:
                patcher.patch_model(load_weights=False)
                self.assertEqual(model.vision_model[0].eps, 1e-6)
                self.assertIs(model.vision_model[1].forward.__func__, original_forward)
                raise RuntimeError("interrupted")
            except RuntimeError:
                pass
            finally:
                patcher.unpatch_model(unpatch_weights=False)
            self.assertEqual(model.vision_model[0].eps, 1e-5)

    def test_cfg_preserves_external_guidance_and_post_callbacks(self):
        from test_nodes_runtime import RuntimeTests

        fixture = RuntimeTests()
        model = fixture.make_model()
        x = torch.full((1, 48, 2, 2, 2), 4.0)
        cond, uncond = torch.ones_like(x), torch.zeros_like(x)
        external_cfg = lambda args: torch.full_like(args["input"], 0.75)

        def post(args):
            torch.testing.assert_close(args["cond_denoised"], cond)
            return args["denoised"] + 10

        model.set_model_sampler_cfg_function(external_cfg, disable_cfg1_optimization=True)
        model.set_model_sampler_post_cfg_function(post)
        with mock.patch.object(nodes.comfy.sample, "sample_custom", return_value=x) as call:
            fixture.sample(model, {"samples": x}, guidance_rescale=0.0, sampler_name="euler")
        used = call.call_args.args[0]
        self.assertIs(used.model_options["sampler_cfg_function"], external_cfg)
        self.assertTrue(used.model_options["disable_cfg1_optimization"])
        actual = nodes.comfy.samplers.cfg_function(
            used.model, cond, uncond, 3.0, x, torch.tensor([0.5]), model_options=used.model_options
        )
        torch.testing.assert_close(actual, x - 0.75 + 10)
        self.assertEqual(model.model_options["sampler_post_cfg_function"], [post])

    def test_rescale_composes_with_guided_prediction(self):
        x = torch.ones(1, 1, 1, 2, 2)
        prediction = torch.tensor([[[[[0.0, 1.0], [2.0, 3.0]]]]])
        args = {"input": x, "denoised": prediction, "cond_denoised": prediction, "cond_scale": 5.5}
        torch.testing.assert_close(samplers._make_vimoo_rescale_function(0.5)(args), prediction)

    def test_vae_decode_accepts_native_positional_options(self):
        vae = object.__new__(vae_module.VimooVAE)
        vae.device = torch.device("cpu")
        vae.decode_mode = "auto"
        vae.convolution_backend = "comfy"
        samples = torch.zeros(1, 48, 1, 2, 2)
        with mock.patch.object(nodes.comfy.sd.VAE, "decode", return_value="decoded") as call:
            self.assertEqual(vae.decode(samples, {}), "decoded")
        self.assertEqual(call.call_args.args[1], {})

    def test_vae_fallback_logs_effective_native_tiles_after_backend_cap(self):
        for backend, expected in (
            ("comfy", (7, 32, 24, (1, 8, 6))),
            ("im2col_bounded", (7, 16, 16, (1, 4, 4))),
        ):
            with self.subTest(backend=backend):
                vae = object.__new__(vae_module.VimooVAE)
                vae.convolution_backend = backend
                samples = torch.zeros(1, 48, 7, 45, 80)
                with (
                    mock.patch.object(
                        nodes.comfy.sd.VAE, "decode_tiled_3d", return_value="decoded"
                    ) as decode,
                    self.assertLogs(level="INFO") as logs,
                ):
                    self.assertEqual(vae.decode_tiled_3d(samples, 7, 32, 24, (1, 8, 6)), "decoded")
                self.assertIs(decode.call_args.args[0], samples)
                self.assertEqual(decode.call_args.args[1:], expected)
                self.assertEqual(vae.last_decode_mode, "comfy_tiled")
                self.assertIn("actual=comfy_tiled", logs.output[0])
                self.assertIn(f"tile_latent_txy={expected[:3]}", logs.output[0])
                self.assertIn(f"overlap_latent_txy={expected[3]}", logs.output[0])

    def test_tiled_decode_forwards_native_options_and_bounds_tiles(self):
        seen = []

        def decode(tile, *, marker):
            seen.append((tile.shape, marker))
            return torch.ones(tile.shape[0], 3, 1, tile.shape[3] * 16, tile.shape[4] * 16)

        model = mock.Mock(decode=decode)
        result = vae_module._spatial_tiled_decode(model, torch.zeros(1, 48, 1, 17, 20), {"marker": 7})
        self.assertEqual(result.shape, (1, 3, 1, 272, 320))
        self.assertEqual(len(seen), 4)
        self.assertTrue(all(shape[3] <= 16 and shape[4] <= 16 and marker == 7 for shape, marker in seen))
        self.assertTrue(torch.all(result == 1))

    def test_method_patches_rebind_fresh_models_without_overwriting_other_patches(self):
        patching = sys.modules[f"{spec.name}.vimoo_precision"]

        def make(value):
            model = torch.nn.Module()
            model.layer = torch.nn.Linear(1, 1, bias=False)
            model.layer.weight.data.fill_(value)
            return nodes.comfy.model_patcher.ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))

        def forward(module, x):
            return x * module.weight + 1

        source, fresh = make(2), make(7)
        patching.add_method_patches(
            source, "test_math", {"layer.forward": MethodType(forward, source.model.layer)}
        )
        clone = source.clone(model_override=fresh.get_clone_model_override())
        self.assertIs(clone.object_patches["layer.forward"].__self__, fresh.model.layer)
        try:
            clone.patch_model(load_weights=False)
            self.assertEqual(clone.model.layer(torch.ones(1)).item(), 8)
            self.assertEqual(source.model.layer(torch.ones(1)).item(), 2)
        finally:
            clone.unpatch_model(unpatch_weights=False)
        other = lambda x: x + 20
        source.add_object_patch("layer.forward", other)
        clone = source.clone(model_override=fresh.get_clone_model_override())
        self.assertIs(clone.object_patches["layer.forward"], other)

    def test_tiled_vae_obeys_memory_managed_batch_size(self):
        from nodes import VAEDecode

        vae = object.__new__(vae_module.VimooVAE)
        vae.device = vae.output_device = torch.device("cpu")
        vae.decode_mode = "tiled"
        vae.convolution_backend = "comfy"
        vae.tile_height, vae.tile_width = 576, 512
        vae.stride_height, vae.stride_width = 512, 448
        vae.vae_dtype = torch.float32
        vae.first_stage_model = object()
        vae.prepare_decode = mock.Mock(return_value=1)
        vae.process_output = lambda x: x
        vae.vae_output_dtype = lambda: torch.float32
        with mock.patch.object(
            vae_module, "_spatial_tiled_decode", return_value=torch.zeros(1, 3, 1, 592, 528)
        ) as decode:
            (result,) = VAEDecode().decode(vae, {"samples": torch.zeros(2, 48, 1, 37, 33)})
        self.assertEqual(result.shape, (2, 592, 528, 3))
        self.assertEqual(vae.last_decode_mode, "reference_tiled")
        self.assertTrue(all(call.args[3:] == (576, 512, 512, 448) for call in decode.call_args_list))
        self.assertEqual(decode.call_count, 2)
        self.assertTrue(all(call.args[1].shape[0] == 1 for call in decode.call_args_list))

    def test_tiled_vae_honors_small_tiles_for_small_images(self):
        vae = object.__new__(vae_module.VimooVAE)
        vae.device = vae.output_device = torch.device("cpu")
        vae.decode_mode = "tiled"
        vae.convolution_backend = "comfy"
        vae.tile_height = vae.tile_width = 64
        vae.stride_height = vae.stride_width = 48
        vae.vae_dtype = torch.float32
        vae.first_stage_model = object()
        vae.prepare_decode = mock.Mock(return_value=1)
        vae.process_output = lambda x: x
        vae.vae_output_dtype = lambda: torch.float32
        with mock.patch.object(
            vae_module, "_spatial_tiled_decode", return_value=torch.zeros(1, 3, 1, 144, 256)
        ) as decode:
            result = vae.decode(torch.zeros(1, 48, 1, 9, 16))
        self.assertEqual(result.shape, (1, 1, 144, 256, 3))
        self.assertEqual(vae.last_decode_mode, "reference_tiled")
        vae.prepare_decode.assert_called_once_with((1, 48, 1, 4, 4))
        self.assertEqual(decode.call_args.args[3:], (64, 64, 48, 48))

    def test_tiled_vae_reserves_memory_for_selected_spatial_tiles(self):
        for height, width, expected in (
            (576, 512, (1, 48, 3, 36, 32)),
            (128, 160, (1, 48, 3, 8, 10)),
            (256, 256, (1, 48, 3, 16, 16)),
        ):
            with self.subTest(tile_height=height, tile_width=width):
                vae = object.__new__(vae_module.VimooVAE)
                vae.device = vae.output_device = torch.device("cpu")
                vae.decode_mode = "tiled"
                vae.convolution_backend = "comfy"
                vae.tile_height, vae.tile_width = height, width
                vae.stride_height, vae.stride_width = height, width
                vae.vae_dtype = torch.float32
                vae.first_stage_model = object()
                vae.prepare_decode = mock.Mock(return_value=1)
                vae.process_output = lambda x: x
                vae.vae_output_dtype = lambda: torch.float32
                with mock.patch.object(
                    vae_module, "_spatial_tiled_decode", return_value=torch.zeros(1, 3, 1, 16, 16)
                ):
                    vae.decode(torch.zeros(1, 48, 3, 45, 80))
                vae.prepare_decode.assert_called_once_with(expected)

    def test_spatial_tiling_accepts_full_width_nonoverlapping_tiles(self):
        seen = []

        def decode(tile, **kwargs):
            seen.append(tile.shape)
            return torch.ones(tile.shape[0], 3, 1, tile.shape[3] * 16, tile.shape[4] * 16)

        model = mock.Mock(decode=decode)
        result = vae_module._spatial_tiled_decode(
            model,
            torch.zeros(1, 48, 1, 45, 80),
            tile_height=368,
            tile_width=1280,
            stride_height=368,
            stride_width=1280,
        )
        self.assertEqual(result.shape, (1, 3, 1, 720, 1280))
        self.assertEqual([shape[3:] for shape in seen], [(23, 80), (22, 80)])

    def test_tiled_vae_blends_before_unpacking_and_clamping(self):
        previous = torch.full((1, 12, 1, 4, 2), 2.0, dtype=torch.bfloat16)
        current = torch.full_like(previous, -2.0)
        result = vae_module._blend_tiles(previous, current, axis=3, extent=4)
        expected = torch.tensor([2.0, 1.0, 0.0, -1.0], dtype=torch.bfloat16)
        torch.testing.assert_close(result[0, 0, 0, :, 0], expected, rtol=0, atol=0)
        self.assertIs(result, current)
        # Clamping individual tiles before blending would give 0.5 here.
        self.assertEqual(result.clamp(-1, 1)[0, 0, 0, 1, 0].item(), 1.0)
        short = torch.full((1, 12, 1, 1, 2), -2.0, dtype=torch.bfloat16)
        vae_module._blend_tiles(previous, short, axis=3)
        self.assertTrue(torch.all(short == 2.0))

    def test_vectorized_tile_blend_is_bit_exact_for_every_overlap_extent(self):
        generator = torch.Generator().manual_seed(7)
        previous = torch.randn(1, 3, 2, 35, 36, generator=generator).bfloat16()
        source = torch.randn(1, 3, 2, 35, 36, generator=generator).bfloat16()
        for axis in (3, 4):
            for extent in range(1, 33):
                with self.subTest(axis=axis, extent=extent):
                    expected = source.clone()
                    for index in range(extent):
                        expected.select(axis, index).copy_(
                            previous.select(axis, previous.shape[axis] - extent + index)
                            * (1 - index / extent)
                            + expected.select(axis, index) * (index / extent)
                        )
                    actual = vae_module._blend_tiles(previous, source.clone(), axis, extent)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_latent_statistics_use_bf16_boundaries(self):
        fmt = vae_module.VimooLatentFormat()
        x = torch.randn(1, 48, 2, 2, 2)
        inverse_std = (1.0 / fmt.latents_std).bfloat16()
        expected = (x.bfloat16() - fmt.latents_mean.bfloat16()) * inverse_std
        torch.testing.assert_close(fmt.process_in(x), expected, rtol=0, atol=0)
        torch.testing.assert_close(
            fmt.process_out(expected), expected / inverse_std + fmt.latents_mean.bfloat16(), rtol=0, atol=0
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA numerical regression requires a GPU")
class CUDAActivationTests(unittest.TestCase):
    def test_bounded_conv_matches_exact_cpu_arithmetic_with_bias_stride_and_dilation(self):
        from comfy.ldm.wan.vae import CausalConv3d

        for batch, kernel, stride, dilation, use_bias in (
            (1, (3, 3, 3), (1, 1, 1), (1, 1, 1), True),
            (2, (3, 3, 3), (1, 2, 2), (1, 1, 1), False),
            (1, (3, 3, 3), (1, 1, 1), (2, 1, 1), True),
            (2, (1, 3, 3), (1, 1, 1), (1, 1, 1), True),
        ):
            with self.subTest(batch=batch, kernel=kernel, stride=stride, dilation=dilation):
                conv = CausalConv3d(2, 3, kernel, padding=0, stride=stride, dilation=dilation, bias=use_bias)
                with torch.no_grad():
                    conv.weight.copy_(
                        torch.arange(conv.weight.numel()).reshape_as(conv.weight).remainder(3) * 0.5 - 0.5
                    )
                    if use_bias:
                        conv.bias.copy_(torch.tensor([-0.125, 0.0, 0.125]))
                inputs = (
                    torch.arange(batch * 2 * 7 * 9 * 11).reshape(batch, 2, 7, 9, 11).remainder(5).float()
                    * 0.25
                    - 0.5
                )
                padded = torch.nn.functional.pad(inputs, (1, 1, 1, 1, 0, 0))
                expected = torch.nn.functional.conv3d(
                    padded,
                    conv.weight.detach(),
                    conv.bias.detach() if use_bias else None,
                    stride=stride,
                    dilation=dilation,
                ).bfloat16()
                with torch.inference_mode():
                    actual = vae_module._causal_conv_forward(
                        conv.cuda(), inputs.cuda().bfloat16(), spatial_padding=(1, 1), im2col_bounded=True
                    )
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    def test_wan22_activation_patches_and_explicit_conv_under_bf16_autocast(self):
        from comfy.ldm.wan.vae import CausalConv3d, RMS_norm
        from comfy.ldm.wan.vae2_2 import ResidualBlock, Up_ResidualBlock, WanVAE

        def previous_rms(module, x):
            normalized = torch.nn.functional.normalize(x, dim=1 if module.channel_first else -1)
            gamma = nodes.comfy.ops.cast_to_input(module.gamma, normalized)
            bias = nodes.comfy.ops.cast_to_input(module.bias, normalized) if module.bias is not None else 0
            return normalized * module.scale * gamma + bias

        generator = torch.Generator().manual_seed(67)
        base = WanVAE(dim=2, dec_dim=4, z_dim=4)
        for parameter in base.parameters():
            parameter.data.copy_(
                torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype) * 0.03
            )
        reference = copy.deepcopy(base)
        optimized = copy.deepcopy(base)
        for module in reference.modules():
            if isinstance(module, RMS_norm):
                module.forward = MethodType(previous_rms, module)
            elif isinstance(module, CausalConv3d):
                spatial_padding = module.padding[1:]
                module.forward = MethodType(
                    partial(vae_module._causal_conv_forward, spatial_padding=spatial_padding),
                    module,
                )
                module.padding = (0, 0, 0)
        for _, module in optimized.named_modules():
            if isinstance(module, RMS_norm):
                module.forward = MethodType(vae_module._rms_forward, module)
            elif isinstance(module, torch.nn.SiLU):
                module.forward = MethodType(vae_module._silu_forward, module)
            elif isinstance(module, ResidualBlock):
                module.forward = MethodType(vae_module._residual_forward_inplace, module)
            elif isinstance(module, Up_ResidualBlock):
                module.forward = MethodType(vae_module._up_residual_forward_inplace, module)
            elif isinstance(module, CausalConv3d):
                spatial_padding = module.padding[1:]
                module.forward = MethodType(
                    partial(vae_module._causal_conv_forward, spatial_padding=spatial_padding),
                    module,
                )
                module.padding = (0, 0, 0)
        optimized.decode = MethodType(vae_module._wan_decode_streaming, optimized)

        reference.cuda()
        optimized.cuda()
        latent = torch.randn(1, 4, 2, 2, 2, generator=generator).cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            expected = reference.decode(latent)
            actual = optimized.decode(latent)
            streamed = torch.empty(expected.shape, dtype=torch.float32, device="cpu")
            optimized.decode(latent, output_buffer=streamed)
        # In-place elementwise/shortcut math and streaming remain bit exact
        # with the previous Vimoo arithmetic and convolution path.
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(streamed, expected.float().cpu(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
