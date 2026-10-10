"""Lite TI2V schedule, observed-frame math and native lifecycle contracts."""

import importlib
import math
from dataclasses import asdict
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from test_nodes_runtime import nodes, spec
from test_patchify import tiny_hms

runtime = importlib.import_module(f"{spec.name}.vimoo_ti2v")
routes = importlib.import_module(f"{spec.name}.vimoo_ti2v_nodes")
text = importlib.import_module(f"{spec.name}.vimoo_text")


class AnchorTI2VTests(unittest.TestCase):
    def test_profile_linear_bias_boundaries_keep_qkv_fused(self):
        precision = importlib.import_module(f"{spec.name}.vimoo_precision")
        model = tiny_hms()
        linears = {
            "txt_in.input_embedder": model.txt_in.input_embedder,
            "double_blocks.0.img_attn.proj": model.double_blocks[0].img_attn.proj,
        }
        qkv = model.txt_in.individual_token_refiner.blocks[0].self_attn.qkv
        original_qkv_forward = qkv.forward.__func__
        precision.configure_anchor_linears(model)
        self.assertIs(qkv.forward.__func__, original_qkv_forward)
        for name, module in linears.items():
            with torch.no_grad():
                module.weight.zero_()
                module.weight[:, 0] = 1.0078125
                module.bias.fill_(0.00390625)
            value = torch.zeros(1, module.in_features, dtype=torch.bfloat16)
            value[:, 0] = 1.0078125
            with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
                actual = module(value)
            if name.startswith("txt_in"):
                self.assertEqual(actual.dtype, torch.bfloat16)
                self.assertTrue(torch.all(actual == 1.015625))
            else:
                self.assertEqual(actual.dtype, torch.float32)
                self.assertTrue(torch.all(actual == 1.01953125))

    def test_qk_gamma_is_applied_before_bf16_rounding(self):
        precision = importlib.import_module(f"{spec.name}.vimoo_precision")
        value = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16).reshape(1, 1, 1, 4)
        gamma = torch.tensor([1.3, 1.4, 1.5, 1.6], dtype=torch.bfloat16)
        expected = (value.float() / math.sqrt(7.5) * gamma.float()).bfloat16()
        actual, _ = precision.anchor_qk_norm(value, value, gamma, gamma, 0.0)
        legacy, _ = precision.precise_qk_norm(value, value, gamma, gamma, 0.0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(actual, legacy))

    def test_new_time_embedding_preserves_double_precision_phase(self):
        precision = importlib.import_module(f"{spec.name}.vimoo_precision")
        timesteps = torch.cat((runtime.ti2v_sigmas(50)[:-1] * 1000, torch.tensor([1000.0, 1.0, 1.0])))
        angles = [
            [float(timestep) * math.exp(-math.log(10000) * index / 128) for index in range(128)]
            for timestep in timesteps
        ]
        expected = torch.tensor(
            [[math.cos(value) for value in row] + [math.sin(value) for value in row] for row in angles]
        )
        actual = precision.timestep_embedding(
            timesteps,
            256,
            time_factor=1.0,
            frequencies=torch.zeros(128),
            frequency_float64=True,
        )
        self.assertEqual(actual.dtype, torch.float64)
        torch.testing.assert_close(actual.float(), expected, rtol=0, atol=0)
        legacy = precision.timestep_embedding(timesteps, 256, time_factor=1.0)
        self.assertEqual(legacy.dtype, torch.float32)
        self.assertFalse(torch.equal(legacy, expected))

    def test_single_frame_sampling_fails_before_loading_model(self):
        model = Mock()
        model.model_options = {"vimoo_sampling_profile": runtime.PROFILE}
        model.model.get_dtype.return_value = torch.bfloat16
        with self.assertRaisesRegex(ValueError, "video length greater than one"):
            runtime.sample_lite_ti2v(
                model,
                [],
                [],
                {"samples": torch.zeros(1, 48, 1, 9, 16)},
                runtime.ti2v_sigmas(50),
                4.0,
                0.85,
                1234,
            )
        model.clone.assert_not_called()

    def test_lognorm_50_grid_and_metadata_routing(self):
        q = torch.linspace(1, 0, 51, dtype=torch.float64)
        expected = torch.sigmoid(math.log(8) + 2.5 * torch.special.ndtri(q)).float()
        model = SimpleNamespace(model_options={"vimoo_sampling_profile": runtime.PROFILE})
        for task, cfg in (("t2v", 4), ("i2v", 5)):
            result = routes.VimooTI2VSchedule().resolve(model, task, 50, 8, 4, 5, 0.85, 0.85, True)
            torch.testing.assert_close(result[0], expected, rtol=0, atol=0)
            self.assertEqual(result[1:], (cfg, 0.85, False, "vimoo_lite_ti2v_euler", 1400))
        model.model_options["vimoo_distilled"] = {"task": "ti2v"}
        with self.assertRaisesRegex(ValueError, "No matching Distilled"):
            routes.VimooTI2VSchedule().resolve(model, "i2v", 50, 8, 4, 5, 0.85, 0.85, False)

    def test_full50_bf16_trajectory_fixed_anchor_and_clean_final(self):
        shape = (1, 2, 3, 2, 2)
        initial = runtime.i2v_initial_state(shape, 1234, "cpu")
        reference = torch.linspace(-1, 1, 8).reshape(1, 2, 1, 2, 2).bfloat16()
        sigmas = runtime.ti2v_sigmas(50)
        seen = []
        flow = torch.full(shape, 0.137, dtype=torch.bfloat16)

        def model(value, sigma):
            seen.append(value.clone())
            return flow

        for tau in (0.001, 0.0, 1.0):
            seen.clear()
            result = runtime.sample_ti2v_euler(model, initial, sigmas, reference=reference, tau=tau)
            expected = initial.clone()
            anchor = runtime.anchor_frame(reference, initial[:, :, :1], tau)
            expected[:, :, :1] = anchor
            for index in range(50):
                torch.testing.assert_close(seen[index], expected.float(), rtol=0, atol=0)
                expected = (expected.float() + flow.float() * (sigmas[index + 1] - sigmas[index])).bfloat16()
                expected[:, :, :1] = anchor
            expected[:, :, :1] = reference
            self.assertEqual(result.dtype, torch.bfloat16)
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            if tau == 0:
                torch.testing.assert_close(anchor, reference, rtol=0, atol=0)
            elif tau == 1:
                torch.testing.assert_close(anchor, initial[:, :, :1], rtol=0, atol=0)

    def test_sampler_guidance_domains_match_runtime_validation(self):
        model = Mock()
        model.model_options = {"vimoo_sampling_profile": runtime.PROFILE}
        model.model.get_dtype.return_value = torch.bfloat16
        model.clone.side_effect = RuntimeError("validated before weight loading")
        latent = {"samples": torch.zeros(1, 48, 2, 1, 1)}
        controls = dict(cfg=0.5, guidance_rescale=1.2, reference_guidance_scale=0.0)
        for tau in (0.0, 1.0):
            values = {**controls, "i2v_anchor_tau": tau}
            self.assertIs(
                nodes.VimooSampler.VALIDATE_INPUTS("vimoo_lite_ti2v_euler", denoise=1, **values), True
            )
            with self.assertRaisesRegex(RuntimeError, "validated before weight loading"):
                runtime.sample_lite_ti2v(model, [], [], latent, runtime.ti2v_sigmas(2), seed=1234, **values)
        for name in (*controls, "i2v_anchor_tau"):
            invalid = (-1.0, float("nan"), float("inf"))
            if name == "i2v_anchor_tau":
                invalid += (1.1,)
            for value in invalid:
                with self.subTest(control=name, value=value):
                    values = {**controls, "i2v_anchor_tau": 0.001, name: value}
                    model.clone.reset_mock()
                    self.assertIsNot(
                        nodes.VimooSampler.VALIDATE_INPUTS("vimoo_lite_ti2v_euler", **values), True
                    )
                    with self.assertRaisesRegex(ValueError, "must be finite"):
                        runtime.sample_lite_ti2v(
                            model, [], [], latent, runtime.ti2v_sigmas(2), seed=1234, **values
                        )
                    model.clone.assert_not_called()
                    if name in ("cfg", "reference_guidance_scale"):
                        with self.assertRaisesRegex(ValueError, "must be finite"):
                            routes.VimooTI2VTextEncodePair().encode(
                                Mock(),
                                "i2v",
                                "positive",
                                "negative",
                                True,
                                1400,
                                values["cfg"],
                                values["reference_guidance_scale"],
                            )

    def test_three_branch_guidance_rescales_generated_region(self):
        generator = torch.Generator().manual_seed(7)
        predictions = [torch.randn(1, 2, 3, 2, 2, generator=generator) for _ in range(3)]
        uncond, ref, positive = predictions
        expected = uncond + 3 * (ref - uncond) + 5 * (positive - ref)
        part = expected[:, :, 1:]
        scaled = part * (
            positive[:, :, 1:].std((1, 2, 3, 4), keepdim=True) / part.std((1, 2, 3, 4), keepdim=True)
        )
        expected[:, :, 1:] = 0.85 * scaled + 0.15 * part
        expected[:, :, :1] = 0
        actual = runtime.ti2v_guidance(predictions, 5, 3, 0.85, True)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        changed = [value.clone() for value in predictions]
        for value in changed:
            value[:, :, :1] *= 1000
        torch.testing.assert_close(runtime.ti2v_guidance(changed, 5, 3, 0.85, True), actual, rtol=0, atol=0)

    def test_profile_precision_rounds_residual_once_and_keeps_head_fp32(self):
        precision = importlib.import_module(f"{spec.name}.vimoo_precision")
        generator = torch.Generator().manual_seed(43)
        residual, value = [torch.randn(2, 3, 128, generator=generator).bfloat16() for _ in range(2)]
        gate = torch.randn(2, 128, generator=generator).bfloat16()
        modulation = SimpleNamespace(gate=gate)
        expected = (residual.float() + value.float() * gate.float().unsqueeze(1)).bfloat16()
        actual = precision.source_gate_add(residual, value, modulation)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(actual, residual + value * gate.unsqueeze(1)))
        head = tiny_hms().final_layer
        vec = torch.randn(2, 128, generator=generator)
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            legacy = head(residual, vec)
            source = head(residual, vec, preserve_fp32=True)
        with torch.inference_mode():
            activated = torch.nn.functional.silu(vec).bfloat16()
            modulation = torch.stack(head.adaLN_modulation[1](activated).chunk(2, dim=-1))
            expected_head = head(residual.float(), vec, precomputed_modulation=modulation)
        self.assertEqual(legacy.dtype, torch.bfloat16)
        self.assertEqual(source.dtype, torch.float32)
        torch.testing.assert_close(source, expected_head, rtol=0, atol=0)

    def test_observed_time_reaches_only_image_blocks_and_not_final_head(self):
        model = tiny_hms()
        captures = []
        hook = model.final_layer.register_forward_pre_hook(
            lambda module, args: captures.append(args[1].clone())
        )
        inputs = (
            torch.randn(1, 2, 2, 3, 4).bfloat16(),
            torch.tensor([555.125]),
            torch.randn(1, 3, 6).bfloat16(),
        )
        options = {"vimoo_block_prefetch": False}
        try:
            with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
                ordinary = model(*inputs, fps=24.0, transformer_options=options)
                observed = model(
                    *inputs,
                    fps=24.0,
                    transformer_options=options,
                    observed_first_frame_timesteps=torch.tensor([1.0]),
                )
                same_time = model(
                    *inputs, fps=24.0, transformer_options=options, observed_first_frame_timesteps=inputs[1]
                )
                source = model(
                    *inputs,
                    fps=24.0,
                    transformer_options={**options, "vimoo_lite_ti2v_anchor": True},
                    observed_first_frame_timesteps=torch.tensor([1.0]),
                )
            torch.testing.assert_close(captures[0], captures[1], rtol=0, atol=0)
            torch.testing.assert_close(captures[0], captures[2], rtol=0, atol=0)
            torch.testing.assert_close(ordinary, same_time, rtol=0, atol=0)
            self.assertFalse(torch.equal(ordinary, observed))
            self.assertEqual(observed.shape, inputs[0].shape)
            self.assertEqual(captures[0].dtype, torch.bfloat16)
            self.assertEqual(captures[3].dtype, torch.float32)
            self.assertEqual(source.dtype, torch.float32)
            self.assertTrue(source.is_contiguous())
            torch.testing.assert_close(source, source.bfloat16().float(), rtol=0, atol=0)
        finally:
            hook.remove()

    def test_native_guider_branch_order_and_no_reference_frame(self):
        model = SimpleNamespace(model_options={})
        guider = runtime.TI2VFlowGuider(model, 5, 3, 0, True)
        guider.inner_model = object()
        guider.conds = {name: [name] for name in ("unreferenced", "negative", "positive")}
        outputs = [torch.full((1, 2, 2, 2, 2), value) for value in (1.0, 2.0, 4.0)]
        with patch.object(runtime.comfy.samplers, "calc_cond_batch", return_value=outputs) as call:
            result = guider.predict_noise(torch.zeros_like(outputs[0]), torch.tensor([1.0]))
        self.assertEqual(call.call_args.args[1], [["unreferenced"], ["negative"], ["positive"]])
        self.assertTrue(torch.all(result[:, :, :1] == 0))
        self.assertTrue(torch.all(result[:, :, 1:] == 14))

    def test_fp16_qwen_rotary_rounding_does_not_inherit_bf16(self):
        module = SimpleNamespace(config=SimpleNamespace(head_dim=128, rope_theta=1000000.0))
        positions = torch.arange(1400).reshape(1, -1)
        fp16 = text._qwen_frequencies(module, positions, "cpu", compute_dtype=torch.float16)
        legacy = text._qwen_frequencies(module, positions, "cpu")
        inv = (1 / (1000000.0 ** (torch.arange(0, 128, 2).float() / 128))).half().float()
        phase = (inv[None, :, None] @ positions[:, None].float()).transpose(1, 2)
        expected = torch.cat((phase, phase), dim=-1).cos().half().unsqueeze(1)
        torch.testing.assert_close(fp16[0], expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(fp16[0].float(), legacy[0].float()))

    def test_lazy_negative_is_required_for_reference_guidance_at_cfg_one(self):
        model = SimpleNamespace(model_options={"vimoo_sampling_profile": runtime.PROFILE})
        positive = [[torch.zeros(1), {"concat_latent_image": torch.ones(1)}]]
        sampler = nodes.VimooSampler()
        self.assertEqual(
            sampler.check_lazy_status(1.0, model=model, positive=positive, negative=None), ["negative"]
        )
        self.assertEqual(
            sampler.check_lazy_status(
                1.0, model=model, positive=positive, negative=None, reference_guidance_scale=1.0
            ),
            [],
        )
        self.assertEqual(
            sampler.check_lazy_status(1.0, model=model, positive=[[torch.zeros(1), {}]], negative=None), []
        )
        model.model_options["vimoo_distilled"] = {"profile": "lite_ti2v_distilled"}
        self.assertEqual(sampler.check_lazy_status(1.0, model=model, positive=positive, negative=None), [])

    def test_paired_qwen_retains_distinct_unreferenced_and_reference_rows(self):
        clip = Mock()
        clip.cond_stage_model.vimoo_precision_profile = runtime.PROFILE
        clip.tokenize.side_effect = lambda prompt, **kwargs: {"qwen25_7b": [[(len(prompt), 1.0)]]}

        def encode(tokens):
            batch = len(tokens["vimoo_cfg_prompts"])
            return [
                [
                    torch.arange(batch).reshape(batch, 1, 1),
                    {
                        "attention_mask": torch.ones(batch, 1),
                        "conditioning_byt5small": torch.zeros(batch, 256, 1472),
                        "vimoo_byt5_mask": torch.zeros(batch, 256),
                    },
                ]
            ]

        clip.encode_from_tokens_scheduled.side_effect = encode
        cases = (
            ("t2v", 4.0, 3.0, ("negative", "positive")),
            ("t2v", 1.0, 3.0, ("positive",)),
            ("t2v", 0.5, 3.0, ("positive",)),
            ("i2v", 5.0, 3.0, ("unreferenced", "negative", "positive")),
            ("i2v", 1.0, 3.0, ("unreferenced", "negative", "positive")),
            ("i2v", 3.0, 3.0, ("unreferenced", "positive")),
            ("i2v", 5.0, 1.0, ("negative", "positive")),
            ("i2v", 1.0, 1.0, ("positive",)),
            ("i2v", 0.5, 1.0, ("positive",)),
            ("i2v", 0.5, 0.0, ("unreferenced", "negative", "positive")),
        )
        for task, cfg, reference_scale, expected_names in cases:
            with self.subTest(task=task, cfg=cfg, reference_scale=reference_scale):
                clip.encode_from_tokens_scheduled.reset_mock()
                positive, negative, scale = routes.VimooTI2VTextEncodePair().encode(
                    clip, task, "pos", "negative", True, 1400, cfg, reference_scale
                )
                clip.encode_from_tokens_scheduled.assert_called_once()
                tokens = clip.encode_from_tokens_scheduled.call_args.args[0]
                self.assertEqual(
                    tokens["qwen25_7b"],
                    [[(3 if name == "positive" else 8, 1.0)] for name in expected_names],
                )
                self.assertEqual(positive[0][0].item(), len(expected_names) - 1)
                if len(expected_names) > 1:
                    negative_index = expected_names.index("negative") if "negative" in expected_names else 0
                    self.assertEqual(negative[0][0].item(), negative_index)
                else:
                    self.assertIsNone(negative)
                self.assertEqual(scale, reference_scale if task == "i2v" else 1.0)

                guider = runtime.TI2VFlowGuider(
                    SimpleNamespace(model_options={}), cfg, scale, 0, task == "i2v"
                )
                guider.inner_model = object()
                guider.conds = {"positive": positive, "negative": negative}
                if "unreferenced" in expected_names:
                    unreferenced = negative[0][1]["vimoo_unreferenced_conditioning"]
                    self.assertEqual(unreferenced[0].item(), 0)
                    guider.conds["unreferenced"] = [unreferenced]
                elif negative is not None:
                    self.assertNotIn("vimoo_unreferenced_conditioning", negative[0][1])
                outputs = [torch.full((1, 1, 2, 1, 1), float(index)) for index in range(len(expected_names))]
                with patch.object(runtime.comfy.samplers, "calc_cond_batch", return_value=outputs) as call:
                    guider.predict_noise(torch.zeros_like(outputs[0]), torch.tensor([1.0]))
                consumed_rows = [condition[0][0].item() for condition in call.call_args.args[1]]
                self.assertEqual(consumed_rows, list(range(len(expected_names))))

    def test_encoder_executes_one_qwen_batch_and_each_glyph_prompt_separately(self):
        encoder_module = importlib.import_module(f"{spec.name}.vimoo_encoder")
        encoder = SimpleNamespace(qwen25_7b=Mock(), byt5_small=Mock())
        encoder.qwen25_7b.encode.return_value = (
            torch.randn(3, 2, 6),
            None,
            {"attention_mask": torch.ones(3, 2, dtype=torch.int64)},
        )
        encoder.byt5_small.encode_token_weights.return_value = (torch.ones(1, 256, 1472),)
        negative = {"qwen25_7b": [[(5, 1.0), (6, 1.0)]]}
        positive = {"qwen25_7b": [[(7, 1.0), (8, 1.0)]], "byt5": [[(9, 1.0)] + [(0, 1.0)] * 255]}
        cond, _, values = encoder_module.VimooTextEncoder.encode_cfg_batch(
            encoder, [negative, negative, positive]
        )
        encoder.qwen25_7b.encode.assert_called_once_with([[5, 6], [5, 6], [7, 8]])
        encoder.byt5_small.encode_token_weights.assert_called_once_with(positive["byt5"])
        self.assertEqual(cond.dtype, torch.bfloat16)
        self.assertEqual(tuple(cond.shape), (3, 2, 6))
        self.assertEqual(values["conditioning_byt5small"].dtype, torch.bfloat16)
        self.assertEqual(values["vimoo_byt5_mask"].sum().item(), 1)

    def test_lite_compact_cfg_one_uses_single_positive_and_boolean_masks(self):
        encoder_module = importlib.import_module(f"{spec.name}.vimoo_encoder")
        encoder = SimpleNamespace(qwen25_7b=Mock(), byt5_small=Mock(), vimoo_precision_profile="lite_compact")
        encoder.qwen25_7b.encode.return_value = (
            torch.ones(1, 2, 6, dtype=torch.float16),
            None,
            {"attention_mask": torch.tensor([[1, 0]], dtype=torch.int64)},
        )
        encoder.byt5_small.encode_token_weights.return_value = (torch.ones(1, 256, 1472),)
        positive = {"qwen25_7b": [[(7, 1.0), (8, 1.0)]], "byt5": [[(9, 1.0)] + [(0, 1.0)] * 255]}
        clip = Mock()
        clip.cond_stage_model = encoder
        clip.tokenize.side_effect = lambda prompt, **kwargs: positive if prompt == "positive" else {}
        clip.encode_from_tokens_scheduled.side_effect = lambda tokens: [
            [result[0], result[2]]
            for result in [
                encoder_module.VimooTextEncoder.encode_cfg_batch(encoder, tokens["vimoo_cfg_prompts"])
            ]
        ]
        for task in ("t2v", "i2v"):
            encoder.qwen25_7b.encode.reset_mock()
            clip.tokenize.reset_mock()
            positive_output, negative_output, _ = routes.VimooTI2VTextEncodePair().encode(
                clip, task, "positive", "negative", True, 1400, 1.0, 1.0
            )
            self.assertIsNone(negative_output)
            clip.tokenize.assert_called_once_with(
                "positive", glyph_conditioning=True, max_sequence_length=1400
            )
            output = positive_output[0]
            encoder.qwen25_7b.encode.assert_called_once_with([[7, 8]])
            self.assertEqual(output[0].dtype, torch.bfloat16)
            self.assertEqual(output[1]["conditioning_byt5small"].dtype, torch.bfloat16)
            self.assertEqual(output[1]["attention_mask"].dtype, torch.bool)
            self.assertEqual(output[1]["vimoo_byt5_mask"].dtype, torch.bool)
            self.assertEqual(output[1]["vimoo_byt5_mask"].sum(), 1)
            plain = nodes.VimooTextEncode().encode(clip, "positive", True, 1400)[0][0]
            torch.testing.assert_close(plain[0], output[0], rtol=0, atol=0)
            for key, value in output[1].items():
                torch.testing.assert_close(plain[1][key], value, rtol=0, atol=0)

    def test_patchify_no_glyph_executes_exact_empty_byt5_then_zeroes(self):
        encoder_module = importlib.import_module(f"{spec.name}.vimoo_encoder")
        encoder = SimpleNamespace(qwen25_7b=Mock(), byt5_small=Mock(), vimoo_precision_profile="lite_compact")
        encoder.qwen25_7b.encode.return_value = (
            torch.ones(1, 2, 6, dtype=torch.float16),
            None,
            {"attention_mask": torch.ones(1, 2, dtype=torch.int64)},
        )
        encoder.byt5_small.encode_token_weights.return_value = (torch.full((1, 256, 1472), 3.125),)
        expected = encoder_module.VimooTokenizer().byt5.tokenizer(
            "", padding="max_length", max_length=256, add_special_tokens=True
        )["input_ids"]
        cond, _, values = encoder_module.VimooTextEncoder.encode_cfg_batch(
            encoder, [{"qwen25_7b": [[(7, 1.0), (8, 1.0)]]}]
        )
        actual = encoder.byt5_small.encode_token_weights.call_args.args[0][0]
        self.assertEqual([token[0] for token in actual], expected)
        self.assertEqual(cond.dtype, torch.bfloat16)
        self.assertEqual(values["conditioning_byt5small"].count_nonzero(), 0)
        self.assertEqual(values["vimoo_byt5_mask"].count_nonzero(), 0)
        self.assertEqual(values["vimoo_byt5_mask"].dtype, torch.bool)

    def test_patchify_sampler_preserves_boolean_vision_mask_at_actual_guider_boundary(self):
        latent_format = importlib.import_module(f"{spec.name}.vimoo_vae").VimooLatentFormat()
        sigmas = nodes.VimooScheduler().get_sigmas("lite_ti2v_distilled", 12, 12.0)[0].float()
        recipe = {
            "num_inference_steps": 12,
            "custom_timesteps": [1000, 998.3, 996.5, 994.7, 991, 983.9, 973, 952.4, 923.1, 878, 800, 631.6],
            "timestep_space": "post_shift",
            "flow_shift": 12,
            "i2v_first_frame_replace": False,
            "initial_noise_mode": "cpu_channel_first",
            "initial_noise_seed": 1234,
        }
        initial = torch.zeros(1, 48, 2, 1, 1, dtype=torch.bfloat16)
        conditions = [
            [
                torch.zeros(1, 2, 6),
                {
                    "concat_latent_image": initial[:, :, :1],
                    "clip_vision_output": SimpleNamespace(
                        last_hidden_state=torch.zeros(1, 3, 6, dtype=torch.bfloat16)
                    ),
                    "vimoo_vision_mask": torch.ones(1, 3, dtype=torch.bool),
                },
            ]
        ]
        for distilled in (False, True):
            model = SimpleNamespace(
                model=SimpleNamespace(
                    get_dtype=lambda: torch.bfloat16,
                    model_sampling=SimpleNamespace(),
                    latent_format=latent_format,
                ),
                model_options={
                    "vimoo_sampling_profile": runtime.PROFILE,
                    "vimoo_initial_noise_file": "fixed-noise",
                },
                add_object_patch=lambda *args: None,
            )
            model.clone = lambda: model
            if distilled:
                model.model_options["vimoo_distilled"] = {
                    "profile": "lite_ti2v_distilled",
                    "recipes": {"t2v": recipe, "i2v": recipe},
                }
            with (
                patch.object(runtime.torch, "load", return_value=initial),
                patch.object(runtime, "TI2VFlowGuider") as guider,
            ):
                guider.return_value.sample.return_value = initial
                runtime.sample_lite_ti2v(
                    model,
                    conditions,
                    conditions,
                    {"samples": initial},
                    sigmas,
                    1,
                    0,
                    1234,
                    reference_guidance_scale=1,
                    i2v_anchor_tau=0.001,
                )
            consumed = guider.return_value.inner_set_conds.call_args.args[0]
            expected_dtype = torch.bool if distilled else torch.int64
            self.assertEqual(consumed["positive"][0][1]["vimoo_vision_mask"].dtype, expected_dtype)
            self.assertEqual(consumed["unreferenced"][0][1]["vimoo_vision_mask"].dtype, expected_dtype)
            self.assertTrue(consumed["positive"][0][1]["vimoo_vision_mask"].all())
            self.assertEqual(consumed["unreferenced"][0][1]["vimoo_vision_mask"].count_nonzero(), 0)
            self.assertEqual(conditions[0][1]["vimoo_vision_mask"].dtype, torch.bool)

    def test_distilled_noise_is_fixed_and_ordinary_noise_follows_sampler_seed(self):
        latent = torch.zeros(1, 48, 2, 3, 4, dtype=torch.bfloat16)
        recipe = {
            "num_inference_steps": 2,
            "custom_timesteps": [1000, 500],
            "timestep_space": "post_shift",
            "flow_shift": 12,
            "i2v_first_frame_replace": False,
            "initial_noise_mode": "cpu_channel_first",
            "initial_noise_seed": 1234,
        }
        model = SimpleNamespace(
            model=SimpleNamespace(
                get_dtype=lambda: torch.bfloat16,
                model_sampling=SimpleNamespace(),
                latent_format=SimpleNamespace(process_in=lambda value: value),
            ),
            model_options={
                "vimoo_sampling_profile": runtime.PROFILE,
            },
            add_object_patch=lambda *args: None,
        )
        model.clone = lambda: model
        noises = []

        def solve(active, initial, sigmas, **kwargs):
            noises.append(initial.clone())
            return initial

        def execute(noise, state, sampler, sigmas, **kwargs):
            return sampler.sampler_function(None, state, sigmas)

        with (
            patch.object(runtime, "TI2VFlowGuider") as guider,
            patch.object(runtime, "sample_ti2v_euler", side_effect=solve),
            patch.object(runtime.torch, "load", side_effect=AssertionError("must generate noise locally")),
        ):
            guider.return_value.sample.side_effect = execute
            for distilled in (False, True):
                if distilled:
                    model.model_options["vimoo_distilled"] = {
                        "profile": "lite_ti2v_distilled",
                        "recipes": {"t2v": recipe, "i2v": recipe},
                    }
                for task in ("t2v", "i2v"):
                    noises.clear()
                    values = {} if task == "t2v" else {"concat_latent_image": latent[:, :, :1]}
                    for seed in (0, 1234, 5678):
                        runtime.sample_lite_ti2v(
                            model,
                            [[torch.zeros(1, 2, 6), values]],
                            [],
                            {"samples": latent},
                            torch.tensor([1.0, 0.5, 0.0]),
                            1,
                            0,
                            seed,
                        )
                        expected = torch.randn(
                            latent.shape,
                            generator=torch.Generator("cpu").manual_seed(1234 if distilled else seed),
                            dtype=torch.bfloat16,
                        )
                        torch.testing.assert_close(noises[-1], expected, rtol=0, atol=0)
                    with self.subTest(distilled=distilled, task=task):
                        self.assertEqual(torch.equal(noises[0], noises[1]), distilled)
                        self.assertEqual(torch.equal(noises[0], noises[2]), distilled)
        recipe["initial_noise_mode"] = "fixed"
        model.clone = Mock(side_effect=AssertionError("must reject before model loading"))
        with self.assertRaisesRegex(ValueError, "update the adapter sidecar"):
            runtime.sample_lite_ti2v(
                model, [], [], {"samples": latent}, torch.tensor([1.0, 0.5, 0.0]), 1, 0, 5678
            )
        model.clone.assert_not_called()
        recipe["initial_noise_mode"] = "cpu_channel_first"
        for noise_seed in (None, 5678):
            with self.subTest(noise_seed=noise_seed):
                recipe["initial_noise_seed"] = noise_seed
                with self.assertRaisesRegex(ValueError, "update the adapter sidecar"):
                    runtime.sample_lite_ti2v(
                        model, [], [], {"samples": latent}, torch.tensor([1.0, 0.5, 0.0]), 1, 0, 5678
                    )
        model.clone.assert_not_called()

    def test_lite_distilled_checks_only_the_running_task_grid(self):
        latent_format = importlib.import_module(f"{spec.name}.vimoo_vae").VimooLatentFormat()
        sigmas = nodes.VimooScheduler().get_sigmas("lite_ti2v_distilled", 12, 12.0)[0].float()
        i2v = {
            "num_inference_steps": 12,
            "custom_timesteps": [1000, 998.3, 996.5, 994.7, 991, 983.9, 973, 952.4, 923.1, 878, 800, 631.6],
            "timestep_space": "post_shift",
            "flow_shift": 12,
            "i2v_first_frame_replace": False,
            "initial_noise_mode": "cpu_channel_first",
            "initial_noise_seed": 1234,
        }
        t2v = {**i2v, "num_inference_steps": 2, "custom_timesteps": [1000, 500]}
        initial = torch.zeros(1, 48, 2, 1, 1, dtype=torch.bfloat16)
        vision = SimpleNamespace(last_hidden_state=torch.zeros(1, 3, 6, dtype=torch.bfloat16))
        image = {
            "concat_latent_image": initial[:, :, :1],
            "clip_vision_output": vision,
            "vimoo_vision_mask": torch.ones(1, 3, dtype=torch.bool),
        }
        model = SimpleNamespace(
            model=SimpleNamespace(
                get_dtype=lambda: torch.bfloat16,
                model_sampling=SimpleNamespace(),
                latent_format=latent_format,
            ),
            model_options={
                "vimoo_sampling_profile": runtime.PROFILE,
                "vimoo_initial_noise_file": "fixed-noise",
                "vimoo_distilled": {"profile": "lite_ti2v_distilled", "recipes": {"t2v": t2v, "i2v": i2v}},
            },
            add_object_patch=lambda *args: None,
        )
        model.clone = lambda: model

        def sample(extra):
            conditions = [[torch.zeros(1, 2, 6), dict(extra)]]
            return runtime.sample_lite_ti2v(
                model,
                conditions,
                conditions,
                {"samples": initial},
                sigmas,
                1,
                0,
                1234,
                reference_guidance_scale=1,
                i2v_anchor_tau=0.001,
            )

        with (
            patch.object(runtime.torch, "load", return_value=initial),
            patch.object(runtime, "TI2VFlowGuider") as guider,
        ):
            guider.return_value.sample.return_value = initial
            sample(image)
            with self.assertRaisesRegex(ValueError, "full manual post-shift grid"):
                sample({})

    def test_real_native_lifecycle_executes_i2v_branches_and_restores_model(self):
        support = importlib.import_module(f"{spec.name}.vimoo_support")
        params = asdict(tiny_hms().params)
        params.update(in_channels=97, out_channels=48, block_modulation_rank=4)
        config = support.Vimoo(params)
        config.set_inference_dtype(torch.bfloat16, None, device=torch.device("cpu"))
        base = config.get_model({}, device=torch.device("cpu"))
        generator = torch.Generator().manual_seed(81)
        with torch.no_grad():
            for parameter in base.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)
        model = nodes.comfy.model_patcher.ModelPatcher(base, torch.device("cpu"), torch.device("cpu"))
        model.model_options["vimoo_sampling_profile"] = runtime.PROFILE
        model.model_options["transformer_options"]["vimoo_block_prefetch"] = False
        ref = torch.randn(1, 48, 1, 3, 4, generator=generator)
        mask = torch.ones(1, 1, 2, 3, 4)
        mask[:, :, :1] = 0
        values = {
            "attention_mask": torch.ones(1, 3, dtype=torch.int64),
            "concat_latent_image": ref,
            "concat_mask": mask,
            "clip_vision_output": SimpleNamespace(
                last_hidden_state=torch.randn(1, 3, 6, generator=generator)
            ),
        }
        positive = [[torch.randn(1, 3, 6, generator=generator), values]]
        negative = [[torch.randn(1, 3, 6, generator=generator), values]]
        captured = []
        block_dtypes = []
        hook = base.diffusion_model.img_in.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].clone())
        )
        block_hook = base.diffusion_model.double_blocks[0].register_forward_pre_hook(
            lambda module, args, kwargs: block_dtypes.append((kwargs["img"].dtype, kwargs["txt"].dtype)),
            with_kwargs=True,
        )
        original_process_timestep = base.process_timestep
        try:
            with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
                (result,) = runtime.sample_lite_ti2v(
                    model,
                    positive,
                    negative,
                    {"samples": torch.zeros(1, 48, 2, 3, 4)},
                    runtime.ti2v_sigmas(2),
                    5.0,
                    0.85,
                    1234,
                )
            self.assertTrue(torch.isfinite(result["samples"]).all())
            self.assertEqual(tuple(result["samples"].shape), (1, 48, 2, 3, 4))
            self.assertEqual(sum(value.shape[0] for value in captured), 6)
            self.assertTrue(all(value == (torch.float32, torch.float32) for value in block_dtypes))
            # Comfy keeps a resident clone patched until it is unloaded.
            nodes.comfy.model_management.unload_all_models()
            self.assertEqual(base.process_timestep, original_process_timestep)
            self.assertNotIn("vimoo_lite_ti2v_anchor", model.model_options["transformer_options"])
            expected_ref = base.latent_format.process_out(base.latent_format.process_in(ref)).float()
            torch.testing.assert_close(result["samples"][:, :, :1], expected_ref, rtol=0, atol=0)
        finally:
            hook.remove()
            block_hook.remove()
