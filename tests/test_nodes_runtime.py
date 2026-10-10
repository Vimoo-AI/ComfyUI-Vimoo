"""CPU integration checks; run with ComfyUI on PYTHONPATH in its Python environment."""

import importlib.util
import os
import tempfile
import sys
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import patch

import numpy as np
import numpy
import folder_paths
import torch
from comfy.cli_args import args

args.cpu = True

PLUGIN = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "vimoo_test_plugin", PLUGIN / "__init__.py", submodule_search_locations=[str(PLUGIN)]
)
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
PATH_BEFORE_IMPORT = list(sys.path)
spec.loader.exec_module(plugin)
PATH_AFTER_IMPORT = list(sys.path)
nodes = sys.modules[f"{spec.name}.vimoo_nodes"]
samplers = sys.modules[f"{spec.name}.vimoo_sampler"]


class RuntimeTests(unittest.TestCase):
    def test_retired_text_profile_is_rejected_before_checkpoint_loading(self):
        with mock.patch.object(nodes.comfy.utils, "load_torch_file") as load:
            with self.assertRaisesRegex(ValueError, "text precision profile"):
                nodes.VimooTextEncoderLoader().load("qwen", "byt5", precision_profile="legacy")
        load.assert_not_called()

    def make_model(self):
        module = torch.nn.Linear(1, 1)
        module.get_dtype = lambda: module.weight.dtype
        return nodes.comfy.model_patcher.ModelPatcher(module, torch.device("cpu"), torch.device("cpu"))

    def make_flow_model(self):
        class TinyDiffusion(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.125, dtype=torch.bfloat16))
                self.dtype = torch.bfloat16
                self.forward_calls = 0
                self.precise_math = False
                self.precise_inputs = False

            def forward(self, x, timestep, **kwargs):
                self.forward_calls += 1
                return x * self.weight

        config = nodes.comfy.supported_models_base.BASE({"dtype": torch.bfloat16})
        base = nodes.comfy.model_base.BaseModel(
            config, nodes.comfy.model_base.ModelType.FLOW, unet_model=TinyDiffusion
        )
        return nodes.comfy.model_patcher.ModelPatcher(base, torch.device("cpu"), torch.device("cpu"))

    def sample(self, model, latent, **overrides):
        options = dict(
            model=model,
            positive=[],
            negative=[],
            latent_image=latent,
            sigmas=nodes.VimooScheduler().get_sigmas("vimoo", 3, 6.0)[0],
            cfg=1.0,
            guidance_rescale=1.0,
            seed=42,
            sampler_name="vimoo_uni_pc",
            denoise=1.0,
        )
        options.update(overrides)
        return nodes.VimooSampler().sample(**options)[0]

    def test_package_does_not_register_global_sampler_or_bare_modules(self):
        self.assertNotIn("vimoo_uni_pc", nodes.comfy.samplers.SAMPLER_NAMES)
        self.assertNotIn("vimoo_nodes", sys.modules)
        self.assertEqual(PATH_AFTER_IMPORT.count(str(PLUGIN)), PATH_BEFORE_IMPORT.count(str(PLUGIN)))
        choices = nodes.VimooSampler.INPUT_TYPES()["required"]["sampler_name"][1]["options"]
        self.assertIn("vimoo_uni_pc", choices)

    def test_zero_denoise_keeps_metadata_and_skips_sampling(self):
        latent = {"samples": torch.zeros(1, 48, 2, 2, 2), "batch_index": [2]}
        with mock.patch.object(nodes.comfy.sample, "sample_custom") as call:
            result = self.sample(self.make_model(), latent, denoise=0)
        call.assert_not_called()
        self.assertEqual(result["batch_index"], [2])
        self.assertIs(result["samples"], latent["samples"])

    def test_ordinary_sampler_selection_still_works(self):
        latent = {"samples": torch.zeros(1, 48, 2, 2, 2)}
        with mock.patch.object(nodes.comfy.sample, "sample_custom", return_value=latent["samples"]) as call:
            self.sample(self.make_model(), latent, sampler_name="euler")
        self.assertEqual(call.call_args.args[3].sampler_function.__name__, "sample_euler")

    def test_invalid_schedule_inputs_fail_before_sampling(self):
        for shift in [0.0, -1.0, float("nan"), float("inf")]:
            with self.subTest(shift=shift), self.assertRaisesRegex(ValueError, "flow_shift"):
                nodes.VimooScheduler().get_sigmas("vimoo", 3, shift)
        with self.assertRaisesRegex(ValueError, "steps"):
            nodes.VimooScheduler().get_sigmas("vimoo", 0, 6.0)

    def test_scheduler_preserves_sigma_grids_and_rejects_mismatched_recipe(self):
        scheduler = nodes.VimooScheduler()
        ordinary = scheduler.get_sigmas("vimoo", 50, 12.0)[0]
        training = torch.from_numpy(1.0 - np.linspace(1.0, 0.001, 1000)[::-1].copy()).float()
        shifted = 12.0 * training / (1.0 + 11.0 * training)
        grid = np.linspace(shifted[0].item(), 0.0, 51)[:-1]
        source = np.concatenate((12.0 * grid / (1.0 + 11.0 * grid), [0.0])).astype(np.float32)
        torch.testing.assert_close(ordinary, torch.from_numpy(source), rtol=0, atol=0)
        self.assertTrue(ordinary[0] < 1.0)
        choices = scheduler.INPUT_TYPES()["required"]["scheduler"][0]
        for name in ("distilled", "dmd", "unsupported"):
            with self.subTest(scheduler=name):
                self.assertNotIn(name, choices)
                with self.assertRaisesRegex(ValueError, "Unknown"):
                    scheduler.get_sigmas(name, 8, 6.0)

    def test_sampler_consumes_external_sigmas_without_regenerating_or_mutating_them(self):
        sigmas = torch.tensor([0.99, 0.8, 0.1, 0.0])
        original = sigmas.clone()
        latent = {"samples": torch.zeros(1, 48, 2, 2, 2)}
        model = self.make_flow_model()
        conditioning = [[torch.zeros(1, 1, 1), {}]]
        actual = self.sample(model, latent, positive=conditioning, negative=None, sigmas=sigmas)["samples"]
        changed_grid = torch.tensor([0.99, 0.6, 0.4, 0.0])
        changed = self.sample(model, latent, positive=conditioning, negative=None, sigmas=changed_grid)[
            "samples"
        ]
        self.assertFalse(torch.equal(actual, changed))
        torch.testing.assert_close(sigmas, original, rtol=0, atol=0)
        for values in ([], [1], [0, 1, 0], [1, float("nan"), 0], [1, 0.5]):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "sigmas"):
                self.sample(model, latent, sigmas=torch.tensor(values))

    def test_video_dimensions_are_not_silently_rounded(self):
        for width, height, length in [(769, 512, 33), (768, 513, 33), (768, 512, 34)]:
            with (
                self.subTest(size=(width, height, length)),
                self.assertRaisesRegex(ValueError, "not rounded"),
            ):
                nodes.VimooEmptyLatentVideo().generate(width, height, length)
        samples = nodes.VimooEmptyLatentVideo().generate(1280, 720, 121)[0]["samples"]
        self.assertEqual(samples.shape, (1, 48, 31, 45, 80))

    def test_literal_video_sizes_are_rejected_before_execution(self):
        shared = sys.modules[f"{spec.name}.vimoo_ti2v_nodes"]
        for node in (nodes.VimooEmptyLatentVideo, nodes.VimooImageToVideo, shared.VimooTI2VConditioning):
            with self.subTest(node=node.__name__):
                self.assertIs(node.VALIDATE_INPUTS(1280, 720, 121), True)
                self.assertIn("multiples of 16", node.VALIDATE_INPUTS(1279, 720, 121))
                self.assertIn("multiples of 16", node.VALIDATE_INPUTS(1280, 719, 121))
                self.assertIn("4n+1", node.VALIDATE_INPUTS(1280, 720, 120))
                self.assertIn("at least 5", node.VALIDATE_INPUTS(1280, 720, 1))
                self.assertIs(node.VALIDATE_INPUTS(None, 720, 121), True)
                self.assertIn("4n+1", node.VALIDATE_INPUTS(None, None, 120))
        for node in (nodes.VimooEmptyLatentVideo, nodes.VimooImageToVideo):
            with self.subTest(node=node.__name__):
                self.assertIn("at most 8192", node.VALIDATE_INPUTS(8208, 720, 121))
                self.assertIn("at most 1000", node.VALIDATE_INPUTS(1280, 720, 1001))

    def test_batch_indices_reuse_the_original_bfloat16_noise_stream(self):
        latent = torch.zeros(4, 48, 2, 2, 2)
        original = samplers._prepare_vimoo_noise(latent, 42)
        expected = (
            torch.randn(
                (4, 2, 48, 2, 2),
                generator=torch.Generator("cpu").manual_seed(42),
                dtype=torch.bfloat16,
            )
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .float()
        )
        torch.testing.assert_close(original, expected, rtol=0, atol=0)
        indices = [3, 1, 3]
        selected = samplers._prepare_vimoo_noise(latent[:3], 42, indices)
        torch.testing.assert_close(selected, original[indices], rtol=0, atol=0)
        direct = samplers._prepare_vimoo_noise(latent, 42, source_t2v=True)
        source = torch.randn(
            latent.shape, generator=torch.Generator("cpu").manual_seed(42), dtype=torch.bfloat16
        ).float()
        torch.testing.assert_close(direct, source, rtol=0, atol=0)
        self.assertFalse(torch.equal(direct, original))
        remapped = samplers._prepare_vimoo_noise(latent[:3], 42, indices, source_t2v=True)
        torch.testing.assert_close(remapped, direct[indices], rtol=0, atol=0)

    def test_nonfinite_sampler_parameters_are_rejected(self):
        latent = {"samples": torch.zeros(1, 48, 2, 2, 2)}
        for options in [
            {"denoise": float("nan")},
            {"denoise": -0.1},
            {"denoise": 1.1},
            {"cfg": float("inf")},
            {"guidance_rescale": float("nan")},
        ]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.sample(self.make_model(), latent, **options)

    def test_unipc_rejects_sigma_boundary_before_inference_without_adjusting_grid(self):
        model = self.make_flow_model()
        latent = {"samples": torch.zeros(1, 48, 2, 2, 2)}
        conditioning = [[torch.zeros(1, 1, 1), {}]]
        for first_sigma in (1.0, 1.2, 1.0 - 1e-10):
            sigmas = torch.tensor([first_sigma, 0.8, 0.1, 0.0], dtype=torch.float64)
            original = sigmas.clone()
            with (
                self.subTest(first_sigma=first_sigma),
                self.assertRaisesRegex(ValueError, "strictly between"),
            ):
                self.sample(model, latent, positive=conditioning, negative=conditioning, sigmas=sigmas)
            self.assertEqual(model.model.diffusion_model.forward_calls, 0)
            torch.testing.assert_close(sigmas, original, rtol=0, atol=0)
        for values in ([0.8, 0.8 - 1e-10, 0.1, 0], [0.8, 1e-50, 0]):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "decreasing"):
                self.sample(model, latent, sigmas=torch.tensor(values, dtype=torch.float64))
            self.assertEqual(model.model.diffusion_model.forward_calls, 0)
        for first_sigma, sampler_name in ((0.999999, "vimoo_uni_pc"), (1.0, "euler")):
            sigmas = torch.tensor([first_sigma, 0.8, 0.1, 0.0])
            original = sigmas.clone()
            with self.subTest(first_sigma=first_sigma, sampler_name=sampler_name):
                result = self.sample(
                    model,
                    latent,
                    positive=conditioning,
                    negative=conditioning,
                    sigmas=sigmas,
                    sampler_name=sampler_name,
                )
                self.assertTrue(torch.isfinite(result["samples"]).all())
                torch.testing.assert_close(sigmas, original, rtol=0, atol=0)

    def test_full_i2v_dispatch_preserves_external_euler_trajectory_and_clone_isolation(self):
        from types import SimpleNamespace

        i2v = importlib.import_module(f"{spec.name}.vimoo_i2v")
        model = self.make_flow_model()
        dit = model.model.diffusion_model
        dit.pe_embedder = SimpleNamespace(ref_fps=None, use_reciprocal_temporal_scale=False)
        dit._math_inputs_cache_key = "ordinary"
        latent = {"samples": torch.zeros(1, 48, 2, 2, 2)}
        conditioning = [[torch.zeros(1, 1, 1), {}]]
        ordinary_options = dict(positive=conditioning, negative=None, guidance_rescale=0.0)
        ordinary_before = self.sample(model, latent, **ordinary_options)["samples"]
        calls_before_full = dit.forward_calls
        sigmas = torch.tensor([1.0, 0.7, 0.2, 0.0])
        original_sigmas = sigmas.clone()
        original_format = model.model.latent_format
        expected = i2v.i2v_initial_state(latent["samples"].shape, 42, "cpu").float()
        for current, following in zip(sigmas[:-1], sigmas[1:]):
            flow = (expected.bfloat16() * dit.weight.detach()).float()
            expected = expected + flow * (following - current)
        expected = i2v.VimooFullI2VLatentFormat().process_out(expected)
        result = self.sample(
            model,
            latent,
            positive=conditioning,
            negative=None,
            sigmas=sigmas,
            sampler_name="vimoo_i2v_euler",
            guidance_rescale=0.0,
        )
        torch.testing.assert_close(result["samples"], expected, rtol=0, atol=0)
        torch.testing.assert_close(sigmas, original_sigmas, rtol=0, atol=0)
        self.assertEqual(dit.forward_calls, calls_before_full + 3)
        # Native clones share modules; reloading the original restores its object patches.
        ordinary_after = self.sample(model, latent, **ordinary_options)["samples"]
        torch.testing.assert_close(ordinary_after, ordinary_before, rtol=0, atol=0)
        self.assertIs(model.model.latent_format, original_format)
        self.assertIsNone(dit.pe_embedder.ref_fps)
        self.assertFalse(dit.pe_embedder.use_reciprocal_temporal_scale)
        self.assertEqual(dit._math_inputs_cache_key, "ordinary")
        self.assertEqual(conditioning[0][1], {})

    def test_full_i2v_unsupported_inputs_fail_before_inference(self):
        invalid = (
            {"denoise": 0.0},
            {"denoise": 0.5},
            {"cfg": 0.5},
            {"guidance_rescale": 1.2},
            {"fps": 0.0},
            {"fps": float("inf")},
            {"sigmas": torch.tensor([0.999, 0.5, 0.0])},
            {"sigmas": torch.tensor([1.0, 0.5, 0.5, 0.0])},
            {"sigmas": torch.tensor([1.0, float("nan"), 0.0])},
            {"sigmas": torch.tensor([1.0, 0.5])},
        )
        for overrides in invalid:
            model = self.make_flow_model()
            options = dict(sampler_name="vimoo_i2v_euler", sigmas=torch.tensor([1.0, 0.0]))
            options.update(overrides)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.sample(model, {"samples": torch.zeros(1, 48, 2, 2, 2)}, **options)
            self.assertEqual(model.model.diffusion_model.forward_calls, 0)
        for extra in ({"noise_mask": torch.ones(1, 1, 2, 2, 2)}, {"batch_index": [0]}):
            model = self.make_flow_model()
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.sample(
                    model,
                    {"samples": torch.zeros(1, 48, 2, 2, 2), **extra},
                    sampler_name="vimoo_i2v_euler",
                    sigmas=torch.tensor([1.0, 0.0]),
                )
            self.assertEqual(model.model.diffusion_model.forward_calls, 0)

    def test_full_i2v_rejects_incompatible_models_and_cfg_patches(self):
        for mode in ("distilled", "lite", "fp32", "cfg_patch"):
            model = self.make_flow_model()
            if mode == "distilled":
                model.model_options["vimoo_distilled"] = {"task": "i2v"}
            elif mode == "lite":
                model.model.diffusion_model.hms = True
            elif mode == "fp32":
                model.model.diffusion_model.dtype = torch.float32
            else:
                model.model_options["sampler_cfg_function"] = lambda args: args["cond"]
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.sample(
                    model,
                    {"samples": torch.zeros(1, 48, 2, 2, 2)},
                    sampler_name="vimoo_i2v_euler",
                    sigmas=torch.tensor([1.0, 0.0]),
                )
            self.assertEqual(model.model.diffusion_model.forward_calls, 0)

    def test_first_frame_conditioning_requires_full_i2v_sampler(self):
        latent = {"samples": torch.zeros(1, 48, 2, 2, 2)}
        positive = [[torch.zeros(1, 1, 1), {"vimoo_ref_latent": torch.zeros(1, 48, 1, 2, 2)}]]
        for sampler_name in ("euler", "vimoo_uni_pc"):
            with self.subTest(sampler=sampler_name), self.assertRaisesRegex(ValueError, "vimoo_i2v_euler"):
                self.sample(self.make_flow_model(), latent, positive=positive, sampler_name=sampler_name)

    def test_reused_image_conditioning_clears_previous_first_frame_pin(self):
        old_ref = torch.ones(1, 48, 1, 2, 2)
        conditioning = [[torch.zeros(1, 1, 1), {"vimoo_ref_latent": old_ref}]]
        vae = mock.Mock()
        vae.encode.return_value = torch.zeros(1, 48, 1, 2, 2)
        with mock.patch.object(nodes, "encode_vimoo_vision", return_value=object()):
            positive, negative, _ = nodes.VimooImageToVideo().encode(
                conditioning,
                vae,
                object(),
                torch.zeros(1, 32, 32, 3),
                32,
                32,
                5,
                first_frame_replace=False,
                negative=conditioning,
            )
        self.assertIsNone(samplers._find_ref_latent(positive))
        self.assertIsNone(samplers._find_ref_latent(negative))
        self.assertIs(conditioning[0][1]["vimoo_ref_latent"], old_ref)

    def test_image_conditioning_accepts_only_positive_text(self):
        conditioning = [[torch.zeros(1, 1, 1), {"text": "keep"}]]
        vae = mock.Mock()
        vae.encode.return_value = torch.zeros(1, 48, 1, 2, 2)
        with mock.patch.object(nodes, "encode_vimoo_vision", return_value=object()):
            positive, negative, latent = nodes.VimooImageToVideo().encode(
                conditioning, vae, object(), torch.zeros(1, 32, 32, 3), 32, 32, 5
            )
        self.assertIsNone(negative)
        self.assertEqual(positive[0][1]["text"], "keep")
        self.assertEqual(conditioning[0][1], {"text": "keep"})
        self.assertEqual(tuple(latent["samples"].shape), (1, 48, 2, 2, 2))

    def test_sampler_accepts_omitted_negative_only_when_guidance_does_not_need_it(self):
        sampler = nodes.VimooSampler()
        parameters = dict(
            model=self.make_flow_model(),
            positive=[[torch.zeros(1, 1, 1), {}]],
            latent_image={"samples": torch.zeros(1, 48, 2, 2, 2)},
            sigmas=nodes.VimooScheduler().get_sigmas("vimoo", 3, 6.0)[0],
            cfg=1.0,
            guidance_rescale=0.0,
            seed=42,
            sampler_name="vimoo_uni_pc",
            denoise=1.0,
        )
        with mock.patch.object(
            nodes, "sample_standard", return_value=(parameters["latent_image"],)
        ) as sample:
            sampler.sample(**parameters)
            self.assertIsNone(sample.call_args.args[2])
            self.assertEqual(sampler.check_lazy_status(cfg=4.5, negative=None), ["negative"])
            self.assertEqual(sampler.check_lazy_status(cfg=4.5), [])
            with self.assertRaisesRegex(ValueError, "negative conditioning"):
                sampler.sample(**{**parameters, "cfg": 4.5})
            sample.assert_called_once()

    def test_unipc_supports_batch_shaped_timesteps(self):
        noise = torch.randn(2, 48, 2, 2, 2)
        sigmas = nodes.vimoo_unipc_sigmas(6.0, 3)
        visited = []

        def model(x, sigma):
            self.assertEqual(sigma.shape, (2,))
            return torch.full_like(x, 0.125)

        result = samplers.sample_vimoo_uni_pc(
            model, noise, sigmas, callback=lambda state: visited.append(state["i"])
        )
        self.assertEqual(visited, [0, 1, 2])
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(result).all())

    def test_custom_api_matches_bf16_flow_arithmetic_with_tiny_model(self):
        model = self.make_flow_model()
        latent = {"samples": torch.zeros(1, 48, 2, 2, 2), "batch_index": [0]}
        sigmas = nodes.vimoo_unipc_sigmas(12.0, 3)
        noise = torch.randn(
            latent["samples"].shape,
            generator=torch.Generator("cpu").manual_seed(42),
            dtype=torch.bfloat16,
        )
        glyph = torch.zeros(1, 256, 1472, dtype=torch.bfloat16)
        positive = [[torch.zeros(1, 1, 1), {"conditioning_byt5small": glyph}]]
        negative = [[torch.zeros(1, 1, 1), {"conditioning_byt5small": glyph}]]

        def reference_model(x, sigma):
            cond = (x.bfloat16() * model.model.diffusion_model.weight.detach()).bfloat16()
            guided = cond + 4.5 * (cond - cond)
            std_cond = cond.std(dim=(1, 2, 3, 4), keepdim=True)
            std_guided = guided.std(dim=(1, 2, 3, 4), keepdim=True)
            rescaled = guided * (std_cond / std_guided)
            guided = 1.2 * rescaled + (1.0 - 1.2) * guided
            return guided.bfloat16()

        expected = samplers.sample_vimoo_uni_pc(reference_model, noise, sigmas)
        actual = self.sample(
            model,
            latent,
            positive=positive,
            negative=negative,
            cfg=4.5,
            guidance_rescale=1.2,
            sigmas=sigmas,
        )
        torch.testing.assert_close(actual["samples"], expected.float(), rtol=0, atol=0)
        self.assertEqual(actual["batch_index"], [0])
        for conditioning in (positive, negative):
            self.assertEqual(set(conditioning[0][1]), {"conditioning_byt5small"})
            self.assertIs(conditioning[0][1]["conditioning_byt5small"], glyph)
        self.assertEqual(glyph.dtype, torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "inpainting masks"):
            self.sample(
                model,
                {"samples": latent["samples"], "noise_mask": torch.ones(1, 1, 2, 2, 2)},
                positive=positive,
                negative=negative,
            )


VimooSaveVideo = nodes.VimooSaveVideo
encoder_available = nodes.encoder_available
frame_bytes = nodes.frame_bytes
write_video = nodes.write_video


class VideoWriting(unittest.TestCase):
    def test_libx264_round_trip_keeps_size_and_frame_count(self):
        frames = torch.rand(9, 64, 96, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "clip.mp4")
            self.assertEqual(write_video(frames, path, 24, "libx264", 19), "libx264")
            self.assertGreater(os.path.getsize(path), 0)
            import av

            with av.open(path) as container:
                stream = container.streams.video[0]
                self.assertEqual((stream.codec_context.height, stream.codec_context.width), (64, 96))
                self.assertEqual(sum(1 for _ in container.decode(video=0)), 9)

    def test_conversion_falls_back_to_the_host_when_the_card_is_full(self):
        frames = torch.rand(3, 8, 8, 3)
        move = torch.Tensor.to

        def refuse(self, *args, **kwargs):
            if args and args[0] == "cuda":
                raise torch.cuda.OutOfMemoryError("no room")
            return move(self, *args, **kwargs)

        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.Tensor, "to", refuse),
            patch.object(torch.cuda, "empty_cache"),
        ):
            converted = frame_bytes(frames)
        self.assertEqual(converted.shape, (3, 8, 8, 3))
        self.assertEqual(converted.dtype, numpy.uint8)

    def test_unknown_encoder_is_reported_as_unavailable(self):
        self.assertFalse(encoder_available("not_a_real_encoder"))

    def test_node_writes_the_clip_and_points_the_ui_at_it(self):
        frames = torch.rand(5, 64, 96, 3)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(folder_paths, "get_output_directory", return_value=directory):
                result = VimooSaveVideo().save(frames, "unit_test", 24.0, "libx264", 30)
        entry = result["ui"]["images"][0]
        self.assertTrue(entry["filename"].endswith(".mp4"))
        self.assertEqual(result["ui"]["animated"], (True,))
        self.assertEqual(result["ui"]["encoder"], ["libx264"])

    def test_node_rejects_an_encoder_this_build_cannot_open(self):
        with self.assertRaises(ValueError):
            VimooSaveVideo().save(torch.rand(2, 16, 16, 3), "unit_test", 24.0, "not_a_real_encoder", 30)

    def test_saved_metadata_restores_workflow_without_changing_video_packets(self):
        import av
        import hashlib
        import json

        frames = torch.rand(5, 64, 96, 3, generator=torch.Generator().manual_seed(1234))
        workflow = {"nodes": [{"id": 1, "type": "VimooSampler", "widgets_values": [1234]}]}
        prompt = {"1": {"class_type": "VimooTextEncode", "inputs": {"text": 'A sign says "Vimoo". 中文'}}}
        payloads = []
        with tempfile.TemporaryDirectory() as directory:
            for disabled in (False, True):
                with (
                    patch.object(folder_paths, "get_output_directory", return_value=directory),
                    patch.object(nodes.args, "disable_metadata", disabled),
                ):
                    result = VimooSaveVideo().save(
                        frames,
                        "metadata",
                        24.0,
                        "libx264",
                        19,
                        prompt=prompt,
                        extra_pnginfo={"workflow": workflow},
                    )
                path = os.path.join(directory, result["ui"]["images"][0]["filename"])
                with av.open(path) as container:
                    if disabled:
                        self.assertNotIn("workflow", container.metadata)
                        self.assertNotIn("prompt", container.metadata)
                    else:
                        self.assertEqual(json.loads(container.metadata["workflow"]), workflow)
                        self.assertEqual(json.loads(container.metadata["prompt"]), prompt)
                    packets = b"".join(
                        bytes(packet) for packet in container.demux(video=0) if packet.dts is not None
                    )
                    payloads.append(hashlib.sha256(packets).hexdigest())
        self.assertEqual(payloads[0], payloads[1])

    def test_encoder_fallback_preserves_metadata(self):
        frames = torch.rand(2, 16, 16, 3)
        metadata = {"workflow": {"nodes": []}, "prompt": {"1": {"inputs": {"seed": 1234}}}}
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(folder_paths, "get_output_directory", return_value=directory),
                patch.object(nodes.args, "disable_metadata", False),
                patch.object(nodes, "encoder_available", return_value=True),
                patch.object(
                    nodes, "write_video", side_effect=[RuntimeError("encoder failed"), "libx264"]
                ) as writer,
            ):
                result = VimooSaveVideo().save(
                    frames,
                    "metadata",
                    24.0,
                    "h264_nvenc",
                    19,
                    prompt=metadata["prompt"],
                    extra_pnginfo={"workflow": metadata["workflow"]},
                )
        self.assertEqual(result["ui"]["encoder"], ["libx264"])
        self.assertEqual(writer.call_args_list[0].kwargs["metadata"], metadata)
        self.assertEqual(writer.call_args_list[1].kwargs["metadata"], metadata)


if __name__ == "__main__":
    unittest.main()
