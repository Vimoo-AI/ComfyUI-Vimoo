"""Full I2V solver boundaries; run in the native ComfyUI environment."""

import importlib
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageOps
import torch

from test_nodes_runtime import spec

i2v = importlib.import_module(f"{spec.name}.vimoo_i2v")


class FullI2VTests(unittest.TestCase):
    def test_distilled_native_solver_emits_fp32_reference_after_bf16_normalization(self):
        nodes = sys.modules[f"{spec.name}.vimoo_nodes"]
        times = [
            1000.0,
            988.2352905273438,
            975.0,
            960.0,
            911.2781982421875,
            837.73583984375,
            713.924072265625,
            461.5384521484375,
        ]
        sigmas = nodes.VimooScheduler().get_sigmas("i2v_distilled", 8, 6.0)[0]
        latent = torch.zeros(1, 48, 3, 2, 2)
        reference = torch.linspace(-2, 2, 48 * 4).reshape(1, 48, 1, 2, 2)
        latent_format = sys.modules[f"{spec.name}.vimoo_vae"].VimooLatentFormat()
        mean = latent_format.latents_mean.bfloat16()
        inverse_std = (1.0 / latent_format.latents_std).bfloat16()
        source_reference = ((reference.bfloat16() - mean) * inverse_std).float()
        recorded = []

        class LifecycleGuider:
            def __init__(self, *args):
                pass

            def set_conds(self, *args):
                pass

            def set_cfg(self, *args):
                pass

            def sample(self, noise, initial, sampler, active_sigmas, **kwargs):
                return sampler.sampler_function(self, initial, active_sigmas)

        def observe_solver(guider, initial, active_sigmas, **kwargs):
            recorded.append(kwargs["reference"])
            return initial

        for distilled in (False, True):
            recipe = (
                None
                if not distilled
                else {
                    "profile": "i2v_distilled",
                    "task": "i2v",
                    "num_inference_steps": 8,
                    "custom_timesteps": times,
                    "timestep_space": "post_shift",
                    "flow_shift": 6.0,
                    "i2v_first_frame_replace": True,
                }
            )
            model = SimpleNamespace(
                model=SimpleNamespace(
                    diffusion_model=SimpleNamespace(hms=False),
                    model_sampling=SimpleNamespace(),
                    latent_format=latent_format,
                    get_dtype=lambda: torch.bfloat16,
                ),
                model_options={},
                add_object_patch=lambda *args: None,
            )
            model.clone = lambda: model
            with (
                patch.object(i2v, "I2VFlowGuider", LifecycleGuider),
                patch.object(i2v, "sample_i2v_euler", side_effect=observe_solver),
            ):
                i2v.sample_full_i2v(
                    model,
                    [[torch.zeros(1), {"vimoo_ref_latent": reference}]],
                    [],
                    {"samples": latent},
                    sigmas,
                    cfg=1.0,
                    guidance_rescale=0.0,
                    recipe=recipe,
                )
            self.assertEqual(recorded[-1].dtype, torch.float32 if distilled else torch.bfloat16)
            torch.testing.assert_close(recorded[-1].float(), source_reference, rtol=0, atol=0)

    def test_distilled_manual_grid_uses_raw_timestep_delta_and_post_update_fp32_replacement(self):
        times = [
            1000.0,
            988.2352905273438,
            975.0,
            960.0,
            911.2781982421875,
            837.73583984375,
            713.924072265625,
            461.5384521484375,
        ]
        nodes = sys.modules[f"{spec.name}.vimoo_nodes"]
        connected = nodes.VimooScheduler().get_sigmas("i2v_distilled", 8, 6.0)[0]
        torch.testing.assert_close(
            connected,
            torch.tensor([*(value / 1000 for value in times), 0.0], dtype=torch.float64),
            rtol=0,
            atol=0,
        )
        initial = i2v.i2v_initial_state((1, 2, 3, 2, 2), 1234, "cpu")
        reference = torch.full_like(initial[:, :, :1].float(), 1.0 + 2**-10)
        flow = torch.full_like(initial, 0.123)
        observed = []

        def predict(value, timestep):
            observed.append(value.clone())
            return flow.float()

        output = i2v.sample_i2v_euler(
            predict, initial, connected.float(), reference=reference, noise=initial, manual_timesteps=times
        )
        expected = initial
        for index, current in enumerate(times):
            torch.testing.assert_close(observed[index], expected.float(), rtol=0, atol=0)
            following = times[index + 1] if index + 1 < len(times) else 0.0
            expected = (expected.float() + ((following - current) / 1000.0) * flow.float()).bfloat16()
            pinned = (
                (1 - following / 1000.0) * reference + (following / 1000.0) * initial[:, :, :1].float()
            ).bfloat16()
            expected = torch.cat((pinned, expected[:, :, 1:]), 2)
        self.assertEqual(output.dtype, torch.bfloat16)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_public_scheduler_has_exact_full_noise_endpoint_and_shift(self):
        nodes = sys.modules[f"{spec.name}.vimoo_nodes"]
        sigmas = nodes.VimooScheduler().get_sigmas("vimoo_i2v_euler", 4, 6.0)[0]
        torch.testing.assert_close(sigmas, torch.tensor([1.0, 18 / 19, 6 / 7, 2 / 3, 0.0]), rtol=0, atol=0)
        self.assertEqual(sigmas.dtype, torch.float32)
        self.assertNotEqual(nodes.VimooScheduler().get_sigmas("vimoo", 4, 6.0)[0][0].item(), 1.0)

    def test_decode_preserves_sub_bf16_euler_precision(self):
        latent = torch.tensor([1.0 + 2**-10], dtype=torch.float32)
        full = i2v.VimooFullI2VLatentFormat()
        full.latents_mean = torch.tensor([1.0])
        full.latents_std = torch.tensor([2.0])
        decoded_input = full.process_out(latent)
        torch.testing.assert_close(decoded_input, torch.tensor([3.0 + 2**-9]), rtol=0, atol=0)
        legacy = sys.modules[f"{spec.name}.vimoo_vae"].VimooLatentFormat()
        legacy.latents_mean, legacy.latents_std = full.latents_mean, full.latents_std
        torch.testing.assert_close(
            legacy.process_out(latent), torch.tensor([3.0], dtype=torch.bfloat16), rtol=0, atol=0
        )

    def test_pure50_preserves_fp32_trajectory_and_first_frame(self):
        sigmas = i2v.i2v_sigmas(50, 6.0)
        reference_video = torch.linspace(-1, 1, 24).reshape(1, 2, 3, 2, 2)
        initial = i2v.i2v_initial_state(reference_video.shape, 1234, "cpu")
        reference = reference_video[:, :, :1] / 3
        states = []
        flow = torch.full_like(initial, 0.123, dtype=torch.bfloat16)

        def model(x, timestep):
            states.append(x.clone())
            return flow.float()

        output = i2v.sample_i2v_euler(model, initial, sigmas, reference=reference, noise=initial)
        expected = initial
        for index in range(50):
            torch.testing.assert_close(states[index], expected.float(), rtol=0, atol=0)
            expected = expected.float() + flow.float() * (sigmas[index + 1] - sigmas[index])
            s = sigmas[index + 1]
            expected[:, :, :1] = (1 - s) * reference + s * initial[:, :, :1].float()
        self.assertEqual(len(states), 50)
        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(output[:, :, :1], reference, rtol=0, atol=0)
        self.assertFalse(torch.equal(output, output.bfloat16().float()))

    def test_pure_noise_draws_bcfhw_bf16(self):
        shape = (1, 2, 3, 2, 2)
        initial = i2v.i2v_initial_state(shape, 42, "cpu")
        expected = torch.randn(shape, generator=torch.Generator("cpu").manual_seed(42), dtype=torch.bfloat16)
        torch.testing.assert_close(initial, expected, rtol=0, atol=0)

    def test_guidance_keeps_bf16_rescale_order(self):
        cond = torch.tensor([0.1, 2.3, -1.7, 0.6], dtype=torch.bfloat16).reshape(1, 1, 1, 2, 2)
        uncond = torch.tensor([-0.8, 0.2, 0.4, 1.1], dtype=torch.bfloat16).reshape_as(cond)
        guided = uncond + 3 * (cond - uncond)
        rescaled = guided * (
            cond.std(dim=(1, 2, 3, 4), keepdim=True) / guided.std(dim=(1, 2, 3, 4), keepdim=True)
        )
        torch.testing.assert_close(
            i2v.i2v_guidance(cond, uncond, 3, 0.5), 0.5 * rescaled + 0.5 * guided, rtol=0, atol=0
        )
        torch.testing.assert_close(i2v.i2v_guidance(cond, uncond, 1, 0.5), cond, rtol=0, atol=0)


images = importlib.import_module(f"{spec.name}.vimoo_vision")
inputs = importlib.import_module(f"{spec.name}.vimoo_ti2v_nodes")
nodes = importlib.import_module(f"{spec.name}.vimoo_nodes")


class ImageDecodeTests(unittest.TestCase):
    def test_standard_i2v_keeps_fp32_normalization_before_vae_autocast(self):
        from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

        pixels = np.arange(11 * 17 * 3, dtype=np.uint16).reshape(11, 17, 3).astype(np.uint8)
        source = Image.fromarray(pixels)
        image = torch.from_numpy(pixels.astype(np.float32) / 255.0)[None]
        model = SimpleNamespace(model_options={"vimoo_image_preprocessing": "standard_i2v"})
        captured = []

        class VAE:
            def encode_normalized(self, value):
                captured.append(value)
                return torch.zeros(1, 48, 1, 1, 1)

        vision = SimpleNamespace(last_hidden_state=torch.zeros(1, 729, 1152))
        scale = max(16 / source.width, 16 / source.height)
        expected = Compose(
            [
                Resize(
                    (round(source.height * scale), round(source.width * scale)),
                    interpolation=InterpolationMode.LANCZOS,
                ),
                CenterCrop((16, 16)),
                ToTensor(),
                Normalize([0.5], [0.5]),
            ]
        )(source).permute(1, 2, 0)[None]
        with patch.object(nodes, "encode_vimoo_vision", return_value=vision) as encode_vision:
            positive, _, _ = nodes.VimooImageToVideo().encode(
                [[torch.zeros(1), {}]], VAE(), object(), image, 16, 16, 5, True, model=model, negative=[]
            )
        self.assertEqual(captured[0].dtype, torch.float32)
        torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)
        self.assertEqual(encode_vision.call_args.kwargs["processor_backend"], "pil")
        self.assertEqual(positive[0][1]["vimoo_vision_mask"].dtype, torch.int64)
        self.assertIsNotNone(positive[0][1]["vimoo_ref_latent"])

    def test_bound_jpeg_uses_oriented_pil_rgb(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "input.jpg"
            pixels = np.arange(8 * 12 * 3, dtype=np.uint16).reshape(8, 12, 3).astype(np.uint8)
            exif = Image.Exif()
            exif[274] = 6
            Image.fromarray(pixels).save(path, quality=91, exif=exif)
            with Image.open(path) as source:
                expected = np.asarray(ImageOps.exif_transpose(source).convert("RGB")).copy()
            model = SimpleNamespace(model_options={"vimoo_source_image_decode": "pil"})
            with (
                patch.object(inputs.folder_paths, "exists_annotated_filepath", return_value=True),
                patch.object(inputs.folder_paths, "get_annotated_filepath", return_value=str(path)),
                patch(
                    "nodes.LoadImage.load_image", side_effect=AssertionError("source decoder must use PIL")
                ),
            ):
                task, actual = inputs.VimooTI2VInput().load("i2v", path.name, model=model)
                image_only = inputs.VimooImageLoader().load(path.name, model=model)[0]
            self.assertEqual(task, "i2v")
            torch.testing.assert_close(actual, image_only, rtol=0, atol=0)
            self.assertEqual(actual.dtype, torch.float32)
            self.assertEqual(tuple(actual.shape), (1, 12, 8, 3))
            restored = np.clip(actual[0].numpy() * 255.0, 0, 255).astype(np.uint8)
            np.testing.assert_array_equal(restored, expected)

    def test_all_uint8_levels_reach_bf16_normalization_unchanged(self):
        pixels = np.repeat(np.arange(256, dtype=np.uint8).reshape(16, 16, 1), 3, axis=2)
        model = SimpleNamespace(
            model_options={
                "vimoo_source_image_decode": "pil",
                "vimoo_image_preprocessing": "lanczos_floor_bf16",
            }
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "levels.png"
            Image.fromarray(pixels).save(path)
            image = images.load_pil_image(path)
        recorded = []

        class VAE:
            def encode_normalized(self, value):
                recorded.append(value)
                return torch.zeros(1, 48, 1, 1, 1)

        vision = SimpleNamespace(last_hidden_state=torch.zeros(1, 729, 1152, dtype=torch.bfloat16))
        with patch.object(nodes, "encode_vimoo_vision", return_value=vision):
            nodes.VimooImageToVideo().encode([], VAE(), object(), image, 16, 16, 5, model=model, negative=[])
        expected = torch.from_numpy(pixels.copy()).bfloat16()[None] / 127.5 - 1.0
        self.assertEqual(recorded[0].dtype, torch.bfloat16)
        torch.testing.assert_close(recorded[0], expected, rtol=0, atol=0)

    def test_unbound_input_keeps_native_loader_and_t2v_skips_image(self):
        image = torch.ones(1, 2, 3, 3)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "reference.png"
            Image.new("RGB", (3, 2)).save(path)
            with (
                patch.object(inputs.folder_paths, "exists_annotated_filepath", return_value=True),
                patch.object(inputs.folder_paths, "get_annotated_filepath", return_value=str(path)),
                patch("nodes.LoadImage.load_image", return_value=(image, object())) as native,
            ):
                actual = inputs.VimooTI2VInput().load("i2v", path.name)
                self.assertIs(actual[1], image)
                native.assert_called_once_with(path.name)
                self.assertEqual(inputs.VimooTI2VInput().load("t2v", "none"), ("t2v", None))

    def test_corrupt_uploaded_image_is_rejected_before_execution(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "reference.png"
            path.write_bytes(b"This is not an image")
            with (
                patch.object(inputs.folder_paths, "exists_annotated_filepath", return_value=True),
                patch.object(inputs.folder_paths, "get_annotated_filepath", return_value=str(path)),
            ):
                self.assertIn(
                    "Cannot decode reference image", inputs.VimooImageLoader.VALIDATE_INPUTS(path.name)
                )
                self.assertIn(
                    "Cannot decode reference image", inputs.VimooTI2VInput.VALIDATE_INPUTS("i2v", path.name)
                )
                self.assertIs(inputs.VimooTI2VInput.VALIDATE_INPUTS("t2v", path.name), True)

    def test_patchify_bound_conditioner_preserves_crop_and_normalization_order(self):
        pixels = np.arange(11 * 17 * 3, dtype=np.uint16).reshape(11, 17, 3).astype(np.uint8)
        image = torch.from_numpy(pixels.astype(np.float32) / 255.0)[None]
        model = SimpleNamespace(model_options={"vimoo_image_preprocessing": "lite_compact"})
        recorded = []

        class VAE:
            def encode_normalized(self, value):
                recorded.append(value)
                return torch.zeros(1, 48, 1, 1, 1)

        vision = SimpleNamespace(last_hidden_state=torch.zeros(1, 729, 1152, dtype=torch.bfloat16))
        source = Image.fromarray(pixels)
        scale = max(16 / source.width, 16 / source.height)
        size = (int(round(source.width * scale)), int(round(source.height * scale)))
        resized = source.resize(size, Image.Resampling.LANCZOS)
        vision_crop = resized.crop(
            ((size[0] - 16) / 2, (size[1] - 16) / 2, (size[0] + 16) / 2, (size[1] + 16) / 2)
        )
        left, top = int(round((size[0] - 16) / 2.0)), int(round((size[1] - 16) / 2.0))
        vae_crop = resized.crop((left, top, left + 16, top + 16))
        source_pixels = torch.from_numpy(np.asarray(vae_crop).copy()).float()
        expected = source_pixels.div(255.0).sub(0.5).div(0.5).bfloat16()[None]
        bf16_pixels = torch.from_numpy(np.asarray(vae_crop).copy()).bfloat16()[None] / 127.5 - 1
        self.assertFalse(torch.equal(expected, bf16_pixels))
        with patch.object(nodes, "encode_vimoo_vision", return_value=vision) as encode_vision:
            positive, _, _ = inputs.VimooTI2VConditioning().encode(
                "i2v",
                [[torch.zeros(1), {}]],
                16,
                16,
                5,
                False,
                image=image,
                vae=VAE(),
                clip_vision=object(),
                attention_backend="precise",
                model=model,
                negative=[],
            )
        torch.testing.assert_close(recorded[0], expected, rtol=0, atol=0)
        self.assertEqual(encode_vision.call_args.kwargs["processor_backend"], "pil")
        actual_pixels = encode_vision.call_args.args[1]
        expected_pixels = torch.from_numpy(np.asarray(vision_crop, dtype=np.float32) / 255.0)[None]
        torch.testing.assert_close(actual_pixels, expected_pixels, rtol=0, atol=0)
        self.assertEqual(positive[0][1]["vimoo_vision_mask"].dtype, torch.bool)
        self.assertTrue(positive[0][1]["vimoo_vision_mask"].all())
        self.assertIsNone(positive[0][1]["vimoo_ref_latent"])

    def test_unknown_bound_decoder_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "image decoder"):
            images.validate_source_image_decode("unknown")


if __name__ == "__main__":
    unittest.main()
