"""Output sizing, image orientation and the reduced release interfaces."""

from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import torch
from test_nodes_runtime import plugin, spec

resolution = sys.modules[f"{spec.name}.vimoo_resolution"]
performance = sys.modules[f"{spec.name}.vimoo_performance"]
vision = sys.modules[f"{spec.name}.vimoo_vision"]
vae = sys.modules[f"{spec.name}.vimoo_vae"]


class ResolutionTests(unittest.TestCase):
    def test_mlx_examples_and_nonstandard_ratio(self):
        for source, expected in [
            ((3840, 2160), (1280, 720)),
            ((2160, 3840), (720, 1280)),
            ((4000, 4000), (960, 960)),
            ((640, 360), (1280, 720)),
            ((100000, 1), (57600, 16)),
            ((1, 100000), (16, 57600)),
        ]:
            self.assertEqual(resolution.reference_output_size(*source), expected)
        width, height = resolution.reference_output_size(1500, 1000)
        self.assertEqual((width, height), (1168, 768))
        self.assertLessEqual(width * height, 1280 * 720)

    def test_matches_actual_mlx_function(self):
        mlx = Path(__file__).resolve().parents[2] / "Vimoo-1.0-MLX/vimoo_mlx/image_processing.py"
        if not mlx.exists():
            self.skipTest("Sibling MLX checkout is not installed")
        # Execute just the pure sizing function; MLX itself is not required on CUDA hosts.
        import ast
        from math import sqrt

        tree = ast.parse(mlx.read_text())
        function = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "reference_output_size"
        )
        namespace = {"sqrt": sqrt}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(mlx), "exec"), namespace)
        for size in [(3840, 2160), (2160, 3840), (4000, 4000), (1500, 1000), (640, 360), (100000, 1)]:
            self.assertEqual(
                resolution.reference_output_size(*size),
                namespace["reference_output_size"](SimpleNamespace(width=size[0], height=size[1])),
            )

    def test_manual_ignores_image_and_auto_ignores_manual_values(self):
        node = resolution.VimooResolution()
        image = torch.empty(1, 900, 600, 3)
        self.assertEqual(node.resolve("manual", 640, 480, "i2v", image)["result"], (640, 480))
        self.assertEqual(node.resolve("auto", 1, 1, "i2v", image)["result"], (768, 1168))
        self.assertEqual(node.resolve("auto", 1, 1, "t2v")["result"], (1280, 720))
        self.assertEqual(node.check_lazy_status("auto", "t2v"), [])
        self.assertEqual(node.check_lazy_status("manual", "i2v"), [])
        self.assertEqual(node.check_lazy_status("auto", "i2v"), ["image"])

    def test_errors(self):
        node = resolution.VimooResolution()
        with self.assertRaisesRegex(ValueError, "reference image"):
            node.resolve("auto", 1280, 720, "i2v")
        with self.assertRaisesRegex(ValueError, "manual"):
            node.resolve("auto", 1280, 720, "i2v", torch.empty(1, 1, 100000, 3))
        for width in (15, 8193, 0):
            with self.assertRaises(ValueError):
                node.resolve("manual", width, 720, "t2v")

    def test_exif_orientation_before_sizing(self):
        image = Image.new("RGB", (600, 900))
        exif = image.getexif()
        exif[274] = 6
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rotated.png"
            image.save(path, exif=exif)
            loaded = vision.load_pil_image(path)
        self.assertEqual(tuple(loaded.shape[1:3]), (600, 900))
        self.assertEqual(
            resolution.VimooResolution().resolve("auto", 1280, 720, "i2v", loaded)["result"], (1168, 768)
        )

    def test_public_controls_and_release_performance(self):
        self.assertEqual(
            set(performance.VimooPerformanceOptions.INPUT_TYPES()["required"]),
            {"model", "attention", "compile"},
        )
        self.assertEqual(
            set(vae.VimooVAELoader.INPUT_TYPES()["required"]), {"vae_name", "tiled_decode", "model"}
        )
        self.assertNotIn("optional", plugin.NODE_CLASS_MAPPINGS["VimooModelLoader"].INPUT_TYPES())
        node = performance.VimooPerformanceOptions()
        model = SimpleNamespace(model_options={"vimoo_sampling_profile": "standard_t2v"})
        with patch.object(performance, "apply_performance", return_value=model) as configure:
            node.patch(model, "sage", False)
            options = vars(configure.call_args.args[1])
            self.assertEqual(options["compile"], "disabled")
            self.assertTrue(options["precise_math"])
            self.assertFalse(options["sage_image_only"])
            self.assertEqual(options["text_padding"], "drop")
            self.assertEqual(options["sage_pv_dtype"], "fp8")
            node.patch(model, "pytorch", True)
            self.assertEqual(configure.call_args.args[1].compile, "regional")
        with self.assertRaisesRegex(ValueError, "Boolean"):
            node.patch(model, "pytorch", "disabled")

    def test_all_release_profiles_preserve_effective_performance_defaults(self):
        profiles = [
            ({"vimoo_sampling_profile": "standard_t2v"}, True),
            ({"vimoo_sampling_profile": "standard_i2v"}, True),
            ({"vimoo_sampling_profile": "lite_ti2v_anchor"}, False),
            ({"vimoo_distilled": {"profile": "t2v_distilled"}}, True),
            ({"vimoo_distilled": {"profile": "i2v_distilled"}}, True),
            ({"vimoo_distilled": {"profile": "lite_ti2v_distilled"}}, True),
        ]
        for options, precise in profiles:
            distilled = options.get("vimoo_distilled", {}).get("profile")
            lite_distilled = distilled == "lite_ti2v_distilled"
            # Ordinary Lite TI2V keeps the conservative padding and image-only split.
            joint = options.get("vimoo_sampling_profile") != "lite_ti2v_anchor"
            for attention, compiled in (("sage", False), ("fa2", True)):
                with self.subTest(options=options, attention=attention, compile=compiled):
                    config = performance.release_performance_config(
                        SimpleNamespace(model_options=options), attention, compiled
                    )
                    self.assertEqual(config.attention, attention)
                    self.assertEqual(config.compile, "regional" if compiled else "disabled")
                    self.assertEqual(config.compile_mode, "default")
                    self.assertEqual(config.compile_dynamic, "auto")
                    self.assertEqual(config.precise_math, precise)
                    self.assertTrue(config.block_prefetch)
                    self.assertEqual(config.memory_usage_factor, 4.0)
                    self.assertEqual(config.text_padding, "drop" if joint else "keep")
                    self.assertEqual(config.fused_qkv, lite_distilled)
                    self.assertEqual(config.sage_scope, "joint")
                    self.assertEqual(config.sage_fallback, "pytorch")
                    self.assertEqual(config.sage_qk_quant_gran, "per_thread")
                    self.assertEqual(config.sage_pv_dtype, "fp8")
                    self.assertEqual(config.sage_pv_accum_dtype, "fp32+fp32")
                    self.assertTrue(config.sage_smooth_k)
                    self.assertFalse(config.sage_smooth_v)
                    self.assertEqual(config.sage_image_only, not joint)
                    self.assertEqual(config.sage_exact_kernel, "fp32")
                    self.assertEqual(config.sage_fa2_first_blocks, 0)
                    self.assertEqual(config.sage_fa2_last_blocks, 0)

    def test_vae_role_defaults_and_adapter_profile(self):
        model = SimpleNamespace(
            model_options={
                "vimoo_distilled": {"profile": "lite_ti2v_distilled"},
                "vimoo_vae_numerical_profile": "wan_fp32_norm",
            }
        )
        with patch.object(vae, "_load_vimoo_vae") as load:
            vae.VimooReferenceVAELoader().load_reference_vae("vae", model)
            reference = load.call_args.kwargs
            vae.VimooVAELoader().load_vae("vae", True, model)
            output = load.call_args.kwargs
        self.assertEqual((reference["tile_height"], reference["stride_height"]), (256, 192))
        self.assertEqual((output["tile_height"], output["stride_height"]), (512, 448))
        self.assertEqual(output["numerical_profile"], "wan_fp32_norm")


class AdapterPerformanceTests(unittest.TestCase):
    def test_adapter_rebinds_profile_and_preserves_user_controls(self):
        model = SimpleNamespace(model_options={"vimoo_sampling_profile": "lite_ti2v_anchor"})
        with patch.object(performance, "apply_performance", return_value=model) as configure:
            performance.VimooPerformanceOptions().patch(model, "sage", False)
            self.assertFalse(configure.call_args.args[1].precise_math)
            self.assertTrue(configure.call_args.args[1].block_prefetch)
            self.assertEqual(configure.call_args.args[1].text_padding, "keep")
            model.model_options["vimoo_distilled"] = {"profile": "lite_ti2v_distilled"}
            self.assertIs(performance.reapply_release_performance(model), model)
            self.assertTrue(configure.call_args.args[1].precise_math)
            self.assertTrue(configure.call_args.args[1].block_prefetch)
            self.assertEqual(configure.call_args.args[1].text_padding, "drop")
            self.assertFalse(configure.call_args.args[1].sage_image_only)
            self.assertEqual(configure.call_args.args[1].attention, "sage")
            self.assertEqual(configure.call_args.args[1].compile, "disabled")
