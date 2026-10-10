"""Task routing must not evaluate or retain inactive image branches."""

import copy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from test_nodes_runtime import spec

ti2v = sys.modules[spec.name + ".vimoo_ti2v_nodes"]


class TI2VTests(unittest.TestCase):
    def test_fixed_noise_sampler_omits_unused_controls_and_rejects_other_profiles(self):
        node = ti2v.VimooLiteDistilledSampler()
        for section, name in (("required", "seed"), ("required", "renoise_seed"), ("optional", "add_fps")):
            self.assertNotIn(name, node.INPUT_TYPES()[section])
            self.assertIn(name, ti2v.VimooSampler.INPUT_TYPES()[section])
        with self.assertRaisesRegex(ValueError, "Use Vimoo Lite Distilled Sampler"):
            ti2v.VimooSampler().sample(
                SimpleNamespace(model_options={"vimoo_distilled": {"profile": "lite_ti2v_distilled"}}),
                None,
                None,
                None,
                1.0,
                0.0,
                1234,
                "vimoo_lite_ti2v_euler",
                1.0,
            )
        for recipe in (None, {"profile": "t2v_distilled"}, {"profile": "i2v_distilled"}):
            with (
                self.subTest(recipe=recipe),
                self.assertRaisesRegex(ValueError, "Lite TI2V Distilled adapter"),
            ):
                node.sample(model=SimpleNamespace(model_options={"vimoo_distilled": recipe}))

    def test_t2v_never_reads_an_image(self):
        with patch.object(ti2v.folder_paths, "exists_annotated_filepath", side_effect=AssertionError):
            self.assertIs(ti2v.VimooTI2VInput.VALIDATE_INPUTS("t2v", "missing.png"), True)
            self.assertEqual(ti2v.VimooTI2VInput.IS_CHANGED("t2v", "missing.png"), "t2v")
            self.assertEqual(ti2v.VimooTI2VInput().load("t2v", "missing.png"), ("t2v", None))
        self.assertNotEqual(ti2v.VimooTI2VInput.VALIDATE_INPUTS("i2v", "none"), True)

    def test_lazy_routing_and_condition_cleanup(self):
        node = ti2v.VimooTI2VConditioning()
        self.assertEqual(node.check_lazy_status("t2v"), [])
        self.assertEqual(set(node.check_lazy_status("i2v")), {"image", "vae", "clip_vision"})
        original = [
            [
                object(),
                {
                    "clip_vision_output": object(),
                    "concat_mask": object(),
                    "vimoo_vision_mask": object(),
                    "text": "keep",
                },
            ]
        ]
        positive, negative, latent = node.encode("t2v", original, 32, 32, 5, negative=original)
        self.assertEqual(positive[0][1], {"text": "keep"})
        self.assertIn("clip_vision_output", original[0][1])
        self.assertEqual(tuple(latent["samples"].shape), (1, 48, 2, 2, 2))
        with self.assertRaisesRegex(ValueError, "requires"):
            node.encode("i2v", original, 32, 32, 5, negative=original)
        positive_only, absent_negative, _ = node.encode("t2v", original, 32, 32, 5)
        self.assertEqual(positive_only[0][1], {"text": "keep"})
        self.assertIsNone(absent_negative)

    def test_same_adapter_selects_both_tasks_without_mutating_it(self):
        from types import SimpleNamespace

        common = {
            "num_inference_steps": 12,
            "custom_timesteps": [
                1000,
                998.3,
                996.5,
                994.7,
                991.0,
                983.9,
                973.0,
                952.4,
                923.1,
                878.0,
                800.0,
                631.6,
            ],
            "timestep_space": "post_shift",
            "flow_shift": 12.0,
            "max_sequence_length": 1400,
        }
        document = {
            "task": "ti2v",
            "profile": "lite_ti2v_distilled",
            "recipes": {"t2v": common, "i2v": {**common, "i2v_first_frame_replace": False}},
        }
        before = copy.deepcopy(document)
        model = SimpleNamespace(
            model_options={"vimoo_sampling_profile": "lite_ti2v_anchor", "vimoo_distilled": document}
        )
        schedule = ti2v.VimooTI2VSchedule()
        for task in ("t2v", "i2v", "t2v"):
            with self.subTest(task=task):
                result = schedule.resolve(model, task, 12, 12.0, 1, 1, 0, 0, False)
                self.assertEqual(len(result[0]), 13)
                self.assertEqual(result[1:], (1, 0, False, "vimoo_lite_ti2v_euler", 1400))
        self.assertEqual(document, before)
        with self.assertRaisesRegex(ValueError, "workflow controls"):
            schedule.resolve(model, "i2v", 8, 12.0, 1, 1, 0, 0, False)

    def test_connected_sampler_controls_validate_before_values_are_known(self):
        sampler = sys.modules[spec.name + ".vimoo_nodes"].VimooSampler
        self.assertIs(sampler.VALIDATE_INPUTS(None, None, None, 1.0), True)
        self.assertNotEqual(sampler.VALIDATE_INPUTS("stochastic_flowmatch", 2.0, 0.0, 1.0), True)
        self.assertNotEqual(sampler.VALIDATE_INPUTS(None, None, 3.0, 1.0), True)

    def test_schedule_rejects_models_without_the_current_lite_profile(self):
        from types import SimpleNamespace

        node = ti2v.VimooTI2VSchedule()
        for options in ({}, {"vimoo_distilled": {"task": "ti2v", "recipes": {}}}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "Vimoo-1.0-Lite"):
                node.resolve(SimpleNamespace(model_options=options), "i2v", 50, 12, 4.5, 5.5, 1.2, 1, True)

    def test_full_recipe_task_transitions_preserve_tuning_and_lite_isolation(self):
        from types import SimpleNamespace
        import torch

        nodes = sys.modules[spec.name + ".vimoo_nodes"]
        model = SimpleNamespace(
            model_options={}, model=SimpleNamespace(diffusion_model=SimpleNamespace(hms=False))
        )
        schedule = ti2v.VimooTI2VSchedule()
        for task, scheduler, shift, replace, expected in (
            ("i2v", "vimoo_i2v_euler", 4.0, True, (8.0, 0.3, True, "vimoo_i2v_euler", 1000)),
            ("t2v", "vimoo", 2.0, True, (9.0, 0.2, False, "vimoo_uni_pc", 1000)),
            ("i2v", "vimoo_i2v_euler", 4.0, False, (8.0, 0.3, False, "vimoo_i2v_euler", 1000)),
        ):
            with self.subTest(task=task, replace=replace):
                result = schedule.resolve(
                    model,
                    task,
                    7,
                    2.0,
                    9.0,
                    8.0,
                    0.2,
                    0.3,
                    replace,
                    recipe="full",
                    full_i2v_flow_shift=4.0,
                )
                torch.testing.assert_close(
                    result[0], nodes.VimooScheduler().get_sigmas(scheduler, 7, shift)[0], rtol=0, atol=0
                )
                self.assertEqual(result[1:], expected)
        lite_model = SimpleNamespace(model_options={"vimoo_sampling_profile": "lite_ti2v_anchor"})
        lite = schedule.resolve(lite_model, "i2v", 7, 2.0, 9.0, 8.0, 0.2, 0.3, False, full_i2v_flow_shift=4.0)
        torch.testing.assert_close(
            lite[0], nodes.VimooScheduler().get_sigmas("vimoo_lite_ti2v_euler", 7, 2.0)[0], rtol=0, atol=0
        )
        self.assertEqual(lite[1:], (8.0, 0.3, False, "vimoo_lite_ti2v_euler", 1400))

    def test_full_recipe_rejects_lite_and_distilled_for_both_tasks(self):
        from types import SimpleNamespace

        schedule = ti2v.VimooTI2VSchedule()
        for options, hms in (({"vimoo_distilled": {"task": "ti2v"}}, False), ({}, True)):
            model = SimpleNamespace(
                model_options=options, model=SimpleNamespace(diffusion_model=SimpleNamespace(hms=hms))
            )
            for task in ("i2v", "t2v"):
                with self.subTest(task=task, hms=hms), self.assertRaises(ValueError):
                    schedule.resolve(model, task, 50, 12.0, 4.5, 5.5, 1.2, 1.0, True, recipe="full")

    def test_task_transition_cannot_reuse_missing_reference_image(self):
        from unittest.mock import Mock
        import torch

        nodes = sys.modules[spec.name + ".vimoo_nodes"]
        conditioning = [[torch.zeros(1, 1, 1), {"text": "keep"}]]
        vae = Mock()
        vae.encode.return_value = torch.ones(1, 48, 1, 2, 2)
        router = ti2v.VimooTI2VConditioning()
        with patch.object(nodes, "encode_vimoo_vision", return_value=object()):
            positive, negative, _ = router.encode(
                "i2v",
                conditioning,
                32,
                32,
                5,
                True,
                image=torch.zeros(1, 32, 32, 3),
                vae=vae,
                clip_vision=object(),
                negative=conditioning,
            )
        self.assertIsNotNone(sys.modules[spec.name + ".vimoo_sampler"]._find_ref_latent(positive))
        text_positive, text_negative, _ = router.encode("t2v", positive, 32, 32, 5, negative=negative)
        self.assertEqual(text_positive[0][1], {"text": "keep"})
        self.assertEqual(text_negative[0][1], {"text": "keep"})
        self.assertIsNotNone(sys.modules[spec.name + ".vimoo_sampler"]._find_ref_latent(positive))
        with self.assertRaises(ValueError):
            router.encode(
                "i2v", text_positive, 32, 32, 5, True, vae=vae, clip_vision=object(), negative=text_negative
            )
        with self.assertRaises(ValueError):
            ti2v.VimooTI2VInput().load("i2v", "none")

    def test_image_content_invalidates_cache(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "reference.png"
            image.write_bytes(b"first image")
            with patch.object(ti2v.folder_paths, "get_annotated_filepath", return_value=str(image)):
                first = ti2v.VimooTI2VInput.IS_CHANGED("i2v", image.name)
                self.assertEqual(first, ti2v.VimooImageLoader.IS_CHANGED(image.name))
                image.write_bytes(b"second image")
                self.assertNotEqual(first, ti2v.VimooTI2VInput.IS_CHANGED("i2v", image.name))
                self.assertNotEqual(first, ti2v.VimooImageLoader.IS_CHANGED(image.name))
