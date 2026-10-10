"""hidden_merge_split model loading, precision boundaries, patch padding, and fps conditioning."""

from dataclasses import asdict
import json
import math
import sys
import unittest
from unittest import mock

import torch

from test_model_runtime import model_module, tiny_vimoo
from test_nodes_runtime import nodes, spec

patchify = sys.modules[f"{spec.name}.vimoo_patchify"]


def tiny_hms(policy="fp32_io_time"):
    params = asdict(tiny_vimoo(hidden_size=128).params)
    params.update(
        patch_architecture="hidden_merge_split",
        precision_policy=policy,
        rope_temporal_interpolation_ref_fps=16.0,
    )
    model = model_module.VimooModel(
        **params, dtype=torch.bfloat16, device="cpu", operations=nodes.comfy.ops.manual_cast
    ).eval()
    generator = torch.Generator().manual_seed(77)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
    return model


def checkpoint(model):
    params = model.params
    config = dict(
        text_projection="single_refiner",
        qk_norm_type="rms",
        qk_norm=True,
        hidden_size=params.hidden_size,
        mm_double_blocks_depth=params.depth,
        out_channels=params.out_channels,
        patch_size=params.patch_size,
        text_states_dim=params.context_in_dim,
        in_channels=params.in_channels,
        concat_condition=False,
        heads_num=params.num_heads,
        rope_dim_list=params.axes_dim,
        rope_theta=params.theta,
        mlp_width_ratio=params.mlp_ratio,
        patch_architecture=params.patch_architecture,
        precision_policy=params.precision_policy,
        numerical_profile="lite" if params.patch_architecture == "hidden_merge_split" else "standard",
        sampling_profile="lite_ti2v_anchor" if params.patch_architecture == "hidden_merge_split" else None,
        rope_temporal_interpolation_ref_fps=params.rope_temporal_interpolation_ref_fps,
    )
    state = dict(model.state_dict())
    for stream in ("img", "txt"):
        for suffix in ("weight", "bias"):
            prefix = f"double_blocks.0.{stream}_attn"
            state[f"{prefix}.qkv.{suffix}"] = torch.cat(
                [state.pop(f"{prefix}.to_{part}.{suffix}") for part in "qkv"]
            )
    return state, config


class PatchifyTests(unittest.TestCase):
    def load(self, state, config):
        metadata = {"vimoo_conversion_version": "1", "vimoo_config": json.dumps(config)}
        with (
            mock.patch.object(nodes.folder_paths, "get_full_path_or_raise", return_value="hms.safetensors"),
            mock.patch.object(
                nodes.comfy.utils, "load_torch_file", side_effect=lambda *a, **kw: (dict(state), metadata)
            ),
        ):
            return nodes.VimooModelLoader().load("hms.safetensors", "bf16", disable_dynamic=True)[0]

    def test_loader_keeps_projection_precision_and_can_rebuild(self):
        source = tiny_hms()
        source.final_layer.adaLN_modulation.to(dtype=torch.bfloat16)
        state, config = checkpoint(source)
        patcher = self.load(state, config)
        model = patcher.model.diffusion_model
        for key in (
            "img_in.proj.weight",
            "img_in.patch_merge.weight",
            "final_layer.patch_split.weight",
            "final_layer.adaLN_modulation.1.weight",
            "time_in.in_layer.weight",
            "txt_in.t_embedder.in_layer.weight",
        ):
            actual, expected = model.state_dict()[key], source.state_dict()[key]
            self.assertEqual(actual.dtype, expected.dtype)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        factory, arguments, index = patcher.cached_patcher_init
        with (
            mock.patch.object(nodes.folder_paths, "get_full_path_or_raise", return_value="hms.safetensors"),
            mock.patch.object(
                nodes.comfy.utils,
                "load_torch_file",
                side_effect=lambda *a, **kw: (
                    dict(state),
                    {"vimoo_conversion_version": "1", "vimoo_config": json.dumps(config)},
                ),
            ),
        ):
            rebuilt = factory(*arguments, disable_dynamic=True)[index]
        self.assertEqual(rebuilt.model.diffusion_model.precision_policy, "fp32_io_time")
        self.assertEqual(model.patch_size, [1, 2, 2])

    def test_retired_hms_profiles_and_projection_policies_are_rejected(self):
        for policy in ("bf16", "fp32_io"):
            with (
                self.subTest(policy=policy),
                self.assertRaisesRegex(ValueError, "precision_policy=fp32_io_time"),
            ):
                tiny_hms(policy)
        state, config = checkpoint(tiny_hms())
        config.pop("sampling_profile")
        with self.assertRaisesRegex(ValueError, "current Lite TI2V"):
            self.load(state, config)

    def test_loader_defaults_spatial_rope_area_like_diffusers(self):
        state, config = checkpoint(tiny_hms())
        self.assertNotIn("rope_spatial_interpolation_ref_area", config)
        model = self.load(state, config).model.diffusion_model
        self.assertEqual(model.pe_embedder.ref_area, 1600.0)
        shape = (1, 2, 2, 45, 80)
        rope, _ = model.prepare_math_inputs(shape, 3, "cpu", fps=24.0)
        dim = model.params.axes_dim[1]
        freqs = 1 / (256.0 ** (torch.arange(0, dim, 2).float() / dim))
        mix = ((torch.linspace(0, 1, dim // 2) - 0.25) / 0.5).clamp(0, 1)
        expected = torch.cos(2 * freqs / (1 + (math.sqrt(45 * 80 / 1600) - 1) * mix)).repeat_interleave(2)
        torch.testing.assert_close(rope[0, 0, 40, 16:72], expected, rtol=0, atol=0)
        explicit = self.load(
            state, {**config, "rope_spatial_interpolation_ref_area": 1600.0}
        ).model.diffusion_model
        reference, _ = explicit.prepare_math_inputs(shape, 3, "cpu", fps=24.0)
        torch.testing.assert_close(rope, reference, rtol=0, atol=0)

    def test_loader_rejects_missing_or_mislabeled_hms_weights(self):
        state, config = checkpoint(tiny_hms())
        for missing in (
            "img_in.patch_merge.weight",
            "final_layer.patch_split.weight",
            "final_layer.patch_split.bias",
        ):
            incomplete = dict(state)
            incomplete.pop(missing)
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "Incomplete"):
                self.load(incomplete, config)
        for changed in ({"patch_architecture": "direct"}, {"patch_size": [1, 1, 1]}):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "mismatch"):
                self.load(state, {**config, **changed})
        without_config = dict(config)
        without_config.pop("patch_architecture")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.load(state, without_config)

    def test_odd_latent_is_zero_padded_and_prediction_cropped(self):
        model = tiny_hms()
        seen = []
        handle = model.img_in.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
        x = torch.randn(1, 2, 2, 3, 5).bfloat16()
        with torch.inference_mode():
            output = model(x, torch.tensor([713.0]), torch.randn(1, 3, 6).bfloat16(), fps=24.0)
        handle.remove()
        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(seen[0].shape[-2:], (4, 6))
        self.assertEqual(torch.count_nonzero(seen[0][..., -1, :]), 0)
        self.assertEqual(torch.count_nonzero(seen[0][..., -1]), 0)

    def test_rope_uses_patch_origins_original_area_and_generation_fps(self):
        model = tiny_hms()
        model.pe_embedder.ref_area = 1600.0
        shape = (1, 2, 2, 45, 80)
        ropes = {}
        for fps in (16.0, 24.0):
            rope, _ = model.prepare_math_inputs(shape, 3, "cpu", fps=fps)
            ropes[fps] = rope
        # Frame 1 starts after 23 * 40 patches, at temporal position 16/24.
        self.assertAlmostEqual(ropes[24.0][0, 0, 23 * 40, 0].item(), math.cos(16 / 24), places=6)
        self.assertFalse(torch.equal(ropes[16.0], ropes[24.0]))
        # Second patch row starts at latent row 2, using the original 45*80 area.
        dim = model.params.axes_dim[1]
        freqs = 1 / (256.0 ** (torch.arange(0, dim, 2).float() / dim))
        mix = ((torch.linspace(0, 1, dim // 2) - 0.25) / 0.5).clamp(0, 1)
        expected = torch.cos(2 * freqs / (1 + (math.sqrt(45 * 80 / 1600) - 1) * mix)).repeat_interleave(2)
        torch.testing.assert_close(ropes[24.0][0, 0, 40, 16:72], expected, rtol=0, atol=0)
        for fps in (None, 0.0, float("nan")):
            with self.subTest(fps=fps), self.assertRaisesRegex(ValueError, "fps"):
                model.prepare_math_inputs(shape, 3, "cpu", fps=fps)

    def test_serialized_temporal_rounding_at_121_frames(self):
        rope = patchify.VimooRoPE((2, 2, 2), 256.0, None, 1.0, 16.0)
        positions = torch.arange(31, dtype=torch.float32)
        ids = torch.zeros(1, 31, 3)
        ids[0, :, 0] = positions
        legacy = rope(ids, (31, 1, 1), fps=24.0)[0, 0, :, 0, 0, 0]
        torch.testing.assert_close(legacy, (positions / 1.5).cos(), rtol=0, atol=0)
        rope = patchify.VimooRoPE((2, 2, 2), 256.0, None, 1.0, 16.0, temporal_mode="multiply_reciprocal")
        actual = rope(ids, (31, 1, 1), fps=24.0)[0, 0, :, 0, 0, 0]
        torch.testing.assert_close(actual, (positions * (16.0 / 24.0)).cos(), rtol=0, atol=0)

    def test_fp32_time_does_not_round_frequency_inputs_to_bf16(self):
        model = tiny_hms()
        frequencies = torch.randn(1, 256)
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            actual = model.time_in(frequencies, output_dtype=torch.bfloat16)
        with torch.inference_mode(), torch.autocast("cpu", enabled=False):
            expected = model.time_in.out_layer(model.time_in.silu(model.time_in.in_layer(frequencies)))
        torch.testing.assert_close(actual, expected.bfloat16(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()


class ConfigurationContractTests(unittest.TestCase):
    load = PatchifyTests.load

    def test_numerical_profile_matches_the_checkpoint_architecture(self):
        for source, accepted in ((tiny_hms(), "lite"), (tiny_vimoo(hidden_size=128), "standard")):
            state, config = checkpoint(source)
            self.load(state, config)
            for rejected in ("legacy", "compact", "standard" if accepted == "lite" else "lite"):
                with self.subTest(architecture=config["patch_architecture"], profile=rejected):
                    with self.assertRaisesRegex(ValueError, "numerical_profile"):
                        self.load(state, {**config, "numerical_profile": rejected})

    def test_refiner_mask_metadata_precedence(self):
        for fields, expected in (
            ({}, True),
            ({"use_attention_mask": False}, False),
            ({"use_text_refiner_attention_mask": False}, False),
            ({"use_attention_mask": False, "use_text_refiner_attention_mask": True}, True),
        ):
            with self.subTest(fields=fields):
                state, config = checkpoint(tiny_hms())
                model = self.load(state, {**config, **fields}).model.diffusion_model
                self.assertEqual(model.params.use_text_refiner_attention_mask, expected)
