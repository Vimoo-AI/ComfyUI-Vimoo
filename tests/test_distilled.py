"""Distilled adapter arithmetic, stochastic recurrence and clone isolation."""

import json
import sys
import tempfile
from functools import partial
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from safetensors.torch import save_file
from test_nodes_runtime import spec, nodes

lora = sys.modules[spec.name + ".vimoo_lora"]
sampler = sys.modules[spec.name + ".vimoo_sampler"]
distilled_nodes = sys.modules[spec.name + ".vimoo_distilled_nodes"]
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import CallbacksMP


def release_recipe():
    return {
        "task": "t2v",
        "profile": "t2v_distilled",
        "adapter_format": "diffusers_lora",
        "adapter_storage_dtype": "bfloat16",
        "effective_scale": 0.5,
        "seed": 1234,
        "rank": 128,
        "alpha": 64,
        "adapter_compute_dtype": "bfloat16",
        "num_inference_steps": 8,
        "custom_timesteps": [999, 988, 974, 951, 923, 880, 808, 622],
        "timestep_space": "post_shift",
        "flow_shift": 12.0,
    }


class DistilledTests(unittest.TestCase):
    def test_lora_loader_only_returns_the_requested_model(self):
        with patch.object(
            distilled_nodes.folder_paths, "get_filename_list", return_value=["example.safetensors"]
        ):
            self.assertEqual(
                distilled_nodes.VimooDistilledLoRALoader.INPUT_TYPES()["required"]["lora_name"][0],
                ["example.safetensors"],
            )
        base, adapted = object(), object()
        loader = distilled_nodes.VimooDistilledLoRALoader()
        with patch.object(distilled_nodes, "load_distilled_lora", return_value=adapted) as load:
            with patch.object(distilled_nodes.folder_paths, "get_full_path_or_raise", return_value="adapter"):
                self.assertEqual(loader.load(base, "example.safetensors"), (adapted,))
            load.assert_called_once_with(base, "adapter")

    def test_distilled_node_accepts_shared_conditioning_and_keeps_source_model_unmodified(self):
        root = torch.nn.Linear(1, 1, dtype=torch.bfloat16)
        root.get_dtype = lambda: torch.bfloat16
        root.model_config = None
        model = ModelPatcher(root, torch.device("cuda"), torch.device("cpu"))
        model.model_options["vimoo_distilled"] = release_recipe()
        positive = [
            [
                torch.zeros(1, 2, 3),
                {
                    "conditioning_byt5small": torch.zeros(1, 256, 1472),
                    "vimoo_byt5_mask": torch.zeros(1, 256),
                },
            ]
        ]
        latent = {"samples": torch.zeros(1, 48, 2, 2, 2)}
        sigmas = sampler.recipe_sigmas(release_recipe())
        node = nodes.VimooSampler()

        def run(model, positive, latent, sigmas, seed, renoise_seed=..., **overrides):
            options = dict(
                model=model,
                positive=positive,
                negative=None,
                latent_image=latent,
                sigmas=sigmas,
                seed=seed,
                cfg=1,
                sampler_name="stochastic_flowmatch",
                guidance_rescale=0,
                denoise=1,
            )
            if renoise_seed is not ...:
                options["renoise_seed"] = renoise_seed
            options.update(overrides)
            return node.sample(**options)

        with (
            patch.object(distilled_nodes.torch, "Generator") as generator,
            patch.object(distilled_nodes, "video_noise", return_value=latent["samples"]),
            patch.object(
                distilled_nodes.comfy.sample, "sample_custom", return_value=latent["samples"]
            ) as sample,
        ):
            run(model, positive, latent, sigmas, 1234, 42)
        used_model, _, cfg, used_sampler, used_sigmas, used_positive, negative = sample.call_args.args
        self.assertIsNot(used_model, model)
        self.assertEqual(model.object_patches, {})
        self.assertIsInstance(
            used_model.object_patches["model_sampling"], distilled_nodes.DistilledModelSampling
        )
        self.assertEqual(cfg, 1.0)
        self.assertIs(used_positive[0][0], positive[0][0])
        for key, value in positive[0][1].items():
            self.assertIs(used_positive[0][1][key], value)
        self.assertEqual(used_positive[0][1]["fps"], 24.0)
        self.assertEqual(negative, [])
        self.assertEqual(used_sampler.extra_options["renoise_seed"], 42)
        generator.return_value.manual_seed.assert_called_once_with(1234)
        for seed, renoise, expected_seed in (
            (42, ..., 1001237),
            (0, ..., 1001237),
            (0, -1, 0),
            (42, None, 42),
            (42, 0, 0),
        ):
            with (
                patch.object(distilled_nodes.torch, "Generator"),
                patch.object(distilled_nodes, "video_noise", return_value=latent["samples"]),
                patch.object(
                    distilled_nodes.comfy.sample, "sample_custom", return_value=latent["samples"]
                ) as sample,
            ):
                run(model, positive, latent, sigmas, seed, renoise)
                self.assertEqual(sample.call_args.args[3].extra_options["renoise_seed"], expected_seed)
        torch.testing.assert_close(used_sigmas, sigmas.float(), rtol=0, atol=0)
        torch.testing.assert_close(used_sampler.extra_options["coefficients"], sigmas, rtol=0, atol=0)
        for overrides in ({"cfg": 8}, {"guidance_rescale": 1}, {"denoise": 0.5}, {"sampler_name": "euler"}):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, "requires"):
                run(model, positive, latent, sigmas, 1234, 42, **overrides)
        with self.assertRaisesRegex(ValueError, "sidecar sampling recipe"):
            run(model, positive, latent, torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64), 1234, 42)
        with self.assertRaisesRegex(ValueError, "Vimoo Text Encode"):
            run(model, [[positive[0][0], {}]], latent, sigmas, 1234, 42)

    def fixture(self):
        g = torch.Generator().manual_seed(12)
        state, shapes = {}, {}
        for part in "qkv":
            prefix = f"transformer.double_blocks.0.attn.to_{part}"
            state[prefix + ".lora_A.weight"] = torch.randn(128, 4, generator=g).bfloat16()
            state[prefix + ".lora_B.weight"] = torch.randn(4, 128, generator=g).bfloat16()
            shapes[f"diffusion_model.double_blocks.0.img_attn.to_{part}.weight"] = (4, 4)
        metadata = {"lora_adapter_metadata": json.dumps({"transformer.r": 128, "transformer.lora_alpha": 64})}
        return state, metadata, shapes

    def test_lite_lora_loads_without_reference_noise_and_preserves_weight_bindings(self):
        state, metadata, _ = self.fixture()
        root = torch.nn.Module()
        root.diffusion_model = torch.nn.Module()
        block = torch.nn.Module()
        block.img_attn = torch.nn.Module()
        for part in "qkv":
            block.img_attn.add_module(f"to_{part}", torch.nn.Linear(4, 4, bias=False).bfloat16())
        root.diffusion_model.double_blocks = torch.nn.ModuleList([block])
        model = ModelPatcher(root, torch.device("cpu"), torch.device("cpu"))
        task_recipe = {
            "num_inference_steps": 2,
            "custom_timesteps": [1000, 500],
            "timestep_space": "post_shift",
            "flow_shift": 12,
            "max_sequence_length": 1400,
            "initial_noise_mode": "cpu_channel_first",
            "initial_noise_seed": 1234,
        }
        with tempfile.TemporaryDirectory() as directory:
            base_path = Path(directory) / "base"
            base_path.write_bytes(b"tiny-vimoo-lite-base")
            model.model_options["vimoo_source_file"] = str(base_path)
            path = Path(directory) / "adapter.safetensors"
            save_file({key: value.float() for key, value in state.items()}, str(path), metadata=metadata)
            sidecar = Path(str(path) + ".json")
            recipe = {
                "format": "vimoo-distilled-lora-v1",
                "profile": "lite_ti2v_distilled",
                "task": "ti2v",
                "base_sha256": lora.sha256(base_path),
                "adapter_sha256": lora.sha256(path),
                "rank": 128,
                "alpha": 64,
                "effective_scale": 0.5,
                "adapter_storage_dtype": "float32",
                "adapter_compute_dtype": "bfloat16",
                "text_encoder_precision_profile": "lite_compact",
                "image_preprocessing": "lite_compact",
                "source_image_decode": "pil",
                "vae_numerical_profile": "wan_fp32_norm",
                "target_modules": [f"double_blocks.0.attn.to_{part}" for part in "qkv"],
                "recipes": {"t2v": task_recipe, "i2v": {**task_recipe, "i2v_first_frame_replace": False}},
            }
            sidecar.write_text(json.dumps(recipe))
            with patch.object(lora, "sha256", wraps=lora.sha256) as checksum:
                adapted = lora.load_distilled_lora(model, path)
            self.assertEqual({Path(call.args[0]) for call in checksum.call_args_list}, {base_path, path})
            self.assertEqual(len(adapted.object_patches), 3)
            self.assertEqual(model.object_patches, {})
            for field, message in (("base_sha256", "base checksum"), ("adapter_sha256", "LoRA checksum")):
                sidecar.write_text(json.dumps({**recipe, field: "0" * 64}))
                with self.subTest(binding=field), self.assertRaisesRegex(ValueError, message):
                    lora.load_distilled_lora(model, path)

    def test_adapter_mapping_preserves_separate_qkv_weights(self):
        state, metadata, shapes = self.fixture()
        adapters = lora.build_adapters(state, metadata, shapes, expected_modules=3)
        self.assertEqual(set(adapters), {key.removesuffix(".weight") for key in shapes})
        for part in "qkv":
            a, b = adapters[f"diffusion_model.double_blocks.0.img_attn.to_{part}"]
            prefix = f"transformer.double_blocks.0.attn.to_{part}"
            self.assertIs(a, state[prefix + ".lora_A.weight"])
            self.assertIs(b, state[prefix + ".lora_B.weight"])

    def test_invalid_adapter_pairs_targets_shapes_and_alpha_fail(self):
        state, metadata, shapes = self.fixture()
        state.pop(next(iter(state)))
        with self.assertRaisesRegex(ValueError, "A/B"):
            lora.build_adapters(state, metadata, shapes, expected_modules=3)
        with self.assertRaises(ValueError):
            lora.map_module("unknown", {})
        state, metadata, shapes = self.fixture()
        shapes[next(iter(shapes))] = (8, 4)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            lora.build_adapters(state, metadata, shapes, expected_modules=3)
        metadata["lora_adapter_metadata"] = json.dumps({"transformer.r": 128, "transformer.lora_alpha": 128})
        with self.assertRaisesRegex(ValueError, "alpha"):
            lora.build_adapters(state, metadata, shapes, expected_modules=3)

    def test_legacy_adapter_serialization_is_rejected(self):
        from types import SimpleNamespace

        state, _, _ = self.fixture()
        legacy = {key.replace(".weight", ".generator.weight"): value for key, value in state.items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.safetensors"
            save_file(
                legacy, path, metadata={"format": "vimoo_lora_safetensors_v2", "component": "generator"}
            )
            recipe = {
                **release_recipe(),
                "format": "vimoo-distilled-lora-v1",
                "adapter_format": "vimoo_lora_v2",
            }
            recipe["adapter_sha256"] = lora.sha256(path)
            Path(str(path) + ".json").write_text(json.dumps(recipe))
            model = SimpleNamespace(
                model_options={"vimoo_source_file": "base", "vimoo_sampling_profile": "t2v_distilled"},
                patches={},
            )
            with (
                patch.object(lora, "base_matches", return_value=True),
                self.assertRaisesRegex(ValueError, "BF16 Diffusers adapter"),
            ):
                lora.load_distilled_lora(model, path)

    def test_release_diffusers_lora_module_names_normalize(self):
        cases = {
            "double_blocks.0.img_mlp.net.0.proj": "double_blocks.0.img_mlp.fc1",
            "double_blocks.0.txt_mlp.net.2": "double_blocks.0.txt_mlp.fc2",
            "double_blocks.0.attn.to_out.0.projection": "double_blocks.0.attn.to_out.0",
            "double_blocks.0.attn.to_add_out.projection": "double_blocks.0.attn.to_add_out",
            "double_blocks.0.img_mlp.in_proj.projection": "double_blocks.0.img_mlp.fc1",
            "double_blocks.0.txt_mlp.out_proj.projection": "double_blocks.0.txt_mlp.fc2",
        }
        self.assertEqual(
            {key: lora.normalize_diffusers_module(key) for key in cases},
            cases,
        )

    def test_adapter_rejects_unsupported_peft_options(self):
        state, metadata, shapes = self.fixture()
        config = json.loads(metadata["lora_adapter_metadata"])
        unsupported = {
            "use_dora": True,
            "use_rslora": True,
            "rank_pattern": {"double_blocks.0.attn.to_q": 64},
            "alpha_pattern": {"double_blocks.0.attn.to_q": 32},
            "modules_to_save": ["final_layer"],
            "lora_bias": True,
            "fan_in_fan_out": True,
            "bias": "all",
        }
        for option, value in unsupported.items():
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, "unsupported PEFT"):
                lora.build_adapters(
                    state,
                    {"lora_adapter_metadata": json.dumps({**config, f"transformer.{option}": value})},
                    shapes,
                    expected_modules=3,
                )
        defaults = {
            "use_dora": False,
            "use_rslora": False,
            "rank_pattern": {},
            "alpha_pattern": {},
            "modules_to_save": None,
            "lora_bias": False,
            "fan_in_fan_out": False,
            "bias": "none",
        }
        adapters = lora.build_adapters(
            state,
            {
                "lora_adapter_metadata": json.dumps(
                    {**config, **{f"transformer.{k}": v for k, v in defaults.items()}}
                )
            },
            shapes,
            expected_modules=3,
        )
        self.assertEqual(len(adapters), 3)

    def test_base_provenance_binds_source_task_and_actual_file(self):
        provenance = sys.modules[spec.name + ".vimoo_checkpoint"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base.safetensors"
            source = Path(directory) / "source.safetensors"
            base.write_bytes(b"test derivative")
            source.write_bytes(b"test source")
            source_hash = lora.sha256(source)
            sidecar = Path(str(base) + ".vimoo.json")
            document = {
                "format": "vimoo-checkpoint-binding-v2",
                "task": "i2v",
                "sha256": lora.sha256(base),
                "base": {"file": source.name, "sha256": source_hash},
            }
            sidecar.write_text(json.dumps(document))
            self.assertTrue(lora.base_matches(base, source_hash, task="i2v"))
            self.assertFalse(lora.base_matches(base, source_hash, task="t2v"))
            self.assertFalse(lora.base_matches(base, None))
            base.write_bytes(b"substituted derivative")
            self.assertFalse(lora.base_matches(base, source_hash, task="i2v"))
            with self.assertRaisesRegex(ValueError, "checksum"):
                provenance.checkpoint_provenance(base)
            for content in ("{}", "null", "[]", "not JSON"):
                sidecar.write_text(content)
                self.assertFalse(lora.base_matches(base, source_hash))
                with self.assertRaises(ValueError):
                    provenance.checkpoint_provenance(base)
            sidecar.unlink()
            with self.assertRaisesRegex(ValueError, "Missing"):
                provenance.checkpoint_provenance(base)
            self.assertTrue(lora.base_matches(source, source_hash))

    def test_public_binding_rejects_legacy_and_missing_public_base(self):
        provenance = sys.modules[spec.name + ".vimoo_checkpoint"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base.safetensors"
            base.write_bytes(b"public base")
            sidecar = Path(str(base) + ".vimoo.json")
            document = {"format": "vimoo-checkpoint-binding-v2", "task": "t2v", "sha256": lora.sha256(base)}
            sidecar.write_text(json.dumps(document))
            self.assertTrue(lora.base_matches(base, document["sha256"], task="t2v"))
            self.assertFalse(lora.base_matches(base, document["sha256"], task="i2v"))
            for extra in (
                {"derived_from": {"sha256": document["sha256"]}},
                {"base": {"file": "missing.safetensors", "sha256": document["sha256"]}},
                {"base": {"file": "../base.safetensors", "sha256": document["sha256"]}},
            ):
                sidecar.write_text(json.dumps({**document, **extra}))
                with self.assertRaises(ValueError):
                    provenance.checkpoint_provenance(base)

    def test_renoise_default_and_explicit_seed_reproduce_trajectory(self):
        recipe = release_recipe()
        sigmas = sampler.recipe_sigmas(recipe)
        initial = torch.ones(1, 2, 3, 2, 2, dtype=torch.bfloat16)
        model = lambda x, sigma, **kwargs: x * 0.125
        default = sampler.sample_vimoo_distilled(model, initial, sigmas, recipe=recipe)
        explicit = sampler.sample_vimoo_distilled(model, initial, sigmas, renoise_seed=1001237, recipe=recipe)
        torch.testing.assert_close(default, explicit, rtol=0, atol=0)
        fixed_with_zero_seed = sampler.sample_vimoo_distilled(
            model, initial, sigmas, extra_args={"seed": 0}, recipe=recipe
        )
        torch.testing.assert_close(default, fixed_with_zero_seed, rtol=0, atol=0)
        zero = sampler.sample_vimoo_distilled(
            model, initial, sigmas, extra_args={"seed": 0}, renoise_seed=None, recipe=recipe
        )
        explicit_zero = sampler.sample_vimoo_distilled(model, initial, sigmas, renoise_seed=0, recipe=recipe)
        torch.testing.assert_close(zero, explicit_zero, rtol=0, atol=0)
        self.assertFalse(torch.equal(default, zero))

    def test_stochastic_route_rejects_euler_adapters_before_sampling(self):
        from types import SimpleNamespace

        with patch.object(distilled_nodes.comfy.sample, "sample_custom") as sample:
            for task in ("i2v", "ti2v"):
                model = SimpleNamespace(model_options={"vimoo_distilled": {"task": task}})
                with self.subTest(task=task), self.assertRaisesRegex(ValueError, "Euler sampler"):
                    distilled_nodes.sample_distilled(model, [], {}, None, 1, 42, 9)
            sample.assert_not_called()

    def test_sampling_consumes_the_explicit_recipe_grid(self):
        recipe = {
            "task": "t2v",
            "num_inference_steps": 3,
            "custom_timesteps": [1000, 800, 300],
            "flow_shift": 12,
            "timestep_space": "post_shift",
            "seed": 1234,
        }
        sigmas = sampler.recipe_sigmas(recipe)
        sampler.validate_distilled_sigmas(sigmas, recipe)
        result = sampler.sample_vimoo_distilled(
            lambda x, sigma, **kw: torch.zeros_like(x), torch.ones(1, 2, 2, 1, 1), sigmas, recipe=recipe
        )
        torch.testing.assert_close(result, torch.zeros_like(result))

    def test_unknown_or_mismatched_adapter_profiles_fail_before_weight_io(self):
        from types import SimpleNamespace

        model = SimpleNamespace(model_options={"vimoo_source_file": "base"}, patches={})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.safetensors"
            for profile, task in (
                ("distilled", "t2v"),
                ("ti2v-distilled", "ti2v"),
                ("t2v_distilled", "i2v"),
                ("lite_ti2v_distilled", "t2v"),
            ):
                Path(str(path) + ".json").write_text(
                    json.dumps({"format": "vimoo-distilled-lora-v1", "profile": profile, "task": task})
                )
                with (
                    self.subTest(profile=profile, task=task),
                    patch.object(lora, "sha256", side_effect=AssertionError("unexpected weight read")),
                    self.assertRaisesRegex(ValueError, "Unsupported Vimoo LoRA sidecar"),
                ):
                    lora.load_distilled_lora(model, path)

    def test_lora_uses_peft_arithmetic_and_restores_base_after_unpatch(self):
        state, metadata, shapes = self.fixture()
        a, b = next(iter(lora.build_adapters(state, metadata, shapes, expected_modules=3).values()))
        root = torch.nn.Module()
        root.layer = torch.nn.Linear(4, 4, dtype=torch.bfloat16)
        original = root.layer
        model = ModelPatcher(root, torch.device("cpu"), torch.device("cpu"))
        patched = model.clone()
        patched.add_object_patch("layer", lora.LoRALinear(original, a, b))
        x = torch.randn(2, 4).bfloat16()
        base = root.layer(x).clone()
        expected = base + torch.nn.functional.linear(torch.nn.functional.linear(x, a), b) * 0.5
        try:
            patched.patch_model(load_weights=False)
            torch.testing.assert_close(root.layer(x), expected, rtol=0, atol=0)
            self.assertEqual(
                {name for name, _ in root.layer.named_parameters()},
                {"base.weight", "base.bias", "down.weight", "up.weight"},
            )
        finally:
            patched.unpatch_model(unpatch_weights=False)
        self.assertIs(root.layer, original)
        self.assertEqual(model.object_patches, {})
        torch.testing.assert_close(root.layer(x), base, rtol=0, atol=0)

    def test_fp32_lora_restores_the_bf16_base_dtype_after_peft_addition(self):
        base = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        a = torch.randn(2, 4, generator=torch.Generator().manual_seed(7), dtype=torch.float32)
        b = torch.randn(4, 2, generator=torch.Generator().manual_seed(8), dtype=torch.float32)
        layer = lora.LoRALinear(base, a, b)
        x = torch.randn(3, 4, generator=torch.Generator().manual_seed(9), dtype=torch.bfloat16)
        base_result = base(x)
        delta = torch.nn.functional.linear(torch.nn.functional.linear(x.float(), a), b) * 0.5
        expected = (base_result.float() + delta).bfloat16()
        actual = layer(x)
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_lora_reuses_owned_outputs_and_preserves_mixed_dtype_arithmetic(self):
        for adapter_dtype in (torch.bfloat16, torch.float32):
            with self.subTest(adapter_dtype=adapter_dtype), torch.inference_mode():
                generator = torch.Generator().manual_seed(73)
                base = torch.nn.Linear(4, 16, dtype=torch.bfloat16)
                a = torch.randn(3, 4, generator=generator, dtype=adapter_dtype)
                b = torch.randn(16, 3, generator=generator, dtype=adapter_dtype)
                layer = lora.LoRALinear(base, a, b)
                x = torch.randn(7, 4, generator=generator, dtype=torch.bfloat16)
                original_input = x.clone()
                original_weights = {name: value.clone() for name, value in layer.named_parameters()}
                delta = (
                    torch.nn.functional.linear(torch.nn.functional.linear(x.to(adapter_dtype), a), b)
                    * layer.scale
                )
                expected = (base(x) + delta).to(x.dtype)
                pointers = {}
                handles = [
                    module.register_forward_hook(
                        lambda module, args, output, name=name: pointers.update({name: output.data_ptr()})
                    )
                    for name, module in (("base", base), ("up", layer.up))
                ]
                try:
                    actual = layer(x)
                    self.assertEqual(actual.data_ptr(), pointers["base"])
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    actual_delta = layer.vimoo_lora_delta(x)
                    self.assertEqual(actual_delta.data_ptr(), pointers["up"])
                    torch.testing.assert_close(actual_delta, delta, rtol=0, atol=0)
                finally:
                    for handle in handles:
                        handle.remove()
                torch.testing.assert_close(x, original_input, rtol=0, atol=0)
                for name, value in layer.named_parameters():
                    torch.testing.assert_close(value, original_weights[name], rtol=0, atol=0)

    def test_lora_reload_uses_original_modules_while_sibling_remains_resident(self):
        root = torch.nn.Module()
        root.diffusion_model = torch.nn.Module()
        root.diffusion_model.double_blocks = torch.nn.ModuleList([torch.nn.Module() for _ in range(40)])
        # Public Diffusers adapter names for the distilled T2V release.
        targets = {
            **{
                f"{stream}_attn_{part}": f"{stream}_attn.to_{part}"
                for stream in ("img", "txt")
                for part in "qkv"
            },
            "img_attn_proj": "img_attn.proj",
            "txt_attn_proj": "txt_attn.proj",
            **{
                f"{stream}_mlp.fc{index}": f"{stream}_mlp.{2 * (index - 1)}"
                for stream in ("img", "txt")
                for index in (1, 2)
            },
        }
        state = {}
        for index, block in enumerate(root.diffusion_model.double_blocks):
            for source, target in targets.items():
                parent, name = target.split(".")
                if parent not in block._modules:
                    block.add_module(parent, torch.nn.Module())
                block.get_submodule(parent).add_module(name, torch.nn.Linear(4, 4, bias=False).bfloat16())
                module = (
                    sys.modules[spec.name + ".vimoo_checkpoint"]
                    .normalize_backend_transformer_key(f"double_blocks.{index}.{source}.weight")
                    .removesuffix(".weight")
                )
                prefix = f"transformer.{module}"
                state[prefix + ".lora_A.weight"] = torch.full((128, 4), 0.01).bfloat16()
                state[prefix + ".lora_B.weight"] = torch.full((4, 128), 0.02).bfloat16()
        metadata = {
            "format": "pt",
            "lora_adapter_metadata": json.dumps({"transformer.r": 128, "transformer.lora_alpha": 64}),
        }
        model = ModelPatcher(root, torch.device("cpu"), torch.device("cpu"))
        model.model_options["vimoo_sampling_profile"] = "t2v_distilled"
        name = "diffusion_model.double_blocks.0.img_attn.to_q"
        original = root.get_submodule(name)
        x = torch.ones(2, 4).bfloat16()
        base_output = original(x)
        with tempfile.TemporaryDirectory() as directory:
            base_path = Path(directory) / "base"
            base_path.write_bytes(b"tiny-vimoo-base")
            model.model_options["vimoo_source_file"] = str(base_path)
            paths = []
            for index in (1, 2):
                path = Path(directory) / f"adapter-{index}.safetensors"
                save_file({key: value * index for key, value in state.items()}, str(path), metadata=metadata)
                Path(str(path) + ".json").write_text(
                    json.dumps(
                        {
                            **release_recipe(),
                            "format": "vimoo-distilled-lora-v1",
                            "base_sha256": lora.sha256(base_path),
                            "adapter_sha256": lora.sha256(path),
                        }
                    )
                )
                paths.append(path)
            first = lora.load_distilled_lora(model, paths[0])
            try:
                first.patch_model(load_weights=False)
                first.cleanup()
                resident_output = root.get_submodule(name)(x)
                second = lora.load_distilled_lora(model, paths[1])
                self.assertIs(second.object_patches[name].base, original)
                torch.testing.assert_close(root.get_submodule(name)(x), resident_output, rtol=0, atol=0)
            finally:
                first.unpatch_model(unpatch_weights=False)
            try:
                second.patch_model(load_weights=False)
                a = state["transformer.double_blocks.0.attn.to_q.lora_A.weight"] * 2
                b = state["transformer.double_blocks.0.attn.to_q.lora_B.weight"] * 2
                expected = base_output + torch.nn.functional.linear(torch.nn.functional.linear(x, a), b) * 0.5
                torch.testing.assert_close(root.get_submodule(name)(x), expected, rtol=0, atol=0)
            finally:
                second.unpatch_model(unpatch_weights=False)
        self.assertIs(root.get_submodule(name), original)
        self.assertEqual(model.object_patches, {})
        torch.testing.assert_close(original(x), base_output, rtol=0, atol=0)

    def test_lora_fresh_model_clone_rebinds_owner_and_preserves_later_patch(self):
        def make_model(weight):
            root = torch.nn.Module()
            root.layer = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
            root.layer.weight.data.fill_(weight)
            return ModelPatcher(root, torch.device("cpu"), torch.device("cpu"))

        source, fresh = make_model(1), make_model(3)
        a, b = torch.ones(2, 4).bfloat16(), torch.ones(4, 2).bfloat16()
        source.add_object_patch("layer", lora.LoRALinear(source.model.layer, a, b))
        source.add_callback_with_key(
            CallbacksMP.ON_CLONE,
            "vimoo_distilled_lora",
            partial(lora._rebind_lora_modules, adapters={"layer": (a, b)}),
        )
        cloned = source.clone(model_override=fresh.get_clone_model_override())
        self.assertIs(cloned.object_patches["layer"].base, fresh.model.layer)
        x = torch.ones(1, 4).bfloat16()
        original, base = source.model.layer(x).clone(), fresh.model.layer(x).clone()
        expected = base + torch.nn.functional.linear(torch.nn.functional.linear(x, a), b) * 0.5
        try:
            cloned.patch_model(load_weights=False)
            torch.testing.assert_close(fresh.model.layer(x), expected, rtol=0, atol=0)
            torch.testing.assert_close(source.model.layer(x), original, rtol=0, atol=0)
        finally:
            cloned.unpatch_model(unpatch_weights=False)
        torch.testing.assert_close(fresh.model.layer(x), base, rtol=0, atol=0)
        override = torch.nn.Identity()
        source.add_object_patch("layer", override)
        cloned = source.clone(model_override=fresh.get_clone_model_override())
        self.assertIs(cloned.object_patches["layer"], override)

    def test_distilled_recurrence_and_terminal_rng(self):
        shape = (1, 2, 3, 2, 2)
        recipe = release_recipe()
        sigmas = sampler.recipe_sigmas(recipe)
        initial = torch.zeros(shape, dtype=torch.bfloat16)
        predictions = [torch.full(shape, i / 10, dtype=torch.bfloat16) for i in range(len(sigmas) - 1)]
        seen = []

        def model(x, sigma, **kw):
            seen.append(x.clone())
            return predictions[len(seen) - 1]

        result = sampler.sample_vimoo_distilled(model, initial, sigmas, renoise_seed=5, recipe=recipe)
        g = torch.Generator().manual_seed(5)
        for i in range(len(predictions) - 1):
            noise = (
                torch.randn((1, 3, 2, 2, 2), generator=g, dtype=torch.bfloat16)
                .permute(0, 2, 1, 3, 4)
                .contiguous()
            )
            next_sigma = float(sigmas[i + 1])
            expected = (1 - next_sigma) * predictions[i] + next_sigma * noise
            torch.testing.assert_close(seen[i + 1], expected, rtol=0, atol=0)
        torch.testing.assert_close(result, predictions[-1], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            sampler.sample_vimoo_distilled(model, initial, sigmas[:3], recipe=recipe)

    def test_model_sampling_x0_uses_fp32_timestep_before_double_precision(self):
        cls = sys.modules[spec.name + ".vimoo_distilled_nodes"].DistilledModelSampling
        sampling = cls(None, sigmas=torch.tensor([1, 18 / 19, 6 / 7, 2 / 3, 0], dtype=torch.float64))
        # Near cancellation exposes the rounding difference from using raw sigma.
        x = torch.full((1, 1, 1, 1, 1), -14.625, dtype=torch.bfloat16)
        flow = torch.full_like(x, -15.4375)
        sigma = torch.tensor([18 / 19], dtype=torch.float32)
        timestep = torch.tensor([1000 * 18 / 19], dtype=torch.float32)
        result = sampling.calculate_denoised(sigma, flow.float(), x)
        expected = (x.double() - timestep.double().reshape(1, 1, 1, 1, 1) / 1000 * flow.double()).bfloat16()
        raw_sigma_result = (
            x.double() - sigma.float().double().reshape(1, 1, 1, 1, 1) * flow.double()
        ).bfloat16()
        self.assertFalse(torch.equal(raw_sigma_result, expected))
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_t2v_distilled_schedule_matches_release_recipe(self):
        (sigmas,) = nodes.VimooScheduler().get_sigmas("t2v_distilled", 8, 12.0)
        expected = torch.tensor(
            [value / 1000.0 for value in (999, 988, 974, 951, 923, 880, 808, 622)] + [0.0],
            dtype=torch.float64,
        )
        torch.testing.assert_close(sigmas, expected, rtol=0, atol=0)
        sampler.validate_distilled_sigmas(sigmas, release_recipe())
        with self.assertRaisesRegex(ValueError, "steps=8 and flow_shift=12"):
            nodes.VimooScheduler().get_sigmas("t2v_distilled", 8, 6.0)

    def test_recipe_requires_explicit_post_shift_fields(self):
        for overrides in ({"timestep_space": "pre_shift"}, {"custom_timesteps": None}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                sampler.recipe_sigmas({**release_recipe(), **overrides})
        with self.assertRaisesRegex(ValueError, "timestep count"):
            sampler.recipe_sigmas({"task": "t2v"})
        with self.assertRaisesRegex(ValueError, "matching LoRA sidecar"):
            sampler.validate_distilled_sigmas(sampler.recipe_sigmas(release_recipe()), None)

    def test_shared_tokenizer_keeps_fixed_padding_and_glyph_tokens(self):
        encoder = sys.modules[spec.name + ".vimoo_encoder"]
        tokenizer = encoder.VimooTokenizer()
        prompt = 'A cat holds a sign that says "Vimoo".'
        tokens = tokenizer.tokenize_with_weights(prompt)
        row = tokens["qwen25_7b"][0]
        self.assertEqual(len(row) - encoder.crop_start(row), 1000)
        self.assertEqual(len(tokens["byt5"][0]), 256)
        self.assertEqual(tokenizer.tokenize_with_weights(prompt), tokens)
