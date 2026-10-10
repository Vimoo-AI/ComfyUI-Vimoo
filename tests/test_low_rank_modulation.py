"""Shared low-rank timestep modulation, mixed storage, loading and distilled-adapter boundaries."""

from dataclasses import asdict
import json
import sys
import tempfile
from pathlib import Path
import unittest

import torch
from safetensors.torch import load_file, save_file

from test_model_runtime import model_module, tiny_vimoo
from test_nodes_runtime import nodes, spec
from comfy.model_patcher import ModelPatcher
import test_patchify
from test_patchify import checkpoint

mapping = sys.modules[f"{spec.name}.vimoo_checkpoint"]
lora = sys.modules[f"{spec.name}.vimoo_lora"]
distilled_nodes = sys.modules[f"{spec.name}.vimoo_distilled_nodes"]


def tiny_factorized():
    params = asdict(tiny_vimoo(hidden_size=128).params)
    params.update(block_modulation_rank=4, block_modulation_precision="bf16_expand_fp32_bias")
    model = model_module.VimooModel(
        **params, dtype=torch.bfloat16, device="cpu", operations=nodes.comfy.ops.manual_cast
    ).eval()
    generator = torch.Generator().manual_seed(77)
    with torch.no_grad():
        for value in model.parameters():
            value.copy_(torch.randn(value.shape, generator=generator) * 0.1)
    return model


class LowRankModulationTests(unittest.TestCase):
    def test_standard_profiles_load_factorized_masters_and_match_sampler(self):
        from unittest.mock import patch

        for profile, sampler in (("standard_t2v", "vimoo_uni_pc"), ("standard_i2v", "vimoo_i2v_euler")):
            with self.subTest(profile=profile):
                source = tiny_factorized()
                state, config = checkpoint(source)
                config.update(
                    block_modulation_rank=4,
                    block_modulation_precision="bf16_expand_fp32_bias",
                    sampling_profile=profile,
                    text_encoder_precision_profile=profile,
                )
                loaded = test_patchify.PatchifyTests().load(state, config)
                self.assertEqual(loaded.model_options["vimoo_sampling_profile"], profile)
                self.assertEqual(loaded.model_options["vimoo_text_encoder_precision_profile"], profile)
                self.assertTrue(loaded.model_options["transformer_options"]["vimoo_symmetric_self_attention"])
                for name in (
                    "mod_projection.weight",
                    "double_blocks.0.img_mod.lin.weight",
                    "double_blocks.0.img_mod.lin.bias",
                ):
                    actual = loaded.model.diffusion_model.state_dict()[name]
                    self.assertEqual(actual.dtype, state[name].dtype)
                    torch.testing.assert_close(actual, state[name], rtol=0, atol=0)
                options = dict(
                    model=loaded,
                    positive=[],
                    negative=[],
                    latent_image={"samples": torch.zeros(1)},
                    sigmas=torch.tensor([0.9, 0.0]),
                    cfg=3.0,
                    guidance_rescale=0.5,
                    seed=1234,
                    sampler_name="euler",
                    denoise=1.0,
                )
                with self.assertRaisesRegex(ValueError, f"requires sampler_name={sampler}"):
                    nodes.VimooSampler().sample(**options)
                options["sampler_name"] = sampler
                if profile == "standard_t2v":
                    target, function = nodes, "sample_standard"
                else:
                    target, function = sys.modules[f"{spec.name}.vimoo_i2v"], "sample_full_i2v"
                with patch.object(target, function, return_value=(options["latent_image"],)) as execute:
                    nodes.VimooSampler().sample(**options)
                execute.assert_called_once()

    def test_post_shift_schedule_and_raw_first_noise_preserve_rounding_boundaries(self):
        times = [999, 988, 974, 951, 923, 880, 808, 622]
        sigmas = nodes.VimooScheduler().get_sigmas("t2v_distilled", 8, 12.0)[0]
        expected = torch.tensor([*(value / 1000 for value in times), 0.0], dtype=torch.float64)
        torch.testing.assert_close(sigmas, expected, rtol=0, atol=0)
        noise = torch.tensor([0.99609375, -1.5], dtype=torch.bfloat16)
        source = distilled_nodes.DistilledModelSampling(sigmas=sigmas)
        self.assertIs(source.noise_scaling(sigmas[0].float(), noise, torch.zeros_like(noise)), noise)
        torch.testing.assert_close(
            source.timestep(sigmas[:-1].float()), torch.tensor(times, dtype=torch.float32), rtol=0, atol=0
        )
        for steps, shift in ((4, 12.0), (8, 6.0)):
            with self.assertRaisesRegex(ValueError, "requires"):
                nodes.VimooScheduler().get_sigmas("t2v_distilled", steps, shift)

    def test_mixed_expansion_matches_operation_order_without_fused_bias(self):
        model = tiny_factorized()
        vec = torch.linspace(-3.7, 2.2, 128, dtype=torch.bfloat16).unsqueeze(0)
        projection = model.mod_projection
        expected_code = torch.nn.functional.linear(
            torch.nn.functional.silu(vec).float(), projection.weight, projection.bias
        )
        code = model.modulation_code(vec)
        torch.testing.assert_close(code, expected_code, rtol=0, atol=0)
        block = model.double_blocks[0]
        for stream in ("img", "txt"):
            module = getattr(block, stream + "_mod")
            expected = (
                torch.nn.functional.linear(code.bfloat16(), module.lin.weight, None).float() + module.lin.bias
            ).to(module.output_dtype)
            actual = module(code)
            actual = torch.cat(
                [value for group in actual for value in (group.shift, group.scale, group.gate)], -1
            )
            torch.testing.assert_close(actual.squeeze(1), expected, rtol=0, atol=0)

    def test_compiled_expansion_keeps_native_cast_context_outside_fullgraph(self):
        from contextlib import contextmanager
        from unittest.mock import patch

        module = tiny_factorized().double_blocks[0].img_mod.lin
        code = torch.linspace(-3.7, 2.2, 4).unsqueeze(0)
        expected = torch.nn.functional.linear(code.bfloat16(), module.weight, None).float() + module.bias
        events = []

        @torch.compiler.disable
        def native_fault():
            events.append("cast")

        @contextmanager
        def cast_context(*args, **kwargs):
            native_fault()
            try:
                yield module.weight, module.bias
            finally:
                events.append("release")

        def backend(graph, inputs):
            events.append("compile")
            return graph.forward

        module.math = torch.compile(module.math, backend=backend, fullgraph=True, dynamic=True)
        with patch.object(nodes.comfy.ops, "CastBiasWeightContext", side_effect=cast_context):
            actual = module(code)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(events, ["cast", "compile", "release"])
            events.clear()
            torch.testing.assert_close(module(code), expected, rtol=0, atol=0)
            self.assertEqual(events, ["cast", "release"])

    def test_mixed_modulation_preserves_text_precision_in_eager_and_cached_execution(self):
        model = tiny_factorized()
        block = model.double_blocks[0]
        code = model.modulation_code(torch.linspace(-3.7, 2.2, 128).bfloat16().unsqueeze(0))
        direct = block._resolve_modulations(code, None, None, torch.bfloat16, torch.float32)
        cached = block._resolve_modulations(
            code, None, block.precompute_modulation(code), torch.bfloat16, torch.float32
        )
        for index, (want, got) in enumerate(zip(direct[:4], cached[:4])):
            dtype = torch.bfloat16 if index < 2 else torch.float32
            for expected, actual in zip(
                (want.shift, want.scale, want.gate), (got.shift, got.scale, got.gate)
            ):
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(actual, expected.squeeze(1), rtol=0, atol=0)
        self.assertFalse(torch.equal(direct[2].shift, direct[2].shift.bfloat16().float()))
        for stream in ("img", "txt"):
            self.assertEqual(getattr(block, stream + "_mod").lin.weight.dtype, torch.bfloat16)
            self.assertEqual(getattr(block, stream + "_mod").lin.bias.dtype, torch.float32)

    def test_save_reload_and_module_dtype_migration_preserve_exact_modulation_storage(self):
        model = tiny_factorized()
        expected = {
            name: value.clone()
            for name, value in model.state_dict().items()
            if name.startswith("mod_projection.") or "_mod.lin." in name
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "low_rank.safetensors"
            save_file(model.state_dict(), path)
            loaded = tiny_factorized()
            loaded.load_state_dict(load_file(path), strict=True)
        loaded.to(dtype=torch.float16)
        for name, before in expected.items():
            after = loaded.state_dict()[name]
            self.assertEqual(before.dtype, after.dtype)
            torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_loader_keeps_mixed_precision_and_rejects_a_rounded_bias(self):
        model = tiny_factorized()
        state, config = checkpoint(model)
        config.update(
            block_modulation_rank=4,
            block_modulation_precision="bf16_expand_fp32_bias",
            vae_numerical_profile="wan_fp32_norm",
        )
        patcher = test_patchify.PatchifyTests().load(state, config)
        self.assertEqual(patcher.model_options["vimoo_vae_numerical_profile"], "wan_fp32_norm")
        loaded = patcher.model.diffusion_model
        for name in (
            "mod_projection.weight",
            "double_blocks.0.img_mod.lin.bias",
            "double_blocks.0.txt_mod.lin.weight",
        ):
            actual, expected = loaded.state_dict()[name], state[name]
            self.assertEqual(actual.dtype, expected.dtype)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        state["double_blocks.0.txt_mod.lin.bias"] = state["double_blocks.0.txt_mod.lin.bias"].bfloat16()
        with self.assertRaisesRegex(ValueError, "must be torch.float32"):
            test_patchify.PatchifyTests().load(state, config)

    def test_native_name_mapping_keeps_unpacked_qkv_and_original_tensor_storage(self):
        tensors = {
            "block_modulation_proj.weight": torch.randn(4, 128),
            "double_blocks.0.img_mod.compact_linear.weight": torch.randn(768, 4).bfloat16(),
            "double_blocks.0.img_mod.compact_bias": torch.randn(768),
        }
        for stream in ("img", "txt"):
            for part in "qkv":
                for suffix in ("weight", "bias"):
                    shape = (128, 128) if suffix == "weight" else (128,)
                    tensors[f"double_blocks.0.{stream}_attn_{part}.{suffix}"] = torch.randn(shape).bfloat16()
        actual = mapping.normalize_transformer_state(
            tensors.items(), {"mm_double_blocks_depth": 1}, packed_qkv=False
        )
        self.assertIs(actual["mod_projection.weight"], tensors["block_modulation_proj.weight"])
        self.assertIs(
            actual["double_blocks.0.img_mod.lin.bias"], tensors["double_blocks.0.img_mod.compact_bias"]
        )
        self.assertIs(
            actual["double_blocks.0.img_attn.to_q.weight"], tensors["double_blocks.0.img_attn_q.weight"]
        )
        self.assertFalse(any(".qkv." in name for name in actual))

    def test_lite_ti2v_distilled_preserves_fp32_masters_and_bf16_autocast_through_clone_offload(self):
        def model_fixture():
            root = torch.nn.Module()
            root.diffusion_model = torch.nn.Module()
            block = torch.nn.Module()
            block.img_attn = torch.nn.Module()
            block.img_attn.to_q = nodes.comfy.ops.manual_cast.Linear(4, 4, bias=False, dtype=torch.bfloat16)
            with torch.no_grad():
                block.img_attn.to_q.weight.copy_(torch.arange(16).reshape(4, 4) * 0.03)
            root.diffusion_model.double_blocks = torch.nn.ModuleList([block])
            return ModelPatcher(root, torch.device("cpu"), torch.device("cpu"))

        a = torch.randn(128, 4, generator=torch.Generator().manual_seed(7)) * 0.13
        b = torch.randn(4, 128, generator=torch.Generator().manual_seed(8)) * 0.17
        state = {
            "transformer.double_blocks.0.attn.to_q.lora_A.weight": a,
            "transformer.double_blocks.0.attn.to_q.lora_B.weight": b,
        }
        metadata = {"lora_adapter_metadata": json.dumps({"transformer.r": 128, "transformer.lora_alpha": 64})}
        task_recipe = {
            "num_inference_steps": 12,
            "custom_timesteps": [1000, 998.3, 996.5, 994.7, 991, 983.9, 973, 952.4, 923.1, 878, 800, 631.6],
            "flow_shift": 12,
            "timestep_space": "post_shift",
            "max_sequence_length": 1400,
            "cfg": 1,
            "guidance_rescale": 0,
            "i2v_first_frame_replace": False,
        }
        model, fresh = model_fixture(), model_fixture()
        target = "diffusion_model.double_blocks.0.img_attn.to_q"
        base = model.get_model_object(target)
        x = torch.randn(32, 4, generator=torch.Generator().manual_seed(9)).bfloat16()
        base_result = base(x)
        expected = (
            base_result
            + torch.nn.functional.linear(torch.nn.functional.linear(x, a.bfloat16()), b.bfloat16()) * 0.5
        )
        incorrect_fp32 = (
            base_result.float()
            + torch.nn.functional.linear(torch.nn.functional.linear(x.float(), a), b) * 0.5
        ).bfloat16()
        self.assertFalse(torch.equal(expected, incorrect_fp32))
        with tempfile.TemporaryDirectory() as directory:
            base_path = Path(directory) / "base"
            base_path.write_bytes(b"small-existing-lite-base")
            model.model_options["vimoo_source_file"] = str(base_path)
            path = Path(directory) / "lite_ti2v_distilled.safetensors"
            save_file(state, path, metadata=metadata)
            recipe = {
                "format": "vimoo-distilled-lora-v1",
                "task": "ti2v",
                "profile": "lite_ti2v_distilled",
                "vae_numerical_profile": "wan_fp32_norm",
                "text_encoder_precision_profile": "lite_compact",
                "image_preprocessing": "lite_compact",
                "source_image_decode": "pil",
                "base_sha256": lora.sha256(base_path),
                "adapter_sha256": lora.sha256(path),
                "rank": 128,
                "alpha": 64,
                "effective_scale": 0.5,
                "adapter_storage_dtype": "float32",
                "adapter_compute_dtype": "bfloat16",
                "target_modules": ["double_blocks.0.attn.to_q"],
                "recipes": {"t2v": task_recipe, "i2v": task_recipe},
            }
            Path(str(path) + ".json").write_text(json.dumps(recipe))
            patched = lora.load_distilled_lora(model, path)
            self.assertEqual(patched.model_options["vimoo_vae_numerical_profile"], "wan_fp32_norm")
            self.assertEqual(patched.model_options["vimoo_text_encoder_precision_profile"], "lite_compact")
            self.assertNotIn("vimoo_text_encoder_precision_profile", model.model_options)
            self.assertEqual(patched.model_options["vimoo_image_preprocessing"], "lite_compact")
            self.assertTrue(patched.model_options["transformer_options"]["vimoo_flash_padded_capacity"])
            for parameter in (
                patched.object_patches[target].down.weight,
                patched.object_patches[target].up.weight,
            ):
                self.assertEqual(parameter.dtype, torch.float32)
            torch.testing.assert_close(patched.object_patches[target].down.weight, a, rtol=0, atol=0)
            torch.testing.assert_close(patched.object_patches[target].up.weight, b, rtol=0, atol=0)
            for key, value in load_file(path).items():
                self.assertEqual(value.dtype, torch.float32)
                torch.testing.assert_close(value, state[key], rtol=0, atol=0)
            try:
                patched.patch_model(device_to=torch.device("cpu"), load_weights=False)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    actual = model.model.get_submodule(target)(x)
                self.assertEqual(actual.dtype, torch.bfloat16)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            finally:
                patched.unpatch_model(device_to=torch.device("cpu"), unpatch_weights=False)
            fresh.model.load_state_dict(model.model.state_dict(), strict=True)
            cloned = patched.clone(model_override=fresh.get_clone_model_override())
            self.assertEqual(cloned.model_options["vimoo_text_encoder_precision_profile"], "lite_compact")
            self.assertEqual(cloned.model_options["vimoo_image_preprocessing"], "lite_compact")
            self.assertIs(cloned.object_patches[target].base, fresh.model.get_submodule(target))
            try:
                cloned.patch_model(device_to=torch.device("cpu"), load_weights=False)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    actual = fresh.model.get_submodule(target)(x)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(fresh.model.get_submodule(target).down.weight.dtype, torch.float32)
                torch.testing.assert_close(fresh.model.get_submodule(target).down.weight, a, rtol=0, atol=0)
                torch.testing.assert_close(fresh.model.get_submodule(target).up.weight, b, rtol=0, atol=0)
            finally:
                cloned.unpatch_model(device_to=torch.device("cpu"), unpatch_weights=False)
            recipe["adapter_compute_dtype"] = "float32"
            Path(str(path) + ".json").write_text(json.dumps(recipe))
            with self.assertRaisesRegex(ValueError, "BF16 compute"):
                lora.load_distilled_lora(model, path)

    def test_lite_ti2v_distilled_separate_bias_dtype_matches_each_source_target(self):
        precision = sys.modules[f"{spec.name}.vimoo_precision"]
        x = torch.randn(32, 4, generator=torch.Generator().manual_seed(46)).bfloat16()
        a = torch.randn(128, 4, generator=torch.Generator().manual_seed(47)).bfloat16()
        b = torch.randn(4, 128, generator=torch.Generator().manual_seed(48)).bfloat16()
        delta = torch.nn.functional.linear(torch.nn.functional.linear(x, a), b) * 0.5
        for name, upcast in (
            ("img_attn.proj", True),
            ("txt_attn.proj", False),
            ("img_mlp.0", False),
            ("txt_mlp.0", False),
            ("img_mlp.2", True),
            ("txt_mlp.2", True),
        ):
            with self.subTest(target=name):
                base = nodes.comfy.ops.manual_cast.Linear(4, 4, dtype=torch.bfloat16)
                with torch.no_grad():
                    base.weight.copy_(torch.randn(4, 4, generator=torch.Generator().manual_seed(49)))
                    base.bias.copy_(torch.randn(4, generator=torch.Generator().manual_seed(50)))
                # Configure through the same named module inventory as the model loader.
                block = torch.nn.Module()
                group, index = name.split(".")
                setattr(block, group, torch.nn.Module())
                getattr(block, group).add_module(index, base)
                model = torch.nn.Module()
                model.double_blocks = torch.nn.ModuleList([block])
                precision.configure_anchor_linears(model)
                rounded = torch.nn.functional.linear(x, base.weight, None) + delta
                source = rounded.float() + base.bias.float() if upcast else rounded + base.bias
                old = lora.LoRALinear(base, a, b)(x)
                actual = lora.LoRALinear(base, a, b, before_bias=True)(x)
                self.assertEqual(actual.dtype, torch.float32 if upcast else torch.bfloat16)
                self.assertFalse(torch.equal(old, source))
                torch.testing.assert_close(actual, source, rtol=0, atol=0)
