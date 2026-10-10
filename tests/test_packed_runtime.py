"""Packed checkpoint layout and runtime adapter contracts using real CPU ops."""

import sys
import unittest
from unittest.mock import patch

import torch
from test_nodes_runtime import nodes, spec
from test_model_runtime import _tiny_kwargs, model_module

precision = sys.modules[spec.name + ".vimoo_precision"]
lora = sys.modules[spec.name + ".vimoo_lora"]


class PackedRuntimeTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for resident adapter replacement")
    def test_resident_lora_replacement_runs_compiled_without_reloading_base(self):
        from comfy.model_patcher import ModelPatcher

        device = torch.device("cuda:0")
        for packed in (False, True):
            with self.subTest(packed=packed):
                root = torch.nn.Module()
                root.layer = nodes.comfy.ops.disable_weight_init.Linear(4, 12 if packed else 4, bias=False)
                root.layer.weight = torch.nn.Parameter(torch.randn(12 if packed else 4, 4, device=device))
                source = ModelPatcher(root, device, torch.device("cpu"))
                base = root.layer
                root.current_weight_patches_uuid = source.patches_uuid
                root.model_lowvram = False
                root.model_loaded_weight_memory = base.weight.numel() * base.weight.element_size()
                pairs = tuple((torch.randn(2, 4), torch.randn(4, 2)) for _ in range(3))
                replacement = source.clone()
                wrapper = lora._wrap_adapter(base, pairs if packed else pairs[0])
                replacement.add_object_patch("layer", wrapper)
                with patch.object(replacement, "load", side_effect=AssertionError("base reloaded")):
                    self.assertEqual(replacement.partially_load(device), 0)
                self.assertIs(root.layer.base, base)
                for branch in tuple(wrapper.down) if packed else (wrapper.down,):
                    self.assertEqual(branch.weight.device, device)
                    self.assertEqual(branch.weight.dtype, torch.float32)
                x = torch.randn(1, 5, 4, device=device)
                compiled = torch.compile(root.layer, backend="eager", fullgraph=True)
                torch.testing.assert_close(compiled(x), root.layer(x))
                replacement.unpatch_model(unpatch_weights=False)

    def test_packed_clone_uses_resident_targets_original_base(self):
        from comfy.model_patcher import ModelPatcher

        source_root, target_root = torch.nn.Module(), torch.nn.Module()
        source_root.layer = nodes.comfy.ops.manual_cast.Linear(4, 12, bias=False)
        target_root.layer = nodes.comfy.ops.manual_cast.Linear(4, 12, bias=False)
        source = ModelPatcher(source_root, torch.device("cpu"), torch.device("cpu"))
        target = ModelPatcher(target_root, torch.device("cpu"), torch.device("cpu"))
        pairs = tuple((torch.ones(2, 4), torch.ones(4, 2)) for _ in range(3))
        original = target_root.layer
        target.object_patches_backup["layer"] = original
        target_root.layer = lora.PackedLoRALinear(original, pairs)
        target.add_object_patch("layer", lora.PackedLoRALinear(source_root.layer, pairs))
        lora._rebind_lora_modules(source, target, {"layer": pairs})
        self.assertIs(target.object_patches["layer"].base, original)
        later = torch.nn.Identity()
        target.add_object_patch("layer", later)
        lora._rebind_lora_modules(source, target, {"layer": pairs})
        self.assertIs(target.object_patches["layer"], later)

    def test_packed_int8_can_load_split_views_for_matched_baseline(self):
        prefix = "double_blocks.0.img_attn."
        weight = torch.arange(48, dtype=torch.int8).reshape(12, 4)
        scales = torch.arange(1, 13, dtype=torch.float32).reshape(12, 1)
        config = torch.tensor(list(b'{"format": "int8_tensorwise"}'), dtype=torch.uint8)
        state = {
            prefix + "qkv.weight": weight,
            prefix + "qkv.weight_scale": scales,
            prefix + "qkv.comfy_quant": config,
            prefix + "qkv.bias": torch.zeros(12),
        }
        self.assertFalse(nodes._prepare_qkv_layout(state, {"vimoo_qkv_layout": "packed"}, "split", 1))
        for index, part in enumerate("qkv"):
            got = state[prefix + f"to_{part}.weight_scale"]
            self.assertEqual(got.untyped_storage().data_ptr(), scales.untyped_storage().data_ptr())
            self.assertTrue(torch.equal(got, scales[index * 4 : (index + 1) * 4]))
            self.assertIs(state[prefix + f"to_{part}.comfy_quant"], config)
        self.assertFalse(any(".qkv." in key for key in state))

    def test_packed_layout_selection_preserves_storage_and_rejects_split_input(self):
        select = getattr(nodes, "_prepare_qkv_layout", None)
        self.assertIsNotNone(select)
        state = {
            f"double_blocks.0.{stream}_attn.qkv.weight": torch.randn(36, 12) for stream in ("img", "txt")
        }
        original = dict(state)
        self.assertTrue(select(state, {}, "packed", 1))
        self.assertTrue(all(state[key] is value for key, value in original.items()))
        state["double_blocks.0.img_attn.to_q.weight"] = torch.randn(12, 12)
        with self.assertRaisesRegex(ValueError, "packed"):
            select(state, {}, "packed", 1)

    def test_packed_lora_adapters_group_in_qkv_order(self):
        group = getattr(lora, "_group_packed_adapters", None)
        self.assertIsNotNone(group)
        prefix = "diffusion_model.double_blocks.0.img_attn"
        pairs = {f"{prefix}.to_{part}": (torch.randn(2, 4), torch.randn(4, 2)) for part in "qkv"}
        grouped = group(pairs, {prefix + ".qkv.weight": (12, 4)})
        self.assertEqual(tuple(grouped), (prefix + ".qkv",))
        for got, part in zip(grouped[prefix + ".qkv"], "qkv"):
            self.assertIs(got, pairs[prefix + ".to_" + part])

    def test_packed_projection_never_concatenates_weights(self):
        attention_type = getattr(precision, "PackedSelfAttention", None)
        self.assertIsNotNone(attention_type)
        attn = attention_type(12, 3, True, operations=nodes.comfy.ops.manual_cast)
        with torch.no_grad():
            attn.qkv.weight.normal_()
            attn.qkv.bias.normal_()
        x = torch.randn(1, 5, 12)
        expected = torch.nn.functional.linear(x, attn.qkv.weight, attn.qkv.bias)
        with patch.object(torch, "cat", side_effect=AssertionError("online concat")):
            result = precision._fused_qkv(attn, x)
        for actual, reference in zip(result, expected.chunk(3, -1)):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_packed_model_exposes_one_weight_per_stream(self):
        packed = model_module.VimooModel(**{**_tiny_kwargs(), "packed_qkv": True})
        keys = packed.state_dict()
        self.assertIn("double_blocks.0.img_attn.qkv.weight", keys)
        self.assertNotIn("double_blocks.0.img_attn.to_q.weight", keys)

    def test_packed_lora_preserves_three_independent_low_rank_branches(self):
        wrapper_type = getattr(lora, "PackedLoRALinear", None)
        self.assertIsNotNone(wrapper_type)
        base = nodes.comfy.ops.manual_cast.Linear(4, 12, bias=True, dtype=torch.float64)
        with torch.no_grad():
            base.weight.normal_()
            base.bias.normal_()
        pairs = tuple(
            (torch.randn(2, 4, dtype=torch.float64), torch.randn(4, 2, dtype=torch.float64)) for _ in range(3)
        )
        module = wrapper_type(base, pairs)
        x = torch.randn(1, 5, 4, dtype=torch.float64)
        expected = tuple(
            y + torch.nn.functional.linear(torch.nn.functional.linear(x, a), b) * 0.5
            for y, (a, b) in zip(base(x).chunk(3, -1), pairs)
        )
        concatenate = torch.cat

        def only_outputs(tensors, *args, **kwargs):
            tensors = tuple(tensors)
            # Output branches may be joined; adapter/base weights must stay separate.
            self.assertTrue(all(tensor.shape == (1, 5, 4) for tensor in tensors))
            return concatenate(tensors, *args, **kwargs)

        with torch.no_grad(), patch.object(torch, "cat", side_effect=only_outputs):
            actual = module(x).chunk(3, -1)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_split_runtime_fusion_requires_packed_loading(self):
        split = precision.SplitSelfAttention(12, 3, True, operations=nodes.comfy.ops.manual_cast)
        with self.assertRaisesRegex(ValueError, "packed"):
            precision._fused_qkv(split, torch.randn(1, 5, 12))


if __name__ == "__main__":
    unittest.main()
